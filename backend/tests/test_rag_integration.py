"""RAG 数据库回归：仅复用随机 schema fixture，不访问应用患者或真实模型。"""
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text

from test_workspace_integration import workspace, register  # noqa: F401
from app.core.config import get_settings
from app.db.models import Patient, ClinicalNote, EmbeddingChunk, KnowledgeChunk
from app.main import app
from app.integrations.dox_mysql_importer import DoxMySqlImporter
from app.maintenance.rebuild_embeddings import rebuild_index
from app.services.embedding_service import EmbeddingService, EmbeddingError
from app.services.embedding_index import check_index, EmbeddingIndexError
from app.services.vector_store import VectorStore


def legacy_index(factory):
    """在本次测试 schema 内构造旧 384 维索引，同时保留可恢复的原始笔记。"""
    with factory.begin() as db:
        patient = Patient(name="Synthetic migration patient")
        db.add(patient)
        db.flush()
        note = ClinicalNote(patient_id=patient.id, note_type="test", content="牙龈出血，建议复诊。" * 60)
        db.add(note)
        db.flush()
        for table in ("embedding_chunks", "knowledge_chunks"):
            db.execute(text(f"ALTER TABLE {table} ALTER COLUMN embedding TYPE vector(384)"))
        params = {"id": uuid4(), "pid": patient.id, "source": note.id,
                  "vector": "[" + ",".join(["1"] + ["0"] * 383) + "]"}
        db.execute(text("INSERT INTO embedding_chunks (id, patient_id, source_type, source_id, chunk_text, metadata, embedding) "
                        "VALUES (:id, :pid, 'clinical_note', :source, 'old truncated note', '{}'::jsonb, CAST(:vector AS vector))"), params)
        params["id"] = uuid4()
        db.execute(text("INSERT INTO knowledge_chunks (id, source_type, source_id, chunk_text, metadata, embedding) "
                        "VALUES (:id, 'synthetic_knowledge', :source, 'Brush twice daily.', '{}'::jsonb, CAST(:vector AS vector))"), params)
        return patient.id, note.id


def column_dimension(db):
    return db.scalar(text("SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                          "WHERE attrelid='embedding_chunks'::regclass AND attname='embedding'"))


def test_legacy_rebuild_atomicity_and_idempotence(workspace, monkeypatch):
    patient_id, note_id = legacy_index(workspace)
    with workspace() as db:
        with pytest.raises(EmbeddingIndexError):
            check_index(db, EmbeddingService())

    service = EmbeddingService()
    def failed_batch(texts):
        raise EmbeddingError("Synthetic provider failure")
    monkeypatch.setattr(service, "embed_documents", failed_batch)
    with pytest.raises(EmbeddingError), workspace.begin() as db:
        rebuild_index(db, service)
    with workspace() as db:
        assert column_dimension(db) == "vector(384)"
        assert db.scalar(select(func.count()).select_from(EmbeddingChunk)) == 1
        assert db.get(ClinicalNote, note_id) is not None

    with workspace.begin() as db:
        first = rebuild_index(db)
        assert first["embedding_chunks"] > 1
        assert first["knowledge_chunks"] == 1
        assert column_dimension(db) == "vector(1024)"
        check_index(db, EmbeddingService())
    with workspace.begin() as db:
        second = rebuild_index(db)
        assert first == second
        assert db.scalar(select(func.count()).select_from(Patient)) == 1
        assert db.scalar(select(func.count()).select_from(ClinicalNote)) == 1
        assert all(row.patient_id == patient_id for row in db.scalars(select(EmbeddingChunk)))


def test_sql_failure_rolls_back_dimension_and_old_vectors(workspace, monkeypatch):
    """模拟远端成功而第二张表迁移失败，第一张表的删除和维度修改也必须回滚。"""
    legacy_index(workspace)
    with workspace() as db:
        original_execute = db.execute
        def fail_second_alter(statement, *args, **kwargs):
            if str(statement).startswith("ALTER TABLE knowledge_chunks"):
                raise RuntimeError("Synthetic DDL failure")
            return original_execute(statement, *args, **kwargs)
        monkeypatch.setattr(db, "execute", fail_second_alter)
        with pytest.raises(RuntimeError), db.begin():
            rebuild_index(db)
    with workspace() as db:
        assert column_dimension(db) == "vector(384)"
        assert db.scalar(select(func.count()).select_from(EmbeddingChunk)) == 1


def test_fingerprint_blocks_same_dimension_model_change(workspace, monkeypatch):
    with workspace.begin() as db:
        VectorStore(db).add_knowledge_chunk("test", uuid4(), "synthetic evidence", {})
    monkeypatch.setattr(get_settings(), "embedding_provider", "openrouter")
    with workspace() as db, pytest.raises(EmbeddingIndexError):
        # hash 与 Liquid 都是 1024 维，仍然不能混用；检测不需要远程请求。
        VectorStore(db).search_knowledge("synthetic")


def test_patient_scoped_retrieval_and_reindex_cleanup(workspace):
    client = TestClient(app)
    register(client)
    a = client.post("/api/patients", json={"name": "Synthetic A"}).json()["id"]
    b = client.post("/api/patients", json={"name": "Synthetic B"}).json()["id"]
    for patient_id in (a, b):
        response = client.post(f"/api/patients/{patient_id}/notes", json={"note_type": "test", "content": "gum bleeding " * 100})
        assert response.status_code == 200, response.text
    first = client.post(f"/api/patients/{a}/notes/ingest").json()["chunks_created"]
    assert client.post(f"/api/patients/{a}/notes/ingest").json()["chunks_created"] == first
    from uuid import UUID
    with workspace() as db:
        matches = VectorStore(db).search(UUID(a), "gum bleeding", 100)
        assert len(matches) == first
        assert all(str(row.patient_id) == a for row, _ in matches)


def test_failed_embedding_rolls_back_note_and_returns_safe_error(workspace, monkeypatch):
    client = TestClient(app)
    register(client)
    patient = client.post("/api/patients", json={"name": "Synthetic failure"}).json()
    def fail(self, texts):
        raise EmbeddingError("Embedding provider request failed")
    monkeypatch.setattr(EmbeddingService, "embed_documents", fail)
    response = client.post(f"/api/patients/{patient['id']}/notes", json={"note_type": "test", "content": "synthetic"})
    assert response.status_code == 503
    with workspace() as db:
        assert db.scalar(select(func.count()).select_from(ClinicalNote)) == 0
        assert db.scalar(select(func.count()).select_from(EmbeddingChunk)) == 0


def test_dox_shortened_source_removes_old_chunks(workspace):
    with workspace.begin() as db:
        patient = Patient(name="Synthetic DOX")
        db.add(patient)
        db.flush()
        importer = DoxMySqlImporter(db)
        source_id = uuid4()
        importer._upsert_patient_chunks(patient.id, "dox_test", source_id, "long note " * 100, {})
        db.flush()
        assert db.scalar(select(func.count()).select_from(EmbeddingChunk)) > 1
        importer._upsert_patient_chunks(patient.id, "dox_test", source_id, "short note", {})
        db.flush()
        assert db.scalar(select(func.count()).select_from(EmbeddingChunk)) == 1


def test_rebuild_refuses_concurrent_index_operation(workspace):
    with workspace() as reader:
        check_index(reader, EmbeddingService())
        with workspace.begin() as writer, pytest.raises(EmbeddingError, match="in use"):
            rebuild_index(writer)
