"""清空功能的真实 PostgreSQL 回归；只操作随机 schema 中的合成资料。"""
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from test_workspace_integration import workspace, register  # noqa: F401
from test_rag_integration import legacy_index, column_dimension
from app.db.models import User
from app.main import app
from app.services.workspace_lock import workspace_lock
from app.services.workspace_service import BUSINESS_TABLES, CHECKPOINT_TABLES, clear_workspace
from app.services.conversation_service import conversation_lock

CONFIRM = {"confirmation": "CLEAR_CLINIC_DATA"}


def test_reset_requires_login_origin_and_explicit_scope(workspace):
    client = TestClient(app)
    assert client.post("/api/workspace/reset", json=CONFIRM).status_code == 401
    register(client)
    assert client.post("/api/workspace/reset", json={}).status_code == 422
    assert client.post("/api/workspace/reset", json={"confirmation": "yes"}).status_code == 422
    assert client.post("/api/workspace/reset", json={**CONFIRM, "table": "users"}).status_code == 422
    assert client.post("/api/workspace/reset", json=CONFIRM, headers={"Origin": "https://untrusted.example"}).status_code == 403


def test_reset_clears_memory_and_legacy_vectors_but_keeps_accounts(workspace):
    """跨账号会话一起清空，但两个账号均继续登录；旧维度转换不调用模型。"""
    patient_id, _ = legacy_index(workspace)
    clients = [TestClient(app), TestClient(app)]
    conversation_ids = []
    for i, client in enumerate(clients):
        register(client, f"reset_user_{i}")
        cid = client.post("/api/conversations", json={"patient_ids": [str(patient_id)]}).json()["id"]
        conversation_ids.append(cid)
        response = client.post(f"/api/conversations/{cid}/messages", json={"message": "Synthetic hello"})
        assert '"type":"result"' in response.text
    with workspace() as db:
        assert db.scalar(text("SELECT count(*) FROM checkpoints")) > 0
        migrations_before = db.scalar(text("SELECT count(*) FROM checkpoint_migrations"))
    response = clients[0].post("/api/workspace/reset", json=CONFIRM)
    assert response.status_code == 200, response.text
    assert response.json()["deleted_counts"]["patients"] == 1
    assert response.json()["deleted_counts"]["conversations"] == 2
    assert response.json()["embedding_dim"] == 1024
    with workspace() as db:
        for table in (*BUSINESS_TABLES, *CHECKPOINT_TABLES):
            assert db.scalar(text(f"SELECT count(*) FROM {table}")) == 0, table
        assert len(db.scalars(select(User)).all()) == 2
        assert db.scalar(text("SELECT count(*) FROM checkpoint_migrations")) == migrations_before
        assert column_dimension(db) == "vector(1024)"
    for client, cid in zip(clients, conversation_ids):
        assert client.get("/api/auth/me").status_code == 200
        assert client.get(f"/api/conversations/{cid}").status_code == 404
        assert client.post(f"/api/conversations/{cid}/resume", json={"interrupt_id": "old", "decisions": []}).status_code in (404, 422)
    # 空工作区可立即添加新笔记并索引，而不是只把页面显示清空。
    patient = clients[0].post("/api/patients", json={"name": "New synthetic patient"}).json()
    assert clients[0].post(f"/api/patients/{patient['id']}/notes", json={"note_type": "test", "content": "New synthetic note"}).status_code == 200
    assert clients[0].post("/api/workspace/reset", json=CONFIRM).status_code == 200
    assert clients[0].post("/api/workspace/reset", json=CONFIRM).json()["deleted_counts"]["patients"] == 0


def test_reset_refuses_active_agent_and_maintenance_blocks_business(workspace):
    client = TestClient(app)
    register(client)
    cid = client.post("/api/conversations", json={}).json()["id"]
    from uuid import UUID
    # 使用真实 Agent 运行锁，而非只模拟前端 loading。
    with conversation_lock(UUID(cid)):
        assert client.post("/api/workspace/reset", json=CONFIRM).status_code == 409
    with workspace_lock(workspace.kw["bind"], exclusive=True):
        assert client.post("/api/patients", json={"name": "Must not be created"}).status_code == 409
        assert client.get("/api/auth/me").status_code == 200
    assert client.post("/api/workspace/reset", json=CONFIRM).status_code == 200


def test_reset_ddl_failure_rolls_back_everything(workspace, monkeypatch):
    legacy_index(workspace)
    with workspace() as db:
        original = db.execute
        def failed_alter(statement, *args, **kwargs):
            if str(statement).startswith("ALTER TABLE"):
                return original(text("SELECT 1 / 0"))
            return original(statement, *args, **kwargs)
        monkeypatch.setattr(db, "execute", failed_alter)
        with pytest.raises(HTTPException) as error:
            clear_workspace(db)
        assert error.value.status_code == 503
    with workspace() as db:
        assert db.scalar(text("SELECT count(*) FROM patients")) == 1
        assert db.scalar(text("SELECT count(*) FROM clinical_notes")) == 1
        assert db.scalar(text("SELECT count(*) FROM embedding_chunks")) == 1
        assert column_dimension(db) == "vector(384)"
