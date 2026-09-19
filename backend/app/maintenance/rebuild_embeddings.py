"""显式、事务化重建两张向量表；默认只展示计数，不访问外部模型或修改数据库。

运行位置为 backend：python -m app.maintenance.rebuild_embeddings [--apply]。
维护时先停止 API 写入；模型失败或 SQL 失败会回滚整个迁移，包括列维度变更。
不会删除患者、原始笔记、预约或事实。大数据生产迁移应改为分批影子索引切换。
"""
import argparse
import json
from uuid import uuid5

from sqlalchemy import MetaData, delete, func, select, text
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import Session

from app.db.models import ClinicalNote, EmbeddingChunk, KnowledgeChunk
from app.db.session import SessionLocal
from app.services.chunking import chunk_text
from app.services.embedding_index import INDEX_LOCK, index_metadata
from app.services.embedding_service import EmbeddingError, EmbeddingService


def index_counts(db: Session) -> dict[str, int]:
    """只输出数量，避免维护日志暴露患者或证据正文。"""
    return {model.__tablename__: db.scalar(select(func.count()).select_from(model)) or 0
            for model in (ClinicalNote, EmbeddingChunk, KnowledgeChunk)}


def rebuild_index(db: Session, embeddings: EmbeddingService | None = None) -> dict[str, int]:
    """在调用者事务中重建全部索引，不自行 commit；可以安全地在测试 schema 中运行。

    笔记从完整原文重新切片；没有独立原文表的知识/其它来源保留原块边界，必要时
    再细分。不会猜测或拼接旧重叠文本，原始 source_id 和患者关联保持不变。
    所有远端编码成功后才删除旧向量、改变维度并插入新向量。
    """
    service = embeddings or EmbeddingService()
    acquired = db.scalar(text("SELECT pg_try_advisory_xact_lock(:key, hashtext(current_schema()))"),
                         {"key": INDEX_LOCK})
    if not acquired:
        raise EmbeddingError("Index is in use; stop API writes and retry the rebuild")
    # 旧版本应用不认识 advisory lock；表锁也保护重建期间的原文及向量快照。
    # 锁超时意味着回滚退出，不无限等待正在使用的诊所工作区。
    db.execute(text("SET LOCAL lock_timeout = '5s'"))
    db.execute(text("LOCK TABLE clinical_notes, embedding_chunks, knowledge_chunks IN SHARE ROW EXCLUSIVE MODE"))
    notes = db.scalars(select(ClinicalNote)).all()
    note_keys = {(note.patient_id, note.id) for note in notes}
    # 一次读取旧来源元数据，避免几万条笔记产生几万次单独 SQL 查询。
    old_patient_rows = db.execute(select(
        EmbeddingChunk.id, EmbeddingChunk.source_type, EmbeddingChunk.source_id,
        EmbeddingChunk.chunk_text, EmbeddingChunk.meta, EmbeddingChunk.patient_id,
    ).order_by(EmbeddingChunk.id)).all()
    old_sources = {}
    for row in old_patient_rows:
        old_sources.setdefault((row.patient_id, row.source_id), row)
    patient_rows: list[EmbeddingChunk] = []
    knowledge_rows: list[KnowledgeChunk] = []

    for note in notes:
        # 原始笔记是权威来源，重建也补上以前尚未索引的笔记。
        old = old_sources.get((note.patient_id, note.id))
        source_type = old.source_type if old else "clinical_note"
        metadata = old.meta if old else {"note_type": note.note_type}
        for i, chunk in enumerate(chunk_text(note.content)):
            patient_rows.append(EmbeddingChunk(
                id=uuid5(note.id, f"rag:{i}"), patient_id=note.patient_id,
                source_type=source_type, source_id=note.id, chunk_text=chunk,
                meta=index_metadata(service, {**metadata, "chunk_index": i}),
            ))

    for model, target in ((EmbeddingChunk, patient_rows), (KnowledgeChunk, knowledge_rows)):
        # 不读取旧 embedding；旧列可以是任意历史维度，且无需把旧向量装进内存。
        columns = [model.id, model.source_type, model.source_id, model.chunk_text, model.meta]
        if model is EmbeddingChunk:
            columns.append(model.patient_id)
        source_rows = old_patient_rows if model is EmbeddingChunk else db.execute(select(*columns)).all()
        for row in source_rows:
            if model is EmbeddingChunk and (row.patient_id, row.source_id) in note_keys:
                continue
            parts = chunk_text(row.chunk_text)
            for i, chunk in enumerate(parts):
                data = dict(id=row.id if len(parts) == 1 else uuid5(row.id, f"part:{i}"),
                            source_type=row.source_type, source_id=row.source_id, chunk_text=chunk,
                            meta=index_metadata(service, {**row.meta, "rebuild_part": i}))
                if model is EmbeddingChunk:
                    data["patient_id"] = row.patient_id
                target.append(model(**data))

    # 数据库临时表保存已编码批次，Python 不同时持有几万个 1024 维浮点列表。
    # 临时表在当前事务结束时自动删除；这不是可跨失败恢复的永久缓存。
    stages = {}
    batch_size = service.settings.embedding_batch_size
    for model, rows in ((EmbeddingChunk, patient_rows), (KnowledgeChunk, knowledge_rows)):
        stage_name = "rag_rebuild_" + model.__tablename__
        db.execute(text(f"CREATE TEMP TABLE {stage_name} (LIKE {model.__tablename__} INCLUDING DEFAULTS) ON COMMIT DROP"))
        db.execute(text(f"ALTER TABLE {stage_name} ALTER COLUMN embedding TYPE vector({service.dim})"))
        stage = model.__table__.to_metadata(MetaData(), name=stage_name)
        stage.c.embedding.type = Vector(service.dim)
        stages[model] = stage
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            vectors = service.embed_documents([row.chunk_text for row in batch])
            data = []
            for row, vector in zip(batch, vectors):
                item = {"id": row.id, "source_type": row.source_type, "source_id": row.source_id,
                        "chunk_text": row.chunk_text, "metadata": row.meta, "embedding": vector}
                if model is EmbeddingChunk:
                    item["patient_id"] = row.patient_id
                data.append(item)
            db.execute(stage.insert(), data)

    for model in (EmbeddingChunk, KnowledgeChunk):
        db.execute(delete(model))
        # 表名是固定 ORM 常量，dim 来自已校验的整数，不能接受外部 SQL 标识符。
        db.execute(text(f"ALTER TABLE {model.__tablename__} ALTER COLUMN embedding TYPE vector({service.dim})"))
        stage = stages[model]
        columns = list(stage.c.keys())
        db.execute(model.__table__.insert().from_select(columns, select(*(stage.c[name] for name in columns))))
    db.flush()
    db.info.pop("embedding_index_checked", None)
    return {"embedding_chunks": len(patient_rows), "knowledge_chunks": len(knowledge_rows),
            "dimension": service.dim}


def main() -> None:
    """默认只读；--apply 明确表示向配置的提供方发送现有文本并提交重建。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Embed existing text and atomically replace both vector indexes")
    args = parser.parse_args()
    with SessionLocal.begin() as db:
        if not args.apply:
            print(json.dumps({"mode": "preview", **index_counts(db)}))
            return
        result = rebuild_index(db)
    # 只有事务提交成功才报告完成，避免 flush 成功但 commit 失败时误报。
    print(json.dumps({"mode": "rebuilt", **result}))


if __name__ == "__main__":
    main()
