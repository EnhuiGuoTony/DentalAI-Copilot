"""向量空间一致性检查：同维度也可能来自不同模型，不能只依赖 pgvector 类型。

索引操作获取事务级共享 advisory lock；重建获取独占锁，防止新代码并发混写。
锁按当前 schema 隔离，因此随机 schema 集成测试不会锁住应用索引。
"""
from sqlalchemy import or_, select, text
from sqlalchemy.orm import Session

from app.db.models import EmbeddingChunk, KnowledgeChunk
from app.services.embedding_service import EmbeddingError, EmbeddingService

INDEX_LOCK = 741923
REBUILD_MESSAGE = "Embedding index is incompatible. Run python -m app.maintenance.rebuild_embeddings --apply before indexing or retrieval."


class EmbeddingIndexError(EmbeddingError):
    """旧索引必须显式重建，不能把缺失标记的旧哈希向量当作语义向量。"""


def check_index(db: Session, embeddings: EmbeddingService) -> None:
    """每个事务/指纹检查一次两张向量表；失败时不会调用外部模型。

    create_all 不会修改旧列的维度，所以也检查 PostgreSQL 实际列类型。
    事务缓存只避免同一次批量导入反复扫描；下一事务仍重新检查。
    """
    db.execute(text("SELECT pg_advisory_xact_lock_shared(:key, hashtext(current_schema()))"), {"key": INDEX_LOCK})
    transaction = db.get_transaction()
    cache_key = (transaction, embeddings.fingerprint)
    if db.info.get("embedding_index_checked") == cache_key:
        return
    for model in (EmbeddingChunk, KnowledgeChunk):
        actual = db.execute(text(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid=to_regclass(:table) AND attname='embedding' AND NOT attisdropped"
        ), {"table": model.__tablename__}).scalar_one_or_none()
        if actual != f"vector({embeddings.dim})":
            raise EmbeddingIndexError(REBUILD_MESSAGE)
        incompatible = db.scalar(select(model.id).where(or_(
            model.meta["embedding_fingerprint"].astext.is_(None),
            model.meta["embedding_fingerprint"].astext != embeddings.fingerprint,
        )).limit(1))
        if incompatible is not None:
            raise EmbeddingIndexError(REBUILD_MESSAGE)
    db.info["embedding_index_checked"] = cache_key


def index_metadata(embeddings: EmbeddingService, metadata: dict) -> dict:
    """覆盖保留字段，避免外部导入元数据伪造当前向量版本。"""
    return {**metadata, "embedding_fingerprint": embeddings.fingerprint,
            "embedding_provider": embeddings.settings.embedding_provider,
            "embedding_model": (embeddings.settings.embedding_model
                                if embeddings.settings.embedding_provider == "openrouter" else "signed-sha256-v1"),
            "embedding_dim": embeddings.dim, "chunk_strategy": "recursive-utf8-v1"}
