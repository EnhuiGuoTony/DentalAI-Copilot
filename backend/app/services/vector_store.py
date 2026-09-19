from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import EmbeddingChunk, KnowledgeChunk
from app.services.embedding_service import EmbeddingService
from app.services.embedding_index import check_index, index_metadata
from app.services.chunking import chunk_text


class VectorStore:
    """封装向量写入与召回，连接 Python 的向量生成器和 PostgreSQL 的 pgvector。

    患者病历与全局知识分表保存。此层只返回文本块和分数，不调用聊天模型生成答案。
    写入方法只加入当前 Session，由调用方决定何时 commit，便于批量处理。
    """
    def __init__(self, db: Session) -> None:
        self.db = db
        self.embeddings = EmbeddingService()
        # 仅在本次工具/请求的实例中复用查询向量，多路检索不重复调用模型。
        # 不使用全局文本缓存，避免患者查询在进程中长期留存。
        self._query_vectors: dict[str, list[float]] = {}

    def _query_vector(self, query: str) -> list[float]:
        """检查索引兼容性后编码；同一个 store 中的多路检索共享查询向量。"""
        check_index(self.db, self.embeddings)
        if query not in self._query_vectors:
            self._query_vectors[query] = self.embeddings.embed_query(query)
        return self._query_vectors[query]

    def add_chunk(self, patient_id: UUID, source_type: str, source_id: UUID, text: str, metadata: dict) -> EmbeddingChunk:
        """保存一个患者文本块；保留原文、来源和元数据，便于检索后展示与追溯。"""
        check_index(self.db, self.embeddings)
        chunk = EmbeddingChunk(
            patient_id=patient_id,
            source_type=source_type,
            source_id=source_id,
            chunk_text=text,
            meta=index_metadata(self.embeddings, metadata),
            embedding=self.embeddings.embed(text),
        )
        self.db.add(chunk)
        return chunk

    def add_knowledge_chunk(self, source_type: str, source_id: UUID, text: str, metadata: dict) -> KnowledgeChunk:
        """保存不属于具体患者的共享知识块；不要把患者私有病历写入这个共享语料库。"""
        check_index(self.db, self.embeddings)
        chunk = KnowledgeChunk(
            source_type=source_type,
            source_id=source_id,
            chunk_text=text,
            meta=index_metadata(self.embeddings, metadata),
            embedding=self.embeddings.embed(text),
        )
        self.db.add(chunk)
        return chunk

    def add_document(self, patient_id: UUID, source_type: str, source_id: UUID,
                     text: str, metadata: dict) -> list[EmbeddingChunk]:
        """笔记原文一次切片、批量编码后入库；事务由 records_service 或 API 提交。

        模型失败时没有部分向量写入；外层事务回滚笔记，保持笔记与索引一致。
        """
        check_index(self.db, self.embeddings)
        chunks = chunk_text(text)
        vectors = self.embeddings.embed_documents(chunks)
        rows = [EmbeddingChunk(patient_id=patient_id, source_type=source_type, source_id=source_id,
                               chunk_text=chunk, embedding=vector,
                               meta=index_metadata(self.embeddings, {**metadata, "chunk_index": i}))
                for i, (chunk, vector) in enumerate(zip(chunks, vectors))]
        self.db.add_all(rows)
        return rows

    def search(self, patient_id: UUID, query: str, top_k: int = 5) -> list[tuple[EmbeddingChunk, float]]:
        """在指定患者范围内返回最多 top_k 个文本块，按余弦距离从近到远排列。"""
        # 查询和入库使用同一个编码器，才能在相同向量空间比较。
        query_embedding = self._query_vector(query)
        # 这里构建 SQL 表达式，由数据库执行距离计算，而不是把全部向量读回 Python。
        # pgvector 的余弦距离为 1 - 余弦相似度；距离越小，向量方向越接近。
        distance = EmbeddingChunk.embedding.cosine_distance(query_embedding)
        # WHERE 在候选集合上限制患者，再排序取 Top-K；患者隔离不能只依赖提示词。
        stmt = (
            select(EmbeddingChunk, distance.label("distance"))
            .where(EmbeddingChunk.patient_id == patient_id)
            .order_by(distance)
            .limit(top_k)
        )
        rows = self.db.execute(stmt).all()
        # 转成越大越相关的分数，并把负相似度截为 0；它不是概率或临床置信度。
        # 没有最低相关性阈值，所以只要有候选，即使相关性很低也可能被返回。
        return [(row[0], max(0.0, 1.0 - float(row[1]))) for row in rows]

    def search_all_patients(self, query: str, top_k: int = 8) -> list[tuple[EmbeddingChunk, float]]:
        """跨患者检索文本块，供 PatientContextService 组合通用病历上下文。

        本方法没有患者过滤或访问权限判断，与单患者 search 的范围不同。
        调用者必须了解这一范围；用于生产环境前需要补充相应的授权过滤。
        """
        query_embedding = self._query_vector(query)
        distance = EmbeddingChunk.embedding.cosine_distance(query_embedding)
        stmt = select(EmbeddingChunk, distance.label("distance")).order_by(distance).limit(top_k)
        rows = self.db.execute(stmt).all()
        return [(row[0], max(0.0, 1.0 - float(row[1]))) for row in rows]

    def search_knowledge(self, query: str, top_k: int = 5) -> list[tuple[KnowledgeChunk, float]]:
        """检索共享知识表，使用与患者病历相同的编码和评分方式，但不绑定患者。"""
        query_embedding = self._query_vector(query)
        distance = KnowledgeChunk.embedding.cosine_distance(query_embedding)
        stmt = select(KnowledgeChunk, distance.label("distance")).order_by(distance).limit(top_k)
        rows = self.db.execute(stmt).all()
        return [(row[0], max(0.0, 1.0 - float(row[1]))) for row in rows]
