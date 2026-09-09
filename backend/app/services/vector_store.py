from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import EmbeddingChunk, KnowledgeChunk
from app.services.embedding_service import EmbeddingService


class VectorStore:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.embeddings = EmbeddingService()

    def add_chunk(self, patient_id: UUID, source_type: str, source_id: UUID, text: str, metadata: dict) -> EmbeddingChunk:
        chunk = EmbeddingChunk(
            patient_id=patient_id,
            source_type=source_type,
            source_id=source_id,
            chunk_text=text,
            meta=metadata,
            embedding=self.embeddings.embed(text),
        )
        self.db.add(chunk)
        return chunk

    def add_knowledge_chunk(self, source_type: str, source_id: UUID, text: str, metadata: dict) -> KnowledgeChunk:
        chunk = KnowledgeChunk(
            source_type=source_type,
            source_id=source_id,
            chunk_text=text,
            meta=metadata,
            embedding=self.embeddings.embed(text),
        )
        self.db.add(chunk)
        return chunk

    def search(self, patient_id: UUID, query: str, top_k: int = 5) -> list[tuple[EmbeddingChunk, float]]:
        query_embedding = self.embeddings.embed(query)
        distance = EmbeddingChunk.embedding.cosine_distance(query_embedding)
        stmt = (
            select(EmbeddingChunk, distance.label("distance"))
            .where(EmbeddingChunk.patient_id == patient_id)
            .order_by(distance)
            .limit(top_k)
        )
        rows = self.db.execute(stmt).all()
        return [(row[0], max(0.0, 1.0 - float(row[1]))) for row in rows]

    def search_all_patients(self, query: str, top_k: int = 8) -> list[tuple[EmbeddingChunk, float]]:
        """Return chart-note evidence across the accessible patient corpus.

        This is deliberately separate from ``search``: case RAG remains strictly
        patient-scoped, while the general chart assistant needs a corpus-wide view.
        Authorization must be applied before exposing this endpoint in production.
        """
        query_embedding = self.embeddings.embed(query)
        distance = EmbeddingChunk.embedding.cosine_distance(query_embedding)
        stmt = select(EmbeddingChunk, distance.label("distance")).order_by(distance).limit(top_k)
        rows = self.db.execute(stmt).all()
        return [(row[0], max(0.0, 1.0 - float(row[1]))) for row in rows]

    def search_knowledge(self, query: str, top_k: int = 5) -> list[tuple[KnowledgeChunk, float]]:
        query_embedding = self.embeddings.embed(query)
        distance = KnowledgeChunk.embedding.cosine_distance(query_embedding)
        stmt = select(KnowledgeChunk, distance.label("distance")).order_by(distance).limit(top_k)
        rows = self.db.execute(stmt).all()
        return [(row[0], max(0.0, 1.0 - float(row[1]))) for row in rows]
