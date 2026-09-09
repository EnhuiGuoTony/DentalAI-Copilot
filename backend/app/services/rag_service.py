from uuid import UUID

from sqlalchemy.orm import Session

from app.schemas.rag import EvidenceItem, RagQueryResponse
from app.services.vector_store import VectorStore


class RagService:
    def __init__(self, db: Session) -> None:
        self.vector_store = VectorStore(db)

    def query(self, patient_id: UUID, question: str, top_k: int = 5) -> RagQueryResponse:
        patient_results = self.vector_store.search(patient_id, question, top_k)
        knowledge_results = self.vector_store.search_knowledge(question, max(2, top_k // 2))
        results = [*patient_results, *knowledge_results]
        results.sort(key=lambda item: item[1], reverse=True)
        results = results[:top_k]
        evidence = [
            EvidenceItem(
                source_type=chunk.source_type,
                source_id=chunk.source_id,
                snippet=chunk.chunk_text[:360],
                score=round(score, 4),
                metadata=chunk.meta,
            )
            for chunk, score in results
        ]
        answer = self._compose_mock_answer(question, evidence)
        return RagQueryResponse(
            answer=answer,
            evidence=evidence,
            limitations=[
                "RAG evidence is limited to the demo patient's indexed notes.",
                "AI output is for clinician review and is not a definitive diagnosis.",
            ],
        )

    @staticmethod
    def _compose_mock_answer(question: str, evidence: list[EvidenceItem]) -> str:
        if not evidence:
            return f"No indexed patient history was found for: {question}"
        top = evidence[0].snippet
        return f"Relevant patient context found for '{question}': {top}"
