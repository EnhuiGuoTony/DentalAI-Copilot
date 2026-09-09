from uuid import UUID

from pydantic import BaseModel


class EvidenceItem(BaseModel):
    source_type: str
    source_id: UUID
    snippet: str
    score: float
    metadata: dict


class RagQueryRequest(BaseModel):
    question: str
    top_k: int = 5


class RagQueryResponse(BaseModel):
    answer: str
    evidence: list[EvidenceItem]
    limitations: list[str]

