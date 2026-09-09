from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from app.schemas.rag import EvidenceItem


class AgentRunRequest(BaseModel):
    question: str = "Generate an evidence-backed clinical summary for this case."


class SuspectedFinding(BaseModel):
    category: str
    confidence: float
    tooth_number: str | None = None
    evidence_refs: list[str] = []


class AgentOutput(BaseModel):
    clinical_summary: str
    suspected_findings: list[SuspectedFinding]
    relevant_history: list[str]
    recommended_next_steps: list[str]
    patient_friendly_explanation: str
    limitations: list[str]


class ToolTraceItem(BaseModel):
    tool: str
    input: dict
    output_summary: str


class AgentRunResponse(BaseModel):
    id: UUID
    patient_id: UUID
    case_id: UUID
    question: str
    output: AgentOutput
    evidence: list[EvidenceItem]
    tool_trace: list[ToolTraceItem]
    created_at: datetime

