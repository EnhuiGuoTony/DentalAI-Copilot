from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from app.schemas.rag import EvidenceItem


class AgentRunRequest(BaseModel):
    # 以下是结构化临床报告的数据契约；当前聊天 Agent 返回 ChatResponse，
    # 并没有调用这些模型进行 with_structured_output 或保存 AgentRun。
    question: str = "Generate an evidence-backed clinical summary for this case."


class SuspectedFinding(BaseModel):
    category: str
    confidence: float
    tooth_number: str | None = None
    evidence_refs: list[str] = []


class AgentOutput(BaseModel):
    # Schema 约束数据形状，不验证医学结论；字段齐全不代表报告已经过临床验证。
    clinical_summary: str
    suspected_findings: list[SuspectedFinding]
    relevant_history: list[str]
    recommended_next_steps: list[str]
    patient_friendly_explanation: str
    limitations: list[str]


class ToolTraceItem(BaseModel):
    # 这里描述的完整轨迹比聊天接口的工具名称列表更丰富，二者不可混为一谈。
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
