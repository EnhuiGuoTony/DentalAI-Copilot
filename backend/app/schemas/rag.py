from uuid import UUID

from pydantic import BaseModel


class EvidenceItem(BaseModel):
    # 来源字段用于追溯，snippet 是截断后的展示文本，不保证包含完整原始记录。
    source_type: str
    source_id: UUID
    snippet: str
    # 检索相似度分数，用于排序；不是模型置信度，也不是诊断概率。
    score: float
    metadata: dict


class RagQueryRequest(BaseModel):
    # top_k 控制最终证据条数，当前 Schema 没有限制其正负或最大值。
    question: str
    top_k: int = 5


class RagQueryResponse(BaseModel):
    # 把回答、证据与局限分开，方便界面展示；目前 answer 来自演示模板。
    answer: str
    evidence: list[EvidenceItem]
    limitations: list[str]
