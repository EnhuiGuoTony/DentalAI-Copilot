"""流式会话协议：只传本轮输入，不接受客户端伪造历史、角色或审批参数。"""
from typing import Literal
from uuid import UUID
from pydantic import BaseModel, Field
from app.schemas.operations import StrictModel, AgentAnswer


class ConversationCreate(StrictModel):
    patient_ids: list[UUID] = Field(default_factory=list, max_length=10)


class TurnRequest(StrictModel):
    message: str = Field(min_length=1, max_length=4000)


class Decision(StrictModel):
    type: Literal["approve", "reject"]


class ResumeRequest(StrictModel):
    interrupt_id: str
    decisions: list[Decision] = Field(min_length=1, max_length=20)


class ReviewAction(BaseModel):
    name: str
    arguments: dict
    description: str = ""


class PendingReview(BaseModel):
    interrupt_id: str
    actions: list[ReviewAction]


class StreamEvent(BaseModel):
    """只暴露可观测执行状态，不暴露模型隐藏推理或任意 checkpoint 内容。"""
    type: Literal["status", "tool", "approval", "result", "error"]
    message: str = ""
    tool_name: str | None = None
    review: PendingReview | None = None
    result: AgentAnswer | None = None
    mock: bool = False
