from pydantic import BaseModel, Field
from uuid import UUID

# 字符数与消息条数限制控制请求体积，但不能精确代表模型的 token 消耗。
MAX_CHAT_MESSAGE_CHARS = 4_000
MAX_CHAT_HISTORY_MESSAGES = 4


class ChatMessage(BaseModel):
    # 浏览器只能提交用户/助手角色，系统提示词由服务端持有。
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=MAX_CHAT_MESSAGE_CHARS)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_CHAT_MESSAGE_CHARS)
    history: list[ChatMessage] = Field(default_factory=list, max_length=MAX_CHAT_HISTORY_MESSAGES)
    # 显式选择优先于问题中的姓名匹配，Agent 据此构造患者工具的查询范围。
    patient_ids: list[UUID] = Field(default_factory=list, max_length=10)


class ChatResponse(BaseModel):
    reply: str
    provider: str
    mock: bool
    # 只记录工具名称；不是完整工具输入/输出，也不表示每次调用已经成功。
    tool_trace: list[str] = Field(default_factory=list)
    token_usage: "TokenUsage"


class TokenUsage(BaseModel):
    """一轮 Agent 问答中所有返回模型消息的累计用量；缺失的指标按 0 处理。"""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ChatConnectionResponse(BaseModel):
    connected: bool
    provider: str
    mock: bool
    detail: str
