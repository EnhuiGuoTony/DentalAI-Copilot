from pydantic import BaseModel, Field
from uuid import UUID

MAX_CHAT_MESSAGE_CHARS = 4_000
MAX_CHAT_HISTORY_MESSAGES = 6


class ChatMessage(BaseModel):
    # System prompts are owned by the server, never supplied by the browser.
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=MAX_CHAT_MESSAGE_CHARS)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_CHAT_MESSAGE_CHARS)
    history: list[ChatMessage] = Field(default_factory=list, max_length=MAX_CHAT_HISTORY_MESSAGES)
    patient_ids: list[UUID] = Field(default_factory=list, max_length=10)


class ChatResponse(BaseModel):
    reply: str
    provider: str
    mock: bool
    tool_trace: list[str] = Field(default_factory=list)


class ChatConnectionResponse(BaseModel):
    connected: bool
    provider: str
    mock: bool
    detail: str
