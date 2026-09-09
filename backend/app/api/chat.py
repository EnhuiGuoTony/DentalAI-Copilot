from fastapi import APIRouter

from app.schemas.chat import ChatConnectionResponse, ChatRequest, ChatResponse
from app.services.llm_service import LlmService

router = APIRouter(tags=["chat"])


@router.post("/chat/connect", response_model=ChatConnectionResponse)
def connect_chat() -> ChatConnectionResponse:
    """Initialize and validate the configured provider without generating text."""
    llm = LlmService()
    connected, detail = llm.connect()
    return ChatConnectionResponse(
        connected=connected,
        provider=llm.provider_label(),
        mock=not llm.enabled,
        detail=detail,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    llm = LlmService()
    return ChatResponse(
        reply=llm.chat(request.message, request.history),
        provider=llm.provider_label(),
        mock=not llm.enabled,
    )
