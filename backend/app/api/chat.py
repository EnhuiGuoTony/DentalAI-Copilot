from fastapi import APIRouter

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm_service import LlmService

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    llm = LlmService()
    return ChatResponse(
        reply=llm.chat(request.message, request.history),
        provider=llm.provider_label(),
        mock=not llm.enabled,
    )
