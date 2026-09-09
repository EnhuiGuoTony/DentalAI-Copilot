from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.chat import ChatConnectionResponse, ChatRequest, ChatResponse
from app.services.llm_service import LlmService
from app.services.patient_context_service import PatientContextService

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
def chat(request: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    llm = LlmService()
    context_service = PatientContextService(db)
    context = context_service.build(request.message)
    if not llm.enabled or context_service.requires_deterministic_answer(request.message):
        reply = context_service.mock_answer(context)
    else:
        reply = llm.chat(request.message, request.history, patient_context=context)
        if context_service.is_unhelpful_model_reply(reply):
            reply = context_service.mock_answer(context)
    return ChatResponse(
        reply=reply,
        provider=llm.provider_label(),
        mock=not llm.enabled,
    )
