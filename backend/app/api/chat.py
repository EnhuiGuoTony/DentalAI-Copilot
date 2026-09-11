from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.services.llm_service import LlmService
from app.db.session import get_db
from app.schemas.chat import ChatConnectionResponse, ChatRequest, ChatResponse
from app.services.pms_agent_service import PmsAgentService

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
    pms_agent = PmsAgentService(db)
    reply, trace, token_usage = pms_agent.run(request.message, request.patient_ids, request.history)
    return ChatResponse(
        reply=reply,
        provider="langchain:pms-agent",
        mock=not pms_agent.llm.enabled,
        tool_trace=trace,
        token_usage=token_usage,
    )
