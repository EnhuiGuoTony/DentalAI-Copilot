from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.services.llm_service import LlmService
from app.db.session import get_db
from app.schemas.chat import ChatConnectionResponse, ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat/connect", response_model=ChatConnectionResponse)
def connect_chat() -> ChatConnectionResponse:
    """只检查模型客户端配置，不生成文本，也不验证远端服务是否实际可达。"""
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
    """旧接口退役，客户端需使用有持久化和审批语义的会话 API。"""
    raise HTTPException(410, "Use /api/conversations for persisted, approval-aware streaming chat")
