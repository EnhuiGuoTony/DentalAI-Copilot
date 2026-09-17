from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.services.llm_service import LlmService
from app.db.session import get_db
from app.schemas.chat import ChatConnectionResponse, ChatRequest, ChatResponse
from app.services.pms_agent_service import PmsAgentService

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
    """聊天入口：将用户问题、选中患者和历史交给 Agent，再封装本轮结果。"""
    # db 由 FastAPI 依赖注入管理；本路由不把数据库访问能力直接交给模型。
    # 模型只能通过服务中显式注册的工具间接读取数据。
    pms_agent = PmsAgentService(db)
    reply, trace, token_usage = pms_agent.run(request.message, request.patient_ids, request.history)
    return ChatResponse(
        reply=reply,
        provider="langchain:pms-agent",
        mock=not pms_agent.llm.enabled,
        tool_trace=trace,
        token_usage=token_usage,
    )
