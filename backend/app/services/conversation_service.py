"""拥有者校验、checkpoint 恢复和 SSE 状态输出。

PostgreSQL advisory lock 跨进程串行化同一会话；断流会释放锁但不会回滚已提交工具。
再次读取会话可看到已保存的中断/结果，未结束的图通过 continue 恢复，写工具回执防重放。
"""
import json
import logging
import traceback
from contextlib import contextmanager
from uuid import UUID
from fastapi import HTTPException
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langgraph.types import Command
from sqlalchemy import select, text
from app.db.models import Conversation, ApprovalAudit
from app.db.session import SessionLocal, engine
from app.schemas.conversation import PendingReview, ReviewAction, ResumeRequest, StreamEvent
from app.schemas.operations import AgentAnswer
from app.services.agent_memory import memory
from app.services.pms_agent_service import PatientAliases, PmsAgentService
from app.services.structured_output_middleware import needs_finalization
from app.services.workspace_lock import workspace_lock


class DemoModel(BaseChatModel):
    """明确标注的无网络 mock；仅返回结构化说明，不假装理解并执行自然语言写请求。"""
    @property
    def _llm_type(self):
        return "dentalai-demo"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        answer = AgentAnswer(answer="当前是 Mock 模式：会话已保存，但不会解析或执行自然语言操作。请配置支持工具调用的模型。", limitations=["Mock model; no clinical reasoning or record changes."])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[{
            "name": "AgentAnswer", "args": answer.model_dump(), "id": "mock-answer"}]))])


def owned_conversation(db, conversation_id: UUID, user_id: UUID) -> Conversation:
    """返回 404 隐藏其他账号的会话是否存在；不能依赖不可猜测 UUID 代替授权。"""
    row = db.scalar(select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user_id))
    if row is None:
        raise HTTPException(404, "Conversation not found")
    return row


@contextmanager
def conversation_lock(conversation_id: UUID):
    # 会话锁使用专用连接，所有退出路径都解锁；锁在不同 API worker 间同样有效。
    key = int.from_bytes(conversation_id.bytes[:8], "big", signed=True)
    # advisory lock 属于连接，无需持有数据库事务。长时间的模型调用若保留事务快照，
    # 会阻塞 CREATE INDEX CONCURRENTLY 等维护操作，甚至使 checkpoint 初始化一直等待。
    # FastAPI 的请求依赖可能在 SSE 迭代前释放；这里的共享锁覆盖模型、工具和 checkpoint。
    with workspace_lock(engine), engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        locked = connection.scalar(text("SELECT pg_try_advisory_lock(:key)"), {"key": key})
        if not locked:
            raise HTTPException(409, "This conversation is already running")
        try:
            yield
        finally:
            connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": key})


def pending_review(snapshot, aliases: PatientAliases) -> PendingReview | None:
    interrupts = [i for task in snapshot.tasks for i in task.interrupts]
    if not interrupts:
        return None
    item = interrupts[0]
    return PendingReview(interrupt_id=item.id, actions=[ReviewAction(name=action["name"], arguments=aliases.transform(action["args"], reveal=True), description=aliases.reveal(action.get("description", "")))
        for action in item.value["action_requests"]])


@contextmanager
def graph_context(conversation_id: UUID, user_id: UUID):
    with SessionLocal() as db:
        row = owned_conversation(db, conversation_id, user_id)
        service = PmsAgentService(row.id, user_id, [UUID(v) for v in row.patient_ids], PatientAliases(row.privacy_map))
    with memory() as saver:
        graph = service.build(saver, model=DemoModel() if not service.llm.enabled else None)
        config = {"configurable": {"thread_id": str(conversation_id)}, "recursion_limit": 50, "max_concurrency": 1}
        yield graph, config, service


def conversation_state(conversation_id: UUID, user_id: UUID) -> dict:
    with graph_context(conversation_id, user_id) as (graph, config, service):
        snapshot = graph.get_state(config)
        review = pending_review(snapshot, service.aliases)
        transcript = []
        messages = snapshot.values.get("messages", [])
        # 结构化工具可能因校验失败重试；只展示有成功工具回执的答案，不能展示被拒绝的草稿。
        accepted = {m.tool_call_id for m in messages if isinstance(m, ToolMessage) and m.name == "AgentAnswer" and m.status != "error"}
        for msg in messages:
            if isinstance(msg, HumanMessage):
                transcript.append({"role": "user", "content": service.aliases.reveal(str(msg.content))})
            elif isinstance(msg, AIMessage):
                for call in msg.tool_calls:
                    if call["name"] == "AgentAnswer" and call["id"] in accepted:
                        answer = AgentAnswer.model_validate(call["args"])
                        transcript.append({"role": "assistant", "content": service.aliases.reveal(answer.answer)})
        return {"id": str(conversation_id), "messages": transcript,
                "review": review.model_dump(mode="json") if review else None,
                "can_continue": bool(snapshot.next or needs_finalization(snapshot.values)) and not review}


def stream_turn(conversation_id: UUID, user_id: UUID, message: str | None = None, resume: ResumeRequest | None = None):
    """SSE 每条消息是一行 JSON；只发布阶段/工具事件和校验后的最终答案。

    不直接透传 provider token，因为结构化输出的中间 token 可能是无效 JSON 或尚未脱敏的数据。
    审批只接收决定，不接收修改后的工具参数；UI 显示内容来自服务器 checkpoint。
    """
    def encode(event: StreamEvent):
        return f"data: {event.model_dump_json()}\n\n"

    try:
        with conversation_lock(conversation_id), graph_context(conversation_id, user_id) as (graph, config, service):
            snapshot = graph.get_state(config)
            review = pending_review(snapshot, service.aliases)
            if resume:
                if review is None or review.interrupt_id != resume.interrupt_id or len(review.actions) != len(resume.decisions):
                    raise HTTPException(409, "Approval is stale or decisions do not match the pending actions")
                decisions = [d.model_dump() for d in resume.decisions]
                with SessionLocal() as db:
                    audit = db.scalar(select(ApprovalAudit).where(ApprovalAudit.conversation_id == conversation_id, ApprovalAudit.interrupt_id == resume.interrupt_id))
                    if audit and audit.decisions != decisions:
                        raise HTTPException(409, "This approval already has a different recorded decision")
                    if not audit:
                        db.add(ApprovalAudit(conversation_id=conversation_id, user_id=user_id, interrupt_id=resume.interrupt_id, decisions=decisions))
                        db.commit()
                graph_input = Command(resume={"decisions": [dict(d, **({"message": "User rejected this action. Do not retry it."} if d["type"] == "reject" else {})) for d in decisions]})
            elif message is not None:
                if review or snapshot.next:
                    raise HTTPException(409, "Resolve or continue the existing run before sending another message")
                graph_input = {"messages": [HumanMessage(content=service.aliases.hide(message))],
                               "finalization_only": False, "structured_output_retries": 0, "output_error": None}
            else:
                if review or (not snapshot.next and not needs_finalization(snapshot.values)):
                    raise HTTPException(409, "No interrupted execution is available to continue")
                # 旧图已结束但缺少结构化结果时，只重新生成最终答案，绝不重放整轮业务操作。
                graph_input = None if snapshot.next else {"finalization_only": True,
                    "structured_output_retries": 0, "output_error": None}
            yield encode(StreamEvent(type="status", message="正在恢复执行" if resume or message is None else "正在处理请求", mock=not service.llm.enabled))
            for update in graph.stream(graph_input, config=config, stream_mode="updates"):
                for node, values in update.items():
                    if node == "__interrupt__":
                        continue
                    if isinstance(values, dict):
                        for msg in values.get("messages", []):
                            if isinstance(msg, AIMessage):
                                for call in msg.tool_calls:
                                    if call["name"] != "AgentAnswer":
                                        yield encode(StreamEvent(type="tool", tool_name=call["name"], message="已提出工具调用，写操作需要审批"))
                            elif isinstance(msg, ToolMessage) and msg.name != "AgentAnswer":
                                # 完整参数只出现在审批卡片；状态不泄漏患者原文。
                                label = "工具已返回结果"
                                if msg.status == "error":
                                    label = "工具调用失败，未得到有效结果"
                                if msg.name == "change_records":
                                    try:
                                        outcome = json.loads(str(msg.content))
                                    except (ValueError, TypeError):
                                        outcome = {}
                                    if isinstance(outcome, dict) and outcome.get("status") == "applied":
                                        label = "审批后的操作已成功提交"
                                    elif isinstance(outcome, dict) and outcome.get("status") == "failed":
                                        label = "操作未执行：记录冲突、不存在或超出会话范围；请重新读取"
                                    elif "User rejected this action." in str(msg.content):
                                        label = "用户已拒绝，未执行操作"
                                yield encode(StreamEvent(type="tool", tool_name=msg.name, message=label))
                    if node == "model":
                        yield encode(StreamEvent(type="status", message="模型步骤完成"))
                    elif node == "StructuredOutputMiddleware.after_model" and isinstance(values, dict) and values.get("jump_to") == "model":
                        yield encode(StreamEvent(type="status", message="正在纠正模型工具调用格式"))
            snapshot = graph.get_state(config)
            review = pending_review(snapshot, service.aliases)
            if review:
                yield encode(StreamEvent(type="approval", message="等待你的审批，尚未执行写操作", review=review))
            else:
                raw = snapshot.values.get("structured_response")
                if raw is None:
                    code = snapshot.values.get("output_error") or "structured_output_missing"
                    detail = ("模型提供方返回了错误或无效响应，请稍后重新发送。"
                              if code == "provider_response_error" else
                              "模型未按要求返回结构化答案。请刷新会话后点击继续，仅补全回答；若仍失败，请更换支持工具调用的模型。")
                    yield encode(StreamEvent(type="error", message=detail, error_code=code))
                    return
                answer = AgentAnswer.model_validate(raw)
                answer = AgentAnswer.model_validate(service.aliases.transform(answer.model_dump(), reveal=True))
                yield encode(StreamEvent(type="result", result=answer, mock=not service.llm.enabled))
    except HTTPException as exc:
        yield encode(StreamEvent(type="error", message=str(exc.detail)))
    except Exception as exc:
        # 不把 provider 异常/SQL 参数返回浏览器，避免异常携带密钥或患者内容。
        frames = [(frame.name, frame.lineno) for frame in traceback.extract_tb(exc.__traceback__)]
        logging.getLogger(__name__).error("Agent failure type=%s frames=%s", type(exc).__name__, frames)
        yield encode(StreamEvent(type="error", message="Agent 执行失败。请刷新会话查看已保存的状态；已提交的操作不会撤销。"))
