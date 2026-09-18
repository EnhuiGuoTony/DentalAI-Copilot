"""处理工具型模型忽略结构化输出约束的情况，而不是把正常文本误当成已校验答案。

create_agent 在有业务工具且模型未调用工具时可以结束循环，即使配置了 ToolStrategy。
本中间件最多纠正两次，每次都经过模型调用计数和 PII 中间件；纠正期间只开放
AgentAnswer，不重新开放业务工具。状态存进 checkpoint，刷新/重启仍可解释执行阶段。
"""
from typing import NotRequired
from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse, hook_config
from langchain_core.messages import AIMessage, SystemMessage

FORMAT_INSTRUCTION = (
    "For this response, complete only the final answer using the AgentAnswer tool. "
    "Do not reply with plain text or a JSON code block. Include answer, evidence_ids and limitations. "
    "Preserve the user's language. Use only the existing conversation and recorded tool results. "
    "Do not perform, request or repeat any business operation. "
    "Do not claim that a proposed, rejected or failed operation succeeded."
)
MAX_FORMAT_RETRIES = 2


class StructuredOutputState(AgentState):
    """服务端控制的恢复状态，不接受客户端传入，也不是模型生成的业务字段。"""
    finalization_only: NotRequired[bool]
    structured_output_retries: NotRequired[int]
    output_error: NotRequired[str | None]


class StructuredOutputMiddleware(AgentMiddleware):
    state_schema = StructuredOutputState

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        """只在纠正/恢复答案时移除业务工具；常规读写和 HITL 路径保持不变。"""
        if not request.state.get("finalization_only", False):
            return handler(request)
        original = request.system_message.text if request.system_message else ""
        response = handler(request.override(
            tools=[], system_message=SystemMessage(content=original + "\n\n" + FORMAT_INSTRUCTION)))
        # 即使提供方忽略 tools=[] 并捏造业务调用，也不能让它进入工具节点。
        if any(isinstance(m, AIMessage) and any(c["name"] != "AgentAnswer" for c in m.tool_calls)
               for m in response.result):
            return ModelResponse(result=[AIMessage(content="", response_metadata={"finish_reason": "error"})])
        return response

    @hook_config(can_jump_to=["model", "end"])
    def after_model(self, state: StructuredOutputState, runtime):
        """普通文本进入受限纠正循环；提供方错误或达到次数上限则明确结束，不无限重试。"""
        if state.get("structured_response") is not None:
            return {"output_error": None, "finalization_only": False, "structured_output_retries": 0}
        last = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
        if last is not None and last.tool_calls:
            # 业务调用交回原图，写操作仍由 HumanInTheLoopMiddleware 中断。
            return None
        if last is None or last.response_metadata.get("finish_reason") in {"error", "content_filter"}:
            return {"output_error": "provider_response_error", "jump_to": "end"}
        retries = state.get("structured_output_retries", 0)
        if retries >= MAX_FORMAT_RETRIES:
            return {"output_error": "structured_output_missing", "jump_to": "end"}
        return {"structured_output_retries": retries + 1, "finalization_only": True,
                "output_error": None, "jump_to": "model"}


def needs_finalization(values: dict) -> bool:
    """识别旧版本遗留的普通文本结尾，允许恢复答案，但不自动恢复未执行的业务调用。"""
    if values.get("structured_response") is not None:
        return False
    messages = values.get("messages", [])
    if not messages or not isinstance(messages[-1], AIMessage):
        return False
    last = messages[-1]
    return bool(last.content) and not last.tool_calls and not last.invalid_tool_calls and last.response_metadata.get("finish_reason") not in {"error", "content_filter"}
