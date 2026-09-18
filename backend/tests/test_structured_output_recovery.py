"""复现提供方忽略 tool_choice、返回普通文本的情况，使用真实 Agent 图而非伪造最终状态。

确保格式重试次数有限、业务工具在重试时不可见、不会误用上一轮的答案。
"""
from uuid import uuid4
from pydantic import Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from app.services.pms_agent_service import PmsAgentService, PatientAliases
from app.services.structured_output_middleware import MAX_FORMAT_RETRIES, needs_finalization


class IrregularModel(BaseChatModel):
    """模拟提供方不遵循工具协议；只保存工具名称，不保存患者文本。"""
    responses: list[str]
    calls: int = 0
    tool_sets: list[list[str]] = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "irregular-test-model"

    def bind_tools(self, tools, **kwargs):
        self.tool_sets.append([t.get("function", {}).get("name", t.get("name", "")) if isinstance(t, dict) else t.name for t in tools])
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        mode = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        if mode == "structured":
            msg = AIMessage(content="", tool_calls=[{"id": f"answer-{self.calls}", "name": "AgentAnswer", "args": {"answer": f"Validated answer {self.calls}", "evidence_ids": [], "limitations": []}}])
        elif mode == "business":
            msg = AIMessage(content="", tool_calls=[{"id": "unwanted-call", "name": "read_records", "args": {}}])
        elif mode == "error":
            msg = AIMessage(content="", response_metadata={"finish_reason": "error"})
        else:
            msg = AIMessage(content="Plain text instead of the required output tool.", response_metadata={"finish_reason": "stop"})
        return ChatResult(generations=[ChatGeneration(message=msg)])


def make_graph(model):
    service = PmsAgentService(uuid4(), uuid4(), [], PatientAliases({}))
    invoked = []

    @tool
    def read_records() -> dict:
        """Read synthetic records for a protocol regression test."""
        invoked.append(True)
        return {"patients": []}

    service._tools = lambda: [read_records]
    return service.build(InMemorySaver(), model), invoked


def test_plain_text_is_retried_with_only_answer_tool():
    model = IrregularModel(responses=["plain", "structured"])
    graph, invoked = make_graph(model)
    result = graph.invoke({"messages": [HumanMessage(content="Hello")]}, {"configurable": {"thread_id": "format-retry"}})
    assert result["structured_response"].answer == "Validated answer 2"
    assert model.calls == 2
    assert "read_records" in model.tool_sets[0]
    assert model.tool_sets[1] == ["AgentAnswer"]
    assert graph.get_state({"configurable": {"thread_id": "format-retry"}}).values["thread_model_call_count"] == 2
    assert invoked == []


def test_persistent_plain_text_ends_with_specific_failure_after_bounded_retries():
    model = IrregularModel(responses=["plain"])
    graph, invoked = make_graph(model)
    result = graph.invoke({"messages": [HumanMessage(content="Hello")]}, {"configurable": {"thread_id": "bounded"}})
    assert model.calls == 1 + MAX_FORMAT_RETRIES
    assert result["structured_response"] is None
    assert result["output_error"] == "structured_output_missing"
    assert needs_finalization(result)
    assert invoked == []


def test_old_plain_text_checkpoint_can_finalize_without_business_tools():
    model = IrregularModel(responses=["structured"])
    graph, invoked = make_graph(model)
    config = {"configurable": {"thread_id": "legacy"}}
    graph.update_state(config, {"messages": [HumanMessage(content="Hello"), AIMessage(content="Legacy plain reply")], "structured_response": None}, as_node="StructuredOutputMiddleware.after_model")
    assert needs_finalization(graph.get_state(config).values)
    result = graph.invoke({"finalization_only": True, "structured_output_retries": 0, "output_error": None}, config)
    assert result["structured_response"].answer == "Validated answer 1"
    assert model.tool_sets == [["AgentAnswer"]]
    assert invoked == []


def test_provider_error_is_not_retried_as_a_successful_answer():
    model = IrregularModel(responses=["error"])
    graph, _ = make_graph(model)
    result = graph.invoke({"messages": [HumanMessage(content="Hello")]}, {"configurable": {"thread_id": "provider-error"}})
    assert result["output_error"] == "provider_response_error"
    assert model.calls == 1
    assert not needs_finalization(result)


def test_provider_cannot_reopen_business_tools_during_finalization():
    model = IrregularModel(responses=["plain", "business"])
    graph, invoked = make_graph(model)
    result = graph.invoke({"messages": [HumanMessage(content="Hello")]}, {"configurable": {"thread_id": "no-side-effects"}})
    assert result["output_error"] == "provider_response_error"
    assert invoked == []


def test_missing_second_answer_does_not_reuse_previous_structured_response():
    model = IrregularModel(responses=["structured", "plain"])
    graph, _ = make_graph(model)
    config = {"configurable": {"thread_id": "two-turns"}}
    graph.invoke({"messages": [HumanMessage(content="First")]}, config)
    result = graph.invoke({"messages": [HumanMessage(content="Second")], "finalization_only": False, "structured_output_retries": 0}, config)
    assert result["structured_response"] is None
    assert result["output_error"] == "structured_output_missing"
