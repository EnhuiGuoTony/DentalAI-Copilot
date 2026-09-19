"""在选定患者的每轮对话开始时读取真实记录，避免模型猜测患者是否存在。

调用现有 read_records 工具，范围仍由服务器会话 UUID 限定，不依赖 DOX 导入编号。
把真实工具结果加入本轮工具消息链，因此 checkpoint 与 SSE 都能观察到这次读取。
工具内部使用独立数据库 Session 和既有身份替换；不会信任客户端提交的病历。
"""
import json
from typing import NotRequired
from uuid import uuid4
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool


class PatientContextState(AgentState):
    """独立标记自动读取，避免长记录被摘要移除后，每次模型调用都重新加载。"""
    patient_context_loaded: NotRequired[bool]


class PatientContextMiddleware(AgentMiddleware):
    state_schema = PatientContextState

    def __init__(self, read_tool: BaseTool):
        """只在会话已选择患者时注册，复用同一个只读工具以保持数据契约一致。"""
        self.read_tool = read_tool

    def before_agent(self, state, runtime):
        """新一轮图输入才重置；从审批 checkpoint 恢复不会重新经过图入口。"""
        return {"patient_context_loaded": False}

    def before_model(self, state, runtime):
        """每轮最多自动读取一次；审批恢复和最终答案补全不重放已有工具链。"""
        if state.get("finalization_only") or state.get("patient_context_loaded"):
            return None
        # 先成功读取，再记录消息；异常交给统一错误处理，不能伪造成功的工具返回。
        result = self.read_tool.invoke({})
        call_id = "selected-records-" + uuid4().hex
        return {"patient_context_loaded": True, "messages": [
            AIMessage(content="", tool_calls=[{"name": "read_records", "args": {}, "id": call_id}]),
            ToolMessage(name="read_records", tool_call_id=call_id, content=json.dumps(result, ensure_ascii=False)),
        ]}
