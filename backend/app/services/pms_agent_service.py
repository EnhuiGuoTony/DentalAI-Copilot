from typing import Any
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from app.db.models import ClinicalFact, ClinicalNote, Patient
from app.services.llm_service import LlmService
from app.services.llm_privacy_service import LlmPrivacyService
from app.services.vector_store import VectorStore
from app.schemas.chat import ChatMessage, MAX_CHAT_HISTORY_MESSAGES


class PmsAgentService:
    """当前 /chat 使用的工具型 Agent（智能体），围绕消息历史完成一轮问答。

    主链路：解析患者 -> 构造受限工具 -> 模型决定调用哪些工具 -> 执行工具 ->
    将工具结果交回模型 -> 返回回答、工具名称轨迹和 token 用量。
    create_agent 负责组织模型与工具之间的循环，本文件没有手写 LangGraph 节点。
    返回的是聊天文本，不是 schemas/agent.py 中的结构化临床报告。
    """
    def __init__(self, db: Session) -> None:
        self.db, self.llm = db, LlmService()

    def run(
        self, question: str, selected_ids: list[Any], history: list[ChatMessage] | None = None
    ) -> tuple[str, list[str], dict[str, int]]:
        """处理本轮问题；数据库 Session 由 API 注入，历史消息由调用方传入。"""
        patients = self._resolve(question, selected_ids)
        # 匿名别名仅在本轮有效，不是患者永久编号，也不是访问权限凭证。
        aliases = {p.id: f"PATIENT_{i + 1}" for i, p in enumerate(patients)}
        safe_question = question
        # 当前问题只替换已解析患者的这些标识，属于精确字符串替换，非全面脱敏。
        # 不同大小写、其他格式或未解析患者的信息可能仍然保留。
        for p in patients:
            for value in (p.name, p.dox_patient_id, p.patient_number, p.medical_record_number):
                if value: safe_question = safe_question.replace(str(value), aliases[p.id])
        # 工具闭包捕获本轮患者范围，模型不能通过工具参数任意指定另一位患者 ID。
        tools = self._tools(patients, aliases)
        if not self.llm.enabled:
            reply = "请配置支持工具调用的模型。" if not patients else "已解析患者：" + "、".join(p.name for p in patients)
            return reply, [], self._empty_token_usage()
        # 未启用真实模型时，上面的分支直接返回，不会进入模型/工具循环。
        model = self.llm._chat_model()
        # required 表达必须调用工具，parallel_tool_calls 允许模型提出并行工具调用。
        # 具体支持情况取决于提供方和框架；当前未显式配置重试或循环终止策略，
        # 也未在此处把 required 切回 auto，不能将它理解为“仅第一轮强制调用”。
        if patients: model = model.bind_tools(tools, tool_choice="required", parallel_tool_calls=True)
        # 英文系统提示词规定行为；工具名称、参数类型和英文 docstring 构成工具说明。
        # 数据库事实来自工具，模型负责选择工具与组织回答，不能依靠模型记忆编造记录。
        agent = create_agent(model=model, tools=tools, system_prompt=(
            "You are a dental PMS assistant. Never invent records. Use patient tools for patient questions. "
            "For chart summaries call profiles, treatments/diagnoses, and recent notes. For general dental questions, "
            "use clinical knowledge search when evidence improves the answer. Patient references are anonymous. "
            "Reply in the user's language and never give a final diagnosis."))
        # 这里只带入最近几条消息，不是持久化 Agent 记忆；没有配置 checkpointer。
        # HumanMessage/AIMessage 分别表示用户与助手，工具消息由运行中的 Agent 维护。
        history_messages = []
        for item in (history or [])[-MAX_CHAT_HISTORY_MESSAGES:]:
            content = item.content
            # 历史消息目前仅替换患者姓名，覆盖范围比本轮问题的替换更窄。
            for patient in patients:
                if patient.name:
                    content = content.replace(patient.name, aliases[patient.id])
            history_messages.append(
                HumanMessage(content=content) if item.role == "user" else AIMessage(content=content)
            )
        # invoke 是同步执行：等待模型与工具循环结束后，得到包含中间消息的最终状态。
        result = agent.invoke({"messages": [*history_messages, HumanMessage(content=safe_question)]})
        messages = result["messages"]
        answer = str(messages[-1].content)
        # 此轨迹提取模型提出的工具调用名称，不含参数、执行结果或成功状态，
        # 因此不是完整审计日志，也不是模型内部的推理过程。
        trace = [call["name"] for m in messages for call in getattr(m, "tool_calls", [])]
        # 在服务端把别名还原成姓名再返回界面；这一轮不会在此处保存 AgentRun。
        for p in patients: answer = answer.replace(aliases[p.id], p.name)
        return answer, trace, self._token_usage(messages)

    @staticmethod
    def _empty_token_usage() -> dict[str, int]:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def _token_usage(self, messages: list[Any]) -> dict[str, int]:
        """累加返回消息中的模型用量；一轮用户问答可能触发多次模型请求。"""
        usage = self._empty_token_usage()
        for message in messages:
            metadata = getattr(message, "usage_metadata", None) or {}
            response_metadata = getattr(message, "response_metadata", None) or {}
            provider_usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
            # 优先使用 LangChain 统一字段，缺失时兼容提供方原始字段；不会自行估算。
            source = metadata or provider_usage
            usage["input_tokens"] += self._usage_value(source, "input_tokens", "prompt_tokens", "inputTokenCount")
            usage["output_tokens"] += self._usage_value(source, "output_tokens", "completion_tokens", "candidatesTokenCount")
            usage["total_tokens"] += self._usage_value(source, "total_tokens", "totalTokenCount")
        if not usage["total_tokens"]:
            usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        return usage

    @staticmethod
    def _usage_value(source: dict[str, Any], *keys: str) -> int:
        """按优先级读取同一指标的不同命名；缺失的统计值记为 0。"""
        for key in keys:
            value = source.get(key)
            if isinstance(value, (int, float)):
                return int(value)
        return 0

    def _tools(self, patients: list[Patient], aliases: dict[Any, str]):
        """为本轮构造工具集合；@tool 下的英文文档字符串会作为说明发送给模型。"""
        privacy = LlmPrivacyService(self.db)
        # 共享知识检索也是一个工具：模型生成查询词，VectorStore 执行召回。
        # 返回完整 chunk_text；这里没有经过 privacy.redact，依赖共享语料本身的边界。
        @tool
        def search_clinical_knowledge(query: str) -> list[dict]:
            """Search the global dental knowledge RAG corpus."""
            return [{"source_type": c.source_type, "snippet": c.chunk_text, "score": round(s, 4)} for c, s in VectorStore(self.db).search_knowledge(query, 6)]
        # 精确数量应由 SQL COUNT 给出：Top-K 检索只得到部分记录，不能用来推断总数。
        # 此统计工具当前统计全库，不受本轮所选患者列表限制。
        @tool
        def get_pms_statistics() -> dict:
            """Get exact PMS counts."""
            return {"patients": int(self.db.scalar(select(func.count()).select_from(Patient)) or 0), "notes": int(self.db.scalar(select(func.count()).select_from(ClinicalNote)) or 0), "clinical_facts": int(self.db.scalar(select(func.count()).select_from(ClinicalFact)) or 0)}
        tools = [search_clinical_knowledge, get_pms_statistics]
        # 没解析到患者时，只开放全局知识和统计，不开放患者详情工具。
        if not patients: return tools
        ids = [p.id for p in patients]
        # 下列患者工具不接受 patient_id 参数，查询范围由后端闭包中的 ids 固定。
        @tool
        def get_patient_profiles() -> list[dict]:
            """Get selected anonymized patient profiles."""
            return privacy.redact([{"patient": aliases[p.id], "date_of_birth": p.date_of_birth, "address": p.address, "dox_patient_id": p.dox_patient_id, "patient_number": p.patient_number, "medical_record_number": p.medical_record_number} for p in patients])
        # 结构化事实适合直接筛选，避免把诊断/治疗记录全部当作自由文本向量检索。
        # 120 条是所有选中患者合计的上限，不是每位患者各 120 条。
        @tool
        def get_treatments_and_diagnoses() -> list[dict]:
            """Get selected patients' diagnosis, treatment and periodontal facts."""
            facts = self.db.execute(select(ClinicalFact).where(ClinicalFact.patient_id.in_(ids)).order_by(ClinicalFact.effective_at.desc().nullslast()).limit(120)).scalars().all()
            return privacy.redact([{"patient": aliases[f.patient_id], "type": f.fact_type, "label": f.label, "summary": f.summary, "date": f.effective_at} for f in facts])
        # 按时间读取与按语义检索互补：近期记录不一定是与问题最相关的记录。
        @tool
        def get_recent_patient_notes() -> list[dict]:
            """Get up to 60 recent notes for selected patients."""
            notes = self.db.execute(select(ClinicalNote).where(ClinicalNote.patient_id.in_(ids)).order_by(ClinicalNote.created_at.desc()).limit(60)).scalars().all()
            return privacy.redact([{"patient": aliases[n.patient_id], "type": n.note_type, "content": n.content, "date": n.created_at} for n in notes])
        # 每位患者先召回最多 5 条，再跨这些候选按分数取最多 12 条；
        # 这是两阶段截断，不等价于对所有所选患者的全部文本一次性取全局 Top-12。
        @tool
        def search_patient_notes(query: str) -> list[dict]:
            """RAG-search selected patients' notes."""
            out = []
            store = VectorStore(self.db)
            for p in patients: out += [{"patient": aliases[p.id], "score": round(s,4), "snippet": c.chunk_text} for c,s in store.search(p.id, query, 5)]
            return privacy.redact(sorted(out, key=lambda x: x["score"], reverse=True)[:12])
        return [*tools, get_patient_profiles, get_treatments_and_diagnoses, get_recent_patient_notes, search_patient_notes]

    def _resolve(self, question: str, ids: list[Any]) -> list[Patient]:
        """优先采用有效的显式选择；否则扫描患者标识并在问题中做子串匹配。

        这是 Demo 的规则匹配，不是模型实体识别，也没有授权校验。
        全表扫描不适合大规模数据，短编号的子串匹配也可能产生误匹配。
        """
        selected = list(self.db.execute(select(Patient).where(Patient.id.in_(ids))).scalars()) if ids else []
        if selected: return selected
        q = question.casefold()
        return [p for p in self.db.execute(select(Patient)).scalars() if any(v and str(v).casefold() in q for v in (p.name, p.dox_patient_id, p.patient_number, p.medical_record_number))]
