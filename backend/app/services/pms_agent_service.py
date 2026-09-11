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
    def __init__(self, db: Session) -> None:
        self.db, self.llm = db, LlmService()

    def run(
        self, question: str, selected_ids: list[Any], history: list[ChatMessage] | None = None
    ) -> tuple[str, list[str], dict[str, int]]:
        patients = self._resolve(question, selected_ids)
        aliases = {p.id: f"PATIENT_{i + 1}" for i, p in enumerate(patients)}
        safe_question = question
        for p in patients:
            for value in (p.name, p.dox_patient_id, p.patient_number, p.medical_record_number):
                if value: safe_question = safe_question.replace(str(value), aliases[p.id])
        tools = self._tools(patients, aliases)
        if not self.llm.enabled:
            reply = "请配置支持工具调用的模型。" if not patients else "已解析患者：" + "、".join(p.name for p in patients)
            return reply, [], self._empty_token_usage()
        model = self.llm._chat_model()
        if patients: model = model.bind_tools(tools, tool_choice="required", parallel_tool_calls=True)
        agent = create_agent(model=model, tools=tools, system_prompt=(
            "You are a dental PMS assistant. Never invent records. Use patient tools for patient questions. "
            "For chart summaries call profiles, treatments/diagnoses, and recent notes. For general dental questions, "
            "use clinical knowledge search when evidence improves the answer. Patient references are anonymous. "
            "Reply in the user's language and never give a final diagnosis."))
        history_messages = []
        for item in (history or [])[-MAX_CHAT_HISTORY_MESSAGES:]:
            content = item.content
            for patient in patients:
                if patient.name:
                    content = content.replace(patient.name, aliases[patient.id])
            history_messages.append(
                HumanMessage(content=content) if item.role == "user" else AIMessage(content=content)
            )
        result = agent.invoke({"messages": [*history_messages, HumanMessage(content=safe_question)]})
        messages = result["messages"]
        answer = str(messages[-1].content)
        trace = [call["name"] for m in messages for call in getattr(m, "tool_calls", [])]
        for p in patients: answer = answer.replace(aliases[p.id], p.name)
        return answer, trace, self._token_usage(messages)

    @staticmethod
    def _empty_token_usage() -> dict[str, int]:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def _token_usage(self, messages: list[Any]) -> dict[str, int]:
        """Aggregate every LLM call made during one agent/tool-call turn."""
        usage = self._empty_token_usage()
        for message in messages:
            metadata = getattr(message, "usage_metadata", None) or {}
            response_metadata = getattr(message, "response_metadata", None) or {}
            provider_usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
            source = metadata or provider_usage
            usage["input_tokens"] += self._usage_value(source, "input_tokens", "prompt_tokens", "inputTokenCount")
            usage["output_tokens"] += self._usage_value(source, "output_tokens", "completion_tokens", "candidatesTokenCount")
            usage["total_tokens"] += self._usage_value(source, "total_tokens", "totalTokenCount")
        if not usage["total_tokens"]:
            usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        return usage

    @staticmethod
    def _usage_value(source: dict[str, Any], *keys: str) -> int:
        for key in keys:
            value = source.get(key)
            if isinstance(value, (int, float)):
                return int(value)
        return 0

    def _tools(self, patients: list[Patient], aliases: dict[Any, str]):
        privacy = LlmPrivacyService(self.db)
        @tool
        def search_clinical_knowledge(query: str) -> list[dict]:
            """Search the global dental knowledge RAG corpus."""
            return [{"source_type": c.source_type, "snippet": c.chunk_text, "score": round(s, 4)} for c, s in VectorStore(self.db).search_knowledge(query, 6)]
        @tool
        def get_pms_statistics() -> dict:
            """Get exact PMS counts."""
            return {"patients": int(self.db.scalar(select(func.count()).select_from(Patient)) or 0), "notes": int(self.db.scalar(select(func.count()).select_from(ClinicalNote)) or 0), "clinical_facts": int(self.db.scalar(select(func.count()).select_from(ClinicalFact)) or 0)}
        tools = [search_clinical_knowledge, get_pms_statistics]
        if not patients: return tools
        ids = [p.id for p in patients]
        @tool
        def get_patient_profiles() -> list[dict]:
            """Get selected anonymized patient profiles."""
            return privacy.redact([{"patient": aliases[p.id], "date_of_birth": p.date_of_birth, "address": p.address, "dox_patient_id": p.dox_patient_id, "patient_number": p.patient_number, "medical_record_number": p.medical_record_number} for p in patients])
        @tool
        def get_treatments_and_diagnoses() -> list[dict]:
            """Get selected patients' diagnosis, treatment and periodontal facts."""
            facts = self.db.execute(select(ClinicalFact).where(ClinicalFact.patient_id.in_(ids)).order_by(ClinicalFact.effective_at.desc().nullslast()).limit(120)).scalars().all()
            return privacy.redact([{"patient": aliases[f.patient_id], "type": f.fact_type, "label": f.label, "summary": f.summary, "date": f.effective_at} for f in facts])
        @tool
        def get_recent_patient_notes() -> list[dict]:
            """Get up to 60 recent notes for selected patients."""
            notes = self.db.execute(select(ClinicalNote).where(ClinicalNote.patient_id.in_(ids)).order_by(ClinicalNote.created_at.desc()).limit(60)).scalars().all()
            return privacy.redact([{"patient": aliases[n.patient_id], "type": n.note_type, "content": n.content, "date": n.created_at} for n in notes])
        @tool
        def search_patient_notes(query: str) -> list[dict]:
            """RAG-search selected patients' notes."""
            out = []
            store = VectorStore(self.db)
            for p in patients: out += [{"patient": aliases[p.id], "score": round(s,4), "snippet": c.chunk_text} for c,s in store.search(p.id, query, 5)]
            return privacy.redact(sorted(out, key=lambda x: x["score"], reverse=True)[:12])
        return [*tools, get_patient_profiles, get_treatments_and_diagnoses, get_recent_patient_notes, search_patient_notes]

    def _resolve(self, question: str, ids: list[Any]) -> list[Patient]:
        selected = list(self.db.execute(select(Patient).where(Patient.id.in_(ids))).scalars()) if ids else []
        if selected: return selected
        q = question.casefold()
        return [p for p in self.db.execute(select(Patient)).scalars() if any(v and str(v).casefold() in q for v in (p.name, p.dox_patient_id, p.patient_number, p.medical_record_number))]
