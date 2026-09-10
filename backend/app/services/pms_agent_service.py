"""Tool-calling PMS chart-summary agent."""
from __future__ import annotations

from typing import Any
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ClinicalFact, ClinicalNote, Patient
from app.services.llm_service import LlmService
from app.services.llm_privacy_service import LlmPrivacyService
from app.services.vector_store import VectorStore


class PmsAgentService:
    def __init__(self, db: Session) -> None:
        self.db, self.llm = db, LlmService()

    def run(self, question: str, selected_ids: list[Any]) -> tuple[str, list[str]]:
        patients = self._resolve_patients(question, selected_ids)
        if not patients:
            return "请先从患者列表选择患者，或在问题中提供完整姓名、DOX ID、患者编号或病历号。", []
        aliases = {patient.id: f"PATIENT_{index + 1}" for index, patient in enumerate(patients)}
        safe_question = question
        for patient in patients:
            for value in (patient.name, patient.dox_patient_id, patient.patient_number, patient.medical_record_number):
                if value:
                    safe_question = safe_question.replace(str(value), aliases[patient.id])
        tools = self._tools(patients, aliases)
        if not self.llm.enabled:
            return self._fallback(patients, aliases), [item.name for item in tools]
        agent = create_agent(
            model=self.llm._chat_model(), tools=tools,
            system_prompt=("You are a dental PMS chart assistant. Use the available tools before answering any "
                           "patient-specific question. The patient references are anonymous tokens. Summarize "
                           "only tool results; distinguish notes, treatment/diagnosis facts, and uncertainty. "
                           "Reply in the user's language and never make a final diagnosis.")
        )
        result = agent.invoke({"messages": [HumanMessage(content=safe_question)]})
        messages = result["messages"]
        answer = str(messages[-1].content)
        trace = [call["name"] for message in messages for call in getattr(message, "tool_calls", [])]
        for patient in patients:
            answer = answer.replace(aliases[patient.id], patient.name)
        return answer, trace

    def matches_question(self, question: str) -> bool:
        return bool(self._resolve_patients(question, []))

    def _tools(self, patients: list[Patient], aliases: dict[Any, str]):
        ids = [patient.id for patient in patients]
        privacy = LlmPrivacyService(self.db)
        @tool
        def get_patient_profiles() -> list[dict]:
            """Get selected patients' anonymized demographics and identifiers."""
            return privacy.redact([{"patient": aliases[p.id], "date_of_birth": p.date_of_birth, "address": p.address,
                                    "dox_patient_id": p.dox_patient_id, "patient_number": p.patient_number,
                                    "medical_record_number": p.medical_record_number} for p in patients])
        @tool
        def get_treatments_and_diagnoses() -> list[dict]:
            """Get diagnosis, treatment, and periodontal facts for selected patients."""
            facts = self.db.execute(select(ClinicalFact).where(ClinicalFact.patient_id.in_(ids)).order_by(ClinicalFact.effective_at.desc().nullslast()).limit(120)).scalars().all()
            return privacy.redact([{"patient": aliases[f.patient_id], "type": f.fact_type, "label": f.label, "summary": f.summary, "date": f.effective_at} for f in facts])
        @tool
        def get_recent_patient_notes() -> list[dict]:
            """Get recent clinical note text for selected patients."""
            notes = self.db.execute(select(ClinicalNote).where(ClinicalNote.patient_id.in_(ids)).order_by(ClinicalNote.created_at.desc()).limit(60)).scalars().all()
            return privacy.redact([{"patient": aliases[n.patient_id], "type": n.note_type, "content": n.content, "date": n.created_at} for n in notes])
        @tool
        def search_patient_notes(query: str) -> list[dict]:
            """Semantically search selected patients' indexed clinical notes for a focused question."""
            results = []
            store = VectorStore(self.db)
            for patient in patients:
                for chunk, score in store.search(patient.id, query, top_k=5):
                    results.append({"patient": aliases[patient.id], "score": round(score, 4), "source_type": chunk.source_type, "snippet": chunk.chunk_text})
            return privacy.redact(sorted(results, key=lambda item: item["score"], reverse=True)[:12])
        return [get_patient_profiles, get_treatments_and_diagnoses, get_recent_patient_notes, search_patient_notes]

    def _resolve_patients(self, question: str, selected_ids: list[Any]) -> list[Patient]:
        selected = list(self.db.execute(select(Patient).where(Patient.id.in_(selected_ids))).scalars()) if selected_ids else []
        if selected:
            return selected
        q = question.casefold()
        return [p for p in self.db.execute(select(Patient)).scalars() if any(v and str(v).casefold() in q for v in (p.name, p.dox_patient_id, p.patient_number, p.medical_record_number))]

    def _fallback(self, patients: list[Patient], aliases: dict[Any, str]) -> str:
        return "已选择患者：" + "、".join(f"{aliases[p.id]}（{p.name}）" for p in patients) + "。请配置支持工具调用的模型以生成完整病历总结。"
