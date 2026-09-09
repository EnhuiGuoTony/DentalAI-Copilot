"""Read-only chart context for the general DentalAI assistant.

RAG is useful for narrative chart notes, but exact operational questions such as
"how many patients?" must be answered from relational data rather than similarity
search.  This service combines both sources into a bounded, auditable context.
"""

from collections import Counter
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import ClinicalCase, ClinicalFact, ClinicalNote, Patient, XrayFinding, XrayImage
from app.services.vector_store import VectorStore


class PatientContextService:
    max_patients = 50
    max_notes = 12
    max_facts = 12
    max_evidence = 8

    def __init__(self, db: Session) -> None:
        self.db = db
        self.vector_store = VectorStore(db)

    def build(self, question: str) -> dict[str, Any]:
        """Build only data that the assistant is allowed to cite; never writes."""
        stats = self._stats()
        patients = self.db.execute(
            select(Patient).order_by(Patient.created_at.desc()).limit(self.max_patients)
        ).scalars().all()
        patient_ids = [patient.id for patient in patients]
        case_counts = self._counts_by_patient(ClinicalCase.patient_id, patient_ids)
        note_counts = self._counts_by_patient(ClinicalNote.patient_id, patient_ids)
        image_counts = self._counts_by_patient(XrayImage.patient_id, patient_ids)

        patient_summaries = [
            {
                "id": str(patient.id),
                "name": patient.name,
                "date_of_birth": patient.date_of_birth.isoformat() if patient.date_of_birth else None,
                "address": patient.address,
                "dox_patient_id": patient.dox_patient_id,
                "patient_number": patient.patient_number,
                "medical_record_number": patient.medical_record_number,
                "case_count": case_counts.get(patient.id, 0),
                "note_count": note_counts.get(patient.id, 0),
                "image_count": image_counts.get(patient.id, 0),
            }
            for patient in patients
        ]
        evidence = self.vector_store.search_all_patients(question, self.max_evidence)
        evidence_patient_ids = {chunk.patient_id for chunk, _ in evidence}
        named_patient_ids = {
            patient.id for patient in patients if patient.name and patient.name.lower() in question.lower()
        }
        focus_ids = evidence_patient_ids | named_patient_ids
        # For a small demo corpus this provides genuinely useful answers even when
        # the deterministic demo embedding cannot understand the user's language.
        if len(patients) <= 10:
            focus_ids.update(patient_ids)

        notes = []
        facts = []
        if focus_ids:
            notes = self.db.execute(
                select(ClinicalNote)
                .where(ClinicalNote.patient_id.in_(focus_ids))
                .order_by(ClinicalNote.created_at.desc())
                .limit(self.max_notes)
            ).scalars().all()
            facts = self.db.execute(
                select(ClinicalFact)
                .where(ClinicalFact.patient_id.in_(focus_ids))
                .order_by(ClinicalFact.effective_at.desc().nullslast(), ClinicalFact.created_at.desc())
                .limit(self.max_facts)
            ).scalars().all()

        names = {patient.id: patient.name for patient in patients}
        return {
            "statistics": stats,
            "patients": patient_summaries,
            "notes": [
                {"patient": names.get(note.patient_id, str(note.patient_id)), "type": note.note_type, "content": note.content,
                 "created_at": note.created_at.isoformat() if note.created_at else None}
                for note in notes
            ],
            "clinical_facts": [
                {"patient": names.get(fact.patient_id, str(fact.patient_id)), "type": fact.fact_type,
                 "label": fact.label, "summary": fact.summary,
                 "effective_at": fact.effective_at.isoformat() if fact.effective_at else None}
                for fact in facts
            ],
            "retrieved_evidence": [
                {"patient": names.get(chunk.patient_id, str(chunk.patient_id)), "source_type": chunk.source_type,
                 "snippet": chunk.chunk_text[:500], "score": round(score, 4)}
                for chunk, score in evidence
            ],
            "limits": {
                "patient_list_limit": self.max_patients,
                "note_limit": self.max_notes,
                "fact_limit": self.max_facts,
                "evidence_limit": self.max_evidence,
            },
        }

    def mock_answer(self, context: dict[str, Any]) -> str:
        """Useful, non-hallucinatory fallback when no model provider is configured."""
        stats = context["statistics"]
        lines = [
            "当前数据库中的病人概览：",
            f"- 病人：{stats['patients']} 位",
            f"- 病例：{stats['cases']} 个",
            f"- 病历笔记：{stats['notes']} 条",
            f"- X 光影像：{stats['images']} 张",
            f"- X 光发现：{stats['findings']} 条",
            f"- 结构化临床事实：{stats['clinical_facts']} 条",
        ]
        patients = context["patients"]
        if patients:
            lines.append("病人列表：")
            lines.extend(
                f"- {item['name']}（病例 {item['case_count']}，笔记 {item['note_count']}，影像 {item['image_count']}）"
                for item in patients
            )
        else:
            lines.append("目前尚未导入或创建任何病人，因此没有可检索的病历详情。")
        if context.get("notes"):
            lines.append("相关病历笔记：")
            lines.extend(
                f"- {item['patient']} / {item['type']}：{item['content']}" for item in context["notes"]
            )
        if context.get("clinical_facts"):
            lines.append("相关结构化临床事实：")
            lines.extend(
                f"- {item['patient']} / {item['type']} / {item['label']}：{item['summary']}"
                for item in context["clinical_facts"]
            )
        lines.append("提示：当前未配置真实 LLM，以上是从数据库实时读取的确定性结果。")
        return "\n".join(lines)

    @staticmethod
    def requires_deterministic_answer(question: str) -> bool:
        """Identify questions for which an LLM must not be the source of truth."""
        normalized = question.lower()
        chart_terms = ("病人", "患者", "病历", "病例", "笔记", "x光", "影像", "patient", "patients", "case", "cases", "note", "notes")
        operation_terms = ("多少", "几个", "几位", "总数", "数量", "列表", "列出", "全部", "count", "how many", "total", "list", "all")
        return any(term in normalized for term in chart_terms) and any(term in normalized for term in operation_terms)

    @staticmethod
    def is_unhelpful_model_reply(reply: str) -> bool:
        normalized = reply.lower()
        markers = ("didn't understand", "did not understand", "could you please ask", "无法理解", "不明白您的问题")
        return any(marker in normalized for marker in markers)

    def _stats(self) -> dict[str, int]:
        return {
            "patients": int(self.db.scalar(select(func.count()).select_from(Patient)) or 0),
            "cases": int(self.db.scalar(select(func.count()).select_from(ClinicalCase)) or 0),
            "notes": int(self.db.scalar(select(func.count()).select_from(ClinicalNote)) or 0),
            "images": int(self.db.scalar(select(func.count()).select_from(XrayImage)) or 0),
            "findings": int(self.db.scalar(select(func.count()).select_from(XrayFinding)) or 0),
            "clinical_facts": int(self.db.scalar(select(func.count()).select_from(ClinicalFact)) or 0),
        }

    def _counts_by_patient(self, patient_column: Any, patient_ids: list[Any]) -> Counter:
        if not patient_ids:
            return Counter()
        rows = self.db.execute(
            select(patient_column, func.count()).where(patient_column.in_(patient_ids)).group_by(patient_column)
        ).all()
        return Counter({patient_id: count for patient_id, count in rows})
