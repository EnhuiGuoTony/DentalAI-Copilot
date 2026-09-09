from typing import Any, TypedDict
from uuid import UUID

from langgraph.graph import END, StateGraph
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ClinicalCase, ClinicalFact, XrayFinding
from app.schemas.agent import AgentOutput, SuspectedFinding, ToolTraceItem
from app.schemas.rag import EvidenceItem
from app.services.llm_service import ClinicalDraftInput, LlmService
from app.services.rag_service import RagService


class ClinicalAgentState(TypedDict, total=False):
    case_id: UUID
    patient_id: UUID
    question: str
    findings: list[XrayFinding]
    evidence: list[EvidenceItem]
    structured_facts: list[ClinicalFact]
    output: AgentOutput
    tool_trace: list[ToolTraceItem]


class ClinicalAgentGraph:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.rag = RagService(db)
        self.llm = LlmService()
        self.graph = self._build_graph()

    def run(self, case: ClinicalCase, question: str) -> ClinicalAgentState:
        initial: ClinicalAgentState = {
            "case_id": case.id,
            "patient_id": case.patient_id,
            "question": question,
            "tool_trace": [],
        }
        return self.graph.invoke(initial)

    def _build_graph(self):
        graph = StateGraph(ClinicalAgentState)
        graph.add_node("load_xray_findings", self._load_xray_findings)
        graph.add_node("retrieve_patient_context", self._retrieve_patient_context)
        graph.add_node("retrieve_structured_context", self._retrieve_structured_context)
        graph.add_node("generate_clinical_draft", self._generate_clinical_draft)
        graph.set_entry_point("load_xray_findings")
        graph.add_edge("load_xray_findings", "retrieve_patient_context")
        graph.add_edge("retrieve_patient_context", "retrieve_structured_context")
        graph.add_edge("retrieve_structured_context", "generate_clinical_draft")
        graph.add_edge("generate_clinical_draft", END)
        return graph.compile()

    def _load_xray_findings(self, state: ClinicalAgentState) -> dict[str, Any]:
        findings = (
            self.db.execute(select(XrayFinding).where(XrayFinding.case_id == state["case_id"]))
            .scalars()
            .all()
        )
        return {
            "findings": findings,
            "tool_trace": [
                *state.get("tool_trace", []),
                ToolTraceItem(
                    tool="langgraph.load_xray_findings",
                    input={"case_id": str(state["case_id"])},
                    output_summary=f"Loaded {len(findings)} structured X-ray finding(s).",
                ),
            ],
        }

    def _retrieve_patient_context(self, state: ClinicalAgentState) -> dict[str, Any]:
        rag = self.rag.query(state["patient_id"], state["question"], top_k=5)
        return {
            "evidence": rag.evidence,
            "tool_trace": [
                *state.get("tool_trace", []),
                ToolTraceItem(
                    tool="langgraph.search_patient_history",
                    input={"patient_id": str(state["patient_id"]), "question": state["question"]},
                    output_summary=f"Retrieved {len(rag.evidence)} patient-scoped evidence chunk(s).",
                ),
            ],
        }

    def _retrieve_structured_context(self, state: ClinicalAgentState) -> dict[str, Any]:
        facts = (
            self.db.execute(
                select(ClinicalFact)
                .where(ClinicalFact.patient_id == state["patient_id"])
                .order_by(ClinicalFact.effective_at.desc().nullslast(), ClinicalFact.created_at.desc())
                .limit(12)
            )
            .scalars()
            .all()
        )
        fact_types = sorted({fact.fact_type for fact in facts})
        return {
            "structured_facts": facts,
            "tool_trace": [
                *state.get("tool_trace", []),
                ToolTraceItem(
                    tool="langgraph.get_structured_clinical_facts",
                    input={"patient_id": str(state["patient_id"]), "limit": 12},
                    output_summary=f"Loaded {len(facts)} structured fact(s): {', '.join(fact_types) or 'none'}.",
                ),
            ],
        }

    def _generate_clinical_draft(self, state: ClinicalAgentState) -> dict[str, Any]:
        if self.llm.enabled:
            output = self.llm.generate_clinical_draft(
                ClinicalDraftInput(
                    question=state["question"],
                    xray_findings=[self._finding_to_payload(item) for item in state.get("findings", [])],
                    evidence=[item.model_dump() for item in state.get("evidence", [])],
                    structured_facts=[self._fact_to_payload(item) for item in state.get("structured_facts", [])],
                )
            )
            mode = "LangChain ChatOpenAI structured output"
        else:
            output = self._mock_structured_output(
                state.get("findings", []),
                state.get("evidence", []),
                state.get("structured_facts", []),
            )
            mode = "mock LLM fallback"

        return {
            "output": output,
            "tool_trace": [
                *state.get("tool_trace", []),
                ToolTraceItem(
                    tool="langgraph.generate_clinical_draft",
                    input={
                        "llm_provider": self.llm.settings.llm_provider,
                        "llm_model": self.llm.provider_label(),
                        "mock_llm": self.llm.settings.mock_llm,
                    },
                    output_summary=f"Generated clinician-review draft using {mode}.",
                ),
            ],
        }

    @staticmethod
    def _finding_to_payload(finding: XrayFinding) -> dict[str, Any]:
        return {
            "id": str(finding.id),
            "category": finding.category,
            "confidence": finding.confidence,
            "tooth_number": finding.tooth_number,
            "bbox": finding.bbox,
            "polygon": finding.polygon,
        }

    @staticmethod
    def _fact_to_payload(fact: ClinicalFact) -> dict[str, Any]:
        return {
            "id": str(fact.id),
            "fact_type": fact.fact_type,
            "source_table": fact.source_table,
            "source_pk": fact.source_pk,
            "label": fact.label,
            "summary": fact.summary,
            "effective_at": fact.effective_at.isoformat() if fact.effective_at else None,
            "data": fact.data,
        }

    @staticmethod
    def _mock_structured_output(findings: list[XrayFinding], evidence: list[EvidenceItem], facts: list[ClinicalFact] | None = None) -> AgentOutput:
        suspected = [
            SuspectedFinding(
                category=f.category,
                confidence=f.confidence,
                tooth_number=f.tooth_number,
                evidence_refs=[f"finding:{f.id}"],
            )
            for f in findings
        ]
        fact_summaries = [fact.summary for fact in (facts or [])[:4]]
        history = [item.snippet for item in evidence[:3]] + fact_summaries
        categories = ", ".join(sorted({f.category for f in findings})) or "no suspicious findings"
        return AgentOutput(
            clinical_summary=(
                f"The LangGraph clinical workflow reviewed {len(findings)} X-ray finding(s): {categories}. "
                "The draft combines image findings with retrieved patient history for clinician review."
            ),
            suspected_findings=suspected,
            relevant_history=history,
            recommended_next_steps=[
                "Review highlighted radiographic regions in the image viewer.",
                "Validate tooth mapping, severity, and symptoms during the clinical exam.",
                "Use the evidence list to update the final diagnosis and treatment plan.",
            ],
            patient_friendly_explanation=(
                "The assistant highlighted areas that may need attention and compared them with the chart history. "
                "A dentist must review the result before any diagnosis or treatment decision."
            ),
            limitations=[
                "Mock LLM mode is enabled until an API key is configured.",
                "Vision findings are preliminary and require licensed clinician interpretation.",
            ],
        )
