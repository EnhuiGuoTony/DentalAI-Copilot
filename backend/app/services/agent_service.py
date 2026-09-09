from uuid import UUID

from sqlalchemy.orm import Session

from app.db.models import AgentRun, ClinicalCase
from app.schemas.agent import AgentRunResponse
from app.services.clinical_agent_graph import ClinicalAgentGraph


class AgentService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.agent_graph = ClinicalAgentGraph(db)

    def run_case_agent(self, case_id: UUID, question: str) -> AgentRunResponse:
        case = self.db.get(ClinicalCase, case_id)
        if case is None:
            raise ValueError("Case not found")

        state = self.agent_graph.run(case, question)
        output = state["output"]
        evidence = state.get("evidence", [])
        tool_trace = state.get("tool_trace", [])

        run = AgentRun(
            patient_id=case.patient_id,
            case_id=case_id,
            user_question=question,
            tool_trace=[item.model_dump() for item in tool_trace],
            final_output=output.model_dump(),
        )
        self.db.add(run)
        self.db.commit()
        self.db.refresh(run)

        return AgentRunResponse(
            id=run.id,
            patient_id=case.patient_id,
            case_id=case_id,
            question=question,
            output=output,
            evidence=evidence,
            tool_trace=tool_trace,
            created_at=run.created_at,
        )
