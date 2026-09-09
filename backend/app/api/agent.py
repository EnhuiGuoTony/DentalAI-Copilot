from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.agent import AgentRunRequest, AgentRunResponse
from app.services.agent_service import AgentService

router = APIRouter(prefix="/cases", tags=["agent"])


@router.post("/{case_id}/agent/run", response_model=AgentRunResponse)
def run_agent(case_id: UUID, req: AgentRunRequest, db: Session = Depends(get_db)):
    try:
        return AgentService(db).run_case_agent(case_id, req.question)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

