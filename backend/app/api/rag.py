from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models import ClinicalCase
from app.db.session import get_db
from app.schemas.rag import RagQueryRequest, RagQueryResponse
from app.services.rag_service import RagService

router = APIRouter(prefix="/cases", tags=["rag"])


@router.post("/{case_id}/rag/query", response_model=RagQueryResponse)
def query_case_rag(case_id: UUID, req: RagQueryRequest, db: Session = Depends(get_db)):
    case = db.get(ClinicalCase, case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return RagService(db).query(case.patient_id, req.question, req.top_k)

