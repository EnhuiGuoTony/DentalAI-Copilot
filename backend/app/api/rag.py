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
    """先由病例查出患者，再按患者范围检索，避免客户端另行指定不一致的患者 ID。"""
    case = db.get(ClinicalCase, case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")
    # 这是“病例所属患者”的病历检索，并未进一步只限定该病例的笔记。
    return RagService(db).query(case.patient_id, req.question, req.top_k)
