from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models import ClinicalCase, Patient
from app.db.session import get_db
from app.schemas.case import CaseCreate, CaseRead
from app.services.records_service import patient_lock

router = APIRouter(prefix="", tags=["cases"])


@router.post("/patients/{patient_id}/cases", response_model=CaseRead)
def create_case(patient_id: UUID, req: CaseCreate, db: Session = Depends(get_db)):
    patient_lock(db, patient_id)
    case = ClinicalCase(patient_id=patient_id, title=req.title, status="draft")
    db.add(case)
    db.commit()
    db.refresh(case)
    return case


@router.get("/cases/{case_id}", response_model=CaseRead)
def get_case(case_id: UUID, db: Session = Depends(get_db)):
    case = db.get(ClinicalCase, case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return case
