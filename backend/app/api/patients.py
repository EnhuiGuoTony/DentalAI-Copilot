from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ClinicalCase, ClinicalNote, Patient
from app.db.session import get_db
from app.schemas.patient import ClinicalNoteCreate, ClinicalNoteRead, PatientCreate, PatientRead, TimelineItem
from app.services.chunking import chunk_text
from app.services.vector_store import VectorStore

router = APIRouter(prefix="/patients", tags=["patients"])


@router.get("", response_model=list[PatientRead])
def list_patients(db: Session = Depends(get_db)):
    return db.execute(select(Patient).order_by(Patient.created_at.desc())).scalars().all()


@router.post("", response_model=PatientRead)
def create_patient(req: PatientCreate, db: Session = Depends(get_db)):
    patient = Patient(name=req.name, date_of_birth=req.date_of_birth)
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


@router.get("/{patient_id}", response_model=PatientRead)
def get_patient(patient_id: UUID, db: Session = Depends(get_db)):
    patient = db.get(Patient, patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@router.post("/{patient_id}/notes", response_model=ClinicalNoteRead)
def create_note(patient_id: UUID, req: ClinicalNoteCreate, db: Session = Depends(get_db)):
    if db.get(Patient, patient_id) is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    note = ClinicalNote(patient_id=patient_id, note_type=req.note_type, content=req.content)
    db.add(note)
    db.commit()
    db.refresh(note)
    return note


@router.post("/{patient_id}/notes/ingest")
def ingest_notes(patient_id: UUID, db: Session = Depends(get_db)):
    notes = db.execute(select(ClinicalNote).where(ClinicalNote.patient_id == patient_id)).scalars().all()
    store = VectorStore(db)
    created = 0
    for note in notes:
        for idx, chunk in enumerate(chunk_text(note.content)):
            store.add_chunk(
                patient_id=patient_id,
                source_type="clinical_note",
                source_id=note.id,
                text=chunk,
                metadata={"note_type": note.note_type, "chunk_index": idx},
            )
            created += 1
    db.commit()
    return {"patient_id": patient_id, "chunks_created": created}


@router.get("/{patient_id}/timeline", response_model=list[TimelineItem])
def get_timeline(patient_id: UUID, db: Session = Depends(get_db)):
    notes = db.execute(select(ClinicalNote).where(ClinicalNote.patient_id == patient_id)).scalars().all()
    cases = db.execute(select(ClinicalCase).where(ClinicalCase.patient_id == patient_id)).scalars().all()
    items: list[TimelineItem] = []
    items.extend(
        TimelineItem(type="note", id=n.id, title=n.note_type, content=n.content, created_at=n.created_at)
        for n in notes
    )
    items.extend(
        TimelineItem(type="case", id=c.id, title=c.title, content=c.status, created_at=c.created_at)
        for c in cases
    )
    return sorted(items, key=lambda item: item.created_at, reverse=True)

