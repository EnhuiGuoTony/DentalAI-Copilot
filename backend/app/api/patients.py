from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, delete
from sqlalchemy.orm import Session

from app.db.models import ClinicalCase, ClinicalNote, Patient, EmbeddingChunk
from app.schemas.operations import UpdatePatient, DeletePatient, AddNote, DeleteNote
from app.services.records_service import apply_change, patient_lock, patient_version, version
from app.db.session import get_db
from app.schemas.patient import ClinicalNoteCreate, ClinicalNoteRead, PatientCreate, PatientRead, TimelineItem
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
    # 与 Agent 共用事务服务：笔记和向量同时成功，避免新笔记在检索中不可见。
    result = apply_change(db, AddNote(operation="add_note", patient_id=patient_id, **req.model_dump()))
    db.commit()
    return db.get(ClinicalNote, result.record_id)


@router.post("/{patient_id}/notes/ingest")
def ingest_notes(patient_id: UUID, db: Session = Depends(get_db)):
    """患者笔记的索引构建入口：读取原文 -> 重叠分块 -> 生成向量 -> 批量提交。

    在患者锁内重建笔记索引，先移除旧块，重复调用不会累积重复向量。
    它只是构建检索数据，不是训练或微调大模型。
    """
    patient_lock(db, patient_id)
    notes = db.execute(select(ClinicalNote).where(ClinicalNote.patient_id == patient_id)).scalars().all()
    db.execute(delete(EmbeddingChunk).where(EmbeddingChunk.patient_id == patient_id, EmbeddingChunk.source_id.in_([n.id for n in notes])))
    store = VectorStore(db)
    created = 0
    for note in notes:
        created += len(store.add_document(patient_id, "clinical_note", note.id, note.content,
                                          {"note_type": note.note_type}))
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


@router.get("/{patient_id}/version")
def get_version(patient_id: UUID, db: Session = Depends(get_db)):
    patient = patient_lock(db, patient_id)
    return {"version": patient_version(db, patient)}


@router.put("/{patient_id}")
def update_patient(patient_id: UUID, req: UpdatePatient, db: Session = Depends(get_db)):
    if req.patient_id != patient_id:
        raise HTTPException(422, "Patient ID mismatch")
    result = apply_change(db, req)
    db.commit()
    return result


@router.delete("/{patient_id}")
def delete_patient(patient_id: UUID, expected_version: str, db: Session = Depends(get_db)):
    result = apply_change(db, DeletePatient(operation="delete_patient", patient_id=patient_id, expected_version=expected_version))
    db.commit()
    return result


@router.delete("/{patient_id}/notes/{note_id}")
def delete_note(patient_id: UUID, note_id: UUID, expected_version: str, db: Session = Depends(get_db)):
    result = apply_change(db, DeleteNote(operation="delete_note", patient_id=patient_id, note_id=note_id, expected_version=expected_version))
    db.commit()
    return result
