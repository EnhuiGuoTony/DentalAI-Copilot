"""患者、预约、笔记的事务边界；普通 API 与审批后的 Agent 使用同一业务规则。

按患者行加锁使重叠预约检查与写入串行执行；删除原文时同步清理向量，避免 RAG 残留。
调用者负责 commit，以便 Agent 将业务修改和幂等回执原子提交。
"""
import hashlib
import json
from uuid import UUID
from fastapi import HTTPException
from sqlalchemy import delete, select
from sqlalchemy.orm import Session
from app.db.models import Patient, Appointment, ClinicalNote, EmbeddingChunk, ClinicalFact, AgentRun, ClinicalCase
from app.schemas.operations import Change, MutationResult
from app.services.chunking import chunk_text
from app.services.vector_store import VectorStore


def version(record) -> str:
    """摘要不依赖 ORM 对象地址，只覆盖数据库列值；它是并发检测值而非权限凭证。"""
    values = {c.name: getattr(record, c.key) for c in record.__table__.columns}
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()


def patient_version(db: Session, patient: Patient) -> str:
    """删除患者会影响整张病历，关联记录变更也必须使原审批失效。"""
    parts = [version(patient)]
    for model in (ClinicalNote, Appointment, ClinicalCase, ClinicalFact, EmbeddingChunk, AgentRun):
        rows = db.scalars(select(model).where(model.patient_id == patient.id)).all()
        # EmbeddingChunk 的 metadata 列名和 Python 属性不同，摘要只需覆盖其唯一 ID。
        parts.extend(sorted(str(row.id) for row in rows))
        if model in (ClinicalNote, Appointment, ClinicalCase, ClinicalFact):
            parts.extend(sorted(version(row) for row in rows))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def patient_lock(db: Session, patient_id: UUID) -> Patient:
    patient = db.scalar(select(Patient).where(Patient.id == patient_id).with_for_update())
    if patient is None:
        raise HTTPException(404, "Patient not found")
    return patient


def ensure_version(actual: str, expected: str) -> None:
    if actual != expected:
        raise HTTPException(409, "Record changed since review. Read again and request a new approval.")


def apply_change(db: Session, change: Change, allowed_ids: list[UUID] | None = None) -> MutationResult:
    """检查固定患者范围与版本，再修改数据库；模型不能用工具参数扩大授权范围。"""
    op = change.operation
    if op in {"create_patient", "update_patient"} and isinstance(change.data.date_of_birth, str):
        raise HTTPException(422, "Unresolved identity placeholder in date_of_birth")
    if op == "create_patient":
        record = Patient(**change.data.model_dump())
        db.add(record)
    else:
        if allowed_ids is not None and change.patient_id not in allowed_ids:
            raise HTTPException(403, "Patient is outside this conversation's selected scope")
        patient = patient_lock(db, change.patient_id)
        if op in {"update_patient", "delete_patient"}:
            ensure_version(patient_version(db, patient), change.expected_version)
            record = patient
            if op == "update_patient":
                for key, value in change.data.model_dump().items():
                    setattr(record, key, value)
            else:
                # 显式依外键顺序删除，不依赖 create_all 修改已有数据库的级联约束。
                for model in (AgentRun, EmbeddingChunk, ClinicalFact, Appointment, ClinicalNote, ClinicalCase):
                    db.execute(delete(model).where(model.patient_id == patient.id))
                db.execute(delete(Patient).where(Patient.id == patient.id))
        elif op == "add_note":
            record = ClinicalNote(patient_id=patient.id, note_type=change.note_type, content=change.content)
            db.add(record)
            db.flush()
            for index, text in enumerate(chunk_text(change.content)):
                VectorStore(db).add_chunk(patient.id, "clinical_note", record.id, text,
                                          {"note_type": change.note_type, "chunk_index": index})
        elif op == "delete_note":
            record = db.get(ClinicalNote, change.note_id)
            if record is None or record.patient_id != patient.id:
                raise HTTPException(404, "Note not found for this patient")
            ensure_version(version(record), change.expected_version)
            db.execute(delete(EmbeddingChunk).where(EmbeddingChunk.patient_id == patient.id, EmbeddingChunk.source_id == record.id))
            db.delete(record)
        else:
            if op == "create_appointment":
                record = Appointment(patient_id=patient.id)
                db.add(record)
            else:
                record = db.get(Appointment, change.appointment_id)
                if record is None or record.patient_id != patient.id:
                    raise HTTPException(404, "Appointment not found for this patient")
                ensure_version(version(record), change.expected_version)
            if op == "delete_appointment":
                db.delete(record)
            else:
                data = change.data
                if data.status in {"scheduled", "checked_in"}:
                    conflict = db.scalar(select(Appointment).where(
                        Appointment.patient_id == patient.id,
                        Appointment.status.in_(["scheduled", "checked_in"]),
                        Appointment.starts_at < data.ends_at, Appointment.ends_at > data.starts_at,
                        Appointment.id != record.id if record.id else True))
                    if conflict:
                        raise HTTPException(409, "Patient already has an appointment in this time range")
                for key, value in data.model_dump().items():
                    setattr(record, key, value)
    db.flush()
    return MutationResult(operation=op, record_id=record.id)
