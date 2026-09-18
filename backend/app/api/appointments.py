"""预约列表和表单写入；带版本的更新/删除防止覆盖其他成员刚刚保存的数据。"""
from uuid import UUID
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.db.models import Appointment
from app.db.session import get_db
from app.schemas.operations import AppointmentCreate, AppointmentRead, CreateAppointment, UpdateAppointment, DeleteAppointment
from app.services.records_service import apply_change, version

router = APIRouter(prefix="/appointments", tags=["appointments"])


@router.get("", response_model=list[AppointmentRead])
def appointments(patient_id: UUID | None = None, db: Session = Depends(get_db)):
    stmt = select(Appointment).order_by(Appointment.starts_at)
    if patient_id:
        stmt = stmt.where(Appointment.patient_id == patient_id)
    return [AppointmentRead.model_validate({**{c.key: getattr(row, c.key) for c in row.__table__.columns}, "version": version(row)})
            for row in db.scalars(stmt).all()]


@router.post("", status_code=201)
def create(req: AppointmentCreate, db: Session = Depends(get_db)):
    result = apply_change(db, CreateAppointment(operation="create_appointment", patient_id=req.patient_id, data=req.model_dump(exclude={"patient_id"})))
    db.commit()
    return result


@router.put("/{appointment_id}")
def update(appointment_id: UUID, req: UpdateAppointment, db: Session = Depends(get_db)):
    from fastapi import HTTPException
    if appointment_id != req.appointment_id:
        raise HTTPException(422, "Appointment ID mismatch")
    result = apply_change(db, req)
    db.commit()
    return result


@router.delete("/{appointment_id}")
def remove(appointment_id: UUID, patient_id: UUID, expected_version: str, db: Session = Depends(get_db)):
    result = apply_change(db, DeleteAppointment(operation="delete_appointment", patient_id=patient_id,
                                               appointment_id=appointment_id, expected_version=expected_version))
    db.commit()
    return result
