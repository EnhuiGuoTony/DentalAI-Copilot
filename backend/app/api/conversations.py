"""会话创建、刷新与 SSE 执行入口；所有读取和恢复都检查当前用户归属。"""
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.db.models import Conversation, Patient, User
from app.db.session import get_db
from app.schemas.conversation import ConversationCreate, TurnRequest, ResumeRequest
from app.services.auth_service import current_user
from app.services.conversation_service import owned_conversation, conversation_state, stream_turn

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", status_code=201)
def create(req: ConversationCreate, user: User = Depends(current_user), db: Session = Depends(get_db)):
    ids = list(dict.fromkeys(req.patient_ids))
    patients = db.scalars(select(Patient).where(Patient.id.in_(ids))).all()
    if len(patients) != len(ids):
        raise HTTPException(404, "Selected patient not found")
    mapping = {}
    for patient in patients:
        for field in ("name", "date_of_birth", "address", "patient_number", "medical_record_number", "dox_patient_id"):
            value = getattr(patient, field)
            if value:
                mapping[f"[P_{patient.id.hex}_{field.upper()}]"] = str(value)
    row = Conversation(user_id=user.id, patient_ids=[str(v) for v in ids], privacy_map=mapping)
    db.add(row)
    db.commit()
    return {"id": row.id, "patient_ids": row.patient_ids, "created_at": row.created_at}


@router.get("")
def conversations(user: User = Depends(current_user), db: Session = Depends(get_db)):
    return [{"id": r.id, "patient_ids": r.patient_ids, "created_at": r.created_at} for r in db.scalars(
        select(Conversation).where(Conversation.user_id == user.id).order_by(Conversation.created_at.desc()).limit(100))]


@router.get("/{conversation_id}")
def state(conversation_id: UUID, user: User = Depends(current_user), db: Session = Depends(get_db)):
    owned_conversation(db, conversation_id, user.id)
    return conversation_state(conversation_id, user.id)


def response(conversation_id, user, db, **kwargs):
    owned_conversation(db, conversation_id, user.id)
    # 生成器持有自己的连接；不依赖 FastAPI 在响应开始前可能关闭的请求 Session。
    return StreamingResponse(stream_turn(conversation_id, user.id, **kwargs), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/{conversation_id}/messages")
def send(conversation_id: UUID, req: TurnRequest, user: User = Depends(current_user), db: Session = Depends(get_db)):
    return response(conversation_id, user, db, message=req.message)


@router.post("/{conversation_id}/resume")
def resume(conversation_id: UUID, req: ResumeRequest, user: User = Depends(current_user), db: Session = Depends(get_db)):
    return response(conversation_id, user, db, resume=req)


@router.post("/{conversation_id}/continue")
def continue_run(conversation_id: UUID, user: User = Depends(current_user), db: Session = Depends(get_db)):
    return response(conversation_id, user, db)
