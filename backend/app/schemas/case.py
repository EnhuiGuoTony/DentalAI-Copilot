from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class CaseCreate(BaseModel):
    title: str = "Patient chart review"


class CaseRead(BaseModel):
    id: UUID
    patient_id: UUID
    title: str
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}
