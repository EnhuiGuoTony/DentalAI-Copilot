from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, Field


class PatientCreate(BaseModel):
    model_config = {"str_strip_whitespace": True}
    name: str = Field(min_length=1, max_length=200)
    date_of_birth: date | None = None


class PatientRead(BaseModel):
    id: UUID
    name: str
    date_of_birth: date | None
    dox_patient_id: str | None = None
    patient_number: str | None = None
    medical_record_number: str | None = None
    address: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ClinicalNoteCreate(BaseModel):
    note_type: str = Field(default="clinical", min_length=1, max_length=80)
    content: str = Field(min_length=1, max_length=20000)


class ClinicalNoteRead(BaseModel):
    id: UUID
    patient_id: UUID
    case_id: UUID | None
    note_type: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


class TimelineItem(BaseModel):
    type: str
    id: UUID
    title: str
    content: str
    created_at: datetime
