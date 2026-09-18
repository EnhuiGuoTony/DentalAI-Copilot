"""读工具返回契约；UUID/日期先规范化，再在模型边界做身份替换。

typed result 让新增字段和缺失数据可被测试发现，避免任意字典逐层传递。
"""
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel
from app.schemas.operations import PatientData, AppointmentData


class NoteRecord(BaseModel):
    id: UUID
    content: str
    note_type: str
    version: str


class FactRecord(BaseModel):
    id: UUID
    type: str
    label: str
    summary: str


class AppointmentRecord(AppointmentData):
    id: UUID
    version: str


class PatientRecord(BaseModel):
    patient_id: UUID
    data: PatientData
    version: str
    notes: list[NoteRecord]
    appointments: list[AppointmentRecord]
    facts: list[FactRecord]


class RecordsResult(BaseModel):
    patients: list[PatientRecord]
    limits: str = "At most 60 notes, 60 appointments and 120 facts per patient."


class EvidenceItem(BaseModel):
    source_id: UUID
    snippet: str
    score: float


class EvidenceResult(BaseModel):
    evidence: list[EvidenceItem]


class ClinicStatistics(BaseModel):
    patients: int
    notes: int
    appointments: int
