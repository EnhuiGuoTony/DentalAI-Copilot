"""业务写入与 Agent 工具共用 Pydantic 契约，避免模型参数绕过 API 校验。

带 discriminator 的联合类型明确每种操作需要哪些字段；版本摘要用于拒绝过期审批。
"""
from datetime import date, datetime
from typing import Annotated, Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, AwareDatetime, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PatientData(StrictModel):
    name: str = Field(min_length=1, max_length=200)
    # 模型可原样回传匿名生日 token；执行前必须还原并通过日期校验。
    date_of_birth: date | Annotated[str, Field(pattern=r"^\[P_[a-f0-9]{32}_DATE_OF_BIRTH\]$")] | None = None
    address: str | None = Field(default=None, max_length=1000)
    patient_number: str | None = Field(default=None, max_length=120)
    medical_record_number: str | None = Field(default=None, max_length=120)


class AppointmentData(StrictModel):
    starts_at: AwareDatetime
    ends_at: AwareDatetime
    reason: str = Field(min_length=1, max_length=500)
    status: Literal["scheduled", "checked_in", "completed", "cancelled"] = "scheduled"

    @model_validator(mode="after")
    def valid_interval(self):
        """要求明确时区和正时长，避免浏览器本地时间被当成 UTC。"""
        if self.ends_at <= self.starts_at:
            raise ValueError("ends_at must be after starts_at")
        return self


class AppointmentCreate(AppointmentData):
    patient_id: UUID


class AppointmentRead(AppointmentCreate):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    created_at: datetime
    version: str


class CreatePatient(StrictModel):
    operation: Literal["create_patient"]
    data: PatientData


class PatientTarget(StrictModel):
    patient_id: UUID


class VersionedTarget(PatientTarget):
    expected_version: str = Field(min_length=64, max_length=64, description="Version returned by the record read tool. Do not invent it.")


class UpdatePatient(VersionedTarget):
    operation: Literal["update_patient"]
    data: PatientData


class DeletePatient(VersionedTarget):
    operation: Literal["delete_patient"]


class CreateAppointment(PatientTarget):
    operation: Literal["create_appointment"]
    data: AppointmentData


class UpdateAppointment(VersionedTarget):
    operation: Literal["update_appointment"]
    appointment_id: UUID
    data: AppointmentData


class DeleteAppointment(VersionedTarget):
    operation: Literal["delete_appointment"]
    appointment_id: UUID


class AddNote(PatientTarget):
    operation: Literal["add_note"]
    note_type: str = Field(default="clinical", min_length=1, max_length=80)
    content: str = Field(min_length=1, max_length=20000)


class DeleteNote(VersionedTarget):
    operation: Literal["delete_note"]
    note_id: UUID


Change = Annotated[CreatePatient | UpdatePatient | DeletePatient | CreateAppointment | UpdateAppointment | DeleteAppointment | AddNote | DeleteNote, Field(discriminator="operation")]


class ChangeRequest(StrictModel):
    change: Change


class MutationResult(BaseModel):
    operation: str
    record_id: UUID
    status: Literal["applied"] = "applied"


class AgentAnswer(BaseModel):
    """Validated final answer with evidence source IDs and explicit limitations."""
    # docstring 会成为模型工具说明，必须使用英文；证据 ID 必须来自实际工具结果。
    answer: str = Field(description="Answer in the user's language. Distinguish applied changes, rejected actions and evidence-based observations.")
    evidence_ids: list[str] = Field(default_factory=list, description="Source IDs from retrieved evidence; empty when none was used.")
    limitations: list[str] = Field(default_factory=list, description="Missing data and clinical or mock limitations.")
