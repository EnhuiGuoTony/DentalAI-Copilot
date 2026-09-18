"""不依赖数据库的协议与隐私回归测试，防止身份替换破坏控制字段。"""
from uuid import uuid4
from datetime import date
import pytest
from pydantic import ValidationError
from app.schemas.operations import ChangeRequest, AppointmentData
from app.services.pms_agent_service import PatientAliases
from app.schemas.auth import RegisterRequest


def test_identity_mapping_preserves_control_fields_and_json_quotes():
    pid = uuid4()
    prefix = f"[P_{pid.hex}_"
    mapping = {prefix + "NAME]": 'O\"Brien', prefix + "PATIENT_NUMBER]": "1", prefix + "DATE_OF_BIRTH]": "2001-01-01"}
    aliases = PatientAliases(mapping)
    raw = {"patient_id": str(pid), "version": "1" * 64, "data": {"name": 'O\"Brien', "patient_number": "1", "date_of_birth": "2001-01-01"}, "starts_at": "2001-01-01T01:00:00+00:00"}
    hidden = aliases.transform(raw)
    assert hidden["patient_id"] == str(pid)
    assert hidden["version"] == "1" * 64
    assert hidden["starts_at"] == raw["starts_at"]
    assert hidden["data"]["patient_number"] != "1"
    assert aliases.transform(hidden, reveal=True) == raw
    change = {"change": {"operation": "update_patient", "patient_id": str(pid), "expected_version": "1" * 64, "data": hidden["data"]}}
    ChangeRequest.model_validate(change)
    restored = ChangeRequest.model_validate(aliases.transform(change, reveal=True)).change
    assert restored.data.date_of_birth == date(2001, 1, 1)


def test_registration_does_not_trim_password():
    req = RegisterRequest(username="test_user", password="  password-with-spaces  ", display_name="  Doctor  ")
    assert req.password == "  password-with-spaces  "
    assert req.display_name == "Doctor"


def test_operation_union_rejects_extra_fields_and_naive_times():
    with pytest.raises(ValidationError):
        ChangeRequest.model_validate({"change": {"operation": "create_patient", "data": {"name": "Demo"}, "user_id": str(uuid4())}})
    with pytest.raises(ValidationError):
        AppointmentData(starts_at="2026-10-01T09:00:00", ends_at="2026-10-01T10:00:00", reason="Demo")
