from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class XrayImageRead(BaseModel):
    id: UUID
    patient_id: UUID
    case_id: UUID
    file_path: str
    width: int | None
    height: int | None
    created_at: datetime

    model_config = {"from_attributes": True}


class XrayFindingRead(BaseModel):
    id: UUID
    image_id: UUID
    case_id: UUID
    category: str
    confidence: float
    tooth_number: str | None
    bbox: dict
    polygon: list | None
    review_status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class FindingReviewUpdate(BaseModel):
    review_status: str
    tooth_number: str | None = None

