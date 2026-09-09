from datetime import datetime

from pydantic import BaseModel, Field


class DoxImportRequest(BaseModel):
    patient_ids: list[int] | None = None
    patient_limit: int | None = Field(default=None, ge=1, le=500)
    include_global_knowledge: bool = True
    include_patient_history: bool = True
    deidentify: bool = True


class DoxImportSummary(BaseModel):
    run_id: str | None = None
    status: str
    patients_requested: int = 0
    patients_imported: int = 0
    notes_imported: int = 0
    knowledge_chunks_imported: int = 0
    clinical_facts_imported: int = 0
    warnings: list[str] = []
    started_at: datetime | None = None
    finished_at: datetime | None = None


class DoxPreviewResponse(BaseModel):
    connected: bool
    tables: dict[str, int | None]
    warnings: list[str] = []
