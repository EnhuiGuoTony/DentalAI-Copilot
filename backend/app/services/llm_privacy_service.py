"""Remove the agreed patient identifiers only at the LLM boundary.

The database intentionally retains source text for the test-data workflow.  This
service is used immediately before an external model invocation; it does not
modify stored notes, facts, chunks, or API responses.
"""

from __future__ import annotations

from datetime import date, datetime
import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Patient


class LlmPrivacyService:
    _field_tokens = {
        "name": "[PATIENT_NAME]",
        "patient_name": "[PATIENT_NAME]",
        "date_of_birth": "[DATE_OF_BIRTH]",
        "dob": "[DATE_OF_BIRTH]",
        "address": "[ADDRESS]",
    }

    def __init__(self, db: Session) -> None:
        self.replacements: list[tuple[str, str]] = []
        for patient in db.execute(select(Patient)).scalars():
            self._add(patient.name, "[PATIENT_NAME]")
            self._add_date(patient.date_of_birth)
            self._add(patient.address, "[ADDRESS]")
        self.replacements.sort(key=lambda item: len(item[0]), reverse=True)

    def redact(self, value: Any, field_name: str = "") -> Any:
        field_token = self._field_tokens.get(field_name.lower())
        if field_token and value not in (None, ""):
            return field_token
        if isinstance(value, dict):
            return {key: self.redact(item, key) for key, item in value.items()}
        if isinstance(value, list):
            return [self.redact(item) for item in value]
        if isinstance(value, (date, datetime)):
            return "[DATE_OF_BIRTH]" if field_name.lower() in {"date_of_birth", "dob"} else value.isoformat()
        if not isinstance(value, str):
            return value
        result = value
        for source, replacement in self.replacements:
            result = re.sub(re.escape(source), replacement, result, flags=re.IGNORECASE)
        return result

    def _add(self, value: str | None, replacement: str) -> None:
        normalized = (value or "").strip()
        if normalized:
            self.replacements.append((normalized, replacement))

    def _add_date(self, value: date | None) -> None:
        if value is None:
            return
        self._add(value.isoformat(), "[DATE_OF_BIRTH]")
        self._add(f"{value.month}/{value.day}/{value.year}", "[DATE_OF_BIRTH]")
        self._add(f"{value.month:02d}/{value.day:02d}/{value.year}", "[DATE_OF_BIRTH]")
