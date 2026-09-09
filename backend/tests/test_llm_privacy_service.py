from datetime import date

from app.services.llm_privacy_service import LlmPrivacyService


def test_redact_replaces_patient_identifiers_in_nested_llm_payload():
    service = LlmPrivacyService.__new__(LlmPrivacyService)
    service.replacements = [
        ("Ada Lovelace", "[PATIENT_NAME]"),
        ("1980-12-10", "[DATE_OF_BIRTH]"),
        ("12 Example Road, Austin", "[ADDRESS]"),
    ]
    payload = {
        "name": "Ada Lovelace",
        "date_of_birth": "1980-12-10",
        "address": "12 Example Road, Austin",
        "notes": ["Ada Lovelace lives at 12 Example Road, Austin; DOB 1980-12-10."],
        "clinical_text": "No identifiers should be removed besides the configured values.",
    }

    redacted = service.redact(payload)

    assert redacted["name"] == "[PATIENT_NAME]"
    assert redacted["date_of_birth"] == "[DATE_OF_BIRTH]"
    assert redacted["address"] == "[ADDRESS]"
    assert redacted["notes"] == ["[PATIENT_NAME] lives at [ADDRESS]; DOB [DATE_OF_BIRTH]."]
    assert redacted["clinical_text"] == payload["clinical_text"]


def test_redact_is_case_insensitive_inside_free_text():
    service = LlmPrivacyService.__new__(LlmPrivacyService)
    service.replacements = [("Ada Lovelace", "[PATIENT_NAME]")]

    assert service.redact("ADA LOVELACE needs a recall visit.") == "[PATIENT_NAME] needs a recall visit."
