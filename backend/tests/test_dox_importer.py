from app.integrations.dox_mysql_importer import (
    deidentified_patient_name,
    deidentify_clinical_text,
    html_to_text,
    stable_dox_uuid,
    template_content_to_text,
)


def test_stable_dox_uuid_is_deterministic():
    assert stable_dox_uuid("Notes", 123) == stable_dox_uuid("Notes", 123)
    assert stable_dox_uuid("Notes", 123) != stable_dox_uuid("Notes", 124)


def test_html_to_text_normalizes_template_content():
    assert html_to_text("<p>Recall <strong>exam</strong></p>") == "Recall exam"


def test_deidentified_patient_name_does_not_use_phi():
    assert deidentified_patient_name(42) == "DOX Patient P000042"


def test_deidentify_clinical_text_redacts_names_and_dates():
    text = "Treatment Narrative - 9/1/2016 Doctor: Sample, Steven Hygienist: Person, Alice"

    redacted = deidentify_clinical_text(text)

    assert "9/1/2016" not in redacted
    assert "Sample" not in redacted
    assert "Alice" not in redacted
    assert "[DATE]" in redacted


def test_deidentify_clinical_text_does_not_redact_lowercase_phrase_as_name():
    redacted = deidentify_clinical_text("Fluoride placed, advised parent not to eat.")

    assert "placed, advised" in redacted


def test_template_content_to_text_extracts_readable_json_fields():
    raw = '{"plainText":"Recall note","items":[{"_text":"Untreated Decay","_selected":false}]}'

    extracted = template_content_to_text(raw)

    assert "Recall note" in extracted
    assert "Untreated Decay" in extracted
    assert "_selected" not in extracted
