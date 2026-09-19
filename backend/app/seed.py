from datetime import date

from sqlalchemy import select

from app.db.init_db import init_db
from app.db.models import ClinicalCase, ClinicalNote, Patient
from app.db.session import SessionLocal
from app.services.vector_store import VectorStore


def seed() -> None:
    init_db()
    db = SessionLocal()
    try:
        existing = db.execute(select(Patient).where(Patient.name == "Emily Carter")).scalar_one_or_none()
        if existing:
            print(existing.id)
            return

        patient = Patient(name="Emily Carter", date_of_birth=date(2014, 5, 18))
        db.add(patient)
        db.flush()

        case = ClinicalCase(patient_id=patient.id, title="Patient chart review", status="draft")
        db.add(case)
        db.flush()

        notes = [
            ClinicalNote(
                patient_id=patient.id,
                case_id=case.id,
                note_type="recall_exam",
                content=(
                    "Recall exam noted moderate caries risk. Previous restoration on tooth 19. "
                    "Parent reported sensitivity when chewing on the lower left side. Oral hygiene fair."
                ),
            ),
            ClinicalNote(
                patient_id=patient.id,
                note_type="treatment_history",
                content=(
                    "Prior bitewing showed early interproximal radiolucency near posterior teeth. "
                    "Fluoride varnish recommended. Monitor lower molars for progression."
                ),
            ),
            ClinicalNote(
                patient_id=patient.id,
                note_type="medical_history",
                content="No known drug allergies. No major medical contraindications recorded.",
            ),
        ]
        db.add_all(notes)
        db.flush()

        store = VectorStore(db)
        for note in notes:
            store.add_document(patient.id, "clinical_note", note.id, note.content,
                               {"note_type": note.note_type})

        db.commit()
        print(patient.id)
    finally:
        db.close()


if __name__ == "__main__":
    seed()
