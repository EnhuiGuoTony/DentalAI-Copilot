from sqlalchemy import text

from app.db.session import SessionLocal


STATEMENTS = [
    "DELETE FROM agent_runs WHERE patient_id IN (SELECT id FROM patients WHERE name LIKE 'DOX Patient%')",
    "DELETE FROM clinical_cases WHERE patient_id IN (SELECT id FROM patients WHERE name LIKE 'DOX Patient%')",
    "DELETE FROM embedding_chunks WHERE source_type LIKE 'dox_%'",
    "DELETE FROM knowledge_chunks WHERE source_type LIKE 'dox_%'",
    "DELETE FROM clinical_notes WHERE note_type LIKE 'dox_%'",
    "DELETE FROM clinical_facts WHERE source_system = 'dox'",
    "DELETE FROM patients WHERE name LIKE 'DOX Patient%'",
    "DELETE FROM dox_import_runs",
]


def main() -> None:
    db = SessionLocal()
    try:
        for statement in STATEMENTS:
            db.execute(text(statement))
        db.commit()
        print("Cleared DOX imported data.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
