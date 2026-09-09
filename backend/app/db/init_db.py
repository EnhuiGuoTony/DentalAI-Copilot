from app.db.models import *  # noqa: F401,F403
from sqlalchemy import text

from app.db.session import Base, engine


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    # This project currently has no migration framework.  Keep existing local
    # demo databases compatible when patient master-data fields are introduced.
    if engine.dialect.name == "postgresql":
        statements = (
            "ALTER TABLE patients ADD COLUMN IF NOT EXISTS dox_patient_id VARCHAR(80)",
            "ALTER TABLE patients ADD COLUMN IF NOT EXISTS patient_number VARCHAR(120)",
            "ALTER TABLE patients ADD COLUMN IF NOT EXISTS medical_record_number VARCHAR(120)",
            "ALTER TABLE patients ADD COLUMN IF NOT EXISTS address TEXT",
            "CREATE UNIQUE INDEX IF NOT EXISTS ix_patients_dox_patient_id ON patients (dox_patient_id)",
            "CREATE INDEX IF NOT EXISTS ix_patients_patient_number ON patients (patient_number)",
            "CREATE INDEX IF NOT EXISTS ix_patients_medical_record_number ON patients (medical_record_number)",
        )
        with engine.begin() as connection:
            for statement in statements:
                connection.execute(text(statement))


if __name__ == "__main__":
    init_db()
