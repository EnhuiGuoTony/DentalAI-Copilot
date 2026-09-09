from sqlalchemy import text

from app.db.session import SessionLocal


def main() -> None:
    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM knowledge_chunks WHERE source_type LIKE 'dox_%'"))
        db.commit()
        print("Cleared DOX knowledge chunks.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
