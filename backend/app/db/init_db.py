from app.db.models import *  # noqa: F401,F403
from app.db.session import Base, engine


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()

