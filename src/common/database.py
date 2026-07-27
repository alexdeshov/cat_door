import sqlite3
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import DateTime, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    status: Mapped[str] = mapped_column(String(20), default="recording")
    duration_seconds: Mapped[int] = mapped_column(Integer)
    media_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    side_media_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    label: Mapped[str] = mapped_column(String(30), default="pending")
    cat_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)


def apply_migrations(database_path: Path) -> None:
    migrations_dir = Path(__file__).parent / "migrations"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(version TEXT PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        applied = {
            row[0] for row in connection.execute("SELECT version FROM schema_migrations")
        }
        for migration in sorted(migrations_dir.glob("*.sql")):
            if migration.stem in applied:
                continue
            connection.executescript(migration.read_text(encoding="utf-8"))
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (migration.stem, datetime.now(UTC).isoformat()),
            )
        connection.commit()


def create_session_factory(database_path: Path) -> sessionmaker[Session]:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    apply_migrations(database_path)
    engine = create_engine(
        f"sqlite:///{database_path.as_posix()}",
        connect_args={"check_same_thread": False, "timeout": 10},
    )
    return sessionmaker(engine, expire_on_commit=False)


def session_dependency(factory: sessionmaker[Session]) -> Generator[Session, None, None]:
    with factory() as session:
        yield session
