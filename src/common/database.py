import sqlite3
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, create_engine
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
    training_target: Mapped[str] = mapped_column(String(20), default="excluded")
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    trigger_source: Mapped[str] = mapped_column(String(20), default="manual")
    initiating_camera: Mapped[str | None] = mapped_column(String(10), nullable=True)
    detector_confidence_face: Mapped[float | None] = mapped_column(Float, nullable=True)
    detector_confidence_side: Mapped[float | None] = mapped_column(Float, nullable=True)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(30), index=True)
    file_name: Mapped[str] = mapped_column(String(255))
    sha256: Mapped[str] = mapped_column(String(64), unique=True)
    metadata_json: Mapped[str] = mapped_column(Text)
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    run_kind: Mapped[str] = mapped_column(String(20), default="identity")
    filter_json: Mapped[str] = mapped_column(Text)
    detector_model_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id"), nullable=True
    )
    face_model_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id"), nullable=True
    )
    side_model_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(20))
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class EventPrediction(Base):
    __tablename__ = "event_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[int] = mapped_column(
        ForeignKey("events.id", ondelete="CASCADE"), index=True
    )
    model_version_id: Mapped[int] = mapped_column(
        ForeignKey("model_versions.id"), index=True
    )
    view: Mapped[str] = mapped_column(String(20))
    origin: Mapped[str] = mapped_column(String(20), index=True)
    evaluation_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    decision: Mapped[str] = mapped_column(String(120))
    predicted_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    frames_used: Mapped[int] = mapped_column(Integer, default=0)
    max_detector_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_inference_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw_scores_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


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
