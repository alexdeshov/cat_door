import sqlite3
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

SQLITE_BUSY_TIMEOUT_SECONDS = 10


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
    # Секунда записи, на которой детектор был увереннее всего в коте. Плеер
    # открывает проход с неё; `None` — момент неизвестен, тогда с начала.
    preview_offset_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)


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


class GatePass(Base):
    """Проход у калитки: событие Frigate и решение системы о нём.

    Отдельно от `Event`: у кошачьих проходов свои клипы, метки и обучение, а
    здесь — люди, имена из библиотеки лиц Frigate и журнал решений.
    """

    __tablename__ = "gate_passes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Ключ Frigate, а не наш: одно событие перечитывается несколько опросов
    # подряд, пока распознавание не поставит имя. Уникальность превращает
    # повторное чтение в обновление строки.
    frigate_event_id: Mapped[str] = mapped_column(String(100), unique=True)
    # Время начала события у Frigate, а не время записи строки.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )
    camera: Mapped[str] = mapped_column(String(100))
    person_name: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    zone: Mapped[str | None] = mapped_column(String(100), nullable=True)
    decision: Mapped[str] = mapped_column(String(40), index=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    snapshot_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    # Какие факторы сказали «за», «против» и «не знаю». Пусто до появления
    # функции решения; без этого разбирать отказы невозможно.
    factors_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Что ушло на плату. Пусто: механика не подключена и кода записи нет.
    pulse_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    modbus_ok: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    # Открылась ли калитка на самом деле. Геркона на ней нет — поля заведены
    # заранее и остаются пустыми.
    reed_before: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    reed_after: Mapped[bool | None] = mapped_column(Boolean, nullable=True)


def apply_migrations(database_path: Path) -> None:
    migrations_dir = Path(__file__).parent / "migrations"
    with sqlite3.connect(database_path, timeout=SQLITE_BUSY_TIMEOUT_SECONDS) as connection:
        _configure_sqlite_connection(connection)
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
            try:
                connection.executescript(migration.read_text(encoding="utf-8"))
            except sqlite3.OperationalError as exc:
                # `executescript` коммитит и работает вне транзакции, поэтому
                # потеря питания между скриптом и записью версии оставляет
                # миграцию применённой, но не отмеченной. При следующем старте
                # повтор упирается в «duplicate column» — и раньше это клало
                # обе службы в вечный рестарт до ручной правки базы. Такой
                # повтор — свидетельство уже сделанной работы, а не ошибка.
                if not _already_applied(exc):
                    raise
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (migration.stem, datetime.now(UTC).isoformat()),
            )
        connection.commit()


def _already_applied(exc: sqlite3.OperationalError) -> bool:
    message = str(exc).lower()
    return "duplicate column" in message or "already exists" in message


def _configure_sqlite_connection(connection: Any, _: Any = None) -> None:
    cursor = connection.cursor()
    try:
        cursor.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_SECONDS * 1000}")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
    finally:
        cursor.close()


def create_session_factory(database_path: Path) -> sessionmaker[Session]:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    apply_migrations(database_path)
    engine = create_engine(
        f"sqlite:///{database_path.as_posix()}",
        connect_args={
            "check_same_thread": False,
            "timeout": SQLITE_BUSY_TIMEOUT_SECONDS,
        },
    )
    event.listen(engine, "connect", _configure_sqlite_connection)
    return sessionmaker(engine, expire_on_commit=False)


def session_dependency(factory: sessionmaker[Session]) -> Generator[Session, None, None]:
    with factory() as session:
        yield session
