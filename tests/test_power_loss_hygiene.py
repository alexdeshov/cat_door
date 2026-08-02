"""Проверки находок аудита отказоустойчивости: питание, диск, база."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.common.database import Event, apply_migrations, create_session_factory
from src.recorder.events import expire_stale_recordings
from src.recorder.preview import local_mjpeg_stream, publish_preview


def test_interrupted_recordings_are_marked_failed(runtime_path: Path) -> None:
    """Строка «идёт запись» без второго коммита раньше жила вечно."""
    database = runtime_path / "test.sqlite3"
    factory = create_session_factory(database)
    with factory() as session:
        session.add(Event(duration_seconds=8, status="recording",
                          created_at=datetime.now(UTC) - timedelta(hours=1)))
        session.add(Event(duration_seconds=8, status="recording"))
        session.add(Event(duration_seconds=8, status="ready"))
        session.commit()

    expired = expire_stale_recordings(database)

    assert expired == 1
    with factory() as session:
        statuses = sorted(
            (event.status, event.error_code or "") for event in session.query(Event).all()
        )
    # Свежая запись не тронута: она может идти прямо сейчас.
    assert statuses == [("failed", "interrupted"), ("ready", ""), ("recording", "")]


def test_replayed_migration_is_not_fatal(runtime_path: Path) -> None:
    """Потеря питания между executescript и записью версии — не приговор.

    Прежде повтор миграции падал на «duplicate column», и обе службы уходили
    в вечный рестарт до ручной правки schema_migrations.
    """
    import sqlite3

    database = runtime_path / "test.sqlite3"
    apply_migrations(database)
    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM schema_migrations WHERE version LIKE '002%'")
        connection.commit()

    apply_migrations(database)  # не должно упасть

    with sqlite3.connect(database) as connection:
        versions = {row[0] for row in connection.execute("SELECT version FROM schema_migrations")}
    assert any(version.startswith("002") for version in versions)


def test_stale_preview_ends_the_local_stream(runtime_path: Path) -> None:
    """Вечный цикл без yield сжигал поток пула админки навсегда."""
    import numpy as np

    path = runtime_path / "runtime" / "previews" / "face.jpg"
    publish_preview(path, np.zeros((4, 4, 3), dtype=np.uint8))
    import os
    stale = datetime.now(UTC).timestamp() - 3600
    os.utime(path, (stale, stale))

    frames = list(local_mjpeg_stream(path, fps=5, idle_timeout_seconds=20.0))

    assert frames == []
