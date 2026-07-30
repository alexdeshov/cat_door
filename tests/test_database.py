import sqlite3
from pathlib import Path
from uuid import uuid4

from src.common.database import create_session_factory


def test_initial_migration_is_applied_once() -> None:
    runtime_path = Path("data") / "test-runtime" / uuid4().hex
    runtime_path.mkdir(parents=True)
    database_path = runtime_path / "catdoor.sqlite3"

    create_session_factory(database_path)
    create_session_factory(database_path)

    with sqlite3.connect(database_path) as connection:
        versions = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert versions == [
        ("001_initial",),
        ("002_side_media",),
        ("003_training_target",),
        ("004_automatic_event_metadata",),
        ("005_model_metrics",),
    ]
    assert "events" in tables
    assert {"model_versions", "event_predictions", "evaluation_runs"} <= tables


def test_sqlite_uses_wal_and_normal_synchronous_mode(runtime_path: Path) -> None:
    database_path = runtime_path / "catdoor.sqlite3"

    factory = create_session_factory(database_path)

    with factory() as session:
        connection = session.connection()
        journal_mode = connection.exec_driver_sql("PRAGMA journal_mode").scalar_one()
        synchronous = connection.exec_driver_sql("PRAGMA synchronous").scalar_one()
        busy_timeout = connection.exec_driver_sql("PRAGMA busy_timeout").scalar_one()

    assert journal_mode == "wal"
    assert synchronous == 1
    assert busy_timeout == 10_000
