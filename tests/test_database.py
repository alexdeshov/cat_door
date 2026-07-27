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
    ]
    assert "events" in tables
