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
        ("006_preview_offset",),
        ("007_gate_passes",),
        ("008_gate_passes_indexes",),
        ("009_gate_pass_context",),
        ("010_gate_pass_movement",),
        ("011_gate_passes_repair",),
        ("012_gate_pass_label",),
        ("013_gate_pass_pulses",),
        ("014_gate_pass_pulse_state",),
    ]
    assert "events" in tables
    assert {"model_versions", "event_predictions", "evaluation_runs"} <= tables
    assert "gate_passes" in tables


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


def test_indexes_survive_a_migration_torn_by_power_loss(runtime_path: Path) -> None:
    """`executescript` фиксирует каждую команду сама: транзакции вокруг нет.

    Обрыв между `CREATE TABLE gate_passes` и созданием индексов оставлял
    таблицу без них навсегда: повтор миграции упирается в «table already
    exists», это считается признаком уже сделанной работы, и версия помечается
    применённой. Заметить пропажу можно было бы только по медленному журналу
    через полгода.
    """
    database_path = runtime_path / "catdoor.sqlite3"
    create_session_factory(database_path)

    # Повторяем обрыв: таблица есть, индексов нет, версии не отмечены.
    with sqlite3.connect(database_path) as connection:
        for name in ("created_at", "person_name", "decision"):
            connection.execute(f"DROP INDEX IF EXISTS ix_gate_passes_{name}")
        connection.execute(
            "DELETE FROM schema_migrations WHERE version IN "
            "('007_gate_passes', '008_gate_passes_indexes')"
        )
        connection.commit()

    create_session_factory(database_path)

    with sqlite3.connect(database_path) as connection:
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='gate_passes'"
            )
        }

    assert {
        "ix_gate_passes_created_at",
        "ix_gate_passes_person_name",
        "ix_gate_passes_decision",
    } <= indexes


def test_a_migration_torn_in_the_middle_heals_itself(runtime_path: Path) -> None:
    """Обрыв питания посреди миграции из нескольких команд.

    Воспроизведено 2026-08-12: миграция 009 добавляет четыре столбца, и обрыв
    оставлял один. Повтор упирался в «duplicate column» на первой команде, это
    считалось признаком сделанной работы, и остальные три не выполнялись уже
    никогда — а версия помечалась применённой. Служба после этого падала на
    «no such column» при каждой записи прохода.
    """
    database_path = runtime_path / "catdoor.sqlite3"
    create_session_factory(database_path)
    torn = ("first_box_bottom", "gate_state_before", "gate_state_after", "area_ratio")

    with sqlite3.connect(database_path) as connection:
        for column in torn:
            connection.execute(f"ALTER TABLE gate_passes DROP COLUMN {column}")
        connection.execute(
            "DELETE FROM schema_migrations WHERE version IN "
            "('009_gate_pass_context', '010_gate_pass_movement', '011_gate_passes_repair')"
        )
        connection.commit()

    create_session_factory(database_path)

    with sqlite3.connect(database_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(gate_passes)")}

    assert set(torn) <= columns


def test_a_semicolon_inside_a_comment_does_not_split_a_statement() -> None:
    """Комментарии в миграциях по-русски, а точка с запятой в русской фразе обычна.

    12 августа такая запятая разрезала `ALTER TABLE` пополам, и миграция упала
    с `near "импульс": syntax error`.
    """
    from src.common.database import _statements

    parts = _statements(
        "-- один прошёл; другой остался\n"
        "ALTER TABLE gate_passes ADD COLUMN a TEXT;\n"
        "-- и ещё; хвост\n"
        "ALTER TABLE gate_passes ADD COLUMN b TEXT;\n"
    )

    assert parts == [
        "ALTER TABLE gate_passes ADD COLUMN a TEXT",
        "ALTER TABLE gate_passes ADD COLUMN b TEXT",
    ]
