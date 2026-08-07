"""Глубина архива записей Frigate."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from src.common.archive_depth import read_archive_depth


def make_database(path: Path, spans: list[tuple[float, float]]) -> None:
    connection = sqlite3.connect(path)
    connection.execute("create table recordings (start_time real, end_time real)")
    connection.executemany("insert into recordings values (?, ?)", spans)
    connection.commit()
    connection.close()


def test_depth_spans_from_oldest_to_newest(tmp_path: Path) -> None:
    database = tmp_path / "frigate.db"
    oldest = datetime(2026, 8, 1, 12, 0, tzinfo=UTC).timestamp()
    newest = datetime(2026, 8, 8, 12, 0, tzinfo=UTC).timestamp()
    make_database(database, [(oldest, oldest + 60), (newest - 60, newest)])

    depth = read_archive_depth(database, tmp_path, retention_days=7)

    assert depth.depth_hours == 168.0
    assert depth.oldest == datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
    assert depth.short is False


def test_short_archive_means_space_ran_out_early(tmp_path: Path) -> None:
    """Глубина меньше настроенной — единственное, ради чего показатель заведён.

    Настроена неделя, а записей трое суток: значит Frigate вытесняет свои же
    записи, не дожив до срока. Узнавать об этом по пропавшей записи, когда она
    понадобилась, — худший из возможных способов.
    """
    database = tmp_path / "frigate.db"
    newest = datetime(2026, 8, 8, 12, 0, tzinfo=UTC).timestamp()
    make_database(database, [(newest - 3 * 86400, newest)])

    depth = read_archive_depth(database, tmp_path, retention_days=7)

    assert depth.short is True


def test_missing_database_is_not_an_error(tmp_path: Path) -> None:
    """Контур калитки может быть не развёрнут — вкладка про наблюдение."""
    depth = read_archive_depth(tmp_path / "нет.db", tmp_path, retention_days=7)

    assert depth.depth_hours is None
    assert depth.short is False
    assert depth.free_gb is not None


def test_almost_full_retention_is_not_reported_as_short(tmp_path: Path) -> None:
    """Архив всегда чуть короче настройки: старый час уходит раньше нового."""
    database = tmp_path / "frigate.db"
    newest = datetime(2026, 8, 8, 12, 0, tzinfo=UTC).timestamp()
    make_database(database, [(newest - 7 * 86400 + 3600, newest)])

    depth = read_archive_depth(database, tmp_path, retention_days=7)

    assert depth.short is False
