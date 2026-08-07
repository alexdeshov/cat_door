"""Глубина архива записей Frigate: с какого момента записи вообще есть.

У Frigate нет пометки «эта запись скоро удалится», и завести её нечем. Но
полезнее другое: видеть, за какой период архив существует. Настроена неделя, а
глубина просела до трёх суток — значит место кончилось раньше срока и Frigate
вытесняет свои же записи. Заметить это по пропавшей записи, когда она
понадобилась, — худший из возможных способов.

База Frigate читается только на чтение и только ради двух чисел. Своей копии
этих данных у CatDoor нет и заводить её незачем: источник правды — тот, кто
пишет.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class ArchiveDepth:
    oldest: datetime | None
    newest: datetime | None
    retention_days: int
    free_gb: float | None
    total_gb: float | None

    @property
    def depth_hours(self) -> float | None:
        if self.oldest is None or self.newest is None:
            return None
        return (self.newest - self.oldest).total_seconds() / 3600

    @property
    def short(self) -> bool:
        """Глубина заметно меньше настроенной — место кончается раньше срока.

        Порог в четверть суток, а не ноль: архив всегда чуть короче настройки,
        потому что самый старый час удаляется до того, как запишется новый.
        """
        hours = self.depth_hours
        if hours is None or self.retention_days <= 0:
            return False
        return hours < self.retention_days * 24 - 6


def _volume(path: Path) -> tuple[float | None, float | None]:
    try:
        stats = os.statvfs(path)
    except OSError:
        return None, None
    free = stats.f_bavail * stats.f_frsize / 1024**3
    total = stats.f_blocks * stats.f_frsize / 1024**3
    return free, total


def read_archive_depth(
    database_path: Path,
    recordings_path: Path,
    retention_days: int,
) -> ArchiveDepth:
    """Границы архива и место на томе. Недоступная база — не ошибка.

    Контур калитки может быть развёрнут не на этом узле или выключен вовсе, и
    вкладка не должна падать из-за отсутствующего файла: она про наблюдение, а
    не про управление.
    """
    free_gb, total_gb = _volume(recordings_path)
    oldest = newest = None
    if database_path.is_file():
        try:
            connection = sqlite3.connect(
                f"file:{database_path}?mode=ro", uri=True, timeout=2.0
            )
            try:
                row = connection.execute(
                    "select min(start_time), max(end_time) from recordings"
                ).fetchone()
            finally:
                connection.close()
        except sqlite3.Error:
            row = None
        if row and row[0] and row[1]:
            oldest = datetime.fromtimestamp(row[0], UTC)
            newest = datetime.fromtimestamp(row[1], UTC)
    return ArchiveDepth(
        oldest=oldest,
        newest=newest,
        retention_days=retention_days,
        free_gb=free_gb,
        total_gb=total_gb,
    )
