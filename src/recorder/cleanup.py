"""Уборка того, что осталось после отключения питания.

Все временные файлы в проекте удаляются в `finally`, и этого достаточно ровно
до того момента, когда процессу не дают доработать: пропажа питания и `SIGKILL`
блока `finally` не исполняют. После такого в каталоге данных остаются
недописанный `.part.mp4`, список для склейки `.concat.txt`, временный файл
настроек и обрезанный на полуслове сегмент кольцевого буфера.

Само по себе это не ломает ничего — но копится от отключения к отключению, а
пустой сегмент ещё и валит сборку события, в которое он попал. Уборка делается
при старте recorder, потому что это единственный момент, когда точно известно:
предыдущая попытка записи уже не продолжится.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path

# Возраст, начиная с которого файл считается брошенным. Веб-процесс может
# писать настройки прямо сейчас, и его временный файл трогать нельзя; минута
# заведомо больше любой такой записи и заведомо меньше промежутка между
# отключениями питания.
ABANDONED_AFTER_SECONDS = 60.0

_TEMPORARY_PATTERNS = ("**/.*.tmp", "**/*.part.mp4", "**/*.concat.txt")


def _older_than(path: Path, cutoff: float) -> bool:
    try:
        return path.stat().st_mtime < cutoff
    except OSError:
        return False


def _abandoned_temporary_files(root: Path, cutoff: float) -> Iterator[Path]:
    for pattern in _TEMPORARY_PATTERNS:
        for path in root.glob(pattern):
            if path.is_file() and _older_than(path, cutoff):
                yield path


def _empty_segments(ring_root: Path) -> Iterator[Path]:
    """Сегменты нулевой длины.

    Ровно то, что остаётся от сегмента, который ffmpeg создал, но не успел
    наполнить. Обрезанный на середине сегмент так не опознать, зато он хотя бы
    воспроизводится; пустой же гарантированно валит склейку события.
    """
    for segment in ring_root.glob("*/*.mp4"):
        with suppress(OSError):
            if segment.is_file() and segment.stat().st_size == 0:
                yield segment


def sweep_incomplete_files(data_dir: Path, *, now: float | None = None) -> int:
    """Удалить брошенные временные файлы и пустые сегменты. Вернуть их число."""
    cutoff = (time.time() if now is None else now) - ABANDONED_AFTER_SECONDS
    removed = 0
    candidates = [
        *_abandoned_temporary_files(data_dir / "runtime", cutoff),
        *_abandoned_temporary_files(data_dir / "events", cutoff),
        *_empty_segments(data_dir / "runtime" / "ring"),
    ]
    for path in candidates:
        try:
            path.unlink()
        except OSError:
            continue
        removed += 1
    return removed
