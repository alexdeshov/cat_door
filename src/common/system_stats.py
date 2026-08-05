"""Загрузка процессора и память узла — для показателей в шапке админки.

Читается напрямую из `/proc`, без psutil: ради двух чисел не стоит тянуть
зависимость в процесс, который работает без аутентификации. На системах без
`/proc` (разработка на macOS) функции честно возвращают `None`, и шапка просто
не показывает числа.

Процент процессора — всегда разность двух замеров: мгновенного «сейчас» у ядра
нет, есть счётчики тактов с загрузки. Кто хранит предыдущий замер, решает
вызывающая сторона.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROC_STAT = Path("/proc/stat")
PROC_MEMINFO = Path("/proc/meminfo")


@dataclass(frozen=True)
class CpuSample:
    busy: int
    total: int


def read_cpu_sample(path: Path | None = None) -> CpuSample | None:
    """Счётчики тактов из первой строки `/proc/stat`."""
    source = PROC_STAT if path is None else path
    try:
        line = source.read_text(encoding="ascii").splitlines()[0]
    except (OSError, IndexError, UnicodeDecodeError):
        return None
    parts = line.split()
    if not parts or parts[0] != "cpu":
        return None
    try:
        values = [int(value) for value in parts[1:]]
    except ValueError:
        return None
    if len(values) < 5:
        return None
    # iowait считается простоем: ядро в это время готово исполнять что угодно,
    # и записывать ожидание диска в занятость значило бы пугать без причины.
    idle = values[3] + values[4]
    total = sum(values)
    return CpuSample(busy=total - idle, total=total)


def cpu_percent(previous: CpuSample | None, current: CpuSample | None) -> int | None:
    """Процент занятости между двумя замерами. Нечего сравнивать — `None`."""
    if previous is None or current is None:
        return None
    elapsed = current.total - previous.total
    if elapsed <= 0:
        return None
    share = (current.busy - previous.busy) / elapsed
    return max(0, min(100, round(share * 100)))


def read_memory_mb(path: Path | None = None) -> tuple[int, int] | None:
    """Память узла: (занято, всего) в мегабайтах.

    Занято — через `MemAvailable`, а не `MemFree`: свободной ядро считает
    только нетронутую память, а страничный кэш отдаст любому желающему, и
    по `MemFree` узел выглядел бы задыхающимся при полном здоровье.
    """
    source = PROC_MEMINFO if path is None else path
    try:
        text = source.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError):
        return None
    fields: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        value = rest.split()
        if value and value[0].isdigit():
            fields[key] = int(value[0])
    try:
        total_kb = fields["MemTotal"]
        available_kb = fields["MemAvailable"]
    except KeyError:
        return None
    return ((total_kb - available_kb) // 1024, total_kb // 1024)
