"""Показатели узла, которых не хватало шапке: очередь, тепло, видеоядро, диски.

Дополняет `src/common/system_stats.py`, а не повторяет его: процент процессора,
память и температура живут там и читаются оттуда. Здесь — то, что появилось с
вопросом «а не стало ли хуже»: длина очереди на процессор, температура диска,
частота видеоядра и место на томах.

Читается только `/proc` и `/sys`, без внешних утилит и без прав root: веб-служба
работает от непривилегированного пользователя, и метрика, требующая пароля, на
странице просто не появится. Каждый показатель вправе быть `None` — датчика,
которого нет на этой машине, честнее показать прочерком, чем ронять страницу.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from src.common.system_stats import read_cpu_temperature, read_memory_mb

# Зона `acpitz` на узле отдаёт -273.3 °C — абсолютный ноль вместо показания.
# Датчик есть, значения нет, и печатать его нельзя.
ABSOLUTE_ZERO_C = -273.0


@dataclass(frozen=True)
class DiskUsage:
    path: str
    total_gb: float
    free_gb: float

    @property
    def used_percent(self) -> float:
        if self.total_gb <= 0:
            return 0.0
        return round((1 - self.free_gb / self.total_gb) * 100, 1)


@dataclass(frozen=True)
class NodeMetrics:
    load_average: tuple[float, float, float] | None
    cpu_count: int | None
    memory_used_mb: int | None
    memory_total_mb: int | None
    cpu_temperature_c: int | None
    disk_temperature_c: float | None
    gpu_frequency_mhz: int | None
    gpu_frequency_max_mhz: int | None
    uptime_seconds: float | None
    disks: tuple[DiskUsage, ...]

    @property
    def load_per_core(self) -> float | None:
        """Очередь на ядро — единственная форма, сравнимая между машинами.

        Само по себе `load average` не значит ничего: восемь на четырёх ядрах
        это беда, а на тридцати двух — простой.
        """
        if self.load_average is None or not self.cpu_count:
            return None
        return round(self.load_average[0] / self.cpu_count, 2)

    @property
    def memory_total_gb(self) -> float | None:
        if self.memory_total_mb is None:
            return None
        return round(self.memory_total_mb / 1024, 1)

    @property
    def memory_used_percent(self) -> float | None:
        if not self.memory_total_mb or self.memory_used_mb is None:
            return None
        return round(self.memory_used_mb / self.memory_total_mb * 100, 1)

    @property
    def gpu_frequency_percent(self) -> float | None:
        """Частота видеоядра долей от потолка.

        **Это не загрузка.** Настоящую утилизацию даёт только `intel_gpu_top`
        через PMU ядра: он требует прав, которых у веб-службы нет, и на узле не
        установлен. Частота — косвенный признак: простаивающее ядро её
        сбрасывает, занятое держит у потолка. Выдавать её за «загрузку GPU»
        нельзя, и на странице она подписана частотой.
        """
        if not self.gpu_frequency_max_mhz or self.gpu_frequency_mhz is None:
            return None
        return round(self.gpu_frequency_mhz / self.gpu_frequency_max_mhz * 100, 1)


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return None


def _read_int(path: Path) -> int | None:
    raw = _read_text(path)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def read_load_average(root: Path = Path("/")) -> tuple[float, float, float] | None:
    raw = _read_text(root / "proc" / "loadavg")
    if raw is None:
        return None
    parts = raw.split()
    if len(parts) < 3:
        return None
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        return None


def read_uptime_seconds(root: Path = Path("/")) -> float | None:
    raw = _read_text(root / "proc" / "uptime")
    if raw is None:
        return None
    try:
        return float(raw.split()[0])
    except (IndexError, ValueError):
        return None


def read_disk_temperature_c(root: Path = Path("/")) -> float | None:
    """Температура NVMe. Диск греется молча и умирает тоже молча."""
    hwmon = root / "sys" / "class" / "hwmon"
    try:
        candidates = sorted(hwmon.glob("hwmon*"))
    except OSError:
        return None
    for sensor in candidates:
        if _read_text(sensor / "name") != "nvme":
            continue
        raw = _read_int(sensor / "temp1_input")
        if raw is None:
            continue
        celsius = raw / 1000
        return round(celsius, 1) if celsius > ABSOLUTE_ZERO_C else None
    return None


def read_gpu_frequency_mhz(root: Path = Path("/")) -> tuple[int | None, int | None]:
    """Текущая и предельная частота видеоядра, если оно вообще есть."""
    drm = root / "sys" / "class" / "drm"
    try:
        cards = sorted(drm.glob("card*"))
    except OSError:
        return None, None
    for card in cards:
        current = _read_int(card / "gt_act_freq_mhz")
        maximum = _read_int(card / "gt_max_freq_mhz")
        if current is not None or maximum is not None:
            return current, maximum
    return None, None


def read_disks(paths: list[Path]) -> tuple[DiskUsage, ...]:
    """Место на томах. Пути, которых нет, молча пропускаются.

    Молча — потому что состав томов зависит от узла: внешний архив может быть
    не подключён, и это отдельный разговор, а не повод ломать страницу.
    """
    usages: list[DiskUsage] = []
    for path in paths:
        try:
            usage = shutil.disk_usage(path)
        except OSError:
            continue
        usages.append(
            DiskUsage(
                path=str(path),
                total_gb=round(usage.total / 1024**3, 1),
                free_gb=round(usage.free / 1024**3, 1),
            )
        )
    return tuple(usages)


def read_node_metrics(
    disk_paths: list[Path] | None = None,
    root: Path = Path("/"),
    cpu_count: int | None = None,
) -> NodeMetrics:
    memory = read_memory_mb()
    frequency, frequency_max = read_gpu_frequency_mhz(root)
    return NodeMetrics(
        load_average=read_load_average(root),
        cpu_count=cpu_count if cpu_count is not None else os.cpu_count(),
        memory_used_mb=memory[0] if memory else None,
        memory_total_mb=memory[1] if memory else None,
        cpu_temperature_c=read_cpu_temperature(),
        disk_temperature_c=read_disk_temperature_c(root),
        gpu_frequency_mhz=frequency,
        gpu_frequency_max_mhz=frequency_max,
        uptime_seconds=read_uptime_seconds(root),
        disks=read_disks(disk_paths or []),
    )
