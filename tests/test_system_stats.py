"""Показатели узла читаются из /proc — здесь /proc подменён файлами."""

from pathlib import Path

from src.common.system_stats import (
    CpuSample,
    cpu_percent,
    read_cpu_sample,
    read_memory_mb,
)


def test_cpu_sample_counts_iowait_as_idle(tmp_path: Path) -> None:
    stat = tmp_path / "stat"
    stat.write_text("cpu  100 0 50 800 50 0 0 0 0 0\ncpu0 1 2 3 4 5\n")

    sample = read_cpu_sample(stat)

    assert sample == CpuSample(busy=150, total=1000)


def test_cpu_percent_is_a_delta_between_samples() -> None:
    first = CpuSample(busy=150, total=1000)
    second = CpuSample(busy=250, total=1400)

    assert cpu_percent(first, second) == 25
    # Нечего сравнивать — честный None, а не ноль, похожий на простаивающий узел.
    assert cpu_percent(None, second) is None
    assert cpu_percent(first, first) is None


def test_memory_uses_available_not_free(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       8000000 kB\n"
        "MemFree:         100000 kB\n"
        "MemAvailable:   6000000 kB\n"
        "Cached:         3000000 kB\n"
    )

    assert read_memory_mb(meminfo) == (1953, 7812)


def test_missing_proc_degrades_to_none(tmp_path: Path) -> None:
    assert read_cpu_sample(tmp_path / "absent") is None
    assert read_memory_mb(tmp_path / "absent") is None
    assert read_cpu_sample(tmp_path / "absent") is None


def test_cpu_temperature_takes_the_hottest_zone(tmp_path: Path) -> None:
    (tmp_path / "thermal_zone0").mkdir()
    (tmp_path / "thermal_zone1").mkdir()
    (tmp_path / "thermal_zone0" / "temp").write_text("27000\n")
    (tmp_path / "thermal_zone1" / "temp").write_text("76500\n")

    from src.common.system_stats import read_cpu_temperature

    assert read_cpu_temperature(str(tmp_path / "thermal_zone*" / "temp")) == 76
    assert read_cpu_temperature(str(tmp_path / "absent*" / "temp")) is None
