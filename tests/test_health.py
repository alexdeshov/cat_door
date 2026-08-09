import os
from pathlib import Path

import pytest

from src.common import health
from src.common.node_metrics import (
    DiskUsage,
    NodeMetrics,
    read_disk_temperature_c,
    read_gpu_frequency_mhz,
    read_load_average,
)


def metrics(**overrides: object) -> NodeMetrics:
    values: dict[str, object] = {
        "load_average": (2.0, 2.0, 2.0),
        "cpu_count": 16,
        "memory_used_mb": 4096,
        "memory_total_mb": 32000,
        "cpu_temperature_c": 55,
        "disk_temperature_c": 42.0,
        "gpu_frequency_mhz": 1200,
        "gpu_frequency_max_mhz": 1500,
        "uptime_seconds": 3600.0,
        "disks": (DiskUsage("/srv/data", 360.0, 315.0),),
    }
    values.update(overrides)
    return NodeMetrics(**values)  # type: ignore[arg-type]


REFERENCE_STATUS = {
    "detector_device_actual": "GPU.0",
    "detector_model": "yolo11n-416.onnx",
    "hwaccel_actual": "vaapi",
    "frames_per_second": {"face": 12.1, "side": 6.0},
    "inference_ms": 13.5,
}

REFERENCE_CONFIG = {
    "face": {"confidence_threshold": 0.5, "sample_fps": 12.0},
    "side": {"confidence_threshold": 0.5, "sample_fps": 6.0},
}

REFERENCE_STATS = {
    "detectors": {"openvino": {"inference_speed": 4.78}},
    "cameras": {"test_cam": {"skipped_fps": 0.0}},
    "detection_fps": 96.0,
}


def build(**overrides: object) -> list[health.HealthGroup]:
    values: dict[str, object] = {
        "recorder_status": REFERENCE_STATUS,
        "detection_config": REFERENCE_CONFIG,
        "frigate_stats": REFERENCE_STATS,
        "gate_detector_device": "GPU",
        "metrics": metrics(),
        "gpu_name": "Intel(R) Iris(R) Xe Graphics (iGPU)",
    }
    values.update(overrides)
    return health.build_health(**values)  # type: ignore[arg-type]


def find(groups: list[health.HealthGroup], key: str) -> health.Check:
    for group in groups:
        for check in group.checks:
            if check.key == key:
                return check
    raise AssertionError(f"проверки {key} нет")


def test_the_accepted_configuration_is_entirely_ok() -> None:
    """Приёмка 2026-08-09 прошла 24 из 24 — эталон обязан это подтверждать.

    Если этот тест покраснеет после правки эталона, значит эталон разошёлся с
    той конфигурацией, которую объявили рабочей.
    """
    assert health.overall_status(build()) == "ok"


def test_detector_back_on_the_processor_is_a_failure() -> None:
    groups = build(
        recorder_status={**REFERENCE_STATUS, "detector_device_actual": "CPU"},
    )

    assert find(groups, "detector_device").status == "fail"
    assert health.overall_status(groups) == "fail"


def test_door_threshold_slipping_back_is_a_failure() -> None:
    """0.30 на дверце — это посылки, принятые за котов."""
    config = {**REFERENCE_CONFIG, "face": {"confidence_threshold": 0.3, "sample_fps": 12.0}}

    assert find(build(detection_config=config), "threshold_face").status == "fail"


def test_the_real_rate_is_checked_apart_from_the_setting() -> None:
    """Настройка может говорить одно, а конвейер делать другое."""
    groups = build(
        recorder_status={**REFERENCE_STATUS, "frames_per_second": {"face": 4.0, "side": 6.0}},
    )

    assert find(groups, "sample_fps_face").status == "ok"
    assert find(groups, "rate_face").status == "fail"


def test_skipped_frames_on_the_gate_are_visible() -> None:
    """До переезда детектора Frigate на видеоядро калитка теряла 0.3 к/с."""
    stats = {**REFERENCE_STATS, "cameras": {"test_cam": {"skipped_fps": 0.3}}}

    assert find(build(frigate_stats=stats), "gate_skipped_test_cam").status != "ok"


def test_unreachable_frigate_is_not_the_same_as_broken() -> None:
    groups = build(frigate_stats=None)

    assert find(groups, "gate_reachable").status == "fail"
    assert not any(check.key.startswith("gate_skipped") for check in groups[1].checks)


def test_uhd_naming_means_single_channel_memory() -> None:
    """То же ядро зовётся UHD при одноканальной памяти — это диагноз, не имя."""
    groups = build(gpu_name="Intel(R) UHD Graphics (iGPU)")

    assert find(groups, "gpu_name").status == "fail"


def test_a_missing_sensor_is_unknown_and_never_ok() -> None:
    """Прочерк не должен выглядеть как здоровье."""
    groups = build(metrics=metrics(cpu_temperature_c=None))

    assert find(groups, "cpu_temperature").status == "unknown"
    assert find(groups, "cpu_temperature").actual == "—"


def test_overheating_is_a_failure_and_warmth_is_a_warning() -> None:
    assert find(build(metrics=metrics(cpu_temperature_c=95)), "cpu_temperature").status == "fail"
    assert find(build(metrics=metrics(cpu_temperature_c=82)), "cpu_temperature").status == "warn"


def test_gpu_occupancy_is_derived_from_both_circuits() -> None:
    """Оценка занятости складывается из работы обоих контуров.

    Коты: (12.1 + 6.0) кадров по 13.5 мс. Калитка: 96 детекций по 4.78 мс.
    """
    occupancy = health.gpu_occupancy_percent(REFERENCE_STATUS, REFERENCE_STATS)

    assert occupancy == pytest.approx((18.1 * 13.5 + 96.0 * 4.78) / 10, abs=0.2)


def test_occupancy_is_unknown_when_nothing_reports_timings() -> None:
    assert health.gpu_occupancy_percent({}, None) is None


def test_load_is_read_per_core_because_bare_load_means_nothing() -> None:
    assert metrics(load_average=(8.0, 8.0, 8.0), cpu_count=16).load_per_core == 0.5
    assert metrics(load_average=(8.0, 8.0, 8.0), cpu_count=4).load_per_core == 2.0


def test_absolute_zero_from_a_dead_sensor_is_not_a_temperature(tmp_path: Path) -> None:
    """Зона acpitz на узле отдаёт -273.3 °C: датчик есть, показания нет."""
    sensor = tmp_path / "sys" / "class" / "hwmon" / "hwmon0"
    sensor.mkdir(parents=True)
    (sensor / "name").write_text("nvme", encoding="utf-8")
    (sensor / "temp1_input").write_text("-273300", encoding="utf-8")

    assert read_disk_temperature_c(tmp_path) is None


def test_metrics_survive_a_machine_without_proc(tmp_path: Path) -> None:
    """На маке ни /proc, ни /sys нет — страница обязана открыться и там."""
    assert read_load_average(tmp_path) is None
    assert read_gpu_frequency_mhz(tmp_path) == (None, None)
    assert read_disk_temperature_c(tmp_path) is None


def test_the_same_volume_is_listed_once(tmp_path: Path) -> None:
    """На узле /var/lib/catdoor подставлен bind-монтированием в /srv/data.

    Без схлопывания по устройству страница показывала один том дважды с
    одинаковыми числами — шум ровно там, где всё затевалось ради сигнала.
    """
    from src.common.node_metrics import read_disks

    nested = tmp_path / "bind"
    nested.mkdir()

    disks = read_disks([tmp_path, nested])

    assert len(disks) == 1
    assert disks[0].path == str(tmp_path)


def test_a_missing_volume_is_skipped_without_breaking_the_rest(tmp_path: Path) -> None:
    from src.common.node_metrics import read_disks

    disks = read_disks([tmp_path / "нет-такого", tmp_path])

    assert [disk.path for disk in disks] == [str(tmp_path)]


def test_acceptance_loads_the_node_environment_before_settings(tmp_path: Path, monkeypatch) -> None:
    """Службам окружение подсовывает systemd, запущенному руками скрипту — никто.

    Без этого `Settings()` поднимался на умолчаниях, каталог данных становился
    относительным `data`, и приёмка честно сверяла эталон с пустотой. Первый же
    прогон на узле так и сделал.
    """
    from scripts.acceptance import load_env_file

    env = tmp_path / "catdoor.env"
    env.write_text(
        "# комментарий\n"
        "CATDOOR_DATA_DIR=/var/lib/catdoor\n"
        'CATDOOR_GATE_DETECTOR_DEVICE="GPU"\n'
        "\n"
        "CATDOOR_ALREADY_SET=из-файла\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CATDOOR_ALREADY_SET", "из-оболочки")
    monkeypatch.delenv("CATDOOR_DATA_DIR", raising=False)
    monkeypatch.delenv("CATDOOR_GATE_DETECTOR_DEVICE", raising=False)

    loaded = load_env_file(env)

    assert loaded == 2
    assert os.environ["CATDOOR_DATA_DIR"] == "/var/lib/catdoor"
    assert os.environ["CATDOOR_GATE_DETECTOR_DEVICE"] == "GPU"
    # Явный запуск с другим окружением должен побеждать файл.
    assert os.environ["CATDOOR_ALREADY_SET"] == "из-оболочки"


def test_acceptance_without_an_env_file_does_not_crash(tmp_path: Path) -> None:
    from scripts.acceptance import load_env_file

    assert load_env_file(tmp_path / "нет-такого") == 0


def test_an_unreadable_environment_is_a_refusal_not_an_absence(tmp_path: Path) -> None:
    """Файл есть, но не читается — это не «окружения нет».

    Молча приняв одно за другое, приёмка сверяла бы эталон с умолчаниями и
    печатала таблицу прочерков с одним выдуманным провалом. Так и вышло на узле
    2026-08-09: `alex` не читает root-овский /etc/catdoor/catdoor.env.
    """
    import pytest as _pytest

    from scripts.acceptance import EnvironmentUnreadable, load_env_file

    env = tmp_path / "catdoor.env"
    env.write_text("CATDOOR_DATA_DIR=/var/lib/catdoor\n", encoding="utf-8")
    env.chmod(0o000)
    try:
        with _pytest.raises(EnvironmentUnreadable, match="не читается"):
            load_env_file(env)
    finally:
        env.chmod(0o644)
