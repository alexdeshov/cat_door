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
    "detector_model": "yolo26n-416.onnx",
    "hwaccel_actual": "vaapi",
    "frames_per_second": {"face": 12.1, "side": 6.0},
    "inference_ms": 13.5,
}

REFERENCE_CONFIG = {
    "face": {"confidence_threshold": 0.65, "sample_fps": 12.0},
    "side": {"confidence_threshold": 0.65, "sample_fps": 6.0},
}

REFERENCE_STATS = {
    "detectors": {"openvino": {"inference_speed": 4.78}},
    "cameras": {"test_cam": {"skipped_fps": 0.0, "camera_fps": 5.0}},
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
        # Эталон включает прошедшую копию: узел, чья копия ни разу не
        # отчитывалась, приёмку проходить не должен — разметку восстановить
        # нечем, она сделана руками.
        "backup_status": {"finished_at": 1000.0, "ok": True, "detail": "копия прошла",
                          "free_gb": 79.0},
        # Служба калитки на связи и слышит брокер: слепота по створке с
        # 2026-08-21 — провал приёмки, а не невидимая деталь.
        "gate_service": {"updated_at": 1000.0 + 3600 - 1, "broker_connected": True},
        "now": 1000.0 + 3600,
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


def test_the_same_loss_means_different_things_at_different_rates() -> None:
    """Пропуск меряется долей кадров, а не их числом.

    Историческая беда — 0.3 к/с на камере в 5 к/с: каждый шестнадцатый кадр мимо
    детектора, и это тревога. Ровно те же 0.3 на камере в 10 к/с — вдвое меньшая
    доля, и будить телефон незачем.
    """
    slow = {**REFERENCE_STATS, "cameras": {"test_cam": {"camera_fps": 5.0, "skipped_fps": 0.3}}}
    fast = {**REFERENCE_STATS, "cameras": {"test_cam": {"camera_fps": 10.0, "skipped_fps": 0.3}}}

    assert find(build(frigate_stats=slow), "gate_skipped_test_cam").status == "warn"
    assert find(build(frigate_stats=fast), "gate_skipped_test_cam").status == "ok"


def test_a_single_lost_frame_does_not_wake_the_phone() -> None:
    """2026-08-26: калитка поехала на 10 к/с, и абсолютный порог начал ловить
    единичный кадр.

    Frigate округляет частоты до одного знака, поэтому 0.1 — наименьшее
    ненулевое показание вообще: порог «0.1 к/с» означал «не потеряно ни кадра».
    Дважды за день короткая заминка стоила одного-двух кадров из десяти, и
    сторож слал беду, хотя камера отдавала детектору вдвое больше кадров, чем
    накануне.
    """
    stats = {**REFERENCE_STATS, "cameras": {"test_cam": {"camera_fps": 10.1, "skipped_fps": 0.2}}}

    assert find(build(frigate_stats=stats), "gate_skipped_test_cam").status == "ok"


def test_losing_a_fifth_of_the_frames_is_a_failure() -> None:
    """Потеря пятой части кадров — это уже не заминка, а неуспевающий контур."""
    stats = {**REFERENCE_STATS, "cameras": {"test_cam": {"camera_fps": 10.0, "skipped_fps": 2.5}}}

    assert find(build(frigate_stats=stats), "gate_skipped_test_cam").status == "fail"


def test_a_dead_camera_does_not_divide_by_its_own_zero() -> None:
    """У мёртвой камеры частота нулевая, и доля от неё — вечная авария.

    2026-08-21 обрыв питания уже показывал, что «0 пропущено» у камеры без
    кадров выглядит здоровьем; отвечает за это отдельная проверка «Кадры идут»,
    а здесь ноль не должен превращаться в ложную беду.
    """
    stats = {**REFERENCE_STATS, "cameras": {"test_cam": {"camera_fps": 0.0, "skipped_fps": 0.0}}}
    groups = build(frigate_stats=stats)

    assert find(groups, "gate_alive_test_cam").status == "fail"
    assert find(groups, "gate_skipped_test_cam").status == "ok"


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


def _disk(path: str, total_gb: float, free_gb: float) -> object:
    from src.common.node_metrics import DiskUsage

    return DiskUsage(path=path, total_gb=total_gb, free_gb=free_gb)


def test_a_full_frigate_volume_is_not_an_alarm() -> None:
    """Frigate не «заполняет диск», а держит его полным.

    Проверено на узле 13 августа при 97% занятости: 6 августа удалено целиком,
    5 августа стёрто до 150 МБ, 7 августа съедено со 130 до 23 ГБ. Уборка идёт,
    и полнота здесь — рабочее состояние.

    Общая мерка в 85% дала бы вечное жёлтое, а постоянное предупреждение хуже
    отсутствующего: оно приучает не смотреть на страницу состояния.
    """
    from src.common.health import _disk_check

    check = _disk_check(_disk("/var/lib/frigate/media/recordings", 916.0, 31.1))

    assert check.status == "ok"
    assert "так и задумано" in check.expected


def test_a_frigate_volume_without_free_space_is_an_alarm() -> None:
    """Смотреть надо не на занятость, а на то, успевает ли уборка."""
    from src.common.health import _disk_check

    tight = _disk_check(_disk("/var/lib/frigate/media", 916.0, 4.0))
    empty = _disk_check(_disk("/var/lib/frigate/media", 916.0, 1.0))

    assert tight.status == "warn"
    assert empty.status == "fail"
    assert "не успевает" in tight.hint


def test_an_ordinary_volume_is_still_measured_the_ordinary_way() -> None:
    """Послабление касается только тома с собственной уборкой."""
    from src.common.health import _disk_check

    check = _disk_check(_disk("/var/lib/catdoor", 360.0, 20.0))

    assert check.status == "warn"
    assert "занято до" in check.expected


def test_a_backup_that_failed_is_said_out_loud() -> None:
    """Копия ломается молча: таймер отработал, скрипт упал, никто не заметил.

    Восстановить разметку нечем — она сделана руками за недели.
    """
    groups = build(
        backup_status={"finished_at": 1000.0, "ok": False, "detail": "диск полон", "free_gb": 1.0},
        now=1000.0 + 60,
    )

    assert find(groups, "backup_last_run").status == "fail"
    assert "не прошла" in find(groups, "backup_last_run").actual
    assert "диск полон" in find(groups, "backup_last_run").actual


def test_a_backup_that_stopped_running_is_noticed() -> None:
    """Остановленный таймер выглядит как тишина, а не как поломка."""
    groups = build(
        backup_status={"finished_at": 1000.0, "ok": True, "detail": "прошла", "free_gb": 79.0},
        now=1000.0 + 40 * 3600,
    )

    assert find(groups, "backup_last_run").status == "warn"
    assert "40 ч назад" in find(groups, "backup_last_run").actual


def test_space_for_the_backup_is_counted_in_gigabytes() -> None:
    """В процентах мерить нельзя: том заполнен на 97% всегда.

    Frigate держит архивный диск полным по устройству, и копии нужны не
    проценты, а гигабайты.
    """
    groups = build(
        backup_status={"finished_at": 1000.0, "ok": True, "detail": "прошла", "free_gb": 8.0},
        now=1000.0 + 60,
    )

    assert find(groups, "backup_free_space").status == "warn"
    assert find(groups, "backup_last_run").status == "ok"


def test_a_backup_that_never_reported_is_unknown_not_fine() -> None:
    """«Отметок нет» — это и первый запуск, и остановленный таймер.

    Молча считать это благополучием нельзя, объявлять поломкой — тоже.
    """
    groups = build(backup_status=None)

    assert find(groups, "backup_last_run").status == "unknown"
