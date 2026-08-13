from __future__ import annotations

import yaml

from src.common.camera_registry import CameraRecord, CameraRegistry
from src.common.circuits import NodeCircuits
from src.common.frigate_config import (
    _camera_block,
    build_config,
    build_env,
    foreign_camera_settings,
    foreign_sections,
    frigate_camera_name,
    frigate_cameras,
)


def camera(camera_id: str, circuit: str | None, **extra: object) -> CameraRecord:
    values: dict[str, object] = {
        "id": camera_id,
        "host": "192.168.1.50",
        "stream_path": "/Streaming/Channels/101",
        "username": "admin",
        "password": "secret",
        "circuit": circuit,
    }
    values.update(extra)
    return CameraRecord.model_validate(values)


def registry(*cameras: CameraRecord) -> CameraRegistry:
    return CameraRegistry(cameras=list(cameras))


def test_only_active_frigate_circuits_reach_the_config() -> None:
    """Выключенный регистратор не должен отнимать процессор у калитки."""
    inventory = registry(
        camera("gate-1", "gate"),
        camera("yard", "nvr"),
        camera("face", "cats", role="face"),
    )

    only_gate = frigate_cameras(inventory, NodeCircuits(cats=False, gate=True))
    both = frigate_cameras(inventory, NodeCircuits(cats=False, gate=True, nvr=True))

    assert [item.id for item in only_gate] == ["gate-1"]
    assert [item.id for item in both] == ["gate-1", "yard"]


def test_cat_cameras_never_reach_frigate() -> None:
    """Кошачий контур считает сам: отдать его камеры Frigate значило бы
    декодировать один поток дважды."""
    inventory = registry(camera("face", "cats", role="face"))

    assert frigate_cameras(inventory, NodeCircuits(cats=True)) == []


def test_disabled_and_addressless_cameras_are_skipped() -> None:
    inventory = registry(
        camera("off", "nvr", enabled=False),
        camera("blank", "nvr", host=""),
    )

    assert frigate_cameras(inventory, NodeCircuits(cats=False, nvr=True)) == []


def test_config_is_valid_yaml_with_a_camera_per_entry() -> None:
    inventory = registry(
        camera("gate-1", "gate", title="Калитка"),
        camera("yard-cam", "nvr", detect_stream_path="/Streaming/Channels/102"),
    )

    parsed = yaml.safe_load(build_config(inventory, NodeCircuits(cats=False, gate=True, nvr=True)))

    assert set(parsed["cameras"]) == {"gate_1", "yard_cam"}
    # У калитки два вопроса: кто пришёл и кто приехал. Машины нужны только
    # для отчёта, но пока их не было в списке, Frigate не создавал события
    # и не читал номер, сколько бы хорошо тот ни был виден.
    assert parsed["cameras"]["gate_1"]["objects"]["track"] == ["person", "car"]
    assert parsed["cameras"]["yard_cam"]["objects"]["track"] == ["person", "car"]


def test_a_substream_splits_detection_from_recording() -> None:
    """Декодирование основного потока дороже самого распознавания."""
    inventory = registry(camera("yard", "nvr", detect_stream_path="/Streaming/Channels/102"))

    parsed = yaml.safe_load(build_config(inventory, NodeCircuits(cats=False, nvr=True)))
    inputs = parsed["cameras"]["yard"]["ffmpeg"]["inputs"]

    assert [item["roles"] for item in inputs] == [["detect"], ["record"]]
    assert inputs[0]["path"].endswith("/Streaming/Channels/102")
    assert inputs[1]["path"].endswith("/Streaming/Channels/101")


def test_without_a_substream_one_input_serves_both_roles() -> None:
    inventory = registry(camera("yard", "nvr"))

    parsed = yaml.safe_load(build_config(inventory, NodeCircuits(cats=False, nvr=True)))
    inputs = parsed["cameras"]["yard"]["ffmpeg"]["inputs"]

    assert len(inputs) == 1
    assert inputs[0]["roles"] == ["detect", "record"]


def test_passwords_never_land_in_the_config() -> None:
    """Конфигурацию читают и сравнивают; пароли камер живут в env-файле."""
    inventory = registry(camera("yard", "nvr"))
    circuits = NodeCircuits(cats=False, nvr=True)

    config = build_config(inventory, circuits)
    env = build_env(frigate_cameras(inventory, circuits))

    assert "secret" not in config
    assert "192.168.1.50" not in config
    assert "{FRIGATE_YARD_PASSWORD}" in config
    assert "FRIGATE_YARD_PASSWORD=secret" in env
    assert "FRIGATE_YARD_HOST=192.168.1.50" in env


def test_a_node_without_frigate_cameras_still_gets_a_valid_config() -> None:
    """Пустой конфиг лучше отсутствующего: Frigate поднимется и скажет об этом
    сам, вместо того чтобы падать на разборе."""
    parsed = yaml.safe_load(build_config(registry(), NodeCircuits(cats=True)))

    assert parsed["cameras"] == {}
    assert parsed["detectors"]["openvino"]["device"] == "CPU"


def test_frigate_detector_device_is_a_setting_and_defaults_to_the_processor() -> None:
    """Устройство детектора Frigate раньше было зашито и стоило узлу пяти ядер.

    Умолчание остаётся прежним нарочно: очередь у видеоядра одна, и там уже
    считает детектор котов. Переключение должно быть решением с замером, а не
    побочным следствием обновления.
    """
    circuits = NodeCircuits(cats=True, gate=True)

    default = yaml.safe_load(build_config(registry(), circuits))
    moved = yaml.safe_load(build_config(registry(), circuits, detector_device="GPU"))

    assert default["detectors"]["openvino"]["device"] == "CPU"
    assert moved["detectors"]["openvino"]["device"] == "GPU"


def test_camera_names_are_made_acceptable_to_frigate() -> None:
    """Frigate не принимает дефисы, а инвентарь ими полон."""
    assert frigate_camera_name("nobelic-4mp") == "nobelic_4mp"
    assert frigate_camera_name("Yard.Cam 2") == "yard_cam_2"
    assert frigate_camera_name("---") == "camera"


def test_continuous_recording_follows_the_node_setting() -> None:
    inventory = registry(camera("yard", "nvr"))
    circuits = NodeCircuits(cats=False, nvr=True, gate_recording="continuous")

    parsed = yaml.safe_load(build_config(inventory, circuits))

    assert parsed["record"]["continuous"]["days"] == 1


def test_motion_recording_keeps_a_week_and_thus_keeps_previews() -> None:
    """Режим «по движению» держит сегменты неделю — и вместе с ними превью.

    Превью в интерфейсе Frigate живут по правилам записей: превью за час, не
    перекрытый ни одной сохранённой записью, уборщик сносит вместе со строкой в
    базе. Пока оба срока были нулевыми, разбор события отвечал «Предпросмотр не
    найден» почти всегда, и чинить это в самом Frigate было нечем.
    """
    inventory = registry(camera("yard", "nvr"))
    circuits = NodeCircuits(cats=False, nvr=True, gate_recording="motion")

    parsed = yaml.safe_load(build_config(inventory, circuits))

    assert parsed["record"]["motion"]["days"] == 7
    assert parsed["record"]["continuous"]["days"] == 0
    # Камера наследует тот же режим: Frigate берёт настройку записи с уровня
    # камеры, если она там есть, и разъехавшиеся уровни дали бы разное
    # поведение у разных камер одного контура.
    assert parsed["cameras"]["yard"]["record"]["motion"]["days"] == 7


def test_events_recording_keeps_nothing_beyond_events() -> None:
    inventory = registry(camera("yard", "nvr"))
    circuits = NodeCircuits(cats=False, nvr=True, gate_recording="events")

    parsed = yaml.safe_load(build_config(inventory, circuits))

    assert parsed["record"]["continuous"]["days"] == 0
    assert parsed["record"]["motion"]["days"] == 0


def test_foreign_sections_survive_a_rebuild() -> None:
    """Настройки, включённые в самом Frigate, переживают пересборку из инвентаря.

    Распознавание лиц и номера включаются в интерфейсе Frigate и попадают в тот
    же файл, который собираем мы. Пересборка «с нуля» стирала их молча, а
    замечалось это через неделю по отсутствию имён в событиях.
    """
    existing = """mqtt:
  enabled: false
cameras:
  old_camera:
    detect:
      fps: 5
# Лица включены вручную в интерфейсе Frigate.
face_recognition:
  enabled: true
  model_size: small
lpr:
  enabled: false
version: 0.17-0
"""

    preserved = foreign_sections(existing)

    assert "face_recognition:" in preserved
    assert "  model_size: small" in preserved
    assert "lpr:" in preserved
    assert "version: 0.17-0" in preserved
    # Комментарий человека едет вместе со своим разделом.
    assert "# Лица включены вручную" in preserved
    # Наши разделы не дублируются: их соберут заново.
    assert "mqtt:" not in preserved
    assert "old_camera" not in preserved


def test_rebuilt_config_keeps_foreign_sections_and_new_cameras() -> None:
    """Пересобранная конфигурация содержит и свежие камеры, и чужие разделы."""
    registry = CameraRegistry(
        cameras=[
            CameraRecord(
                id="gate-1",
                host="192.0.2.5",
                stream_path="/live",
                circuit="gate",
            )
        ]
    )
    preserved = foreign_sections("face_recognition:\n  enabled: true\n")

    config = build_config(registry, NodeCircuits(cats=False, gate=True), preserved=preserved)

    assert "gate_1:" in config
    assert config.rstrip().endswith("enabled: true")
    # Раздел остаётся единственным: собранная часть его не дублирует.
    assert config.count("face_recognition:") == 1


def test_detect_resolution_is_left_to_frigate() -> None:
    """Разрешение детекции не навязывается: его задаёт сам поток.

    Жёсткие 640×480 сплющивали субпоток 16:9 и, что важнее, уменьшали лицо до
    размера, который детектор лиц уже не находит: 95×112 точек в записи
    становились 32×50 в кадре детекции.
    """
    registry = CameraRegistry(
        cameras=[
            CameraRecord(
                id="gate-1",
                host="192.0.2.5",
                stream_path="/media/video1",
                detect_stream_path="/media/video2",
                circuit="gate",
            )
        ]
    )

    config = build_config(registry, NodeCircuits(cats=False, gate=True))
    parsed = yaml.safe_load(config)
    detect = parsed["cameras"]["gate_1"]["detect"]

    assert detect == {"enabled": True, "fps": 5}


def test_camera_named_with_a_digit_does_not_topple_frigate() -> None:
    """Ключ `3:` YAML считает числом, а Frigate ждёт строку — и бракует всё.

    2026-08-08 камера, названная цифрой, подняла Frigate в safe mode вообще без
    камер: легли разом и регистратор, и калитка. Цена ошибки несоразмерна
    причине, поэтому имя уходит в кавычках.
    """
    camera = CameraRecord(
        id="3",
        host="192.168.1.213",
        username="admin",
        password="secret",
        stream_path="/Streaming/Channels/101",
        circuit="nvr",
    )

    inventory = CameraRegistry(cameras=[camera])

    parsed = yaml.safe_load(build_config(inventory, NodeCircuits(cats=False, nvr=True)))

    # Ключ обязан остаться строкой: именно на этом Frigate забраковал всё.
    assert list(parsed["cameras"]) == ["3"]
    assert _camera_block(camera, 0, 7).startswith('  "3":')


def test_ordinary_names_stay_without_quotes() -> None:
    camera = CameraRecord(
        id="ursa-minor",
        host="192.168.1.213",
        username="admin",
        password="secret",
        stream_path="/Streaming/Channels/101",
        circuit="nvr",
    )

    assert _camera_block(camera, 0, 7).startswith("  ursa_minor:")


EXISTING_WITH_MASK = """\
mqtt:
  enabled: false

cameras:
  test_cam:
    ffmpeg:
      inputs:
        - path: rtsp://old
          roles:
            - detect
    detect:
      enabled: true
      fps: 5
    # кусты справа: качаются от ветра, интереса не представляют
    motion:
      contour_area: 40
      mask:
        - 0.7,0,1,0,1,1,0.7,1
    zones:
      gate:
        coordinates: 0,0,0.5,0,0.5,1,0,1
  "3":
    detect:
      enabled: true
    motion:
      threshold: 40

face_recognition:
  enabled: true
"""


def test_hand_drawn_camera_settings_survive_a_rebuild() -> None:
    """Маску рисуют в интерфейсе Frigate, а конфигурацию пересобираем мы.

    До 2026-08-09 раздел `cameras` пересобирался целиком, и нарисованная маска
    жила до первой правки камеры в админке. Узнать об этом можно было только по
    вернувшимся ложным срабатываниям.
    """
    kept = foreign_camera_settings(EXISTING_WITH_MASK)

    assert set(kept) == {"test_cam", "3"}
    text = "\n".join(kept["test_cam"])
    assert "motion:" in text
    assert "contour_area: 40" in text
    assert "0.7,0,1,0,1,1,0.7,1" in text
    assert "zones:" in text
    assert "качаются от ветра" in text, "комментарий человека принадлежит его настройке"
    # Наше собственное мы не переносим: оно будет собрано заново.
    assert "ffmpeg:" not in text
    assert "fps: 5" not in text


def test_a_numeric_camera_name_keeps_its_settings() -> None:
    """Камеру, названную цифрой, YAML отдаёт в кавычках — ключ надо очистить."""
    kept = foreign_camera_settings(EXISTING_WITH_MASK)

    assert "threshold: 40" in "\n".join(kept["3"])


def test_rebuilt_config_carries_the_mask_into_the_camera() -> None:
    inventory = registry(camera("test_cam", "gate"))
    circuits = NodeCircuits(cats=True, gate=True)

    config = build_config(
        inventory,
        circuits,
        preserved_cameras=foreign_camera_settings(EXISTING_WITH_MASK),
    )
    parsed = yaml.safe_load(config)

    built = parsed["cameras"]["test_cam"]
    assert built["motion"]["contour_area"] == 40
    assert built["motion"]["mask"] == ["0.7,0,1,0,1,1,0.7,1"]
    assert built["zones"]["gate"]["coordinates"] == "0,0,0.5,0,0.5,1,0,1"
    # Наше при этом на месте и не потеряно.
    assert built["detect"]["fps"] == 5
    assert "ffmpeg" in built


def test_a_camera_without_hand_settings_is_unchanged() -> None:
    inventory = registry(camera("test_cam", "gate"))
    circuits = NodeCircuits(cats=True, gate=True)

    without = build_config(inventory, circuits)
    with_empty = build_config(inventory, circuits, preserved_cameras={})

    assert without == with_empty


def test_a_camera_can_be_forced_to_software_decoding() -> None:
    """У nvr-2 VA-API падал на её потоке примерно раз в минуту.

    «Failed to sync surface», watchdog Frigate бесконечно перезапускал ffmpeg —
    детекция теряла поток на каждом круге. Смена устройства детектора замером
    исключена: дело в самом потоке. Пустая `hwaccel_args` выключает аппаратный
    путь одной камере, не трогая остальных.
    """
    inventory = registry(
        camera("nvr-2", "nvr"),
        camera("cam-3", "nvr"),
        camera("yard", "nvr"),
    )
    inventory.cameras[0].decode = "software"
    inventory.cameras[1].decode = "qsv"

    parsed = yaml.safe_load(
        build_config(inventory, NodeCircuits(cats=False, nvr=True))
    )

    assert parsed["cameras"]["nvr_2"]["ffmpeg"]["hwaccel_args"] == ""
    assert parsed["cameras"]["cam_3"]["ffmpeg"]["hwaccel_args"] == "preset-intel-qsv-h264"
    assert "hwaccel_args" not in parsed["cameras"]["yard"]["ffmpeg"]


def test_mqtt_stays_off_until_a_broker_exists() -> None:
    """Включённый MQTT без брокера сыпал бы ошибками подключения в лог."""
    parsed = yaml.safe_load(build_config(registry(), NodeCircuits(gate=True)))

    assert parsed["mqtt"]["enabled"] is False
    assert "host" not in parsed["mqtt"]


def test_the_broker_address_is_the_one_frigate_can_reach() -> None:
    """Frigate живёт в контейнере, и `127.0.0.1` там — это он сам.

    Брокер поднимается соседним сервисом того же compose, и достучаться до него
    можно по имени сервиса. Адрес узла (`127.0.0.1`) в конфигурации Frigate
    означал бы «подключайся к самому себе» — и не подключился бы никто.
    """
    parsed = yaml.safe_load(
        build_config(registry(), NodeCircuits(gate=True), mqtt=("mqtt", 1883))
    )

    assert parsed["mqtt"]["enabled"] is True
    assert parsed["mqtt"]["host"] == "mqtt"
    assert parsed["mqtt"]["port"] == 1883
