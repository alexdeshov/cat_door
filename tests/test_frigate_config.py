from __future__ import annotations

import yaml

from src.common.camera_registry import CameraRecord, CameraRegistry
from src.common.circuits import NodeCircuits
from src.common.frigate_config import (
    build_config,
    build_env,
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
    # У калитки один вопрос — кто пришёл; регистратору участка нужны и машины.
    assert parsed["cameras"]["gate_1"]["objects"]["track"] == ["person"]
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
