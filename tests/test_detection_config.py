from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.common.detection_config import (
    CameraDetectionConfig,
    DetectionConfig,
    detection_config_path,
    load_detection_config,
    save_detection_config,
)
from src.common.settings import Settings


def make_config(**overrides: object) -> DetectionConfig:
    values = {
        "face": {
            "confidence_threshold": 0.30,
            "box_min_confidence": 0.15,
            "confirmation_frames": 2,
            "window_frames": 5,
            "sample_fps": 8.0,
            "roi": None,
        },
        "side": {
            "confidence_threshold": 0.55,
            "box_min_confidence": 0.20,
            "confirmation_frames": 2,
            "window_frames": 5,
            "sample_fps": 3.0,
            "roi": None,
        },
        "candidate_timeout_seconds": 3.0,
        "absence_seconds": 3.0,
        "tail_seconds": 10.0,
        "cooldown_seconds": 5.0,
        **overrides,
    }
    return DetectionConfig.model_validate(values)


def test_detection_config_round_trip(runtime_path) -> None:
    path = runtime_path / "runtime" / "detection-settings.json"
    config = make_config()

    save_detection_config(path, config)

    assert load_detection_config(path, make_config()) == config


def test_invalid_detection_config_falls_back_to_defaults(runtime_path) -> None:
    path = runtime_path / "runtime" / "detection-settings.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"confirmation_frames": 99}', encoding="utf-8")
    defaults = make_config()

    assert load_detection_config(path, defaults) == defaults


def test_nested_legacy_config_gets_new_camera_defaults(runtime_path) -> None:
    path = runtime_path / "runtime" / "detection-settings.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        """
        {
          "face": {
            "confidence_threshold": 0.25,
            "box_min_confidence": 0.10,
            "confirmation_frames": 2,
            "window_frames": 5
          },
          "side": {
            "confidence_threshold": 0.50,
            "box_min_confidence": 0.20,
            "confirmation_frames": 2,
            "window_frames": 5
          },
          "candidate_timeout_seconds": 3,
          "absence_seconds": 3,
          "tail_seconds": 10,
          "cooldown_seconds": 5
        }
        """,
        encoding="utf-8",
    )

    loaded = load_detection_config(path, make_config())

    assert loaded.face.confidence_threshold == 0.25
    assert loaded.face.sample_fps == 8.0
    assert loaded.face.roi is None
    assert loaded.side.confidence_threshold == 0.50
    assert loaded.side.sample_fps == 3.0


def test_door_camera_starts_at_the_same_threshold_as_the_approach(runtime_path) -> None:
    """Дверце долго доставался порог мягче подхода — 0.30 против 0.55.

    Расчёт был на то, что камера смотрит в упор и кот занимает половину кадра.
    2026-08-09 владелец показал обратное: оставленная у двери посылка стабильно
    проходила как кот. Порог дверцы поднят до 0.50.
    """
    defaults = DetectionConfig.from_settings(Settings(data_dir=runtime_path))

    assert defaults.face.confidence_threshold == 0.50


def test_flat_legacy_config_caps_the_door_camera_at_its_own_default(runtime_path) -> None:
    """Плоский файл писался, когда порог был один на обе камеры.

    По смыслу это был порог подхода, и дверце он годится только как потолок:
    поднять её выше собственного умолчания он не вправе, опустить — вправе.
    """
    path = detection_config_path(runtime_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"confidence_threshold": 0.80}), encoding="utf-8")
    defaults = DetectionConfig.from_settings(Settings(data_dir=runtime_path))

    loaded = load_detection_config(path, defaults)

    assert loaded.face.confidence_threshold == defaults.face.confidence_threshold
    assert loaded.side.confidence_threshold == 0.80


def test_roi_outside_frame_falls_back_to_defaults(runtime_path) -> None:
    path = runtime_path / "runtime" / "detection-settings.json"
    config = make_config()
    payload = config.model_dump(mode="json")
    payload["face"]["roi"] = {"x": 0.8, "y": 0.0, "width": 0.4, "height": 1.0}
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_detection_config(path, config) == config


def test_sampling_ceiling_follows_the_hardware() -> None:
    """Потолок частоты ставился под детектор на процессоре.

    Там кадр стоил 26–41 мс, и одиннадцать кадров в секунду с двух камер были
    практическим пределом. На видеоядре кадр стоит 10 мс, и двадцать кадров
    занимают пятую часть его времени.
    """
    config = CameraDetectionConfig(
        confidence_threshold=0.3,
        box_min_confidence=0.15,
        confirmation_frames=2,
        window_frames=5,
        sample_fps=12.0,
    )

    assert config.sample_fps == 12.0

    with pytest.raises(ValidationError):
        CameraDetectionConfig(
            confidence_threshold=0.3,
            box_min_confidence=0.15,
            confirmation_frames=2,
            window_frames=5,
            sample_fps=25.0,
        )


def test_rejected_settings_leave_a_trace(runtime_path, caplog) -> None:
    """Молчаливый откат неотличим от применённой настройки.

    2026-08-08 веб-служба записала частоту выше потолка, который знал recorder:
    она была с новым кодом, он со старым. Recorder оставил прежние значения, и
    админка полчаса показывала двенадцать кадров там, где шло десять.
    """
    path = detection_config_path(runtime_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"face": {"sample_fps": 999.0}, "side": {}}), encoding="utf-8")
    defaults = DetectionConfig.from_settings(Settings(data_dir=runtime_path))

    with caplog.at_level("WARNING", logger="catdoor.detection-config"):
        config = load_detection_config(path, defaults)

    assert config.face.sample_fps == defaults.face.sample_fps
    assert "не приняты" in caplog.text


def test_missing_settings_are_not_worth_a_warning(runtime_path, caplog) -> None:
    """Файла нет — это первый запуск, а не беда."""
    defaults = DetectionConfig.from_settings(Settings(data_dir=runtime_path))

    with caplog.at_level("WARNING", logger="catdoor.detection-config"):
        assert load_detection_config(detection_config_path(runtime_path), defaults) == defaults

    assert caplog.text == ""
