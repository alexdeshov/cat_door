from __future__ import annotations

import json

from src.common.detection_config import (
    DetectionConfig,
    load_detection_config,
    save_detection_config,
)


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
    assert loaded.face.motion_gating_enabled is False
    assert loaded.face.motion_delta == 12.0
    assert loaded.face.motion_min_area_fraction == 0.005
    assert loaded.face.motion_alpha == 0.1
    assert loaded.side.confidence_threshold == 0.50
    assert loaded.side.sample_fps == 3.0
    assert loaded.side.motion_gating_enabled is False


def test_roi_outside_frame_falls_back_to_defaults(runtime_path) -> None:
    path = runtime_path / "runtime" / "detection-settings.json"
    config = make_config()
    payload = config.model_dump(mode="json")
    payload["face"]["roi"] = {"x": 0.8, "y": 0.0, "width": 0.4, "height": 1.0}
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_detection_config(path, config) == config
