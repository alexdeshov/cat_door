from __future__ import annotations

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
        },
        "side": {
            "confidence_threshold": 0.55,
            "box_min_confidence": 0.20,
            "confirmation_frames": 2,
            "window_frames": 5,
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
