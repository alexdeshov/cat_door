import json
from pathlib import Path

import pytest

from src.common.settings import Settings
from src.recorder.app import build_automatic_config, publish_heartbeat, recorder_status_path


def test_disabled_recorder_writes_safe_heartbeat(runtime_path: Path) -> None:
    settings = Settings(
        data_dir=runtime_path,
        automatic_capture_enabled=False,
        automatic_capture_dry_run=True,
    )

    publish_heartbeat(settings)

    payload = json.loads(recorder_status_path(settings).read_text(encoding="utf-8"))
    assert payload["process"] == "CatDoorRecorder"
    assert payload["state"] == "disabled"
    assert payload["dry_run"] is True
    assert payload["detector_connected"] is False
    assert "rtsp" not in json.dumps(payload).lower()


def test_enabled_recorder_waits_for_local_detector(runtime_path: Path) -> None:
    settings = Settings(data_dir=runtime_path, automatic_capture_enabled=True)

    publish_heartbeat(settings)

    payload = json.loads(recorder_status_path(settings).read_text(encoding="utf-8"))
    assert payload["state"] == "waiting_for_detector"


def test_invalid_detection_window_is_rejected() -> None:
    settings = Settings(
        detection_confirmation_frames=4,
        detection_window_frames=3,
    )

    with pytest.raises(ValueError):
        build_automatic_config(settings)
