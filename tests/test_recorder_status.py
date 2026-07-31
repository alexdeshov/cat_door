from pathlib import Path

from src.recorder.status import (
    RecentFrameCounters,
    read_recorder_status,
    write_recorder_status,
)


def test_status_write_retries_transient_windows_lock(
    runtime_path: Path,
    monkeypatch,
) -> None:
    status_path = runtime_path / "runtime" / "recorder-status.json"
    original_replace = Path.replace
    attempts = 0

    def flaky_replace(source: Path, target: Path) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("temporarily locked")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)

    write_recorder_status(status_path, state="idle")

    assert attempts == 3
    assert read_recorder_status(status_path)["state"] == "idle"


def test_recent_frame_counters_only_include_last_minute() -> None:
    counters = RecentFrameCounters(window_seconds=60.0)
    counters.record_processed("face", 10.0)
    counters.record_processed("face", 70.0)
    counters.record_skipped_no_motion("side", 69.0)

    snapshot = counters.snapshot(70.1)

    assert snapshot == {
        "frames_processed": {"face": 1, "side": 0},
        "frames_skipped_no_motion": {"face": 0, "side": 1},
    }
