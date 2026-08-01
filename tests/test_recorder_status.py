from pathlib import Path

from src.recorder.status import read_recorder_status, write_recorder_status


def test_status_write_round_trips_and_stamps_time(runtime_path: Path) -> None:
    status_path = runtime_path / "runtime" / "recorder-status.json"

    write_recorder_status(status_path, state="idle", frames_processed=0)

    payload = read_recorder_status(status_path)
    assert payload is not None
    assert payload["state"] == "idle"
    assert payload["frames_processed"] == 0
    assert isinstance(payload["updated_at"], str)


def test_status_write_replaces_previous_and_leaves_no_temporaries(
    runtime_path: Path,
) -> None:
    status_path = runtime_path / "runtime" / "recorder-status.json"

    write_recorder_status(status_path, state="idle")
    write_recorder_status(status_path, state="active")

    payload = read_recorder_status(status_path)
    assert payload is not None
    assert payload["state"] == "active"
    assert list(status_path.parent.glob(".*.tmp")) == []


def test_reading_a_missing_status_returns_none(runtime_path: Path) -> None:
    assert read_recorder_status(runtime_path / "runtime" / "absent.json") is None
