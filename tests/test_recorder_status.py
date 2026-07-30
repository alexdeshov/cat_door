from pathlib import Path

from src.recorder.status import read_recorder_status, write_recorder_status


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
