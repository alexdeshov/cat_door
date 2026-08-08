import os
import time
from pathlib import Path

from src.recorder.cleanup import ABANDONED_AFTER_SECONDS, sweep_incomplete_files


def _aged(path: Path, seconds: float, content: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    stale = time.time() - seconds
    import os

    os.utime(path, (stale, stale))
    return path


def test_leftovers_of_a_killed_process_are_removed(runtime_path: Path) -> None:
    """Отключение питания не исполняет ни одного блока `finally`."""
    old = ABANDONED_AFTER_SECONDS + 60
    temporary = _aged(runtime_path / "runtime" / ".cameras.json.abc123.tmp", old)
    part = _aged(runtime_path / "events" / "00000007" / "face.part.mp4", old)
    concat = _aged(runtime_path / "events" / "00000007" / "face.concat.txt", old)
    empty = _aged(runtime_path / "runtime" / "ring" / "face" / "20260802T091500.mp4", 5, b"")
    keep = _aged(runtime_path / "runtime" / "ring" / "face" / "20260802T091502.mp4", 5, b"video")

    removed = sweep_incomplete_files(runtime_path)

    assert removed == 4
    assert not temporary.exists()
    assert not part.exists()
    assert not concat.exists()
    assert not empty.exists()
    assert keep.exists()


def test_a_write_in_progress_is_left_alone(runtime_path: Path) -> None:
    """Веб-процесс может писать настройки прямо сейчас — его файл не трогаем."""
    fresh = _aged(runtime_path / "runtime" / ".cameras.json.def456.tmp", 1)

    assert sweep_incomplete_files(runtime_path) == 0
    assert fresh.exists()


def test_empty_data_directory_is_not_an_error(runtime_path: Path) -> None:
    assert sweep_incomplete_files(runtime_path / "нет-такого") == 0


def test_notification_snapshots_do_not_survive_a_power_cut(runtime_path: Path) -> None:
    """Снимок удаляется в `finally`, а он при пропаже питания не исполняется.

    Каталог отдельный именно поэтому: всё, что в нём осталось, заведомо
    брошено — сообщение либо ушло, либо уже никогда не уйдёт.
    """
    snapshots = runtime_path / "runtime" / "notifications"
    snapshots.mkdir(parents=True)
    abandoned = snapshots / "pass-7.jpg"
    abandoned.write_bytes(b"\xff\xd8jpeg\xff\xd9")
    fresh = snapshots / "gate-1786171119.jpg"
    fresh.write_bytes(b"\xff\xd8jpeg\xff\xd9")
    old = time.time() - 3600
    os.utime(abandoned, (old, old))

    removed = sweep_incomplete_files(runtime_path)

    assert removed == 1
    assert not abandoned.exists()
    # Свежий снимок мог писаться прямо сейчас — его не трогаем.
    assert fresh.exists()
