import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from src.recorder.ffmpeg import (
    ffmpeg_creation_flags,
    mjpeg_stream,
    record_clip,
)


def test_ffmpeg_creation_flags_hide_windows_console(
    monkeypatch,
) -> None:
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 123, raising=False)

    assert ffmpeg_creation_flags() == 123


def test_record_clip_hides_ffmpeg_window(runtime_path: Path, monkeypatch) -> None:
    destination = runtime_path / "clip.mp4"
    run_kwargs = {}

    def fake_run(command, **kwargs):
        run_kwargs.update(kwargs)
        Path(command[-1]).write_bytes(b"clip")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("src.recorder.ffmpeg.subprocess.run", fake_run)

    record_clip("ffmpeg", "rtsp://camera", destination, 5)

    assert destination.read_bytes() == b"clip"
    assert run_kwargs["creationflags"] == getattr(
        subprocess,
        "CREATE_NO_WINDOW",
        0,
    )


def test_mjpeg_stream_hides_ffmpeg_window(monkeypatch) -> None:
    process = Mock()
    process.stdout.read.side_effect = [b"\xff\xd8frame\xff\xd9", b""]
    popen_kwargs = {}

    def fake_popen(command, **kwargs):
        popen_kwargs.update(kwargs)
        return process

    monkeypatch.setattr("src.recorder.ffmpeg.subprocess.Popen", fake_popen)

    frames = list(mjpeg_stream("ffmpeg", "rtsp://camera", 3))

    assert len(frames) == 1
    assert popen_kwargs["creationflags"] == getattr(
        subprocess,
        "CREATE_NO_WINDOW",
        0,
    )
    process.terminate.assert_called_once()
