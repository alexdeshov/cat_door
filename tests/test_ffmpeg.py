from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from src.recorder.ffmpeg import mjpeg_stream, record_clip


def test_record_clip_writes_destination(runtime_path: Path, monkeypatch) -> None:
    destination = runtime_path / "clip.mp4"
    run_kwargs = {}

    def fake_run(command, **kwargs):
        run_kwargs.update(kwargs)
        Path(command[-1]).write_bytes(b"clip")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("src.recorder.ffmpeg.subprocess.run", fake_run)

    record_clip("ffmpeg", "rtsp://camera", destination, 5)

    assert destination.read_bytes() == b"clip"
    assert "creationflags" not in run_kwargs


def test_mjpeg_stream_yields_complete_frames(monkeypatch) -> None:
    process = Mock()
    process.stdout.read.side_effect = [b"\xff\xd8frame\xff\xd9", b""]
    popen_kwargs = {}

    def fake_popen(command, **kwargs):
        popen_kwargs.update(kwargs)
        return process

    monkeypatch.setattr("src.recorder.ffmpeg.subprocess.Popen", fake_popen)

    frames = list(mjpeg_stream("ffmpeg", "rtsp://camera", 3))

    assert len(frames) == 1
    assert "creationflags" not in popen_kwargs
    process.terminate.assert_called_once()
