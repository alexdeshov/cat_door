import io
import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.recorder.ffmpeg import RecorderError, mask_credentials, mjpeg_stream, record_clip


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
    process.stderr = io.BytesIO(b"")
    popen_kwargs = {}

    def fake_popen(command, **kwargs):
        popen_kwargs.update(kwargs)
        return process

    monkeypatch.setattr("src.recorder.ffmpeg.subprocess.Popen", fake_popen)

    frames = list(mjpeg_stream("ffmpeg", "rtsp://camera", 3))

    assert len(frames) == 1
    assert "creationflags" not in popen_kwargs
    process.terminate.assert_called_once()


def test_mask_credentials_removes_the_camera_password() -> None:
    message = "Error opening input file rtsp://admin:s3cret@192.168.1.64:554/Streaming"

    assert mask_credentials(message) == (
        "Error opening input file rtsp://***@192.168.1.64:554/Streaming"
    )


def test_mask_credentials_leaves_a_url_without_credentials_alone() -> None:
    assert mask_credentials("rtsp://192.168.1.64:554/Streaming") == (
        "rtsp://192.168.1.64:554/Streaming"
    )


def test_empty_stream_reports_why_without_leaking_the_password(
    monkeypatch,
    caplog,
) -> None:
    """A silent failure looks exactly like a working camera showing black."""
    process = Mock()
    process.stdout.read.side_effect = [b""]
    process.stderr = io.BytesIO(
        b"[rtsp @ 0x1] method DESCRIBE failed: 401 Unauthorized\n"
        b"Error opening input file rtsp://admin:s3cret@192.168.1.64:554/x.\n"
    )
    monkeypatch.setattr("src.recorder.ffmpeg.subprocess.Popen", lambda command, **kw: process)

    with caplog.at_level(logging.ERROR, logger="catdoor.ffmpeg"):
        assert list(mjpeg_stream("ffmpeg", "rtsp://admin:s3cret@camera/x", 3)) == []

    reported = caplog.text
    assert "401 Unauthorized" in reported
    assert "s3cret" not in reported
    assert "rtsp://***@192.168.1.64" in reported


def test_failed_recording_reports_why_without_leaking_the_password(
    runtime_path: Path,
    monkeypatch,
    caplog,
) -> None:
    def fake_run(command, **kwargs):
        return SimpleNamespace(
            returncode=1,
            stderr="Error opening input file rtsp://admin:s3cret@192.168.1.64:554/x.",
        )

    monkeypatch.setattr("src.recorder.ffmpeg.subprocess.run", fake_run)

    with (
        caplog.at_level(logging.ERROR, logger="catdoor.ffmpeg"),
        pytest.raises(RecorderError, match="rtsp_recording_failed"),
    ):
        record_clip("ffmpeg", "rtsp://admin:s3cret@camera/x", runtime_path / "c.mp4", 5)

    assert "s3cret" not in caplog.text
    assert "rtsp://***@192.168.1.64" in caplog.text


def test_preview_stream_gives_up_on_a_camera_that_stopped_sending(monkeypatch) -> None:
    """Забытая вкладка с мёртвой камерой не должна держать ffmpeg вечно.

    Чтение блокирующее, поэтому молчание замечает отдельный сторож; здесь
    проверяется, что он действительно прерывает процесс.
    """
    terminated = threading.Event()
    process = Mock()
    process.stderr = io.BytesIO(b"")
    process.poll.return_value = None
    process.terminate.side_effect = lambda: terminated.set()

    def read_until_terminated(_size: int) -> bytes:
        # Камера приняла соединение и замолчала: чтение висит, пока процесс не
        # прервут снаружи.
        terminated.wait(timeout=5)
        return b""

    process.stdout.read.side_effect = read_until_terminated
    monkeypatch.setattr("src.recorder.ffmpeg.subprocess.Popen", lambda command, **kwargs: process)

    frames = list(mjpeg_stream("ffmpeg", "rtsp://camera", 3, idle_timeout_seconds=5.0))

    assert frames == []
    assert terminated.is_set()


def test_preview_stream_is_not_interrupted_while_frames_arrive(monkeypatch) -> None:
    process = Mock()
    process.stdout.read.side_effect = [b"\xff\xd8frame\xff\xd9", b""]
    process.stderr = io.BytesIO(b"")
    process.poll.return_value = 0
    monkeypatch.setattr("src.recorder.ffmpeg.subprocess.Popen", lambda command, **kwargs: process)

    frames = list(mjpeg_stream("ffmpeg", "rtsp://camera", 3, idle_timeout_seconds=30.0))

    assert len(frames) == 1
