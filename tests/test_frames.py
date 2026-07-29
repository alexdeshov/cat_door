from unittest.mock import Mock

import numpy as np

from src.recorder.frames import FRAME_BYTES, rgb_frame_stream


def test_rgb_frame_stream_decodes_fixed_frame(monkeypatch) -> None:
    process = Mock()
    process.stdout.read.side_effect = [
        bytes([7]) * 100,
        bytes([7]) * (FRAME_BYTES - 100),
        b"",
    ]
    commands = []

    def fake_popen(command, **kwargs):
        commands.append(command)
        return process

    monkeypatch.setattr("src.recorder.frames.subprocess.Popen", fake_popen)

    frames = list(rgb_frame_stream("ffmpeg", "rtsp://camera", 3.0))

    assert len(frames) == 1
    assert frames[0].shape == (320, 320, 3)
    assert frames[0].dtype == np.uint8
    assert int(frames[0][0, 0, 0]) == 7
    assert "fps=3.0,scale=320:320,hflip,vflip" in commands[0]
    process.terminate.assert_called_once()


def test_rgb_frame_stream_can_write_ring_segments(runtime_path, monkeypatch) -> None:
    process = Mock()
    process.stdout.read.return_value = b""
    commands = []
    monkeypatch.setattr(
        "src.recorder.frames.subprocess.Popen",
        lambda command, **kwargs: commands.append(command) or process,
    )

    list(
        rgb_frame_stream(
            "ffmpeg",
            "rtsp://camera",
            3.0,
            segment_directory=runtime_path / "ring",
            segment_seconds=2,
            retention_seconds=180,
        )
    )

    assert "-f" in commands[0]
    assert "segment" in commands[0]
    assert "-segment_time" in commands[0]
    assert str(runtime_path / "ring" / "%Y%m%dT%H%M%S.mp4") in commands[0]
