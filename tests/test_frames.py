import subprocess
from unittest.mock import Mock

import numpy as np

from src.common.detection_config import DetectorRoi
from src.inference.preprocessing import prepare_detector_frame
from src.recorder.frames import FRAME_BYTES, FRAME_HEIGHT, FRAME_WIDTH, rgb_frame_stream


def test_rgb_frame_stream_decodes_fixed_frame(monkeypatch) -> None:
    process = Mock()
    process.stdout.read.side_effect = [
        bytes([7]) * 100,
        bytes([7]) * (FRAME_BYTES - 100),
        b"",
    ]
    commands = []
    popen_kwargs = []

    def fake_popen(command, **kwargs):
        commands.append(command)
        popen_kwargs.append(kwargs)
        return process

    monkeypatch.setattr("src.recorder.frames.subprocess.Popen", fake_popen)

    frames = list(rgb_frame_stream("ffmpeg", "rtsp://camera", 3.0))

    assert len(frames) == 1
    assert frames[0].preview.shape == (360, 640, 3)
    assert frames[0].detector.shape == (360, 640, 3)
    assert frames[0].preview.dtype == np.uint8
    assert int(frames[0].preview[0, 0, 0]) == 7
    assert (
        "fps=3.0,scale=640:360:force_original_aspect_ratio=decrease,"
        "pad=640:360:(ow-iw)/2:(oh-ih)/2,hflip,vflip"
    ) in commands[0]
    assert popen_kwargs[0]["creationflags"] == getattr(
        subprocess,
        "CREATE_NO_WINDOW",
        0,
    )
    process.terminate.assert_called_once()


def test_detector_frame_remains_square_without_distortion() -> None:
    preview = np.full((360, 640, 3), 100, dtype=np.uint8)

    detector = prepare_detector_frame(preview)

    assert detector.shape == (320, 320, 3)
    assert np.all(detector[:70] == 0)
    assert np.all(detector[70:250] == 100)
    assert np.all(detector[250:] == 0)


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


def test_roi_is_cropped_before_detector_scaling_and_keeps_full_preview(
    runtime_path,
    monkeypatch,
) -> None:
    preview = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 7, dtype=np.uint8)
    detector = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 9, dtype=np.uint8)
    combined = np.hstack((preview, detector))
    process = Mock()
    process.stdout.read.side_effect = [combined.tobytes(), b""]
    commands = []
    monkeypatch.setattr(
        "src.recorder.frames.subprocess.Popen",
        lambda command, **kwargs: commands.append(command) or process,
    )
    roi = DetectorRoi(x=0.2, y=0.1, width=0.5, height=0.7)

    frames = list(
        rgb_frame_stream(
            "ffmpeg",
            "rtsp://camera",
            8.0,
            segment_directory=runtime_path / "ring",
            roi=roi,
            rotate_180=False,
        )
    )

    assert len(frames) == 1
    assert np.all(frames[0].preview == 7)
    assert np.all(frames[0].detector == 9)
    filter_complex = commands[0][commands[0].index("-filter_complex") + 1]
    assert "hflip" not in filter_complex
    assert "vflip" not in filter_complex
    assert (
        "[detector_source]crop=iw*0.5:ih*0.7:iw*0.2:ih*0.1,"
        "scale=640:360"
    ) in filter_complex
    assert commands[0].index("copy") < commands[0].index("-filter_complex")


def test_rgb_frame_stream_can_keep_native_orientation(monkeypatch) -> None:
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
            rotate_180=False,
        )
    )

    video_filter = commands[0][commands[0].index("-vf") + 1]
    assert "hflip" not in video_filter
    assert "vflip" not in video_filter
