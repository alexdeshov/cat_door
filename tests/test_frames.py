import io
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from src.common.detection_config import DetectorRoi
from src.inference.preprocessing import prepare_detector_frame
from src.recorder.frames import (
    FRAME_BYTES,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    active_hwaccel,
    reset_hwaccel_state,
    rgb_frame_stream,
)

RENDER_NODE = Path("/dev/dri/renderD128")


@pytest.fixture(autouse=True)
def _forget_hwaccel_verdict() -> None:
    """Защёлка отказа живёт на весь процесс, а тесты идут в одном."""
    reset_hwaccel_state()


def _hardware_process(reads: list[bytes] | None = None) -> Mock:
    process = Mock()
    if reads:
        process.stdout.read.side_effect = reads
    else:
        # Молчащий поток: чтение отвечает пустотой сколько угодно раз — процесс
        # может быть переиспользован и на программной попытке.
        process.stdout.read.return_value = b""
    # На аппаратной попытке жалобы ffmpeg читает отдельный поток, поэтому
    # канал ошибок обязан быть настоящим итератором строк.
    process.stderr = io.BytesIO(b"")
    return process


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
    assert frames[0].preview.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
    assert frames[0].detector.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
    assert frames[0].preview.dtype == np.uint8
    assert int(frames[0].preview[0, 0, 0]) == 7
    assert (
        f"fps=3.0,scale={FRAME_WIDTH}:{FRAME_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={FRAME_WIDTH}:{FRAME_HEIGHT}:(ow-iw)/2:(oh-ih)/2,hflip,vflip"
    ) in commands[0]
    assert "creationflags" not in popen_kwargs[0]
    process.terminate.assert_called_once()


def test_detector_frame_remains_square_without_distortion() -> None:
    preview = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 100, dtype=np.uint8)

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
        "scale=960:540"
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


def test_stream_hands_out_its_process_so_a_stall_can_be_broken(monkeypatch) -> None:
    """Чтение кадра блокирующее: разбудить застрявший поток можно только извне.

    Поэтому сам процесс отдаётся наружу сразу после запуска, а не после первого
    успешного кадра — камера, принявшая соединение и замолчавшая, кадра не даст
    никогда.
    """
    process = Mock()
    process.stdout.read.side_effect = [b""]
    handed: list[object] = []
    monkeypatch.setattr("src.recorder.frames.subprocess.Popen", lambda command, **kwargs: process)

    list(rgb_frame_stream("ffmpeg", "rtsp://camera", 3.0, on_process=handed.append))

    assert handed == [process]


def test_hardware_decoding_scales_on_the_media_engine(monkeypatch) -> None:
    """Кадр уменьшается видеоядром, в память забирается уже 960×540."""
    process = _hardware_process([bytes([7]) * FRAME_BYTES, b""])
    commands = []
    monkeypatch.setattr(
        "src.recorder.frames.subprocess.Popen",
        lambda command, **kwargs: commands.append(command) or process,
    )

    frames = list(
        rgb_frame_stream("ffmpeg", "rtsp://camera", 3.0, vaapi_device=RENDER_NODE)
    )

    assert len(frames) == 1
    command = commands[0]
    # Флаги ускорения обязаны стоять до входа: после -i ffmpeg их не применит.
    assert command.index("-hwaccel") < command.index("-i")
    assert command[command.index("-hwaccel_device") + 1] == str(RENDER_NODE)
    assert command[command.index("-hwaccel_output_format") + 1] == "vaapi"
    assert command[command.index("-vf") + 1] == (
        "fps=3.0,"
        f"scale_vaapi={FRAME_WIDTH}:{FRAME_HEIGHT}:"
        "force_original_aspect_ratio=decrease:force_divisible_by=2,"
        "hwdownload,format=nv12,"
        f"pad={FRAME_WIDTH}:{FRAME_HEIGHT}:(ow-iw)/2:(oh-ih)/2,hflip,vflip"
    )
    assert active_hwaccel() == "vaapi"


def test_ring_segments_are_still_copied_under_hardware_decoding(
    runtime_path,
    monkeypatch,
) -> None:
    """Запись прохода идёт копированием: декодирование её не касается."""
    process = _hardware_process()
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
            vaapi_device=RENDER_NODE,
        )
    )

    assert commands[0][commands[0].index("-c:v") + 1] == "copy"


def test_software_path_stays_untouched_without_a_device(monkeypatch) -> None:
    """Значение по умолчанию не должно менять команду ни на байт."""
    process = Mock()
    process.stdout.read.return_value = b""
    commands = []
    monkeypatch.setattr(
        "src.recorder.frames.subprocess.Popen",
        lambda command, **kwargs: commands.append(command) or process,
    )

    list(rgb_frame_stream("ffmpeg", "rtsp://camera", 3.0))

    assert "-hwaccel" not in commands[0]
    assert "scale_vaapi" not in commands[0][commands[0].index("-vf") + 1]
    assert active_hwaccel() is None


def test_roi_camera_keeps_software_decoding(monkeypatch) -> None:
    """Обрезку видеоядро не умеет, а скачивать кадр 4 Мп ради неё незачем."""
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
            roi=DetectorRoi(x=0.2, y=0.1, width=0.5, height=0.7),
            vaapi_device=RENDER_NODE,
        )
    )

    assert "-hwaccel" not in commands[0]
    assert "crop=iw*0.5" in commands[0][commands[0].index("-filter_complex") + 1]


def test_failed_hardware_attempt_falls_back_without_losing_the_camera(
    monkeypatch,
    caplog,
) -> None:
    """Молчаливая потеря камеры хуже потерянной экономии."""
    broken = _hardware_process([b""])
    broken.stderr = io.BytesIO(b"Device creation failed: -22.\n")
    working = Mock()
    working.stdout.read.side_effect = [bytes([5]) * FRAME_BYTES, b""]
    processes = [broken, working]
    commands = []

    def fake_popen(command, **kwargs):
        commands.append(command)
        return processes.pop(0)

    monkeypatch.setattr("src.recorder.frames.subprocess.Popen", fake_popen)

    with caplog.at_level("WARNING", logger="catdoor.frames"):
        frames = list(
            rgb_frame_stream("ffmpeg", "rtsp://camera", 3.0, vaapi_device=RENDER_NODE)
        )

    assert len(frames) == 1
    assert int(frames[0].preview[0, 0, 0]) == 5
    assert "-hwaccel" in commands[0]
    assert "-hwaccel" not in commands[1]
    assert "Device creation failed" in caplog.text
    assert active_hwaccel() == "none"


def test_hardware_is_not_retried_after_it_failed_once(monkeypatch) -> None:
    """`_frame_worker` переподключается вечно, и каждый цикл пробовал бы снова."""
    processes = [_hardware_process(), Mock(), Mock()]
    processes[1].stdout.read.return_value = b""
    processes[2].stdout.read.return_value = b""
    commands = []

    def fake_popen(command, **kwargs):
        commands.append(command)
        return processes.pop(0)

    monkeypatch.setattr("src.recorder.frames.subprocess.Popen", fake_popen)

    list(rgb_frame_stream("ffmpeg", "rtsp://camera", 3.0, vaapi_device=RENDER_NODE))
    list(rgb_frame_stream("ffmpeg", "rtsp://camera", 3.0, vaapi_device=RENDER_NODE))

    assert len(commands) == 3
    assert "-hwaccel" not in commands[2]


def test_hardware_stream_that_died_after_frames_is_not_blamed(monkeypatch) -> None:
    """Замолчавшая камера — не отказ видеоядра, переподключение не здесь."""
    process = _hardware_process([bytes([7]) * FRAME_BYTES, b""])
    commands = []
    monkeypatch.setattr(
        "src.recorder.frames.subprocess.Popen",
        lambda command, **kwargs: commands.append(command) or process,
    )

    list(rgb_frame_stream("ffmpeg", "rtsp://camera", 3.0, vaapi_device=RENDER_NODE))

    assert len(commands) == 1
    assert active_hwaccel() == "vaapi"


def test_unreachable_camera_does_not_blame_the_media_engine(monkeypatch, caplog) -> None:
    """Ровно это и случилось после перезагрузки: сеть камер поднялась позже.

    Программный путь на недоступной камере провалился бы точно так же, поэтому
    переоткрывать поток без ускорения незачем — и уж тем более нельзя терять
    экономию до перезапуска службы.
    """
    commands = []

    def fake_popen(command, **kwargs):
        commands.append(command)
        # Свой процесс на каждую попытку: канал ошибок читается один раз, и
        # общий на всех молча превратил бы вторую жалобу в пустую.
        process = _hardware_process()
        process.stderr = io.BytesIO(
            b"[tcp @ 0x1] Connection to tcp://192.168.1.170:554 failed: "
            b"Network is unreachable\n"
        )
        return process

    monkeypatch.setattr("src.recorder.frames.subprocess.Popen", fake_popen)

    with caplog.at_level("WARNING", logger="catdoor.frames"):
        list(rgb_frame_stream("ffmpeg", "rtsp://camera", 3.0, vaapi_device=RENDER_NODE))
        list(rgb_frame_stream("ffmpeg", "rtsp://camera", 3.0, vaapi_device=RENDER_NODE))

    assert len(commands) == 2
    assert all("-hwaccel" in command for command in commands)
    assert caplog.text == ""
    assert active_hwaccel() is None


def test_unknown_complaint_is_retried_after_the_cooldown(monkeypatch) -> None:
    """Незнакомая жалоба — не приговор: причина могла быть разовой."""
    monkeypatch.setattr("src.recorder.frames.HWACCEL_COOLDOWN_SECONDS", 0.0)
    processes = [_hardware_process(), Mock(), _hardware_process(), Mock()]
    for process in processes:
        process.stdout.read.return_value = b""
    commands = []

    def fake_popen(command, **kwargs):
        commands.append(command)
        return processes.pop(0)

    monkeypatch.setattr("src.recorder.frames.subprocess.Popen", fake_popen)

    list(rgb_frame_stream("ffmpeg", "rtsp://camera", 3.0, vaapi_device=RENDER_NODE))
    list(rgb_frame_stream("ffmpeg", "rtsp://camera", 3.0, vaapi_device=RENDER_NODE))

    # Первая попытка аппаратная, откат программный, а следующее подключение
    # снова пробует видеоядро.
    assert "-hwaccel" in commands[0]
    assert "-hwaccel" not in commands[1]
    assert "-hwaccel" in commands[2]


def test_device_failure_still_locks_the_hardware_path(monkeypatch) -> None:
    """Жалоба про устройство — приговор до перезапуска службы."""
    broken = _hardware_process()
    broken.stderr = io.BytesIO(
        b"[AVHWDeviceContext @ 0x1] Failed to initialise VAAPI connection: -1.\n"
        b"Device creation failed: -22.\n"
    )
    software = Mock()
    software.stdout.read.return_value = b""
    processes = [broken, software, Mock()]
    processes[2].stdout.read.return_value = b""
    commands = []

    def fake_popen(command, **kwargs):
        commands.append(command)
        return processes.pop(0)

    monkeypatch.setattr("src.recorder.frames.subprocess.Popen", fake_popen)

    list(rgb_frame_stream("ffmpeg", "rtsp://camera", 3.0, vaapi_device=RENDER_NODE))
    list(rgb_frame_stream("ffmpeg", "rtsp://camera", 3.0, vaapi_device=RENDER_NODE))

    assert "-hwaccel" in commands[0]
    assert "-hwaccel" not in commands[1]
    assert "-hwaccel" not in commands[2]
    assert active_hwaccel() == "none"
