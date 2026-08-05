from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

import numpy as np

from src.inference.preprocessing import NormalizedRoi
from src.recorder.ffmpeg import _base_input
from src.recorder.ring import prune_segments

# Размер кадра, в котором работает кошачий контур. Он же задаёт предел
# детализации кропа для идентификации: кроп режется по рамке кота именно
# отсюда, а не из исходного потока.
#
# Было 640×360, и на камере подхода это резало по живому. Замер по 118 рамкам
# из восьми размеченных проходов: кот занимал там в среднем 112×105 точек, кроп
# с запасом выходил 142 точки — меньше входа классификатора в 160, то есть две
# трети примеров модель получала растянутыми. На камере дверцы кот вдвое
# крупнее, и её это не задевало.
#
# 960×540 закрывает обе камеры с запасом: медиана кропа 490 и 213 точек, ниже
# входа модели не опускается ни один. Дальше расти незачем — классификатор
# принимает 160×160, и лишние пиксели снова ушли бы в мусор, а поток сырых
# кадров по каналу растёт линейно.
#
# Записи проходов от этого размера не зависят вовсе: кольцевой буфер пишет
# сегменты копированием исходного потока.
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
FRAME_BYTES = FRAME_WIDTH * FRAME_HEIGHT * 3


@dataclass(frozen=True)
class FrameSample:
    preview: np.ndarray
    detector: np.ndarray


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            break
        chunks.extend(chunk)
    return bytes(chunks)


def _scale_filter() -> str:
    return (
        f"scale={FRAME_WIDTH}:{FRAME_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={FRAME_WIDTH}:{FRAME_HEIGHT}:(ow-iw)/2:(oh-ih)/2"
    )


def _roi_filter(fps: float, roi: NormalizedRoi, rotate_180: bool) -> str:
    crop = (
        f"crop=iw*{roi.width:g}:ih*{roi.height:g}:"
        f"iw*{roi.x:g}:ih*{roi.y:g}"
    )
    scale = _scale_filter()
    orientation = ",hflip,vflip" if rotate_180 else ""
    return (
        f"[0:v:0]fps={fps}{orientation},split=2[preview_source][detector_source];"
        f"[preview_source]{scale}[preview];"
        f"[detector_source]{crop},{scale}[detector];"
        "[preview][detector]hstack=inputs=2[combined]"
    )


def rgb_frame_stream(
    ffmpeg_path: str,
    rtsp_url: str,
    fps: float,
    *,
    segment_directory: Path | None = None,
    segment_seconds: int = 2,
    retention_seconds: int = 180,
    roi: NormalizedRoi | None = None,
    rotate_180: bool = True,
    on_process: Callable[[subprocess.Popen[bytes]], None] | None = None,
) -> Iterator[FrameSample]:
    """Кадры с камеры. `on_process` отдаёт наружу сам процесс ffmpeg.

    Отдавать его приходится потому, что чтение кадра блокирующее: если камера
    пропала так, что TCP-сессия не закрылась, поток роли останется в `read`
    навсегда, и разбудить его можно только со стороны — убив процесс. Кто
    именно следит за застоем, здесь не решается.
    """
    command = [
        *_base_input(ffmpeg_path, rtsp_url),
    ]
    if segment_directory is not None:
        segment_directory.mkdir(parents=True, exist_ok=True)
        command.extend(
            [
                "-map",
                "0:v:0",
                "-an",
                "-c:v",
                "copy",
                "-f",
                "segment",
                "-segment_time",
                str(segment_seconds),
                "-reset_timestamps",
                "1",
                "-strftime",
                "1",
                str(segment_directory / "%Y%m%dT%H%M%S.mp4"),
            ]
        )
    if roi is None:
        command.extend(
            [
                "-map",
                "0:v:0",
                "-vf",
                (
                    f"fps={fps},{_scale_filter()}"
                    f"{',hflip,vflip' if rotate_180 else ''}"
                ),
            ]
        )
        frame_bytes = FRAME_BYTES
        output_width = FRAME_WIDTH
    else:
        command.extend(
            [
                "-filter_complex",
                _roi_filter(fps, roi, rotate_180),
                "-map",
                "[combined]",
            ]
        )
        frame_bytes = FRAME_BYTES * 2
        output_width = FRAME_WIDTH * 2
    command.extend(
        [
            "-an",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
    )
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if on_process is not None:
        on_process(process)
    try:
        assert process.stdout is not None
        stream = cast(BinaryIO, process.stdout)
        last_prune_at = 0.0
        while True:
            frame = _read_exact(stream, frame_bytes)
            if len(frame) != frame_bytes:
                return
            now = time.monotonic()
            if segment_directory is not None and now - last_prune_at >= 5:
                prune_segments(segment_directory, retention_seconds)
                last_prune_at = now
            decoded = np.frombuffer(frame, dtype=np.uint8).reshape(
                (FRAME_HEIGHT, output_width, 3)
            )
            if roi is None:
                yield FrameSample(preview=decoded, detector=decoded)
            else:
                yield FrameSample(
                    preview=np.ascontiguousarray(decoded[:, :FRAME_WIDTH, :]),
                    detector=np.ascontiguousarray(decoded[:, FRAME_WIDTH:, :]),
                )
    finally:
        if process.stdout:
            process.stdout.close()
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
