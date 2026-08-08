from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

import numpy as np

from src.inference.preprocessing import NormalizedRoi
from src.recorder.ffmpeg import _base_input, _collect_stderr, mask_credentials
from src.recorder.ring import prune_segments

logger = logging.getLogger("catdoor.frames")

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


def _vaapi_filter(fps: float, rotate_180: bool) -> str:
    """Тот же кадр 960×540, но уменьшенный медиадвижком.

    В обычную память забирается уже уменьшенный кадр: `hwdownload` стоит тем
    дороже, чем больше картинка. Поля добавляются после скачивания — на
    960×540 это копейки, а `pad_vaapi` есть не во всякой сборке ffmpeg.
    Поворот тоже после полей: они симметричные, картинка выходит та же, что и
    на программном пути, поэтому превью и детектор видят одну ориентацию.
    """
    orientation = ",hflip,vflip" if rotate_180 else ""
    return (
        f"fps={fps},"
        f"scale_vaapi={FRAME_WIDTH}:{FRAME_HEIGHT}:"
        "force_original_aspect_ratio=decrease:force_divisible_by=2,"
        "hwdownload,format=nv12,"
        f"pad={FRAME_WIDTH}:{FRAME_HEIGHT}:(ow-iw)/2:(oh-ih)/2"
        f"{orientation}"
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


# Состояние аппаратного декодирования на весь процесс. Отказ по железу
# запирается защёлкой: `_frame_worker` переподключается вечно, и без неё каждый
# цикл тратил бы секунду на заведомо мёртвую попытку и строку в журнал.
_hwaccel_lock = threading.Lock()
_hwaccel_failed = False
_hwaccel_confirmed = False
_hwaccel_retry_after = 0.0
_roi_notice_shown = False

# Пауза после непонятного отказа. Заперать видеоядро навсегда из-за жалобы,
# которую мы не узнали, слишком дорого; молотить попытками каждую секунду —
# слишком шумно.
HWACCEL_COOLDOWN_SECONDS = 60.0

# Жалоба про видеоядро: попытка провалилась там, где мы можем помочь только
# отказом от аппаратного пути.
_HARDWARE_COMPLAINTS = (
    "vaapi",
    "hwaccel",
    "hwdownload",
    "renderd",
    "device creation failed",
    "no va display",
    "function not implemented",
    "impossible to convert between the formats",
    "no device available for decoder",
)

# Жалоба про вход: камера не ответила. Видеоядро тут ни при чём — программный
# путь на той же камере провалился бы точно так же.
_INPUT_COMPLAINTS = (
    "network is unreachable",
    "no route to host",
    "connection refused",
    "connection timed out",
    "operation timed out",
    "name or service not known",
    "immediate exit requested",
    "401 unauthorized",
    "403 forbidden",
    "404 not found",
    "invalid data found when processing input",
)


def _hwaccel_verdict(complaints: list[str]) -> str:
    """`hardware`, `camera` или `unknown` по жалобам ffmpeg.

    Порядок проверок не переставлять: при мёртвом видеоядре ffmpeg жалуется и
    на устройство, и на вход («Error while opening decoder for input stream»),
    поэтому железо смотрится первым.
    """
    text = " ".join(complaints).lower()
    if any(marker in text for marker in _HARDWARE_COMPLAINTS):
        return "hardware"
    if any(marker in text for marker in _INPUT_COMPLAINTS):
        return "camera"
    return "unknown"


def active_hwaccel() -> str | None:
    """Что декодирует кадры на самом деле, а не что попросили в настройках.

    `vaapi` — хотя бы один поток отдал кадр через медиадвижок; `none` —
    видеоядро отказало и до перезапуска службы работает программный путь;
    `None` — ещё не известно.
    """
    with _hwaccel_lock:
        if _hwaccel_failed:
            return "none"
        return "vaapi" if _hwaccel_confirmed else None


def reset_hwaccel_state() -> None:
    """Только для тестов: защёлка живёт на весь процесс."""
    global _hwaccel_failed, _hwaccel_confirmed, _hwaccel_retry_after, _roi_notice_shown
    with _hwaccel_lock:
        _hwaccel_failed = False
        _hwaccel_confirmed = False
        _hwaccel_retry_after = 0.0
        _roi_notice_shown = False


def _confirm_hwaccel() -> None:
    global _hwaccel_confirmed
    with _hwaccel_lock:
        _hwaccel_confirmed = True


def _disable_hwaccel(complaints: list[str]) -> None:
    global _hwaccel_failed
    with _hwaccel_lock:
        already_known = _hwaccel_failed
        _hwaccel_failed = True
    if already_known:
        return
    reason = mask_credentials(" ".join(complaints).strip())
    logger.warning(
        "Видеоядро отказало, перехожу на программное декодирование до "
        "перезапуска службы: %s",
        reason or "ffmpeg ничего не сообщил",
    )


def _defer_hwaccel(complaints: list[str]) -> None:
    """Отложить аппаратный путь, не отказываясь от него насовсем.

    Сюда попадают жалобы, которых мы не узнали. Камера важнее экономии, поэтому
    сейчас поток пересобирается программным, но через паузу видеоядро пробуется
    снова: причина могла быть разовой.
    """
    global _hwaccel_retry_after
    with _hwaccel_lock:
        already_waiting = time.monotonic() < _hwaccel_retry_after
        _hwaccel_retry_after = time.monotonic() + HWACCEL_COOLDOWN_SECONDS
    if already_waiting:
        return
    reason = mask_credentials(" ".join(complaints).strip())
    logger.warning(
        "Аппаратное декодирование не дало кадров по непонятной причине, "
        "работаю программно и повторю попытку через %.0f с: %s",
        HWACCEL_COOLDOWN_SECONDS,
        reason or "ffmpeg ничего не сообщил",
    )


def _hwaccel_device(vaapi_device: Path | None, roi: NormalizedRoi | None) -> Path | None:
    global _roi_notice_shown
    if vaapi_device is None:
        return None
    if roi is not None:
        # Обрезку видеоядро в ffmpeg не умеет (`crop_vaapi` не существует), а
        # скачивание кадра 4 Мп ради неё съело бы весь выигрыш. Оговорка раз на
        # процесс: мёртвая камера переподключается раз в секунду.
        with _hwaccel_lock:
            already_shown, _roi_notice_shown = _roi_notice_shown, True
        if not already_shown:
            logger.info(
                "Камера с областью интереса остаётся на программном "
                "декодировании: обрезка кадра выполняется до масштабирования"
            )
        return None
    with _hwaccel_lock:
        if _hwaccel_failed or time.monotonic() < _hwaccel_retry_after:
            return None
    return vaapi_device


def _frame_command(
    ffmpeg_path: str,
    rtsp_url: str,
    fps: float,
    *,
    segment_directory: Path | None,
    segment_seconds: int,
    roi: NormalizedRoi | None,
    rotate_180: bool,
    vaapi_device: Path | None,
) -> list[str]:
    command = [
        *_base_input(ffmpeg_path, rtsp_url, vaapi_device=vaapi_device),
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
        filters = (
            _vaapi_filter(fps, rotate_180)
            if vaapi_device is not None
            else f"fps={fps},{_scale_filter()}{',hflip,vflip' if rotate_180 else ''}"
        )
        command.extend(["-map", "0:v:0", "-vf", filters])
    else:
        command.extend(
            [
                "-filter_complex",
                _roi_filter(fps, roi, rotate_180),
                "-map",
                "[combined]",
            ]
        )
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
    return command


def _stream_once(
    command: list[str],
    *,
    frame_bytes: int,
    output_width: int,
    roi: NormalizedRoi | None,
    segment_directory: Path | None,
    retention_seconds: int,
    on_process: Callable[[subprocess.Popen[bytes]], None] | None,
    hardware: bool,
    complaints: list[str],
    delivered: list[int],
) -> Iterator[FrameSample]:
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        # Жалобы ffmpeg нужны только на аппаратной попытке — по ним человек
        # поймёт, чего не хватило видеоядру. Читать их обязан отдельный поток:
        # заполненный канал ошибок остановил бы ffmpeg навсегда.
        stderr=subprocess.PIPE if hardware else subprocess.DEVNULL,
    )
    drain: threading.Thread | None = None
    if hardware:
        drain = threading.Thread(
            target=_collect_stderr,
            args=(process.stderr, complaints),
            daemon=True,
            name="catdoor-frames-stderr",
        )
        drain.start()
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
            delivered[0] += 1
            if hardware and delivered[0] == 1:
                _confirm_hwaccel()
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
        if drain is not None:
            drain.join(timeout=2)


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
    vaapi_device: Path | None = None,
) -> Iterator[FrameSample]:
    """Кадры с камеры. `on_process` отдаёт наружу сам процесс ffmpeg.

    Отдавать его приходится потому, что чтение кадра блокирующее: если камера
    пропала так, что TCP-сессия не закрылась, поток роли останется в `read`
    навсегда, и разбудить его можно только со стороны — убив процесс. Кто
    именно следит за застоем, здесь не решается.

    `vaapi_device` просит декодировать поток медиадвижком. Если попытка не дала
    ни одного кадра, поток молча не теряется: команда пересобирается программной
    и запускается заново, а аппаратный путь выключается до перезапуска службы.
    """
    device = _hwaccel_device(vaapi_device, roi)
    frame_bytes = FRAME_BYTES if roi is None else FRAME_BYTES * 2
    output_width = FRAME_WIDTH if roi is None else FRAME_WIDTH * 2
    while True:
        complaints: list[str] = []
        delivered = [0]
        yield from _stream_once(
            _frame_command(
                ffmpeg_path,
                rtsp_url,
                fps,
                segment_directory=segment_directory,
                segment_seconds=segment_seconds,
                roi=roi,
                rotate_180=rotate_180,
                vaapi_device=device,
            ),
            frame_bytes=frame_bytes,
            output_width=output_width,
            roi=roi,
            segment_directory=segment_directory,
            retention_seconds=retention_seconds,
            on_process=on_process,
            hardware=device is not None,
            complaints=complaints,
            delivered=delivered,
        )
        if device is None or delivered[0]:
            # Поток, отдавший хотя бы кадр, закончился по обычной причине —
            # камера замолчала или её убили снаружи. Переподключение не здесь.
            return
        verdict = _hwaccel_verdict(complaints)
        if verdict == "camera":
            # Камеры нет — программный путь провалился бы ровно так же, и
            # выключать из-за этого видеоядро нельзя. Ровно это и случилось
            # после перезагрузки узла: сеть камер поднимается позже службы.
            return
        if verdict == "hardware":
            _disable_hwaccel(complaints)
        else:
            _defer_hwaccel(complaints)
        device = None
