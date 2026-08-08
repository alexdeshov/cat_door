import logging
import re
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path

from src.common.durable_write import publish_file_durably

logger = logging.getLogger("catdoor.ffmpeg")

# FFmpeg quotes the whole input URL back in its diagnostics, and the RTSP URL
# carries the camera password. Everything that leaves this module is masked.
_CREDENTIALS = re.compile(r"(?<=://)[^/@\s]+:[^/@\s]*@")


class RecorderError(RuntimeError):
    pass


def _collect_stderr(stream: object, sink: list[str]) -> None:
    if stream is None:
        return
    try:
        for line in stream:  # type: ignore[attr-defined]
            sink.append(line.decode("utf-8", "replace").strip())
    except (OSError, ValueError):
        return


def mask_credentials(text: str) -> str:
    """Replace `user:password@` inside any URL with a marker.

    Logs and error messages are read by people and shipped to the journal, so
    they must never carry the camera password.
    """
    return _CREDENTIALS.sub("***@", text)


def _base_input(
    ffmpeg_path: str,
    rtsp_url: str,
    *,
    vaapi_device: Path | None = None,
) -> list[str]:
    """Общее начало любой команды ffmpeg проекта.

    `vaapi_device` включает декодирование медиадвижком и задаётся только там,
    где кадры действительно декодируются, — в `src/recorder/frames.py`. Запись
    прохода идёт копированием потока (`-c:v copy`), декодирования не требует
    вовсе, и `-hwaccel` заставил бы ffmpeg делать лишнюю работу.
    """
    acceleration = (
        [
            "-hwaccel",
            "vaapi",
            "-hwaccel_device",
            str(vaapi_device),
            # Кадр остаётся в памяти видеоядра: масштабирует его тоже видеоядро,
            # а в обычную память забирается уже уменьшенный.
            "-hwaccel_output_format",
            "vaapi",
        ]
        if vaapi_device is not None
        else []
    )
    return [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-rtsp_transport",
        "tcp",
        "-timeout",
        "5000000",
        *acceleration,
        "-i",
        rtsp_url,
    ]


def record_clip(
    ffmpeg_path: str,
    rtsp_url: str,
    destination: Path,
    duration_seconds: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part.mp4")
    command = [
        *_base_input(ffmpeg_path, rtsp_url),
        "-t",
        str(duration_seconds),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "copy",
        "-movflags",
        "+faststart",
        "-y",
        str(temporary),
    ]
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=duration_seconds + 15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        temporary.unlink(missing_ok=True)
        raise RecorderError("ffmpeg_unavailable_or_timed_out") from exc

    if result.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        reason = mask_credentials((result.stderr or "").strip())
        logger.error("Запись RTSP не удалась: %s", reason or "ffmpeg ничего не сообщил")
        raise RecorderError("rtsp_recording_failed")
    publish_file_durably(temporary, destination)


def probe_stream(
    ffmpeg_path: str,
    rtsp_url: str,
    *,
    timeout_seconds: float = 12.0,
) -> str | None:
    """Забрать с камеры один кадр. `None` — получилось.

    Нужно после каждой перестановки: адрес мог достаться другой камере, пароль
    мог не совпасть, а поток на этом пути мог не существовать вовсе. В браузере
    все три случая выглядят одинаково — пустым прямоугольником.
    """
    command = [
        *_base_input(ffmpeg_path, rtsp_url),
        "-map",
        "0:v:0",
        "-frames:v",
        "1",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"камера не ответила за {timeout_seconds:g} с"
    except OSError:
        return "ffmpeg недоступен"
    if result.returncode == 0:
        return None
    complaints = (result.stderr or "").strip().splitlines()
    return mask_credentials(complaints[-1]) if complaints else "ffmpeg ничего не сообщил"


def _terminate_when_idle(
    process: subprocess.Popen[bytes],
    progress: list[float],
    idle_timeout_seconds: float,
) -> None:
    """Прервать поток, который перестал отдавать кадры.

    Чтение из ffmpeg блокирующее, поэтому заметить молчание может только кто-то
    снаружи. Без этого забытая вкладка админки с мёртвой камерой держала бы свой
    процесс ffmpeg до перезапуска веб-службы, а таких вкладок за вечер
    накапливается сколько угодно.
    """
    while process.poll() is None:
        if time.monotonic() - progress[0] < idle_timeout_seconds:
            time.sleep(min(1.0, idle_timeout_seconds))
            continue
        with suppress(OSError):
            process.terminate()
        # ffmpeg, застрявший в сетевом вводе-выводе, SIGTERM не разбудит —
        # тогда чтение так и осталось бы заблокированным навсегда.
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            with suppress(OSError):
                process.kill()
        return


def mjpeg_stream(
    ffmpeg_path: str,
    rtsp_url: str,
    fps: int,
    *,
    rotate_180: bool = False,
    idle_timeout_seconds: float = 20.0,
) -> Iterator[bytes]:
    # Same pair of flips the detector pipeline uses in src/recorder/frames.py,
    # so an upside-down camera looks the same while being aimed as it will once
    # it is recording.
    orientation = ",hflip,vflip" if rotate_180 else ""
    command = [
        *_base_input(ffmpeg_path, rtsp_url),
        "-map",
        "0:v:0",
        "-vf",
        f"fps={fps},scale=960:-2{orientation}",
        "-q:v",
        "5",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Drained on a thread: a full stderr pipe would block ffmpeg forever, and
    # the frames we are here for come from stdout.
    complaints: list[str] = []
    drain = threading.Thread(
        target=_collect_stderr,
        args=(process.stderr, complaints),
        daemon=True,
        name="catdoor-ffmpeg-stderr",
    )
    drain.start()
    # Отсчёт молчания начинается с запуска: камера, которая приняла соединение
    # и не отдала ни одного кадра, ничем не лучше замолчавшей посреди потока.
    progress = [time.monotonic()]
    idle_guard = threading.Thread(
        target=_terminate_when_idle,
        args=(process, progress, idle_timeout_seconds),
        daemon=True,
        name="catdoor-preview-idle",
    )
    idle_guard.start()
    frames_sent = 0
    buffer = bytearray()
    try:
        assert process.stdout is not None
        while chunk := process.stdout.read(8192):
            progress[0] = time.monotonic()
            buffer.extend(chunk)
            while True:
                start = buffer.find(b"\xff\xd8")
                end = buffer.find(b"\xff\xd9", start + 2)
                if start < 0 or end < 0:
                    break
                frame = bytes(buffer[start : end + 2])
                del buffer[: end + 2]
                frames_sent += 1
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(frame)}\r\n\r\n".encode()
                    + frame
                    + b"\r\n"
                )
    finally:
        if process.stdout:
            process.stdout.close()
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
        drain.join(timeout=2)
        idle_guard.join(timeout=2)
        # A stream that produced nothing is the case worth explaining: in the
        # browser it looks exactly like a working camera showing black.
        if not frames_sent:
            reason = mask_credentials(" ".join(complaints).strip())
            logger.error(
                "Поток с камеры не дал ни одного кадра: %s",
                reason or "ffmpeg ничего не сообщил",
            )
