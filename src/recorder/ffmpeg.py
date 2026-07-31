import subprocess
from collections.abc import Iterator
from pathlib import Path


class RecorderError(RuntimeError):
    pass


def ffmpeg_creation_flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _base_input(ffmpeg_path: str, rtsp_url: str) -> list[str]:
    return [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-rtsp_transport",
        "tcp",
        "-timeout",
        "5000000",
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
            creationflags=ffmpeg_creation_flags(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        temporary.unlink(missing_ok=True)
        raise RecorderError("ffmpeg_unavailable_or_timed_out") from exc

    if result.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RecorderError("rtsp_recording_failed")
    temporary.replace(destination)


def mjpeg_stream(ffmpeg_path: str, rtsp_url: str, fps: int) -> Iterator[bytes]:
    command = [
        *_base_input(ffmpeg_path, rtsp_url),
        "-map",
        "0:v:0",
        "-vf",
        f"fps={fps},scale=960:-2",
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
        stderr=subprocess.DEVNULL,
        creationflags=ffmpeg_creation_flags(),
    )
    buffer = bytearray()
    try:
        assert process.stdout is not None
        while chunk := process.stdout.read(8192):
            buffer.extend(chunk)
            while True:
                start = buffer.find(b"\xff\xd8")
                end = buffer.find(b"\xff\xd9", start + 2)
                if start < 0 or end < 0:
                    break
                frame = bytes(buffer[start : end + 2])
                del buffer[: end + 2]
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
