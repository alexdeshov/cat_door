from __future__ import annotations

from collections.abc import Iterator

from src.web.live_stream import no_signal_jpeg, resilient_mjpeg_stream


class Clock:
    """Часы, которые двигает только сон: тест не должен ничего ждать."""

    def __init__(self) -> None:
        self.now = 0.0

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def monotonic(self) -> float:
        return self.now


def use_clock(monkeypatch) -> Clock:
    clock = Clock()
    monkeypatch.setattr("src.web.live_stream._wait", clock.sleep)
    monkeypatch.setattr("src.web.live_stream._now", clock.monotonic)
    return clock


def one_frame(payload: bytes) -> Iterator[bytes]:
    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"


def test_stream_outlives_a_camera_that_lost_power(monkeypatch) -> None:
    """Именно этого не хватало: картинка обязана ожить без перезагрузки страницы.

    Прежде ответ заканчивался вместе с источником, и `<img>` навсегда замирал
    на последнем кадре — камере вернули питание, а в админке всё та же ложь.
    """
    use_clock(monkeypatch)
    sources = [one_frame(b"live"), None, None, one_frame(b"back")]

    def open_source() -> Iterator[bytes] | None:
        return sources.pop(0) if sources else None

    stream = resilient_mjpeg_stream(open_source, silence_limit_seconds=60.0)
    chunks = [next(stream) for _ in range(5)]

    assert b"live" in chunks[0]
    # Обрыв и каждая неудачная попытка отдают заглушку: пока она обновляется,
    # видно, что поток ждёт камеру, а не замер.
    assert all(b"\xff\xd8" in chunk for chunk in chunks[1:4])
    assert b"back" in chunks[4]


def test_stream_gives_up_on_a_camera_that_never_returns(monkeypatch) -> None:
    """Забытая вкладка не должна поднимать ffmpeg до перезапуска службы."""
    use_clock(monkeypatch)

    stream = resilient_mjpeg_stream(
        lambda: None,
        reconnect_delay_seconds=2.0,
        silence_limit_seconds=10.0,
    )

    assert len(list(stream)) == 5


def test_returning_frames_reset_the_countdown(monkeypatch) -> None:
    """Живая камера не приближает конец ответа, сколько бы он ни длился."""
    use_clock(monkeypatch)
    attempts = {"count": 0}

    def open_source() -> Iterator[bytes] | None:
        attempts["count"] += 1
        return one_frame(b"live") if attempts["count"] % 2 else None

    stream = resilient_mjpeg_stream(open_source, silence_limit_seconds=10.0)
    # Без сброса отсчёта поток кончился бы на седьмом куске, и `next` упал бы.
    chunks = [next(stream) for _ in range(30)]

    assert sum(b"live" in chunk for chunk in chunks) == 10


def test_no_signal_frame_shows_that_it_is_still_counting() -> None:
    """Заглушка со счётчиком — единственное отличие от замершего кадра."""
    fresh = no_signal_jpeg(0.0)
    later = no_signal_jpeg(97.0)

    assert fresh.startswith(b"\xff\xd8") and fresh.endswith(b"\xff\xd9")
    assert fresh != later
