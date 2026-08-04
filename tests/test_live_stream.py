from __future__ import annotations

from collections.abc import Iterator

from src.web.live_stream import no_signal_jpeg, resilient_mjpeg_stream, until


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
    chunks = [next(stream) for _ in range(4)]

    assert b"live" in chunks[0]
    # Каждая неудачная попытка отдаёт заглушку: пока она обновляется, видно,
    # что поток ждёт камеру, а не замер.
    assert all(b"\xff\xd8" in chunk for chunk in chunks[1:3])
    assert b"back" in chunks[3]


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

    # Кадр и заглушка чередуются: продуктивный источник заглушку не вызывает.
    assert sum(b"live" in chunk for chunk in chunks) == 15


def test_no_signal_frame_shows_that_it_is_still_counting() -> None:
    """Заглушка со счётчиком — единственное отличие от замершего кадра."""
    fresh = no_signal_jpeg(0.0)
    later = no_signal_jpeg(97.0)

    assert fresh.startswith(b"\xff\xd8") and fresh.endswith(b"\xff\xd9")
    assert fresh != later


def test_switching_sources_does_not_flash_no_signal(monkeypatch) -> None:
    """Смена источника — не пропажа камеры.

    Превью и камера сменяют друг друга при живой картинке: recorder то
    публикует кадры, то перестаёт. Заглушка между ними означала бы «нет
    сигнала» посреди исправного изображения.
    """
    use_clock(monkeypatch)
    sources = [one_frame(b"annotated"), one_frame(b"raw"), one_frame(b"annotated")]

    stream = resilient_mjpeg_stream(
        lambda: sources.pop(0) if sources else None,
        silence_limit_seconds=60.0,
    )
    chunks = [next(stream) for _ in range(3)]

    assert [b"annotated", b"raw", b"annotated"] == [
        b"annotated" if b"annotated" in chunk else b"raw" for chunk in chunks
    ]


def test_a_source_that_dies_instantly_does_not_spin(monkeypatch) -> None:
    """Камера, отдающая кадр и обрывающаяся, не должна крутить запуск ffmpeg."""
    clock = use_clock(monkeypatch)

    stream = resilient_mjpeg_stream(
        lambda: one_frame(b"flap"),
        silence_limit_seconds=60.0,
    )
    for _ in range(10):
        next(stream)

    assert clock.now >= 4.5


def test_until_stops_the_endless_source_when_a_better_one_appears(monkeypatch) -> None:
    """Рамки детекции пропадали именно здесь.

    Поток с живой камеры не кончается сам, поэтому выбор, сделанный при
    протухшем превью, оставался навсегда: картинка живая, рамок нет.
    """
    clock = use_clock(monkeypatch)
    preview_ready = {"value": False}

    def endless() -> Iterator[bytes]:
        while True:
            clock.sleep(0.5)
            yield b"raw"

    bounded = until(endless(), lambda: preview_ready["value"])
    assert [next(bounded) for _ in range(4)] == [b"raw"] * 4

    preview_ready["value"] = True

    # Проверка идёт не чаще раза в секунду, поэтому счёт идёт на кадры, а не на
    # мгновенность: важно, что поток кончается сам и рамки возвращаются.
    assert len(list(bounded)) <= 2


def test_a_stopping_service_is_not_held_by_the_stream(monkeypatch) -> None:
    """Ответ, который по замыслу не кончается, держал бы остановку службы.

    uvicorn при остановке ждёт завершения запросов, и одной открытой вкладки
    админки хватало, чтобы перезапуск занимал полторы минуты и заканчивался
    убийством по таймауту.
    """
    use_clock(monkeypatch)
    stopping = {"value": False}

    def endless() -> Iterator[bytes]:
        while True:
            yield b"frame"

    stream = resilient_mjpeg_stream(
        endless,
        silence_limit_seconds=600.0,
        stopping=lambda: stopping["value"],
    )
    assert [next(stream) for _ in range(3)] == [b"frame"] * 3

    stopping["value"] = True

    assert list(stream) == []


def test_a_stopping_service_does_not_even_open_a_source(monkeypatch) -> None:
    use_clock(monkeypatch)

    opened = []

    def open_source() -> Iterator[bytes] | None:
        opened.append(1)
        return one_frame(b"live")

    assert list(resilient_mjpeg_stream(open_source, stopping=lambda: True)) == []
    assert opened == []
