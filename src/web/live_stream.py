"""Живое превью, которое переживает пропажу камеры.

Каждый источник кадров рано или поздно заканчивается: recorder перестал
публиковать превью, ffmpeg убит сторожем молчания, камере выдернули питание.
Раньше на этом заканчивался и HTTP-ответ, а `<img>` в браузере навсегда
оставался с последним кадром: картинка врала, будто камера жива, и оживала
только перезагрузкой страницы. Здесь ответ переживает обрыв — вместо вранья
идёт заглушка «нет сигнала», а источник переоткрывается, пока зритель смотрит.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from functools import lru_cache
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

NO_SIGNAL_SIZE = (960, 540)
# Пол между двумя открытиями источника, отдавшего хоть кадр. Смена источника
# должна быть незаметной, а не выдержанной, поэтому шаг маленький.
MIN_REOPEN_INTERVAL_SECONDS = 0.5
# Шрифты с кириллицей на Ubuntu и на macOS. Встроенный шрифт Pillow знает
# только латиницу, поэтому без системного текст пришлось бы рисовать по-русски
# прямоугольниками — лучше честная латиница, чем нечитаемая заглушка.
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
)


def _now() -> float:
    return time.monotonic()


def _wait(seconds: float) -> None:
    time.sleep(seconds)


def multipart_frame(jpeg: bytes) -> bytes:
    return (
        b"--frame\r\nContent-Type: image/jpeg\r\n"
        + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
        + jpeg
        + b"\r\n"
    )


@lru_cache(maxsize=1)
def _load_font() -> tuple[ImageFont.ImageFont | ImageFont.FreeTypeFont, bool]:
    """Шрифт для заглушки и признак того, что он умеет кириллицу."""
    for candidate in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, 36), True
        except OSError:
            continue
    return ImageFont.load_default(), False


def no_signal_jpeg(silent_seconds: float) -> bytes:
    """Кадр-заглушка со счётчиком тишины.

    Счётчик здесь не украшение: он единственное, чем «нет сигнала» отличается
    от замершего изображения — видно, что поток жив и продолжает ждать камеру.
    """
    font, cyrillic = _load_font()
    caption = "НЕТ СИГНАЛА" if cyrillic else "NO SIGNAL"
    elapsed = f"{int(silent_seconds)} с" if cyrillic else f"{int(silent_seconds)} s"
    image = Image.new("RGB", NO_SIGNAL_SIZE, (24, 24, 27))
    draw = ImageDraw.Draw(image)
    width, height = NO_SIGNAL_SIZE
    # Растровый запасной шрифт не умеет якорей, и с ними отрисовка упала бы —
    # заглушка обязана получаться на любой машине, где нет системных шрифтов.
    anchor = "mm" if isinstance(font, ImageFont.FreeTypeFont) else None
    draw.text((width / 2, height / 2 - 24), caption, fill="#f87171", font=font, anchor=anchor)
    draw.text((width / 2, height / 2 + 28), elapsed, fill="#a1a1aa", font=font, anchor=anchor)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=70)
    return buffer.getvalue()


def until(
    source: Iterator[bytes],
    should_stop: Callable[[], bool],
    *,
    check_interval_seconds: float = 1.0,
) -> Iterator[bytes]:
    """Кадры источника, пока не появился источник получше.

    Два источника превью несимметричны, и на этом однажды сломались рамки
    детекции: аннотированные кадры кончаются сами, когда recorder перестаёт их
    публиковать, а поток с живой камеры не кончается никогда. Без этой обёртки
    выбор, сделанный в момент, когда превью протухло, оставался бы навсегда —
    картинка живая, рамок нет, и помогает только перезагрузка страницы.
    """
    checked_at = _now()
    for chunk in source:
        now = _now()
        if now - checked_at >= check_interval_seconds:
            checked_at = now
            if should_stop():
                return
        yield chunk


def resilient_mjpeg_stream(
    open_source: Callable[[], Iterator[bytes] | None],
    *,
    reconnect_delay_seconds: float = 2.0,
    silence_limit_seconds: float = 300.0,
    frame: Callable[[float], bytes] = no_signal_jpeg,
    stopping: Callable[[], bool] = lambda: False,
) -> Iterator[bytes]:
    """Кадры из источника, переоткрывая его, пока зритель не ушёл.

    `open_source` вызывается заново на каждую попытку: за время обрыва камера
    могла получить другой адрес, а recorder — начать публиковать превью, и
    держаться за выбор, сделанный в начале запроса, значило бы переподключаться
    к камере, которой больше нет. `None` — камеру убрали или выключили.

    Заглушка отдаётся не только для зрителя: пока генератор молчит, обрыв
    соединения браузером остаётся незамеченным, и каждая закрытая вкладка
    держала бы свой ffmpeg. Регулярный yield — то, чем этот цикл отличается от
    вечного цикла без единого yield, который однажды парализовал админку.

    Ожидание не бесконечно: забытая вкладка перед мёртвой камерой иначе
    поднимала бы ffmpeg заново до перезапуска веб-службы.

    `stopping` — остановка службы. Ответ, который по замыслу не кончается,
    держал бы её до принудительного убийства по таймауту: uvicorn при остановке
    ждёт завершения запросов, и одной открытой вкладки хватало, чтобы каждый
    перезапуск занимал полторы минуты.
    """
    silent_since: float | None = None
    while not stopping():
        opened_at = _now()
        source = open_source()
        delivered = False
        if source is not None:
            for chunk in source:
                silent_since = None
                delivered = True
                yield chunk
                if stopping():
                    return
        if delivered:
            # Источник отдавал кадры и закончился сам — это смена источника, а
            # не пропажа камеры: превью протухло и уступило место камере, либо
            # наоборот recorder снова начал публиковать кадры. Заглушка и пауза
            # здесь означали бы «нет сигнала» посреди исправной картинки.
            # Минимальный шаг всё же нужен: камера, отдающая кадр и сразу
            # обрывающаяся, иначе крутила бы запуск ffmpeg без передышки.
            streamed = _now() - opened_at
            if streamed < MIN_REOPEN_INTERVAL_SECONDS:
                _wait(MIN_REOPEN_INTERVAL_SECONDS - streamed)
            continue
        now = _now()
        if silent_since is None:
            silent_since = now
        silence = now - silent_since
        if silence >= silence_limit_seconds:
            return
        yield multipart_frame(frame(silence))
        _wait(reconnect_delay_seconds)
