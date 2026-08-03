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


def resilient_mjpeg_stream(
    open_source: Callable[[], Iterator[bytes] | None],
    *,
    reconnect_delay_seconds: float = 2.0,
    silence_limit_seconds: float = 300.0,
    frame: Callable[[float], bytes] = no_signal_jpeg,
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
    """
    silent_since: float | None = None
    while True:
        source = open_source()
        if source is not None:
            for chunk in source:
                silent_since = None
                yield chunk
        now = _now()
        if silent_since is None:
            silent_since = now
        silence = now - silent_since
        if silence >= silence_limit_seconds:
            return
        yield multipart_frame(frame(silence))
        _wait(reconnect_delay_seconds)
