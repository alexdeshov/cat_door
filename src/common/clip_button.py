"""Кнопка «Видео» под уведомлением и разбор нажатия на неё.

Уведомление приходит со снимком: одного кадра хватает, чтобы понять, надо ли
смотреть дальше. Ролик же весит мегабайты, и слать его каждым сообщением —
значит забивать чат и мобильный трафик ради проходов, которые чаще всего
никому не интересны. Поэтому ролик едет по нажатию.

Нажатие возвращается в бот единственным способом: строкой `callback_data`,
которую мы сами и записали в кнопку. Сборка и разбор этой строки живут здесь,
а не в службе, которая её читает: пишут её двое (recorder и уведомления
калитки), читает третий, и разъехавшиеся форматы означали бы мёртвую кнопку в
чате — то есть поломку, которую видно только на телефоне владельца.

Ограничение Телеграма — 64 байта на нагрузку. Идентификаторы Frigate
(`1786171119.126838-b1jywn`) укладываются с запасом, идентификаторы наших
проходов — тем более, но предел проверяется явно: сообщение с длинной
нагрузкой Телеграм отвергает целиком, вместе с уведомлением.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common.telegram import CALLBACK_DATA_LIMIT

# Приставка нужна, чтобы кнопка «видео» не оказалась единственным, что бот
# когда-либо умел: нажатие чужого вида просто не разберётся и будет пропущено.
CLIP_PREFIX = "clip"
# Событие ленты Frigate (калитка) — ролик тянется по HTTP у самого Frigate.
GATE_KIND = "g"
# Проход кота — наша запись в базе, ролик лежит файлом в каталоге данных.
PASS_KIND = "p"
# Ракурсы кошачьего прохода. Буквой, а не словом: в 64 байта помещается всё,
# но короткая нагрузка оставляет запас на идентификаторы, которые длиннее.
FACE_VIEW = "f"
SIDE_VIEW = "s"

VIEW_NAMES = {FACE_VIEW: "face", SIDE_VIEW: "side"}


@dataclass(frozen=True)
class ClipRequest:
    """Что именно попросили показать."""

    kind: str
    event_id: str
    view: str = ""


def clip_callback_data(kind: str, event_id: str, view: str = "") -> str | None:
    """Нагрузка кнопки или `None`, если она не влезает в предел Телеграма.

    `None`, а не исключение: уведомление без кнопки — потеря удобства, а
    уведомление, не ушедшее вовсе, — потеря сообщения о человеке у калитки.
    """
    parts = [CLIP_PREFIX, kind, event_id]
    if view:
        parts.append(view)
    data = ":".join(parts)
    if len(data.encode()) > CALLBACK_DATA_LIMIT:
        return None
    return data


def _button(text: str, kind: str, event_id: str, view: str = "") -> list[tuple[str, str]]:
    data = clip_callback_data(kind, event_id, view)
    return [] if data is None else [(text, data)]


def gate_clip_buttons(event_id: str) -> tuple[tuple[str, str], ...]:
    """Кнопка под уведомлением калитки.

    Ставится независимо от того, готов ли клип: в момент уведомления его ещё
    нет — Frigate дописывает запись после конца события, — а к нажатию он уже
    появляется. Проверять готовность заранее значило бы прятать кнопку у самых
    свежих событий, то есть у тех единственных, ради которых её нажимают.
    """
    return tuple(_button("🎬 Видео", GATE_KIND, event_id))


def pass_clip_buttons(
    event_id: str,
    *,
    has_face: bool,
    has_side: bool,
) -> tuple[tuple[str, str], ...]:
    """Кнопки под уведомлением о проходе кота — по одной на записанный ракурс.

    Ракурсы называются по-человечески, а не «face/side»: в чат смотрит не тот,
    кто знает наши роли камер.
    """
    buttons: list[tuple[str, str]] = []
    if has_face:
        buttons += _button("🎬 Видео с дверцы", PASS_KIND, event_id, FACE_VIEW)
    if has_side:
        buttons += _button("🎬 Видео с террасы", PASS_KIND, event_id, SIDE_VIEW)
    return tuple(buttons)


def parse_clip_callback(data: str) -> ClipRequest | None:
    """Разобрать нажатие. `None` — это не наша кнопка, и трогать её не нужно."""
    parts = data.split(":")
    if len(parts) < 3 or parts[0] != CLIP_PREFIX:
        return None
    kind, event_id = parts[1], parts[2]
    if kind not in (GATE_KIND, PASS_KIND) or not event_id:
        return None
    view = parts[3] if len(parts) > 3 else ""
    if kind == PASS_KIND and view not in VIEW_NAMES:
        return None
    return ClipRequest(kind=kind, event_id=event_id, view=view)
