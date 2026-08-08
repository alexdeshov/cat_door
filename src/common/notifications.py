"""Кого и как часто беспокоить.

Решение отделено от отправки намеренно: молчание — такая же часть уведомлений,
как и сообщение, и проверять его надо без сети. Сейчас правил два, и оба взяты
из измеренного, а не из вкуса.

**Только выбранные роли.** Разбор 170 размеченных проходов: на дверце детектор
разделяет котов и не-котов уверенно (медиана 0.98 против 0.45), а на подходе
распределения совпадают (0.93 против 0.91) — 33 из 38 ложных срабатываний
пришли оттуда, и 22 из них в один солнечный час. Уведомления с подхода до
починки детекции превратили бы телефон в звонок теней от листвы.

**Пауза.** Кот, гуляющий у дверцы, даёт события подряд. Пять минут тишины
после отправленного — это про «пришло уведомление», а не про «пропущен
проход»: сами проходы пишутся все, их всегда видно в админке.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def notification_snapshot_path(data_dir: Path, name: str) -> Path:
    """Куда класть снимок, который вот-вот уедет в сообщение.

    Отдельный каталог не про порядок, а про уборку: снимок удаляется в
    `finally`, а он при пропаже питания не исполняется. Всё, что осталось в
    этом каталоге, заведомо брошено — и подметается при старте recorder.
    """
    return data_dir / "runtime" / "notifications" / f"{name}.jpg"


@dataclass
class NotificationGate:
    roles: frozenset[str]
    cooldown_seconds: float
    _last_sent_at: float | None = field(default=None, repr=False)

    def blocked_reason(self, role: str | None, now: float) -> str | None:
        """`None` — отправляем. Иначе причина молчания, годная для журнала."""
        if not self.roles:
            return "notifications_disabled"
        if role is None:
            return "unknown_camera"
        if role not in self.roles:
            return f"camera_not_notified:{role}"
        if self._last_sent_at is not None and now - self._last_sent_at < self.cooldown_seconds:
            return "cooldown"
        return None

    def remember_sent(self, now: float) -> None:
        self._last_sent_at = now


def pass_caption(
    *,
    when: datetime,
    timezone: str,
    role: str | None,
    confidence: float | None,
    identity: str | None,
) -> str:
    """Подпись к снимку прохода.

    Имя кота идёт первым, если оно есть: остальное — подробности, а вопрос у
    человека с телефоном в руках ровно один.

    Метка без зоны считается UTC — так её записал recorder, и так же её читает
    админка. Принять её за местное время значило бы прислать проход на три
    часа раньше, чем он был.
    """
    moment = when.replace(tzinfo=UTC) if when.tzinfo is None else when
    camera = {"face": "дверца", "side": "подход"}.get(role or "", role or "камера")
    head = f"{identity}: проход" if identity else "Проход"
    parts = [f"{head} в {moment.astimezone(ZoneInfo(timezone)).strftime('%H:%M')}", camera]
    if confidence is not None:
        parts.append(f"уверенность {confidence:.2f}")
    return " · ".join(parts)
