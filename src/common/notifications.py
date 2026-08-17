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
    # Порог уверенности для сообщения. Он выше порога записи намеренно: писать
    # стоит всё сомнительное — потом разберём, — а будить человека только тем,
    # в чём детектор уверен. На дверце при 0.8 остаются 9 из 10 настоящих
    # проходов, а тени отсекаются почти целиком.
    min_confidence: float = 0.0
    _last_sent_at: float | None = field(default=None, repr=False)

    def blocked_reason(
        self,
        role: str | None,
        now: float,
        confidence: float | None = None,
    ) -> str | None:
        """`None` — отправляем. Иначе причина молчания, годная для журнала."""
        if not self.roles:
            return "notifications_disabled"
        if role is None:
            return "unknown_camera"
        if role not in self.roles:
            return f"camera_not_notified:{role}"
        if confidence is not None and confidence < self.min_confidence:
            return f"below_notify_threshold:{confidence:.2f}"
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
    detector_confidence: float | None,
    identity: str | None,
    identity_confidence: float | None = None,
    votes: int | None = None,
    frames_used: int | None = None,
) -> str:
    """Подпись к снимку прохода.

    Имя кота идёт первым, если оно есть: остальное — подробности, а вопрос у
    человека с телефоном в руках ровно один.

    **Рядом с именем стоит уверенность опознания, а не детектора.** Раньше
    стояла детекторная — «в кадре точно кот», — и читалась как «это точно
    Яся». 17 августа чужой рыжий кот приехал в телеграм как «Яся · уверенность
    0.94», где 0.94 значило всего лишь, что перед камерой кот. Безымянный
    проход детекторную уверенность сохраняет, но теперь она подписана словом
    «кот»: это единственное, о чём она и говорит.

    Кворум печатается вместе с именем: «56 из 154 кадров» отличает уверенное
    опознание от собранного по нескольким удачным кадрам, и отличает его до
    того, как человек полезет смотреть запись.

    Метка без зоны считается UTC — так её записал recorder, и так же её читает
    админка. Принять её за местное время значило бы прислать проход на три
    часа раньше, чем он был.
    """
    moment = when.replace(tzinfo=UTC) if when.tzinfo is None else when
    camera = {"face": "дверца", "side": "подход"}.get(role or "", role or "камера")
    head = f"{identity}: проход" if identity else "Проход"
    parts = [f"{head} в {moment.astimezone(ZoneInfo(timezone)).strftime('%H:%M')}", camera]
    if identity is not None:
        if identity_confidence is not None:
            parts.append(f"опознание {identity_confidence:.2f}")
        if votes is not None and frames_used:
            parts.append(f"{votes} из {frames_used} кадров")
    elif detector_confidence is not None:
        parts.append(f"кот {detector_confidence:.2f}")
    return " · ".join(parts)
