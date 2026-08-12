"""Открывать ли калитку этому человеку. Чистая функция, без железа и сети.

Решение принимается по **списку факторов**, а не одним условием. Сегодня фактор
один — лицо, завтра к нему добавляется телефон, и добавление обязано быть одной
строкой, а не переписыванием «узнали и открыли». Каждый фактор отвечает «за»,
«против» или «не знаю»; в журнал пишутся все, иначе разбирать отказы
невозможно.

Функция ничего не знает ни о Frigate, ни о базе, ни о плате: событие, настройки
и время на входе — решение и причина на выходе. Поэтому она проверяется
таблицей случаев, а не поездкой к калитке.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, time
from typing import Literal

from src.common.gate_access_config import GateAccessConfig

Verdict = Literal["for", "against", "unknown"]

FACE_FACTOR = "face"
PHONE_FACTOR = "phone"

DECISION_MODE_OFF = "denied_mode_off"
DECISION_NO_GATE = "denied_no_gate"
DECISION_ZONE = "denied_zone"
DECISION_NOT_ALLOWED = "denied_not_allowed"
DECISION_LOW_SCORE = "denied_low_score"
DECISION_WINDOW = "denied_window"
DECISION_COOLDOWN = "denied_cooldown"
DECISION_SECOND_FACTOR = "denied_second_factor"
DECISION_LEAVING = "denied_leaving"
DECISION_NOT_APPROACHING = "denied_not_approaching"
DECISION_MOVEMENT_UNKNOWN = "denied_movement_unknown"
DECISION_DRY_RUN = "dry_run"

MOVEMENT_FACTOR = "movement"


@dataclass(frozen=True)
class Factor:
    """Что один источник знания говорит об этом человеке."""

    name: str
    verdict: Verdict
    detail: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "verdict": self.verdict, "detail": self.detail}


@dataclass(frozen=True)
class Decision:
    decision: str
    reason: str
    factors: tuple[Factor, ...] = ()

    @property
    def would_open(self) -> bool:
        return self.decision == DECISION_DRY_RUN


def _within_window(moment: time, open_from: str, open_to: str) -> bool:
    """Попадает ли местное время в разрешённое окно.

    Окно, заданное «с вечера до утра», переходит через полночь — и наивное
    сравнение `from <= now <= to` закрывало бы его целиком.
    """
    start = time.fromisoformat(open_from)
    end = time.fromisoformat(open_to)
    if start <= end:
        return start <= moment <= end
    return moment >= start or moment <= end


def _movement_verdict(
    config: GateAccessConfig,
    movement: str,
    *,
    gate_just_closed: bool,
) -> tuple[str, str, Factor] | None:
    """Что говорит направление. `None` — возражений нет.

    Правила умеют только запрещать. Единственное послабление — «стоит рядом
    сразу после того, как створка закрылась»: это возвращение, ради которого
    ветка и заведена.
    """
    rules = config.movement
    if not rules.enabled:
        return None
    factor = Factor(MOVEMENT_FACTOR, "unknown", movement)
    if movement == "leaving":
        return (
            DECISION_LEAVING,
            "Человек уходит от калитки",
            Factor(MOVEMENT_FACTOR, "against", "уходит"),
        )
    if movement == "approaching":
        return None
    if movement == "still":
        if rules.open_when_still and gate_just_closed:
            # Створка только что закрылась, а человек стоит перед ней: так
            # выглядит возвращение — вышел, закрыл за собой, развернулся.
            return None
        return (
            DECISION_NOT_APPROACHING,
            "Человек не идёт к калитке, а створка давно не закрывалась",
            Factor(MOVEMENT_FACTOR, "against", "стоит рядом"),
        )
    # Движение ещё неизвестно: событие обновляется раз в несколько секунд.
    # Считать незнание разрешением значило бы открывать на первом же опросе —
    # ровно та ошибка, из-за которой всё и затевалось.
    return (
        DECISION_MOVEMENT_UNKNOWN,
        "Куда идёт человек, ещё не видно",
        factor,
    )


def decide(
    *,
    camera: str,
    person_name: str | None,
    score: float | None,
    zones: Sequence[str],
    config: GateAccessConfig,
    now: datetime,
    last_opened_at: datetime | None = None,
    factors: Sequence[Factor] = (),
    movement: str = "unknown",
    gate_just_closed: bool = False,
) -> Decision:
    """Решение об одном событии. `now` — местное время, с зоной.

    Порядок отсечек выбран так, чтобы причина в журнале была самой полезной из
    возможных: «человек не в зоне» объясняет отказ лучше, чем «не в списке», а
    «узнан неуверенно» лучше, чем «пауза».
    """
    known = tuple(factors)
    if config.mode == "off":
        return Decision(DECISION_MODE_OFF, "Автооткрытие выключено", known)
    gate = config.gate_for(camera)
    if gate is None:
        return Decision(DECISION_NO_GATE, f"Камера {camera} не назначена калиткой", known)
    if gate.zone and gate.zone not in set(zones):
        return Decision(
            DECISION_ZONE,
            f"Человек не в зоне «{gate.zone}»",
            known,
        )
    face = Factor(
        FACE_FACTOR,
        "for" if person_name else "unknown",
        person_name or "лицо не узнано",
    )
    known = (face, *known)
    if person_name is None:
        return Decision(DECISION_NOT_ALLOWED, "Лицо не узнано", known)
    if person_name not in config.allowed_names:
        return Decision(
            DECISION_NOT_ALLOWED,
            f"{person_name} не в списке разрешённых",
            known,
        )
    if score is None or score < config.min_score:
        shown = "нет оценки" if score is None else f"{score:.2f}"
        return Decision(
            DECISION_LOW_SCORE,
            f"Уверенность {shown} ниже порога {config.min_score:.2f}",
            known,
        )
    if not _within_window(now.time(), config.open_from, config.open_to):
        return Decision(
            DECISION_WINDOW,
            f"Вне разрешённых часов {config.open_from}–{config.open_to}",
            known,
        )
    if last_opened_at is not None:
        waited = (now - last_opened_at).total_seconds()
        if 0 <= waited < config.cooldown_seconds:
            return Decision(
                DECISION_COOLDOWN,
                f"Прошло {waited:.0f} с из {config.cooldown_seconds} с паузы",
                known,
            )
    verdict = _movement_verdict(config, movement, gate_just_closed=gate_just_closed)
    if verdict is not None:
        return Decision(verdict[0], verdict[1], (*known, verdict[2]))
    if config.require_second_factor:
        phone = next((item for item in known if item.name == PHONE_FACTOR), None)
        if phone is None or phone.verdict != "for":
            # «Не знаю» — это отказ, а не разрешение: недоступность сети не
            # должна открывать калитку.
            detail = phone.detail if phone is not None else "фактор не проверялся"
            return Decision(
                DECISION_SECOND_FACTOR,
                f"Второй фактор не подтвердил присутствие: {detail}",
                known,
            )
    return Decision(DECISION_DRY_RUN, f"{person_name} — открыл бы", known)
