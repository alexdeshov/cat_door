"""Открывать ли дверцу этому коту. Чистая функция, без железа и без сети.

Условие владельца, с которого всё началось: **дверца открывается только тогда,
когда опознанный кот появился на камере дверцы.** Не «где-то во дворе» и не «в
записи прохода», а именно перед дверцей и именно сейчас.

Отсюда две особенности, которых нет у калитки.

Во-первых, решение принимается **по живому кадру**, а не по законченному
событию. Опознание прохода в базе появляется, когда проход уже закончился, —
для журнала это нормально, для защёлки поздно: кот к этому времени ушёл или
ждёт под закрытой дверцей.

Во-вторых, роль камеры — не подробность, а само условие. Терраса (`side`)
показывает того же кота, но он там не у дверцы; открыв по ней, мы отпускаем
защёлку неизвестно для кого и на пять секунд раньше, чем нужно.

Функция ничего не знает ни о рекордере, ни о плате: имя, уверенность, роль и
настройки на входе — решение и причина на выходе. Поэтому она проверяется
таблицей случаев.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common.cat_door_config import CatDoorConfig

# Камера дверцы. Вторая роль — терраса, и кот на ней стоит не у дверцы.
DOOR_ROLE = "face"

DECISION_MODE_OFF = "denied_mode_off"
DECISION_WRONG_CAMERA = "denied_wrong_camera"
DECISION_UNKNOWN_CAT = "denied_unknown_cat"
DECISION_NOT_ALLOWED = "denied_not_allowed"
DECISION_LOW_CONFIDENCE = "denied_low_confidence"
DECISION_COOLDOWN = "denied_cooldown"
# Кот подходит по всем условиям, но серия кадров ещё не набрана. Не отказ и не
# открытие: исполнитель копит подтверждения — см. `CatDoorAutomation`.
DECISION_CONFIRMING = "confirming"
DECISION_DRY_RUN = "dry_run"
DECISION_OPEN = "open"


@dataclass(frozen=True)
class CatDoorDecision:
    """Что решено и почему. Причина нужна не меньше решения.

    «Дверца не открылась» без причины разбирается только пересмотром записи, а
    с причиной — взглядом в журнал: не тот кот, не та камера, не хватило
    уверенности или ещё идёт пауза.
    """

    code: str
    reason: str

    @property
    def opens(self) -> bool:
        """Дойдёт ли это решение до защёлки."""
        return self.code == DECISION_OPEN


def decide(
    *,
    name: str | None,
    confidence: float | None,
    role: str,
    config: CatDoorConfig,
    now: float,
    last_opened_at: float | None = None,
) -> CatDoorDecision:
    """Решение по одному кадру.

    Порядок отсечек выбран так, чтобы причина была самой полезной из возможных:
    «не та камера» объясняет отказ лучше, чем «кот не в списке», а «узнан
    неуверенно» лучше, чем «пауза».
    """
    if config.mode == "off":
        return CatDoorDecision(DECISION_MODE_OFF, "Автоматика дверцы выключена")
    if role != DOOR_ROLE:
        return CatDoorDecision(
            DECISION_WRONG_CAMERA,
            f"Кот виден на камере «{role}», а не у дверцы",
        )
    if name is None:
        return CatDoorDecision(DECISION_UNKNOWN_CAT, "Кот не опознан")
    if not config.allows(name):
        # Пустой список — «никого», как и у калитки. Список имён, а не «любой
        # опознанный»: опознание тут удобство, а не пропуск, и открывать
        # незнакомому коту дверцу в дом мы не станем по умолчанию.
        empty = not config.allowed_cats
        return CatDoorDecision(
            DECISION_NOT_ALLOWED,
            "Список своих котов пуст — не открываю никому"
            if empty
            else f"{name} не в списке своих",
        )
    if confidence is None or confidence < config.min_confidence:
        shown = "неизвестна" if confidence is None else f"{confidence:.2f}"
        return CatDoorDecision(
            DECISION_LOW_CONFIDENCE,
            f"{name} узнан неуверенно: {shown} ниже порога {config.min_confidence:.2f}",
        )
    if last_opened_at is not None and config.cooldown_seconds > 0:
        waited = now - last_opened_at
        # Отрицательное — часы шагнули назад после обрыва питания. Тогда ждём
        # полную паузу заново, а не открываем: незнание времени не повод.
        if waited < 0:
            return CatDoorDecision(DECISION_COOLDOWN, "Часы узла сместились — жду паузу заново")
        if waited < config.cooldown_seconds:
            return CatDoorDecision(
                DECISION_COOLDOWN,
                f"Пауза после прошлого открытия: прошло {waited:.0f} с "
                f"из {config.cooldown_seconds} с",
            )
    if config.mode == "dry":
        return CatDoorDecision(DECISION_DRY_RUN, f"{name} — открыл бы дверцу")
    return CatDoorDecision(DECISION_OPEN, f"{name} у дверцы, уверенность {confidence:.2f}")
