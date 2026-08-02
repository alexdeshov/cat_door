"""Какие контуры узла работают прямо сейчас.

На узле два контура: коты у дверцы (этот репозиторий) и люди у калитки
(`gate_access`, Frigate). Процессор один, встроенная графика на текущем железе
сломана драйвером, поэтому считает всё он. Одновременная работа не «замедляет
немного», а деградирует оба контура незаметно: кадры стареют, проходы
пропускаются, и узнаётся это из пустого журнала через неделю.

Отсюда признак `exclusive`. Он не архитектурное украшение, а запись факта про
конкретное железо, и снимается одной строкой после переезда — интерфейс и
модель состояния переделывать не придётся.

Режим контура котов живёт отдельно, в `capture_mode`: там сказано, *как* он
работает, здесь — *разрешено* ли ему работать вообще. Разделение нужно, чтобы
выключение с уровня узла не стирало выбранный режим наблюдения или записи.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from src.common.durable_write import write_text_durably

CircuitName = Literal["cats", "gate"]
GateRecording = Literal["events", "continuous"]

CIRCUIT_TITLES: dict[CircuitName, str] = {
    "cats": "Дверца",
    "gate": "Калитка 1",
}


class CircuitConflict(ValueError):
    """Запрошены оба контура там, где узел тянет только один."""


class NodeCircuits(BaseModel):
    cats: bool = True
    gate: bool = False
    exclusive: bool = True
    # Круглосуточная запись заполняет диск предсказуемо и быстро, а для сбора
    # библиотеки лиц достаточно клипов вокруг событий. Значение попадает в
    # конфигурацию Frigate при развёртывании контура.
    gate_recording: GateRecording = "events"

    @property
    def has_conflict(self) -> bool:
        return self.exclusive and self.cats and self.gate

    def resolved(self) -> NodeCircuits:
        """Обезвредить конфликт в правленом руками файле.

        Отвергать файл целиком нельзя: тогда опечатка останавливала бы запись
        проходов. Побеждает кошачий контур — он работал до появления второго и
        не требует внешней службы.
        """
        return self.model_copy(update={"gate": False}) if self.has_conflict else self

    def is_active(self, circuit: CircuitName) -> bool:
        return self.cats if circuit == "cats" else self.gate

    def with_only(self, circuit: CircuitName) -> NodeCircuits:
        """Оставить работать один контур. Основное действие переключателя."""
        return self.model_copy(update={"cats": circuit == "cats", "gate": circuit == "gate"})

    @property
    def active_names(self) -> tuple[CircuitName, ...]:
        return tuple(name for name in ("cats", "gate") if self.is_active(name))


def ensure_allowed(circuits: NodeCircuits) -> None:
    """Проверка вынесена из валидатора модели намеренно.

    Валидатор pydantic заворачивает своё исключение в `ValidationError`, и
    вызывающий код не смог бы отличить «узел не тянет оба контура» от опечатки
    в типе поля — а это разные ответы: осмысленный отказ и ошибка запроса.
    """
    if circuits.has_conflict:
        raise CircuitConflict(
            "Узел не тянет оба контура сразу: включите один или снимите "
            "ограничение после переезда на новое железо"
        )


def circuits_path(data_dir: Path) -> Path:
    return data_dir / "runtime" / "circuits.json"


def load_circuits(path: Path, defaults: NodeCircuits | None = None) -> NodeCircuits:
    """Прочитать состояние узла. Нечитаемый файл не меняет поведения.

    Умолчание — кошачий контур разрешён: это то, что работало до появления
    второго контура, и неожиданная остановка записи из-за испорченного файла
    была бы худшим исходом, чем игнорирование файла.
    """
    fallback = defaults or NodeCircuits()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback
    try:
        return NodeCircuits.model_validate(payload).resolved()
    except ValueError:
        return fallback


def save_circuits(path: Path, circuits: NodeCircuits) -> None:
    write_text_durably(path, json.dumps(circuits.model_dump(), ensure_ascii=False, indent=2))
