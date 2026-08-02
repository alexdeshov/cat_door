"""Примиритель: приводит службу контура калитки к состоянию узла.

Запускается от root по изменению `runtime/circuits.json` и один раз при
загрузке. Веб-служба этот файл только пишет — привилегий у неё нет и не должно
быть: у админки нет аутентификации, а `NoNewPrivileges` в её юните запрещает
любое повышение прав.

Разделение на «кто решает» и «кто исполняет» даёт ещё одно свойство даром:
после перезагрузки состояние применяется само. Контейнер Frigate объявлен с
`restart: "no"`, а его служба не в автозапуске — значит без примирителя узел,
отданный калитке, поднялся бы с остановленным контуром.
"""

from __future__ import annotations

import sys

from src.common.circuits import circuits_path, load_circuits
from src.common.gate_service import gate_status, set_gate_running
from src.common.settings import Settings


def reconcile(settings: Settings) -> str:
    """Привести службу к записанному состоянию. Вернуть строку для журнала."""
    circuits = load_circuits(circuits_path(settings.data_dir))
    unit = settings.gate_service_unit
    status = gate_status(unit)
    if not status.deployed:
        return f"{unit}: не установлена, приводить нечего"
    if status.running == circuits.gate:
        return f"{unit}: уже {'работает' if circuits.gate else 'остановлена'}"
    failure = set_gate_running(unit, circuits.gate)
    if failure is not None:
        return f"{unit}: не удалось привести к состоянию — {failure}"
    return f"{unit}: {'запущена' if circuits.gate else 'остановлена'}"


def main() -> None:
    print(reconcile(Settings()), file=sys.stderr)


if __name__ == "__main__":
    main()
