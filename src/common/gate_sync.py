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

import subprocess
import sys

from src.common.circuits import circuits_path, load_circuits
from src.common.gate_service import gate_status, set_gate_running
from src.common.settings import Settings


def _container_running(name: str) -> bool | None:
    """Жив ли контейнер. `None` — Docker не ответил, судить не по чему."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", name],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        # Контейнера нет вовсе — для вопроса «жив ли» это твёрдое «нет».
        return False
    return result.stdout.strip() == "true"


def reconcile(settings: Settings) -> str:
    """Привести службу к записанному состоянию. Вернуть строку для журнала."""
    circuits = load_circuits(circuits_path(settings.data_dir))
    unit = settings.gate_service_unit
    status = gate_status(unit)
    if not status.deployed:
        return f"{unit}: не установлена, приводить нечего"
    if status.state == "starting":
        return f"{unit}: запускается, ждём"
    if status.running == circuits.gate:
        if not circuits.gate:
            return f"{unit}: уже остановлена"
        # Юнит «работает» — но это oneshot с RemainAfterExit: контейнер под ним
        # мог умереть, и для systemd это невидимо. Админка показывала бы
        # «Калитка работает» при мёртвом Frigate неделями. Единственный, кто
        # может сверить контейнер с юнитом, — примиритель: он root и приходит
        # по таймеру регулярно.
        alive = _container_running(settings.gate_container_name)
        if alive is False:
            failure = set_gate_running(unit, False)
            if failure is not None:
                return f"{unit}: контейнер мёртв, остановить не удалось — {failure}"
            failure = set_gate_running(unit, True)
            if failure is not None:
                return f"{unit}: контейнер мёртв, перезапустить не удалось — {failure}"
            return f"{unit}: контейнер был мёртв, служба перезапущена"
        return f"{unit}: уже работает"
    failure = set_gate_running(unit, circuits.gate)
    if failure is not None:
        return f"{unit}: не удалось привести к состоянию — {failure}"
    return f"{unit}: {'запущена' if circuits.gate else 'остановлена'}"


def main() -> None:
    print(reconcile(Settings()), file=sys.stderr)


if __name__ == "__main__":
    main()
