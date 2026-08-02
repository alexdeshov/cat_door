"""Управление контуром калитки с пульта узла.

Контур калитки — это служба systemd, которая поднимает Frigate в Docker.
Пульту нужно две вещи: знать, работает ли она, и уметь её включить и выключить.

Состояние берётся у systemd, а не по HTTP у самого Frigate, и это осознанно.
У Frigate есть собственная аутентификация — правильная, в отличие от админки
CatDoor. Ходить к нему за статистикой значило бы держать здесь ещё одни учётные
данные, то есть ухудшить защиту ради цифр, которые и так видны в его
интерфейсе. Вопрос «работает ли контур» systemd отвечает точнее и бесплатно.

Запускать и останавливать службу веб-процесс не может и не должен: в его юните
стоит `NoNewPrivileges=true`, и это правильно — у админки нет аутентификации,
а значит и привилегий у неё быть не должно. Поэтому веб только записывает
желаемое состояние узла, а приводит службу к нему отдельный примиритель от
root (`gate_sync`), следящий за этим файлом.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Literal

GateState = Literal["running", "starting", "stopped", "failed", "absent", "unknown"]

# Пульт ждёт ответа, поэтому команды коротки по времени. Запуск Frigate идёт
# в фоне: systemd возвращает управление, не дожидаясь готовности контейнера.
COMMAND_TIMEOUT_SECONDS = 20.0


@dataclass(frozen=True)
class GateStatus:
    state: GateState
    detail: str = ""

    @property
    def running(self) -> bool:
        """Запускающаяся служба считается работающей.

        Для примирителя это важно: иначе он видел бы «остановлена» во время
        долгого первого запуска и пытался запустить ещё раз.
        """
        return self.state in {"running", "starting"}

    @property
    def deployed(self) -> bool:
        """Развёрнут ли контур вообще.

        Отсутствующая служба и остановленная — разные вещи: первую нельзя
        включить, и предлагать это пользователю значит обещать несбыточное.
        """
        return self.state != "absent"


def _run(command: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 1, ""
    return result.returncode, (result.stdout or "").strip()


def gate_status(unit: str) -> GateStatus:
    """Состояние службы контура калитки.

    `systemctl is-active` не требует прав и отвечает одним словом. Отдельно
    выясняется, существует ли юнит вообще: до развёртывания Frigate его нет, и
    это состояние нужно отличать от остановленного.
    """
    code, output = _run(["systemctl", "is-active", unit])
    if output == "active":
        return GateStatus("running")
    if output == "activating":
        # Первый запуск тянет образ минуты. Показывать это время «остановлена»
        # значило бы подталкивать щёлкать переключателем ещё раз.
        return GateStatus("starting", "служба запускается")
    if output == "failed":
        return GateStatus("failed", "служба завершилась с ошибкой")
    if not output:
        return GateStatus("unknown", "не удалось опросить systemd")
    loaded_code, loaded = _run(["systemctl", "show", unit, "-p", "LoadState", "--value"])
    if loaded_code == 0 and loaded in {"not-found", "masked"}:
        return GateStatus("absent", "служба не установлена на узле")
    del code
    return GateStatus("stopped")


def set_gate_running(unit: str, running: bool) -> str | None:
    """Включить или выключить контур. `None` — получилось.

    Вызывается только примирителем, который работает от root. Из веб-процесса
    эта команда не сработала бы: `NoNewPrivileges` в его юните запрещает любое
    повышение прав, включая `sudo`.
    """
    if running:
        # Запуск не ждёт готовности: первый старт Frigate тянет образ минуты, а
        # блокирующий вызов обрезался бы нашим же таймаутом и лёг в журнал
        # ложным отказом. Сошёлся ли запуск, выяснит следующий прогон
        # примирителя — по таймеру он приходит регулярно.
        code, output = _run(["systemctl", "start", "--no-block", unit])
    else:
        code, output = _run(["systemctl", "stop", unit])
    if code == 0:
        return None
    action = "start" if running else "stop"
    return output or f"не удалось выполнить systemctl {action} {unit}"
