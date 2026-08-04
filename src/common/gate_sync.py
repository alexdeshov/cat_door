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
from pathlib import Path

from src.common.camera_registry import camera_registry_path, load_registry
from src.common.circuits import circuits_path, load_circuits
from src.common.durable_write import write_text_durably
from src.common.frigate_config import build_config, build_env, frigate_cameras
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


def _write_if_changed(path: Path, text: str, *, mode: int) -> bool:
    """Записать файл, если он отличается. Вернуть, была ли запись.

    Сравнение обязательно: перезапись одинакового содержимого сама по себе
    безвредна, но примиритель приходит по таймеру, и «файл изменился» —
    единственный признак, по которому он решает перезапустить Frigate. Без
    сравнения контур калитки перезапускался бы каждые несколько минут.
    """
    try:
        if path.read_text(encoding="utf-8") == text:
            return False
    except OSError:
        pass
    write_text_durably(path, text, mode=mode)
    return True


def apply_frigate_config(settings: Settings) -> bool:
    """Собрать конфигурацию Frigate из инвентаря. Вернуть, изменилась ли она.

    Пишет примиритель, а не веб-служба: у той нет привилегий на `/etc/frigate`,
    и давать их нельзя — у админки нет аутентификации.
    """
    registry = load_registry(camera_registry_path(settings.data_dir))
    if registry is None:
        return False
    circuits = load_circuits(circuits_path(settings.data_dir))
    config_changed = _write_if_changed(
        settings.gate_config_path,
        build_config(registry, circuits),
        mode=0o644,
    )
    # Пароли камер: файл читает только root и сам контейнер.
    env_changed = _write_if_changed(
        settings.gate_env_path,
        build_env(frigate_cameras(registry, circuits)),
        mode=0o600,
    )
    return config_changed or env_changed


def reconcile(settings: Settings) -> str:
    """Привести службу к записанному состоянию. Вернуть строку для журнала."""
    circuits = load_circuits(circuits_path(settings.data_dir))
    # Frigate обслуживает и калитку, и регистратор: он нужен, пока работает
    # хоть один из них.
    wanted = circuits.frigate_wanted
    unit = settings.gate_service_unit
    status = gate_status(unit)
    if not status.deployed:
        return f"{unit}: не установлена, приводить нечего"
    if status.state == "starting":
        return f"{unit}: запускается, ждём"
    config_changed = apply_frigate_config(settings)
    if status.running == wanted:
        if not wanted:
            return f"{unit}: уже остановлена"
        if config_changed:
            # Камеру перевели в другой контур или поправили её адрес. Без
            # перезапуска Frigate продолжал бы снимать по старому списку, и
            # перемещение камеры в админке ничего бы не значило.
            failure = set_gate_running(unit, False)
            if failure is None:
                failure = set_gate_running(unit, True)
            if failure is not None:
                return f"{unit}: конфигурация обновлена, перезапуск не удался — {failure}"
            return f"{unit}: конфигурация камер изменилась, служба перезапущена"
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
    failure = set_gate_running(unit, wanted)
    if failure is not None:
        return f"{unit}: не удалось привести к состоянию — {failure}"
    return f"{unit}: {'запущена' if wanted else 'остановлена'}"


def main() -> None:
    print(reconcile(Settings()), file=sys.stderr)


if __name__ == "__main__":
    main()
