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


def _unchanged(path: Path, text: str) -> bool:
    try:
        return path.read_text(encoding="utf-8") == text
    except OSError:
        return False


def apply_frigate_config(settings: Settings) -> bool:
    """Собрать конфигурацию Frigate из инвентаря. Вернуть, изменилась ли она.

    Пишет примиритель, а не веб-служба: у той нет привилегий на каталоги
    Frigate, и давать их нельзя — у админки нет аутентификации.

    Сравнение идёт с копией прошлой сборки, а не с рабочим файлом. Frigate
    переписывает свою конфигурацию сам — мигрирует между версиями и хранит
    рядом `backup_config.yaml`, — и комментарии при этом не переживают
    round-trip. Сравнение с живым файлом после первой же миграции расходилось бы
    навсегда, и примиритель перезапускал бы контур каждые несколько минут.
    Пока инвентарь не менялся, миграция Frigate остаётся нетронутой.
    """
    registry = load_registry(camera_registry_path(settings.data_dir))
    if registry is None:
        return False
    circuits = load_circuits(circuits_path(settings.data_dir))
    config = build_config(registry, circuits)
    env = build_env(frigate_cameras(registry, circuits))
    if _unchanged(settings.gate_applied_config_path, config) and _unchanged(
        settings.gate_env_path, env
    ):
        return False
    write_text_durably(settings.gate_config_path, config, mode=0o644)
    write_text_durably(settings.gate_applied_config_path, config, mode=0o600)
    # Пароли камер: файл читает только root и Docker как env_file.
    write_text_durably(settings.gate_env_path, env, mode=0o600)
    return True


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
