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
import time
from pathlib import Path

# Сколько ждать, прежде чем судить, поднялся ли контейнер: Frigate валидирует
# конфигурацию уже после старта процесса и на негодной падает через пару секунд.
CONTAINER_SETTLE_SECONDS = 6.0

from src.common.camera_registry import camera_registry_path, load_registry
from src.common.circuits import circuits_path, load_circuits
from src.common.durable_write import write_text_durably
from src.common.frigate_config import (
    build_config,
    build_env,
    foreign_sections,
    frigate_cameras,
)
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


def _foreign_config(settings: Settings) -> str:
    """Чужие разделы из живого файла. Нечитаемый файл — просто нечего переносить."""
    try:
        return foreign_sections(settings.gate_config_path.read_text(encoding="utf-8"))
    except OSError:
        return ""


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
    # Распознавание лиц, номера и классификаторы включаются в интерфейсе самого
    # Frigate и живут в том же файле. Пересобирая его из инвентаря, эти разделы
    # переносим как есть: иначе правка камеры в админке молча стирала бы чужую
    # настройку, а узнавалось бы это через неделю без имён в событиях.
    config = build_config(registry, circuits, preserved=_foreign_config(settings))
    env = build_env(frigate_cameras(registry, circuits), settings.display_timezone)
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
            return _restart(settings, "конфигурация камер изменилась")
        # Юнит «работает» — но это oneshot с RemainAfterExit: контейнер под ним
        # мог умереть, и для systemd это невидимо. Админка показывала бы
        # «Калитка работает» при мёртвом Frigate неделями. Единственный, кто
        # может сверить контейнер с юнитом, — примиритель: он root и приходит
        # по таймеру регулярно.
        alive = _container_running(settings.gate_container_name)
        if alive is False:
            return _restart(settings, "контейнер был мёртв")
        return f"{unit}: уже работает"
    if not wanted:
        failure = set_gate_running(unit, False)
        if failure is not None:
            return f"{unit}: не удалось остановить — {failure}"
        return f"{unit}: остановлена"
    failure = set_gate_running(unit, wanted)
    if failure is not None:
        return f"{unit}: не удалось запустить — {failure}"
    return f"{unit}: запущена{_container_warning(settings)}"


def _restart(settings: Settings, reason: str) -> str:
    """Перезапустить службу и доложить, поднялся ли контейнер на самом деле."""
    unit = settings.gate_service_unit
    failure = set_gate_running(unit, False)
    if failure is None:
        failure = set_gate_running(unit, True)
    if failure is not None:
        return f"{unit}: {reason}, перезапуск не удался — {failure}"
    return f"{unit}: {reason}, служба перезапущена{_container_warning(settings)}"


def _container_warning(settings: Settings) -> str:
    """Хвост к сообщению, если контейнер под живым юнитом так и не поднялся.

    `systemctl start` возвращает успех, как только `docker compose up -d` создал
    контейнер, но Frigate проверяет конфигурацию уже внутри и на негодной
    падает через несколько секунд. Юнит при этом остаётся `active`, и без этой
    проверки примиритель докладывал бы «запущена» о мёртвом контейнере — ровно
    то враньё, от которого сама сверка контейнера и заведена. Пауза даёт
    контейнеру пережить или не пережить валидацию, прежде чем судить.
    """
    _settle()
    if _container_running(settings.gate_container_name) is False:
        return (
            f" — но контейнер не поднялся, смотрите journalctl -u {settings.gate_service_unit}"
            f" и docker logs {settings.gate_container_name}"
        )
    return ""


def _settle() -> None:
    """Пауза, чтобы контейнер успел упасть на валидации, если конфиг негоден."""
    time.sleep(CONTAINER_SETTLE_SECONDS)


def main() -> None:
    print(reconcile(Settings()), file=sys.stderr)


if __name__ == "__main__":
    main()
