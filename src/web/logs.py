"""Журналы служб для админки.

До 31 августа увидеть, что пишет служба, можно было только по ssh: `journalctl`
на узле, `detection.jsonl` — через владельца, потому что каталог данных читает
только служба. Разбор любой беды начинался с «дай мне журнал». Здесь тот же
хвост журнала отдаётся страницей.

Устройство нарочно простое: один вызов `journalctl -u <юнит>` на запрос, без
слежения и без потоков. Страница — для разбора уже случившегося, а не для
наблюдения в реальном времени; за живым состоянием смотрит сторож.

Источники перечислены явно, и тест сверяет список с каталогом `install/`: юнит
без строки здесь — служба, чей журнал придётся снова добывать по ssh.
"""

from __future__ import annotations

import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LINES = 200
MAX_LINES = 2000
# Журнал за раз читается недолго: страница блокирует один воркер, и минутная
# выборка из journald при живом узле делала бы админку недоступной.
TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class LogSource:
    key: str
    title: str
    # Юнит systemd или путь файла относительно каталога данных — одно из двух.
    unit: str = ""
    file: str = ""
    hint: str = ""


SOURCES: tuple[LogSource, ...] = (
    LogSource("recorder", "Recorder — коты", unit="catdoor-recorder",
              hint="Детекция и запись проходов. Почти пуст в норме: решения — в «Решения recorder»."),
    LogSource("web", "Админка", unit="catdoor-web",
              hint="Запросы к страницам и API, фоновые работы, запущенные отсюда."),
    LogSource("gate-access", "Калитка — доступ", unit="catdoor-gate-access",
              hint="Решения по лицам и импульсы. Журнал проходов — на странице калитки, здесь — служебное."),
    LogSource("gate-notify", "Калитка — уведомления", unit="catdoor-gate-notify"),
    LogSource("gate-sync", "Примиритель Frigate", unit="catdoor-gate-sync",
              hint="Раз в две минуты по таймеру: «уже работает» — норма, «перезапущена» — конфигурация камер менялась."),
    LogSource("cloud-backup", "Облачная копия — калитка", unit="catdoor-cloud-backup"),
    LogSource("cloud-backup-cats", "Облачная копия — коты", unit="catdoor-cloud-backup-cats"),
    LogSource("healthwatch", "Сторож здоровья", unit="catdoor-healthwatch"),
    LogSource("backup", "Копия на внешний диск", unit="catdoor-backup",
              hint="По таймеру в 05:30. Строка «подрезка старше N дн» — работа окна хранения."),
    LogSource("cpu-thermal", "Запрет турбо-частот", unit="catdoor-cpu-thermal",
              hint="Выключен по замыслу: журнал пуст, пока не включат установщиком."),
    LogSource("frigate", "Frigate", unit="frigate",
              hint="Только строки службы; подробный журнал Frigate — в его интерфейсе."),
    LogSource("dnsmasq", "DHCP камер", unit="dnsmasq",
              hint="Аренды и запросы камер. Камера, просящая адрес каждые секунды, — отвергнутая аренда."),
    LogSource("wifi-keepalive", "Заплатка Wi-Fi", unit="wifi-keepalive"),
    LogSource("detection", "Решения recorder", file="runtime/detection.jsonl",
              hint="Каждое решение автомата событий строкой JSON: кандидат, подтверждение, уведомление. "
                   "Раньше доставался только через владельца."),
)

SOURCE_BY_KEY = {source.key: source for source in SOURCES}


def clamp_lines(value: int | None) -> int:
    if value is None:
        return DEFAULT_LINES
    return max(1, min(MAX_LINES, int(value)))


def read_unit_log(unit: str, lines: int) -> tuple[str, str | None]:
    """Хвост журнала юнита и причина, если прочитать не вышло.

    Пустой журнал и недоступный журнал — разные ответы: первый значит «служба
    молчит», второй — «службе не дали читать». Путать их нельзя: разбор беды
    по пустому экрану ушёл бы не туда.
    """
    command = [
        "journalctl", "-u", unit, "-n", str(lines), "--no-pager", "-o", "short-iso",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=TIMEOUT_SECONDS, check=False
        )
    except FileNotFoundError:
        return "", "journalctl не найден"
    except subprocess.TimeoutExpired:
        return "", f"journalctl не ответил за {TIMEOUT_SECONDS:.0f} с"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        lowered = detail.lower()
        if any(sign in lowered for sign in ("no journal files", "insufficient permissions", "not permitted")):
            detail = (
                "службе не дали читать журнал: пользователь catdoor должен быть в группе "
                "systemd-journal (делает install/install.sh), после — перезапуск catdoor-web"
            )
        return "", detail or f"journalctl вернул код {result.returncode}"
    text = result.stdout
    if text.startswith("-- No entries --"):
        return "", None
    return text, None


def read_file_tail(path: Path, lines: int) -> tuple[str, str | None]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            tail = deque(handle, maxlen=lines)
    except FileNotFoundError:
        return "", "файла ещё нет — служба ничего не решала"
    except OSError as exc:
        return "", f"не прочитать: {exc}"
    return "".join(tail), None


def read_log(source: LogSource, data_dir: Path, lines: int) -> tuple[str, str | None]:
    if source.unit:
        return read_unit_log(source.unit, lines)
    return read_file_tail(data_dir / source.file, lines)
