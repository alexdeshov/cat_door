"""Приёмка узла: тот же эталон, что и на странице состояния, плюс тяжёлое.

Страница админки отвечает на вопрос «как сейчас» и обязана открываться мгновенно.
Здесь то же самое плюс проверки, которые стоят секунд и потому на странице не
живут: службы и их автозапуск, тома, две планки памяти, совпадение кода с
`origin/main` и тесты проекта прямо на узле.

Эталон намеренно **не дублируется**: он импортируется из `src/common/health.py`.
Два списка контрольных чисел разошлись бы в первый же день, и разошлись бы молча.

Запускать на узле от `catdoor` (нужен доступ к данным) или от root — часть
проверок читает `dmidecode` и состояние служб.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from src.common.health import (
    EXPECTED_MEMORY_TOTAL_GB,
    Check,
    build_health,
    overall_status,
)
from src.common.node_metrics import read_node_metrics
from src.common.settings import Settings
from src.recorder.status import read_recorder_status, recorder_status_path

# Службы, без которых узел не работает, и их ожидаемое состояние автозапуска.
# `frigate` намеренно `disabled`: ею владеет примиритель, он же поднимает её
# после загрузки. Требовать `enabled` значило бы проверять не ту гарантию —
# 2026-08-09 эта проверка так и провалилась, будучи неверной сама.
REQUIRED_UNITS = ("catdoor-recorder", "catdoor-web", "docker", "dnsmasq", "chrony")
REQUIRED_TIMERS = ("catdoor-gate-sync.timer", "catdoor-gate-sync.path", "catdoor-backup.timer")
RECONCILER_OWNED = ("frigate",)


def load_env_file(path: Path) -> int:
    """Подложить окружение узла до создания `Settings`.

    Службам его подсовывает systemd (`EnvironmentFile=`), а запущенному руками
    скрипту — никто. Без этого `Settings()` поднимается на умолчаниях, каталог
    данных становится относительным `data`, и приёмка честно сверяет эталон с
    пустотой. Первый же прогон на узле так и сделал.

    Уже заданные переменные не трогаются: явный запуск с другим окружением
    должен побеждать файл.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return 0
    loaded = 0
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


def _run(command: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return 1, ""
    return result.returncode, (result.stdout or "").strip()


def _api(url: str, timeout: float = 5.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as answer:
            return json.loads(answer.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None


def service_checks() -> list[Check]:
    checks: list[Check] = []
    for unit in REQUIRED_UNITS:
        _, active = _run(["systemctl", "is-active", unit])
        _, enabled = _run(["systemctl", "is-enabled", unit])
        good = active == "active" and enabled in {"enabled", "static", "enabled-runtime"}
        checks.append(
            Check(
                f"unit_{unit}",
                f"Служба {unit}",
                "active + enabled",
                f"{active or '—'} + {enabled or '—'}",
                "ok" if good else "fail",
                "Служба без автозапуска не переживёт перезагрузку.",
            )
        )
    for unit in REQUIRED_TIMERS:
        _, enabled = _run(["systemctl", "is-enabled", unit])
        checks.append(
            Check(
                f"unit_{unit}",
                f"Таймер {unit}",
                "enabled",
                enabled or "—",
                "ok" if enabled == "enabled" else "fail",
            )
        )
    for unit in RECONCILER_OWNED:
        _, active = _run(["systemctl", "is-active", unit])
        checks.append(
            Check(
                f"unit_{unit}",
                f"Служба {unit}",
                "active (автозапуском владеет примиритель)",
                active or "—",
                "ok" if active == "active" else "fail",
            )
        )
        _, boot = _run(["uptime", "-s"])
        code, journal = _run(
            ["journalctl", "-u", "catdoor-gate-sync", "--since", boot, "--no-pager"]
        )
        started = code == 0 and "frigate.service: запущена" in journal
        checks.append(
            Check(
                "gate_autostart",
                "Примиритель поднял Frigate после загрузки",
                "есть в журнале",
                "есть" if started else "нет",
                "ok" if started else "fail",
                "Контур калитки поднимается примирителем, а не systemd — это и надо проверять.",
            )
        )
    _, failed = _run(["systemctl", "--failed", "--no-legend"])
    count = len([line for line in failed.splitlines() if line.strip()])
    checks.append(
        Check(
            "units_failed",
            "Отказавших юнитов",
            "0",
            str(count),
            "ok" if count == 0 else "fail",
        )
    )
    return checks


def volume_checks(paths: list[Path]) -> list[Check]:
    checks: list[Check] = []
    for path in paths:
        exists = path.is_dir()
        free = "—"
        if exists:
            try:
                free = f"{shutil.disk_usage(path).free / 1024**3:.0f} ГБ свободно"
            except OSError:
                exists = False
        checks.append(
            Check(
                f"volume_{path}",
                f"Том {path}",
                "доступен",
                free if exists else "нет",
                "ok" if exists else "fail",
            )
        )
    return checks


def memory_channel_check() -> Check:
    """Две планки — единственный прямой признак двухканального режима.

    Косвенный (имя `Iris Xe` вместо `UHD`) проверяется вместе с остальным
    железом на странице; здесь именно железо, и спрашивается оно у `dmidecode`.
    """
    code, output = _run(["dmidecode", "-t", "memory"])
    if code != 0:
        return Check(
            "memory_modules",
            "Планок памяти",
            "2",
            "не спросить без root",
            "unknown",
        )
    modules = [
        line for line in output.splitlines() if line.strip().startswith("Size:") and "GB" in line
    ]
    return Check(
        "memory_modules",
        "Планок памяти",
        "2",
        str(len(modules)),
        "ok" if len(modules) == 2 else "fail",
        f"Одна планка — одноканальный режим и полоса вдвое ниже "
        f"(ожидается не меньше {EXPECTED_MEMORY_TOTAL_GB} ГБ суммарно).",
    )


def repository_checks(root: Path) -> list[Check]:
    _, _ = _run(["git", "-C", str(root), "fetch", "-q", "origin"])
    _, local = _run(["git", "-C", str(root), "rev-parse", "HEAD"])
    _, remote = _run(["git", "-C", str(root), "rev-parse", "origin/main"])
    _, dirty = _run(["git", "-C", str(root), "status", "--porcelain"])
    checks = [
        Check(
            "code_matches_main",
            "Код узла",
            "совпадает с origin/main",
            "совпадает" if local and local == remote else f"{local[:8]} против {remote[:8]}",
            "ok" if local and local == remote else "fail",
        ),
        Check(
            "code_clean",
            "Рабочее дерево",
            "без правок руками",
            "чистое" if not dirty else "есть правки",
            "ok" if not dirty else "fail",
            "Правка руками переживёт до первого обновления, а потом исчезнет молча.",
        ),
    ]
    return checks


def test_suite_check(root: Path, python: str) -> Check:
    """Тесты прямо на узле. Зелёные на маке не доказывают ничего про узел."""
    code, output = _run([python, "-m", "pytest", "-q"])
    last = output.splitlines()[-1] if output.splitlines() else "нет вывода"
    return Check(
        "tests",
        "Тесты проекта на узле",
        "все проходят",
        last,
        "ok" if code == 0 else "fail",
    )


def measured_rate_check(api_url: str, seconds: float) -> list[Check]:
    """Темп кадров живым замером, а не из статуса.

    Recorder публикует собственную оценку скользящим окном, и обычно её
    достаточно. Здесь считается по счётчикам за окно замера — независимо от
    того, что recorder о себе думает.
    """
    first = _api(f"{api_url}/api/recorder/status")
    if first is None:
        return [Check("rate_measured", "Темп кадров", "замер", "recorder не отвечает", "fail")]
    time.sleep(seconds)
    second = _api(f"{api_url}/api/recorder/status")
    if second is None:
        return [Check("rate_measured", "Темп кадров", "замер", "recorder пропал", "fail")]
    from src.common.health import MIN_SAMPLE_FPS

    checks: list[Check] = []
    for role, floor in MIN_SAMPLE_FPS.items():
        before = (first.get("frames_processed") or {}).get(role)
        after = (second.get("frames_processed") or {}).get(role)
        if before is None or after is None:
            checks.append(
                Check(f"rate_measured_{role}", f"Темп {role}", f"≥ {floor}", "—", "unknown")
            )
            continue
        rate = round((after - before) / seconds, 1)
        checks.append(
            Check(
                f"rate_measured_{role}",
                f"Темп {role}, замер {seconds:.0f} с",
                f"≥ {floor} к/с",
                f"{rate} к/с",
                "ok" if rate >= floor else "fail",
            )
        )
    return checks


def report(title: str, checks: list[Check]) -> tuple[int, int]:
    print(f"\n=== {title} ===")
    good = bad = 0
    for check in checks:
        mark = {"ok": "  ok     ", "warn": "  внимание", "fail": "  ПРОВАЛ ", "unknown": "  ?      "}
        detail = f"{check.title}: {check.actual}"
        if check.status != "ok":
            detail += f" (эталон: {check.expected})"
        print(f"{mark[check.status]} {detail}")
        if check.status == "ok":
            good += 1
        elif check.status != "unknown":
            bad += 1
    return good, bad


def main() -> int:
    parser = argparse.ArgumentParser(description="Приёмка рабочей конфигурации узла CatDoor")
    parser.add_argument("--root", type=Path, default=Path("/opt/catdoor"))
    parser.add_argument("--python", default="/opt/catdoor/.venv/bin/python")
    parser.add_argument("--api", default="http://127.0.0.1:8080")
    parser.add_argument("--rate-seconds", type=float, default=20.0)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path("/etc/catdoor/catdoor.env"),
        help="окружение узла; без него Settings поднимется на умолчаниях",
    )
    arguments = parser.parse_args()

    loaded = load_env_file(arguments.env_file)
    settings = Settings()
    print(f"окружение: {arguments.env_file} ({loaded} переменных), данные: {settings.data_dir}")
    status = read_recorder_status(recorder_status_path(settings)) or {}
    stats = _api(f"{settings.gate_api_url.rstrip('/')}/api/stats")
    metrics = read_node_metrics(
        [settings.data_dir, Path("/srv/data"), settings.gate_recordings_path]
    )
    detection = _api(f"{arguments.api}/api/detection/settings") or {}

    groups = build_health(
        recorder_status=status,
        detection_config=detection,
        frigate_stats=stats,
        gate_detector_device=settings.gate_detector_device,
        metrics=metrics,
        gpu_name=str(status.get("detector_device_name") or "") or None,
    )

    good = bad = 0
    for group in groups:
        a, b = report(group.title, group.checks)
        good, bad = good + a, bad + b
    for title, checks in (
        ("Службы", service_checks()),
        ("Тома", volume_checks([Path("/srv/data"), Path("/srv/archive"), settings.data_dir])),
        ("Память", [memory_channel_check()]),
        ("Живой замер темпа", measured_rate_check(arguments.api, arguments.rate_seconds)),
        ("Код узла", repository_checks(arguments.root)),
    ):
        a, b = report(title, checks)
        good, bad = good + a, bad + b
    if not arguments.skip_tests:
        a, b = report("Тесты", [test_suite_check(arguments.root, arguments.python)])
        good, bad = good + a, bad + b

    print(f"\nИТОГО: пройдено {good}, провалено {bad}")
    print(f"сводный статус страницы состояния: {overall_status(groups)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
