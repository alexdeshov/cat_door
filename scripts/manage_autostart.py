from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASKS = {
    "CatDoorWeb": "web",
    "CatDoorRecorderRecording": "recorder-recording",
}
LEGACY_VALUES = ("CatDoorRecorderDryRun", "CatDoorAdminChrome")
RUN_KEY = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run"
WATCHDOG_TASK_NAME = "CatDoorWatchdog"


def pythonw_path() -> Path:
    candidate = PROJECT_ROOT / ".venv" / "Scripts" / "pythonw.exe"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Не найден {candidate}. Сначала создайте виртуальное окружение проекта."
        )
    return candidate


def task_command(component: str) -> str:
    executable = pythonw_path()
    entry = PROJECT_ROOT / "scripts" / "autostart_entry.py"
    executable_text = f'"{executable}"' if " " in str(executable) else str(executable)
    entry_text = f'"{entry}"' if " " in str(entry) else str(entry)
    return f"{executable_text} {entry_text} {component}"


def watchdog_task_command() -> str:
    executable = pythonw_path()
    entry = PROJECT_ROOT / "scripts" / "watchdog.py"
    executable_text = f'"{executable}"' if " " in str(executable) else str(executable)
    entry_text = f'"{entry}"' if " " in str(entry) else str(entry)
    return f"{executable_text} {entry_text}"


def run_registry(arguments: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["reg.exe", *arguments],
        check=check,
        text=True,
        capture_output=True,
    )


def run_scheduler(
    arguments: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["schtasks.exe", *arguments],
        check=check,
        text=True,
        capture_output=True,
    )


def install() -> None:
    for value_name in LEGACY_VALUES:
        run_registry(["DELETE", RUN_KEY, "/V", value_name, "/F"], check=False)
    for value_name, component in TASKS.items():
        run_registry(
            ["ADD", RUN_KEY, "/V", value_name, "/T", "REG_SZ", "/D", task_command(component), "/F"]
        )
        print(f"Установлено: {value_name}")
    run_scheduler(
        [
            "/Create",
            "/TN",
            WATCHDOG_TASK_NAME,
            "/SC",
            "MINUTE",
            "/MO",
            "5",
            "/TR",
            watchdog_task_command(),
            "/RL",
            "LIMITED",
            "/F",
        ]
    )
    print(f"Установлено: {WATCHDOG_TASK_NAME}")


def uninstall() -> None:
    for value_name in (*TASKS, *LEGACY_VALUES):
        result = run_registry(["DELETE", RUN_KEY, "/V", value_name, "/F"], check=False)
        if result.returncode == 0:
            print(f"Удалено: {value_name}")
        else:
            print(f"Не найдено: {value_name}")
    result = run_scheduler(
        ["/Delete", "/TN", WATCHDOG_TASK_NAME, "/F"],
        check=False,
    )
    if result.returncode == 0:
        print(f"Удалено: {WATCHDOG_TASK_NAME}")
    else:
        print(f"Не найдено: {WATCHDOG_TASK_NAME}")


def status() -> int:
    missing = False
    for value_name in TASKS:
        result = run_registry(["QUERY", RUN_KEY, "/V", value_name], check=False)
        installed = result.returncode == 0
        print(f"{value_name}: {'установлено' if installed else 'не установлено'}")
        missing = missing or not installed
    watchdog = run_scheduler(
        ["/Query", "/TN", WATCHDOG_TASK_NAME],
        check=False,
    )
    watchdog_installed = watchdog.returncode == 0
    print(
        f"{WATCHDOG_TASK_NAME}: "
        f"{'установлено' if watchdog_installed else 'не установлено'}"
    )
    missing = missing or not watchdog_installed
    return 1 if missing else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage CatDoor Windows autostart")
    parser.add_argument("action", choices=("install", "uninstall", "status"))
    arguments = parser.parse_args()

    if arguments.action == "install":
        install()
    elif arguments.action == "uninstall":
        uninstall()
    else:
        raise SystemExit(status())


if __name__ == "__main__":
    main()
