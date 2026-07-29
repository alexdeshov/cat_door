from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASKS = {
    "CatDoorWeb": "web",
    "CatDoorRecorderRecording": "recorder-recording",
    "CatDoorAdminChrome": "browser",
}
LEGACY_VALUES = ("CatDoorRecorderDryRun",)
RUN_KEY = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run"


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


def run_registry(arguments: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["reg.exe", *arguments],
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


def uninstall() -> None:
    for value_name in (*TASKS, *LEGACY_VALUES):
        result = run_registry(["DELETE", RUN_KEY, "/V", value_name, "/F"], check=False)
        if result.returncode == 0:
            print(f"Удалено: {value_name}")
        else:
            print(f"Не найдено: {value_name}")


def status() -> int:
    missing = False
    for value_name in TASKS:
        result = run_registry(["QUERY", RUN_KEY, "/V", value_name], check=False)
        installed = result.returncode == 0
        print(f"{value_name}: {'установлено' if installed else 'не установлено'}")
        missing = missing or not installed
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
