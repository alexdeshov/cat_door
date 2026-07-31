from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import manage_autostart


def test_task_command_uses_project_pythonw_and_absolute_entry() -> None:
    project_root = manage_autostart.PROJECT_ROOT
    pythonw = project_root / ".venv" / "Scripts" / "pythonw.exe"

    command = manage_autostart.task_command("web")

    assert command == f"{pythonw} {project_root / 'scripts' / 'autostart_entry.py'} web"


def test_watchdog_command_uses_project_pythonw_and_absolute_entry() -> None:
    project_root = manage_autostart.PROJECT_ROOT
    pythonw = project_root / ".venv" / "Scripts" / "pythonw.exe"

    command = manage_autostart.watchdog_task_command()

    assert command == f"{pythonw} {project_root / 'scripts' / 'watchdog.py'}"


def test_install_creates_all_user_startup_values(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    scheduler_calls: list[list[str]] = []
    monkeypatch.setattr(
        manage_autostart,
        "task_command",
        lambda component: f"start-{component}",
    )
    monkeypatch.setattr(
        manage_autostart,
        "run_registry",
        lambda arguments, check=True: calls.append(arguments),
    )
    monkeypatch.setattr(
        manage_autostart,
        "run_scheduler",
        lambda arguments, check=True: scheduler_calls.append(arguments),
    )

    manage_autostart.install()

    add_calls = [call for call in calls if call[0] == "ADD"]
    assert len(add_calls) == 3
    assert calls[0][:2] == ["DELETE", manage_autostart.RUN_KEY]
    assert all(call[:2] == ["ADD", manage_autostart.RUN_KEY] for call in add_calls)
    assert any("start-web" in call for call in calls)
    assert any("start-recorder-recording" in call for call in calls)
    assert any("start-browser" in call for call in calls)
    assert scheduler_calls == [
        [
            "/Create",
            "/TN",
            manage_autostart.WATCHDOG_TASK_NAME,
            "/SC",
            "MINUTE",
            "/MO",
            "5",
            "/TR",
            manage_autostart.watchdog_task_command(),
            "/RL",
            "LIMITED",
            "/F",
        ]
    ]


def test_uninstall_removes_watchdog_task(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler_calls: list[list[str]] = []
    monkeypatch.setattr(
        manage_autostart,
        "run_registry",
        lambda arguments, check=True: SimpleNamespace(returncode=1),
    )

    def scheduler(arguments: list[str], check: bool = True) -> SimpleNamespace:
        scheduler_calls.append(arguments)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(manage_autostart, "run_scheduler", scheduler)

    manage_autostart.uninstall()

    assert scheduler_calls == [
        ["/Delete", "/TN", manage_autostart.WATCHDOG_TASK_NAME, "/F"]
    ]


def test_status_requires_watchdog_task(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        manage_autostart,
        "run_registry",
        lambda arguments, check=True: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        manage_autostart,
        "run_scheduler",
        lambda arguments, check=True: SimpleNamespace(returncode=1),
    )

    assert manage_autostart.status() == 1
