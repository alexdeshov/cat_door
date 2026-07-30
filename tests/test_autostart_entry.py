from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts import autostart_entry


def test_main_changes_to_project_root_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}
    events: list[tuple[str, str]] = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["autostart_entry.py", "web"])
    monkeypatch.setattr(
        autostart_entry,
        "append_autostart_event",
        lambda component, status, **kwargs: events.append((component, status)),
    )

    def capture_launch(component: str) -> None:
        observed["component"] = component
        observed["cwd"] = Path.cwd()

    monkeypatch.setattr(autostart_entry, "run_component", capture_launch)

    autostart_entry.main()

    assert observed == {
        "component": "web",
        "cwd": autostart_entry.PROJECT_ROOT,
    }
    assert events == [("web", "starting"), ("web", "started")]


def test_failure_log_omits_exception_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "autostart.jsonl"
    secret_message = "sensitive-value-that-must-not-be-logged"
    monkeypatch.setattr(sys, "argv", ["autostart_entry.py", "web"])
    monkeypatch.setattr(autostart_entry, "autostart_log_paths", lambda: [log_path])

    def fail_launch(component: str) -> None:
        raise RuntimeError(secret_message)

    monkeypatch.setattr(autostart_entry, "run_component", fail_launch)

    with pytest.raises(RuntimeError, match="must-not-be-logged"):
        autostart_entry.main()

    content = log_path.read_text(encoding="utf-8")
    events = [json.loads(line) for line in content.splitlines()]
    assert [event["status"] for event in events] == ["starting", "failed"]
    assert events[-1]["error_type"] == "builtins.RuntimeError"
    assert events[-1]["trace"]
    assert secret_message not in content
