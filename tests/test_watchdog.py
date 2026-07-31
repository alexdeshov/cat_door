from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from scripts.watchdog import Watchdog
from src.common.settings import Settings


class Clock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def now(self) -> datetime:
        return self.value


def test_healthy_components_do_not_create_restart_log(runtime_path: Path) -> None:
    clock = Clock(datetime(2026, 7, 30, 12, tzinfo=UTC))
    starts: list[str] = []
    watchdog = Watchdog(
        Settings(data_dir=runtime_path),
        now=clock.now,
        status_reader=lambda _path: {"updated_at": clock.now().isoformat()},
        web_probe=lambda: True,
        starters={
            "recorder": lambda: starts.append("recorder") or 101,
            "web": lambda: starts.append("web") or 102,
        },
    )

    watchdog.run_once()

    assert starts == []
    assert not watchdog.log_path.exists()


def test_stale_recorder_restarts_once_and_respects_cooldown(
    runtime_path: Path,
) -> None:
    clock = Clock(datetime(2026, 7, 30, 12, tzinfo=UTC))
    starts: list[str] = []
    watchdog = Watchdog(
        Settings(
            data_dir=runtime_path,
            watchdog_restart_cooldown_seconds=600,
        ),
        now=clock.now,
        status_reader=lambda _path: None,
        web_probe=lambda: True,
        starters={
            "recorder": lambda: starts.append("recorder") or 101,
            "web": lambda: starts.append("web") or 102,
        },
    )

    watchdog.run_once()
    clock.value += timedelta(minutes=5)
    watchdog.run_once()

    events = [
        json.loads(line)
        for line in watchdog.log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert starts == ["recorder"]
    assert [(event["component"], event["action"]) for event in events] == [
        ("recorder", "restart_started")
    ]


def test_dead_web_starts_without_touching_fresh_recorder(runtime_path: Path) -> None:
    clock = Clock(datetime(2026, 7, 30, 12, tzinfo=UTC))
    starts: list[str] = []
    watchdog = Watchdog(
        Settings(data_dir=runtime_path),
        now=clock.now,
        status_reader=lambda _path: {"updated_at": clock.now().isoformat()},
        web_probe=lambda: False,
        starters={
            "recorder": lambda: starts.append("recorder") or 101,
            "web": lambda: starts.append("web") or 102,
        },
    )

    watchdog.run_once()

    assert starts == ["web"]


def test_repeated_launcher_failures_enter_backoff_without_logging_secret(
    runtime_path: Path,
) -> None:
    clock = Clock(datetime(2026, 7, 30, 12, tzinfo=UTC))
    calls = 0

    def fail_recorder() -> int:
        nonlocal calls
        calls += 1
        raise RuntimeError("sensitive-value-that-must-not-be-logged")

    watchdog = Watchdog(
        Settings(
            data_dir=runtime_path,
            watchdog_restart_cooldown_seconds=60,
            watchdog_max_restart_attempts=2,
            watchdog_failure_backoff_seconds=300,
        ),
        now=clock.now,
        status_reader=lambda _path: None,
        web_probe=lambda: True,
        starters={"recorder": fail_recorder, "web": lambda: 102},
    )

    watchdog.run_once()
    clock.value += timedelta(seconds=61)
    watchdog.run_once()
    clock.value += timedelta(seconds=61)
    watchdog.run_once()

    content = watchdog.log_path.read_text(encoding="utf-8")
    events = [json.loads(line) for line in content.splitlines()]
    assert calls == 2
    assert [event["action"] for event in events] == [
        "restart_failed",
        "restart_failed",
        "restart_backoff",
    ]
    assert "sensitive-value" not in content
