from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.start_recorder import start_recorder
from scripts.start_web import start_web
from src.common.settings import Settings
from src.recorder.status import read_recorder_status

ComponentStarter = Callable[[], int]


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return _as_utc(datetime.fromisoformat(value))
    except ValueError:
        return None


def recorder_status_is_fresh(
    payload: Mapping[str, object] | None,
    *,
    now: datetime,
    stale_after: timedelta,
) -> bool:
    updated_at = _parse_timestamp(payload.get("updated_at") if payload else None)
    if updated_at is None:
        return False
    return _as_utc(now) - updated_at <= stale_after


def probe_web(port: int, timeout_seconds: float) -> bool:
    request = Request(
        f"http://127.0.0.1:{port}/api/recorder/status",
        headers={"Accept": "application/json"},
    )
    try:
        with build_opener(ProxyHandler({})).open(
            request,
            timeout=timeout_seconds,
        ) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


class Watchdog:
    def __init__(
        self,
        settings: Settings,
        *,
        now: Callable[[], datetime] | None = None,
        status_reader: Callable[[Path], dict[str, object] | None] = read_recorder_status,
        web_probe: Callable[[], bool] | None = None,
        starters: Mapping[str, ComponentStarter] | None = None,
    ) -> None:
        self.settings = settings
        self._now = now or (lambda: datetime.now(UTC))
        self._status_reader = status_reader
        self._web_probe = web_probe or (
            lambda: probe_web(
                settings.web_port,
                settings.watchdog_web_timeout_seconds,
            )
        )
        self._starters = starters or {
            "recorder": lambda: start_recorder(recording=True),
            "web": start_web,
        }
        self.runtime_dir = settings.data_dir / "runtime"
        self.state_path = self.runtime_dir / "watchdog-state.json"
        self.log_path = self.runtime_dir / "watchdog.jsonl"

    def run_once(self) -> None:
        current_time = _as_utc(self._now())
        state = self._load_state()
        components = state.get("components")
        if not isinstance(components, dict):
            components = {}
            state["components"] = components
        changed = False
        health = {
            "recorder": recorder_status_is_fresh(
                self._status_reader(self.runtime_dir / "recorder-status.json"),
                now=current_time,
                stale_after=timedelta(
                    seconds=self.settings.watchdog_recorder_stale_seconds
                ),
            ),
            "web": self._web_probe(),
        }
        for component, healthy in health.items():
            component_state = components.get(component)
            if not isinstance(component_state, dict):
                component_state = {}
                components[component] = component_state
                changed = True
            if healthy:
                if component_state:
                    components[component] = {}
                    changed = True
                continue
            changed = (
                self._handle_unhealthy(component, component_state, current_time)
                or changed
            )
        if changed:
            self._save_state(state)

    def _handle_unhealthy(
        self,
        component: str,
        component_state: dict[str, Any],
        current_time: datetime,
    ) -> bool:
        blocked_until = _parse_timestamp(component_state.get("blocked_until"))
        if blocked_until is not None and blocked_until > current_time:
            return False
        if blocked_until is not None:
            component_state.clear()

        last_restart_at = _parse_timestamp(component_state.get("last_restart_at"))
        cooldown = timedelta(
            seconds=self.settings.watchdog_restart_cooldown_seconds
        )
        if (
            last_restart_at is not None
            and current_time - last_restart_at < cooldown
        ):
            return False

        attempts = int(component_state.get("restart_attempts", 0))
        if attempts >= self.settings.watchdog_max_restart_attempts:
            self._enter_backoff(component, component_state, current_time, attempts)
            return True

        attempts += 1
        component_state["restart_attempts"] = attempts
        component_state["last_restart_at"] = current_time.isoformat()
        try:
            pid = self._starters[component]()
        except Exception as exc:  # noqa: BLE001 - launcher failures must be contained
            self._append_event(
                component,
                "restart_failed",
                attempt=attempts,
                error_type=type(exc).__name__,
            )
            if attempts >= self.settings.watchdog_max_restart_attempts:
                self._enter_backoff(component, component_state, current_time, attempts)
        else:
            self._append_event(
                component,
                "restart_started",
                attempt=attempts,
                pid=pid,
            )
        return True

    def _enter_backoff(
        self,
        component: str,
        component_state: dict[str, Any],
        current_time: datetime,
        attempts: int,
    ) -> None:
        blocked_until = current_time + timedelta(
            seconds=self.settings.watchdog_failure_backoff_seconds
        )
        component_state["blocked_until"] = blocked_until.isoformat()
        self._append_event(
            component,
            "restart_backoff",
            attempt=attempts,
            blocked_until=blocked_until.isoformat(),
        )

    def _load_state(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _save_state(self, state: dict[str, Any]) -> None:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_name(
            f".{self.state_path.name}.{uuid4().hex}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(state, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(self.state_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _append_event(self, component: str, action: str, **values: object) -> None:
        payload = {
            "timestamp": _as_utc(self._now()).isoformat(),
            "component": component,
            "action": action,
            **values,
        }
        try:
            self.runtime_dir.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as log:
                log.write(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        except OSError:
            return


def _acquire_run_lock(path: Path) -> int | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            age = datetime.now(UTC).timestamp() - path.stat().st_mtime
        except OSError:
            return None
        if age <= 240:
            return None
        try:
            path.unlink()
        except OSError:
            return None
        try:
            return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return None


def main() -> None:
    os.chdir(PROJECT_ROOT)
    settings = Settings()
    lock_path = settings.data_dir / "runtime" / "watchdog.lock"
    lock_handle = _acquire_run_lock(lock_path)
    if lock_handle is None:
        return
    try:
        Watchdog(settings).run_once()
    finally:
        os.close(lock_handle)
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
