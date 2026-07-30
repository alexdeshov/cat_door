from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def autostart_log_paths() -> list[Path]:
    paths: list[Path] = []
    try:
        from src.common.settings import Settings

        configured_path = Settings().data_dir.resolve() / "runtime" / "autostart.jsonl"
    except (ImportError, OSError, ValueError):
        configured_path = None
    if configured_path is not None:
        paths.append(configured_path)

    local_app_data = os.environ.get("LOCALAPPDATA")
    fallback_root = Path(local_app_data) if local_app_data else PROJECT_ROOT / "data"
    fallback = fallback_root / "CatDoor" / "runtime" / "autostart.jsonl"
    if fallback not in paths:
        paths.append(fallback)
    return paths


def append_autostart_event(
    component: str,
    status: str,
    *,
    error: BaseException | None = None,
) -> None:
    payload: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "component": component,
        "status": status,
    }
    if error is not None:
        error_type = type(error)
        payload["error_type"] = f"{error_type.__module__}.{error_type.__qualname__}"
        payload["trace"] = [
            {
                "file": frame.filename,
                "line": frame.lineno,
                "function": frame.name,
            }
            for frame in traceback.extract_tb(error.__traceback__)
        ]

    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    for path in autostart_log_paths():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as log:
                log.write(f"{line}\n")
            return
        except OSError:
            continue


def run_component(component: str) -> None:
    if component == "browser":
        from scripts.open_admin import main as open_admin

        open_admin()
        return

    if component == "web":
        from scripts.start_web import main as start_web

        start_web()
        return

    from scripts.start_recorder import start_recorder

    print(start_recorder(recording=component == "recorder-recording"))


def main() -> None:
    os.chdir(PROJECT_ROOT)

    parser = argparse.ArgumentParser(description="Start a CatDoor component from Windows")
    parser.add_argument("component", choices=("web", "recorder", "recorder-recording", "browser"))
    arguments = parser.parse_args()

    append_autostart_event(arguments.component, "starting")
    try:
        run_component(arguments.component)
    except BaseException as error:
        append_autostart_event(arguments.component, "failed", error=error)
        raise
    append_autostart_event(arguments.component, "started")


if __name__ == "__main__":
    main()
