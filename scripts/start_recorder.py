from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.common.settings import Settings


def start_recorder(*, recording: bool = False) -> int:
    settings = Settings()
    log_dir = settings.data_dir / "runtime"
    log_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["CATDOOR_AUTOMATIC_CAPTURE_ENABLED"] = "true"
    environment["CATDOOR_AUTOMATIC_CAPTURE_DRY_RUN"] = "false" if recording else "true"
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    with (
        (log_dir / "recorder.stdout.log").open("ab") as stdout,
        (log_dir / "recorder.stderr.log").open("ab") as stderr,
    ):
        process = subprocess.Popen(
            [sys.executable, "-m", "src.recorder.app"],
            cwd=Path(__file__).resolve().parent.parent,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=flags,
            close_fds=True,
        )
    return process.pid


def main() -> None:
    parser = argparse.ArgumentParser(description="Start detached CatDoorRecorder")
    parser.add_argument(
        "--record",
        action="store_true",
        help="publish confirmed automatic events from the ring buffer",
    )
    arguments = parser.parse_args()
    print(start_recorder(recording=arguments.record))


if __name__ == "__main__":
    main()
