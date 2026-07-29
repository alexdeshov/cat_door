from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ADMIN_URL = "http://127.0.0.1:8080"


def find_chrome() -> Path:
    roots = (
        os.environ.get("PROGRAMFILES"),
        os.environ.get("PROGRAMFILES(X86)"),
        os.environ.get("LOCALAPPDATA"),
    )
    for root in roots:
        if not root:
            continue
        candidate = Path(root) / "Google" / "Chrome" / "Application" / "chrome.exe"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Google Chrome не найден")


def wait_for_admin(*, attempts: int = 60, delay_seconds: float = 1.0) -> bool:
    for _ in range(attempts):
        try:
            with urlopen(ADMIN_URL, timeout=1):
                return True
        except (OSError, URLError):
            time.sleep(delay_seconds)
    return False


def main() -> None:
    if not wait_for_admin():
        raise SystemExit("Админка CatDoor не ответила в течение 60 секунд")
    subprocess.Popen(
        [str(find_chrome()), ADMIN_URL],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )


if __name__ == "__main__":
    main()
