from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Start a CatDoor component from Windows")
    parser.add_argument("component", choices=("web", "recorder", "browser"))
    arguments = parser.parse_args()

    if arguments.component == "browser":
        from scripts.open_admin import main as open_admin

        open_admin()
        return

    if arguments.component == "web":
        from scripts.start_web import main as start_web

        start_web()
        return

    from scripts.start_recorder import start_recorder

    print(start_recorder(recording=False))


if __name__ == "__main__":
    main()
