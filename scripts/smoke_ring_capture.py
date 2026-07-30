from __future__ import annotations

import json
import time
from datetime import UTC, datetime

from src.common.settings import Settings
from src.recorder.ring import concatenate_segments, select_segments


def main() -> None:
    settings = Settings()
    ended_at = time.time()
    started_at = ended_at - 12
    time.sleep(settings.ring_segment_seconds + 1)
    output = (
        settings.data_dir
        / "runtime"
        / "smoke"
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    views: dict[str, object] = {}
    result: dict[str, object] = {"output": str(output), "views": views}
    for role in ("face", "side"):
        segments = select_segments(
            settings.data_dir / "runtime" / "ring" / role,
            started_at,
            ended_at,
            settings.ring_segment_seconds,
        )
        destination = output / f"{role}.mp4"
        concatenate_segments(settings.ffmpeg_path, segments, destination)
        views[role] = {
            "segments": len(segments),
            "bytes": destination.stat().st_size,
        }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
