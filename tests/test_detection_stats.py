import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.common.detection_stats import (
    read_detection_stats,
    read_detector_probe_stats,
)


def test_reads_rotated_detection_observations_and_ignores_invalid_lines(
    runtime_path: Path,
) -> None:
    journal = runtime_path / "detection.jsonl"
    journal.with_suffix(".jsonl.1").write_text(
        json.dumps(
            {
                "event": "observation",
                "camera": "face",
                "detected": False,
                "inference_ms": 40.0,
                "frame_age_ms": 100.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    journal.write_text(
        "\n".join(
            [
                "{broken",
                json.dumps(
                    {
                        "event": "observation",
                        "camera": "face",
                        "detected": True,
                        "inference_ms": 60.0,
                        "frame_age_ms": 140.0,
                    }
                ),
                json.dumps({"event": "session_started"}),
            ]
        ),
        encoding="utf-8",
    )

    result = read_detection_stats(journal)

    assert result["face"] == {
        "observations": 2,
        "above_threshold": 1,
        "median_inference_ms": 50.0,
        "median_frame_age_ms": 120.0,
    }
    assert result["side"]["observations"] == 0


def test_probe_stats_use_only_requested_time_window(runtime_path: Path) -> None:
    journal = runtime_path / "detection.jsonl"
    started_at = datetime(2026, 7, 30, 12, tzinfo=UTC)
    rows = [
        {
            "timestamp": (started_at - timedelta(seconds=1)).isoformat(),
            "event": "observation",
            "camera": "face",
            "inference_ms": 500.0,
        },
        {
            "timestamp": (started_at + timedelta(seconds=1)).isoformat(),
            "event": "observation",
            "camera": "face",
            "inference_ms": 40.0,
        },
        {
            "timestamp": (started_at + timedelta(seconds=2)).isoformat(),
            "event": "observation",
            "camera": "side",
            "inference_ms": 80.0,
        },
    ]
    journal.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    result = read_detector_probe_stats(
        journal,
        started_at=started_at,
        ended_at=started_at + timedelta(seconds=30),
    )

    assert result["face"]["median_inference_ms"] == 40.0
    assert result["face"]["latency_passed"] is True
    assert result["side"]["latency_passed"] is False
    assert result["combined"]["median_inference_ms"] == 60.0
    assert result["combined"]["p95_inference_ms"] == 78.0
