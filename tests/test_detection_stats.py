import json
from pathlib import Path

from src.common.detection_stats import read_detection_stats


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
