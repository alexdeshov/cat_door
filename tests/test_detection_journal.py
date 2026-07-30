from __future__ import annotations

import json

from src.recorder.detection_journal import DetectionJournal


def test_detection_journal_writes_structured_jsonl(runtime_path) -> None:
    path = runtime_path / "runtime" / "detection.jsonl"
    journal = DetectionJournal(path, max_bytes=100_000, backup_count=2)

    journal.write(
        "observation",
        camera="face",
        confidence=0.42,
        threshold=0.55,
        detected=False,
        state="idle",
    )
    journal.close()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["event"] == "observation"
    assert payload["camera"] == "face"
    assert payload["confidence"] == 0.42
    assert payload["threshold"] == 0.55
    assert payload["detected"] is False
    assert "timestamp" in payload
