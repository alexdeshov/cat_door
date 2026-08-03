from __future__ import annotations

import json

from src.recorder.detection_journal import DetectionJournal, frame_journal_path


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

    payload = json.loads(frame_journal_path(path).read_text(encoding="utf-8"))
    assert payload["event"] == "observation"
    assert payload["camera"] == "face"
    assert payload["confidence"] == 0.42
    assert payload["threshold"] == 0.55
    assert payload["detected"] is False
    assert "timestamp" in payload


def test_frame_events_do_not_crowd_out_the_history(runtime_path) -> None:
    """Покадровые наблюдения вытесняли из журнала всё остальное.

    Их тысячи в час, а ротация — шесть файлов: происшествие, случившееся ночью,
    к утру уже не найти, хотя ради таких находок журнал и заведён.
    """
    path = runtime_path / "runtime" / "detection.jsonl"
    journal = DetectionJournal(path, max_bytes=100_000, backup_count=2)

    journal.write("stream_starving", camera="face", camera_id="nobelic-4mp")
    for _ in range(500):
        journal.write("observation", camera="face", confidence=0.01)
    journal.write("stream_recovered", camera="face")
    journal.close()

    history = path.read_text(encoding="utf-8").splitlines()
    frames = frame_journal_path(path).read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["event"] for line in history] == [
        "stream_starving",
        "stream_recovered",
    ]
    assert len(frames) == 500
