import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.recorder.ffmpeg import RecorderError
from src.recorder.ring import concatenate_segments, prune_segments, select_segments


def make_segment(directory: Path, name: str, finalized_at: float) -> Path:
    segment = directory / name
    segment.parent.mkdir(parents=True, exist_ok=True)
    segment.write_bytes(b"segment")
    os.utime(segment, (finalized_at, finalized_at))
    return segment


def test_selects_segments_overlapping_event_window(runtime_path: Path) -> None:
    ring = runtime_path / "ring"
    old = make_segment(ring, "old.mp4", 90.0)
    first = make_segment(ring, "first.mp4", 100.0)
    second = make_segment(ring, "second.mp4", 102.0)
    late = make_segment(ring, "late.mp4", 110.0)

    selected = select_segments(ring, started_at=99.0, ended_at=103.0, segment_seconds=2.0)

    assert selected == [first, second]
    assert old not in selected
    assert late not in selected


def test_prunes_only_expired_mp4_segments(runtime_path: Path) -> None:
    ring = runtime_path / "ring"
    expired = make_segment(ring, "expired.mp4", 10.0)
    current = make_segment(ring, "current.mp4", 90.0)
    note = ring / "keep.txt"
    note.write_text("keep", encoding="utf-8")

    removed = prune_segments(ring, retention_seconds=50.0, now=100.0)

    assert removed == 1
    assert not expired.exists()
    assert current.exists()
    assert note.exists()


def test_concatenates_segments_atomically(runtime_path: Path, monkeypatch) -> None:
    segments = [
        make_segment(runtime_path / "ring", "one.mp4", 10.0),
        make_segment(runtime_path / "ring", "two.mp4", 12.0),
    ]
    destination = runtime_path / "event" / "face.mp4"

    def fake_run(command, **kwargs):
        Path(command[-1]).write_bytes(b"combined")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("src.recorder.ring.subprocess.run", fake_run)

    concatenate_segments("ffmpeg", segments, destination)

    assert destination.read_bytes() == b"combined"
    assert not destination.with_suffix(".concat.txt").exists()


def test_concat_rejects_empty_segment_list(runtime_path: Path) -> None:
    with pytest.raises(RecorderError, match="ring_segments_missing"):
        concatenate_segments("ffmpeg", [], runtime_path / "face.mp4")
