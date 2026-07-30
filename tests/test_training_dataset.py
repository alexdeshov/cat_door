from pathlib import Path

import pytest

from training.dataset import DatasetItem, assign_splits, validate_manifest


def item(event_id: str, image_path: str, split: str | None = None) -> DatasetItem:
    return DatasetItem(
        event_id=event_id,
        captured_at="2026-07-27T12:00:00+03:00",
        view="face",
        image_path=image_path,
        label="barsik",
        split=split,
    )


def test_split_keeps_all_frames_of_event_together() -> None:
    items = [item("event-1", "a.jpg"), item("event-1", "b.jpg"), item("event-2", "c.jpg")]

    result = assign_splits(items)

    event_one_splits = {row.split for row in result if row.event_id == "event-1"}
    assert len(event_one_splits) == 1
    assert validate_manifest(result)["items"] == 3


def test_validation_rejects_event_leakage() -> None:
    items = [item("event-1", "a.jpg", "train"), item("event-1", "b.jpg", "test")]

    with pytest.raises(ValueError, match="multiple splits"):
        validate_manifest(items)


def test_validation_rejects_path_traversal() -> None:
    with pytest.raises(ValueError, match="safe relative"):
        item("event-1", "../secret.jpg").validate()


def test_validation_can_check_files(runtime_path: Path) -> None:
    row = item("event-1", "face/a.jpg", "train")

    with pytest.raises(ValueError, match="missing image"):
        validate_manifest([row], runtime_path)
