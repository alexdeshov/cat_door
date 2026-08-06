from pathlib import Path

import pytest

from training.dataset import DatasetItem, assign_splits, validate_manifest


def item(
    event_id: str,
    image_path: str,
    split: str | None = None,
    label: str = "barsik",
) -> DatasetItem:
    return DatasetItem(
        event_id=event_id,
        captured_at="2026-07-27T12:00:00+03:00",
        view="face",
        image_path=image_path,
        label=label,
        split=split,
    )


def test_split_keeps_all_frames_of_event_together() -> None:
    items = [item("event-1", "a.jpg"), item("event-1", "b.jpg"), item("event-2", "c.jpg")]

    result = assign_splits(items)

    event_one_splits = {row.split for row in result if row.event_id == "event-1"}
    assert len(event_one_splits) == 1
    assert validate_manifest(result)["items"] == 3


def test_split_reserves_test_for_every_label_with_two_events() -> None:
    # Шесть проходов чужих котов против десятков своих: раньше хэш легко
    # оставлял всех чужих вне test, и FAR было не из чего считать.
    items = [item(f"cat-{index}", f"cat-{index}.jpg") for index in range(40)]
    items += [
        item(f"foreign-{index}", f"foreign-{index}.jpg", label="__foreign__")
        for index in range(6)
    ]

    result = assign_splits(items)

    foreign_splits = {row.split for row in result if row.label == "__foreign__"}
    barsik_splits = {row.split for row in result if row.label == "barsik"}
    assert "test" in foreign_splits
    assert {"train", "validation", "test"} <= barsik_splits


def test_split_keeps_single_event_label_in_train() -> None:
    items = [item("only", "a.jpg", label="rare"), item("other", "b.jpg")]

    result = assign_splits(items)

    assert [row.split for row in result if row.label == "rare"] == ["train"]


def test_split_with_two_events_gives_one_test_one_train() -> None:
    items = [item("first", "a.jpg"), item("second", "b.jpg")]

    result = assign_splits(items)

    assert sorted(row.split or "" for row in result) == ["test", "train"]


def test_split_is_deterministic() -> None:
    items = [item(f"event-{index}", f"{index}.jpg") for index in range(10)]

    first = assign_splits(items)
    second = assign_splits(list(reversed(items)))

    by_event = {row.event_id: row.split for row in second}
    assert all(row.split == by_event[row.event_id] for row in first)


def test_split_rejects_event_with_two_labels() -> None:
    items = [item("event-1", "a.jpg"), item("event-1", "b.jpg", label="murka")]

    with pytest.raises(ValueError, match="multiple labels"):
        assign_splits(items)


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
