import pytest

from training.acceptance import calculate_metrics, select_test_items
from training.dataset import DatasetItem


def make_item(view: str, split: str, name: str) -> DatasetItem:
    return DatasetItem(
        event_id=f"event-{name}",
        captured_at="2026-08-06T12:00:00+03:00",
        view=view,
        image_path=f"{name}.jpg",
        label="barsik",
        split=split,
    )


def test_select_test_items_filters_by_view() -> None:
    items = [
        make_item("face", "test", "a"),
        make_item("side", "test", "b"),
        make_item("face", "train", "c"),
    ]

    selected = select_test_items(items, "face")

    assert [item.image_path for item in selected] == ["a.jpg"]


def test_select_test_items_without_view_keeps_all_test() -> None:
    items = [make_item("face", "test", "a"), make_item("side", "test", "b")]

    assert len(select_test_items(items)) == 2


def test_select_test_items_rejects_empty_view() -> None:
    items = [make_item("face", "test", "a")]

    with pytest.raises(ValueError, match="side"):
        select_test_items(items, "side")


def test_acceptance_prioritizes_false_accept_rate() -> None:
    metrics = calculate_metrics(
        expected=["barsik", "barsik", "__foreign__", "__foreign__"],
        predicted=["barsik", None, None, "barsik"],
        inference_ms=[10.0, 12.0, 11.0, 13.0],
        max_far=0.1,
        max_frr=0.6,
    )

    assert metrics.false_accept_rate == 0.5
    assert metrics.false_reject_rate == 0.5
    assert metrics.passed is False


def test_acceptance_requires_known_and_foreign_samples() -> None:
    metrics = calculate_metrics(
        expected=["barsik"],
        predicted=["barsik"],
        inference_ms=[10.0],
        max_far=0.0,
        max_frr=0.0,
    )

    assert metrics.false_accept_rate is None
    assert metrics.passed is False


def test_acceptance_counts_misidentified_known_separately() -> None:
    metrics = calculate_metrics(
        expected=["barsik", "murka", "__foreign__"],
        predicted=["murka", "murka", None],
        inference_ms=[10.0, 10.0, 10.0],
        max_far=0.0,
        max_frr=0.6,
    )

    assert metrics.false_rejects == 1
    assert metrics.misidentified_known == 1
    assert metrics.false_reject_rate == 0.5


def test_acceptance_rejects_slow_model() -> None:
    metrics = calculate_metrics(
        expected=["barsik", "__foreign__"],
        predicted=["barsik", None],
        inference_ms=[250.0, 250.0],
        max_far=0.0,
        max_frr=0.0,
        max_mean_inference_ms=200.0,
    )

    assert metrics.latency_passed is False
    assert metrics.passed is False
