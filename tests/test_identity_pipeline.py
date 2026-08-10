"""Конвейер идентификации: ограничение набора целевыми котами."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.identity_pipeline import PipelineError, restrict_to_cats


def write_manifest(path: Path) -> None:
    rows = [
        {"event_id": "1", "view": "face", "image_path": "a.jpg", "label": "Яся"},
        {"event_id": "2", "view": "face", "image_path": "b.jpg", "label": "Бася"},
        {"event_id": "3", "view": "side", "image_path": "c.jpg", "label": "__foreign__"},
        {"event_id": "4", "view": "side", "image_path": "d.jpg", "label": "__nocat__"},
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_other_named_cats_become_negatives(tmp_path: Path) -> None:
    """Не-целевые имена — лучшие отрицательные примеры, а не классы.

    Шесть классов, где у четырёх по 10–18 проходов, дали 69% принятых чужих.
    """
    source = tmp_path / "dataset.jsonl"
    write_manifest(source)
    destination = tmp_path / "restricted.jsonl"

    restrict_to_cats(source, ["Яся"], destination)

    labels = [
        json.loads(line)["label"]
        for line in destination.read_text(encoding="utf-8").splitlines()
    ]
    assert labels == ["Яся", "__foreign__", "__foreign__", "__nocat__"]


def test_unknown_target_cat_is_a_loud_refusal(tmp_path: Path) -> None:
    """Опечатка в имени тихо обучила бы модель без кота вовсе."""
    source = tmp_path / "dataset.jsonl"
    write_manifest(source)

    with pytest.raises(PipelineError, match="Мурзик"):
        restrict_to_cats(source, ["Мурзик"], tmp_path / "restricted.jsonl")
