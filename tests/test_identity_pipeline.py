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


def test_a_ready_export_is_taken_instead_of_rebuilt() -> None:
    """13 августа конвейер запустил вторую сборку поверх законченной.

    Полтора часа процессора ради того же самого набора — и полтора часа
    ожидания у владельца, который уже видел «Набор собран».
    """
    from datetime import UTC, datetime

    from scripts.identity_pipeline import _is_fresh_export

    just_now = datetime.now(UTC).isoformat()

    assert _is_fresh_export({"archive": "macbook.zip", "finished_at": just_now})


def test_a_stale_export_is_rebuilt() -> None:
    """Вчерашний набор — это вчерашняя разметка, выброшенная молча."""
    from datetime import UTC, datetime, timedelta

    from scripts.identity_pipeline import _is_fresh_export

    yesterday = (datetime.now(UTC) - timedelta(hours=30)).isoformat()

    assert not _is_fresh_export({"archive": "macbook.zip", "finished_at": yesterday})


def test_an_unfinished_export_is_never_reused() -> None:
    """Незаконченная сборка — это отсутствие набора, а не набор."""
    from datetime import UTC, datetime

    from scripts.identity_pipeline import _is_fresh_export

    # Готового набора нет вовсе — поля `ready` в ответе не будет.
    assert not _is_fresh_export(None)
    assert not _is_fresh_export({})
    # Часы узла могли уйти вперёд: набор «из будущего» свежим не считаем.
    ahead = datetime.now(UTC).replace(year=datetime.now(UTC).year + 1).isoformat()
    assert not _is_fresh_export({"archive": "macbook.zip", "finished_at": ahead})
    # Отчёт без времени — не отчёт: судить о свежести не по чему.
    assert not _is_fresh_export({"archive": "macbook.zip"})


def test_cats_may_be_written_with_commas_or_spaces() -> None:
    """`--cats Черныш Яся` и `--cats Черныш,Яся` пишут с равной уверенностью.

    13 августа запятая стоила прогона: аргумент склеился в одно имя, которого в
    наборе, разумеется, не было.
    """
    from scripts.identity_pipeline import split_cats

    assert split_cats(["Черныш", "Яся"]) == {"Черныш", "Яся"}
    assert split_cats(["Черныш,Яся"]) == {"Черныш", "Яся"}
    assert split_cats(["Черныш, Яся"]) == {"Черныш", "Яся"}
    assert split_cats([]) == set()


def test_the_refusal_shows_which_names_the_set_has(tmp_path) -> None:
    """«Нет проходов кота Мурзик» без списка имён отправляет искать не там."""
    import json

    import pytest

    from scripts.identity_pipeline import PipelineError, restrict_to_cats

    manifest = tmp_path / "dataset.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps({"label": name}, ensure_ascii=False)
            for name in ("Черныш", "Яся", "Бася")
        ),
        encoding="utf-8",
    )

    with pytest.raises(PipelineError, match="Есть: Бася, Черныш, Яся"):
        restrict_to_cats(manifest, ["Мурзик"], tmp_path / "out.jsonl")
