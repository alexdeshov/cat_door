"""Сквозная проверка обучения эмбеддингов на игрушечном наборе.

Настоящий прогон стоит десятки минут на живых кропах, и узнать из него, что
экспорт роняется на форме тензора, — дорогое удовольствие. Здесь тот же путь
целиком: обучение, эталоны, порог, выгрузка в ONNX и метаданные.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

CATS = {"Черныш": (200, 40, 40), "Яся": (40, 40, 200)}
FOREIGN_COLOUR = (40, 200, 40)


def _dataset(root: Path) -> Path:
    """Три «кота» разного цвета: двое своих и чужой."""
    from PIL import Image

    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, colour in [*CATS.items(), ("__foreign__", FOREIGN_COLOUR)]:
        for index in range(6):
            name = f"{label}-{index}.jpg"
            Image.new("RGB", (48, 48), colour).save(images / name)
            rows.append(
                {
                    "event_id": f"{label}-{index}",
                    "captured_at": "2026-08-11T10:00:00+00:00",
                    "view": "face",
                    "image_path": f"images/{name}",
                    "label": label,
                    # Двое в обучение, по одному в проверку и в тест.
                    "split": ["train", "train", "train", "train", "validation", "test"][index],
                }
            )
    manifest = root / "manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return manifest


def _imagenet_stub(root: Path) -> Path:
    """Заменитель весов ImageNet: скачивать их в тесте нечего."""
    import torch
    from torchvision import models

    weights = root / "mobilenet.pth"
    torch.save(models.mobilenet_v3_small(weights=None).state_dict(), weights)
    return weights


def test_embeddings_learn_prototypes_and_a_threshold(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from training.train_embeddings import train

    manifest = _dataset(tmp_path)
    checkpoint = tmp_path / "face.pt"

    result = train(
        manifest,
        tmp_path,
        "face",
        checkpoint,
        epochs=1,
        image_size=32,
        weights_path=_imagenet_stub(tmp_path),
    )

    assert checkpoint.is_file()
    # Эталоны только у настоящих котов: «чужой» — это не личность, а все
    # остальные коты мира, и на узле он получается порогом.
    assert sorted(result["prototypes"]) == sorted(CATS)
    assert 0.0 < float(result["threshold"]) <= 0.95


def test_the_exported_model_carries_its_prototypes_and_threshold(tmp_path: Path) -> None:
    """Узлу без эталонов сравнивать не с чем, а без порога — не по чему решать."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from training.export_onnx import export_checkpoint
    from training.train_embeddings import train

    manifest = _dataset(tmp_path)
    checkpoint = tmp_path / "face.pt"
    train(
        manifest,
        tmp_path,
        "face",
        checkpoint,
        epochs=1,
        image_size=32,
        weights_path=_imagenet_stub(tmp_path),
    )

    info = export_checkpoint(checkpoint, tmp_path / "face_model.onnx")

    assert info["kind"] == "embedding"
    assert info["output_name"] == "embedding"
    assert sorted(info["labels"]) == sorted(CATS)
    assert sorted(info["prototypes"]) == sorted(CATS)
    assert len(info["prototypes"]["Яся"]) == info["embedding_dim"]
    assert (tmp_path / "face_model.onnx").is_file()


def test_the_prototypes_are_unit_vectors(tmp_path: Path) -> None:
    """Косинус считается скалярным произведением, и длина эталона его исказит."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import numpy as np

    from training.train_embeddings import train

    manifest = _dataset(tmp_path)
    checkpoint = tmp_path / "face.pt"
    train(
        manifest,
        tmp_path,
        "face",
        checkpoint,
        epochs=1,
        image_size=32,
        weights_path=_imagenet_stub(tmp_path),
    )
    import torch

    prototypes = torch.load(checkpoint, map_location="cpu", weights_only=True)["prototypes"]

    for vector in prototypes.values():
        assert np.isclose(np.linalg.norm(np.asarray(vector)), 1.0, atol=1e-4)
