import argparse
import json
from pathlib import Path
from typing import Any

from src.inference.preprocessing import (
    IDENTITY_NORMALIZATION_MEAN,
    IDENTITY_NORMALIZATION_STD,
)


def export_checkpoint(checkpoint_path: Path, destination: Path) -> dict[str, object]:
    """Выгрузить обученную модель в ONNX вместе с её метаданными.

    Видов два, и различает их сам чекпойнт. Классификатор отдаёт логиты по
    классам. Модель эмбеддингов отдаёт нормированный вектор, а решение «кто
    это» принимается сравнением с эталонами, которые уезжают в метаданные
    рядом: узлу иначе не с чем сравнивать.
    """
    import torch
    from torchvision import models

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    labels = list(checkpoint["labels"])
    image_size = int(checkpoint["image_size"])
    if str(checkpoint.get("kind", "logits")) == "embedding":
        return _export_embedding(checkpoint, destination, image_size)
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(labels))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    example = torch.randn(1, 3, image_size, image_size)
    destination.parent.mkdir(parents=True, exist_ok=True)
    program = torch.onnx.export(
        model,
        (example,),
        input_names=["image"],
        output_names=["logits"],
        dynamic_shapes=({0: "batch"},),
        dynamo=True,
        external_data=False,
    )
    if program is None:
        raise RuntimeError("PyTorch ONNX exporter did not return an ONNX program")
    program.save(destination)
    return {
        "architecture": checkpoint["architecture"],
        "view": checkpoint["view"],
        "labels": labels,
        "image_size": image_size,
        "normalization_mean": list(IDENTITY_NORMALIZATION_MEAN),
        "normalization_std": list(IDENTITY_NORMALIZATION_STD),
        "input_name": "image",
        "output_name": "logits",
        "onnx_file": destination.name,
    }


def _export_embedding(
    checkpoint: dict[str, Any],
    destination: Path,
    image_size: int,
) -> dict[str, object]:
    import torch

    from training.train_embeddings import _build_backbone

    prototypes = checkpoint["prototypes"]
    if not isinstance(prototypes, dict) or not prototypes:
        raise RuntimeError("embedding checkpoint carries no prototypes")
    embedding_dim = int(checkpoint["embedding_dim"])
    model = _build_backbone(embedding_dim, None)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    example = torch.randn(1, 3, image_size, image_size)
    destination.parent.mkdir(parents=True, exist_ok=True)
    program = torch.onnx.export(
        model,
        (example,),
        input_names=["image"],
        output_names=["embedding"],
        dynamic_shapes=({0: "batch"},),
        dynamo=True,
        external_data=False,
    )
    if program is None:
        raise RuntimeError("PyTorch ONNX exporter did not return an ONNX program")
    program.save(destination)
    return {
        "architecture": checkpoint["architecture"],
        "kind": "embedding",
        "view": checkpoint["view"],
        # Наружу — только имена, у которых есть эталон: «чужой» это не личность.
        "labels": sorted(prototypes),
        "prototypes": {name: list(vector) for name, vector in prototypes.items()},
        "embedding_dim": embedding_dim,
        # Порог подобран при обучении на отложенном наборе и едет с моделью:
        # граница «свой/чужой» — свойство пространства, а не настройка узла.
        "threshold": float(checkpoint.get("threshold", 0.55)),
        "image_size": image_size,
        "normalization_mean": list(IDENTITY_NORMALIZATION_MEAN),
        "normalization_std": list(IDENTITY_NORMALIZATION_STD),
        "input_name": "image",
        "output_name": "embedding",
        "onnx_file": destination.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a CatDoor checkpoint to ONNX")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--model-info", type=Path, required=True)
    args = parser.parse_args()
    model_info = export_checkpoint(args.checkpoint, args.destination)
    args.model_info.write_text(
        json.dumps(model_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
