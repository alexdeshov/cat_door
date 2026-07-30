from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import Tensor, nn
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)


class ExportableDetector(nn.Module):
    def __init__(self, weights_path: Path) -> None:
        super().__init__()
        self.detector = ssdlite320_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,
            num_classes=len(SSDLite320_MobileNet_V3_Large_Weights.DEFAULT.meta["categories"]),
        ).eval()
        self.detector.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True)
        )

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        result = self.detector([images[0]])[0]
        return result["boxes"], result["labels"], result["scores"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def export_detector(destination: Path, weights_path: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    model = ExportableDetector(weights_path)
    example = torch.zeros((1, 3, 320, 320), dtype=torch.float32)
    torch.onnx.export(
        model,
        (example,),
        destination,
        input_names=["images"],
        output_names=["boxes", "labels", "scores"],
        opset_version=18,
        dynamo=False,
    )
    categories = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT.meta["categories"]
    manifest = {
        "format": "onnx",
        "architecture": "ssdlite320_mobilenet_v3_large",
        "weights": "COCO_V1",
        "license": "BSD-3-Clause",
        "input": {"name": "images", "shape": [1, 3, 320, 320], "range": [0.0, 1.0]},
        "outputs": ["boxes", "labels", "scores"],
        "cat_class_id": categories.index("cat"),
        "sha256": sha256(destination),
    }
    destination.with_suffix(".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local COCO cat detector")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/cat-detector.onnx"),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth"),
    )
    arguments = parser.parse_args()
    print(export_detector(arguments.output, arguments.weights))


if __name__ == "__main__":
    main()
