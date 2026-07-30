from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.inference.cat_detector import DetectorError
from src.inference.preprocessing import (
    IDENTITY_NORMALIZATION_MEAN,
    IDENTITY_NORMALIZATION_STD,
    prepare_identity_tensor,
)


@dataclass(frozen=True)
class IdentityPrediction:
    name: str | None
    confidence: float


class OpenVinoCatIdentity:
    def __init__(self, model_path: Path, device: str = "CPU") -> None:
        metadata_path = model_path.with_suffix(".json")
        if not model_path.is_file() or not metadata_path.is_file():
            raise DetectorError(f"identity model or metadata is missing: {model_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.labels = [str(label) for label in metadata["labels"]]
        self.image_size = int(metadata.get("image_size", 160))
        self.mean = np.asarray(
            metadata.get("normalization_mean", IDENTITY_NORMALIZATION_MEAN),
            dtype=np.float32,
        ).reshape(1, 3, 1, 1)
        self.std = np.asarray(
            metadata.get("normalization_std", IDENTITY_NORMALIZATION_STD),
            dtype=np.float32,
        ).reshape(1, 3, 1, 1)
        try:
            from openvino import Core

            compiled = Core().compile_model(Core().read_model(model_path), device)
        except Exception as exc:
            raise DetectorError(f"could not load identity model {model_path}: {exc}") from exc
        self._compiled: Any = compiled
        self._output = compiled.output(0)

    def predict(self, frame: np.ndarray, threshold: float) -> IdentityPrediction:
        tensor = prepare_identity_tensor(
            frame,
            self.image_size,
            mean=self.mean,
            std=self.std,
        )
        logits = np.asarray(self._compiled([tensor])[self._output])[0]
        shifted = logits - logits.max()
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        index = int(probabilities.argmax())
        confidence = float(probabilities[index])
        return IdentityPrediction(
            name=self.labels[index] if confidence >= threshold else None,
            confidence=confidence,
        )
