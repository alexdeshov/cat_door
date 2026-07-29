from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class DetectorError(RuntimeError):
    pass


@dataclass(frozen=True)
class CatDetection:
    box: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True)
class CatDetectionResult:
    confidence: float
    detections: tuple[CatDetection, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class OpenVinoCatDetector:
    def __init__(self, model_path: Path, device: str = "CPU") -> None:
        try:
            import openvino as ov
        except ImportError as exc:
            raise DetectorError("openvino_not_installed") from exc

        manifest_path = model_path.with_suffix(".json")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_hash = str(manifest["sha256"])
            self.cat_class_id = int(manifest["cat_class_id"])
        except (FileNotFoundError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DetectorError("detector_manifest_invalid") from exc
        if _sha256(model_path) != expected_hash:
            raise DetectorError("detector_checksum_mismatch")

        core = ov.Core()
        try:
            model = core.read_model(model_path)
            self._compiled: Any = core.compile_model(
                model,
                device,
                {"PERFORMANCE_HINT": "LATENCY"},
            )
        except Exception as exc:
            raise DetectorError("detector_compile_failed") from exc
        self._input = self._compiled.input("images")
        self._boxes = self._compiled.output("boxes")
        self._labels = self._compiled.output("labels")
        self._scores = self._compiled.output("scores")

    def detect(self, rgb_frame: np.ndarray) -> CatDetectionResult:
        if rgb_frame.shape != (320, 320, 3):
            raise ValueError("detector frame must have shape (320, 320, 3)")
        tensor = np.transpose(rgb_frame, (2, 0, 1))[None].astype(np.float32) / 255.0
        result = self._compiled({self._input: tensor})
        boxes = np.asarray(result[self._boxes])
        labels = np.asarray(result[self._labels])
        scores = np.asarray(result[self._scores])
        cat_indices = np.flatnonzero(labels == self.cat_class_id)
        detections = tuple(
            CatDetection(
                box=tuple(float(value) for value in boxes[index]),
                confidence=float(scores[index]),
            )
            for index in cat_indices
        )
        confidence = max((detection.confidence for detection in detections), default=0.0)
        return CatDetectionResult(confidence=confidence, detections=detections)

    def confidence(self, rgb_frame: np.ndarray) -> float:
        return self.detect(rgb_frame).confidence
