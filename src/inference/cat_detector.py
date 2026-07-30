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
            self.architecture = str(manifest["architecture"])
            input_shape = manifest["input"]["shape"]
            if (
                not isinstance(input_shape, list)
                or len(input_shape) != 4
                or input_shape[0] != 1
                or input_shape[1] != 3
                or not isinstance(input_shape[2], int)
                or isinstance(input_shape[2], bool)
                or input_shape[2] != input_shape[3]
                or input_shape[2] <= 0
            ):
                raise ValueError("detector input shape must be [1, 3, size, size]")
            self.input_size = input_shape[2]
        except (FileNotFoundError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DetectorError("detector_manifest_invalid") from exc
        if _sha256(model_path) != expected_hash:
            raise DetectorError("detector_checksum_mismatch")

        core = ov.Core()
        try:
            model = core.read_model(model_path)
            compile_config = {"PERFORMANCE_HINT": "LATENCY"}
            if device.upper().startswith("GPU"):
                compile_config["INFERENCE_PRECISION_HINT"] = "f16"
            self._compiled: Any = core.compile_model(
                model,
                device,
                compile_config,
            )
            execution_devices = self._compiled.get_property("EXECUTION_DEVICES")
            self.execution_devices = (
                (execution_devices,)
                if isinstance(execution_devices, str)
                else tuple(
                    str(execution_device)
                    for execution_device in execution_devices
                )
            )
            if not self.execution_devices:
                raise RuntimeError("OpenVINO returned no execution devices")
            self.device = ",".join(self.execution_devices)
        except Exception as exc:
            raise DetectorError("detector_compile_failed") from exc
        self._input = self._compiled.input("images")
        self._boxes = self._compiled.output("boxes")
        self._labels = self._compiled.output("labels")
        self._scores = self._compiled.output("scores")

    def detect(self, rgb_frame: np.ndarray) -> CatDetectionResult:
        expected_shape = (self.input_size, self.input_size, 3)
        if rgb_frame.shape != expected_shape:
            raise ValueError(f"detector frame must have shape {expected_shape}")
        tensor = np.transpose(rgb_frame, (2, 0, 1))[None].astype(np.float32) / 255.0
        result = self._compiled({self._input: tensor})
        boxes = np.asarray(result[self._boxes])
        labels = np.asarray(result[self._labels])
        scores = np.asarray(result[self._scores])
        cat_indices = np.flatnonzero(labels == self.cat_class_id)
        detections = tuple(
            CatDetection(
                box=(
                    float(boxes[index][0]),
                    float(boxes[index][1]),
                    float(boxes[index][2]),
                    float(boxes[index][3]),
                ),
                confidence=float(scores[index]),
            )
            for index in cat_indices
        )
        confidence = max((detection.confidence for detection in detections), default=0.0)
        return CatDetectionResult(confidence=confidence, detections=detections)

    def confidence(self, rgb_frame: np.ndarray) -> float:
        return self.detect(rgb_frame).confidence
