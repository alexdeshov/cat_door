from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.inference.detector_adapters import (
    DetectorOutputAdapter,
    DetectorOutputError,
    create_output_adapter,
)


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
            self._adapter: DetectorOutputAdapter = create_output_adapter(
                manifest,
                self.input_size,
            )
            input_manifest = manifest["input"]
            self.input_name = str(input_manifest.get("name", "images"))
            self.letterbox_value = int(input_manifest.get("letterbox_value", 0))
            self.resize_mode = str(input_manifest.get("resize_mode", "letterbox"))
            self.letterbox_alignment = str(
                input_manifest.get("letterbox_alignment", "center")
            )
            self.channel_order = str(input_manifest.get("channel_order", "rgb"))
            self.input_scale = float(input_manifest.get("scale", 1.0 / 255.0))
            self.input_mean = np.asarray(
                input_manifest.get("mean", [0.0, 0.0, 0.0]),
                dtype=np.float32,
            )
            self.input_std = np.asarray(
                input_manifest.get("std", [1.0, 1.0, 1.0]),
                dtype=np.float32,
            )
            if not 0 <= self.letterbox_value <= 255:
                raise ValueError("letterbox value must be between 0 and 255")
            if self.resize_mode not in {"letterbox", "stretch"}:
                raise ValueError("resize mode must be letterbox or stretch")
            if self.letterbox_alignment not in {"center", "top_left"}:
                raise ValueError("letterbox alignment must be center or top_left")
            if self.channel_order not in {"rgb", "bgr"}:
                raise ValueError("channel order must be rgb or bgr")
            if self.input_mean.shape != (3,) or self.input_std.shape != (3,):
                raise ValueError("input mean and std must have three values")
            if np.any(self.input_std == 0):
                raise ValueError("input std values must be non-zero")
        except (
            DetectorOutputError,
            FileNotFoundError,
            OSError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
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
        self._input = self._compiled.input(self.input_name)
        self._outputs = {
            name: self._compiled.output(name)
            for name in manifest["outputs"]
        }

    def detect(self, rgb_frame: np.ndarray) -> CatDetectionResult:
        expected_shape = (self.input_size, self.input_size, 3)
        if rgb_frame.shape != expected_shape:
            raise ValueError(f"detector frame must have shape {expected_shape}")
        ordered_frame = (
            rgb_frame
            if self.channel_order == "rgb"
            else rgb_frame[:, :, ::-1]
        )
        tensor = np.transpose(ordered_frame, (2, 0, 1))[None].astype(np.float32)
        tensor = tensor * self.input_scale
        tensor = (
            tensor - self.input_mean.reshape(1, 3, 1, 1)
        ) / self.input_std.reshape(1, 3, 1, 1)
        result = self._compiled({self._input: tensor})
        parsed = self._adapter.parse(
            {
                name: np.asarray(result[output])
                for name, output in self._outputs.items()
            }
        )
        detections = tuple(
            CatDetection(
                box=detection.box,
                confidence=detection.confidence,
            )
            for detection in parsed
        )
        confidence = max((detection.confidence for detection in detections), default=0.0)
        return CatDetectionResult(confidence=confidence, detections=detections)

    def confidence(self, rgb_frame: np.ndarray) -> float:
        return self.detect(rgb_frame).confidence
