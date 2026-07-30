import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.inference.cat_detector import DetectorError, OpenVinoCatDetector


class DeviceReportingCompiled:
    def input(self, name: str) -> str:
        return name

    def output(self, name: str) -> str:
        return name

    def get_property(self, name: str) -> list[str]:
        assert name == "EXECUTION_DEVICES"
        return ["GPU.0"]


def test_detector_rejects_missing_manifest(runtime_path: Path) -> None:
    model = runtime_path / "detector.onnx"
    model.write_bytes(b"not-a-model")

    with pytest.raises(DetectorError, match="detector_manifest_invalid"):
        OpenVinoCatDetector(model)


def test_detector_rejects_checksum_mismatch(runtime_path: Path) -> None:
    model = runtime_path / "detector.onnx"
    model.write_bytes(b"not-a-model")
    model.with_suffix(".json").write_text(
        json.dumps(
            {
                "sha256": "wrong",
                "architecture": "test",
                "cat_class_id": 17,
                "input": {"shape": [1, 3, 320, 320]},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DetectorError, match="detector_checksum_mismatch"):
        OpenVinoCatDetector(model)


def test_detector_requires_expected_frame_shape() -> None:
    detector = object.__new__(OpenVinoCatDetector)
    detector.input_size = 320

    with pytest.raises(ValueError):
        detector.confidence(np.zeros((10, 10, 3), dtype=np.uint8))


def test_detector_returns_cat_boxes_and_maximum_confidence() -> None:
    detector = object.__new__(OpenVinoCatDetector)
    detector.cat_class_id = 17
    detector.input_size = 320
    detector._input = "images"
    detector._boxes = "boxes"
    detector._labels = "labels"
    detector._scores = "scores"
    detector._compiled = lambda inputs: {
        "boxes": np.asarray([[10, 20, 100, 120], [1, 2, 3, 4], [30, 40, 200, 220]]),
        "labels": np.asarray([17, 1, 17]),
        "scores": np.asarray([0.4, 0.99, 0.8]),
    }

    result = detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))

    assert result.confidence == pytest.approx(0.8)
    assert len(result.detections) == 2
    assert result.detections[0].box == (10.0, 20.0, 100.0, 120.0)


def test_detector_reports_actual_openvino_execution_device(
    runtime_path: Path,
    monkeypatch,
) -> None:
    payload = b"synthetic-model"
    model = runtime_path / "detector.onnx"
    model.write_bytes(payload)
    model.with_suffix(".json").write_text(
        json.dumps(
            {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "architecture": "test",
                "cat_class_id": 17,
                "input": {"shape": [1, 3, 416, 416]},
            }
        ),
        encoding="utf-8",
    )
    compile_calls: list[tuple[object, str, dict[str, str]]] = []
    compiled = DeviceReportingCompiled()
    core = SimpleNamespace(
        read_model=lambda _path: object(),
        compile_model=lambda loaded, device, config: (
            compile_calls.append((loaded, device, config)) or compiled
        ),
    )
    monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(Core=lambda: core))

    detector = OpenVinoCatDetector(model, "GPU")

    assert detector.execution_devices == ("GPU.0",)
    assert detector.device == "GPU.0"
    assert detector.input_size == 416
    assert compile_calls[0][1:] == (
        "GPU",
        {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": "f16",
        },
    )


def test_detector_accepts_manifest_declared_416_frame() -> None:
    detector = object.__new__(OpenVinoCatDetector)
    detector.cat_class_id = 17
    detector.input_size = 416
    detector._input = "images"
    detector._boxes = "boxes"
    detector._labels = "labels"
    detector._scores = "scores"
    detector._compiled = lambda inputs: {
        "boxes": np.asarray([[10, 20, 100, 120]]),
        "labels": np.asarray([17]),
        "scores": np.asarray([0.8]),
    }

    result = detector.detect(np.zeros((416, 416, 3), dtype=np.uint8))

    assert result.confidence == pytest.approx(0.8)
