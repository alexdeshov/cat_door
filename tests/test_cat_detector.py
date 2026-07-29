import json
from pathlib import Path

import numpy as np
import pytest

from src.inference.cat_detector import DetectorError, OpenVinoCatDetector


def test_detector_rejects_missing_manifest(runtime_path: Path) -> None:
    model = runtime_path / "detector.onnx"
    model.write_bytes(b"not-a-model")

    with pytest.raises(DetectorError, match="detector_manifest_invalid"):
        OpenVinoCatDetector(model)


def test_detector_rejects_checksum_mismatch(runtime_path: Path) -> None:
    model = runtime_path / "detector.onnx"
    model.write_bytes(b"not-a-model")
    model.with_suffix(".json").write_text(
        json.dumps({"sha256": "wrong", "cat_class_id": 17}),
        encoding="utf-8",
    )

    with pytest.raises(DetectorError, match="detector_checksum_mismatch"):
        OpenVinoCatDetector(model)


def test_detector_requires_expected_frame_shape() -> None:
    detector = object.__new__(OpenVinoCatDetector)

    with pytest.raises(ValueError):
        detector.confidence(np.zeros((10, 10, 3), dtype=np.uint8))


def test_detector_returns_cat_boxes_and_maximum_confidence() -> None:
    detector = object.__new__(OpenVinoCatDetector)
    detector.cat_class_id = 17
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
