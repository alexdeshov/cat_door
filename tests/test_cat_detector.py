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
