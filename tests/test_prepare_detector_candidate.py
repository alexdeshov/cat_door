import hashlib
import json
from pathlib import Path

import pytest

from scripts.prepare_detector_candidate import (
    assert_checksum,
    download_verified,
    export_candidate,
    validate_sha256,
)


def test_validate_sha256_rejects_placeholder() -> None:
    with pytest.raises(ValueError, match="64 hexadecimal"):
        validate_sha256("replace-me")


def test_assert_checksum_accepts_matching_local_file(runtime_path: Path) -> None:
    weights = runtime_path / "weights.bin"
    weights.write_bytes(b"candidate-weights")
    expected = hashlib.sha256(b"candidate-weights").hexdigest()

    assert_checksum(weights, expected)


def test_verified_download_checks_hash_before_publishing(runtime_path: Path) -> None:
    source = runtime_path / "source.bin"
    source.write_bytes(b"official-weights")
    destination = runtime_path / "models" / "weights.bin"
    expected = hashlib.sha256(b"official-weights").hexdigest()

    result = download_verified(source.resolve().as_uri(), destination, expected)

    assert result.read_bytes() == b"official-weights"
    assert not destination.with_name(".weights.bin.download").exists()


def test_verified_download_removes_bad_payload(runtime_path: Path) -> None:
    source = runtime_path / "source.bin"
    source.write_bytes(b"wrong")
    destination = runtime_path / "models" / "weights.bin"

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        download_verified(source.resolve().as_uri(), destination, "0" * 64)

    assert not destination.exists()
    assert not destination.with_name(".weights.bin.download").exists()


def test_export_candidate_writes_reproducible_manifest(
    runtime_path: Path,
    monkeypatch,
) -> None:
    weights = runtime_path / "nanodet.ckpt"
    weights.write_bytes(b"candidate-weights")
    expected = hashlib.sha256(b"candidate-weights").hexdigest()
    output = runtime_path / "nanodet.onnx"
    source_root = runtime_path / "nanodet-source"
    source_root.mkdir()
    monkeypatch.setattr(
        "scripts.prepare_detector_candidate._export_nanodet",
        lambda _python, _weights, candidate_output, _root: candidate_output.write_bytes(
            b"onnx"
        ),
    )
    monkeypatch.setattr(
        "scripts.prepare_detector_candidate._onnx_contract",
        lambda _path: ("data", ["output"], [1, 3, 416, 416]),
    )

    result = export_candidate(
        "nanodet-plus-m-416",
        weights,
        expected,
        output,
        python=Path("python"),
        source_root=source_root,
    )
    manifest = json.loads(result.with_suffix(".json").read_text(encoding="utf-8"))

    assert manifest["architecture"] == "nanodet_plus_m"
    assert manifest["license"] == "Apache-2.0"
    assert manifest["input"]["shape"] == [1, 3, 416, 416]
    assert manifest["weights_sha256"] == expected
    assert manifest["sha256"] == hashlib.sha256(b"onnx").hexdigest()


def test_yolo11_export_takes_its_input_size_from_the_manifest(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Вход 640 держится наравне с 416 ради кота в дальнем конце террасы."""
    weights = runtime_path / "yolo11s.pt"
    weights.write_bytes(b"candidate-weights")
    expected = hashlib.sha256(b"candidate-weights").hexdigest()
    output = runtime_path / "yolo11s-640.onnx"
    requested: list[int] = []
    monkeypatch.setattr(
        "scripts.prepare_detector_candidate._export_yolo11",
        lambda _weights, candidate_output, input_size: (
            requested.append(input_size),
            candidate_output.write_bytes(b"onnx"),
        ),
    )
    monkeypatch.setattr(
        "scripts.prepare_detector_candidate._onnx_contract",
        lambda _path: ("images", ["output0"], [1, 3, 640, 640]),
    )

    result = export_candidate(
        "yolo11s-640",
        weights,
        expected,
        output,
        python=Path("python"),
        source_root=None,
    )
    manifest = json.loads(result.with_suffix(".json").read_text(encoding="utf-8"))

    assert requested == [640]
    assert manifest["architecture"] == "yolo11s"
    assert manifest["input"]["shape"] == [1, 3, 640, 640]
    assert manifest["license"] == "AGPL-3.0"


def test_export_refuses_a_model_of_the_wrong_input_size(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Манифест описывает вход детектору: разойтись с ONNX он не имеет права."""
    weights = runtime_path / "yolo11n.pt"
    weights.write_bytes(b"candidate-weights")
    expected = hashlib.sha256(b"candidate-weights").hexdigest()
    output = runtime_path / "yolo11n-640.onnx"
    monkeypatch.setattr(
        "scripts.prepare_detector_candidate._export_yolo11",
        lambda _weights, candidate_output, _size: candidate_output.write_bytes(b"onnx"),
    )
    monkeypatch.setattr(
        "scripts.prepare_detector_candidate._onnx_contract",
        lambda _path: ("images", ["output0"], [1, 3, 416, 416]),
    )

    with pytest.raises(RuntimeError, match="input shape"):
        export_candidate(
            "yolo11n-640",
            weights,
            expected,
            output,
            python=Path("python"),
            source_root=None,
        )

    assert not output.exists()
    assert not output.with_suffix(".json").exists()
