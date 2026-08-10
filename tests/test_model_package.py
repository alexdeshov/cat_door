import json
import zipfile
from pathlib import Path

import pytest

from training.model_package import build_package, verify_package


def test_build_and_verify_package(runtime_path: Path) -> None:
    source = runtime_path / "source"
    source.mkdir()
    (source / "face_model.onnx").write_bytes(b"synthetic-onnx")
    (source / "model_info.json").write_text(
        json.dumps({"labels": ["barsik"], "image_size": 224}),
        encoding="utf-8",
    )
    (source / "thresholds.json").write_text(
        json.dumps(
            {
                "accept_score": 0.9,
                "max_false_accept_rate": 0.01,
                "max_false_reject_rate": 0.2,
                "max_mean_inference_ms": 200.0,
            }
        ),
        encoding="utf-8",
    )
    package = runtime_path / "model.zip"

    build_package(source, package)
    result = verify_package(package)

    assert result["valid"] is True
    assert result["onnx_files"] == ["face_model.onnx"]
    assert "face_model.onnx" in (source / "checksums.sha256").read_text(encoding="ascii")


def test_per_view_metadata_next_to_its_model_is_welcome(runtime_path: Path) -> None:
    """`<ракурс>_model.json` кладёт конвейер — рекордер ищет их рядом с ONNX.

    Сборщик отверг их на первом же прогоне после починки установщика, и
    обучение падало на шаге пакета.
    """
    source = runtime_path / "source"
    source.mkdir()
    (source / "face_model.onnx").write_bytes(b"onnx")
    (source / "face_model.json").write_text("{}", encoding="utf-8")
    (source / "model_info.json").write_text(
        json.dumps({"labels": ["barsik"], "image_size": 224}),
        encoding="utf-8",
    )
    (source / "thresholds.json").write_text(
        json.dumps(
            {
                "accept_score": 0.9,
                "max_false_accept_rate": 0.01,
                "max_false_reject_rate": 0.2,
                "max_mean_inference_ms": 200.0,
            }
        ),
        encoding="utf-8",
    )

    build_package(source, runtime_path / "model.zip")

    # Метаданные без одноимённой модели — по-прежнему отказ.
    (source / "side_model.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="side_model.json"):
        build_package(source, runtime_path / "model2.zip")


def test_verify_rejects_unsafe_member(runtime_path: Path) -> None:
    package = runtime_path / "unsafe.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("../model.onnx", b"bad")
        archive.writestr("model_info.json", "{}")
        archive.writestr("thresholds.json", "{}")
        archive.writestr("checksums.sha256", "")

    with pytest.raises(ValueError, match="unsafe package member"):
        verify_package(package)


def test_verify_rejects_unexpected_member(runtime_path: Path) -> None:
    package = runtime_path / "unexpected.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("face_model.onnx", b"synthetic-onnx")
        archive.writestr("model_info.json", "{}")
        archive.writestr("thresholds.json", "{}")
        archive.writestr("checksums.sha256", "")
        archive.writestr("checkpoint.pt", b"must-not-be-packaged")

    with pytest.raises(ValueError, match=r"unexpected package files: checkpoint\.pt"):
        verify_package(package)


def test_build_rejects_unexpected_source_file(runtime_path: Path) -> None:
    source = runtime_path / "source"
    source.mkdir()
    (source / "face_model.onnx").write_bytes(b"synthetic-onnx")
    (source / "model_info.json").write_text("{}", encoding="utf-8")
    (source / "thresholds.json").write_text("{}", encoding="utf-8")
    (source / "checkpoint.pt").write_bytes(b"must-not-be-packaged")

    with pytest.raises(ValueError, match=r"unexpected package files: checkpoint\.pt"):
        build_package(source, runtime_path / "model.zip")


def test_build_rejects_destination_inside_source(runtime_path: Path) -> None:
    source = runtime_path / "source"
    source.mkdir()
    (source / "face_model.onnx").write_bytes(b"synthetic-onnx")
    (source / "model_info.json").write_text("{}", encoding="utf-8")
    (source / "thresholds.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="package destination must be outside source_dir"):
        build_package(source, source / "model.zip")
