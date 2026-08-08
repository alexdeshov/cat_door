from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

CANDIDATES: dict[str, dict[str, Any]] = {
    "nanodet-plus-m-416": {
        "architecture": "nanodet_plus_m",
        "license": "Apache-2.0",
        "input": {
            "name": "data",
            "shape": [1, 3, 416, 416],
            "channel_order": "bgr",
            "scale": 1.0,
            "mean": [103.53, 116.28, 123.675],
            "std": [57.375, 57.12, 58.395],
            "letterbox_value": 0,
            "resize_mode": "stretch",
            "letterbox_alignment": "center",
        },
        "postprocessing": {
            "score_threshold": 0.01,
            "nms_iou_threshold": 0.6,
            "max_detections": 100,
            "strides": [8, 16, 32, 64],
            "reg_max": 7,
            "num_classes": 80,
        },
        "source": "https://github.com/RangiLyu/nanodet",
    },
    "yolox-nano-416": {
        "architecture": "yolox_nano",
        "license": "Apache-2.0",
        "input": {
            "name": "images",
            "shape": [1, 3, 416, 416],
            "channel_order": "bgr",
            "scale": 1.0,
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
            "letterbox_value": 114,
            "resize_mode": "letterbox",
            "letterbox_alignment": "top_left",
        },
        "postprocessing": {
            "score_threshold": 0.01,
            "nms_iou_threshold": 0.65,
            "max_detections": 100,
        },
        "source": "https://github.com/Megvii-BaseDetection/YOLOX",
    },
}


def _yolo11(weights_name: str, input_size: int) -> dict[str, Any]:
    """Запись кандидата Ultralytics: отличаются только веса и сторона входа.

    Вход 640 держится наравне с 416 не ради архитектуры, а ради дальнего кота:
    у него на террасе после сжатия до квадрата остаются десятки точек, и
    разрешение может дать больше смены модели.

    Лицензия AGPL-3.0 домашнему некоммерческому применению в приватном
    репозитории не мешает. Публикация репозитория с кодом или дообученными
    весами Ultralytics потребует сделать AGPL весь проект — это оговорка на
    будущее, а не запрет сегодня.
    """
    return {
        "architecture": weights_name,
        "license": "AGPL-3.0",
        "input": {
            "name": "images",
            "shape": [1, 3, input_size, input_size],
            "channel_order": "rgb",
            "scale": 1.0 / 255.0,
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
            "letterbox_value": 114,
            "resize_mode": "letterbox",
            "letterbox_alignment": "center",
        },
        "postprocessing": {
            "score_threshold": 0.001,
            "nms_in_graph": True,
            "nms_iou_threshold": 0.7,
            "max_detections": 300,
        },
        "source": "https://github.com/ultralytics/ultralytics",
    }


CANDIDATES.update(
    {
        "yolo11n-416": _yolo11("yolo11n", 416),
        "yolo11n-640": _yolo11("yolo11n", 640),
        "yolo11s-640": _yolo11("yolo11s", 640),
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sha256(value: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("SHA-256 must contain exactly 64 hexadecimal characters")
    return normalized


def assert_checksum(path: Path, expected_sha256: str) -> None:
    expected = validate_sha256(expected_sha256)
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: expected {expected}, received {actual}"
        )


def download_verified(url: str, destination: Path, expected_sha256: str) -> Path:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    expected = validate_sha256(expected_sha256)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.download")
    if temporary.exists():
        raise FileExistsError(f"temporary download already exists: {temporary}")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "CatDoorAI-model-preparation"},
    )
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            temporary.open("wb") as target,
        ):
            shutil.copyfileobj(response, target)
        assert_checksum(temporary, expected)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _run(command: Sequence[str], source_root: Path | None = None) -> None:
    result = subprocess.run(
        list(command),
        cwd=source_root,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"candidate export failed with exit code {result.returncode}")


def _export_nanodet(
    python: Path,
    weights: Path,
    output: Path,
    source_root: Path,
) -> None:
    exporter = source_root / "tools" / "export_onnx.py"
    config = source_root / "config" / "nanodet-plus-m_416.yml"
    if not exporter.is_file() or not config.is_file():
        raise FileNotFoundError("NanoDet checkout lacks exporter or 416 config")
    _run(
        (
            str(python),
            str(exporter),
            "--cfg_path",
            str(config),
            "--model_path",
            str(weights),
            "--out_path",
            str(output),
            "--input_shape",
            "416,416",
        ),
        source_root,
    )


def _export_yolox(
    python: Path,
    weights: Path,
    output: Path,
    source_root: Path,
) -> None:
    exporter = source_root / "tools" / "export_onnx.py"
    if not exporter.is_file():
        raise FileNotFoundError("YOLOX checkout lacks tools/export_onnx.py")
    _run(
        (
            str(python),
            str(exporter),
            "--output-name",
            str(output),
            "--input",
            "images",
            "--output",
            "output",
            "--opset",
            "18",
            "--name",
            "yolox-nano",
            "--ckpt",
            str(weights),
            "--decode_in_inference",
            "test_size",
            "416,416",
        ),
        source_root,
    )


def _export_yolo11(weights: Path, output: Path, input_size: int) -> None:
    try:
        from ultralytics import YOLO  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required only in the candidate export environment"
        ) from exc
    default_export = weights.with_suffix(".onnx")
    if default_export.exists() and default_export.resolve() != output.resolve():
        raise FileExistsError(f"refusing to overwrite Ultralytics output {default_export}")
    exported = Path(
        YOLO(str(weights)).export(
            format="onnx",
            imgsz=input_size,
            opset=18,
            simplify=True,
            nms=True,
            conf=0.001,
            iou=0.7,
            max_det=300,
            batch=1,
            device="cpu",
        )
    )
    if not exported.is_file():
        raise RuntimeError("Ultralytics did not return an exported ONNX file")
    if exported.resolve() != output.resolve():
        shutil.move(str(exported), output)


def _onnx_contract(path: Path) -> tuple[str, list[str], list[int]]:
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("onnx is required in the candidate export environment") from exc
    model = onnx.load(path)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise RuntimeError("candidate ONNX must have exactly one input and one output")
    input_node = model.graph.input[0]
    dimensions: list[int] = []
    for dimension in input_node.type.tensor_type.shape.dim:
        if not dimension.HasField("dim_value"):
            raise RuntimeError("candidate ONNX input shape must be static")
        dimensions.append(int(dimension.dim_value))
    return input_node.name, [model.graph.output[0].name], dimensions


def export_candidate(
    candidate: str,
    weights: Path,
    weights_sha256: str,
    output: Path,
    *,
    python: Path,
    source_root: Path | None,
) -> Path:
    metadata = CANDIDATES[candidate]
    weights = weights.resolve()
    output = output.resolve()
    python = python.resolve()
    source_root = source_root.resolve() if source_root is not None else None
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to overwrite candidate output {output}")
    if not weights.is_file():
        raise FileNotFoundError(weights)
    assert_checksum(weights, weights_sha256)
    output.parent.mkdir(parents=True, exist_ok=True)
    architecture = str(metadata["architecture"])
    expected_shape = metadata["input"]["shape"]
    try:
        if architecture == "nanodet_plus_m":
            if source_root is None:
                raise ValueError("NanoDet export requires --source-root")
            _export_nanodet(python, weights, output, source_root)
        elif architecture == "yolox_nano":
            if source_root is None:
                raise ValueError("YOLOX export requires --source-root")
            _export_yolox(python, weights, output, source_root)
        elif architecture.startswith("yolo11"):
            _export_yolo11(weights, output, int(expected_shape[2]))
        else:
            raise ValueError(f"unknown detector candidate: {candidate}")
        if not output.is_file():
            raise RuntimeError("candidate exporter did not create the requested ONNX file")
        input_name, output_names, input_shape = _onnx_contract(output)
        if input_shape != expected_shape:
            raise RuntimeError(
                f"candidate input shape is {input_shape}, expected {expected_shape}"
            )
        manifest = {
            "format": "onnx",
            "architecture": metadata["architecture"],
            "license": metadata["license"],
            "weights": weights.name,
            "weights_sha256": validate_sha256(weights_sha256),
            "source": metadata["source"],
            "input": {**metadata["input"], "name": input_name},
            "outputs": output_names,
            "cat_class_id": 15,
            "postprocessing": metadata["postprocessing"],
            "sha256": sha256(output),
        }
        output.with_suffix(".json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        output.unlink(missing_ok=True)
        output.with_suffix(".json").unlink(missing_ok=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explicitly download or locally export detector candidates"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download")
    download.add_argument("--url", required=True)
    download.add_argument("--expected-sha256", required=True)
    download.add_argument("--output", type=Path, required=True)

    export = commands.add_parser("export")
    export.add_argument("candidate", choices=tuple(CANDIDATES))
    export.add_argument("--weights", type=Path, required=True)
    export.add_argument("--weights-sha256", required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--source-root", type=Path)
    export.add_argument("--python", type=Path, default=Path(sys.executable))

    arguments = parser.parse_args()
    if arguments.command == "download":
        result = download_verified(
            arguments.url,
            arguments.output,
            arguments.expected_sha256,
        )
    else:
        result = export_candidate(
            arguments.candidate,
            arguments.weights,
            arguments.weights_sha256,
            arguments.output,
            python=arguments.python,
            source_root=arguments.source_root,
        )
    print(result)


if __name__ == "__main__":
    main()
