import argparse
import json
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training.dataset import DatasetItem, load_manifest, validate_manifest
from training.model_package import verify_package

FOREIGN_LABEL = "__foreign__"


@dataclass(frozen=True)
class AcceptanceMetrics:
    samples: int
    known_samples: int
    foreign_samples: int
    correct: int
    false_accepts: int
    false_rejects: int
    uncertain: int
    accuracy: float
    false_accept_rate: float
    false_reject_rate: float
    uncertain_rate: float
    mean_inference_ms: float
    latency_passed: bool
    passed: bool


def calculate_metrics(
    expected: list[str],
    predicted: list[str | None],
    inference_ms: list[float],
    *,
    max_far: float,
    max_frr: float,
    max_mean_inference_ms: float = float("inf"),
) -> AcceptanceMetrics:
    if not expected or len(expected) != len(predicted) or len(expected) != len(inference_ms):
        raise ValueError("expected, predicted and inference_ms must have the same non-zero length")
    known_samples = sum(label != FOREIGN_LABEL for label in expected)
    foreign_samples = len(expected) - known_samples
    correct = sum(
        prediction == label
        for label, prediction in zip(expected, predicted, strict=True)
        if label != FOREIGN_LABEL
    )
    correct += sum(
        prediction is None
        for label, prediction in zip(expected, predicted, strict=True)
        if label == FOREIGN_LABEL
    )
    false_accepts = sum(
        label == FOREIGN_LABEL and prediction is not None
        for label, prediction in zip(expected, predicted, strict=True)
    )
    false_rejects = sum(
        label != FOREIGN_LABEL and prediction != label
        for label, prediction in zip(expected, predicted, strict=True)
    )
    uncertain = sum(prediction is None for prediction in predicted)
    far = false_accepts / foreign_samples if foreign_samples else 0.0
    frr = false_rejects / known_samples if known_samples else 0.0
    mean_inference_ms = sum(inference_ms) / len(inference_ms)
    latency_passed = mean_inference_ms <= max_mean_inference_ms
    return AcceptanceMetrics(
        samples=len(expected),
        known_samples=known_samples,
        foreign_samples=foreign_samples,
        correct=correct,
        false_accepts=false_accepts,
        false_rejects=false_rejects,
        uncertain=uncertain,
        accuracy=correct / len(expected),
        false_accept_rate=far,
        false_reject_rate=frr,
        uncertain_rate=uncertain / len(expected),
        mean_inference_ms=mean_inference_ms,
        latency_passed=latency_passed,
        passed=(
            foreign_samples > 0
            and known_samples > 0
            and far <= max_far
            and frr <= max_frr
            and latency_passed
        ),
    )


def _softmax(values: Any) -> Any:
    import numpy as np

    shifted = values - np.max(values, axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=1, keepdims=True)


def run_onnx_acceptance(
    model_path: Path,
    items: list[DatasetItem],
    dataset_root: Path,
    model_info: dict[str, Any],
    thresholds: dict[str, Any],
) -> AcceptanceMetrics:
    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    labels = list(model_info["labels"])
    image_size = int(model_info.get("image_size", 224))
    mean = np.asarray(model_info.get("normalization_mean", [0.485, 0.456, 0.406]))
    std = np.asarray(model_info.get("normalization_std", [0.229, 0.224, 0.225]))
    accept_score = float(thresholds["accept_score"])
    max_far = float(thresholds["max_false_accept_rate"])
    max_frr = float(thresholds["max_false_reject_rate"])
    max_mean_inference_ms = float(thresholds["max_mean_inference_ms"])
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    expected: list[str] = []
    predicted: list[str | None] = []
    timings: list[float] = []
    for item in items:
        with Image.open(dataset_root / item.image_path) as image:
            array = np.asarray(
                image.convert("RGB").resize((image_size, image_size)),
                dtype=np.float32,
            )
        tensor = ((array / 255.0 - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
        started = time.perf_counter()
        logits = session.run(None, {input_name: tensor})[0]
        timings.append((time.perf_counter() - started) * 1000)
        scores = _softmax(logits)[0]
        best = int(np.argmax(scores))
        predicted.append(labels[best] if float(scores[best]) >= accept_score else None)
        expected.append(item.label)
    return calculate_metrics(
        expected,
        predicted,
        timings,
        max_far=max_far,
        max_frr=max_frr,
        max_mean_inference_ms=max_mean_inference_ms,
    )


def accept_package(
    package_path: Path,
    manifest_path: Path,
    dataset_root: Path,
    model_name: str,
) -> dict[str, Any]:
    verification = verify_package(package_path)
    all_items = load_manifest(manifest_path)
    validate_manifest(all_items, dataset_root)
    items = [item for item in all_items if item.split == "test"]
    if not items:
        raise ValueError("acceptance manifest has no test items")
    with tempfile.TemporaryDirectory(prefix="catdoor-acceptance-") as temporary:
        temporary_path = Path(temporary)
        with zipfile.ZipFile(package_path) as archive:
            archive.extract(model_name, temporary_path)
        metrics = run_onnx_acceptance(
            temporary_path / model_name,
            items,
            dataset_root,
            verification["model_info"],
            verification["thresholds"],
        )
    return {
        "package": package_path.name,
        "model": model_name,
        "metrics": asdict(metrics),
        "activated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Accept a CatDoor ONNX model on Windows")
    parser.add_argument("package", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--model", default="face_model.onnx")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = accept_package(args.package, args.manifest, args.dataset_root, args.model)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["metrics"]["passed"] else 2)


if __name__ == "__main__":
    main()
