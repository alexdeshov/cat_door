import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.quantize_detector import load_calibration, quantize, tensor_builder

MANIFEST = {
    "format": "onnx",
    "architecture": "yolo11n",
    "license": "AGPL-3.0",
    "weights": "yolo11n.pt",
    "input": {
        "name": "images",
        "shape": [1, 3, 8, 8],
        "channel_order": "rgb",
        "scale": 1 / 255,
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "letterbox_value": 114,
        "resize_mode": "letterbox",
        "letterbox_alignment": "center",
    },
    "outputs": ["output0"],
    "cat_class_id": 15,
    "postprocessing": {
        "score_threshold": 0.001,
        "nms_in_graph": True,
        "nms_iou_threshold": 0.7,
        "max_detections": 300,
    },
}

PROVENANCE = {
    "model": "source.onnx",
    "input_size": 8,
    "letterbox_value": 114,
    "resize_mode": "letterbox",
    "letterbox_alignment": "center",
    "channel_order": "rgb",
    "seed": 5,
    "negative_fraction": 0.4,
    "events": [1, 2, 3],
}


def write_model(path: Path, manifest: dict[str, object], payload: bytes = b"onnx") -> Path:
    path.write_bytes(payload)
    import hashlib

    manifest = {**manifest, "sha256": hashlib.sha256(payload).hexdigest()}
    path.with_suffix(".json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def write_calibration(path: Path, model: Path, count: int = 40, **overrides: object) -> Path:
    manifest = json.loads(model.with_suffix(".json").read_text(encoding="utf-8"))
    provenance = {**PROVENANCE, "model_sha256": manifest["sha256"], **overrides}
    frames = np.full((count, 8, 8, 3), 120, dtype=np.uint8)
    np.savez_compressed(
        path,
        frames=frames,
        provenance=np.frombuffer(json.dumps(provenance).encode("utf-8"), dtype=np.uint8),
    )
    return path


class FakeNncf:
    """Подделка NNCF ровно той формы, какую использует скрипт."""

    class ValidationError(RuntimeError):
        pass

    def __init__(self) -> None:
        self.__version__ = "9.9.9"
        self.calls: list[dict[str, object]] = []
        self.QuantizationPreset = SimpleNamespace(MIXED="mixed")

    def IgnoredScope(self, *, types, names, validate):
        return {"types": types, "names": names, "validate": validate}

    def Dataset(self, frames, transform_func):
        return {"frames": frames, "transform": transform_func}

    def quantize(self, model, dataset, *, preset, subset_size, ignored_scope):
        self.calls.append(
            {
                "model": model,
                "dataset": dataset,
                "preset": preset,
                "subset_size": subset_size,
                "ignored_scope": ignored_scope,
            }
        )
        return "quantized-model"


class FakeOnnx:
    def __init__(self, op_types: tuple[str, ...] = ("Conv", "NonMaxSuppression")) -> None:
        self.graph = SimpleNamespace(
            graph=SimpleNamespace(
                node=[SimpleNamespace(op_type=op_type) for op_type in op_types]
            )
        )
        self.saved: list[str] = []

    def load(self, path: str) -> object:
        return self.graph

    def save(self, model: object, path: str) -> None:
        self.saved.append(path)
        Path(path).write_bytes(b"int8-onnx")


@pytest.fixture
def quantization_stack(monkeypatch) -> tuple[FakeNncf, list[list[float]]]:
    nncf = FakeNncf()
    monkeypatch.setitem(sys.modules, "nncf", nncf)
    monkeypatch.setitem(sys.modules, "onnx", FakeOnnx())
    scored: list[list[float]] = [[0.9] * 8, [0.9] * 8]

    def fake_confidences(model_path, manifest, frames):
        return scored.pop(0)

    monkeypatch.setattr("scripts.quantize_detector.confidences_of", fake_confidences)
    return nncf, scored


def test_quantization_writes_a_manifest_that_keeps_the_contract(
    runtime_path: Path,
    quantization_stack: tuple[FakeNncf, list[list[float]]],
) -> None:
    """Всё, что читают детектор и адаптер, обязано доехать без изменений.

    Иначе квантованная модель не пройдёт ни существующую офлайн-оценку, ни
    реестр моделей, и сравнивать её с исходной будет нечем.
    """
    model = write_model(runtime_path / "source.onnx", MANIFEST)
    calibration = write_calibration(runtime_path / "calib.npz", model)
    output = runtime_path / "source-int8.onnx"

    quantize(model, calibration, output, holdout=8)

    manifest = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    source_manifest = json.loads(model.with_suffix(".json").read_text(encoding="utf-8"))
    for key in ("architecture", "input", "outputs", "cat_class_id", "postprocessing"):
        assert manifest[key] == source_manifest[key]
    assert manifest["sha256"] != source_manifest["sha256"]
    assert manifest["precision"] == "int8"
    assert manifest["quantization"]["source_model"]["sha256"] == source_manifest["sha256"]
    assert manifest["quantization"]["nncf_version"] == "9.9.9"
    assert manifest["quantization"]["subset_size"] == 32
    assert manifest["quantization"]["calibration"]["overlaps_evaluation_set"] is True
    assert manifest["quantization"]["self_check"]["frames_with_a_cat"] == 8


def test_a_holdout_without_cats_is_refused(
    runtime_path: Path,
    monkeypatch,
    quantization_stack: tuple[FakeNncf, list[list[float]]],
) -> None:
    """Ноль расхождения на кадрах без котов — не успех, а пустая проверка.

    Отчёт при этом выглядит безупречно: среднее отклонение 0.0, перевёрнутых
    вердиктов ноль. Ровно тот молчаливый ноль, который дороже явного отказа.
    """
    model = write_model(runtime_path / "source.onnx", MANIFEST)
    calibration = write_calibration(runtime_path / "calib.npz", model)
    monkeypatch.setattr(
        "scripts.quantize_detector.confidences_of",
        lambda path, manifest, frames: [0.0] * 8,
    )

    with pytest.raises(ValueError, match="proves nothing"):
        quantize(model, calibration, runtime_path / "out.onnx", holdout=8)


def test_postprocessing_is_excluded_from_quantization(
    runtime_path: Path,
    quantization_stack: tuple[FakeNncf, list[list[float]]],
) -> None:
    """NMS квантованию не подлежит, и исключается по типу операции.

    Имена узлов в сквозном экспорте ultralytics зависят от версии, а
    `NonMaxSuppression` — имя операции ONNX и стабильно.
    """
    nncf, _ = quantization_stack
    model = write_model(runtime_path / "source.onnx", MANIFEST)
    calibration = write_calibration(runtime_path / "calib.npz", model)

    quantize(model, calibration, runtime_path / "out.onnx", holdout=8)

    scope = nncf.calls[0]["ignored_scope"]
    assert scope["types"] == ["NonMaxSuppression"]
    assert scope["validate"] is True
    assert nncf.calls[0]["preset"] == "mixed"


def test_types_absent_from_the_graph_are_not_requested(
    runtime_path: Path,
    monkeypatch,
    quantization_stack: tuple[FakeNncf, list[list[float]]],
) -> None:
    """`validate=True` отвергает тип, которого в графе нет.

    Сузить список до присутствующих — единственный способ сохранить строгую
    проверку имён узлов и не падать на графе без NMS.
    """
    nncf, _ = quantization_stack
    monkeypatch.setitem(sys.modules, "onnx", FakeOnnx(op_types=("Conv", "Add")))
    manifest = {**MANIFEST, "postprocessing": {**MANIFEST["postprocessing"], "nms_in_graph": False}}
    model = write_model(runtime_path / "source.onnx", manifest)
    calibration = write_calibration(runtime_path / "calib.npz", model)

    quantize(model, calibration, runtime_path / "out.onnx", holdout=8)

    assert nncf.calls[0]["ignored_scope"]["types"] == []


def test_manifest_promising_nms_must_match_the_graph(
    runtime_path: Path,
    monkeypatch,
    quantization_stack: tuple[FakeNncf, list[list[float]]],
) -> None:
    monkeypatch.setitem(sys.modules, "onnx", FakeOnnx(op_types=("Conv",)))
    model = write_model(runtime_path / "source.onnx", MANIFEST)
    calibration = write_calibration(runtime_path / "calib.npz", model)

    with pytest.raises(ValueError, match="no NonMaxSuppression"):
        quantize(model, calibration, runtime_path / "out.onnx", holdout=8)


def test_a_drifted_model_leaves_no_files_behind(
    runtime_path: Path,
    monkeypatch,
    quantization_stack: tuple[FakeNncf, list[list[float]]],
) -> None:
    """Модель, которая втрое быстрее и слепая, до узла доехать не должна."""
    model = write_model(runtime_path / "source.onnx", MANIFEST)
    calibration = write_calibration(runtime_path / "calib.npz", model)
    monkeypatch.setattr(
        "scripts.quantize_detector.confidences_of",
        lambda path, manifest, frames: (
            [0.9] * 8 if path == model else [0.1] * 8
        ),
    )
    output = runtime_path / "out.onnx"

    with pytest.raises(ValueError, match="drifted"):
        quantize(model, calibration, output, holdout=8)

    assert not output.exists()
    assert not output.with_suffix(".json").exists()


def test_quantization_refuses_to_overwrite(
    runtime_path: Path,
    quantization_stack: tuple[FakeNncf, list[list[float]]],
) -> None:
    model = write_model(runtime_path / "source.onnx", MANIFEST)
    calibration = write_calibration(runtime_path / "calib.npz", model)
    output = runtime_path / "out.onnx"
    output.write_bytes(b"already here")

    with pytest.raises(FileExistsError):
        quantize(model, calibration, output, holdout=8)


def test_source_model_must_match_its_own_manifest(
    runtime_path: Path,
    quantization_stack: tuple[FakeNncf, list[list[float]]],
) -> None:
    model = write_model(runtime_path / "source.onnx", MANIFEST)
    calibration = write_calibration(runtime_path / "calib.npz", model)
    model.write_bytes(b"a different model entirely")

    # Соглашение о контрольных суммах импортируется из
    # `scripts.prepare_detector_candidate`, второй его реализации быть не должно.
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        quantize(model, calibration, runtime_path / "out.onnx", holdout=8)


def test_calibration_built_for_another_input_size_is_rejected(runtime_path: Path) -> None:
    """Набор для 640-й модели на 416-й «сработал» бы и дал мусорные диапазоны."""
    model = write_model(runtime_path / "source.onnx", MANIFEST)
    calibration = write_calibration(runtime_path / "calib.npz", model, input_size=640)
    manifest = json.loads(model.with_suffix(".json").read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="different preprocessing"):
        load_calibration(calibration, manifest)


def test_calibration_built_for_another_model_is_rejected(runtime_path: Path) -> None:
    model = write_model(runtime_path / "source.onnx", MANIFEST)
    calibration = write_calibration(runtime_path / "calib.npz", model, model_sha256="deadbeef")
    manifest = json.loads(model.with_suffix(".json").read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="different model"):
        load_calibration(calibration, manifest)


def test_calibration_smaller_than_the_holdout_is_rejected(
    runtime_path: Path,
    quantization_stack: tuple[FakeNncf, list[list[float]]],
) -> None:
    model = write_model(runtime_path / "source.onnx", MANIFEST)
    calibration = write_calibration(runtime_path / "calib.npz", model, count=8)

    with pytest.raises(ValueError, match="not enough"):
        quantize(model, calibration, runtime_path / "out.onnx", holdout=8)


def test_calibration_tensor_matches_what_inference_feeds_the_model() -> None:
    """Калибровка обязана видеть ровно то, что видит инференс."""
    from src.inference.cat_detector import build_detector_tensor

    frame = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    build = tensor_builder(MANIFEST)

    expected = build_detector_tensor(
        frame,
        channel_order="rgb",
        scale=1 / 255,
        mean=np.zeros(3, dtype=np.float32),
        std=np.ones(3, dtype=np.float32),
    )
    assert np.array_equal(build(frame)["images"], expected)
