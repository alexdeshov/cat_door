import json
from pathlib import Path

import numpy as np
import pytest

from scripts.benchmark_detector_model import load_frames, run_benchmark

INPUT_SIZE = 8


class FakeDetector:
    """Детектор, чья уверенность зависит только от имени файла модели.

    Так «квантованная» модель отличается от базовой предсказуемо, и сравнение
    можно проверить, не поднимая OpenVINO.
    """

    def __init__(self, model_path: Path, device: str) -> None:
        self.model_name = model_path.name
        self.model_sha256 = "0" * 64
        self.architecture = "yolo11n"
        self.input_size = INPUT_SIZE
        self.device = f"{device}.0"
        self._shift = 0.3 if "int8" in model_path.name else 0.0
        self.calls = 0

    def detect(self, frame: np.ndarray) -> object:
        self.calls += 1
        confidence = float(frame[0, 0, 0]) / 255.0
        return type("Result", (), {"confidence": max(0.0, confidence - self._shift)})()


@pytest.fixture
def models(runtime_path: Path, monkeypatch) -> list[Path]:
    manifest = json.dumps({"input": {"shape": [1, 3, INPUT_SIZE, INPUT_SIZE]}})
    paths = []
    for name in ("base.onnx", "base-int8.onnx"):
        model = runtime_path / name
        model.write_bytes(b"onnx")
        model.with_suffix(".json").write_text(manifest, encoding="utf-8")
        paths.append(model)
    monkeypatch.setattr(
        "scripts.benchmark_detector_model.OpenVinoCatDetector",
        FakeDetector,
    )
    return paths


def frames_archive(path: Path, count: int = 4) -> Path:
    frames = np.zeros((count, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    for index in range(count):
        frames[index, :, :, :] = 50 * index + 100
    np.savez(path, frames=frames)
    return path


def test_every_model_and_device_pair_appears_in_the_report(
    runtime_path: Path,
    models: list[Path],
) -> None:
    report = run_benchmark(
        models,
        devices=["CPU", "GPU"],
        warmup=2,
        iterations=4,
        frames_path=None,
        verdict_threshold=0.5,
        seed=1,
    )

    pairs = {(cell["model"], cell["device_requested"]) for cell in report["cells"]}
    assert pairs == {
        ("base.onnx", "CPU"),
        ("base-int8.onnx", "CPU"),
        ("base.onnx", "GPU"),
        ("base-int8.onnx", "GPU"),
    }
    assert all(cell["device_actual"].endswith(".0") for cell in report["cells"])


def test_warmup_runs_but_stays_out_of_the_statistics(
    runtime_path: Path,
    models: list[Path],
    monkeypatch,
) -> None:
    """Первые вызовы на видеоядре включают компиляцию ядер и стоят в разы больше."""
    built: list[FakeDetector] = []
    original = FakeDetector

    def remember(model_path: Path, device: str) -> FakeDetector:
        detector = original(model_path, device)
        built.append(detector)
        return detector

    monkeypatch.setattr("scripts.benchmark_detector_model.OpenVinoCatDetector", remember)

    report = run_benchmark(
        models[:1],
        devices=["CPU"],
        warmup=7,
        iterations=5,
        frames_path=None,
        verdict_threshold=0.5,
        seed=1,
    )

    assert built[0].calls == 12
    assert report["cells"][0]["iterations"] == 5
    assert report["cells"][0]["warmup"] == 7


def test_confidence_deltas_need_real_frames(
    runtime_path: Path,
    models: list[Path],
) -> None:
    """На шуме расхождение уверенностей выглядело бы замером, не будучи им."""
    noise = run_benchmark(
        models,
        devices=["CPU"],
        warmup=1,
        iterations=4,
        frames_path=None,
        verdict_threshold=0.5,
        seed=1,
    )
    real = run_benchmark(
        models,
        devices=["CPU"],
        warmup=1,
        iterations=4,
        frames_path=frames_archive(runtime_path / "calibration.npz"),
        verdict_threshold=0.5,
        seed=1,
    )

    noise_candidate = noise["cells"][1]["versus_baseline"]
    assert "speedup" in noise_candidate
    assert "mean_abs_confidence_delta" not in noise_candidate

    real_candidate = real["cells"][1]["versus_baseline"]
    assert real_candidate["mean_abs_confidence_delta"] == pytest.approx(0.3, abs=0.01)
    # Кадры дают уверенности 0.39, 0.59, 0.78, 0.98; сдвиг на 0.3 переводит через
    # порог 0.5 два средних, а крайние оставляет по свою сторону.
    assert real_candidate["detections_changed"] == 2


def test_benchmark_refuses_to_overwrite_a_report(
    runtime_path: Path,
    models: list[Path],
) -> None:
    output = runtime_path / "report.json"
    output.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_benchmark(
            models[:1],
            devices=["CPU"],
            warmup=1,
            iterations=2,
            frames_path=None,
            verdict_threshold=0.5,
            seed=1,
            output=output,
        )


def test_frames_of_the_wrong_size_are_rejected(runtime_path: Path) -> None:
    """Набор, собранный для другой модели, «сработал» бы и дал мусор."""
    path = runtime_path / "wrong.npz"
    np.savez(path, frames=np.zeros((2, INPUT_SIZE + 1, INPUT_SIZE + 1, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="expected"):
        load_frames(path, input_size=INPUT_SIZE, count=2, seed=1)
