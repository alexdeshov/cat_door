import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.benchmark_detector import run_probe

PROBE_STATS = {
    "face": {"observations": 1},
    "side": {"observations": 1},
    "combined": {
        "observations": 2,
        "median_inference_ms": 50.0,
        "p95_inference_ms": 60.0,
        "latency_budget_ms": 70.0,
        "latency_passed": True,
    },
}


@pytest.fixture
def probe_model(runtime_path: Path, monkeypatch) -> tuple[Path, dict[str, object]]:
    monkeypatch.setenv("CATDOOR_DATA_DIR", str(runtime_path))
    model = runtime_path / "candidate.onnx"
    model.write_bytes(b"onnx")
    model.with_suffix(".json").write_text(
        json.dumps({"architecture": "yolox_nano", "sha256": "abc"}),
        encoding="utf-8",
    )
    launched: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        launched["command"] = command
        launched["environment"] = kwargs["env"]
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("scripts.benchmark_detector.subprocess.run", fake_run)
    monkeypatch.setattr(
        "scripts.benchmark_detector.read_detector_probe_stats",
        lambda *_args, **_kwargs: PROBE_STATS,
    )
    return model, launched


def test_probe_pins_cpu_by_default_and_writes_local_report(
    runtime_path: Path,
    probe_model: tuple[Path, dict[str, object]],
) -> None:
    """Умолчание закрепляет процессор, а не наследует окружение.

    Проба наследует `os.environ`, и на узле там вполне может стоять
    `CATDOOR_DETECTOR_DEVICE=GPU`. Все прежние отчёты в `detector-probes` сняты
    на процессоре и сравнимы между собой ровно потому, что устройство
    закреплено здесь.
    """
    model, launched = probe_model
    output = runtime_path / "probe.json"

    result = run_probe(model, seconds=30, output=output)
    report = json.loads(result.read_text(encoding="utf-8"))

    environment = launched["environment"]
    assert isinstance(environment, dict)
    assert environment["CATDOOR_DETECTOR_DEVICE"] == "CPU"
    assert environment["CATDOOR_DETECTOR_MODEL_PATH"] == str(model.resolve())
    assert "--probe-seconds" in launched["command"]
    assert report["device"] == "CPU"
    assert report["architecture"] == "yolox_nano"
    assert report["stats"]["combined"]["latency_passed"] is True


def test_probe_passes_the_requested_device_into_the_report(
    runtime_path: Path,
    probe_model: tuple[Path, dict[str, object]],
) -> None:
    model, launched = probe_model
    output = runtime_path / "probe-gpu.json"

    result = run_probe(model, seconds=30, device="GPU", output=output)
    report = json.loads(result.read_text(encoding="utf-8"))

    environment = launched["environment"]
    assert isinstance(environment, dict)
    assert environment["CATDOOR_DETECTOR_DEVICE"] == "GPU"
    assert report["device"] == "GPU"


def test_probe_reports_the_device_openvino_actually_used(
    runtime_path: Path,
    probe_model: tuple[Path, dict[str, object]],
) -> None:
    """Запрошенное устройство — намерение, и доказательством оно не является.

    `AUTO` выбирает сам, `GPU` без прав на устройство просто не поднимется.
    Фактическое приходит из статуса, который проба пишет о себе сама.
    """
    model, _ = probe_model
    status = runtime_path / "runtime" / "recorder-status.json"
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(
        json.dumps({"detector_device_actual": "GPU.0"}),
        encoding="utf-8",
    )

    result = run_probe(model, seconds=30, device="AUTO", output=runtime_path / "probe.json")
    report = json.loads(result.read_text(encoding="utf-8"))

    assert report["device"] == "AUTO"
    assert report["device_actual"] == "GPU.0"
