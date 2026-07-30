import json
from pathlib import Path
from types import SimpleNamespace

from scripts.benchmark_detector import run_probe


def test_probe_forces_cpu_and_writes_local_report(
    runtime_path: Path,
    monkeypatch,
) -> None:
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
        lambda *_args, **_kwargs: {
            "face": {"observations": 1},
            "side": {"observations": 1},
            "combined": {
                "observations": 2,
                "median_inference_ms": 50.0,
                "p95_inference_ms": 60.0,
                "latency_budget_ms": 70.0,
                "latency_passed": True,
            },
        },
    )
    output = runtime_path / "probe.json"

    result = run_probe(model, seconds=30, output=output)
    report = json.loads(result.read_text(encoding="utf-8"))

    environment = launched["environment"]
    assert isinstance(environment, dict)
    assert environment["CATDOOR_DETECTOR_DEVICE"] == "CPU"
    assert environment["CATDOOR_DETECTOR_MODEL_PATH"] == str(model.resolve())
    assert "--probe-seconds" in launched["command"]
    assert report["architecture"] == "yolox_nano"
    assert report["stats"]["combined"]["latency_passed"] is True
