from pathlib import Path
from typing import Any

from training import evaluation_job


def capture_run(monkeypatch, argv: list[str]) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_run(*args: object, **kwargs: object) -> None:
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(evaluation_job, "run_evaluation", fake_run)
    monkeypatch.setattr("sys.argv", ["evaluation_job", *argv])
    evaluation_job.main()
    return captured


BASE_ARGV = ["db.sqlite3", "data", "ffmpeg", "7", "model.onnx", "status.json"]


def test_evaluation_defaults_to_the_processor(monkeypatch) -> None:
    """Умолчание закрепляет процессор, чтобы прошлые отчёты остались сравнимы.

    `mean_detector_inference_ms` хранится и сравнивается между запусками; как
    только оно может приехать с двух устройств, умолчание перестаёт быть
    вопросом вкуса.
    """
    captured = capture_run(monkeypatch, BASE_ARGV)

    assert captured["kwargs"]["detector_device"] == "CPU"


def test_requested_device_reaches_the_detector(monkeypatch) -> None:
    captured = capture_run(monkeypatch, [*BASE_ARGV, "--detector-device", "GPU"])

    assert captured["kwargs"]["detector_device"] == "GPU"
    assert captured["args"][4] == Path("model.onnx")
