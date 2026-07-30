from __future__ import annotations

import argparse
import json
import shutil
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sqlalchemy import select

from src.common.database import (
    EvaluationRun,
    Event,
    EventPrediction,
    create_session_factory,
)
from src.common.detection_config import (
    DetectionConfig,
    detection_config_path,
    load_detection_config,
)
from src.common.evaluation import (
    EvaluatedEvent,
    ViewEvaluation,
    build_identity_evaluation_report,
    empty_view_evaluation,
    evaluate_view_frames,
)
from src.common.settings import Settings
from src.inference.cat_detector import OpenVinoCatDetector
from src.inference.cat_identity import OpenVinoCatIdentity
from training.workflow import extract_frames, write_status


def _load_filter(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("evaluation filter must be an object")
    return value


def _event_query(filter_value: dict[str, Any]) -> Any:
    query = select(Event).order_by(Event.created_at)
    labels = filter_value.get("labels")
    if labels:
        query = query.where(Event.label.in_([str(label) for label in labels]))
    training_targets = filter_value.get("training_targets")
    if training_targets:
        query = query.where(
            Event.training_target.in_([str(target) for target in training_targets])
        )
    if filter_value.get("created_from"):
        query = query.where(
            Event.created_at >= datetime.fromisoformat(str(filter_value["created_from"]))
        )
    if filter_value.get("created_to"):
        query = query.where(
            Event.created_at <= datetime.fromisoformat(str(filter_value["created_to"]))
        )
    return query


def _load_frames(
    ffmpeg_path: str,
    source: Path,
    destination: Path,
) -> list[np.ndarray]:
    paths = extract_frames(
        ffmpeg_path,
        source,
        destination / "frame-%02d.jpg",
        rotate_180=True,
    )
    frames: list[np.ndarray] = []
    for frame_path in paths:
        with Image.open(frame_path) as image:
            frames.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
    return frames


def _save_prediction(
    factory: Any,
    *,
    run_id: int,
    event_id: int,
    model_version_id: int,
    view: str,
    result: ViewEvaluation,
) -> None:
    with factory() as session:
        session.add(
            EventPrediction(
                event_id=event_id,
                model_version_id=model_version_id,
                view=view,
                origin="evaluation",
                evaluation_run_id=run_id,
                decision=result.aggregate.decision,
                predicted_name=result.aggregate.predicted_name,
                confidence=result.aggregate.confidence,
                frames_used=result.aggregate.frames_used,
                max_detector_confidence=result.max_detector_confidence,
                mean_inference_ms=result.mean_identity_inference_ms,
                raw_scores_json=json.dumps(
                    [asdict_score(score) for score in result.raw_scores],
                    ensure_ascii=False,
                ),
                created_at=datetime.now(UTC),
            )
        )
        session.commit()


def asdict_score(score: Any) -> dict[str, object]:
    return {"name": score.name, "confidence": score.confidence}


def run_evaluation(
    database_path: Path,
    data_dir: Path,
    ffmpeg_path: str,
    run_id: int,
    detector_model: Path,
    identity_models: dict[str, Path],
    *,
    identity_threshold: float,
    status_path: Path,
) -> None:
    factory = create_session_factory(database_path)
    temporary_root = data_dir / "runtime" / "evaluations" / f"{run_id:08d}"
    try:
        with factory() as session:
            run = session.get(EvaluationRun, run_id)
            if run is None:
                raise ValueError(f"evaluation run {run_id} does not exist")
            run.status = "running"
            session.commit()
            filter_value = _load_filter(run.filter_json)
            events = session.scalars(_event_query(filter_value)).all()
            model_version_ids = {
                "face": run.face_model_version_id,
                "side": run.side_model_version_id,
            }
        if not events:
            raise ValueError("evaluation filter did not select any events")
        detector = OpenVinoCatDetector(detector_model, "CPU")
        identities = {
            role: OpenVinoCatIdentity(path, "CPU")
            for role, path in identity_models.items()
        }
        defaults = DetectionConfig.from_settings(Settings(data_dir=data_dir))
        runtime_config = load_detection_config(
            detection_config_path(data_dir),
            defaults,
        )
        results: list[EvaluatedEvent] = []
        root = data_dir.resolve()
        write_status(
            status_path,
            state="running",
            message="Офлайн-оценка сохранённых событий",
            run_id=run_id,
            completed=0,
            total=len(events),
        )
        for index, event in enumerate(events, 1):
            event_directory = temporary_root / f"{event.id:08d}"
            views: dict[str, ViewEvaluation] = {}
            try:
                for role, model in identities.items():
                    relative_media = (
                        event.media_path if role == "face" else event.side_media_path
                    )
                    result = empty_view_evaluation()
                    if relative_media:
                        source = (root / relative_media).resolve()
                        if root not in source.parents or not source.is_file():
                            raise ValueError(
                                f"unsafe or missing media for event {event.id}, view {role}"
                            )
                        frames = _load_frames(
                            ffmpeg_path,
                            source,
                            event_directory / role,
                        )
                        result = evaluate_view_frames(
                            frames,
                            detector,
                            model,
                            getattr(runtime_config, role),
                            identity_threshold,
                        )
                    views[role] = result
                    model_version_id = model_version_ids[role]
                    if model_version_id is None:
                        raise ValueError(f"missing model version for {role}")
                    _save_prediction(
                        factory,
                        run_id=run_id,
                        event_id=event.id,
                        model_version_id=model_version_id,
                        view=role,
                        result=result,
                    )
            finally:
                if event_directory.exists():
                    shutil.rmtree(event_directory)
            results.append(
                EvaluatedEvent(
                    event_id=event.id,
                    label=event.label,
                    cat_name=event.cat_name,
                    training_target=event.training_target,
                    created_at=event.created_at,
                    views=views,
                )
            )
            write_status(
                status_path,
                state="running",
                message=f"Обработано событий: {index} из {len(events)}",
                run_id=run_id,
                completed=index,
                total=len(events),
            )
        evaluated_views = list(identities)
        metrics = build_identity_evaluation_report(
            results,
            evaluated_views,
            {
                role: getattr(runtime_config, role).confidence_threshold
                for role in evaluated_views
            },
            identity_threshold=identity_threshold,
        )
        with factory() as session:
            run = session.get(EvaluationRun, run_id)
            assert run is not None
            run.status = "completed"
            run.completed_at = datetime.now(UTC)
            run.metrics_json = json.dumps(metrics, ensure_ascii=False)
            session.commit()
        write_status(
            status_path,
            state="completed",
            message="Офлайн-оценка завершена",
            run_id=run_id,
            completed=len(events),
            total=len(events),
            metrics=metrics,
        )
    except Exception as exc:  # noqa: BLE001 - detached job must persist failures
        with factory() as session:
            run = session.get(EvaluationRun, run_id)
            if run is not None:
                run.status = "failed"
                run.completed_at = datetime.now(UTC)
                run.error_message = str(exc)[:1000]
                session.commit()
        write_status(
            status_path,
            state="failed",
            message=str(exc),
            run_id=run_id,
            traceback=traceback.format_exc(),
        )
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay saved CatDoor events on OpenVINO CPU")
    parser.add_argument("database_path", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("ffmpeg_path")
    parser.add_argument("run_id", type=int)
    parser.add_argument("detector_model", type=Path)
    parser.add_argument("status_path", type=Path)
    parser.add_argument("--face-model", type=Path)
    parser.add_argument("--side-model", type=Path)
    parser.add_argument("--identity-threshold", type=float, default=0.8)
    args = parser.parse_args()
    identities = {
        role: path
        for role, path in (
            ("face", args.face_model),
            ("side", args.side_model),
        )
        if path is not None
    }
    run_evaluation(
        args.database_path,
        args.data_dir,
        args.ffmpeg_path,
        args.run_id,
        args.detector_model,
        identities,
        identity_threshold=args.identity_threshold,
        status_path=args.status_path,
    )


if __name__ == "__main__":
    main()
