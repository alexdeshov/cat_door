import json
import shutil
import subprocess
import sys
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.common.database import (
    EvaluationRun,
    Event,
    EventPrediction,
    ModelVersion,
    create_session_factory,
    session_dependency,
)
from src.common.detection_config import (
    DetectionConfig,
    detection_config_path,
    load_detection_config,
    save_detection_config,
)
from src.common.detection_stats import read_detection_stats
from src.common.model_registry import register_model_version
from src.common.online_metrics import build_online_metrics
from src.common.settings import Settings
from src.common.windows_power import SleepInhibitor
from src.recorder.ffmpeg import RecorderError, record_clip
from src.recorder.preview import local_mjpeg_stream, preview_path
from src.recorder.status import read_recorder_status
from training.workflow import create_macbook_export, write_status

LABELS = {"pending", "known_cat", "foreign_cat", "not_visible", "multiple_cats"}
TRAINING_TARGETS = {"excluded", "macbook", "local"}


class CapturePayload(BaseModel):
    duration_seconds: int = Field(ge=3, le=30)


class LabelPayload(BaseModel):
    label: str
    cat_name: str | None = Field(default=None, max_length=100)
    training_target: str = "excluded"


class BulkLabelPayload(LabelPayload):
    event_ids: list[int] = Field(min_length=1, max_length=100)


class EvaluationStartPayload(BaseModel):
    created_from: datetime | None = None
    created_to: datetime | None = None
    labels: list[str] = Field(
        default_factory=lambda: [
            "known_cat",
            "foreign_cat",
            "pending",
            "multiple_cats",
        ]
    )
    training_targets: list[str] = Field(default_factory=list)
    detector_model: str | None = None
    face_model: str | None = None
    side_model: str | None = None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    factory = create_session_factory(settings.resolved_database_path)
    templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

    app = FastAPI(title="CatDoor AI", version="0.1.0")
    app.state.settings = settings
    app.state.session_factory = factory

    def current_detection_config() -> DetectionConfig:
        return load_detection_config(
            detection_config_path(settings.data_dir),
            DetectionConfig.from_settings(settings),
        )

    def get_session() -> Generator[Session, None, None]:
        yield from session_dependency(factory)

    def resolve_evaluation_model(
        configured: Path | None,
        requested: str | None,
    ) -> Path | None:
        if requested is None:
            return configured
        candidate = Path(requested)
        if candidate.is_absolute() or ".." in candidate.parts or candidate.suffix != ".onnx":
            raise HTTPException(422, "Имя модели должно быть безопасным относительным ONNX-путём")
        models_root = settings.detector_model_path.parent.resolve()
        resolved = (models_root / candidate).resolve()
        if models_root not in resolved.parents:
            raise HTTPException(422, "Модель должна находиться в локальном каталоге models")
        return resolved

    @app.get("/", response_class=HTMLResponse)
    def index(
        request: Request,
        session: Annotated[Session, Depends(get_session)],
    ) -> HTMLResponse:
        events = session.scalars(select(Event).order_by(Event.created_at.desc()).limit(100)).all()
        event_ids = [event.id for event in events]
        prediction_rows = (
            session.scalars(
                select(EventPrediction)
                .where(
                    EventPrediction.event_id.in_(event_ids),
                    EventPrediction.origin == "online",
                )
                .order_by(EventPrediction.created_at.desc(), EventPrediction.id.desc())
            ).all()
            if event_ids
            else []
        )
        event_predictions: dict[int, dict[str, EventPrediction]] = {}
        for prediction in prediction_rows:
            by_view = event_predictions.setdefault(prediction.event_id, {})
            by_view.setdefault(prediction.view, prediction)
        summary_row = session.execute(
            select(
                func.count(Event.id),
                func.count(Event.id).filter(Event.trigger_source == "automatic"),
                func.count(Event.id).filter(Event.label == "pending"),
                func.count(Event.id).filter(Event.label == "known_cat"),
                func.count(Event.id).filter(Event.label == "foreign_cat"),
                func.count(Event.id).filter(Event.status == "failed"),
            )
        ).one()
        event_summary = {
            "total": summary_row[0],
            "automatic": summary_row[1],
            "manual": summary_row[0] - summary_row[1],
            "pending": summary_row[2],
            "known": summary_row[3],
            "foreign": summary_row[4],
            "failed": summary_row[5],
        }
        known_cat_names = session.scalars(
            select(Event.cat_name)
            .where(Event.label == "known_cat", Event.cat_name.is_not(None))
            .distinct()
            .order_by(Event.cat_name)
        ).all()
        target_counts = {
            target: sum(event.training_target == target for event in events)
            for target in ("macbook", "local")
        }
        local_runs = sorted((settings.data_dir / "training" / "local").glob("*/status.json"))
        local_status = None
        if local_runs:
            try:
                local_status = json.loads(local_runs[-1].read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                local_status = {"state": "unknown", "message": "Статус запуска повреждён"}
        recorder_status = read_recorder_status(
            settings.data_dir / "runtime" / "recorder-status.json"
        )
        recorder_enabled = (
            bool(recorder_status.get("automatic_capture_enabled"))
            if recorder_status
            else settings.automatic_capture_enabled
        )
        recorder_dry_run = (
            bool(recorder_status.get("dry_run"))
            if recorder_status
            else settings.automatic_capture_dry_run
        )
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "events": events,
                "event_predictions": event_predictions,
                "face_configured": bool(settings.camera_face_rtsp),
                "side_configured": bool(settings.camera_side_rtsp),
                "default_duration": settings.capture_duration_seconds,
                "target_counts": target_counts,
                "event_summary": event_summary,
                "known_cat_names": known_cat_names,
                "local_status": local_status,
                "local_training_epochs": settings.local_training_epochs,
                "recorder_status": recorder_status,
                "automatic_capture_enabled": recorder_enabled,
                "automatic_capture_dry_run": recorder_dry_run,
                "stream_version": uuid4().hex,
                "detection_config": current_detection_config(),
            },
        )

    @app.get("/stream/{view}.mjpeg")
    def stream(view: str) -> StreamingResponse:
        urls = {"face": settings.camera_face_rtsp, "side": settings.camera_side_rtsp}
        if view not in urls:
            raise HTTPException(404, "Неизвестная камера")
        if not urls[view]:
            raise HTTPException(503, "Камера не настроена")
        return StreamingResponse(
            local_mjpeg_stream(preview_path(settings.data_dir, view), settings.preview_fps),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/recorder/status")
    def recorder_status() -> dict[str, object]:
        status = read_recorder_status(settings.data_dir / "runtime" / "recorder-status.json")
        if status is None:
            return {
                "state": "offline",
                "message": "CatDoorRecorder не запущен",
                "stale": True,
            }
        updated_at = status.get("updated_at")
        try:
            updated = datetime.fromisoformat(str(updated_at))
            stale = (datetime.now(UTC) - updated).total_seconds() > 10
        except ValueError:
            stale = True
        return {**status, "stale": stale}

    @app.get("/api/detection/settings")
    def get_detection_settings() -> DetectionConfig:
        return current_detection_config()

    @app.put("/api/detection/settings")
    def update_detection_settings(payload: DetectionConfig) -> DetectionConfig:
        try:
            save_detection_config(detection_config_path(settings.data_dir), payload)
        except OSError as exc:
            raise HTTPException(500, "Не удалось сохранить настройки детекции") from exc
        return payload

    @app.post("/api/events", status_code=201)
    def capture(
        payload: CapturePayload,
        session: Annotated[Session, Depends(get_session)],
    ) -> dict[str, object]:
        if not settings.camera_face_rtsp:
            raise HTTPException(503, "Камера face не настроена")
        event = Event(duration_seconds=payload.duration_seconds)
        session.add(event)
        session.commit()
        event_dir = settings.data_dir / "events" / f"{event.id:08d}"
        recordings = {"face": (settings.camera_face_rtsp, event_dir / "face.mp4")}
        if settings.camera_side_rtsp:
            recordings["side"] = (settings.camera_side_rtsp, event_dir / "side.mp4")
        errors: list[str] = []
        completed: set[str] = set()
        with ThreadPoolExecutor(max_workers=len(recordings)) as executor:
            futures = {
                executor.submit(
                    record_clip,
                    settings.ffmpeg_path,
                    url,
                    destination,
                    payload.duration_seconds,
                ): (view, destination)
                for view, (url, destination) in recordings.items()
            }
            for future in as_completed(futures):
                view, destination = futures[future]
                try:
                    future.result()
                    completed.add(view)
                    relative = destination.relative_to(settings.data_dir).as_posix()
                    if view == "face":
                        event.media_path = relative
                    else:
                        event.side_media_path = relative
                except RecorderError as exc:
                    errors.append(f"{view}:{exc}")
        event.status = "ready" if not errors else ("partial" if completed else "failed")
        event.error_code = ",".join(errors) or None
        session.commit()
        return {"id": event.id, "status": event.status}

    @app.post("/api/events/{event_id}/label")
    def label_event(
        event_id: int,
        payload: LabelPayload,
        session: Annotated[Session, Depends(get_session)],
    ) -> dict[str, object]:
        event = session.get(Event, event_id)
        if not event:
            raise HTTPException(404, "Событие не найдено")
        if payload.label not in LABELS:
            raise HTTPException(422, "Неизвестный тип разметки")
        cat_name = payload.cat_name.strip() if payload.cat_name else None
        if payload.label == "known_cat" and not cat_name:
            raise HTTPException(422, "Для домашнего кота укажите имя")
        if payload.label != "known_cat" and cat_name:
            raise HTTPException(422, "Имя допустимо только для домашнего кота")
        if payload.training_target not in TRAINING_TARGETS:
            raise HTTPException(422, "Неизвестное назначение обучения")
        if payload.training_target != "excluded" and payload.label not in {
            "known_cat",
            "foreign_cat",
        }:
            raise HTTPException(
                422,
                "Для обучения можно выбрать только домашнего или чужого кота",
            )
        event.label = payload.label
        event.cat_name = cat_name
        event.training_target = payload.training_target
        session.commit()
        return {
            "id": event.id,
            "label": event.label,
            "cat_name": event.cat_name,
            "training_target": event.training_target,
        }

    @app.post("/api/events/bulk-label")
    def bulk_label_events(
        payload: BulkLabelPayload,
        session: Annotated[Session, Depends(get_session)],
    ) -> dict[str, object]:
        if payload.label != "known_cat":
            raise HTTPException(422, "Группировать можно только события одного домашнего кота")
        cat_name = payload.cat_name.strip() if payload.cat_name else None
        if not cat_name:
            raise HTTPException(422, "Укажите имя кота")
        if payload.training_target not in {"macbook", "excluded"}:
            raise HTTPException(422, "Обучение выполняется офлайн на MacBook")
        unique_ids = sorted(set(payload.event_ids))
        events = session.scalars(select(Event).where(Event.id.in_(unique_ids))).all()
        if len(events) != len(unique_ids):
            raise HTTPException(404, "Одно или несколько событий не найдены")
        for event in events:
            event.label = "known_cat"
            event.cat_name = cat_name
            event.training_target = payload.training_target
        session.commit()
        return {"updated": len(events), "cat_name": cat_name}

    @app.delete("/api/events/{event_id}", status_code=204)
    def delete_event(
        event_id: int,
        session: Annotated[Session, Depends(get_session)],
    ) -> None:
        event = session.get(Event, event_id)
        if not event:
            raise HTTPException(404, "Событие не найдено")

        events_root = (settings.data_dir / "events").resolve()
        event_dir = (events_root / f"{event.id:08d}").resolve()
        if events_root not in event_dir.parents:
            raise HTTPException(500, "Небезопасный путь события")
        if event_dir.exists():
            try:
                shutil.rmtree(event_dir)
            except OSError as exc:
                raise HTTPException(500, "Не удалось удалить файлы события") from exc

        session.delete(event)
        session.commit()

    @app.get("/api/training/macbook/export")
    def export_for_macbook() -> FileResponse:
        try:
            archive = create_macbook_export(
                settings.resolved_database_path,
                settings.data_dir,
                settings.ffmpeg_path,
                settings.detector_model_path,
                settings.detector_device,
                DetectionConfig.from_settings(settings),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return FileResponse(
            archive,
            media_type="application/zip",
            filename=archive.name,
        )

    @app.post("/api/training/local/start", status_code=202)
    def start_local_training() -> dict[str, object]:
        runs_root = settings.data_dir / "training" / "local"
        existing = sorted(runs_root.glob("*/status.json"))
        if existing:
            try:
                latest = json.loads(existing[-1].read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                latest = {}
            if latest.get("state") in {"starting", "preparing", "training"}:
                raise HTTPException(409, "Локальное обучение уже выполняется")

        run_id = f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
        output_dir = runs_root / run_id
        status_path = output_dir / "status.json"
        write_status(status_path, state="starting", message="Запуск отдельного процесса")
        log_path = output_dir / "training.log"
        command = [
            sys.executable,
            "-m",
            "training.local_job",
            str(settings.resolved_database_path.resolve()),
            str(settings.data_dir.resolve()),
            settings.ffmpeg_path,
            str(output_dir.resolve()),
            "--detector-model",
            str(settings.detector_model_path.resolve()),
            "--detector-device",
            settings.detector_device,
            "--epochs",
            str(settings.local_training_epochs),
        ]
        if settings.identity_pretrained_weights_path is not None:
            command.extend(
                [
                    "--weights-path",
                    str(settings.identity_pretrained_weights_path.resolve()),
                ]
            )
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        try:
            with log_path.open("ab") as log:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=Path.cwd(),
                    creationflags=creationflags,
                )
        except OSError as exc:
            write_status(status_path, state="failed", message="Не удалось запустить Python")
            raise HTTPException(500, "Не удалось запустить процесс обучения") from exc
        return {"run_id": run_id, "pid": process.pid, "state": "starting"}

    @app.get("/api/training/local/status")
    def local_training_status() -> dict[str, object]:
        statuses = sorted((settings.data_dir / "training" / "local").glob("*/status.json"))
        if not statuses:
            return {"state": "idle", "message": "Локальное обучение ещё не запускалось"}
        try:
            return json.loads(statuses[-1].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(500, "Не удалось прочитать статус обучения") from exc

    @app.post("/api/evaluations", status_code=202)
    def start_evaluation(
        payload: EvaluationStartPayload,
        session: Annotated[Session, Depends(get_session)],
    ) -> dict[str, object]:
        invalid_labels = sorted(set(payload.labels) - LABELS)
        invalid_targets = sorted(set(payload.training_targets) - TRAINING_TARGETS)
        if invalid_labels:
            raise HTTPException(422, f"Неизвестные метки: {', '.join(invalid_labels)}")
        if invalid_targets:
            raise HTTPException(
                422,
                f"Неизвестные назначения обучения: {', '.join(invalid_targets)}",
            )
        if (
            payload.created_from is not None
            and payload.created_to is not None
            and payload.created_from > payload.created_to
        ):
            raise HTTPException(422, "Начало диапазона должно быть раньше окончания")
        running = session.scalar(
            select(EvaluationRun)
            .where(EvaluationRun.status.in_(["starting", "running"]))
            .limit(1)
        )
        if running is not None:
            raise HTTPException(409, "Офлайн-оценка уже выполняется")

        detector_path = resolve_evaluation_model(
            settings.detector_model_path,
            payload.detector_model,
        )
        face_path = resolve_evaluation_model(
            settings.identity_face_model_path,
            payload.face_model,
        )
        side_path = resolve_evaluation_model(
            settings.identity_side_model_path,
            payload.side_model,
        )
        if detector_path is None or not detector_path.is_file():
            raise HTTPException(422, "Локальная detector-модель не найдена")
        identity_paths = {
            role: path
            for role, path in (("face", face_path), ("side", side_path))
            if path is not None and path.is_file()
        }
        if not identity_paths:
            raise HTTPException(422, "Не установлена ни одна локальная identity-модель")
        try:
            detector_version = register_model_version(
                session,
                kind="detector",
                model_path=detector_path,
            )
            identity_versions = {
                role: register_model_version(
                    session,
                    kind=f"identity_{role}",
                    model_path=path,
                )
                for role, path in identity_paths.items()
            }
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        filter_value = {
            "created_from": (
                payload.created_from.isoformat() if payload.created_from else None
            ),
            "created_to": payload.created_to.isoformat() if payload.created_to else None,
            "labels": payload.labels,
            "training_targets": payload.training_targets,
        }
        run = EvaluationRun(
            run_kind="identity",
            filter_json=json.dumps(filter_value, ensure_ascii=False),
            detector_model_version_id=detector_version.id,
            face_model_version_id=(
                identity_versions["face"].id if "face" in identity_versions else None
            ),
            side_model_version_id=(
                identity_versions["side"].id if "side" in identity_versions else None
            ),
            status="starting",
        )
        session.add(run)
        session.commit()
        run_directory = settings.data_dir / "evaluations" / f"{run.id:08d}"
        status_path = run_directory / "status.json"
        write_status(
            status_path,
            state="starting",
            message="Запуск отдельного процесса оценки",
            run_id=run.id,
        )
        log_path = run_directory / "evaluation.log"
        command = [
            sys.executable,
            "-m",
            "training.evaluation_job",
            str(settings.resolved_database_path.resolve()),
            str(settings.data_dir.resolve()),
            settings.ffmpeg_path,
            str(run.id),
            str(detector_path.resolve()),
            str(status_path.resolve()),
            "--identity-threshold",
            str(settings.identity_confidence_threshold),
        ]
        for role in ("face", "side"):
            if role in identity_paths:
                command.extend(
                    [f"--{role}-model", str(identity_paths[role].resolve())]
                )
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        try:
            with log_path.open("ab") as log:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=Path.cwd(),
                    creationflags=creationflags,
                )
        except OSError as exc:
            run.status = "failed"
            run.completed_at = datetime.now(UTC)
            run.error_message = "Не удалось запустить Python"
            session.commit()
            write_status(
                status_path,
                state="failed",
                message="Не удалось запустить Python",
                run_id=run.id,
            )
            raise HTTPException(500, "Не удалось запустить процесс оценки") from exc
        return {"run_id": run.id, "pid": process.pid, "state": "starting"}

    @app.get("/api/evaluations/{run_id}/status")
    def evaluation_status(
        run_id: int,
        session: Annotated[Session, Depends(get_session)],
    ) -> dict[str, object]:
        run = session.get(EvaluationRun, run_id)
        if run is None:
            raise HTTPException(404, "Запуск оценки не найден")
        status_path = settings.data_dir / "evaluations" / f"{run_id:08d}" / "status.json"
        try:
            return json.loads(status_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"run_id": run.id, "state": run.status}
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(500, "Не удалось прочитать статус оценки") from exc

    @app.get("/api/metrics/online")
    def online_metrics(
        session: Annotated[Session, Depends(get_session)],
    ) -> dict[str, object]:
        return build_online_metrics(session)

    @app.get("/metrics", response_class=HTMLResponse)
    def metrics_page(
        request: Request,
        session: Annotated[Session, Depends(get_session)],
    ) -> HTMLResponse:
        runs = session.scalars(
            select(EvaluationRun).order_by(EvaluationRun.created_at.desc())
        ).all()
        return templates.TemplateResponse(
            request=request,
            name="metrics.html",
            context={
                "online": build_online_metrics(session),
                "detection": read_detection_stats(
                    settings.data_dir / "runtime" / "detection.jsonl"
                ),
                "runs": runs,
                "detector_ready": settings.detector_model_path.is_file(),
                "identity_ready": any(
                    path is not None and path.is_file()
                    for path in (
                        settings.identity_face_model_path,
                        settings.identity_side_model_path,
                    )
                ),
            },
        )

    @app.get("/metrics/runs/{run_id}", response_class=HTMLResponse)
    def evaluation_report_page(
        request: Request,
        run_id: int,
        session: Annotated[Session, Depends(get_session)],
    ) -> HTMLResponse:
        run = session.get(EvaluationRun, run_id)
        if run is None:
            raise HTTPException(404, "Запуск оценки не найден")
        try:
            report = json.loads(run.metrics_json) if run.metrics_json else None
        except json.JSONDecodeError:
            report = None
        version_ids = {
            version_id
            for version_id in (
                run.detector_model_version_id,
                run.face_model_version_id,
                run.side_model_version_id,
            )
            if version_id is not None
        }
        versions = (
            {
                version.id: version
                for version in session.scalars(
                    select(ModelVersion).where(ModelVersion.id.in_(version_ids))
                ).all()
            }
            if version_ids
            else {}
        )
        return templates.TemplateResponse(
            request=request,
            name="metrics-report.html",
            context={"run": run, "report": report, "versions": versions},
        )

    @app.get("/metrics/compare", response_class=HTMLResponse)
    def comparison_page(
        request: Request,
        session: Annotated[Session, Depends(get_session)],
        left: int | None = None,
        right: int | None = None,
    ) -> HTMLResponse:
        runs = session.scalars(
            select(EvaluationRun)
            .where(EvaluationRun.status == "completed")
            .order_by(EvaluationRun.created_at.desc())
        ).all()
        selected: list[EvaluationRun | None] = [
            session.get(EvaluationRun, run_id) if run_id is not None else None
            for run_id in (left, right)
        ]
        reports: list[dict[str, object] | None] = []
        for run in selected:
            try:
                value = json.loads(run.metrics_json) if run and run.metrics_json else None
            except json.JSONDecodeError:
                value = None
            reports.append(value if isinstance(value, dict) else None)
        metric_rows = _comparison_rows(reports[0], reports[1])
        return templates.TemplateResponse(
            request=request,
            name="metrics-compare.html",
            context={
                "runs": runs,
                "left": left,
                "right": right,
                "selected": selected,
                "metric_rows": metric_rows,
            },
        )

    @app.get("/media/{event_id}/{view}")
    def media(
        event_id: int,
        view: str,
        session: Annotated[Session, Depends(get_session)],
    ) -> FileResponse:
        event = session.get(Event, event_id)
        if view not in {"face", "side"}:
            raise HTTPException(404, "Неизвестная камера")
        media_path = (
            event.media_path if event and view == "face" else event.side_media_path if event else None
        )
        if not event or not media_path:
            raise HTTPException(404, "Видео не найдено")
        root = settings.data_dir.resolve()
        path = (root / media_path).resolve()
        if root not in path.parents or not path.is_file():
            raise HTTPException(404, "Видео не найдено")
        return FileResponse(
            path,
            media_type="video/mp4",
            filename=f"event-{event.id}-{view}.mp4",
        )

    return app


def _comparison_rows(
    left: dict[str, object] | None,
    right: dict[str, object] | None,
) -> list[dict[str, object]]:
    names = (
        ("false_accept_rate", "FAR", False),
        ("false_reject_rate", "FRR", False),
        ("misidentified_known", "Ошибочных имён", False),
        ("uncertain_rate", "Uncertain", False),
        ("accuracy", "Accuracy", True),
        ("mean_inference_ms", "Средний inference, мс", False),
    )
    values = [
        _combined_clean_metrics(report)
        for report in (left, right)
    ]
    rows: list[dict[str, object]] = []
    for key, label, higher_is_better in names:
        left_value = values[0].get(key) if values[0] else None
        right_value = values[1].get(key) if values[1] else None
        delta = (
            float(right_value) - float(left_value)
            if isinstance(left_value, (int, float))
            and isinstance(right_value, (int, float))
            else None
        )
        favorable = (
            delta is not None
            and delta != 0
            and ((delta > 0) if higher_is_better else (delta < 0))
        )
        rows.append(
            {
                "label": label,
                "left": left_value,
                "right": right_value,
                "delta": delta,
                "delta_class": (
                    "good" if favorable else "bad" if delta not in {None, 0} else ""
                ),
            }
        )
    return rows


def _combined_clean_metrics(
    report: dict[str, object] | None,
) -> dict[str, object] | None:
    if report is None:
        return None
    clean = report.get("clean_metrics")
    if not isinstance(clean, dict):
        return None
    combined = clean.get("combined")
    return combined if isinstance(combined, dict) else None


def run() -> None:
    settings = Settings()
    with SleepInhibitor():
        uvicorn.run(create_app(settings), host=settings.web_host, port=settings.web_port)


if __name__ == "__main__":
    run()
