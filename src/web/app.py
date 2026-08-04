import asyncio
import json
import shutil
import subprocess
import sys
from collections.abc import AsyncIterator, Callable, Generator, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from math import ceil
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.common.camera_discovery import (
    RTSP_PORT,
    DiscoveredDevice,
    candidate_paths,
    discover_devices,
    identify_stream_path,
    match_camera,
    port_is_open,
)
from src.common.camera_registry import (
    ROLE_TITLES,
    ROLES,
    CameraCircuit,
    CameraRecord,
    CameraRegistry,
    CameraRole,
    camera_from_url,
    camera_registry_path,
    ensure_registry,
    load_registry,
    save_registry,
    slugify_camera_id,
)
from src.common.capture_mode import (
    CaptureMode,
    CaptureState,
    capture_mode_path,
    load_capture_mode,
    save_capture_mode,
)
from src.common.circuits import (
    CIRCUIT_TITLES,
    CircuitConflict,
    CircuitName,
    GateRecording,
    NodeCircuits,
    circuits_path,
    ensure_allowed,
    load_circuits,
    save_circuits,
)
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
from src.common.gate_service import gate_status
from src.common.model_registry import register_model_version
from src.common.online_metrics import build_online_metrics
from src.common.process import spawn_detached
from src.common.sd_notify import Notifier
from src.common.settings import ConfigurationError, Settings, validate_runtime_paths
from src.recorder.detection_journal import frame_journal_path
from src.recorder.ffmpeg import RecorderError, mjpeg_stream, probe_stream, record_clip
from src.recorder.preview import local_mjpeg_stream, preview_is_fresh, preview_path
from src.recorder.status import read_recorder_status
from src.web.live_stream import resilient_mjpeg_stream, until
from training.workflow import create_macbook_export, write_status

LABELS = {"pending", "known_cat", "foreign_cat", "not_visible", "multiple_cats"}
TRAINING_TARGETS = {"excluded", "macbook", "local"}
# Проходов за ночь набирается столько, что страница переставала помещаться в
# экран и её приходилось долго листать. Десяток — та порция, после которой ещё
# видно счётчики и камеры без прокрутки обратно наверх.
EVENTS_PER_PAGE = 10
ACTIVE_EVALUATION_STATES = {"starting", "running"}
ACTIVE_TRAINING_STATES = {"starting", "preparing", "training"}
HEARTBEAT_STALE_AFTER = timedelta(minutes=10)
# Подбор пути потока держит запрос открытым, а администратор ждёт ответ
# страницы. Ограничения переводят «поиск завис» в «часть устройств не
# проверена», о чём в ответе сказано прямым текстом.
DISCOVERY_PROBE_LIMIT = 16
DISCOVERY_PROBE_SECONDS = 6.0
INTERRUPTED_MESSAGE = "Прерван без завершения: heartbeat не обновлялся более 10 минут"


def _asset_version(path: Path) -> str:
    """Короткая контрольная сумма файла стилей для адреса.

    Время изменения не годится: `git pull` на сервере переписывает файл, но
    может сохранить прежнюю метку, и браузер оставит старые стили.
    """
    try:
        return sha256(path.read_bytes()).hexdigest()[:12]
    except OSError:
        return "dev"


def _read_status(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _status_is_stale(
    payload: dict[str, object] | None,
    *,
    fallback: datetime,
    now: datetime | None = None,
) -> bool:
    updated_at = fallback
    raw_updated_at = payload.get("updated_at") if payload is not None else None
    if isinstance(raw_updated_at, str):
        try:
            updated_at = datetime.fromisoformat(raw_updated_at)
        except ValueError:
            pass
    current_time = now or datetime.now(UTC)
    return _as_utc(current_time) - _as_utc(updated_at) > HEARTBEAT_STALE_AFTER


def _expire_stale_evaluations(
    session: Session,
    data_dir: Path,
    *,
    now: datetime | None = None,
) -> int:
    current_time = now or datetime.now(UTC)
    runs = session.scalars(
        select(EvaluationRun).where(EvaluationRun.status.in_(ACTIVE_EVALUATION_STATES))
    ).all()
    expired = 0
    for run in runs:
        status_path = data_dir / "evaluations" / f"{run.id:08d}" / "status.json"
        if not _status_is_stale(
            _read_status(status_path),
            fallback=run.created_at,
            now=current_time,
        ):
            continue
        run.status = "failed"
        run.completed_at = current_time
        run.error_message = INTERRUPTED_MESSAGE
        write_status(
            status_path,
            state="failed",
            message=INTERRUPTED_MESSAGE,
            run_id=run.id,
        )
        expired += 1
    if expired:
        session.commit()
    return expired


def _expire_stale_training(
    status_path: Path,
    payload: dict[str, object],
    *,
    now: datetime | None = None,
) -> bool:
    if payload.get("state") not in ACTIVE_TRAINING_STATES:
        return False
    try:
        fallback = datetime.fromtimestamp(status_path.stat().st_mtime, tz=UTC)
    except OSError:
        fallback = now or datetime.now(UTC)
    if not _status_is_stale(payload, fallback=fallback, now=now):
        return False
    write_status(status_path, state="failed", message=INTERRUPTED_MESSAGE)
    return True


class CapturePayload(BaseModel):
    duration_seconds: int = Field(ge=3, le=30)


class CameraPayload(BaseModel):
    """Поля карточки камеры.

    Пароль отдельно: `None` означает «оставить прежний», пустая строка —
    «стереть». Иначе форма, которая не может показать сохранённый пароль,
    стирала бы его при каждом сохранении соседнего поля.
    """

    id: str | None = None
    title: str = ""
    host: str = ""
    port: int = Field(default=554, ge=1, le=65535)
    stream_path: str = ""
    username: str = ""
    password: str | None = None
    mac: str = ""
    circuit: CameraCircuit | None = None
    role: CameraRole | None = None
    rotate_180: bool = False
    enabled: bool = True
    note: str = ""
    rtsp_url: str = ""


class CaptureModePayload(BaseModel):
    state: CaptureState


class CircuitsPayload(BaseModel):
    """Что просит пульт узла.

    `active` — контур, которому отдаётся узел; остальные останавливаются. Поле
    отдельно от «включить/выключить», потому что до переезда работать может
    только один, и запрос вида «включи калитку» без указания, что делать с
    котами, не имеет однозначного смысла.
    """

    active: CircuitName | None = None
    gate_recording: GateRecording | None = None


class DiscoveryPayload(BaseModel):
    """Что администратор задаёт перед поиском.

    Логин и пароль — общие для всех найденных устройств: у новой камеры их
    взять неоткуда, а у уже заведённой берутся сохранённые, если поля пусты.
    """

    username: str = ""
    password: str = ""
    subnet: str | None = None
    probe_streams: bool = True


def _unique_camera_id(registry: CameraRegistry, candidate: str) -> str:
    if registry.get(candidate) is None:
        return candidate
    suffix = 2
    while registry.get(f"{candidate}-{suffix}") is not None:
        suffix += 1
    return f"{candidate}-{suffix}"


def _role_stream_url(registry: CameraRegistry, role: str) -> str:
    """RTSP-адрес камеры роли. Пустая строка — роли нечем снимать.

    Выключенная камера считается отсутствующей: recorder её тоже не берёт, и
    показывать с неё живое изображение значило бы обещать запись, которой не
    будет. Кнопка проверки потока работает и для выключенной.
    """
    camera = registry.by_role(role)
    return camera.rtsp_url() if camera is not None and camera.enabled else ""


def _role_base_rotation(registry: CameraRegistry, role: str) -> int:
    """Поворот, которым показывают ЗАПИСИ этой роли.

    Клипы пишутся как есть с камеры (`-c:v copy`, без ориентационных фильтров),
    поэтому перевёрнутую камеру отыгрывает показ. Живое превью здесь ни при чём:
    его разворачивает ffmpeg на сервере, и добавить поворот ещё и в CSS значило
    бы развернуть картинку дважды.
    """
    camera = registry.by_role(role)
    return 180 if camera is not None and camera.rotate_180 else 0


def _camera_view(camera: CameraRecord) -> dict[str, object]:
    """Карточка камеры для API. Пароль наружу не выходит никогда."""
    return {
        "id": camera.id,
        "title": camera.title,
        "display_title": camera.display_title,
        "host": camera.host,
        "port": camera.port,
        "stream_path": camera.stream_path,
        "username": camera.username,
        "has_password": bool(camera.password),
        "mac": camera.mac,
        "circuit": camera.circuit,
        "role": camera.role,
        "rotate_180": camera.rotate_180,
        "enabled": camera.enabled,
        "note": camera.note,
        "configured": bool(camera.rtsp_url()),
    }


def _apply_camera_payload(
    payload: CameraPayload,
    camera_id: str,
    existing: CameraRecord | None,
) -> CameraRecord:
    """Наложить присланные поля на сохранённую запись.

    Применяется только то, что действительно пришло в запросе: запрос, где
    поменялся один адрес, не должен заодно снять с камеры роль и поворот.
    Целый RTSP-адрес, если он задан, перекрывает разобранные поля — вставить
    строку из вебморды камеры проще, чем разложить её руками.
    """
    base = existing if existing is not None else CameraRecord(id=camera_id)
    provided = payload.model_dump(exclude_unset=True)
    provided.pop("id", None)
    rtsp_url = str(provided.pop("rtsp_url", "") or "")
    password = provided.pop("password", None)
    values: dict[str, object] = {**base.model_dump(), **provided, "id": camera_id}
    if password is not None:
        values["password"] = password
    if rtsp_url.strip():
        parsed = camera_from_url(camera_id, rtsp_url)
        values.update(
            {
                "host": parsed.host,
                "port": parsed.port,
                "stream_path": parsed.stream_path,
            }
        )
        if parsed.username:
            values["username"] = parsed.username
        if parsed.password:
            values["password"] = parsed.password
    try:
        return CameraRecord.model_validate(values)
    except ValidationError as exc:
        # Берутся только сообщения: полный текст ошибки pydantic цитирует ещё и
        # значения полей, а среди них пароль камеры.
        reason = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        raise HTTPException(422, f"Некорректные поля камеры: {reason}") from exc


class LabelPayload(BaseModel):
    label: str
    cat_name: str | None = Field(default=None, max_length=100)
    training_target: str = "excluded"


class BulkLabelPayload(LabelPayload):
    event_ids: list[int] = Field(min_length=1, max_length=100)


class EvaluationStartPayload(BaseModel):
    run_kind: Literal["identity", "detector"] = "identity"
    created_from: datetime | None = None
    created_to: datetime | None = None
    labels: list[str] = Field(
        default_factory=lambda: [
            "known_cat",
            "foreign_cat",
            "pending",
            "multiple_cats",
            "not_visible",
        ]
    )
    training_targets: list[str] = Field(default_factory=list)
    detector_model: str | None = None
    face_model: str | None = None
    side_model: str | None = None


async def _ping_watchdog(notifier: Notifier, interval_seconds: float) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        notifier.watchdog()


@asynccontextmanager
async def _notify_systemd(app: FastAPI) -> AsyncIterator[None]:
    """Announce readiness once the event loop is actually serving.

    Sending READY=1 before `uvicorn.run` would claim the port is open while it
    is not, so a dependent unit could start too early. Outside systemd the
    notifier is inert, no task is created, and `TestClient` behaves as before.
    """
    notifier = Notifier.from_environment()
    notifier.ready(f"Админка слушает {app.state.settings.web_host}")
    interval = notifier.watchdog_interval_seconds
    task = (
        asyncio.create_task(_ping_watchdog(notifier, interval))
        if interval is not None
        else None
    )
    try:
        yield
    finally:
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        notifier.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    factory = create_session_factory(settings.resolved_database_path)
    templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
    # Общая таблица стилей вместо инлайнового блока в каждом шаблоне: правка
    # темы или отступа не должна означать четыре одинаковые правки. Версия в
    # адресе — контрольная сумма файла, поэтому после обновления служб браузер
    # не показывает вчерашние стили и при этом кэширует их между заходами.
    static_dir = Path(__file__).parent / "static"
    asset_version = _asset_version(static_dir / "app.css")
    registry_path = camera_registry_path(settings.data_dir)
    ensure_registry(registry_path, settings)

    app = FastAPI(title="CatDoor AI", version="0.1.0", lifespan=_notify_systemd)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    app.state.settings = settings
    app.state.session_factory = factory
    templates.env.globals["asset_version"] = asset_version
    templates.env.globals["circuit_titles"] = CIRCUIT_TITLES

    def current_detection_config() -> DetectionConfig:
        return load_detection_config(
            detection_config_path(settings.data_dir),
            DetectionConfig.from_settings(settings),
        )

    def current_circuits() -> NodeCircuits:
        return load_circuits(circuits_path(settings.data_dir))

    def circuits_view(circuits: NodeCircuits) -> dict[str, object]:
        status = gate_status(settings.gate_service_unit)
        return {
            "gate_state": status.state,
            "gate_detail": status.detail,
            "gate_deployed": status.deployed,
            "gate_ui_url": settings.gate_ui_url,
            "cats": circuits.cats,
            "gate": circuits.gate,
            "exclusive": circuits.exclusive,
            "gate_recording": circuits.gate_recording,
            "active": list(circuits.active_names),
        }

    def current_capture_mode() -> CaptureMode:
        return load_capture_mode(
            capture_mode_path(settings.data_dir),
            CaptureMode.from_settings(settings),
        )

    def current_registry() -> CameraRegistry:
        """Инвентарь читается на каждый запрос.

        Файл крошечный, а редактировать его может и recorder, и человек через
        SSH: держать копию в памяти означало бы показывать устаревший адрес
        ровно тогда, когда камеру только что переставили.
        """
        return load_registry(registry_path) or ensure_registry(registry_path, settings)

    def store_registry(registry: CameraRegistry) -> None:
        try:
            save_registry(registry_path, registry)
        except OSError as exc:
            raise HTTPException(500, "Не удалось сохранить список камер") from exc

    def required_camera(registry: CameraRegistry, camera_id: str) -> CameraRecord:
        camera = registry.get(camera_id)
        if camera is None:
            raise HTTPException(404, "Камера не найдена")
        return camera

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
        page: int = 1,
    ) -> HTMLResponse:
        summary_row = session.execute(
            select(
                func.count(Event.id),
                func.count(Event.id).filter(Event.trigger_source == "automatic"),
                func.count(Event.id).filter(Event.label == "pending"),
                func.count(Event.id).filter(Event.label == "known_cat"),
                func.count(Event.id).filter(Event.label == "foreign_cat"),
                func.count(Event.id).filter(Event.status == "failed"),
                func.count(Event.id).filter(Event.training_target == "macbook"),
                func.count(Event.id).filter(Event.training_target == "local"),
            )
        ).one()
        pages = max(1, ceil(summary_row[0] / EVENTS_PER_PAGE))
        # Номер страницы прижимается к диапазону, а не отвергается: после
        # удаления последнего прохода на странице сохранённая ссылка иначе вела
        # бы в пустоту.
        page = min(max(page, 1), pages)
        events = session.scalars(
            select(Event)
            # Тай-брейк по id обязателен: автоматические проходы пишутся пачками
            # и могут делить время создания, а при неустойчивом порядке одна и
            # та же запись прыгала бы между страницами.
            .order_by(Event.created_at.desc(), Event.id.desc())
            .limit(EVENTS_PER_PAGE)
            .offset((page - 1) * EVENTS_PER_PAGE)
        ).all()
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
        # Считается по всей базе, а не по показанной странице: это размер
        # набора для обучения, и он не зависит от того, что сейчас на экране.
        # Прежний подсчёт по срезу врал уже на сотне проходов.
        target_counts = {"macbook": summary_row[6], "local": summary_row[7]}
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
        registry = current_registry()
        capture_mode = current_capture_mode()
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "events": events,
                "event_predictions": event_predictions,
                # Не `page`: так в шаблонах называется вкладка навигации,
                # и страница проходов затёрла бы подсветку «Проходы».
                "events_page": page,
                "events_pages": pages,
                "events_per_page": EVENTS_PER_PAGE,
                "events_shown_from": (page - 1) * EVENTS_PER_PAGE + 1 if events else 0,
                "events_shown_to": (page - 1) * EVENTS_PER_PAGE + len(events),
                "face_configured": bool(_role_stream_url(registry, "face")),
                "side_configured": bool(_role_stream_url(registry, "side")),
                "recording_rotations": {
                    role: _role_base_rotation(registry, role) for role in ("face", "side")
                },
                "default_duration": settings.capture_duration_seconds,
                "target_counts": target_counts,
                "event_summary": event_summary,
                "known_cat_names": known_cat_names,
                "local_status": local_status,
                "local_training_epochs": settings.local_training_epochs,
                "recorder_status": recorder_status,
                "automatic_capture_enabled": recorder_enabled,
                "automatic_capture_dry_run": recorder_dry_run,
                "capture_state": capture_mode.state,
                "circuits": circuits_view(current_circuits()),
                "stream_version": uuid4().hex,
                "detection_config": current_detection_config(),
            },
        )

    @app.get("/cameras", response_class=HTMLResponse)
    def cameras_page(request: Request) -> HTMLResponse:
        registry = current_registry()
        return templates.TemplateResponse(
            request,
            "cameras.html",
            {
                "cameras": [_camera_view(camera) for camera in registry.cameras],
                "role_titles": ROLE_TITLES,
                "roles": list(ROLES),
                "circuit_groups": [
                    {
                        "circuit": circuit,
                        "title": title,
                        "cameras": [
                            _camera_view(camera)
                            for camera in registry.cameras
                            if camera.circuit == circuit
                        ],
                    }
                    for circuit, title in (
                        ("cats", CIRCUIT_TITLES["cats"]),
                        ("gate", CIRCUIT_TITLES["gate"]),
                        (None, "Без контура"),
                    )
                ],
                "detection_config": current_detection_config(),
                "recorder_status": read_recorder_status(
                    settings.data_dir / "runtime" / "recorder-status.json"
                ),
                "discovery_subnet": settings.camera_discovery_subnet,
                "stream_version": uuid4().hex,
            },
        )

    @app.get("/api/cameras")
    def list_cameras() -> dict[str, object]:
        registry = current_registry()
        return {"cameras": [_camera_view(camera) for camera in registry.cameras]}

    @app.post("/api/cameras", status_code=201)
    def create_camera(payload: CameraPayload) -> dict[str, object]:
        registry = current_registry()
        if payload.id:
            camera_id = slugify_camera_id(payload.id)
            if registry.get(camera_id) is not None:
                raise HTTPException(409, f"Камера с идентификатором {camera_id} уже есть")
        else:
            # Идентификатор собран за администратора, поэтому и совпадение
            # разрешается за него: название на кириллице сворачивается в один и
            # тот же слаг, а отказ на этом месте выглядел бы отказом добавить
            # камеру вовсе.
            camera_id = _unique_camera_id(
                registry,
                slugify_camera_id(payload.title or payload.host or "camera"),
            )
        camera = _apply_camera_payload(payload, camera_id, None)
        store_registry(registry.with_camera(camera))
        return _camera_view(camera)

    @app.put("/api/cameras/{camera_id}")
    def update_camera(camera_id: str, payload: CameraPayload) -> dict[str, object]:
        registry = current_registry()
        existing = required_camera(registry, camera_id)
        camera = _apply_camera_payload(payload, camera_id, existing)
        store_registry(registry.with_camera(camera))
        return _camera_view(camera)

    @app.delete("/api/cameras/{camera_id}", status_code=204)
    def delete_camera(camera_id: str) -> None:
        registry = current_registry()
        required_camera(registry, camera_id)
        store_registry(registry.without_camera(camera_id))

    @app.post("/api/cameras/swap-roles")
    def swap_camera_roles() -> dict[str, object]:
        registry = current_registry().with_swapped_roles()
        store_registry(registry)
        return {"cameras": [_camera_view(camera) for camera in registry.cameras]}

    @app.post("/api/cameras/discover")
    def discover_cameras(payload: DiscoveryPayload) -> dict[str, object]:
        registry = current_registry()
        subnet = (
            payload.subnet
            if payload.subnet is not None
            else settings.camera_discovery_subnet
        )
        devices = discover_devices(
            leases_path=settings.camera_dhcp_leases_path,
            arp_path=settings.camera_arp_table_path,
            subnet=subnet,
            onvif_timeout=settings.camera_discovery_timeout_seconds,
        )
        probed = 0

        def examine(device: DiscoveredDevice) -> dict[str, object]:
            nonlocal probed
            camera = match_camera(registry, device)
            rtsp_open = port_is_open(device.host, RTSP_PORT)
            username = payload.username or (camera.username if camera is not None else "")
            found: dict[str, object] = {
                "host": device.host,
                "mac": device.mac,
                "name": device.name,
                "sources": list(device.sources),
                "other_hosts": list(device.other_hosts),
                "matched_camera_id": camera.id if camera is not None else None,
                "matched_camera_title": camera.display_title if camera is not None else None,
                "moved": camera is not None and camera.host != device.host,
                "rtsp_open": rtsp_open,
                "username": username,
                "stream_path": "",
                "reason": "",
                "needs_credentials": False,
            }
            if not rtsp_open or not payload.probe_streams:
                return found
            probed += 1
            if probed > DISCOVERY_PROBE_LIMIT:
                found["reason"] = "проверка потока пропущена: слишком много устройств"
                return found
            guess = identify_stream_path(
                device.host,
                probe=lambda url: probe_stream(
                    settings.ffmpeg_path,
                    url,
                    timeout_seconds=DISCOVERY_PROBE_SECONDS,
                ),
                username=username,
                password=payload.password
                or (camera.password if camera is not None else ""),
                paths=candidate_paths(camera.stream_path if camera is not None else ""),
            )
            found["stream_path"] = guess.stream_path
            found["reason"] = guess.reason
            found["needs_credentials"] = guess.needs_credentials
            return found

        if not devices:
            return {"subnet": subnet, "devices": []}
        with ThreadPoolExecutor(max_workers=min(8, len(devices))) as executor:
            results = list(executor.map(examine, devices))
        return {"subnet": subnet, "devices": results}

    @app.post("/api/cameras/{camera_id}/test")
    def test_camera(camera_id: str) -> dict[str, object]:
        camera = required_camera(current_registry(), camera_id)
        url = camera.rtsp_url()
        if not url:
            raise HTTPException(422, "У камеры не задан адрес")
        reason = probe_stream(settings.ffmpeg_path, url)
        return {"ok": reason is None, "reason": reason}

    def camera_stream(find: Callable[[CameraRegistry], CameraRecord | None]) -> StreamingResponse:
        camera = find(current_registry())
        if camera is None:
            raise HTTPException(404, "Неизвестная камера")
        if not camera.enabled or not camera.rtsp_url():
            raise HTTPException(503, "Камера не настроена")

        def open_source() -> Iterator[bytes] | None:
            # Камера перечитывается на каждую попытку: за время обрыва она могла
            # вернуться по другому адресу, и попытка второй раз открыть прежний
            # означала бы, что картинка не оживёт уже никогда.
            camera = find(current_registry())
            if camera is None or not camera.enabled or not camera.rtsp_url():
                return None
            # Frames annotated by the detector are strictly more useful, so they
            # win whenever the recorder is publishing them. Falling back to the
            # camera keeps the picture available for aiming and placement, which
            # is needed long before a detector model exists on the machine.
            # The recorder publishes previews per role, so a camera without a
            # role simply has none and is streamed directly.
            path = preview_path(settings.data_dir, camera.role or camera.id)
            if preview_is_fresh(path):
                return local_mjpeg_stream(
                    path,
                    settings.preview_fps,
                    idle_timeout_seconds=settings.preview_idle_timeout_seconds,
                )
            # Поток с камеры не кончается сам, поэтому его обрывают снаружи,
            # как только recorder снова публикует кадры: иначе выбор, сделанный
            # в момент протухшего превью, остался бы навсегда — картинка живая,
            # а рамок детекции больше нет до перезагрузки страницы.
            return until(
                mjpeg_stream(
                    settings.ffmpeg_path,
                    camera.rtsp_url(),
                    settings.preview_fps,
                    rotate_180=camera.rotate_180,
                    idle_timeout_seconds=settings.preview_idle_timeout_seconds,
                ),
                lambda: preview_is_fresh(path),
            )

        return StreamingResponse(
            resilient_mjpeg_stream(
                open_source,
                silence_limit_seconds=settings.preview_silence_limit_seconds,
            ),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/stream/camera/{camera_id}.mjpeg")
    def camera_id_stream(camera_id: str) -> StreamingResponse:
        """Поток строго по идентификатору камеры.

        Страница камер показывает железки, а не роли, и обращаться к ним через
        `resolve` она не может: тот сначала ищет по роли, а идентификатор камеры
        вполне может совпасть с именем роли. Стоило переставить роли местами —
        и карточка камеры с id `face` показывала ту, что сейчас на роли `face`,
        то есть обе карточки сходились на одной камере.
        """
        return camera_stream(lambda registry: registry.get(camera_id))

    @app.get("/stream/{view}.mjpeg")
    def stream(view: str) -> StreamingResponse:
        # Здесь роль имеет приоритет намеренно: страница проходов показывает
        # «камеру идентификации», кто бы ею сейчас ни был.
        return camera_stream(lambda registry: registry.resolve(view))

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

    @app.get("/api/circuits")
    def get_circuits() -> dict[str, object]:
        return circuits_view(current_circuits())

    @app.put("/api/circuits")
    def update_circuits(payload: CircuitsPayload) -> dict[str, object]:
        circuits = current_circuits()
        if payload.active is not None:
            circuits = circuits.with_only(payload.active)
        if payload.gate_recording is not None:
            circuits = circuits.model_copy(update={"gate_recording": payload.gate_recording})
        try:
            ensure_allowed(circuits)
        except CircuitConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        try:
            save_circuits(circuits_path(settings.data_dir), circuits)
        except OSError as exc:
            raise HTTPException(500, "Не удалось сохранить состояние узла") from exc
        # Служба к состоянию не приводится здесь: у веб-процесса нет и не
        # должно быть привилегий. Файл читает примиритель `catdoor-gate-sync`,
        # запускаемый от root по изменению этого файла, — вкладка увидит
        # результат при ближайшем обновлении состояния.
        return circuits_view(circuits)

    @app.get("/gate", response_class=HTMLResponse)
    def gate_page(request: Request) -> HTMLResponse:
        circuits = current_circuits()
        return templates.TemplateResponse(
            request,
            "gate.html",
            {
                "circuits": circuits_view(circuits),
                "circuit_titles": CIRCUIT_TITLES,
            },
        )

    @app.get("/api/capture-mode")
    def get_capture_mode() -> dict[str, object]:
        mode = current_capture_mode()
        return {"state": mode.state, "enabled": mode.enabled, "record_events": mode.record_events}

    @app.put("/api/capture-mode")
    def update_capture_mode(payload: CaptureModePayload) -> dict[str, object]:
        mode = CaptureMode.from_state(payload.state)
        try:
            save_capture_mode(capture_mode_path(settings.data_dir), mode)
        except OSError as exc:
            raise HTTPException(500, "Не удалось сохранить режим сбора") from exc
        return {"state": mode.state, "enabled": mode.enabled, "record_events": mode.record_events}

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
        registry = current_registry()
        face_url = _role_stream_url(registry, "face")
        side_url = _role_stream_url(registry, "side")
        if not face_url:
            raise HTTPException(503, "Камера face не настроена")
        try:
            free_gb = shutil.disk_usage(settings.data_dir).free / 1024**3
        except OSError as exc:
            raise HTTPException(507, "Не удалось проверить место на диске") from exc
        if free_gb < settings.disk_min_free_gb:
            raise HTTPException(
                507,
                f"Осталось {free_gb:.1f} ГБ — меньше порога "
                f"{settings.disk_min_free_gb:g} ГБ. Освободите место: новая "
                "запись добила бы диск, после чего замолчала бы и отчётность.",
            )
        event = Event(duration_seconds=payload.duration_seconds)
        session.add(event)
        session.commit()
        event_dir = settings.data_dir / "events" / f"{event.id:08d}"
        recordings = {"face": (face_url, event_dir / "face.mp4")}
        if side_url:
            recordings["side"] = (side_url, event_dir / "side.mp4")
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
            latest = _read_status(existing[-1]) or {}
            if (
                latest.get("state") in ACTIVE_TRAINING_STATES
                and not _expire_stale_training(existing[-1], latest)
            ):
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
        try:
            with log_path.open("ab") as log:
                process = spawn_detached(
                    command,
                    cwd=Path.cwd(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
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
        _expire_stale_evaluations(session, settings.data_dir)
        running = session.scalar(
            select(EvaluationRun)
            .where(EvaluationRun.status.in_(ACTIVE_EVALUATION_STATES))
            .limit(1)
        )
        if running is not None:
            raise HTTPException(409, "Офлайн-оценка уже выполняется")

        detector_path = resolve_evaluation_model(
            settings.detector_model_path,
            payload.detector_model,
        )
        face_path = (
            resolve_evaluation_model(
                settings.identity_face_model_path,
                payload.face_model,
            )
            if payload.run_kind == "identity"
            else None
        )
        side_path = (
            resolve_evaluation_model(
                settings.identity_side_model_path,
                payload.side_model,
            )
            if payload.run_kind == "identity"
            else None
        )
        if detector_path is None or not detector_path.is_file():
            raise HTTPException(422, "Локальная detector-модель не найдена")
        identity_paths = {
            role: path
            for role, path in (("face", face_path), ("side", side_path))
            if path is not None and path.is_file()
        }
        if payload.run_kind == "identity" and not identity_paths:
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
            run_kind=payload.run_kind,
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
            "--run-kind",
            payload.run_kind,
        ]
        for role in ("face", "side"):
            if role in identity_paths:
                command.extend(
                    [f"--{role}-model", str(identity_paths[role].resolve())]
                )
        try:
            with log_path.open("ab") as log:
                process = spawn_detached(
                    command,
                    cwd=Path.cwd(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
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
        # Прогон, убитый отключением питания, иначе висел бы «выполняется»
        # до следующей попытки запустить новую оценку.
        _expire_stale_evaluations(session, settings.data_dir)
        runs = session.scalars(
            select(EvaluationRun).order_by(EvaluationRun.created_at.desc())
        ).all()
        return templates.TemplateResponse(
            request=request,
            name="metrics.html",
            context={
                "online": build_online_metrics(session),
                # Статистика считается по покадровым наблюдениям, а они с
                # недавних пор живут отдельным файлом.
                "detection": read_detection_stats(
                    frame_journal_path(settings.data_dir / "runtime" / "detection.jsonl")
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
        report_template = (
            "metrics-detector-report.html"
            if isinstance(report, dict) and report.get("run_kind") == "detector"
            else "metrics-report.html"
        )
        return templates.TemplateResponse(
            request=request,
            name=report_template,
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
        comparison_kind = (
            selected[0].run_kind
            if selected[0] is not None
            and selected[1] is not None
            and selected[0].run_kind == selected[1].run_kind
            else None
        )
        return templates.TemplateResponse(
            request=request,
            name="metrics-compare.html",
            context={
                "runs": runs,
                "left": left,
                "right": right,
                "selected": selected,
                "metric_rows": metric_rows,
                "comparison_kind": comparison_kind,
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
    left_kind = left.get("run_kind", "identity") if left else None
    right_kind = right.get("run_kind", "identity") if right else None
    if left_kind != right_kind:
        return []
    if left_kind == "detector":
        return _detector_comparison_rows(left, right)
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


def _detector_comparison_rows(
    left: dict[str, object] | None,
    right: dict[str, object] | None,
) -> list[dict[str, object]]:
    names = (
        ("miss_rate", "Miss rate", False),
        ("false_trigger_rate", "False trigger rate", False),
        ("median_inference_ms", "Медиана inference, мс", False),
        ("p95_inference_ms", "p95 inference, мс", False),
    )
    rows: list[dict[str, object]] = []
    for view in ("face", "side"):
        values = [
            _detector_clean_metrics(report, view)
            for report in (left, right)
        ]
        for key, label, higher_is_better in names:
            left_value = values[0].get(key) if values[0] else None
            right_value = values[1].get(key) if values[1] else None
            delta = (
                float(right_value) - float(left_value)
                if isinstance(left_value, (int, float))
                and not isinstance(left_value, bool)
                and isinstance(right_value, (int, float))
                and not isinstance(right_value, bool)
                else None
            )
            favorable = (
                delta is not None
                and delta != 0
                and ((delta > 0) if higher_is_better else (delta < 0))
            )
            rows.append(
                {
                    "label": f"{view}: {label}",
                    "left": left_value,
                    "right": right_value,
                    "delta": delta,
                    "delta_class": (
                        "good"
                        if favorable
                        else "bad"
                        if delta not in {None, 0}
                        else ""
                    ),
                }
            )
    return rows


def _detector_clean_metrics(
    report: dict[str, object] | None,
    view: str,
) -> dict[str, object] | None:
    if report is None:
        return None
    sections = report.get("sections")
    if not isinstance(sections, dict):
        return None
    clean = sections.get("clean")
    if not isinstance(clean, dict):
        return None
    values = clean.get(view)
    return values if isinstance(values, dict) else None


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
    try:
        validate_runtime_paths(settings)
    except ConfigurationError as exc:
        raise SystemExit(f"Конфигурация непригодна для запуска: {exc}") from exc
    uvicorn.run(create_app(settings), host=settings.web_host, port=settings.web_port)


if __name__ == "__main__":
    run()
