import asyncio
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import AsyncIterator, Callable, Generator, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from math import ceil
from pathlib import Path

# Под именем `Event` в этом модуле уже живёт строка события базы, а событие
# потока рядом с ней читалось бы как та же сущность.
from threading import Event as ThreadEvent
from time import monotonic, sleep, time
from typing import Annotated, Literal
from uuid import uuid4
from zoneinfo import ZoneInfo

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import ColumnElement, func, select
from sqlalchemy.orm import Session

from src.common.archive_depth import ArchiveDepth, read_archive_depth
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
from src.common.cat_door_config import (
    CatDoorConfig,
    cat_door_config_path,
    load_cat_door_config,
    save_cat_door_config,
)
from src.common.circuits import (
    CIRCUIT_TITLES,
    RECORDING_TITLES,
    CircuitConflict,
    CircuitName,
    GateRecording,
    NodeCircuits,
    circuits_path,
    ensure_allowed,
    load_circuits,
    save_circuits,
)
from src.common.cloud_backup_config import (
    CloudBackupConfig,
    cats_cloud_backup_state_path,
    cloud_backup_config_path,
    cloud_backup_state_path,
    load_cloud_backup_config,
    save_cloud_backup_config,
)
from src.common.database import (
    EvaluationRun,
    Event,
    EventPrediction,
    GatePass,
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
from src.common.frigate_config import RECORDING_RETENTION, frigate_camera_name
from src.common.gate_access_config import (
    MOVEMENT_PRESET_NAME,
    GateAccessConfig,
    GateSpec,
    apply_movement_preset,
    gate_access_config_path,
    load_gate_access_config,
    mask_mac,
    rollback_movement_preset,
    save_gate_access_config,
)
from src.common.gate_open_request import (
    TARGET_CAT_DOOR,
    TARGET_GATE,
    OpenRequest,
    OpenResult,
    boot_id,
    heartbeat_age,
    heartbeat_path,
    load_open_result,
    open_request_path,
    open_result_path,
    save_open_request,
)
from src.common.gate_service import (
    GateStatus,
    frigate_face_names,
    frigate_stats,
    gate_status,
)
from src.common.health import build_health, gpu_occupancy_percent, overall_status
from src.common.model_registry import identity_labels, register_model_version
from src.common.node_metrics import read_node_metrics
from src.common.online_metrics import build_online_metrics
from src.common.process import spawn_detached
from src.common.sd_notify import Notifier
from src.common.settings import ConfigurationError, Settings, validate_runtime_paths
from src.common.system_stats import (
    cpu_percent,
    read_cpu_sample,
    read_cpu_temperature,
    read_memory_mb,
)
from src.common.telegram import TelegramClient, TelegramError, hint_for
from src.gate.access import gate_health_path
from src.gate.notify import state_path as gate_notify_state_path
from src.monitor.healthwatch import state_path as health_watch_state_path
from src.recorder.detection_journal import frame_journal_path
from src.recorder.ffmpeg import (
    RecorderError,
    extract_frame,
    mjpeg_stream,
    probe_stream,
    record_clip,
)
from src.recorder.preview import local_mjpeg_stream, preview_is_fresh, preview_path
from src.recorder.status import read_recorder_status, recorder_status_path
from src.web.door_monitor import DoorMonitor
from src.web.live_stream import resilient_mjpeg_stream, until
from src.web.logs import (
    DEFAULT_LINES,
    MAX_LINES,
    SOURCE_BY_KEY,
    SOURCES,
    clamp_lines,
    read_log,
)
from training.workflow import write_status

LABELS = {"pending", "known_cat", "foreign_cat", "not_visible", "multiple_cats"}
TRAINING_TARGETS = {"excluded", "macbook", "local"}
# Проходов за ночь набирается столько, что страница переставала помещаться в
# экран и её приходилось долго листать. Десяток — та порция, после которой ещё
# видно счётчики и камеры без прокрутки обратно наверх.
EVENTS_PER_PAGE = 10
# Счётчики в шапке — они же фильтры списка проходов: ключ приходит в `?show=`
# и превращается в условие. Ключи совпадают с полями `event_summary`.
# Раздел «Опознанные»: фильтры по размеченным проходам. Ключ «named» — все
# свои коты с именем — умолчание раздела; конкретное имя приходит отдельным
# параметром cat и проверяется по базе, а не по этому словарю.
LABELED_FILTERS = {
    "named": (Event.label == "known_cat") & Event.cat_name.is_not(None),
    "foreign": Event.label == "foreign_cat",
    "nocat": Event.label == "not_visible",
    "multi": Event.label == "multiple_cats",
    "all": Event.label != "pending",
}
# Журнал калитки: те же чипы-фильтры, что и у проходов дверцы. «Неузнанные»
# отдельным ключом не для симметрии — это самая интересная часть журнала: по
# ней видно, кого система не узнала и почему у калитки набирали код руками.
# Люди и машины разведены намеренно: в журнале они отвечают на разные вопросы —
# «кого пустили» и «кто приезжал», — и общий счётчик не отвечает ни на один.
GATE_PASS_FILTERS = {
    "all": GatePass.label.is_not("car"),
    "named": GatePass.person_name.is_not(None),
    "unknown": (GatePass.person_name.is_(None)) & (GatePass.label.is_not("car")),
    "cars": GatePass.label == "car",
}
# Решения журнала калитки по-русски. Коды заведены все сразу, включая те, что
# появятся с функцией решения: словарь в шаблоне не должен молча показать
# английский код в тот день, когда служба научится его писать.
GATE_PASS_DECISIONS = {
    "denied_mode_off": "автооткрытие выключено",
    "dry_run": "открыл бы",
    "opened": "открыл",
    "denied_not_allowed": "не в списке",
    "denied_low_score": "узнан неуверенно",
    "denied_window": "вне разрешённых часов",
    "denied_cooldown": "пауза после прошлого",
    "denied_second_factor": "нет второго фактора",
    "denied_leaving": "уходит от калитки",
    "denied_not_approaching": "не идёт к калитке",
    "denied_movement_unknown": "куда идёт — пока не видно",
    "denied_stale": "решение опоздало, человек ушёл",
    "denied_leaf_open": "створка открыта или ходила — без импульса",
    "logged": "машина, только запись",
    "error": "отказ",
}
# Состояние службы Frigate показывается в шапке каждой страницы, а узнаётся
# запуском systemctl. Короткая память избавляет от процесса на отрисовку, не
# успевая соврать заметно.
GATE_STATUS_CACHE_SECONDS = 2.0
CAPTURE_STATE_TITLES = {"off": "выключен", "observe": "наблюдение", "record": "запись"}
GATE_STATE_TITLES = {
    "running": "работает",
    "starting": "запускается",
    "stopped": "остановлен",
    "failed": "отказ",
    "absent": "не развёрнут",
    "unknown": "состояние неизвестно",
    "degraded": "работает без камер",
}
ACTIVE_EVALUATION_STATES = {"starting", "running"}
ACTIVE_TRAINING_STATES = {"starting", "preparing", "training"}
HEARTBEAT_STALE_AFTER = timedelta(minutes=10)

# Ширина кадра-постера. Карточка на странице проходов рисуется ~300 точек,
# 480 оставляет запас под плотные экраны и весит ~30 КБ.
POSTER_WIDTH = 480


def backup_status_path(data_dir: Path) -> Path:
    """Отчёт ночной копии. Пишет его `scripts/backup_to_archive.sh` от root.

    Лежит на внутреннем диске, а не в архиве: чаще всего копия не проходит
    из-за переполненного архивного диска, и записать отчёт туда было бы некуда.
    """
    return data_dir / "runtime" / "backup-status.json"


def poster_path(data_dir: Path, event_id: int, view: str) -> Path:
    return data_dir / "runtime" / "posters" / f"{event_id:08d}-{view}.jpg"
# Подбор пути потока держит запрос открытым, а администратор ждёт ответ
# страницы. Ограничения переводят «поиск завис» в «часть устройств не
# проверена», о чём в ответе сказано прямым текстом.
DISCOVERY_PROBE_LIMIT = 16
DISCOVERY_PROBE_SECONDS = 6.0
INTERRUPTED_MESSAGE = "Прерван без завершения: heartbeat не обновлялся более 10 минут"
GONE_MESSAGE = "Прогон оборван: процесс не запущен, а состояние осталось рабочим"


def _asset_version(path: Path):
    """Короткая контрольная сумма файла стилей для адреса — на каждый запрос.

    Контрольная сумма, а не время изменения: `git pull` может сохранить
    прежнюю метку файла. И заново при каждом обращении, а не раз при старте:
    служба переживает деплой без перезапуска, и посчитанная на старте версия
    держала браузеры на старых стилях до следующей перезагрузки. Пересчёт
    закеширован по mtime — хешировать файл на каждый запрос незачем.
    """
    cache: dict[str, object] = {"mtime": None, "value": "dev"}

    def version() -> str:
        try:
            mtime = path.stat().st_mtime_ns
            if cache["mtime"] != mtime:
                cache["value"] = sha256(path.read_bytes()).hexdigest()[:12]
                cache["mtime"] = mtime
        except OSError:
            return "dev"
        return str(cache["value"])

    return version


def _read_status(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _notifications_view(settings: Settings) -> dict[str, object]:
    """Что показать про уведомления, ни разу не назвав токен.

    Страница админки открыта всякому в домашней сети, поэтому наружу уходит
    только «настроен или нет» и правила тишины.
    """
    roles = {"face": "дверца", "side": "подход"}
    return {
        "configured": bool(settings.telegram_bot_token and settings.telegram_chat_id),
        "roles": ", ".join(roles.get(role, role) for role in settings.notify_roles) or "нет",
        "cooldown_minutes": round(settings.notify_cooldown_seconds / 60),
    }


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def local_day_bounds(day: str, timezone: ZoneInfo) -> tuple[datetime, datetime] | None:
    """Границы местных суток в UTC. `None` — день не разобрался.

    Фильтр «за день» человек задаёт по своим часам, а база живёт в UTC: в
    Москве полночь наступает на три часа раньше, и без пересчёта в список за
    сегодня попадали бы три часа вчерашней ночи.
    """
    try:
        start = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone)
    except ValueError:
        return None
    return (start.astimezone(UTC), (start + timedelta(days=1)).astimezone(UTC))


def format_local_time(
    value: datetime,
    timezone: ZoneInfo,
    pattern: str = "%d.%m.%Y %H:%M",
) -> str:
    """Метка времени из базы в том виде, в каком её читает человек.

    Хранится всё в UTC, а sqlite отдаёт метки вообще без зоны — печатать их
    как есть значит показывать проход на несколько часов раньше, чем он был.
    Значение без зоны здесь считается UTC, а не местным временем: так его
    записал recorder.
    """
    return _as_utc(value).astimezone(timezone).strftime(pattern)


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


def _process_is_gone(payload: dict[str, object]) -> bool:
    """Жив ли процесс, записавший это состояние.

    Отвечает «нет» только уверенно. Нет идентификатора — значит запись сделана
    старой версией, и судить о ней нужно по времени, а не по догадке.
    """
    pid = payload.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        # Чужой процесс с этим номером: жив, хоть и не наш.
        return False
    return False


def _expire_stale_training(
    status_path: Path,
    payload: dict[str, object],
    *,
    now: datetime | None = None,
) -> bool:
    """Пометить брошенный прогон брошенным. `True` — пометили.

    Два признака, и первый важнее. Процесс, которого нет, — это ответ сразу:
    12 августа перезапуск веб-службы убил пересчёт эталонов, и восемь часов
    подряд админка показывала «Готовлю кадры размеченных проходов», а кнопка
    оставалась серой. Второй признак — молчание heartbeat — остаётся для
    случаев, когда процесс жив, но работать перестал.
    """
    if payload.get("state") not in ACTIVE_TRAINING_STATES:
        return False
    if _process_is_gone(payload):
        write_status(status_path, state="failed", message=GONE_MESSAGE)
        return True
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
    detect_stream_path: str = ""
    preview_stream_path: str = ""
    username: str = ""
    password: str | None = None
    mac: str = ""
    circuit: CameraCircuit | None = None
    role: CameraRole | None = None
    rotate_180: bool = False
    enabled: bool = True
    decode: str = "auto"
    note: str = ""
    rtsp_url: str = ""


class CaptureModePayload(BaseModel):
    state: CaptureState


class CircuitsPayload(BaseModel):
    """Что просит пульт узла.

    `active` — контур, которому отдаётся узел целиком; остальные
    останавливаются. `circuit` вместе с `enabled` включает или выключает один
    контур, посторонившись перед чужим движком: калитка и регистратор — это
    один Frigate, и гасить один ради другого незачем.

    `exclusive` снимает ограничение узла. Это единственное, что нужно поменять
    после переезда на железо, которое тянет всё сразу.
    """

    active: CircuitName | None = None
    circuit: CircuitName | None = None
    enabled: bool | None = None
    exclusive: bool | None = None
    gate_recording: GateRecording | None = None


class CatDoorPayload(BaseModel):
    """Настройки дверцы из админки. Режим строкой — как у калитки."""

    mode: str = "off"
    relay: int = 2
    hold_ms: int = 5000
    allowed_cats: list[str] = Field(default_factory=list)
    min_confidence: float = 0.9
    confirm_frames: int = 5
    cooldown_seconds: int = 30
    open_delay_seconds: float = 5.0


class GatePersonPayload(BaseModel):
    name: str
    phone_ip: str | None = None
    # `null` — оставить сохранённый MAC. Наружу он выходит только замаскированным,
    # и админка физически не может прислать обратно то, чего не получала.
    phone_mac: str | None = None


class MovementRulesPayload(BaseModel):
    enabled: bool = False
    approaching_ratio: float = 2.0
    leaving_ratio: float = 0.5
    open_when_still: bool = True
    return_window_seconds: int = 30


class GateAccessPayload(BaseModel):
    """Настройки автооткрытия из админки.

    Режим приходит строкой, а не перечислением: значение `on` должно отвергаться
    внятной причиной — «требуется письменное разрешение владельца», — а не
    списком допустимых вариантов, из которого причина не видна.
    """

    mode: str = "off"
    gates: list[GateSpec] = Field(default_factory=list)
    allowed_names: list[str] = Field(default_factory=list)
    require_second_factor: bool = False
    people: list[GatePersonPayload] = Field(default_factory=list)
    min_score: float = 0.93
    pulse_ms: int = 500
    cooldown_seconds: int = 20
    open_from: str = "07:00"
    open_to: str = "23:00"
    movement: MovementRulesPayload | None = None


class CloudBackupPayload(BaseModel):
    enabled: bool = False
    bucket: str = ""
    endpoint: str = "https://storage.yandexcloud.net"
    region: str = "ru-central1"
    prefix: str = "events"
    lookback_hours: int = 24
    cameras: list[str] = Field(default_factory=list)
    retention_days: int = 30
    # Кошачьи проходы — свой бакет, префикс и срок, но те же ключи.
    cats_enabled: bool = False
    cats_bucket: str = ""
    cats_prefix: str = "cat-events"
    cats_retention_days: int = 90
    # Пусто — «оставить прежний»: секрет наружу не выходит и вернуться не может.
    access_key_id: str = ""
    secret_key: str = ""


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
        "detect_stream_path": camera.detect_stream_path,
        "preview_stream_path": camera.preview_stream_path,
        "username": camera.username,
        "has_password": bool(camera.password),
        "mac": camera.mac,
        "circuit": camera.circuit,
        "role": camera.role,
        "rotate_180": camera.rotate_180,
        "enabled": camera.enabled,
        # Путь декодирования обязан быть виден: настройка, которую нельзя
        # прочитать из API, обнаруживается только раскопками в собранной
        # конфигурации Frigate — заметили при переводе камер на QSV.
        "decode": camera.decode,
        "note": camera.note,
        "configured": bool(camera.rtsp_url()),
    }


# Сколько ждать ответа службы на нажатие кнопки. Она смотрит файл раз в
# секунду; три секунды покрывают опрос, импульс и запись в журнал, и человек у
# калитки столько ждёт спокойно.
OPEN_RESULT_TIMEOUT_SECONDS = 5.0
# Насколько свежей должна быть отметка службы, чтобы считать её живой. Она
# отмечается пять раз в секунду, так что три — с большим запасом.
HEARTBEAT_FRESH_SECONDS = 3.0
OPEN_RESULT_POLL_SECONDS = 0.1


def _await_open_result(data_dir: Path, request_id: str) -> OpenResult | None:
    """Дождаться ответа службы на просьбу. `None` — не дождались.

    Ответ ищется по идентификатору просьбы, а не «по свежести файла»: рядом
    лежит ответ на прошлое нажатие, и принять его за свой значило бы сказать
    «открыл» там, где не открыл никто.
    """
    path = open_result_path(data_dir)
    deadline = monotonic() + OPEN_RESULT_TIMEOUT_SECONDS
    while True:
        result = load_open_result(path)
        if result is not None and result.id == request_id:
            return result
        if monotonic() >= deadline:
            return None
        sleep(OPEN_RESULT_POLL_SECONDS)


def _error_messages(exc: ValidationError) -> list[str]:
    """Только сообщения об ошибках.

    Полный текст pydantic цитирует ещё и значения полей — а среди них MAC
    телефона, то есть персональные данные, которым в ответе API делать нечего.
    """
    return [str(error["msg"]).removeprefix("Value error, ") for error in exc.errors()]


def _gate_access_view(config: GateAccessConfig) -> dict[str, object]:
    """Настройки калитки для API. MAC телефона наружу выходит только хвостом."""
    return {
        "mode": config.mode,
        "gates": [gate.model_dump() for gate in config.gates],
        "allowed_names": list(config.allowed_names),
        "require_second_factor": config.require_second_factor,
        "people": [
            {
                "name": person.name,
                "phone_ip": person.phone_ip,
                "phone_mac": mask_mac(person.phone_mac),
                "has_phone_mac": bool(person.phone_mac),
            }
            for person in config.people
        ],
        "min_score": config.min_score,
        "pulse_ms": config.pulse_ms,
        "cooldown_seconds": config.cooldown_seconds,
        "open_from": config.open_from,
        "open_to": config.open_to,
        "movement": config.movement.model_dump(),
        # Есть ли к чему возвращаться: кнопка отката без этого бессмысленна.
        "preset_applied": config.replaced_by_preset is not None,
        "preset_name": MOVEMENT_PRESET_NAME,
    }


def _merge_gate_access(
    payload: GateAccessPayload, stored: GateAccessConfig
) -> GateAccessConfig:
    """Наложить присланное на сохранённое.

    Не присланный MAC означает «оставить как было»: админка получает его
    замаскированным и вернуть настоящий не может. Пустая строка — это «стереть»,
    и это разные намерения.
    """
    people = []
    for person in payload.people:
        known = stored.person(person.name)
        people.append(
            {
                "name": person.name,
                "phone_ip": (
                    person.phone_ip
                    if person.phone_ip is not None
                    else (known.phone_ip if known else "")
                ),
                "phone_mac": (
                    person.phone_mac
                    if person.phone_mac is not None
                    else (known.phone_mac if known else "")
                ),
            }
        )
    values = payload.model_dump(exclude={"people", "movement"})
    # Правила движения без них же остаются прежними: панель настроек и
    # переключатель пресета живут рядом и не должны затирать друг друга.
    values["movement"] = (
        payload.movement.model_dump() if payload.movement is not None else stored.movement.model_dump()
    )
    values["replaced_by_preset"] = stored.replaced_by_preset
    return GateAccessConfig.model_validate({**values, "people": people})


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
    door_monitor = getattr(app.state, "door_monitor", None)
    if door_monitor is not None:
        door_monitor.start()
    interval = notifier.watchdog_interval_seconds
    task = (
        asyncio.create_task(_ping_watchdog(notifier, interval))
        if interval is not None
        else None
    )
    try:
        yield
    finally:
        # Живое превью — длинный ответ, который по замыслу не заканчивается,
        # пока камера жива. При остановке службы uvicorn ждёт завершения
        # запросов, поэтому одна открытая вкладка админки держала бы её до
        # принудительного убийства по таймауту: полторы минуты на каждый
        # перезапуск, и не там, где ждёшь.
        app.state.shutting_down.set()
        if door_monitor is not None:
            door_monitor.stop()
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
    # Монитор создаётся здесь, а стартует в lifespan: TestClient без lifespan
    # не должен плодить потоки, а установка без контроллера — вообще ничего.
    app.state.door_monitor = (
        DoorMonitor(
            host=settings.door_controller_host,
            port=settings.door_controller_port,
            reed_input=settings.door_reed_input,
            lock_relay=settings.door_lock_relay,
            stable_seconds=settings.door_position_stable_seconds,
        )
        if settings.door_controller_host
        else None
    )
    # Признак остановки службы. Событие, а не флаг: живое превью крутится в
    # потоке пула, и ему нужно узнать об остановке, не опрашивая ничего.
    app.state.shutting_down = ThreadEvent()
    # Предыдущий замер тактов CPU: процент — всегда разность двух чтений,
    # и первый ответ после старта честно отдаёт null вместо числа с потолка.
    app.state.cpu_sample = read_cpu_sample()
    templates.env.globals["asset_version"] = asset_version
    templates.env.globals["circuit_titles"] = CIRCUIT_TITLES
    templates.env.globals["recording_titles"] = RECORDING_TITLES
    # Пояс читается один раз на сборку приложения: он меняется правкой
    # конфигурации, то есть с перезапуском службы, а разбор имени зоны на
    # каждый проход в списке был бы работой впустую.
    display_timezone = ZoneInfo(settings.display_timezone)
    def _ago(value: object) -> str:
        """«7 мин назад» из эпохи или ISO-строки; пустое — «никогда»."""
        if value in (None, "", 0):
            return "никогда"
        try:
            moment = (
                datetime.fromisoformat(str(value)).timestamp()
                if isinstance(value, str)
                else float(value)  # type: ignore[arg-type]
            )
        except (TypeError, ValueError):
            return str(value)
        seconds = max(0.0, time() - moment)
        if seconds < 90:
            return f"{seconds:.0f} с назад"
        if seconds < 5400:
            return f"{seconds / 60:.0f} мин назад"
        if seconds < 172800:
            return f"{seconds / 3600:.0f} ч назад"
        return f"{seconds / 86400:.0f} дн назад"

    templates.env.filters["ago"] = _ago

    def _hint(label: str, text: str) -> Markup:
        """Значок «?» с подсказкой. Описания полей и панелей живут здесь, а не
        абзацами под формой: пульт читается по приборам, пояснение — по
        требованию. Один и тот же значок на всех страницах."""
        return Markup(
            '<span class="hint" tabindex="0" role="note" aria-label="Подсказка: {}" '
            'data-tooltip="{}">?</span>'
        ).format(label, text)

    templates.env.globals["hint"] = _hint
    templates.env.filters["local_time"] = lambda value, pattern="%d.%m.%Y %H:%M": (
        format_local_time(value, display_timezone, pattern)
    )

    def current_detection_config() -> DetectionConfig:
        return load_detection_config(
            detection_config_path(settings.data_dir),
            DetectionConfig.from_settings(settings),
        )

    def current_circuits() -> NodeCircuits:
        return load_circuits(circuits_path(settings.data_dir))

    def _latest_dataset(runs_root: Path) -> Path | None:
        """Кропы последнего прогона, если они уцелели."""
        for status in sorted(runs_root.glob("*/status.json"), reverse=True):
            dataset = status.parent / "dataset"
            if (dataset / "dataset.jsonl").is_file():
                return dataset
        return None

    def _latest_status(runs_root: Path) -> dict[str, object] | None:
        """Последнее состояние прогона, с поправкой на оборванные.

        Проверка живучести делается при чтении, а не по расписанию: иначе
        оборванный прогон держал бы кнопку серой до тех пор, пока кто-нибудь не
        попробует запустить новый — то есть ровно до того момента, когда
        пробовать уже незачем.
        """
        statuses = sorted(runs_root.glob("*/status.json"))
        if not statuses:
            return None
        payload = _read_status(statuses[-1])
        if payload is not None and _expire_stale_training(statuses[-1], payload):
            return _read_status(statuses[-1])
        return payload

    def _quality_cell(value: object) -> dict[str, float] | None:
        """Привести отчёт к паре «узнано своих / пущено чужих».

        Формы две: пересчёт эталонов пишет доли `recognised`/`false_accepts`,
        приёмка — `false_reject_rate`/`false_accept_rate`. Смысл один, и в
        таблице он обязан выглядеть одинаково.
        """
        if not isinstance(value, dict):
            return None
        if "recognised" in value:
            return {
                "recognised": round(float(value["recognised"]) * 100, 1),
                "false_accepts": round(float(value.get("false_accepts", 0.0)) * 100, 1),
            }
        if "false_reject_rate" in value:
            return {
                "recognised": round((1 - float(value["false_reject_rate"])) * 100, 1),
                "false_accepts": round(float(value.get("false_accept_rate", 0.0)) * 100, 1),
            }
        return None

    def _installed_identity_models() -> list[dict[str, object]]:
        """Что стоит на узле: вид, вход, порог, коты, дата файла.

        Читается из метаданных рядом с моделью — того же файла, который правит
        пересчёт эталонов. Отдельного реестра нет намеренно: второй список
        разошёлся бы с первым молча.
        """
        rows: list[dict[str, object]] = []
        for role, path in (
            ("face", settings.identity_face_model_path),
            ("side", settings.identity_side_model_path),
        ):
            if path is None or not path.is_file():
                rows.append({"role": role, "present": False})
                continue
            try:
                payload = json.loads(
                    path.with_suffix(".json").read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                rows.append({"role": role, "present": False})
                continue
            rows.append(
                {
                    "role": role,
                    "present": True,
                    "kind": payload.get("kind", "logits"),
                    "image_size": payload.get("image_size"),
                    "threshold": payload.get("threshold"),
                    "labels": payload.get("labels", []),
                    "installed_at": datetime.fromtimestamp(
                        path.stat().st_mtime, tz=UTC
                    ),
                }
            )
        return rows

    def _training_history(limit: int = 8) -> list[dict[str, object]]:
        """Прошлые прогоны — пересчёты эталонов и обучения на узле, вперемешку
        по времени. Это и есть динамика: по строкам видно, двигается ли
        качество, а не кажется ли, что двигается.
        """
        rows: list[dict[str, object]] = []
        for status_path in (settings.data_dir / "training" / "prototypes").glob(
            "*/status.json"
        ):
            payload = _read_status(status_path) or {}
            report = payload.get("report")
            if payload.get("state") != "completed" or not isinstance(report, dict):
                continue
            views = {}
            raw_views = report.get("views")
            if isinstance(raw_views, dict):
                for role, view in raw_views.items():
                    if isinstance(view, dict) and isinstance(view.get("before"), dict):
                        views[role] = {
                            "new": _quality_cell(view.get("after")),
                            "installed": _quality_cell(view.get("before")),
                            "better": bool(view.get("applied")),
                        }
            rows.append(
                {
                    "kind": "prototypes",
                    "title": "Пересчёт эталонов",
                    "run": status_path.parent.name,
                    "at": payload.get("updated_at"),
                    "views": views,
                }
            )
        for status_path in (settings.data_dir / "training" / "local").glob(
            "*/status.json"
        ):
            payload = _read_status(status_path) or {}
            raw = payload.get("views")
            if payload.get("state") != "completed" or not isinstance(raw, dict):
                continue
            views = {
                role: {
                    "new": _quality_cell(view.get("new")),
                    "installed": _quality_cell(view.get("installed")),
                    "better": bool(view.get("better")),
                }
                for role, view in raw.items()
                if isinstance(view, dict)
            }
            rows.append(
                {
                    "kind": "training",
                    "title": "Обучение на узле",
                    "run": status_path.parent.name,
                    "at": payload.get("updated_at"),
                    "views": views,
                }
            )
        rows.sort(key=lambda row: str(row.get("at") or ""), reverse=True)
        return rows[:limit]

    def current_cat_door() -> CatDoorConfig:
        return load_cat_door_config(cat_door_config_path(settings.data_dir), CatDoorConfig())

    def current_gate_access() -> GateAccessConfig:
        return load_gate_access_config(
            gate_access_config_path(settings.data_dir), GateAccessConfig()
        )

    def current_archive_depth() -> ArchiveDepth:
        _, motion_days = RECORDING_RETENTION.get(
            current_circuits().gate_recording, (0, 0)
        )
        return read_archive_depth(
            settings.gate_database_path,
            settings.gate_recordings_path,
            motion_days,
        )

    def archive_depth_view(depth: ArchiveDepth) -> dict[str, object]:
        hours = depth.depth_hours
        return {
            "oldest": (
                format_local_time(depth.oldest, display_timezone)
                if depth.oldest
                else None
            ),
            "hours": round(hours, 1) if hours is not None else None,
            "retention_days": depth.retention_days,
            "free_gb": round(depth.free_gb) if depth.free_gb is not None else None,
            "total_gb": round(depth.total_gb) if depth.total_gb is not None else None,
            # Признак «место кончается раньше срока»: настроена неделя, а архива
            # меньше. Это единственное, ради чего показатель и заведён.
            "short": depth.short,
        }

    def circuits_view(circuits: NodeCircuits) -> dict[str, object]:
        status = gate_status(settings.gate_service_unit, settings.gate_api_url)
        return {
            "gate_state": status.state,
            "gate_detail": status.detail,
            "gate_deployed": status.deployed,
            "gate_ui_url": settings.gate_ui_url,
            "cats": circuits.cats,
            "gate": circuits.gate,
            "nvr": circuits.nvr,
            "exclusive": circuits.exclusive,
            "gate_recording": circuits.gate_recording,
            "active": list(circuits.active_names),
        }

    def current_capture_mode() -> CaptureMode:
        return load_capture_mode(
            capture_mode_path(settings.data_dir),
            CaptureMode.from_settings(settings),
        )

    def cached_gate_status() -> GateStatus:
        """Состояние службы Frigate с короткой памятью.

        Шапка показывает его на каждой странице, а узнаётся оно запуском
        `systemctl` — процессом на каждую отрисовку. Пары секунд достаточно,
        чтобы человек не заметил задержки, и хватает, чтобы страница с
        картинками не плодила по вызову на каждый свой запрос.
        """
        now = monotonic()
        cached = getattr(app.state, "gate_status_cache", None)
        if cached is not None and now - cached[0] < GATE_STATUS_CACHE_SECONDS:
            return cached[1]
        status = gate_status(settings.gate_service_unit, settings.gate_api_url)
        app.state.gate_status_cache = (now, status)
        return status

    def circuit_nav() -> list[dict[str, object]]:
        """Контуры для шапки: где мы, что работает и в каком состоянии.

        Живёт в шапке, а не на страницах контуров, потому что вопрос «что
        сейчас работает» возникает раньше выбора страницы: узел отдан одному
        движку, и увидеть это нужно, не уходя со своей вкладки.
        """
        circuits = current_circuits()
        # Служба спрашивается только когда есть кому работать: при выключенных
        # контурах Frigate ответ и так известен, а процесс стоил бы столько же.
        status = cached_gate_status() if circuits.frigate_wanted else None
        items: list[dict[str, object]] = []
        for name, title in CIRCUIT_TITLES.items():
            if name == "cats":
                mode = current_capture_mode()
                state = CAPTURE_STATE_TITLES[mode.state] if circuits.cats else "остановлен"
                tone = ("beacon-live" if mode.state == "record" else "beacon-ok") if (
                    circuits.cats
                ) else ""
            elif not circuits.is_active(name):
                state, tone = "выключен", ""
            else:
                state = GATE_STATE_TITLES.get(status.state if status else "unknown", "")
                tone = {
                    "running": "beacon-ok",
                    "failed": "beacon-bad",
                    "degraded": "beacon-bad",
                    "unknown": "beacon-bad",
                }.get(status.state if status else "unknown", "")
            items.append(
                {
                    "name": name,
                    "title": title,
                    "href": "/" if name == "cats" else f"/{name}",
                    "state": state,
                    "tone": tone,
                }
            )
        return items

    templates.env.globals["circuit_nav"] = circuit_nav

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
        lpage: int = 1,
        who: str = "named",
        cat: str | None = None,
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
        known_cat_names = session.scalars(
            select(Event.cat_name)
            .where(Event.label == "known_cat", Event.cat_name.is_not(None))
            .distinct()
            .order_by(Event.cat_name)
        ).all()
        # Незнакомые фильтры молча сбрасываются к умолчанию, а не отвергаются —
        # по той же причине, по которой ниже прижимается номер страницы.
        if who not in LABELED_FILTERS:
            who = "named"
        if cat is not None and cat not in known_cat_names:
            cat = None
        labeled_filter = (
            (Event.label == "known_cat") & (Event.cat_name == cat)
            if cat is not None
            else LABELED_FILTERS[who]
        )

        def load_page(condition, page_number: int):
            total = session.execute(
                select(func.count(Event.id)).where(condition)
            ).scalar_one()
            pages = max(1, ceil(total / EVENTS_PER_PAGE))
            # Номер страницы прижимается к диапазону, а не отвергается: после
            # удаления последнего прохода на странице сохранённая ссылка иначе
            # вела бы в пустоту.
            page_number = min(max(page_number, 1), pages)
            rows = session.scalars(
                select(Event)
                .where(condition)
                # Тай-брейк по id обязателен: автоматические проходы пишутся
                # пачками и могут делить время создания, а при неустойчивом
                # порядке одна и та же запись прыгала бы между страницами.
                .order_by(Event.created_at.desc(), Event.id.desc())
                .limit(EVENTS_PER_PAGE)
                .offset((page_number - 1) * EVENTS_PER_PAGE)
            ).all()
            return rows, total, page_number, pages

        events, events_total, page, pages = load_page(Event.label == "pending", page)
        labeled_events, labeled_total, lpage, labeled_pages = load_page(
            labeled_filter, lpage
        )
        event_ids = [event.id for event in events] + [
            event.id for event in labeled_events
        ]
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
                "events_total": events_total,
                "events_shown_from": (page - 1) * EVENTS_PER_PAGE + 1 if events else 0,
                "events_shown_to": (page - 1) * EVENTS_PER_PAGE + len(events),
                "labeled_events": labeled_events,
                "labeled_page": lpage,
                "labeled_pages": labeled_pages,
                "labeled_total": labeled_total,
                "labeled_who": who,
                "labeled_cat": cat,
                "door_configured": app.state.door_monitor is not None,
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
                "notifications": _notifications_view(settings),
                "cat_door": current_cat_door().model_dump(),
                "prototypes_status": _latest_status(
                    settings.data_dir / "training" / "prototypes"
                ),
                "export_status": _latest_status(settings.data_dir / "training" / "exports"),
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
                    # Группы выводятся из списка контуров, а не перечисляются
                    # руками. Перечисленные руками, они однажды разошлись с
                    # ним: контур регистратора появился, а его группа — нет, и
                    # камеры этого контура пропали со страницы совсем. Камера
                    # при этом жила в инвентаре и попадала в конфигурацию
                    # Frigate — то есть работала, но поправить или удалить её
                    # было негде.
                    for circuit, title in (*CIRCUIT_TITLES.items(), (None, "Без контура"))
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

    @app.post("/api/notifications/test")
    def test_notification() -> dict[str, object]:
        """Проверить бота, не дожидаясь кота.

        Настройка уведомлений — единственное место в системе, где ошибка
        проявляется молчанием: неверный токен, чужой чат и остановленный бот
        выглядят одинаково никак. Кнопка превращает это в ответ Телеграма.
        """
        client = TelegramClient(
        settings.telegram_bot_token,
        settings.telegram_chat_id,
        api_base=settings.telegram_api_base,
    )
        if not client.configured:
            raise HTTPException(422, "Бот не настроен: нет токена или адреса чата")
        try:
            client.send_message("CatDoor на связи: проверка уведомлений.")
        except TelegramError as exc:
            hint = hint_for(str(exc))
            return {"ok": False, "reason": f"{exc} — {hint}" if hint else str(exc)}
        return {"ok": True, "reason": None}

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
                    # Субпоток, если он есть: превью шириной 960 точек не стоит
                    # декодирования основного потока. Замер 5 августа — 47%
                    # ядра на карточку камеры 2560×1440 против ~4% с субпотока.
                    camera.preview_rtsp_url(),
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
                stopping=app.state.shutting_down.is_set,
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

    @app.get("/api/system/stats")
    def system_stats() -> dict[str, object]:
        """Загрузка узла для шапки: без psutil, прямо из /proc.

        Процент CPU — разность с прошлым вызовом этого же эндпоинта: браузер
        опрашивает раз в несколько секунд, и интервал опроса и есть окно
        усреднения. На системах без /proc все поля — null, шапка молчит.
        """
        current = read_cpu_sample()
        percent = cpu_percent(app.state.cpu_sample, current)
        if current is not None:
            app.state.cpu_sample = current
        memory = read_memory_mb()
        try:
            load = [round(value, 2) for value in os.getloadavg()]
        except OSError:
            load = None
        return {
            "cpu_percent": percent,
            "memory_used_mb": memory[0] if memory else None,
            "memory_total_mb": memory[1] if memory else None,
            "load": load,
            "cpu_temperature_c": read_cpu_temperature(),
        }

    def current_health() -> dict[str, object]:
        """Сверка факта с эталоном — одним местом для страницы и для API.

        Эталон живёт в `src/common/health.py` вместе с причиной каждого числа:
        расхождение без причины через месяц не даёт решить, чинить факт или
        править эталон.
        """
        status = read_recorder_status(recorder_status_path(settings)) or {}
        stats = frigate_stats(settings.gate_api_url)
        metrics = read_node_metrics(
            [settings.data_dir, Path("/srv/data"), settings.gate_recordings_path]
        )
        groups = build_health(
            recorder_status=status,
            detection_config=current_detection_config().model_dump(mode="json"),
            frigate_stats=stats,
            gate_detector_device=settings.gate_detector_device,
            metrics=metrics,
            gpu_name=str(status.get("detector_device_name") or "") or None,
            backup_status=_read_status(backup_status_path(settings.data_dir)),
            gate_service=_read_status(gate_health_path(settings.data_dir)),
        )
        return {
            "status": overall_status(groups),
            "groups": [
                {
                    "title": group.title,
                    "status": group.status,
                    "checks": [asdict(check) for check in group.checks],
                }
                for group in groups
            ],
            "metrics": {
                "load_average": metrics.load_average,
                "load_per_core": metrics.load_per_core,
                "cpu_count": metrics.cpu_count,
                "cpu_temperature_c": metrics.cpu_temperature_c,
                "disk_temperature_c": metrics.disk_temperature_c,
                "memory_used_percent": metrics.memory_used_percent,
                "memory_total_gb": metrics.memory_total_gb,
                "gpu_frequency_mhz": metrics.gpu_frequency_mhz,
                "gpu_frequency_max_mhz": metrics.gpu_frequency_max_mhz,
                "gpu_frequency_percent": metrics.gpu_frequency_percent,
                "gpu_occupancy_percent": gpu_occupancy_percent(status, stats),
                "uptime_seconds": metrics.uptime_seconds,
                "disks": [asdict(disk) for disk in metrics.disks],
            },
        }

    @app.get("/api/health")
    def health_api() -> dict[str, object]:
        return current_health()

    def _health_dashboard(session: Session) -> dict[str, object]:
        """Сводка для дашборда «Состояния»: модули, проходы, отказы.

        Всё за сегодняшние местные сутки: вопрос владельца к этой странице —
        «как дела сейчас», а не «как дела вообще». Итоги за всё время живут на
        своих страницах — в журнале калитки и в проходах.
        """
        day_start = (
            datetime.now(display_timezone).replace(hour=0, minute=0, second=0, microsecond=0)
        ).astimezone(UTC)

        opened = func.count(GatePass.id).filter(GatePass.decision == "opened")
        gate_row = session.execute(
            select(func.count(GatePass.id), opened).where(
                GatePass.created_at >= day_start, GatePass.label == "person"
            )
        ).one()
        denial_rows = session.execute(
            select(GatePass.decision, func.count(GatePass.id))
            .where(
                GatePass.created_at >= day_start,
                GatePass.label == "person",
                GatePass.decision.notin_(("opened", "dry_run", "logged")),
            )
            .group_by(GatePass.decision)
            .order_by(func.count(GatePass.id).desc())
        ).all()
        gate_people = session.execute(
            select(GatePass.person_name, func.count(GatePass.id))
            .where(
                GatePass.created_at >= day_start,
                GatePass.decision == "opened",
                GatePass.person_name.is_not(None),
            )
            .group_by(GatePass.person_name)
            .order_by(func.count(GatePass.id).desc())
        ).all()

        events_row = session.execute(
            select(
                func.count(Event.id),
                func.count(Event.id).filter(Event.status == "failed"),
            ).where(Event.created_at >= day_start)
        ).one()
        # Опознания — глазами модели, а не разметки: свежие проходы владелец
        # ещё не размечал, а вопрос дашборда — «кого узнала автоматика».
        cat_rows = session.execute(
            select(EventPrediction.predicted_name, func.count(func.distinct(Event.id)))
            .join(Event, Event.id == EventPrediction.event_id)
            .where(
                Event.created_at >= day_start,
                EventPrediction.origin == "online",
            )
            .group_by(EventPrediction.predicted_name)
        ).all()
        recognised = {name: count for name, count in cat_rows if name}
        uncertain = sum(count for name, count in cat_rows if not name)

        heartbeat = heartbeat_age(heartbeat_path(settings.data_dir))
        return {
            "gate": {
                "total": gate_row[0],
                "opened": gate_row[1],
                "denied": sum(count for _, count in denial_rows),
                "denials": [
                    (GATE_PASS_DECISIONS.get(code, code), count)
                    for code, count in denial_rows[:4]
                ],
                "people": gate_people[:5],
                "service_alive": heartbeat is not None
                and heartbeat <= HEARTBEAT_FRESH_SECONDS,
            },
            "cats": {
                "events": events_row[0],
                "failed": events_row[1],
                "recognised": sorted(
                    recognised.items(), key=lambda item: item[1], reverse=True
                )[:5],
                "uncertain": uncertain,
            },
        }

    def _cloud_backup_view(config: CloudBackupConfig) -> dict[str, object]:
        """Настройки для админки: секрет наружу не выходит, только его наличие."""
        values = config.model_dump(exclude={"secret_key", "access_key_id"})
        values["has_secret"] = bool(config.secret_key)
        # Уборку в облаке делает lifecycle бакета, а не служба: её ключ не
        # умеет удалять по построению. Команду настройки печатаем готовой —
        # владелец выполняет один раз со своей машины со своим админским
        # профилем (тот же приём, что с активацией моделей).
        #
        # `put-bucket-lifecycle-configuration` **заменяет** конфигурацию бакета
        # целиком, а не добавляет правило. Поэтому две команды на один бакет —
        # это мина: запустив вторую, владелец молча снёс бы правило из первой и
        # остановил уборку. Отсюда устройство ниже: пока бакеты общие, правила
        # печатаются **одной** командой, и отдельная кошачья не показывается
        # вовсе.
        def _rule(name: str, prefix: str, days: int) -> dict[str, object]:
            return {
                "ID": name,
                "Status": "Enabled",
                "Filter": {"Prefix": f"{prefix}/"},
                "Expiration": {"Days": days},
                "NoncurrentVersionExpiration": {"NoncurrentDays": days},
                "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1},
            }

        def _command(bucket: str, rules: list[dict[str, object]]) -> str:
            payload = json.dumps({"Rules": rules}, separators=(",", ":"))
            return (
                f"aws --endpoint-url {config.endpoint} "
                f"s3api put-bucket-lifecycle-configuration "
                f"--bucket {bucket} "
                f"--lifecycle-configuration '{payload}'"
            )

        cats_rule = (
            _rule("catdoor-cats-retention", config.cats_prefix, config.cats_retention_days)
            if config.cats_target_bucket and config.cats_retention_days
            else None
        )
        shares_bucket = config.cats_target_bucket == config.bucket
        if config.bucket and config.retention_days:
            rules = [_rule("catdoor-events-retention", config.prefix, config.retention_days)]
            if cats_rule is not None and shares_bucket:
                rules.append(cats_rule)
            values["lifecycle_command"] = _command(config.bucket, rules)
        if cats_rule is not None and not shares_bucket:
            values["cats_lifecycle_command"] = _command(
                config.cats_target_bucket, [cats_rule]
            )
        # Длина — безопасная диагностика: у секретов Яндекса она всегда 40, и
        # «37» или «41» мгновенно выдаёт обрезанную или слипшуюся вставку,
        # которую иначе видно только как SignatureDoesNotMatch.
        values["secret_length"] = len(config.secret_key)
        # Key id — не секрет (секретна только вторая половина пары), а сверять
        # его с bucket policy приходится посимвольно: первая подсказка из
        # четырёх букв оказалась бесполезной — с «YCAJ» начинаются все ключи
        # Яндекса, и именно это скрыло несовпадение ключей 14 августа.
        values["access_key_id"] = config.access_key_id
        return values

    @app.get("/api/cloud-backup/settings")
    def get_cloud_backup_settings() -> dict[str, object]:
        return _cloud_backup_view(
            load_cloud_backup_config(
                cloud_backup_config_path(settings.data_dir), CloudBackupConfig()
            )
        )

    @app.put("/api/cloud-backup/settings")
    def update_cloud_backup_settings(payload: CloudBackupPayload) -> dict[str, object]:
        """Пустые ключи в запросе означают «оставить прежние».

        Админка получает секрет замаскированным и вернуть его не может — тот же
        приём, что с MAC телефонов у калитки.
        """
        path = cloud_backup_config_path(settings.data_dir)
        stored = load_cloud_backup_config(path, CloudBackupConfig())
        values = payload.model_dump(exclude={"access_key_id", "secret_key"})
        # strip обязателен: секрет копируют из консоли, и невидимый перенос
        # строки на конце превращается в SignatureDoesNotMatch — 14 августа
        # именно он съел вечер.
        values["access_key_id"] = payload.access_key_id.strip() or stored.access_key_id
        values["secret_key"] = payload.secret_key.strip() or stored.secret_key
        config = CloudBackupConfig.model_validate(values)
        try:
            save_cloud_backup_config(path, config)
        except OSError as exc:
            raise HTTPException(500, "Не удалось сохранить настройки облака") from exc
        return _cloud_backup_view(config)

    @app.get("/api/cloud-backup/state")
    def cloud_backup_state() -> dict[str, object]:
        """Живое состояние выгрузчиков — для прогресс-бара на «Состоянии».

        Кошачья копия — отдельная служба со своим журналом, и её состояние
        лежит рядом ключом `cats`: сутки после запуска его не показывал никто.
        """
        state = _read_status(cloud_backup_state_path(settings.data_dir)) or {
            "status": "idle",
            "message": "Выгрузчик ещё не отчитывался",
        }
        state["cats"] = _read_status(cats_cloud_backup_state_path(settings.data_dir)) or {
            "status": "idle",
            "message": "Кошачья копия ещё не отчитывалась",
        }
        return state

    @app.post("/api/cloud-backup/test")
    def test_cloud_backup() -> dict[str, object]:
        """Прогон четырёх операций S3 — чтобы отказ прав читался картиной.

        PutObject должен пройти; Get и Delete по нашей же схеме обязаны быть
        запрещены — их «отказ» здесь успех. Паттерн «всё запрещено» означает
        bucket policy, глушащую ключ целиком (условие не совпало), «разрешено
        всё» — policy нет и ключ шире задуманного.
        """
        config = load_cloud_backup_config(
            cloud_backup_config_path(settings.data_dir), CloudBackupConfig()
        )
        if not config.ready:
            raise HTTPException(422, "Сначала задайте бакет и ключи")
        try:
            import boto3  # type: ignore[import-not-found]
            from botocore.exceptions import (  # type: ignore[import-not-found]
                ClientError,
            )
        except ImportError as exc:
            raise HTTPException(500, "boto3 не установлен в venv узла") from exc
        client = boto3.client(
            "s3",
            endpoint_url=config.endpoint,
            region_name=config.region,
            aws_access_key_id=config.access_key_id,
            aws_secret_access_key=config.secret_key,
        )
        probe_key = f"{config.prefix}/probe.txt"
        results: dict[str, str] = {}

        def attempt(name: str, call) -> None:
            try:
                call()
                results[name] = "ok"
            except ClientError as exc:
                results[name] = str(
                    exc.response.get("Error", {}).get("Code") or exc
                )[:80]
            except Exception as exc:  # noqa: BLE001 — сеть и DNS тоже ответы
                results[name] = str(exc)[:120]

        attempt("list", lambda: client.list_objects_v2(Bucket=config.bucket, MaxKeys=1))
        attempt(
            "put",
            lambda: client.put_object(
                Bucket=config.bucket, Key=probe_key, Body=b"catdoor probe"
            ),
        )
        # Бакет с блокировкой по умолчанию требует Content-MD5, а свежий boto3
        # шлёт вместо него CRC32 — вторая попытка различает «политика не пускает»
        # и «нужен MD5».
        digest = base64.b64encode(hashlib.md5(b"catdoor probe").digest()).decode()
        attempt(
            "put_md5",
            lambda: client.put_object(
                Bucket=config.bucket,
                Key=probe_key,
                Body=b"catdoor probe",
                ContentMD5=digest,
            ),
        )
        attempt("get", lambda: client.get_object(Bucket=config.bucket, Key=probe_key))
        attempt(
            "delete", lambda: client.delete_object(Bucket=config.bucket, Key=probe_key)
        )
        return {"bucket": config.bucket, "results": results}

    @app.get("/health", response_class=HTMLResponse)
    def health_page(
        request: Request,
        session: Annotated[Session, Depends(get_session)],
    ) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "health.html",
            {
                "circuit": "health",
                "health": current_health(),
                "dashboard": _health_dashboard(session),
                "cloud_backup": _cloud_backup_view(
                    load_cloud_backup_config(
                        cloud_backup_config_path(settings.data_dir), CloudBackupConfig()
                    )
                ),
                "cloud_state": _read_status(
                    cloud_backup_state_path(settings.data_dir)
                ),
                "cats_state": _read_status(cats_cloud_backup_state_path(settings.data_dir)),
                "watch_state": _read_status(health_watch_state_path(settings.data_dir)),
                "notify_state": _read_status(gate_notify_state_path(settings.data_dir)),
            },
        )

    @app.get("/logs", response_class=HTMLResponse)
    def logs_page(request: Request, source: str = "recorder") -> HTMLResponse:
        selected = source if source in SOURCE_BY_KEY else "recorder"
        return templates.TemplateResponse(
            request,
            "logs.html",
            {
                "circuit": "logs",
                "sources": SOURCES,
                "selected": selected,
                "lines": DEFAULT_LINES,
                "max_lines": MAX_LINES,
                "hints": {item.key: item.hint for item in SOURCES},
            },
        )

    @app.get("/api/logs/{source}")
    def logs_api(source: str, lines: int | None = None) -> dict[str, object]:
        """Хвост журнала одним куском. Фильтр — на стороне страницы: строк
        мало, а серверный grep означал бы второй вызов journalctl на каждую
        букву в поле фильтра."""
        item = SOURCE_BY_KEY.get(source)
        if item is None:
            raise HTTPException(404, f"Нет такого журнала: {source}")
        text, error = read_log(item, settings.data_dir, clamp_lines(lines))
        return {"source": item.key, "title": item.title, "text": text, "error": error}

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

    @app.get("/api/door/status")
    def door_status() -> dict[str, object]:
        monitor = app.state.door_monitor
        if monitor is None:
            return {"configured": False}
        shot = monitor.snapshot()
        return {
            "configured": True,
            "available": shot.available,
            "position": shot.position,
            "position_stable_seconds": (
                round(shot.position_stable_seconds, 1)
                if shot.position_stable_seconds is not None
                else None
            ),
            "reed_closed": shot.reed_closed,
            "lock_relay_on": shot.lock_relay_on,
            "error": shot.error,
        }

    @app.get("/api/circuits")
    def get_circuits() -> dict[str, object]:
        return circuits_view(current_circuits())

    @app.put("/api/circuits")
    def update_circuits(payload: CircuitsPayload) -> dict[str, object]:
        circuits = current_circuits()
        if payload.exclusive is not None:
            circuits = circuits.model_copy(update={"exclusive": payload.exclusive})
        if payload.active is not None:
            circuits = circuits.with_only(payload.active)
        if payload.circuit is not None and payload.enabled is not None:
            circuits = circuits.with_circuit(payload.circuit, payload.enabled)
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

    @app.get("/models", response_class=HTMLResponse)
    def models_page(request: Request) -> HTMLResponse:
        """Инженерия опознания: модели, обучение, динамика.

        Отдельно от «Проходов» по ритму работы: разметка ежедневная, обучение
        еженедельное, и им незачем делить один экран.
        """
        runs_root = settings.data_dir / "training"
        local_runs = sorted((runs_root / "local").glob("*/status.json"))
        local_status = None
        if local_runs:
            payload = _read_status(local_runs[-1])
            if payload is not None and _expire_stale_training(local_runs[-1], payload):
                payload = _read_status(local_runs[-1])
            local_status = payload or {"state": "unknown", "message": "Статус повреждён"}
        with factory() as session:
            summary_row = session.execute(
                select(
                    func.count(Event.id).filter(Event.training_target == "macbook"),
                    func.count(Event.id).filter(Event.training_target == "local"),
                )
            ).one()
        return templates.TemplateResponse(
            request,
            "models.html",
            {
                "local_status": local_status,
                "local_training_epochs": settings.local_training_epochs,
                "installed_models": _installed_identity_models(),
                "training_history": _training_history(),
                "prototypes_status": _latest_status(runs_root / "prototypes"),
                "export_status": _latest_status(runs_root / "exports"),
                "target_counts": {"macbook": summary_row[0], "local": summary_row[1]},
            },
        )

    @app.get("/gate", response_class=HTMLResponse)
    def gate_page(
        request: Request,
        session: Annotated[Session, Depends(get_session)],
        gpage: int = 1,
        who: str = "all",
        name: str | None = None,
        day: str | None = None,
    ) -> HTMLResponse:
        circuits = current_circuits()
        registry = current_registry()
        access = current_gate_access()
        # Имена берутся из библиотеки лиц Frigate, а не вводятся руками: два
        # списка имён разъезжаются и, разъехавшись, молча перестают совпадать.
        face_library = frigate_face_names(settings.gate_api_url)
        orphan_names = [
            name for name in access.allowed_names if face_library and name not in face_library
        ]
        timezone = ZoneInfo(settings.display_timezone)
        midnight = datetime.now(timezone).replace(hour=0, minute=0, second=0, microsecond=0)
        today = midnight.strftime("%Y-%m-%d")
        today_bounds = (
            midnight.astimezone(UTC),
            (midnight + timedelta(days=1)).astimezone(UTC),
        )
        # Счётчики считаются по всей базе, а не по показанной странице: это
        # ответ на вопрос «сколько людей у калитки», и он не зависит от того,
        # что сейчас на экране.
        summary_row = session.execute(
            select(
                func.count(GatePass.id).filter(GatePass.label.is_not("car")),
                func.count(GatePass.id).filter(GatePass.person_name.is_not(None)),
                func.count(GatePass.id).filter(
                    (GatePass.person_name.is_(None)) & (GatePass.label.is_not("car"))
                ),
                func.count(GatePass.id).filter(
                    GatePass.created_at >= today_bounds[0],
                    GatePass.created_at < today_bounds[1],
                ),
                func.count(GatePass.id).filter(GatePass.label == "car"),
            )
        ).one()
        person_names = session.scalars(
            select(GatePass.person_name)
            .where(GatePass.person_name.is_not(None))
            .distinct()
            .order_by(GatePass.person_name)
        ).all()
        # Незнакомые значения фильтров молча сбрасываются к умолчанию — так же,
        # как на вкладке дверцы: сохранённая ссылка не должна отдавать ошибку.
        if who not in GATE_PASS_FILTERS:
            who = "all"
        if name is not None and name not in person_names:
            name = None
        bounds = local_day_bounds(day, timezone) if day else None
        if bounds is None:
            day = None
        criteria: list[ColumnElement[bool]] = [
            GatePass.person_name == name if name is not None else GATE_PASS_FILTERS[who]
        ]
        if bounds is not None:
            criteria.append(GatePass.created_at >= bounds[0])
            criteria.append(GatePass.created_at < bounds[1])
        total = session.execute(
            select(func.count(GatePass.id)).where(*criteria)
        ).scalar_one()
        pages = max(1, ceil(total / EVENTS_PER_PAGE))
        gpage = min(max(gpage, 1), pages)
        passes = session.scalars(
            select(GatePass)
            .where(*criteria)
            # Тай-брейк по id обязателен: события у калитки приходят пачками и
            # делят время начала, а при неустойчивом порядке одна и та же
            # строка прыгала бы между страницами.
            .order_by(GatePass.created_at.desc(), GatePass.id.desc())
            .limit(EVENTS_PER_PAGE)
            .offset((gpage - 1) * EVENTS_PER_PAGE)
        ).all()
        return templates.TemplateResponse(
            request,
            "gate.html",
            {
                "circuits": circuits_view(circuits),
                "circuit_titles": CIRCUIT_TITLES,
                "gate_cameras": [
                    # Имя во Frigate — рядом с карточкой камеры намеренно. В
                    # настройках калитки хранится именно оно: события приходят
                    # с ним, а не с идентификатором инвентаря, и `test-cam`
                    # против `test_cam` означал бы «камера не назначена
                    # калиткой» на каждый проход — молча.
                    {**_camera_view(camera), "frigate_name": frigate_camera_name(camera.id)}
                    for camera in registry.cameras
                    if camera.circuit == "gate"
                ],
                "stream_version": uuid4().hex,
                "passes": passes,
                "pass_summary": {
                    "total": summary_row[0],
                    "named": summary_row[1],
                    "unknown": summary_row[2],
                    "today": summary_row[3],
                    "cars": summary_row[4],
                },
                "pass_page": gpage,
                "pass_pages": pages,
                "pass_total": total,
                "pass_who": who,
                "pass_name": name,
                "pass_day": day,
                "pass_today": today,
                "person_names": person_names,
                "pass_decisions": GATE_PASS_DECISIONS,
                "gate_access": _gate_access_view(access),
                # `None` — библиотеку спросить не удалось; это не то же самое,
                # что «лиц не заведено», и страница обязана различать.
                "face_library": face_library,
                # Имя, исчезнувшее из библиотеки, показывается отдельно: молча
                # оставить его в разрешённых значит хранить настройку, которая
                # никогда не сработает.
                "orphan_names": orphan_names,
            },
        )

    @app.get("/api/gate/passes")
    def get_gate_passes(
        session: Annotated[Session, Depends(get_session)],
        since: str | None = None,
        until: str | None = None,
        limit: int = 1000,
    ) -> dict[str, object]:
        """Журнал проходов машинно-читаемо — для аудита логики калитки.

        Страница `/gate` отдаёт то же самое человеку; аудитору нужны голые
        поля: решения, факторы, импульсы и состояние створки сверяются
        скриптом с журналом службы и лентой Frigate, а не глазами. Времена
        `since`/`until` — ISO с зоной или UTC без неё.
        """

        def moment(raw: str | None) -> datetime | None:
            if not raw:
                return None
            try:
                parsed = datetime.fromisoformat(raw)
            except ValueError as exc:
                # Молча проигнорированный фильтр — это весь журнал вместо окна:
                # первый же прогон аудита поймал «+00:00», ставшее пробелом в
                # URL, и завалил отчёт находками недельной давности.
                raise HTTPException(
                    422, f"Время не разобрано: {raw!r} — нужен ISO, например 2026-08-20T09:00:00"
                ) from exc
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)

        criteria: list[ColumnElement[bool]] = []
        low, high = moment(since), moment(until)
        if low is not None:
            criteria.append(GatePass.created_at >= low)
        if high is not None:
            criteria.append(GatePass.created_at < high)
        rows = session.scalars(
            select(GatePass)
            .where(*criteria)
            .order_by(GatePass.created_at.asc(), GatePass.id.asc())
            .limit(min(max(limit, 1), 5000))
        ).all()

        def as_dict(row: GatePass) -> dict[str, object]:
            created = row.created_at
            if created.tzinfo is None:
                # sqlite отдаёт метки без зоны, а записаны они в UTC.
                created = created.replace(tzinfo=UTC)
            factors: object = None
            if row.factors_json:
                try:
                    factors = json.loads(row.factors_json)
                except ValueError:
                    factors = None
            return {
                "id": row.id,
                "frigate_event_id": row.frigate_event_id,
                "created_at": created.isoformat(),
                "camera": row.camera,
                "label": row.label,
                "person_name": row.person_name,
                "plate": row.plate,
                "score": row.score,
                "zone": row.zone,
                "decision": row.decision,
                "reason": row.reason,
                "factors": factors,
                "pulses": row.pulses,
                "last_pulse_at": row.last_pulse_at,
                "pulse_gate_state": row.pulse_gate_state,
                "pulse_ms": row.pulse_ms,
                "modbus_ok": row.modbus_ok,
                "gate_state_before": row.gate_state_before,
                "gate_state_after": row.gate_state_after,
                "approach": row.approach,
                "first_box_bottom": row.first_box_bottom,
                "movement": row.movement,
                "area_ratio": row.area_ratio,
                "has_snapshot": bool(row.snapshot_path),
            }

        return {"passes": [as_dict(row) for row in rows], "count": len(rows)}

    @app.get("/api/gate/access-settings")
    def get_gate_access_settings() -> dict[str, object]:
        return _gate_access_view(current_gate_access())

    @app.put("/api/gate/access-settings")
    def update_gate_access_settings(payload: GateAccessPayload) -> dict[str, object]:
        stored = current_gate_access()
        try:
            config = _merge_gate_access(payload, stored)
        except ValidationError as exc:
            # Пользовательские причины отказа доходят до человека целиком:
            # «режим on требует разрешения владельца» — это то, что он должен
            # прочитать, а не «422 Unprocessable Entity».
            raise HTTPException(422, "; ".join(_error_messages(exc))) from exc
        try:
            save_gate_access_config(gate_access_config_path(settings.data_dir), config)
        except OSError as exc:
            raise HTTPException(500, "Не удалось сохранить настройки калитки") from exc
        # Механику веб не трогает и здесь: файл читает служба
        # `catdoor-gate-access`, у веб-процесса прав на плату нет и не будет.
        return _gate_access_view(config)

    def _request_open(target: str) -> dict[str, object]:
        """Попросить службу открыть и дождаться ответа.

        Общий путь для калитки и дверцы: плату держит один процесс, потому что
        шлюз в прозрачном режиме второго соединения может не принять.
        """
        request = OpenRequest(
            id=uuid4().hex,
            requested_at=time(),
            target=target,
            source="admin",
            # Загрузка, в которой нажали: после перезагрузки эта просьба уже не
            # чья-то, и часам после обрыва питания верить нельзя.
            boot_id=boot_id(),
        )
        try:
            save_open_request(open_request_path(settings.data_dir), request)
        except OSError as exc:
            raise HTTPException(500, "Не удалось передать просьбу службе") from exc
        result = _await_open_result(settings.data_dir, request.id)
        if result is None:
            # «Занята» и «не работает» ведут человека к разным действиям, и
            # путать их нельзя: в первом случае надо повторить через секунду,
            # во втором — идти чинить службу.
            age = heartbeat_age(heartbeat_path(settings.data_dir))
            if age is not None and age <= HEARTBEAT_FRESH_SECONDS:
                raise HTTPException(
                    503, "Служба занята другим делом — нажмите ещё раз через секунду"
                )
            raise HTTPException(504, "Служба не отвечает: проверьте catdoor-gate-access")
        if not result.ok:
            raise HTTPException(502, result.detail)
        return {"ok": True, "detail": result.detail, "pulse_ms": result.pulse_ms}

    @app.get("/api/cat-door/settings")
    def get_cat_door_settings() -> dict[str, object]:
        return current_cat_door().model_dump()

    @app.put("/api/cat-door/settings")
    def update_cat_door_settings(payload: CatDoorPayload) -> dict[str, object]:
        try:
            config = CatDoorConfig.model_validate(payload.model_dump())
        except ValidationError as exc:
            raise HTTPException(422, "; ".join(_error_messages(exc))) from exc
        try:
            save_cat_door_config(cat_door_config_path(settings.data_dir), config)
        except OSError as exc:
            raise HTTPException(500, "Не удалось сохранить настройки дверцы") from exc
        # Механику веб не трогает и здесь: файл читает служба.
        return config.model_dump()

    @app.post("/api/cat-door/open")
    def open_cat_door() -> dict[str, object]:
        """Тестовое открытие дверцы — тем же путём, что и калитка."""
        return _request_open(TARGET_CAT_DOOR)

    @app.post("/api/gate/open")
    def open_gate() -> dict[str, object]:
        """Открыть калитку по кнопке в админке.

        Веб механикой не распоряжается и здесь: он кладёт просьбу файлом, а
        исполняет её служба `catdoor-gate-access` — единственная, которой
        позволено писать на плату. Ответ ждём её же: сказать «открыл», не
        дождавшись подтверждения, — это ровно та ложь, ради отсутствия которой
        всё и разделено.
        """
        config = current_gate_access()
        if not config.gates:
            raise HTTPException(422, "Канал реле калитки не настроен")
        return _request_open(TARGET_GATE)

    @app.post("/api/gate/access-settings/preset")
    def apply_gate_preset(rollback: bool = False) -> dict[str, object]:
        """Накатить пресет правил движения или вернуть, как было.

        Числа пресета подобраны разбором живых проходов 12 августа
        (`docs/gate-direction-experiments.md`), и перебирать их руками человеку
        незачем. Прежние правила запоминаются в том же файле, поэтому откат
        работает и после перезагрузки узла, и через неделю.
        """
        stored = current_gate_access()
        config = rollback_movement_preset(stored) if rollback else apply_movement_preset(stored)
        try:
            save_gate_access_config(gate_access_config_path(settings.data_dir), config)
        except OSError as exc:
            raise HTTPException(500, "Не удалось сохранить настройки калитки") from exc
        return _gate_access_view(config)

    @app.get("/gate/passes/{pass_id}/snapshot")
    def gate_pass_snapshot(
        pass_id: int,
        session: Annotated[Session, Depends(get_session)],
    ) -> FileResponse:
        """Кадр прохода. Запись прохода неизменяема, поэтому кэш навсегда."""
        stored = session.get(GatePass, pass_id)
        if stored is None or not stored.snapshot_path:
            raise HTTPException(404, "Снимок не найден")
        root = settings.data_dir.resolve()
        path = (root / stored.snapshot_path).resolve()
        if root not in path.parents or not path.is_file():
            raise HTTPException(404, "Снимок не найден")
        return FileResponse(
            path,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    @app.get("/nvr", response_class=HTMLResponse)
    def nvr_page(request: Request) -> HTMLResponse:
        registry = current_registry()
        return templates.TemplateResponse(
            request,
            "nvr.html",
            {
                "circuits": circuits_view(current_circuits()),
                "circuit_titles": CIRCUIT_TITLES,
                "nvr_cameras": [
                    _camera_view(camera)
                    for camera in registry.cameras
                    if camera.circuit == "nvr"
                ],
                "archive": archive_depth_view(current_archive_depth()),
                "stream_version": uuid4().hex,
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
        for view in ("face", "side"):
            poster_path(settings.data_dir, event.id, view).unlink(missing_ok=True)

        session.delete(event)
        session.commit()

    @app.post("/api/training/macbook/export", status_code=202)
    def start_macbook_export(scope: str = "macbook") -> dict[str, object]:
        """Начать сборку набора. Ответ приходит сразу, работа идёт фоном.

        Раньше архив собирался прямо здесь. На живом наборе это разваливалось:
        11 августа узел резал кадры больше часа, веб-процесс висел на 145%
        процессора, клиент отвалился по таймауту — и обучение пришлось вести на
        наборе недельной давности. HTTP не умеет ждать час.
        """
        if scope not in {"macbook", "all"}:
            raise HTTPException(422, "scope должен быть macbook или all")
        runs_root = settings.data_dir / "training" / "exports"
        existing = sorted(runs_root.glob("*/status.json"))
        if existing:
            latest = _read_status(existing[-1]) or {}
            if (
                latest.get("state") in ACTIVE_TRAINING_STATES
                and not _expire_stale_training(existing[-1], latest)
            ):
                raise HTTPException(409, "Набор уже собирается")
        run_id = f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
        output_dir = runs_root / run_id
        status_path = output_dir / "status.json"
        write_status(status_path, state="starting", message="Запуск отдельного процесса")
        command = [
            sys.executable,
            "-m",
            "training.export_job",
            str(settings.resolved_database_path.resolve()),
            str(settings.data_dir.resolve()),
            settings.ffmpeg_path,
            str(output_dir.resolve()),
            "--detector-model",
            str(settings.detector_model_path.resolve()),
            "--detector-device",
            settings.detector_device,
            "--scope",
            scope,
        ]
        try:
            with (output_dir / "export.log").open("ab") as log:
                process = spawn_detached(
                    command,
                    cwd=Path.cwd(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
        except OSError as exc:
            write_status(status_path, state="failed", message="Не удалось запустить Python")
            raise HTTPException(500, "Не удалось запустить сборку набора") from exc
        write_status(
            status_path,
            state="starting",
            message="Запуск отдельного процесса",
            pid=process.pid,
        )
        return {"run_id": run_id, "pid": process.pid, "state": "starting"}

    def _ready_export() -> dict[str, object] | None:
        """Самый свежий **собранный** набор, чей архив ещё лежит на диске.

        Считается по всем прогонам, а не по последнему. Последний прогон
        отвечает на вопрос «что происходит сейчас», и это разные вопросы:
        13 августа рядом с законченной сборкой шла вторая, и по последнему
        прогону выходило, что готового набора нет вовсе.
        """
        runs_root = settings.data_dir / "training" / "exports"
        best: dict[str, object] | None = None
        for path in sorted(runs_root.glob("*/status.json")):
            payload = _read_status(path)
            if payload is None or payload.get("state") != "completed":
                continue
            archive = payload.get("archive")
            if not isinstance(archive, str) or not Path(archive).is_file():
                # Архивы подчищаются: остаются три последних. Отчёт о
                # собранном наборе переживает сам набор.
                continue
            best = payload
        return best

    @app.get("/api/training/macbook/export/status")
    def macbook_export_status() -> dict[str, object]:
        latest = _latest_status(settings.data_dir / "training" / "exports")
        ready = _ready_export()
        answer: dict[str, object] = (
            dict(latest) if latest is not None
            else {"state": "idle", "message": "Набор ещё не собирался"}
        )
        # Готовый набор — отдельное поле, а не подмена состояния: «идёт сборка»
        # и «набор уже есть» бывают правдой одновременно.
        if ready is not None:
            answer["ready"] = {
                "archive": Path(str(ready.get("archive"))).name,
                "finished_at": ready.get("updated_at"),
                "size_mb": ready.get("size_mb"),
            }
        return answer

    @app.get("/api/training/macbook/export")
    def export_for_macbook() -> FileResponse:
        """Отдать собранный архив. Сборкой этот маршрут больше не занимается.

        Пока набор не собран, честнее ответить отказом с внятной причиной, чем
        начать часовую работу в ответ на запрос файла.
        """
        ready = _ready_export()
        if ready is None:
            latest = _latest_status(settings.data_dir / "training" / "exports") or {}
            if latest.get("state") in ACTIVE_TRAINING_STATES:
                raise HTTPException(409, "Набор ещё собирается — дождитесь окончания")
            raise HTTPException(404, "Набор не собран: сначала запустите сборку")
        # Идущая сборка отдать готовый набор не мешает: она делает следующий, а
        # этот уже лежит целиком.
        archive = Path(str(ready.get("archive")))
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
        # Автономное переобучение учит тех котов, которых узел уже знает, и
        # сдаёт стоящую модель тому же экзамену, что и новую, — для честного
        # сравнения в отчёте.
        for role, path in (
            ("face", settings.identity_face_model_path),
            ("side", settings.identity_side_model_path),
        ):
            if path is not None and path.is_file():
                command.extend([f"--{role}-model", str(path.resolve())])
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
        write_status(
            status_path,
            state="starting",
            message="Запуск отдельного процесса",
            pid=process.pid,
        )
        return {"run_id": run_id, "pid": process.pid, "state": "starting"}

    @app.get("/api/training/local/status")
    def local_training_status() -> dict[str, object]:
        latest = _latest_status(settings.data_dir / "training" / "local")
        if latest is None:
            return {"state": "idle", "message": "Обучение на узле ещё не запускалось"}
        return latest

    @app.post("/api/identity/prototypes/rebuild", status_code=202)
    def rebuild_prototypes(reuse: bool = False, include: str = "") -> dict[str, object]:
        """Пересчитать эталоны котов по свежей разметке — прямо на узле.

        Сеть при этом не обучается: размеченные проходы прогоняются через уже
        стоящую модель, и эталон каждого кота пересчитывается как среднее его
        векторов. Поэтому это минуты процессора, а не ночь, и поэтому новый кот
        заводится без макбука вовсе.

        Работа уходит отдельным процессом по той же причине, что и обучение:
        перезапуск веб-службы не должен её обрывать.
        """
        runs_root = settings.data_dir / "training" / "prototypes"
        existing = sorted(runs_root.glob("*/status.json"))
        if existing:
            latest = _read_status(existing[-1]) or {}
            if (
                latest.get("state") in ACTIVE_TRAINING_STATES
                and not _expire_stale_training(existing[-1], latest)
            ):
                raise HTTPException(409, "Пересчёт эталонов уже выполняется")
        identity_paths = {
            role: path
            for role, path in (
                ("face", settings.identity_face_model_path),
                ("side", settings.identity_side_model_path),
            )
            if path is not None and path.is_file()
        }
        if not identity_paths:
            raise HTTPException(422, "Не установлена ни одна модель опознания")
        run_id = f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
        output_dir = runs_root / run_id
        status_path = output_dir / "status.json"
        write_status(status_path, state="starting", message="Запуск отдельного процесса")
        # Кадры нарезаются заново по умолчанию, и это стоит часа процессора.
        # Дешёвая альтернатива — взять кропы прошлого прогона — молча
        # игнорировала бы разметку, сделанную после них, а обычный порядок
        # работы именно такой: разметил и нажал. Поэтому `reuse=true`
        # запрашивается явно и годится, когда разметка не менялась: например,
        # чтобы пересчитать с другим набором котов.
        previous = _latest_dataset(runs_root) if reuse else None
        command = [
            sys.executable,
            "-m",
            "training.enrolment",
            str(settings.resolved_database_path.resolve()),
            str(settings.data_dir.resolve()),
            settings.ffmpeg_path,
            str(output_dir.resolve()),
            "--detector-model",
            str(settings.detector_model_path.resolve()),
            "--detector-device",
            settings.detector_device,
        ]
        for role, path in identity_paths.items():
            command.extend([f"--{role}-model", str(path.resolve())])
        if previous is not None:
            command.extend(["--dataset", str(previous)])
        if include.strip():
            command.extend(["--include", *include.split(",")])
        try:
            with (output_dir / "enrolment.log").open("ab") as log:
                process = spawn_detached(
                    command,
                    cwd=Path.cwd(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
        except OSError as exc:
            write_status(status_path, state="failed", message="Не удалось запустить Python")
            raise HTTPException(500, "Не удалось запустить пересчёт эталонов") from exc
        # Теперь состояние знает, за чьей жизнью следить. До первой записи
        # самого процесса это единственный способ отличить «ещё запускается» от
        # «умер на старте».
        write_status(
            status_path,
            state="starting",
            message="Запуск отдельного процесса",
            pid=process.pid,
        )
        return {"run_id": run_id, "pid": process.pid, "state": "starting"}

    @app.get("/api/identity/prototypes/status")
    def prototypes_status() -> dict[str, object]:
        """Состояние пересчёта — то же, что видит страница, включая оборванные.

        Кнопку на странице гасит именно этот ответ, поэтому врать он не имеет
        права: состояние «готовлю кадры» от процесса, которого больше нет, —
        это серая кнопка навсегда.
        """
        latest = _latest_status(settings.data_dir / "training" / "prototypes")
        if latest is None:
            return {"state": "idle", "message": "Эталоны ещё не пересчитывались"}
        return latest

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
            "--quorum",
            str(settings.identity_quorum),
            "--run-kind",
            payload.run_kind,
            "--detector-device",
            settings.evaluation_detector_device,
        ]
        # Замер на настоящем проходе: вырезание кадров стоит 20 секунд
        # процессорного времени на камеру, инференс по ним — полторы. Значит
        # оценку ускоряет не модель, а видеоядро.
        if settings.recorder_hwaccel == "vaapi":
            command.extend(["--vaapi-device", str(settings.recorder_hwaccel_device)])
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

    def installed_cat_names() -> set[str] | None:
        return identity_labels(
            (settings.identity_face_model_path, settings.identity_side_model_path)
        )

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
                "online": build_online_metrics(
                    session,
                    known_names=installed_cat_names(),
                ),
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

    @app.get("/media/{event_id}/{view}/poster")
    def media_poster(
        event_id: int,
        view: str,
        session: Annotated[Session, Depends(get_session)],
    ) -> FileResponse:
        """Кадр-постер прохода: страница проходов вместо метаданных видео.

        До этого страница вешала на каждый проход `<video preload="metadata">`,
        и браузер при открытии тянул метаданные двадцати роликов разом. По
        каналу до админки в 1.7 МБ/с — при том что сами ролики ночного прохода
        весят по полсотни мегабайт — превью появлялись через несколько секунд,
        а нажатый play ждал, пока освободятся соединения. Постер в ~30 КБ
        решает то же самое одним маленьким запросом.

        Кадр берётся на секунде лучшей уверенности детектора — той же, с
        которой открывается ролик, — и кэшируется навсегда: запись прохода
        неизменяема.
        """
        if view not in {"face", "side"}:
            raise HTTPException(404, "Неизвестная камера")
        event = session.get(Event, event_id)
        media_path = (
            event.media_path if event and view == "face" else event.side_media_path if event else None
        )
        if not event or not media_path:
            raise HTTPException(404, "Видео не найдено")
        root = settings.data_dir.resolve()
        source = (root / media_path).resolve()
        if root not in source.parents or not source.is_file():
            raise HTTPException(404, "Видео не найдено")
        poster = poster_path(settings.data_dir, event.id, view)
        if not poster.is_file():
            extracted = extract_frame(
                settings.ffmpeg_path,
                source,
                poster,
                at_seconds=event.preview_offset_seconds or 1.0,
                max_width=POSTER_WIDTH,
            )
            if not extracted:
                raise HTTPException(404, "Кадр не вынулся")
        return FileResponse(
            poster,
            media_type="image/jpeg",
            # Запись неизменяема, значит и кадр из неё вечен: браузер больше не
            # должен спрашивать про него никогда.
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
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
