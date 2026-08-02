"""Инвентарь камер, переживающий их физическую перестановку.

Адрес камеры больше не является её именем. Раньше камера жила в `.env` как
единственная строка `rtsp://логин:пароль@адрес/путь`: смена адреса по DHCP
делала запись бесполезной, а пустое значение убирало камеру из админки целиком.
Здесь камера — это запись с собственным идентификатором, которая остаётся на
месте, даже когда её сейчас не видно в сети, а адрес, учётные данные и роль
`face`/`side` — редактируемые поля этой записи.

Роль остаётся отдельным понятием: на неё завязаны детектор, схема событий и
модели идентификации, поэтому перестановка камер местами — это переназначение
роли, а не переименование камеры. Настройки детекции при этом не переезжают:
они описывают сцену (`data/runtime/detection-settings.json`), а не железку.
"""

from __future__ import annotations

import json
import re
from contextlib import suppress
from pathlib import Path
from typing import Literal
from urllib.parse import quote, urlsplit
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from src.common.durable_write import write_text_durably
from src.common.settings import Settings

CameraRole = Literal["face", "side"]

# Роли, на которых держатся детектор, таблицы событий и модели идентификации.
ROLES: tuple[CameraRole, ...] = ("face", "side")

ROLE_TITLES: dict[CameraRole, str] = {
    "face": "Камера идентификации",
    "side": "Камера подхода",
}

# Идентификатор попадает в адрес потока `/stream/<id>.mjpeg`, поэтому в нём
# только то, что переживает сегмент URL без экранирования.
_CAMERA_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_MAC = re.compile(r"^[0-9a-f]{2}(:[0-9a-f]{2}){5}$")
_NOT_ID_CHARACTER = re.compile(r"[^a-z0-9]+")

REGISTRY_VERSION = 1


def slugify_camera_id(value: str) -> str:
    """Собрать из произвольного текста идентификатор, годный для URL."""
    slug = _NOT_ID_CHARACTER.sub("-", value.strip().lower()).strip("-")
    return slug or "camera"


class CameraRecord(BaseModel):
    """Одна камера инвентаря.

    Адрес хранится разобранным на части, а не одной строкой: обнаружение камер
    меняет только `host`, оставляя путь потока и учётные данные нетронутыми,
    а пароль так не попадает в ответы API вместе с остальными полями.
    """

    id: str
    title: str = ""
    host: str = ""
    port: int = Field(default=554, ge=1, le=65535)
    stream_path: str = ""
    username: str = ""
    password: str = ""
    mac: str = ""
    role: CameraRole | None = None
    rotate_180: bool = False
    enabled: bool = True
    note: str = ""

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        if not _CAMERA_ID.match(value):
            raise ValueError(
                "идентификатор камеры может содержать только латиницу в нижнем "
                f"регистре, цифры, дефис и подчёркивание, задано: {value!r}"
            )
        return value

    @field_validator("host", "username", "title", "note", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("mac", mode="before")
    @classmethod
    def _normalize_mac(cls, value: object) -> object:
        if not isinstance(value, str) or not value.strip():
            return ""
        normalized = value.strip().lower().replace("-", ":")
        if not _MAC.match(normalized):
            raise ValueError(f"MAC-адрес должен иметь вид aa:bb:cc:dd:ee:ff, задано: {value!r}")
        return normalized

    @field_validator("stream_path", mode="before")
    @classmethod
    def _normalize_path(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        path = value.strip()
        if not path or path.startswith("/"):
            return path
        return f"/{path}"

    @property
    def display_title(self) -> str:
        if self.title:
            return self.title
        return ROLE_TITLES[self.role] if self.role is not None else self.id

    @property
    def is_addressable(self) -> bool:
        """Есть ли у камеры адрес, по которому вообще можно постучаться."""
        return bool(self.host) and self.enabled

    def rtsp_url(self) -> str:
        """Собрать RTSP-адрес. Пустая строка — камера не настроена.

        Логин и пароль экранируются: в паролях камер регулярно встречаются
        `@`, `/` и `:`, которые иначе разрежут URL не в том месте.
        """
        if not self.host:
            return ""
        credentials = ""
        if self.username:
            credentials = quote(self.username, safe="")
            if self.password:
                credentials = f"{credentials}:{quote(self.password, safe='')}"
            credentials = f"{credentials}@"
        return f"rtsp://{credentials}{self.host}:{self.port}{self.stream_path}"

    def stream_signature(self) -> tuple[str, bool]:
        """То, из-за чего поток приходится пересоздавать."""
        return (self.rtsp_url(), self.rotate_180)


class CameraRegistry(BaseModel):
    version: int = REGISTRY_VERSION
    cameras: list[CameraRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def _resolve_conflicts(self) -> CameraRegistry:
        """Дубликаты не отвергают файл целиком, а обезвреживаются.

        Файл могли поправить руками, и падение на одной строке оставило бы
        систему вовсе без камер — то есть ровно с тем поведением, от которого
        этот модуль избавляется. Побеждает первая запись, остальным конфликтное
        поле обнуляется.
        """
        seen_ids: set[str] = set()
        seen_roles: set[str] = set()
        kept: list[CameraRecord] = []
        for camera in self.cameras:
            if camera.id in seen_ids:
                continue
            seen_ids.add(camera.id)
            if camera.role is not None and camera.role in seen_roles:
                camera = camera.model_copy(update={"role": None})
            if camera.role is not None:
                seen_roles.add(camera.role)
            kept.append(camera)
        self.cameras = kept
        return self

    def get(self, camera_id: str) -> CameraRecord | None:
        return next((camera for camera in self.cameras if camera.id == camera_id), None)

    def by_role(self, role: str) -> CameraRecord | None:
        return next((camera for camera in self.cameras if camera.role == role), None)

    def resolve(self, name: str) -> CameraRecord | None:
        """Найти камеру по имени роли, а если это не роль — по идентификатору.

        Роль имеет приоритет: `/stream/face.mjpeg` должен показывать ту камеру,
        которая сейчас назначена на роль, даже если камера с таким id тоже есть.
        """
        if name in ROLES:
            by_role = self.by_role(name)
            if by_role is not None:
                return by_role
        return self.get(name)

    def role_of(self, camera_id: str) -> CameraRole | None:
        camera = self.get(camera_id)
        return camera.role if camera is not None else None

    def with_camera(self, camera: CameraRecord) -> CameraRegistry:
        """Добавить или заменить запись, освободив роль у прежнего владельца.

        Порядок списка сохраняется: он же порядок карточек в админке, и
        сохранение настроек не должно перетасовывать их под курсором.
        """
        cameras: list[CameraRecord] = []
        replaced = False
        for existing in self.cameras:
            if existing.id == camera.id:
                cameras.append(camera)
                replaced = True
                continue
            if camera.role is not None and existing.role == camera.role:
                existing = existing.model_copy(update={"role": None})
            cameras.append(existing)
        if not replaced:
            cameras.append(camera)
        return CameraRegistry(version=self.version, cameras=cameras)

    def without_camera(self, camera_id: str) -> CameraRegistry:
        return CameraRegistry(
            version=self.version,
            cameras=[camera for camera in self.cameras if camera.id != camera_id],
        )

    def with_swapped_roles(self) -> CameraRegistry:
        """Поменять местами `face` и `side`, ничего больше не трогая."""
        cameras = [
            (
                camera.model_copy(
                    update={"role": "side" if camera.role == "face" else "face"}
                )
                if camera.role in ROLES
                else camera
            )
            for camera in self.cameras
        ]
        return CameraRegistry(version=self.version, cameras=cameras)


def camera_registry_path(data_dir: Path) -> Path:
    return data_dir / "runtime" / "cameras.json"


def camera_from_url(
    camera_id: str,
    rtsp_url: str,
    *,
    role: CameraRole | None = None,
    title: str = "",
    rotate_180: bool = False,
) -> CameraRecord:
    """Разобрать готовый RTSP-адрес в запись инвентаря.

    Нужно и для переноса прежних `CATDOOR_CAMERA_*_RTSP`, и для формы, куда
    адрес удобнее вставить целиком, чем раскладывать руками по полям.
    """
    record = CameraRecord(id=camera_id, role=role, title=title, rotate_180=rotate_180)
    if not rtsp_url.strip():
        return record
    parts = urlsplit(rtsp_url.strip())
    path = parts.path or ""
    if parts.query:
        path = f"{path}?{parts.query}"
    return record.model_copy(
        update={
            "host": parts.hostname or "",
            "port": parts.port or 554,
            "stream_path": path,
            "username": parts.username or "",
            "password": parts.password or "",
        }
    )


def registry_from_settings(settings: Settings) -> CameraRegistry:
    """Начальный инвентарь из переменных окружения.

    Роли заводятся всегда, даже без адреса: ненастроенная роль — это то, что
    администратору и нужно увидеть в списке, чтобы дать ей адрес.
    """
    cameras = [
        camera_from_url(
            role,
            settings.camera_face_rtsp if role == "face" else settings.camera_side_rtsp,
            role=role,
            title=ROLE_TITLES[role],
            rotate_180=(
                settings.camera_face_rotate_180
                if role == "face"
                else settings.camera_side_rotate_180
            ),
        )
        for role in ROLES
    ]
    registry = CameraRegistry(cameras=cameras)
    for extra in settings.cameras:
        existing = registry.get(extra.id)
        parsed = camera_from_url(
            extra.id,
            extra.rtsp,
            role=existing.role if existing is not None else None,
            title=extra.title or (existing.title if existing is not None else ""),
            rotate_180=extra.rotate_180,
        )
        registry = registry.with_camera(parsed)
    return registry


def load_registry(path: Path) -> CameraRegistry | None:
    """Прочитать инвентарь. `None` — файла нет либо он нечитаем."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    try:
        return CameraRegistry.model_validate(payload)
    except ValueError:
        return None


def save_registry(path: Path, registry: CameraRegistry) -> None:
    """Записать инвентарь целиком, только для владельца и до самого диска.

    Здесь лежат пароли камер: потерять этот файл при обрыве питания — значит
    потерять доступ ко всем камерам, поэтому запись идёт через
    `write_text_durably`, а не через обычное переименование.
    """
    write_text_durably(
        path,
        json.dumps(registry.model_dump(), ensure_ascii=False, indent=2),
        mode=0o600,
    )


def ensure_registry(path: Path, settings: Settings) -> CameraRegistry:
    """Инвентарь на диске, при необходимости созданный из `.env`.

    Испорченный файл не переписывается молча: он отодвигается в сторону, чтобы
    администратор мог достать оттуда адрес камеры, но и служба поднялась.
    """
    registry = load_registry(path)
    if registry is not None:
        return registry
    if path.exists():
        with suppress(OSError):
            path.replace(path.with_name(f"{path.name}.invalid-{uuid4().hex[:8]}"))
    registry = registry_from_settings(settings)
    save_registry(path, registry)
    return registry
