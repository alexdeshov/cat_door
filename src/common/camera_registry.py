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
# Контур, которому принадлежит камера. Совпадает с контурами узла из
# `src.common.circuits`, но живёт здесь отдельным типом: инвентарь не должен
# зависеть от состояния узла, чтобы камеру можно было завести до того, как
# её контур вообще развёрнут.
CameraCircuit = Literal["cats", "gate", "nvr"]

# Роли, на которых держатся детектор, таблицы событий и модели идентификации.
ROLES: tuple[CameraRole, ...] = ("face", "side")
CIRCUITS: tuple[CameraCircuit, ...] = ("cats", "gate", "nvr")

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
    # Субпоток для детекции. Декодирование основного потока — самая дорогая
    # статья расхода Frigate, дороже самого распознавания, поэтому детекция
    # смотрит уменьшенную копию, а в архив пишется основной поток без
    # перекодирования. Пусто — детекция идёт по основному, что честно работает
    # и заметно дороже.
    detect_stream_path: str = ""
    # Поток для картинки в админке. Отдельный от детекции, потому что задачи у
    # них разные: детекции нужно разрешение, достаточное чтобы узнать лицо, а
    # превью показывается шириной 960 точек, и всё сверх этого — впустую
    # потраченный процессор. У камеры калитки развод дал 32% ядра против 2%.
    preview_stream_path: str = ""
    username: str = ""
    password: str = ""
    mac: str = ""
    circuit: CameraCircuit | None = None
    role: CameraRole | None = None
    rotate_180: bool = False
    enabled: bool = True
    # Путь декодирования камеры в Frigate. `auto` — автоопределение Frigate
    # (на этом узле VA-API); `qsv` — явный QuickSync; `software` — процессор.
    #
    # Поле появилось из-за «Failed to sync surface»: автоопределённый VA-API на
    # Gen12 (Iris Xe) ломается в известном месте драйвера, ffmpeg детекции
    # падал раз в 1–3 минуты. Это подтверждено и сообществом Frigate
    # (обсуждение №23319: та же пара ошибок, сопровождающий прямо называет
    # автоопределение «догадкой»), и нашим экспериментом 2026-08-09: камера,
    # переведённая с VA-API, перестала падать, оставшиеся продолжили. QSV —
    # рекомендованное там лекарство: то же видеоядро другим путём, ноль падений
    # у автора за 9.5 часа. Программный путь — запасной, треть ядра на камеру.
    decode: str = "auto"
    note: str = ""

    @field_validator("decode")
    @classmethod
    def _check_decode(cls, value: str) -> str:
        if value not in {"auto", "qsv", "software"}:
            raise ValueError(
                f"путь декодирования бывает auto, qsv или software, задано: {value!r}"
            )
        return value

    @model_validator(mode="before")
    @classmethod
    def _legacy_software_decode(cls, values: object) -> object:
        # Однодневное поле `software_decode` (2026-08-09) уже записано в
        # инвентаре узла; молча потерять его значило бы вернуть камеру на
        # сломанный путь при первом же чтении.
        if isinstance(values, dict) and values.pop("software_decode", False):
            values.setdefault("decode", "software")
        return values

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

    @field_validator("circuit", mode="before")
    @classmethod
    def _forget_unknown_circuit(cls, value: object) -> object:
        """Незнакомый контур обнуляется, а не роняет разбор.

        Инвентарь переживает версии кода. Файл, записанный новой версией, читает
        и старая — например, служба, которую забыли перезапустить при
        обновлении. Строгая проверка превращала такой файл в «испорченный», а
        `ensure_registry` отодвигал его и пересоздавал из `.env`: адреса,
        пароли и роли всех камер разом заменялись умолчаниями. Ровно так и
        случилось 4 августа, когда камеру назначили в контур, которого не
        перезапущенный recorder ещё не знал.

        Камера без понятного контура остаётся камерой: её видно в списке, и
        контур назначается заново одним нажатием. Это несравнимо дешевле потери
        всего инвентаря вместе с паролями.
        """
        return value if value is None or value in CIRCUITS else None

    @field_validator("role", mode="before")
    @classmethod
    def _forget_unknown_role(cls, value: object) -> object:
        """То же и для роли: незнакомая роль лучше потерянной камеры."""
        return value if value is None or value in ROLES else None

    @field_validator("stream_path", "detect_stream_path", "preview_stream_path", mode="before")
    @classmethod
    def _normalize_path(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        path = value.strip()
        if not path or path.startswith("/"):
            return path
        return f"/{path}"

    @model_validator(mode="after")
    def _role_implies_cats(self) -> CameraRecord:
        """Роль бывает только у камеры дверцы.

        `face` и `side` — понятия кошачьего контура: на них завязаны детектор,
        таблицы событий и модели по ракурсам. Камера с ролью принадлежит ему по
        определению, поэтому контур подставляется, а не требуется отдельно, —
        иначе список предлагал бы завести противоречивую запись.
        """
        if self.role is not None and self.circuit != "cats":
            # Меняется сам объект, а не копия: при создании через `__init__`
            # pydantic отбрасывает всё, кроме `self`, и подстановка пропала бы.
            self.circuit = "cats"
        return self

    @property
    def display_title(self) -> str:
        if self.title:
            return self.title
        return ROLE_TITLES[self.role] if self.role is not None else self.id

    @property
    def is_addressable(self) -> bool:
        """Есть ли у камеры адрес, по которому вообще можно постучаться."""
        return bool(self.host) and self.enabled

    def _rtsp_url_for(self, path: str) -> str:
        """Собрать RTSP-адрес с заданным путём потока.

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
        return f"rtsp://{credentials}{self.host}:{self.port}{path}"

    def rtsp_url(self) -> str:
        """Основной поток. Пустая строка — камера не настроена."""
        return self._rtsp_url_for(self.stream_path)

    def preview_rtsp_url(self) -> str:
        """Адрес для живой картинки: субпоток, если он задан.

        Превью показывается в браузере шириной 960 точек, и декодировать ради
        него основной поток — самая дорогая ошибка из дешёвых. Замер 5 августа:
        одна открытая карточка камеры 2560×1440 стоила 47% ядра, тот же кадр из
        субпотока 640×360 — около 4%. Разница уходила в никуда: картинка всё
        равно масштабируется вниз.

        Порядок предпочтения — от самого дешёвого к самому дорогому: путь,
        заданный специально для превью, затем поток детекции, затем основной.
        Пустое поле означает «другого потока нет», а не «показывать нечего».
        """
        return self._rtsp_url_for(
            self.preview_stream_path or self.detect_stream_path or self.stream_path
        )


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

    Пересоздание — крайняя мера, а не обычный путь: в файле лежат адреса,
    пароли и роли всех камер, и подменить его умолчаниями значит стереть
    настройку узла целиком. Поэтому сюда попадают только два случая: файла нет
    вовсе и файл не разбирается как JSON.

    Файл, который разобрался, но не прошёл проверку модели, к этим случаям не
    относится. Раньше относился — и 4 августа служба, оставшаяся на старом коде
    после обновления, встретила незнакомое значение контура, объявила инвентарь
    испорченным и заменила его умолчаниями из `.env`. Теперь незнакомые
    значения обнуляются по полю (см. валидаторы `CameraRecord`), а до
    пересоздания дело не доходит.

    Испорченный файл не удаляется, а отодвигается: администратор достанет
    оттуда адреса, а служба всё равно поднимется.
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
