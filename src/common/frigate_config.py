"""Конфигурация Frigate, собранная из инвентаря камер.

Frigate обслуживает два контура узла — калитку и регистратор, — и разница
между ними только в том, что делают с камерой: у калитки лица, у регистратора
запись по детекции. Движок один, поэтому и конфигурация одна.

Раньше она правилась руками в `gate_access`, и перевод камеры в другой контур
на вкладке «Камеры» не менял ровным счётом ничего: инвентарь и Frigate жили
каждый своей жизнью. Здесь инвентарь становится источником правды.

Секреты остаются отдельно. Frigate подставляет переменные с префиксом
`FRIGATE_` по имени в фигурных скобках, поэтому в `config.yml` попадают только
ссылки, а адреса и пароли — в `frigate.env` с правами `root:root` 0600. Так
конфигурацию можно читать и сравнивать, не рассыпая пароли камер.

Файлы пишет примиритель от root: у веб-службы привилегий нет и не должно быть.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from src.common.camera_registry import CameraRecord, CameraRegistry
from src.common.circuits import FRIGATE_CIRCUITS, CircuitName, NodeCircuits

# Frigate требует имён из латиницы, цифр и подчёркиваний: идентификатор камеры
# вида `nobelic-4mp` он не примет. Дефис заменяется, а не отбрасывается, чтобы
# из двух разных камер не получилось одно имя.
_UNSAFE_NAME = re.compile(r"[^a-z0-9_]+")

# Что ищут на камерах контура. Калитку интересует один вопрос — кто пришёл;
# регистратору участка нужны ещё машины: подъехавшая машина и есть событие.
# Каждый лишний класс — лишняя работа на двух ядрах, поэтому список короткий.
CIRCUIT_OBJECTS: dict[CircuitName, tuple[str, ...]] = {
    "gate": ("person",),
    "nvr": ("person", "car"),
}

# Пять кадров в секунду хватает, чтобы заметить человека: он находится в кадре
# секундами, а не долями секунды, и движется медленнее кота.
DETECT_FPS = 5

# Разрешение детекции не задаётся: Frigate берёт его из самого потока.
#
# Раньше здесь стояли 640×480 для всех камер сразу, и это стоило дороже, чем
# выглядело. Во-первых, пропорции: субпоток 16:9 втискивался в 4:3, и картинка
# детекции была сплющена. Во-вторых — и это важнее — Frigate ищет лица внутри
# найденного человека именно в кадре детекции. Лицо ростом 95×112 точек в
# записи 1920×1080 превращалось в 32×50 после сжатия до 640×480, и детектор лиц
# такие уже не находит: 5 августа владелец прошёл перед камерой с хорошо
# различимым лицом, а распознавание не запустилось ни разу.
#
# Отсюда требование к субпотоку: он и есть разрешение детекции, поэтому держать
# его в пределах 720p (см. docs/cameras.md). Камера без субпотока будет
# считаться по основному потоку — дорого, но честно и заметно.


def frigate_camera_name(camera_id: str) -> str:
    return _UNSAFE_NAME.sub("_", camera_id.lower()).strip("_") or "camera"


def frigate_camera_key(camera_id: str) -> str:
    """Имя камеры так, как его можно положить ключом в YAML.

    Камера, названная цифрой, ложится в конфигурацию ключом `3:` — а YAML
    считает такой ключ числом, тогда как Frigate ждёт строку. Проверка у него
    строгая, и отвергается **вся** конфигурация: 2026-08-08 узел поднял Frigate
    в safe mode вообще без камер, положив разом и регистратор, и калитку.

    Цена ошибки несоразмерна причине — человек ввёл цифру в поле имени, — и
    чинится она одними кавычками.
    """
    name = frigate_camera_name(camera_id)
    return f'"{name}"' if not name[0].isalpha() else name


def _env_prefix(camera: CameraRecord) -> str:
    return f"FRIGATE_{frigate_camera_name(camera.id).upper()}"


def frigate_cameras(registry: CameraRegistry, circuits: NodeCircuits) -> list[CameraRecord]:
    """Камеры, которые Frigate должен взять прямо сейчас.

    Берутся только камеры действующих контуров: выключенный регистратор не
    должен отнимать процессор у калитки, ради которой узел и включили.
    """
    active = {name for name in FRIGATE_CIRCUITS if circuits.is_active(name)}
    return [
        camera
        for camera in registry.cameras
        if camera.circuit in active and camera.enabled and camera.host
    ]


# Сколько суток хранить записи в каждом режиме: (непрерывно, по движению).
#
# Превью в интерфейсе Frigate живут по тем же правилам, что и записи: превью за
# час, который не перекрывается ни одной сохранённой записью, уборщик сносит
# вместе со строкой в базе. При обоих нулях — режим «только события» — это
# значит, что почти все превью исчезают через час после создания, и на странице
# разбора вместо кадра появляется «Предпросмотр не найден».
#
# Отсюда режим «по движению»: он держит сегменты с движением неделю, и превью
# за эти часы выживают. Пустой ночной двор при этом места не занимает.
RECORDING_RETENTION: dict[str, tuple[int, int]] = {
    "events": (0, 0),
    "motion": (0, 7),
    "continuous": (1, 0),
}


def _retention(mode: str) -> tuple[int, int]:
    return RECORDING_RETENTION.get(mode, RECORDING_RETENTION["events"])


def _record_block(indent: str, continuous_days: int, motion_days: int) -> list[str]:
    return [
        f"{indent}record:",
        f"{indent}  enabled: true",
        f"{indent}  continuous:",
        f"{indent}    days: {continuous_days}",
        f"{indent}  motion:",
        f"{indent}    days: {motion_days}",
    ]


# Строки `hwaccel_args` по пути декодирования; `auto` не пишет ничего и
# оставляет решение автоопределению Frigate.
DECODE_ARGS = {
    "qsv": ["      hwaccel_args: preset-intel-qsv-h264"],
    "software": ['      hwaccel_args: ""'],
}


def _camera_block(
    camera: CameraRecord,
    continuous_days: int,
    motion_days: int,
    preserved: list[str] | None = None,
) -> str:
    prefix = _env_prefix(camera)
    name = frigate_camera_key(camera.id)
    address = f"{{{prefix}_USER}}:{{{prefix}_PASSWORD}}@{{{prefix}_HOST}}:{camera.port}"
    detect_path = camera.detect_stream_path or camera.stream_path
    objects = CIRCUIT_OBJECTS.get(camera.circuit or "nvr", ("person",))
    tracked = "\n".join(f"        - {item}" for item in objects)
    separate_detect = bool(camera.detect_stream_path) and detect_path != camera.stream_path
    lines = [
        f"  {name}:",
        f"    # {camera.title or camera.id}",
        "    ffmpeg:",
        "      inputs:",
        f"        - path: rtsp://{address}{detect_path}",
        "          roles:",
        "            - detect",
    ]
    if separate_detect:
        # Запись идёт основным потоком копированием: в архив попадает полное
        # качество, процессор при этом не тратится.
        lines += [
            f"        - path: rtsp://{address}{camera.stream_path}",
            "          roles:",
            "            - record",
        ]
    else:
        # Один поток на обе роли: субпоток не задан, и разделять нечего.
        lines += ["            - record"]
    lines += [
        "      input_args: preset-rtsp-generic",
        # Поканальный путь декодирования: `hwaccel_args` наследуется камерой и
        # переопределяется на её уровне (справочник Frigate). Пустая строка —
        # программный путь; пресет QSV — то же видеоядро в обход сломанного на
        # Gen12 пути VA-API.
        *DECODE_ARGS.get(camera.decode, []),
        "      output_args:",
        "        record: preset-record-generic-audio-copy",
        "    detect:",
        "      enabled: true",
        f"      fps: {DETECT_FPS}",
        "    objects:",
        "      track:",
        tracked,
        *_record_block("    ", continuous_days, motion_days),
        # Маски движения, зоны и прочее, поставленное человеком в интерфейсе
        # Frigate. Мы их не сочиняем и не понимаем — просто переносим как есть.
        *(preserved or []),
    ]
    return "\n".join(lines)


# Разделы верхнего уровня, которые собираются из инвентаря. Всё остальное в
# `config.yml` поставлено человеком или самим Frigate, и наше дело — не потерять.
MANAGED_SECTIONS = ("mqtt", "detectors", "model", "record", "snapshots", "cameras")

# Ключи камеры, которые собираем мы. Всё остальное под камерой поставил человек
# в интерфейсе Frigate — маски движения, зоны, пороги чувствительности — и наше
# дело перенести это нетронутым.
#
# Раздел `cameras` числится управляемым целиком, и до 2026-08-09 это означало,
# что нарисованная маска жила до первой правки камеры в админке. Узнать об этом
# можно было только по вернувшимся ложным срабатываниям.
GENERATED_CAMERA_KEYS = ("ffmpeg", "detect", "objects", "record")


def foreign_camera_settings(config_text: str) -> dict[str, list[str]]:
    """Ручные настройки камер из существующего `config.yml`, по именам камер.

    Тот же разбор по отступам, что и у `foreign_sections`, только на два уровня
    глубже: камеры лежат с отступом в два пробела, их ключи — в четыре.

    Возвращаются строки как есть, вместе с отступами и комментариями: наше дело
    перенести чужое, а не понять его. Маска — это список координат многоугольника,
    и разбирать его нам незачем.
    """
    result: dict[str, list[str]] = {}
    in_cameras = False
    camera: str | None = None
    keeping = False
    pending: list[str] = []
    for raw in config_text.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        if indent == 0 and not stripped.startswith("#"):
            in_cameras = stripped.split(":", 1)[0].strip() == "cameras"
            camera, keeping, pending = None, False, []
            continue
        if not in_cameras:
            continue
        if stripped.startswith("#"):
            pending.append(line)
            continue
        if indent == 2 and stripped.endswith(":"):
            # Имя камеры. В конфигурации оно в кавычках, если это число: YAML
            # иначе принял бы `3` за целое, и Frigate забраковал бы файл целиком.
            camera = stripped[:-1].strip().strip('"').strip("'")
            keeping, pending = False, []
            continue
        if camera is None:
            continue
        if indent == 4 and ":" in stripped:
            key = stripped.split(":", 1)[0].strip()
            keeping = key not in GENERATED_CAMERA_KEYS
            if keeping:
                result.setdefault(camera, []).extend([*pending, line])
            pending = []
            continue
        if keeping:
            result.setdefault(camera, []).extend([*pending, line])
        pending = []
    return result


def foreign_sections(config_text: str) -> str:
    """Разделы чужой настройки из существующего `config.yml`.

    Распознавание лиц, номера, классификаторы Frigate включаются в его же
    интерфейсе и попадают в тот же файл, который собираем мы. Раньше первая же
    правка камеры в админке стирала их молча: конфигурация пересобиралась с
    нуля, и человек узнавал о потере, когда переставали приходить имена.

    Разбор построчный, без YAML-библиотеки: нужно ровно одно — куски текста
    между ключами нулевого отступа. Разбирать значения незачем, а лишняя
    зависимость в примирителе, который работает от root, стоит дороже.
    """
    kept: list[str] = []
    block: list[str] = []
    # Комментарий нулевого отступа принадлежит следующему за ним ключу: человек
    # писал его про свой раздел, и оторванный от раздела он становится загадкой.
    pending: list[str] = []
    keeping = False
    for line in config_text.splitlines():
        if line[:1] == "#":
            pending.append(line)
            continue
        if line[:1] not in ("", " ") and ":" in line:
            if keeping:
                kept.extend(block)
            key = line.split(":", 1)[0].strip()
            keeping = key not in MANAGED_SECTIONS
            block = [*pending, line] if keeping else []
            pending = []
            continue
        if keeping:
            block.extend(pending)
            block.append(line)
        pending = []
    if keeping:
        kept.extend(block)
    text = "\n".join(kept).strip("\n")
    return f"{text}\n" if text else ""


def build_config(
    registry: CameraRegistry,
    circuits: NodeCircuits,
    preserved: str = "",
    detector_device: str = "CPU",
    preserved_cameras: dict[str, list[str]] | None = None,
) -> str:
    """Текст `config.yml`. Без камер — валидный конфиг без единой камеры."""
    cameras = frigate_cameras(registry, circuits)
    continuous_days, motion_days = _retention(circuits.gate_recording)
    header = [
        "# Собран автоматически из инвентаря камер CatDoor. Правки руками",
        "# перетрёт примиритель: меняйте камеры на вкладке «Камеры» в админке.",
        "",
        "mqtt:",
        "  # Понадобится на этапе отчётов о перемещениях. Пока брокера нет, и",
        "  # включённый MQTT только сыпал бы ошибками подключения.",
        "  enabled: false",
        "",
        "detectors:",
        "  openvino:",
        "    type: openvino",
        f"    device: {detector_device}",
        "",
        "model:",
        "  width: 300",
        "  height: 300",
        "  input_tensor: nhwc",
        "  input_pixel_format: bgr",
        "  path: /openvino-model/ssdlite_mobilenet_v2.xml",
        "  labelmap_path: /openvino-model/coco_91cl_bkgr.txt",
        "",
        *_record_block("", continuous_days, motion_days),
        "  alerts:",
        "    retain:",
        "      days: 14",
        "      mode: all",
        "  detections:",
        "    retain:",
        "      days: 14",
        "      mode: all",
        "",
        "snapshots:",
        "  enabled: true",
        "  retain:",
        "    default: 14",
        "",
    ]
    tail = [preserved] if preserved else []
    if not cameras:
        header += [
            "# Ни одна камера не назначена действующему контуру Frigate.",
            "cameras: {}",
            "",
        ]
        return "\n".join([*header, *tail])
    kept = preserved_cameras or {}
    blocks = [
        _camera_block(
            camera,
            continuous_days,
            motion_days,
            kept.get(frigate_camera_key(camera.id).strip('"')),
        )
        for camera in cameras
    ]
    return "\n".join([*header, "cameras:", *blocks, "", *tail])


def build_env(cameras: Iterable[CameraRecord], timezone: str = "Europe/Moscow") -> str:
    """Текст `frigate.env`: адреса и пароли, которых нет в конфигурации."""
    lines = [
        "# Собран автоматически из инвентаря камер CatDoor.",
        "# Здесь лежат пароли камер: файл принадлежит root и закрыт от всех.",
        "",
    ]
    for camera in cameras:
        prefix = _env_prefix(camera)
        lines += [
            f"{prefix}_HOST={camera.host}",
            f"{prefix}_USER={camera.username}",
            f"{prefix}_PASSWORD={camera.password}",
        ]
    lines += ["", f"TZ={timezone}", ""]
    return "\n".join(lines)
