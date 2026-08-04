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
DETECT_WIDTH = 640
DETECT_HEIGHT = 480


def frigate_camera_name(camera_id: str) -> str:
    return _UNSAFE_NAME.sub("_", camera_id.lower()).strip("_") or "camera"


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


def _camera_block(camera: CameraRecord, continuous_days: int) -> str:
    prefix = _env_prefix(camera)
    name = frigate_camera_name(camera.id)
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
        "      output_args:",
        "        record: preset-record-generic-audio-copy",
        "    detect:",
        "      enabled: true",
        f"      fps: {DETECT_FPS}",
        f"      width: {DETECT_WIDTH}",
        f"      height: {DETECT_HEIGHT}",
        "    objects:",
        "      track:",
        tracked,
        "    record:",
        "      enabled: true",
        "      continuous:",
        f"        days: {continuous_days}",
    ]
    return "\n".join(lines)


def build_config(registry: CameraRegistry, circuits: NodeCircuits) -> str:
    """Текст `config.yml`. Без камер — валидный конфиг без единой камеры."""
    cameras = frigate_cameras(registry, circuits)
    continuous_days = 1 if circuits.gate_recording == "continuous" else 0
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
        "    device: CPU",
        "",
        "model:",
        "  width: 300",
        "  height: 300",
        "  input_tensor: nhwc",
        "  input_pixel_format: bgr",
        "  path: /openvino-model/ssdlite_mobilenet_v2.xml",
        "  labelmap_path: /openvino-model/coco_91cl_bkgr.txt",
        "",
        "record:",
        "  enabled: true",
        "  continuous:",
        f"    days: {continuous_days}",
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
    if not cameras:
        header += [
            "# Ни одна камера не назначена действующему контуру Frigate.",
            "cameras: {}",
            "",
        ]
        return "\n".join(header)
    blocks = [_camera_block(camera, continuous_days) for camera in cameras]
    return "\n".join([*header, "cameras:", *blocks, ""])


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
