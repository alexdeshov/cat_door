"""Всё, что принимает и пишет узел, обязано быть видно из админки.

Три раза за два дня встретился один и тот же изъян: поле есть в модели и в
API, но не в форме (`decode`, кошачья копия); служба есть, но не в списке
сторожа; юнит есть, но не в репозитории. Каждый раз лечился тестом, который
сверяет список с источником, а не ведётся руками. Этот файл — тот же приём,
применённый ко всей админке разом.

Конвенция, которую он держит: каждый управляющий элемент, чьё значение уезжает
в полезную нагрузку API, несёт `data-field="<путь.к.полю>"`. Вложенные модели —
через точку (`movement.enabled`), элементы списков — по имени поля списка
(`people.phone_ip`). Так покрытие проверяется точным совпадением, а не
догадками по `id`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import get_args, get_origin

from pydantic import BaseModel

import src.web.app as app_module
from src.common.detection_config import DetectionConfig

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "src" / "web" / "templates"

# Полезная нагрузка → шаблоны, где живёт её форма. Один класс может быть
# размазан по нескольким файлам (тумблер контура — partial, режим записи — на
# странице калитки).
FORMS: dict[type[BaseModel], tuple[str, ...]] = {
    app_module.CameraPayload: ("cameras.html",),
    app_module.DiscoveryPayload: ("cameras.html",),
    DetectionConfig: ("cameras.html",),
    app_module.CloudBackupPayload: ("health.html",),
    app_module.GateAccessPayload: ("gate.html",),
    app_module.CatDoorPayload: ("index.html",),
    app_module.CaptureModePayload: ("index.html",),
    app_module.CircuitsPayload: ("_circuit_switch.html", "gate.html"),
}

# Поля без элемента формы — каждое с причиной. Пустая причина не принимается:
# исключение без объяснения через месяц неотличимо от забытого поля.
EXEMPT: dict[str, str] = {
    "CircuitsPayload.active": (
        "вход API «отдать узел одному контуру целиком»; в интерфейсе это же "
        "делается тумблерами контуров и кнопкой exclusive, третья ручка "
        "путала бы"
    ),
    "CircuitsPayload.enabled": (
        "тумблер контура шлёт circuit и enabled одной кнопкой; включённость — "
        "её собственное состояние в data-enabled, отдельной ручки нет"
    ),
    "GateAccessPayload.people.name": (
        "имя не вводится — строка телефона привязана к имени из белого списка "
        "и несёт его в data-name"
    ),
}

# `{{ role }}` в шаблоне детекции разворачивается в обе роли: форма одна на
# две камеры, и проверять её надо по рендеру, а не по исходнику.
ROLE_EXPANSION = ("face", "side")


def _walk(model: type[BaseModel], prefix: str = "") -> list[str]:
    paths: list[str] = []
    for name, field in model.model_fields.items():
        annotation = field.annotation
        origin = get_origin(annotation)
        args = [a for a in get_args(annotation) if a is not type(None)]
        nested: type[BaseModel] | None = None
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            nested = annotation
        elif origin in (list, tuple) and args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
            nested = args[0]
        elif args and any(isinstance(a, type) and issubclass(a, BaseModel) for a in args):
            nested = next(a for a in args if isinstance(a, type) and issubclass(a, BaseModel))
        if nested is not None and nested is not model:
            paths.extend(_walk(nested, f"{prefix}{name}."))
        else:
            paths.append(f"{prefix}{name}")
    return paths


def _controls(template_names: tuple[str, ...]) -> set[str]:
    found: set[str] = set()
    for name in template_names:
        text = (TEMPLATES / name).read_text(encoding="utf-8")
        for raw in re.findall(r'data-field="([^"]+)"', text):
            if "{{ role }}" in raw:
                for role in ROLE_EXPANSION:
                    found.add(raw.replace("{{ role }}", role))
            else:
                found.add(raw)
    return found


def test_every_payload_field_has_a_control() -> None:
    missing: list[str] = []
    for model, templates in FORMS.items():
        controls = _controls(templates)
        for path in _walk(model):
            key = f"{model.__name__}.{path}"
            if key in EXEMPT:
                assert EXEMPT[key].strip(), f"{key}: исключение без причины"
                continue
            # roi у детекции — вложенная модель с флагом «включено»: поле
            # целиком представлено одним чекбоксом и четырьмя числами.
            if path not in controls and not any(c.startswith(path + ".") for c in controls):
                missing.append(key)
    assert missing == [], "поля без элемента формы:\n  " + "\n  ".join(missing)


def test_no_exemption_is_stale() -> None:
    """Исключение для поля, которого больше нет, — мусор, который прячет
    настоящее упущение под тем же именем."""
    known = {f"{m.__name__}.{p}" for m in FORMS for p in _walk(m)}
    stale = sorted(set(EXEMPT) - known)
    assert stale == []


# --- состояния служб ---------------------------------------------------------

# Файлы в runtime/, которые пишут службы. Делятся на три сорта, и админке
# положено показывать только один из них — состояния. Настройки она и так
# редактирует формами, а межпроцессные файлы и кеши — не для глаз.
SETTINGS_FILES = {
    "cameras.json", "capture-mode.json", "cat-door.json", "circuits.json",
    "cloud-backup.json", "detection-settings.json", "gate-access.json",
}
INTERNAL_FILES = {
    "gate-open-request.json": "просьба об импульсе: живёт секунды, читает только gate-access",
    "gate-open-result.json": "ответ на просьбу: возвращается вызвавшему через API открытия",
    "gate-open-alive": "пульс службы калитки; наружу показан через gate-health.json",
    "ring": "кольцевой буфер сегментов",
    "previews": "кеш кадров превью",
    "posters": "кеш постеров проходов",
    "notifications": "троттлинг уведомлений recorder — внутренняя память",
    "gate-passes": "снимки проходов; каждый отдаётся своей карточкой",
}


def _runtime_files_written_by_services() -> dict[str, Path]:
    """Имя файла → модуль, который его определяет (не веб-служба)."""
    found: dict[str, Path] = {}
    for module in (ROOT / "src").rglob("*.py"):
        if module.parts[-3:-1] == ("src", "web"):
            continue
        text = module.read_text(encoding="utf-8")
        for name in re.findall(r'"runtime"\s*/\s*"([A-Za-z0-9._-]+)"', text):
            found.setdefault(name, module)
    return found


def test_every_service_state_is_shown_in_admin() -> None:
    """Состояние, которое никто не показывает, всё равно что не пишется.

    Кошачья облачная копия сутки писала своё состояние в файл, который не
    читала ни одна страница: очередь, темп, отказы — всё было, и ничего не
    было видно. Проверка сверяет источник (кто пишет) с потребителем (кто
    читает), а не список, составленный по памяти.
    """
    admin = (ROOT / "src" / "web" / "app.py").read_text(encoding="utf-8")
    health = (ROOT / "src" / "common" / "health.py").read_text(encoding="utf-8")
    readers = admin + health
    unseen: list[str] = []
    for name, module in _runtime_files_written_by_services().items():
        if name in SETTINGS_FILES or name in INTERNAL_FILES:
            continue
        # Помощник пути определён в том же модуле, где и литерал имени; админка
        # обязана звать помощника (или, на худой конец, писать имя сама).
        source = module.read_text(encoding="utf-8").splitlines()
        helpers: list[str] = []
        for index, line in enumerate(source):
            if f'"{name}"' not in line:
                continue
            for back in range(index, -1, -1):
                found = re.match(r"\s*def (\w+)\(", source[back])
                if found:
                    helpers.append(found.group(1))
                    break
        if name in readers or any(re.search(rf"\b{h}\b", readers) for h in helpers):
            continue
        unseen.append(f"{name} (пишет {module.relative_to(ROOT)})")
    assert unseen == [], "состояния, которых админка не читает:\n  " + "\n  ".join(unseen)


def test_internal_files_have_reasons() -> None:
    assert all(reason.strip() for reason in INTERNAL_FILES.values())
