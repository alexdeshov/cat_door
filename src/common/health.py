"""Эталон рабочей конфигурации узла и сверка факта с ним.

Числа здесь — не пожелания, а **измеренные** значения принятой конфигурации от
2026-08-09 (приёмка прошла 24 проверки из 24). Каждое сопровождается причиной:
эталон без причины через месяц читается как суеверие, и первый же, кто увидит
расхождение, не сможет решить, чинить факт или править эталон.

Модуль сознательно ничего не читает сам. Он получает уже собранные показания и
сводит их с эталоном — поэтому одна и та же сверка работает и на странице
админки, и в скрипте приёмки, и в тестах без узла.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

Status = Literal["ok", "warn", "fail", "unknown"]

# --- Эталон -----------------------------------------------------------------

# Детектор котов работает на видеоядре с 2026-08-08: 13.5 мс на кадр против
# 18.5 на процессоре, и это освободило запас, ушедший в частоту опроса.
EXPECTED_DETECTOR_DEVICE = "GPU"
EXPECTED_DETECTOR_MODEL = "yolo26n-416.onnx"

# Декодирование медиадвижком: ffmpeg дверцы 59.8% ядра → 10.0% (замер 08-08).
EXPECTED_HWACCEL = "vaapi"

# Пороги подняты до 0.65 при переходе на yolo26n-416 (2026-08-10): её
# калибровка увереннее, и на 0.65 и пропусков, и ложных меньше, чем у прежней
# пары 0.5 + yolo11n. До того пороги сравнялись 2026-08-09: дверцу подняли,
# оставленная у двери посылка стабильно проходила как кот.
EXPECTED_THRESHOLD = 0.65

# Частоты подняты вдвое после переезда детектора на видеоядро. Дверца упирается
# в саму камеру: настроено 12, фактический потолок около 11.5.
EXPECTED_SAMPLE_FPS = {"face": 12.0, "side": 6.0}
MIN_SAMPLE_FPS = {"face": 11.0, "side": 5.5}

# Детектор Frigate переехал на видеоядро 2026-08-09: 8.81 мс → 4.78 и минус
# пять ядер. Порог с запасом: на процессоре под нагрузкой было 11 мс.
EXPECTED_GATE_DETECTOR_DEVICE = "GPU"
MAX_GATE_INFERENCE_MS = 12.0

# Пропуск кадров — признак, что контур не успевает. До переезда детектора
# Frigate на видеоядро камера калитки теряла 0.3 кадра в секунду.
#
# Порог долевой, а не абсолютный, и это куплено ложной тревогой. Абсолютные
# 0.1 к/с были настроены на камеру в 5 к/с; когда калитка 2026-08-26 поехала на
# 10 к/с, тот же порог начал ловить единичный потерянный кадр: Frigate округляет
# частоты до одного знака, поэтому 0.1 — наименьшее ненулевое показание вообще,
# и проверка означала «не потеряно ни кадра». Дважды за день короткая заминка
# стоила одного-двух кадров из десяти — телефон звонил, а камера при этом
# отдавала детектору вдвое больше кадров, чем накануне.
#
# Доля этим не обманывается: 0.3 к/с на пяти кадрах (историческая беда) — это
# 6% и тревога, 0.2 на десяти — 2% и тишина.
SKIPPED_WARN_SHARE = 0.05
SKIPPED_FAIL_SHARE = 0.20
# Камера, чья частота неизвестна, меряется по прежней пятёрке: занизить
# знаменатель безопаснее, чем промолчать.
ASSUMED_CAMERA_FPS = 5.0

# Двухканальная память с 2026-08-09. Признак живого второго канала виден
# невооружённым глазом: OpenVINO опознаёт видеоядро как Iris Xe, а при одном
# канале то же ядро зовётся UHD — правило Intel про 128-битную шину.
EXPECTED_GPU_NAME_MARKER = "Iris"
EXPECTED_MEMORY_TOTAL_GB = 31.0

# Пороги здоровья железа. Температура пакета на узле в работе — 50–58 °C;
# 85 °C у Raptor Lake это начало тротлинга.
WARN_CPU_TEMPERATURE_C = 80.0
FAIL_CPU_TEMPERATURE_C = 90.0
WARN_DISK_TEMPERATURE_C = 60.0
FAIL_DISK_TEMPERATURE_C = 70.0
# Нагрузка на ядро: единица означает, что очередь на процессор сравнялась с
# числом ядер. Полтора — узел уже не успевает.
WARN_LOAD_PER_CORE = 1.0
FAIL_LOAD_PER_CORE = 1.5
WARN_MEMORY_USED_PERCENT = 85.0
FAIL_MEMORY_USED_PERCENT = 95.0
WARN_DISK_USED_PERCENT = 85.0
FAIL_DISK_USED_PERCENT = 95.0
# Тома, которые сами распоряжаются своим местом. Frigate не «заполняет диск», а
# **держит его полным**: пишет, пока есть место, и стирает самое старое, когда
# места не осталось. Полнота там — рабочее состояние, а не беда.
#
# Проверено на узле 13 августа: при 97% занятости 6 августа удалено целиком,
# 5 августа стёрто до 150 МБ, 7 августа съедено со 130 до 23 ГБ. Уборка идёт.
#
# Мерить такой том общей меркой нельзя: 85% он перешагнёт в первый же день и
# будет светиться жёлтым вечно. Постоянное предупреждение хуже, чем никакого, —
# оно приучает не смотреть на страницу состояния.
#
# Смотреть надо не на занятость, а на то, **успевает ли уборка**: пока остаётся
# запас свободного, всё идёт как задумано.
SELF_MANAGED_DISK_MARKERS = ("frigate",)
# Копия делается раз в сутки в 05:35. Полутора суток хватает, чтобы не поднимать
# тревогу из-за узла, выключенного на пару часов утром.
BACKUP_STALE_AFTER_HOURS = 36.0
# Сколько места нужно копии на архивном диске. В процентах мерить нельзя: том
# заполнен на 97% всегда, потому что Frigate держит его полным.
BACKUP_WARN_FREE_GB = 15.0
BACKUP_FAIL_FREE_GB = 5.0
# Часовая запись четырёх камер — около 7 ГБ. Ниже этого уборка уже не успевает
# за записью, и вот тогда стоит сказать человеку.
WARN_SELF_MANAGED_FREE_GB = 5.0
FAIL_SELF_MANAGED_FREE_GB = 2.0


@dataclass(frozen=True)
class Check:
    """Одна строка сверки: что ждали, что получили, чем это грозит."""

    key: str
    title: str
    expected: str
    actual: str
    status: Status
    hint: str = ""


@dataclass(frozen=True)
class HealthGroup:
    title: str
    checks: list[Check] = field(default_factory=list)

    @property
    def status(self) -> Status:
        for level in ("fail", "warn", "unknown"):
            if any(check.status == level for check in self.checks):
                return level  # type: ignore[return-value]
        return "ok"


def _compare(value: Any, expected: Any) -> Status:
    if value is None:
        return "unknown"
    return "ok" if value == expected else "fail"


def _threshold(
    value: float | None,
    warn: float,
    fail: float,
    *,
    higher_is_worse: bool = True,
) -> Status:
    if value is None:
        return "unknown"
    if higher_is_worse:
        if value >= fail:
            return "fail"
        return "warn" if value >= warn else "ok"
    if value <= fail:
        return "fail"
    return "warn" if value <= warn else "ok"


def _text(value: Any, suffix: str = "") -> str:
    return "—" if value is None else f"{value}{suffix}"


def detector_checks(status: dict[str, Any], config: dict[str, Any]) -> HealthGroup:
    """Кошачий контур: чем считает, по какой модели и на каких порогах."""
    device = status.get("detector_device_actual")
    checks = [
        Check(
            "detector_device",
            "Устройство детектора",
            EXPECTED_DETECTOR_DEVICE,
            _text(device),
            "unknown"
            if device is None
            else ("ok" if str(device).startswith(EXPECTED_DETECTOR_DEVICE) else "fail"),
            "На процессоре кадр стоит 18.5 мс вместо 13.5 — частоту опроса придётся снижать.",
        ),
        Check(
            "detector_model",
            "Модель",
            EXPECTED_DETECTOR_MODEL,
            _text(status.get("detector_model")),
            _compare(status.get("detector_model"), EXPECTED_DETECTOR_MODEL),
            "Другая модель — другие пороги; сравнивать метрики с прежними нельзя.",
        ),
        Check(
            "hwaccel",
            "Декодирование",
            EXPECTED_HWACCEL,
            _text(status.get("hwaccel_actual")),
            _compare(status.get("hwaccel_actual"), EXPECTED_HWACCEL),
            "Программный откат стоит почти три четверти ядра на камеру.",
        ),
    ]
    for role, title in (("face", "дверца"), ("side", "подход")):
        camera = config.get(role) or {}
        threshold = camera.get("confidence_threshold")
        checks.append(
            Check(
                f"threshold_{role}",
                f"Порог уверенности, {title}",
                str(EXPECTED_THRESHOLD),
                _text(threshold),
                _compare(threshold, EXPECTED_THRESHOLD),
                "Ниже — посылки и тени пойдут как коты; выше — начнутся пропуски.",
            )
        )
        fps = camera.get("sample_fps")
        checks.append(
            Check(
                f"sample_fps_{role}",
                f"Частота опроса, {title}",
                str(EXPECTED_SAMPLE_FPS[role]),
                _text(fps),
                _compare(fps, EXPECTED_SAMPLE_FPS[role]),
                "Настройка; фактический темп — строкой ниже.",
            )
        )
        measured = (status.get("frames_per_second") or {}).get(role)
        checks.append(
            Check(
                f"rate_{role}",
                f"Фактический темп, {title}",
                f"не ниже {MIN_SAMPLE_FPS[role]}",
                _text(measured, " к/с"),
                _threshold(
                    measured,
                    MIN_SAMPLE_FPS[role],
                    MIN_SAMPLE_FPS[role] * 0.8,
                    higher_is_worse=False,
                ),
                "Настройка может говорить одно, а конвейер делать другое — это уже случалось.",
            )
        )
    return HealthGroup("Кошачий контур", checks)


def gate_checks(stats: dict[str, Any] | None, configured_device: str | None) -> HealthGroup:
    """Контур калитки: где считает детектор Frigate и успевает ли он."""
    checks = [
        Check(
            "gate_detector_device",
            "Устройство детектора Frigate",
            EXPECTED_GATE_DETECTOR_DEVICE,
            _text(configured_device),
            _compare(configured_device, EXPECTED_GATE_DETECTOR_DEVICE),
            "На процессоре OpenVINO занимал почти пять ядер: 524% против 39.5%.",
        )
    ]
    if stats is None:
        checks.append(
            Check(
                "gate_reachable",
                "Frigate отвечает",
                "да",
                "нет",
                "fail",
                "Контур калитки не опрашивается — служба стоит или порт закрыт.",
            )
        )
        return HealthGroup("Контур калитки", checks)
    for name, detector in (stats.get("detectors") or {}).items():
        speed = detector.get("inference_speed")
        checks.append(
            Check(
                f"gate_inference_{name}",
                f"Инференс, {name}",
                f"до {MAX_GATE_INFERENCE_MS} мс",
                _text(speed, " мс"),
                _threshold(speed, MAX_GATE_INFERENCE_MS * 0.75, MAX_GATE_INFERENCE_MS),
                "Рост означает, что видеоядро перегружено и очередь общая с котами.",
            )
        )
    for camera, values in (stats.get("cameras") or {}).items():
        fps = values.get("camera_fps")
        checks.append(
            Check(
                f"gate_alive_{camera}",
                f"Кадры идут, {camera}",
                "да",
                _text(fps, " к/с"),
                # Ноль кадров — камера мертва, и это провал, а не «медленно».
                # 2026-08-21 обрыв питания убил test_cam, а проверка пропуска
                # кадров честно показывала «0 пропущено — ok»: пропускать
                # нечего, когда кадров нет вовсе.
                "unknown" if fps is None else ("ok" if fps >= 1.0 else "fail"),
                "Ноль — камера не отдаёт поток: питание, сеть или сама камера.",
            )
        )
        skipped = values.get("skipped_fps")
        # Знаменатель — фактическая частота этой камеры, а не настройка: так
        # порог сам едет за любой сменой частоты и не требует правки здесь.
        # Мёртвая камера в знаменатель не годится — за неё отвечает проверка
        # выше, а здесь она дала бы деление на ноль и вечную аварию.
        rate = fps if fps is not None and fps >= 1.0 else ASSUMED_CAMERA_FPS
        warn = round(rate * SKIPPED_WARN_SHARE, 2)
        checks.append(
            Check(
                f"gate_skipped_{camera}",
                f"Пропуск кадров, {camera}",
                f"до {warn} к/с",
                _text(skipped, " к/с"),
                _threshold(skipped, warn, round(rate * SKIPPED_FAIL_SHARE, 2)),
                "Камера не успевает: часть кадров детектор не увидит вовсе. "
                "Единичный кадр — не беда, тревога с потери двадцатой части.",
            )
        )
    return HealthGroup("Контур калитки", checks)


# Свежесть файла здоровья службы калитки: опрос идёт раз в секунду-две, и
# полминуты тишины означают, что служба стоит или зациклилась.
GATE_HEALTH_FRESH_SECONDS = 30.0


def gate_service_checks(
    payload: dict[str, Any] | None, *, now: float | None = None
) -> HealthGroup:
    """Служба калитки изнутри: слушается ли створка.

    Появилось после 2026-08-21: обрыв питания перезагрузил узел, служба
    поднялась раньше брокера и три часа жила без створки — правило стороны
    стояло разоружённым, и снаружи это не было видно ничем.
    """
    moment = time.time() if now is None else now
    if payload is None:
        return HealthGroup(
            "Служба калитки",
            [
                Check(
                    "gate_service_health",
                    "Файл здоровья службы",
                    "есть",
                    "нет",
                    "unknown",
                    "Служба ещё не писала здоровье — стоит или старой версии.",
                )
            ],
        )
    updated = payload.get("updated_at")
    age = moment - updated if isinstance(updated, (int, float)) else None
    checks = [
        Check(
            "gate_service_fresh",
            "Служба на связи",
            f"обновлялось < {GATE_HEALTH_FRESH_SECONDS:.0f} с назад",
            _text(age, " с") if age is not None else "нет отметки",
            "unknown"
            if age is None
            else ("ok" if 0 <= age <= GATE_HEALTH_FRESH_SECONDS else "fail"),
            "Старый файл — служба калитки стоит, решения не принимаются.",
        ),
        Check(
            "gate_broker",
            "Створка слушается (брокер)",
            "подключено",
            "подключено" if payload.get("broker_connected") else "нет",
            "ok" if payload.get("broker_connected") else "fail",
            "Без брокера правило стороны разоружено: слепые импульсы возможны.",
        ),
    ]
    return HealthGroup("Служба калитки", checks)


def hardware_checks(metrics: Any, gpu_name: str | None) -> HealthGroup:
    """Железо: память, видеоядро, температуры, нагрузка."""
    checks = [
        Check(
            "gpu_name",
            "Видеоядро",
            f"…{EXPECTED_GPU_NAME_MARKER}…",
            _text(gpu_name),
            "unknown"
            if not gpu_name
            else ("ok" if EXPECTED_GPU_NAME_MARKER in gpu_name else "fail"),
            "Имя UHD вместо Iris Xe означает одноканальную память: полоса вдвое ниже.",
        ),
        Check(
            "memory_total",
            "Память",
            f"не меньше {EXPECTED_MEMORY_TOTAL_GB} ГБ",
            _text(metrics.memory_total_gb, " ГБ"),
            _threshold(
                metrics.memory_total_gb,
                EXPECTED_MEMORY_TOTAL_GB,
                EXPECTED_MEMORY_TOTAL_GB * 0.6,
                higher_is_worse=False,
            ),
            "Меньше — вторая планка отвалилась, и полоса памяти упала вдвое.",
        ),
        Check(
            "memory_used",
            "Память занята",
            f"до {WARN_MEMORY_USED_PERCENT}%",
            _text(metrics.memory_used_percent, "%"),
            _threshold(
                metrics.memory_used_percent,
                WARN_MEMORY_USED_PERCENT,
                FAIL_MEMORY_USED_PERCENT,
            ),
        ),
        Check(
            "load",
            "Нагрузка на ядро",
            f"до {WARN_LOAD_PER_CORE}",
            _text(metrics.load_per_core),
            _threshold(metrics.load_per_core, WARN_LOAD_PER_CORE, FAIL_LOAD_PER_CORE),
            "Больше единицы — очередь на процессор длиннее, чем ядер.",
        ),
        Check(
            "cpu_temperature",
            "Температура процессора",
            f"до {WARN_CPU_TEMPERATURE_C} °C",
            _text(metrics.cpu_temperature_c, " °C"),
            _threshold(
                metrics.cpu_temperature_c,
                WARN_CPU_TEMPERATURE_C,
                FAIL_CPU_TEMPERATURE_C,
            ),
            "В работе узел держит 50–58 °C.",
        ),
        Check(
            "disk_temperature",
            "Температура диска",
            f"до {WARN_DISK_TEMPERATURE_C} °C",
            _text(metrics.disk_temperature_c, " °C"),
            _threshold(
                metrics.disk_temperature_c,
                WARN_DISK_TEMPERATURE_C,
                FAIL_DISK_TEMPERATURE_C,
            ),
        ),
    ]
    for disk in metrics.disks:
        checks.append(_disk_check(disk))
    return HealthGroup("Железо узла", checks)


def _is_self_managed(path: str) -> bool:
    lowered = path.lower()
    return any(marker in lowered for marker in SELF_MANAGED_DISK_MARKERS)


def _disk_check(disk: Any) -> Check:
    """Строка про один том. Том с собственной уборкой меряется своей меркой."""
    if not _is_self_managed(disk.path):
        return Check(
            f"disk_{disk.path}",
            f"Диск {disk.path}",
            f"занято до {WARN_DISK_USED_PERCENT}%",
            f"{disk.used_percent}% занято, свободно {disk.free_gb} ГБ",
            _threshold(disk.used_percent, WARN_DISK_USED_PERCENT, FAIL_DISK_USED_PERCENT),
        )
    # Свободного тем меньше, чем лучше работает уборка, поэтому пороги
    # перевёрнуты: тревога внизу, а не наверху.
    status: Status = "ok"
    if disk.free_gb <= FAIL_SELF_MANAGED_FREE_GB:
        status = "fail"
    elif disk.free_gb <= WARN_SELF_MANAGED_FREE_GB:
        status = "warn"
    return Check(
        f"disk_{disk.path}",
        f"Диск {disk.path}",
        f"полон — так и задумано, свободно от {WARN_SELF_MANAGED_FREE_GB} ГБ",
        f"{disk.used_percent}% занято, свободно {disk.free_gb} ГБ",
        status,
        "Frigate держит диск полным и стирает самое старое. "
        "Тревога здесь означает, что уборка не успевает за записью.",
    )


def gpu_occupancy_percent(
    recorder_status: dict[str, Any],
    frigate_stats: dict[str, Any] | None,
) -> float | None:
    """Занятость видеоядра — вычисленная, а не измеренная.

    Настоящую утилизацию отдаёт только `intel_gpu_top` через PMU ядра, а он
    требует прав, которых у веб-службы нет, и на узле не установлен. Зато обе
    стороны честно сообщают, сколько кадров в секунду они считают и сколько
    миллисекунд стоит кадр, — а произведение и есть доля занятого потока.

    Число приблизительное: оно не видит ни декодирование, ни накладные расходы
    очереди. Читать его как «сколько ещё влезет», а не как точный процент.
    """
    busy_ms = 0.0
    seen = False
    rates = recorder_status.get("frames_per_second") or {}
    inference = recorder_status.get("inference_ms")
    if isinstance(inference, (int, float)):
        for rate in rates.values():
            if isinstance(rate, (int, float)):
                busy_ms += rate * inference
                seen = True
    if frigate_stats:
        for detector in (frigate_stats.get("detectors") or {}).values():
            speed = detector.get("inference_speed")
            if isinstance(speed, (int, float)):
                busy_ms += float(frigate_stats.get("detection_fps") or 0) * speed
                seen = True
    if not seen:
        return None
    return round(busy_ms / 1000 * 100, 1)


def _as_float(value: object) -> float | None:
    """Число из чужого JSON. `None` — там не число, и врать про него нельзя."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def backup_checks(status: dict[str, Any] | None, *, now: float | None = None) -> HealthGroup:
    """Прошла ли последняя резервная копия и не устарела ли она.

    Копия — единственное, что стоит между случайным `rm` и потерянной за месяц
    разметкой: её нечем восстановить, она сделана руками. При этом ломается она
    молча — таймер отработал, скрипт упал, никто не заметил.

    Отдельная беда этого узла: архивный диск общий с записями Frigate, а тот
    держит его полным по устройству. Место кончится не постепенно, а ровно в
    ночь, когда записей окажется чуть больше обычного, — и «свободно» здесь
    важно не меньше, чем «прошла».
    """
    current = time.time() if now is None else now
    if not status:
        return HealthGroup(
            "Резервная копия",
            [
                Check(
                    "backup_last_run",
                    "Последняя копия",
                    f"не старше {BACKUP_STALE_AFTER_HOURS} ч",
                    "отметок нет",
                    "unknown",
                    "Копия ни разу не отчитывалась. Так выглядит и первый запуск, "
                    "и остановленный таймер `catdoor-backup.timer`.",
                )
            ],
        )
    finished_at = _as_float(status.get("finished_at"))
    ok = bool(status.get("ok"))
    detail = str(status.get("detail") or "")
    age_hours = None if finished_at is None else (current - finished_at) / 3600
    if not ok:
        run_status: Status = "fail"
        actual = f"не прошла: {detail}" if detail else "не прошла"
    elif age_hours is None:
        run_status = "unknown"
        actual = "время последней копии неизвестно"
    elif age_hours > BACKUP_STALE_AFTER_HOURS or age_hours < 0:
        # Отрицательный возраст — часы узла сместились. Свежей такую копию
        # считать нельзя: она может быть и вчерашней, и месячной давности.
        run_status = "warn"
        actual = f"прошла {abs(age_hours):.0f} ч назад"
    else:
        run_status = "ok"
        actual = f"прошла {age_hours:.0f} ч назад: {detail}" if detail else "прошла"
    checks = [
        Check(
            "backup_last_run",
            "Последняя копия",
            f"не старше {BACKUP_STALE_AFTER_HOURS} ч",
            actual,
            run_status,
            "Копию делает `catdoor-backup.timer` в 05:35. Восстановить разметку "
            "нечем: она сделана руками.",
        )
    ]
    free_gb = _as_float(status.get("free_gb"))
    if free_gb is not None:
        # Порог не в процентах: архивный диск заполнен на 97% всегда, потому что
        # Frigate держит его полным. Копии нужны не проценты, а гигабайты.
        space_status: Status = "ok"
        if free_gb <= BACKUP_FAIL_FREE_GB:
            space_status = "fail"
        elif free_gb <= BACKUP_WARN_FREE_GB:
            space_status = "warn"
        checks.append(
            Check(
                "backup_free_space",
                "Место под копию",
                f"не меньше {BACKUP_WARN_FREE_GB} ГБ",
                f"{free_gb:.0f} ГБ на архивном диске",
                space_status,
                "Диск общий с записями Frigate, а тот держит его полным. "
                "Место кончится не постепенно, а в одну ночь.",
            )
        )
    return HealthGroup("Резервная копия", checks)


def build_health(
    *,
    recorder_status: dict[str, Any],
    detection_config: dict[str, Any],
    frigate_stats: dict[str, Any] | None,
    gate_detector_device: str | None,
    metrics: Any,
    gpu_name: str | None,
    backup_status: dict[str, Any] | None = None,
    gate_service: dict[str, Any] | None = None,
    now: float | None = None,
) -> list[HealthGroup]:
    return [
        detector_checks(recorder_status, detection_config),
        gate_checks(frigate_stats, gate_detector_device),
        gate_service_checks(gate_service, now=now),
        hardware_checks(metrics, gpu_name),
        backup_checks(backup_status, now=now),
    ]


def overall_status(groups: list[HealthGroup]) -> Status:
    for level in ("fail", "warn", "unknown"):
        if any(group.status == level for group in groups):
            return level  # type: ignore[return-value]
    return "ok"
