"""Сжатие ролика под предел Телеграма.

Bot API принимает от бота файл не больше 50 МБ. Это предел протокола, а не
настройки: обойти его нечем, и до сих пор кнопка на тяжёлом ролике отвечала
отказом. Отказывала она ровно там, где ролик нужнее всего, — событие калитки
в несколько минут весит сотни мегабайт, и смотреть хочется как раз его.
Теперь узел пережимает ролик сам: хуже качеством, зато целиком и тем же
нажатием.

Битрейт берётся не из таблицы, а делением бюджета на длину: пятьдесят
мегабайт на тридцать секунд и те же пятьдесят на пять минут — это разные
картинки, и постоянное «сожми в 2 Мбит/с» промахивалось бы в обе стороны.
Ширина кадра выбирается под получившийся битрейт: 1080p на 600 кбит/с
разваливается в квадраты, а 854 точки при том же битрейте остаются
смотрибельными — уменьшить кадр выгоднее, чем сохранить его и получить кашу.

Кодирует процессор (`libx264`, `veryfast`). Видеоядро узла умеет кодировать
вдвое быстрее, но доступ к `/dev/dri` служба получает только новыми правами в
юните и группой `render` у пользователя — то есть выкатом `install.sh` и
руками владельца. Здесь это не окупается: ролик уезжает по нажатию, а не
потоком, и минута работы процессора раз в день никому не мешает.

Ролик, которому на секунду достаётся меньше `MIN_TOTAL_KBPS`, не сжимается
вовсе: под этой границей видео перестаёт быть видео, и честнее сказать словами
«смотрите в админке», чем прислать полчаса серой мути.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

logger = logging.getLogger("catdoor.telegram-bot")

# Бюджет берётся с запасом от предела: заданный битрейт ffmpeg держит в
# среднем, а не покадрово, плюс контейнер добавляет заголовки и индекс.
# Промах вверх стоит целой перекодировки, промах вниз — незаметен.
BUDGET_SHARE = 0.90
# Звук в ролике Frigate есть (запись идёт пресетом с копированием звука), и
# терять его незачем: голос у калитки объясняет кадр не хуже картинки. Моно
# 48 кбит/с — столько, чтобы речь оставалась разборчивой.
AUDIO_KBPS = 48
# Ниже этого совокупного битрейта сжимать нечего: получится равномерная муть,
# на которой не видно ни лица, ни кота. Полчаса записи в пятьдесят мегабайт
# укладываются ровно на этой границе.
MIN_TOTAL_KBPS = 220
# Какой ширины кадр имеет смысл при таком битрейте. Порядок — от щедрого к
# скупому; кадр никогда не растягивается, только уменьшается.
WIDTH_LADDER: tuple[tuple[int, int], ...] = (
    (2200, 1920),
    (1100, 1280),
    (550, 854),
    (0, 640),
)
# Запасной путь, когда длину ролика измерить не удалось: постоянное качество
# со ступенями «хуже и мельче». Без длины бюджет не поделить, но остаться без
# видео из-за молчащего ffprobe — хуже, чем прислать лишний раз пережатое.
QUALITY_LADDER: tuple[tuple[int, int], ...] = ((1280, 30), (854, 34), (640, 38))
# Сколько раз пробовать. Первая попытка обычно и последняя: промах
# исправляется пересчётом битрейта по факту, а не слепым повтором.
MAX_ATTEMPTS = 3


class ShrinkFailed(RuntimeError):
    """Сжать не вышло, и человеку надо сказать почему — его словами."""


@dataclass(frozen=True)
class Plan:
    """Чем именно давить: ширина кадра и либо битрейт, либо качество."""

    width: int
    video_kbps: int = 0
    crf: int = 0


def ffprobe_path(ffmpeg_path: str) -> str:
    """Соседний `ffprobe` рядом с тем `ffmpeg`, которым живёт весь узел.

    Настройка на узле одна (`CATDOOR_FFMPEG_PATH`), и вторая, отдельная, для
    `ffprobe` означала бы возможность развести их по разным сборкам — то есть
    ещё один способ сломать узел молча.
    """
    path = Path(ffmpeg_path)
    if path.name.startswith("ffmpeg"):
        return str(path.with_name("ffprobe" + path.name[len("ffmpeg") :]))
    return "ffprobe"


def probe_duration(
    ffmpeg_path: str,
    video: Path,
    *,
    timeout_seconds: float = 20.0,
) -> float | None:
    """Длина ролика в секундах. `None` — измерить не удалось.

    Не исключение: без длины остаётся запасной путь по качеству, и ронять из-за
    этого целую отправку незачем.
    """
    command = [
        ffprobe_path(ffmpeg_path),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video),
    ]
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("Длину ролика %s измерить не вышло: %s", video.name, exc)
        return None
    if result.returncode != 0:
        logger.warning(
            "ffprobe отказал на %s: %s",
            video.name,
            (result.stderr or "").strip() or "без объяснений",
        )
        return None
    try:
        duration = float(json.loads(result.stdout)["format"]["duration"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return duration if duration > 0 else None


def width_for(total_kbps: int) -> int:
    for floor, width in WIDTH_LADDER:
        if total_kbps >= floor:
            return width
    return WIDTH_LADDER[-1][1]


def _too_long(duration_seconds: float, limit_bytes: int) -> ShrinkFailed:
    return ShrinkFailed(
        f"Ролик длиной {duration_seconds / 60:.0f} мин не ужать до "
        f"{limit_bytes // 1024 // 1024} МБ так, чтобы на нём было что-то видно. "
        "Смотрите его в админке."
    )


def first_plan(duration_seconds: float | None, budget_bytes: int, limit_bytes: int) -> Plan:
    """С чего начать. Битрейт — от бюджета на секунду, ширина — от битрейта."""
    if duration_seconds is None:
        width, crf = QUALITY_LADDER[0]
        return Plan(width=width, crf=crf)
    total_kbps = int(budget_bytes * 8 / duration_seconds / 1000)
    if total_kbps < MIN_TOTAL_KBPS:
        raise _too_long(duration_seconds, limit_bytes)
    return Plan(width=width_for(total_kbps), video_kbps=max(total_kbps - AUDIO_KBPS, 1))


def tighter_plan(
    plan: Plan,
    *,
    overshoot: float,
    duration_seconds: float | None,
    limit_bytes: int,
) -> Plan:
    """Следующая попытка. Промах по битрейту правится по факту, а не на глаз.

    `overshoot` — во сколько раз получившийся файл больше предела. Множитель
    даётся с небольшим запасом: попасть в предел со второго раза дешевле, чем
    подобраться к нему с третьего.
    """
    if plan.video_kbps:
        total_kbps = int((plan.video_kbps + AUDIO_KBPS) / overshoot * 0.95)
        if total_kbps < MIN_TOTAL_KBPS:
            raise _too_long(duration_seconds or 0.0, limit_bytes)
        return Plan(width=width_for(total_kbps), video_kbps=max(total_kbps - AUDIO_KBPS, 1))
    for width, crf in QUALITY_LADDER:
        if crf > plan.crf:
            return replace(plan, width=width, crf=crf)
    raise ShrinkFailed(
        "Ролик не ужимается до предела Телеграма даже на самом грубом качестве. "
        "Смотрите его в админке."
    )


def encode_command(ffmpeg_path: str, source: Path, destination: Path, plan: Plan) -> list[str]:
    """Команда одной попытки.

    `min(iw,…)` вместо простой ширины: кадр с террасной камеры уже меньше
    любой нашей ступени, и растягивать его значило бы платить битрейтом за
    придуманные пиксели. `0:a:0?` — звук, если он есть: вопросительный знак
    оставляет команду рабочей на записи прохода, где звука нет вовсе.
    """
    quality = (
        [
            "-b:v",
            f"{plan.video_kbps}k",
            # Потолок и буфер держат пики: без них сцена с движением съедает
            # бюджет целого ролика за десяток секунд.
            "-maxrate",
            f"{int(plan.video_kbps * 1.4)}k",
            "-bufsize",
            f"{plan.video_kbps * 2}k",
        ]
        if plan.video_kbps
        else ["-crf", str(plan.crf)]
    )
    return [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-vf",
        f"scale='min(iw,{plan.width})':-2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        *quality,
        # Телефон отказывается показывать кадр в 10 бит и в 4:2:2 — камеры
        # такого не отдают, но склейка Frigate переживает смену камеры, а
        # ролик обязан открываться на том единственном экране, где его смотрят.
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        f"{AUDIO_KBPS}k",
        "-ac",
        "1",
        # Без этого Телеграм отдаёт файл только целиком: индекс лежит в конце,
        # и на телефоне ролик открывается после полной загрузки.
        "-movflags",
        "+faststart",
        "-y",
        str(destination),
    ]


def _encode(
    ffmpeg_path: str,
    source: Path,
    destination: Path,
    plan: Plan,
    *,
    timeout_seconds: float,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            encode_command(ffmpeg_path, source, destination, plan),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        destination.unlink(missing_ok=True)
        raise ShrinkFailed(
            "Сжатие ролика не уложилось в отведённое время. Смотрите его в админке."
        ) from exc
    except OSError as exc:
        destination.unlink(missing_ok=True)
        raise ShrinkFailed("Сжать ролик нечем: ffmpeg на узле не запускается.") from exc
    if result.returncode != 0 or not destination.is_file() or destination.stat().st_size == 0:
        complaint = (result.stderr or "").strip() or "ffmpeg ничего не сообщил"
        destination.unlink(missing_ok=True)
        logger.error("Сжатие %s не удалось: %s", source.name, complaint)
        raise ShrinkFailed("Сжать ролик не вышло — ffmpeg отказал. Смотрите его в админке.")


def encode_timeout(duration_seconds: float | None) -> float:
    """Сколько ждать одну попытку.

    Считается от длины ролика, а не константой: `veryfast` идёт быстрее
    реального времени, но запас нужен и на четырёхмегапиксельный кадр, и на
    узел, занятый детекцией. Предел не про качество, а про то, чтобы зависший
    ffmpeg не держал службу вечно; сторожа в это время кормит нить.
    """
    if duration_seconds is None:
        return 900.0
    return min(max(120.0, duration_seconds * 6.0), 1800.0)


def shrink_to_fit(
    ffmpeg_path: str,
    source: Path,
    destination: Path,
    *,
    limit_bytes: int,
    duration_seconds: float | None = None,
) -> Path:
    """Пережать `source` так, чтобы он влез в `limit_bytes`.

    Возвращает `destination`. Не влезло за `MAX_ATTEMPTS` попыток или влезть
    не могло — `ShrinkFailed` с текстом для чата.
    """
    if duration_seconds is None:
        duration_seconds = probe_duration(ffmpeg_path, source)
    budget = int(limit_bytes * BUDGET_SHARE)
    plan = first_plan(duration_seconds, budget, limit_bytes)
    timeout = encode_timeout(duration_seconds)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        logger.info(
            "Сжатие %s, попытка %d: ширина %d, %s",
            source.name,
            attempt,
            plan.width,
            f"{plan.video_kbps} кбит/с" if plan.video_kbps else f"crf {plan.crf}",
        )
        _encode(ffmpeg_path, source, destination, plan, timeout_seconds=timeout)
        size = destination.stat().st_size
        if size <= limit_bytes:
            return destination
        if attempt == MAX_ATTEMPTS:
            break
        plan = tighter_plan(
            plan,
            overshoot=size / limit_bytes,
            duration_seconds=duration_seconds,
            limit_bytes=limit_bytes,
        )
    destination.unlink(missing_ok=True)
    raise ShrinkFailed(
        "Ролик не удалось ужать до предела Телеграма. Смотрите его в админке."
    )
