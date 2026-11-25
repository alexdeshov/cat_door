"""Облачная копия кошачьих проходов: страховка от выноса оборудования.

Клипы проходов до сих пор лежали только на узле и на внешнем диске рядом с ним
— оба в одном доме. Кража, пожар, залив уносили единственную видеозапись
кошачьих проходов целиком, тогда как у калитки такая страховка есть с 14
августа.

Почему отдельная служба, а не второй источник в `uploader.py`:

- **заливка калитки — самый ответственный путь в проекте**, и ронять её ошибкой
  в кошачьем коде нельзя. Разные процессы — разные аварии;
- цикл калитки сросся с Frigate: он спрашивает `/api/events`, смотрит
  `has_clip`, тянет клип по HTTP. Кошачьи события — строки нашей базы и файлы
  на диске, никакого HTTP;
- у догона другой темп. Калитка живёт свежими событиями, здесь же надо увезти
  накопленное: 84 ГБ уехали за пять часов при полосе около 20 Мбит/с.

Что перенесено из калиточного выгрузчика дословно, потому что куплено авариями:

- **сторожа кормить внутри передачи** (`feed_watchdog`): один клип уезжает минуты,
  и systemd убивал заливку посреди работы;
- **отметка на каждый объект, а не в конце цикла**: догон идёт часами и может
  не дожить до конца — иначе всё поедет заново;
- **нечитаемое состояние — пропустить цикл, а не начать с чистого листа**:
  чистый лист перезаписал бы журнал залитого;
- **ошибки botocore не наследуют `OSError`** — ловится `Exception`;
- удалять служба не умеет и не должна: ключ только на запись, уборку делает
  lifecycle бакета.

Порядок заливки — **от старых к новым**. Это не вкусовщина: локальное окно
хранения удаляет как раз самое старое, поэтому в первую очередь спасать надо
то, что первым пойдёт под нож.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

from sqlalchemy import select

from src.cloud_backup.uploader import (
    _alert_sender,
    _attempt_entry,
    _load_state,
    _remember,
    _save_state_quietly,
    deferred_ids,
    human_size,
    retry_delay,
)
from src.common.cloud_backup_config import (
    CloudBackupConfig,
    cats_cloud_backup_state_path,
    cloud_backup_config_path,
    load_cloud_backup_config,
)
from src.common.database import Event, create_session_factory
from src.common.sd_notify import Notifier, feed_watchdog
from src.common.settings import Settings

logger = logging.getLogger("catdoor.cats-cloud-backup")

# Пауза между кругами, когда заливать нечего. Проходы случаются десятками в
# сутки, спешить некуда: минута против тридцати секунд у калитки, потому что
# кошачий клип — не улика по краже.
POLL_SECONDS = 60.0
# Сколько проходов брать за круг. Предел нужен, чтобы отметки успевали лечь на
# диск и служба оставалась отзывчивой: при догоне в тысячу с лишним событий
# один бесконечный круг означал бы часы без единой записи состояния.
BATCH = 10
# Сколько идентификаторов помнить. Проходов ~30 в сутки; двадцати тысяч хватает
# на годы, а журнал остаётся файлом в несколько сотен килобайт.
REMEMBER_IDS = 20_000
# Ракурсы прохода и поля записи, где лежат их пути.
VIEWS = (("face", "media_path"), ("side", "side_media_path"))
# Статусы, при которых клип считается пригодным. `partial` — это проход, где
# один ракурс не записался; второй заливать всё равно надо.
USABLE_STATUSES = ("ready", "partial")
# Сколько отказов подряд у одного прохода терпеть молча. Пятый отказ — это
# уже четверть часа бэкоффа (1+2+4+8 мин), то есть не мигание связи, а беда:
# с 18 по 25 ноября бакет упёрся в предел размера, и неделю об этом знал только
# журнал службы.
ALERT_AT_ATTEMPT = 5
# Как часто повторять тревогу о застрявших проходах.
ALERT_REPEAT_SECONDS = 6 * 3600.0


def record_failure(attempts: dict[str, dict[str, float]], key: str, now: float) -> int:
    """Учесть отказ прохода; возвращает номер попытки.

    Поле — `last_at`, как его читает `deferred_ids`. До 25 ноября здесь писалось
    `at`: отсрочка всегда считалась от нуля, проход повторялся без паузы, и
    неделя упора в предел бакета стоила ~780 тысяч платных POST-запросов.
    """
    entry = attempts.setdefault(key, {"count": 0, "last_at": 0.0})
    entry["count"] = int(entry.get("count") or 0) + 1
    entry["last_at"] = now
    entry.pop("at", None)
    return int(entry["count"])


def pause_after_round(done: int, remaining: int) -> float:
    """Сколько спать после круга.

    Без паузы — только если круг что-то залил и очередь не пуста: догон должен
    идти подряд. Круг без единой заливки (отказ, всё отложено) спит обычную
    паузу — иначе отказ превращается в цикл по десять запросов в секунду.
    """
    return 0.0 if done and remaining else POLL_SECONDS


def object_key(prefix: str, event_id: int, created_at: Any, view: str) -> str:
    """Ключ в бакете. Детерминированный: потерянный журнал даёт перезапись того
    же объекта новой версией, а не второй экземпляр и второй счёт."""
    day = created_at.strftime("%Y-%m-%d") if created_at else "unknown"
    return f"{prefix.strip('/')}/{day}/{event_id:08d}/{view}.mp4"


def pending_events(
    factory: Any,
    uploaded: set[str],
    deferred: set[str],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Проходы, которые ещё не в облаке, — от старых к новым.

    Отбор идёт по базе, а не по каталогу: каталог не знает ни статуса прохода,
    ни того, какой ракурс вообще записался.
    """
    with factory() as session:
        rows = session.scalars(
            select(Event)
            .where(Event.status.in_(USABLE_STATUSES))
            .order_by(Event.id)
        ).all()
        chosen: list[dict[str, Any]] = []
        for event in rows:
            key = str(event.id)
            if key in uploaded or key in deferred:
                continue
            views = {
                view: getattr(event, field)
                for view, field in VIEWS
                if getattr(event, field)
            }
            if not views:
                continue
            chosen.append(
                {"id": event.id, "created_at": event.created_at, "views": views}
            )
            if limit is not None and len(chosen) >= limit:
                break
        return chosen


def _upload_event(
    client: Any,
    bucket: str,
    prefix: str,
    data_dir: Path,
    event: dict[str, Any],
    notifier: Notifier,
) -> tuple[int, list[str]]:
    """Залить оба ракурса прохода. Возвращает залитые байты и список пропусков.

    Пропуск — не отказ: файла может не быть, если окно хранения уже съело его.
    Такой проход отмечается залитым, иначе служба вечно спотыкалась бы об него.
    """
    sent = 0
    missing: list[str] = []
    for view, relative in event["views"].items():
        source = (data_dir / relative).resolve()
        if not source.is_file():
            missing.append(view)
            continue
        key = object_key(prefix, event["id"], event["created_at"], view)
        size = source.stat().st_size
        # Сторож кормится нитью на время передачи: файл в полсотни мегабайт
        # едет десятки секунд, а при занятом канале и минуты, тогда как
        # `WatchdogSec` у службы — 900 секунд.
        with feed_watchdog(notifier):
            client.upload_file(str(source), bucket, key)
        sent += size
    return sent, missing


def run(settings: Settings, *, once: bool = False) -> None:
    notifier = Notifier.from_environment()
    notifier.ready("Облачная копия кошачьих проходов на связи")
    config_path = cloud_backup_config_path(settings.data_dir)
    state_path = cats_cloud_backup_state_path(settings.data_dir)
    config = CloudBackupConfig()
    factory = create_session_factory(settings.resolved_database_path)
    alert = _alert_sender(settings)
    client = None
    client_key: tuple[str, str, str, str] | None = None

    while True:
        notifier.watchdog()
        config = load_cloud_backup_config(config_path, config)
        loaded = _load_state(state_path)
        if loaded is None:
            logger.warning("Состояние не читается — пропускаю круг, журнал не трогаю")
            if once:
                return
            time.sleep(POLL_SECONDS)
            continue
        state = loaded
        if not config.cats_enabled or not config.cats_ready:
            state["status"] = "disabled" if not config.cats_enabled else "not_configured"
            state["message"] = (
                "Выключено в админке"
                if not config.cats_enabled
                else "Не заданы бакет или ключи"
            )
            _save_state_quietly(state_path, state)
            if once:
                return
            time.sleep(POLL_SECONDS)
            continue

        uploaded = set(state.get("uploaded", []))
        order = list(state.get("order", []))
        attempts = {str(k): _attempt_entry(v) for k, v in state.get("attempts", {}).items()}
        now = time.time()
        deferred = deferred_ids(attempts, now)
        # Заливки этого круга: от них зависит, спать ли после него. Ноль и при
        # упавшем круге — тогда пауза обязательна.
        done = 0

        try:
            import boto3  # type: ignore[import-not-found]  # ставится на узле, в dev-среде не нужен

            wanted = (
                config.endpoint,
                config.region,
                config.access_key_id,
                config.secret_key,
            )
            if client is None or client_key != wanted:
                extra: dict[str, Any] = {}
                try:
                    from botocore.config import (  # type: ignore[import-not-found]
                        Config as BotoConfig,
                    )

                    extra["config"] = BotoConfig(
                        connect_timeout=15,
                        read_timeout=300,
                        retries={"max_attempts": 2},
                    )
                except ImportError:
                    pass
                client = boto3.client(
                    "s3",
                    **extra,
                    endpoint_url=config.endpoint,
                    region_name=config.region,
                    aws_access_key_id=config.access_key_id,
                    aws_secret_access_key=config.secret_key,
                )
                client_key = wanted

            # Очередь считается целиком один раз за круг: её размер нужен и для
            # отчёта в админке, и для решения, спать ли до следующего круга.
            queue = pending_events(factory, uploaded, deferred)
            batch = queue[:BATCH]
            bucket = config.cats_target_bucket
            sent_bytes = 0
            started = time.monotonic()
            for event in batch:
                key = str(event["id"])
                try:
                    size, missing = _upload_event(
                        client, bucket, config.cats_prefix, settings.data_dir, event, notifier
                    )
                except Exception as exc:  # noqa: BLE001 — botocore не наследует OSError
                    count = record_failure(attempts, key, time.time())
                    logger.warning(
                        "Проход %s не уехал (%s), повтор через %.0f с",
                        key,
                        str(exc)[:200],
                        retry_delay(count),
                    )
                    # Попытки — в состояние сразу: круг перечитывает его с
                    # диска, и несохранённый счётчик обнулялся бы каждый раз.
                    state["attempts"] = attempts
                    if count >= ALERT_AT_ATTEMPT and alert is not None:
                        last = float(state.get("stuck_alert_at") or 0.0)
                        if not 0 <= time.time() - last < ALERT_REPEAT_SECONDS:
                            state["stuck_alert_at"] = time.time()
                            alert(
                                f"Кошачья облачная копия: проход {key} не уезжает "
                                f"{count} попыток подряд: {str(exc)[:150]}"
                            )
                    _save_state_quietly(state_path, state)
                    break
                if missing:
                    logger.info(
                        "Проход %s: нет файлов ракурсов %s — отмечаю без них",
                        key,
                        ", ".join(missing),
                    )
                attempts.pop(key, None)
                _remember(uploaded, order, key)
                sent_bytes += size
                done += 1
                # Отметка на каждый проход, а не в конце круга: догон идёт
                # часами и может не дожить до конца.
                state["uploaded"] = order[-REMEMBER_IDS:]
                state["order"] = order[-REMEMBER_IDS:]
                state["attempts"] = attempts
                _save_state_quietly(state_path, state)

            spent = max(0.001, time.monotonic() - started)
            remaining = max(0, len(queue) - done)
            if done:
                rate = sent_bytes / spent
                logger.info(
                    "Залито проходов: %d, %s, темп %s/с",
                    done,
                    human_size(sent_bytes),
                    human_size(rate),
                )
                state["status"] = "ok"
                state["message"] = (
                    f"Залито {done} проходов, {human_size(sent_bytes)}, "
                    f"темп {human_size(rate)}/с"
                    + (", очередь не пуста" if remaining else ", очередь пуста")
                )
            else:
                state["status"] = "ok"
                state["message"] = (
                    "Очередь пуста: все проходы в облаке"
                    if not remaining
                    else "Очередь ждёт: все попытки отложены после отказов"
                )
            state["pending"] = remaining
            state["bucket"] = bucket
            _save_state_quietly(state_path, state)
        except Exception as exc:  # noqa: BLE001 — botocore не наследует OSError
            logger.warning("Круг не удался: %s", str(exc)[:200])
            state["status"] = "fail"
            state["message"] = f"Отказ: {str(exc)[:150]}"
            _save_state_quietly(state_path, state)
            if alert is not None:
                alert(f"Кошачья облачная копия: {str(exc)[:150]}")

        if once:
            return
        time.sleep(
            pause_after_round(done, int(state.get("pending") or 0))
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser(description="Облачная копия кошачьих проходов")
    parser.add_argument("--once", action="store_true", help="один круг и выход")
    args = parser.parse_args()
    run(Settings(), once=args.once)


if __name__ == "__main__":
    main()
