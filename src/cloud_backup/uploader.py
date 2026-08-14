"""Выгрузка клипов событий Frigate в S3-бакет — раз в минуту.

Назначение — страховка от выноса оборудования: к моменту, когда вор дошёл от
калитки до сервера, клип его подхода уже в облаке. Отсюда устройство:

- **опрос, а не подписка**: раз в минуту спрашиваем у Frigate завершённые
  события за окно оглядки и заливаем недостающие. Опрос идемпотентен и
  догоняет после любого простоя — облака, узла, самого выгрузчика; подписка
  на MQTT теряла бы всё, что случилось, пока служба лежала;
- **клип целиком, а не сегменты**: `GET /api/events/<id>/clip.mp4` отдаёт
  склеенный ролик события — смотреть его после кражи можно без плясок с
  10-секундными кусками;
- **журнал залитого** живёт в состоянии на диске: перезапуск не приводит ни к
  повторной заливке, ни к дырам. Ключи в бакете детерминированы
  (`events/дата/камера/id.mp4`), так что даже потерянное состояние даёт лишь
  перезапись того же объекта новой версией, а не дубль;
- деньги: входящий трафик у Яндекса бесплатен, DELETE не тарифицируется,
  чистит бакет lifecycle — службе удалять ничего не нужно (и нельзя: ключ
  только на запись).

Настройки перечитываются каждый цикл — включение из админки подхватывается
без перезапуска.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.common.cloud_backup_config import (
    CloudBackupConfig,
    cloud_backup_config_path,
    cloud_backup_state_path,
    load_cloud_backup_config,
)
from src.common.durable_write import write_text_durably
from src.common.sd_notify import Notifier
from src.common.settings import Settings

logger = logging.getLogger("catdoor.cloud-backup")

POLL_SECONDS = 60.0
# Сколько идентификаторов помнить. Событий ~400 в сутки, окно оглядки сутки —
# четырёх тысяч хватает с десятикратным запасом, а файл состояния не растёт.
REMEMBER_IDS = 4000


def _fetch_json(url: str, timeout: float = 15.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", "replace"))


def _fetch_bytes(url: str, timeout: float = 120.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def eligible_events(
    events: list[dict[str, Any]],
    config: CloudBackupConfig,
    uploaded: set[str],
) -> list[dict[str, Any]]:
    """Что заливать: завершённые, с записью, нужной камеры, ещё не залитые.

    Незавершённые события пропускаются без пометки — их клип ещё растёт, и
    заливать половину значило бы никогда не залить целое: пометка «сделано»
    закрыла бы дорогу повтору.
    """
    selected = []
    # Старые вперёд: хвост должен рассасываться хронологически, иначе одно
    # долгое событие в середине навсегда заслоняет всё, что старше.
    events = sorted(events, key=lambda item: float(item.get("start_time") or 0))
    for event in events:
        event_id = str(event.get("id") or "")
        if not event_id or event_id in uploaded:
            continue
        if not event.get("end_time"):
            continue
        if not event.get("has_clip", True):
            # Записи нет — заливать нечего; помечаем, чтобы не спрашивать вечно.
            selected.append({**event, "_skip": True})
            continue
        camera = str(event.get("camera") or "")
        if config.cameras and camera not in config.cameras:
            continue
        selected.append(event)
    return selected


def _object_key(config: CloudBackupConfig, event: dict[str, Any], suffix: str) -> str:
    day = datetime.fromtimestamp(float(event["start_time"]), UTC).strftime("%Y-%m-%d")
    camera = str(event.get("camera") or "unknown")
    return f"{config.prefix}/{day}/{camera}/{event['id']}.{suffix}"


def _load_state(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(UTC).isoformat()
    write_text_durably(path, json.dumps(state, ensure_ascii=False), mode=0o600)


def run(settings: Settings, *, once: bool = False) -> None:
    notifier = Notifier.from_environment()
    notifier.ready("Облачная копия событий на связи")
    config_path = cloud_backup_config_path(settings.data_dir)
    state_path = cloud_backup_state_path(settings.data_dir)
    config = CloudBackupConfig()
    client = None
    client_key: tuple[str, str, str, str] | None = None
    while True:
        notifier.watchdog()
        config = load_cloud_backup_config(config_path, config)
        state = _load_state(state_path)
        if not config.enabled or not config.ready:
            state["status"] = "disabled" if not config.enabled else "not_configured"
            state["message"] = (
                "Выключено в админке" if not config.enabled
                else "Не заданы бакет или ключи"
            )
            _save_state(state_path, state)
            if once:
                return
            time.sleep(POLL_SECONDS)
            continue
        try:
            import boto3  # type: ignore[import-not-found]  # ставится на узле, в dev-среде не нужен

            wanted = (config.endpoint, config.region, config.access_key_id, config.secret_key)
            if client is None or client_key != wanted:
                client = boto3.client(
                    "s3",
                    endpoint_url=config.endpoint,
                    region_name=config.region,
                    aws_access_key_id=config.access_key_id,
                    aws_secret_access_key=config.secret_key,
                )
                client_key = wanted
            after = time.time() - config.lookback_hours * 3600
            events = _fetch_json(
                f"{settings.gate_api_url.rstrip('/')}/api/events?"
                f"after={after:.0f}&limit=500"
            )
            if not isinstance(events, list):
                raise RuntimeError("Frigate вернул не список событий")  # noqa: TRY004 — это отказ API, не тип
            uploaded = set(state.get("uploaded") or [])
            attempts: dict[str, int] = dict(state.get("attempts") or {})
            sent_bytes = 0
            sent_events = 0
            failed = 0
            batch = eligible_events(events, config, uploaded)
            for position, event in enumerate(batch):
                event_id = str(event["id"])
                if event.get("_skip"):
                    uploaded.add(event_id)
                    continue
                # Одно ядовитое событие не должно ронять цикл: Frigate может
                # собирать клип часового события дольше любого таймаута.
                # 14 августа такой клип оборвал догоняющую выгрузку, а
                # застрявший сторож перезапустил службу целиком.
                try:
                    clip = _fetch_bytes(
                        f"{settings.gate_api_url.rstrip('/')}/api/events/{event_id}/clip.mp4"
                    )
                    client.put_object(
                        Bucket=config.bucket,
                        Key=_object_key(config, event, "mp4"),
                        Body=clip,
                        ContentType="video/mp4",
                    )
                except Exception as exc:  # noqa: BLE001 — пропуск, не крушение
                    attempts[event_id] = attempts.get(event_id, 0) + 1
                    failed += 1
                    if attempts[event_id] >= 5:
                        # Пять попыток по циклу на каждую — этот клип не отдать.
                        # Прекращаем жечь на нём таймауты; снимок ниже мог уйти.
                        uploaded.add(event_id)
                        logger.warning(
                            "Событие %s не отдалось за %d попыток, пропускаю: %s",
                            event_id, attempts[event_id], str(exc)[:120],
                        )
                    else:
                        logger.info(
                            "Событие %s отложено (попытка %d): %s",
                            event_id, attempts[event_id], str(exc)[:120],
                        )
                    notifier.watchdog()
                    continue
                attempts.pop(event_id, None)
                sent_bytes += len(clip)
                try:
                    snapshot = _fetch_bytes(
                        f"{settings.gate_api_url.rstrip('/')}/api/events/{event_id}/snapshot.jpg"
                    )
                    client.put_object(
                        Bucket=config.bucket,
                        Key=_object_key(config, event, "jpg"),
                        Body=snapshot,
                        ContentType="image/jpeg",
                    )
                    sent_bytes += len(snapshot)
                except OSError:
                    # Снимка может не быть — клип важнее, не срываем цикл.
                    logger.info("Событие %s без снимка", event_id)
                uploaded.add(event_id)
                sent_events += 1
                notifier.watchdog()
            state["uploaded"] = list(uploaded)[-REMEMBER_IDS:]
            state["attempts"] = {
                key: value for key, value in attempts.items() if value < 5
            }
            today = datetime.now(UTC).strftime("%Y-%m-%d")
            if state.get("day") != today:
                state["day"] = today
                state["day_events"] = 0
                state["day_bytes"] = 0
            state["day_events"] = int(state.get("day_events") or 0) + sent_events
            state["day_bytes"] = int(state.get("day_bytes") or 0) + sent_bytes
            state["status"] = "ok"
            tail = f", отложено {failed}" if failed else ""
            state["message"] = (
                f"За сутки: {state['day_events']} событий, "
                f"{state['day_bytes'] / 1e6:.0f} МБ{tail}"
            )
            state["last_success_at"] = time.time()
            state.pop("last_error", None)
            if sent_events:
                logger.info(
                    "Залито %d событий, %.1f МБ", sent_events, sent_bytes / 1e6
                )
        except Exception as exc:  # noqa: BLE001 — облако и Frigate падают, служба нет
            state["status"] = "error"
            state["last_error"] = str(exc)[:300]
            state["message"] = f"Ошибка выгрузки: {str(exc)[:120]}"
            logger.warning("Цикл выгрузки не удался: %s", exc)
        _save_state(state_path, state)
        if once:
            return
        time.sleep(POLL_SECONDS)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    run(Settings())


if __name__ == "__main__":
    main()
