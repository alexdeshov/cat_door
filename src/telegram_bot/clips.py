"""Бот Телеграма: нажали кнопку под уведомлением — уехал ролик.

Почему отдельная служба, а не поток в уведомлениях калитки или в recorder.

- **Читать обновления бота может только один процесс.** Телеграм отдаёт их
  длинным опросом `getUpdates`, и второй читающий получает `Conflict: terminated
  by other getUpdates request`. Кнопки ставят двое (уведомления калитки и
  recorder), значит читающий обязан быть третьим и единственным.
- Отправка ролика идёт минутами, а recorder в это время должен писать проходы.
  Разные процессы — разные аварии: зависшая отдача видео не стоит пропущенного
  кота.

Вебхука здесь быть не может: снаружи в домашнюю сеть не войти (три NAT,
`docs/second-node-plan.md`), и это не временное неудобство, а устройство узла.
Поэтому опрос — не запасной путь, а единственный.

Два источника роликов, и они устроены по-разному:

- **калитка** — событие Frigate, ролик тянется по HTTP у самого Frigate;
- **проход кота** — строка нашей базы, ролик уже лежит файлом в каталоге данных.

Общее у них одно: пока человек не нажал, наружу не уходит ни байта.

Ролик тяжелее пятидесяти мегабайт Телеграм от бота не принимает — это
предел Bot API. Такой ролик здесь не отвергается, а пережимается на узле
(`src/telegram_bot/shrink.py`): событие калитки в несколько минут весит
сотни мегабайт, и раньше кнопка отказывала ровно на том, ради чего её и
нажимают. Любое уведомление обязано доезжать видео.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from sqlalchemy import select

from src.common.clip_button import (
    GATE_KIND,
    PASS_KIND,
    VIEW_NAMES,
    ClipRequest,
    parse_clip_callback,
)
from src.common.database import Event, create_session_factory
from src.common.durable_write import write_text_durably
from src.common.sd_notify import Notifier, feed_watchdog
from src.common.settings import ConfigurationError, Settings, validate_runtime_paths
from src.common.telegram import (
    UPLOAD_LIMIT_BYTES,
    TelegramClient,
    TelegramError,
    TelegramUnreachable,
)
from src.telegram_bot.shrink import ShrinkFailed, shrink_to_fit

logger = logging.getLogger("catdoor.telegram-bot")

# Пауза после обрыва связи. Телеграм недоступен целиком, а не для одного
# сообщения: долбиться в него без передышки значит греть журнал и процессор.
UNREACHABLE_PAUSE_SECONDS = 15.0
# Сколько помнить нажатие. Ролик едет не мгновенно, и нетерпеливый человек
# жмёт кнопку второй раз — второй ролик ему не нужен.
REPEAT_PRESS_SECONDS = 180.0
# Ракурс прохода — поле записи, где лежит путь к ролику.
VIEW_FIELDS = {"face": "media_path", "side": "side_media_path"}
VIEW_TITLES = {"face": "дверца", "side": "терраса"}


class ClipUnavailable(RuntimeError):
    """Ролика не будет, и человеку надо сказать почему — его словами."""


def offset_path(data_dir: Path) -> Path:
    return data_dir / "runtime" / "telegram-bot.json"


def clip_directory(data_dir: Path) -> Path:
    """Куда складывать скачанный ролик перед отправкой.

    В каталоге данных, а не в `/tmp`: приватный `/tmp` службы умирает вместе с
    её перезапуском, и половина скачанного клипа уехала бы в никуда. Заодно
    здесь видно брошенное: всё, что тут лежит, — недоделанная отправка.
    """
    return data_dir / "runtime" / "telegram-clips"


def load_offset(path: Path) -> int:
    """Какое обновление читать следующим.

    Ноль значит «с самого начала того, что Телеграм ещё хранит». Потеря файла
    не страшна: Телеграм держит обновления сутки, и худшее последствие — один
    старый ролик, уехавший повторно.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return int(payload["offset"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return 0


def save_offset(path: Path, offset: int) -> None:
    write_text_durably(path, json.dumps({"offset": offset}), mode=0o640)


def sweep_abandoned_clips(data_dir: Path) -> None:
    """Убрать то, что осталось от прерванной отправки.

    Файл удаляется в `finally`, а `finally` не исполняется при пропаже питания
    и при `SIGKILL` от сторожа. Каталог отведён под одну задачу, поэтому любой
    файл в нём при старте заведомо брошен — и удалять его можно не глядя на
    возраст.
    """
    directory = clip_directory(data_dir)
    if not directory.is_dir():
        return
    for leftover in directory.glob("*"):
        if leftover.is_file():
            leftover.unlink(missing_ok=True)


def _megabytes(size: int) -> str:
    return f"{size / 1024 / 1024:.0f} МБ"


def _reject_beyond_ceiling(size: int, ceiling: int) -> None:
    """Потолок исходника — не предел Телеграма, а предел здравого смысла.

    Пятьдесят мегабайт Bot API добираются сжатием, но тянуть ради этого с
    Frigate гигабайтную склейку припаркованной машины бессмысленно вдвойне:
    такой ролик и не сожмётся во что-то смотрибельное, а том данных и время
    узла потратит целиком.
    """
    if size > ceiling:
        raise ClipUnavailable(
            f"Ролик весит {_megabytes(size)} — это больше, чем узел берётся "
            f"пережимать ({_megabytes(ceiling)}). Смотрите его в админке."
        )


def _safe_name(event_id: str) -> str:
    """Имя файла из идентификатора события.

    Идентификатор приходит нажатием, а нажатие — строка снаружи. В имя файла
    она попадает только буквами, цифрами и точками: путь наружу каталога
    отправки не нужен ни одному настоящему событию.
    """
    return "".join(symbol if symbol.isalnum() or symbol in "._-" else "_" for symbol in event_id)


def fetch_gate_clip(
    api_url: str,
    event_id: str,
    destination: Path,
    *,
    timeout: float,
    ceiling_bytes: int,
) -> Path:
    """Скачать ролик события калитки у Frigate.

    Отказ `404` — это почти всегда «ещё не готов», а не «нет никогда»: Frigate
    склеивает запись после конца события, и та же просьба через полминуты
    выполняется. Разделять эти случаи нечем, поэтому человеку предлагается
    подождать: он всё равно нажмёт ещё раз, и лучше пусть знает зачем.

    Размер проверяется дважды — по заголовку и по ходу чтения. Заголовка может
    не быть (Frigate отдаёт склейку потоком), а качать до конца гигабайтный
    клип припаркованной машины, чтобы потом сказать «не влезло», — значит
    занять том данных заведомо бесполезным. Предел здесь — потолок исходника, а
    не предел Телеграма: всё, что ниже потолка, доедет сжатым.
    """
    url = f"{api_url.rstrip('/')}/api/events/{event_id}/clip.mp4"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            declared = response.headers.get("Content-Length")
            if declared is not None and declared.isdigit():
                _reject_beyond_ceiling(int(declared), ceiling_bytes)
            destination.parent.mkdir(parents=True, exist_ok=True)
            size = 0
            with destination.open("wb") as target:
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    _reject_beyond_ceiling(size, ceiling_bytes)
                    target.write(chunk)
    except urllib.error.HTTPError as exc:
        destination.unlink(missing_ok=True)
        if exc.code == 404:
            raise ClipUnavailable(
                "Ролик ещё не готов — Frigate дописывает запись. "
                "Нажмите ещё раз через полминуты."
            ) from exc
        raise ClipUnavailable(f"Frigate отказал ({exc.code})") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        destination.unlink(missing_ok=True)
        raise ClipUnavailable(f"Ролик не забрался: {exc}") from exc
    except ClipUnavailable:
        destination.unlink(missing_ok=True)
        raise
    if not destination.is_file() or destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        raise ClipUnavailable("Frigate отдал пустой ролик")
    return destination


def pass_clip_path(factory: Any, data_dir: Path, event_id: str, view: str) -> tuple[Path, str]:
    """Путь к ролику кошачьего прохода и подпись к нему.

    Пути в базе относительные: том данных переезжал, и абсолютные пути в
    строках пережили бы переезд только на бумаге.
    """
    try:
        number = int(event_id)
    except ValueError as exc:
        raise ClipUnavailable("Проход не опознан") from exc
    with factory() as session:
        event = session.scalars(select(Event).where(Event.id == number)).one_or_none()
        if event is None:
            raise ClipUnavailable("Проход не найден — запись уже удалена")
        relative = getattr(event, VIEW_FIELDS[view], None)
        moment = event.created_at
    if not relative:
        raise ClipUnavailable(f"Ракурс «{VIEW_TITLES[view]}» у этого прохода не записан")
    path = data_dir / str(relative)
    if not path.is_file():
        # Окно хранения съедает старые записи, и это норма, а не поломка: в
        # чате строка живёт дольше, чем ролик на диске.
        raise ClipUnavailable("Ролик уже удалён окном хранения")
    when = moment.strftime("%d.%m %H:%M") if moment is not None else ""
    return path, f"Проход №{number} · {VIEW_TITLES[view]}{' · ' + when if when else ''}"


def _allowed(client: TelegramClient, chat_id: str) -> bool:
    """Отвечать только своим.

    Бота может найти кто угодно — достаточно знать его имя, — а кнопка из
    чужого чата не появится только потому, что сообщений туда не было. Список
    получателей уведомлений и есть список тех, кому положено видео.
    """
    return chat_id in client.chat_ids


def handle_press(
    settings: Settings,
    client: TelegramClient,
    factory: Any,
    callback: dict[str, Any],
    *,
    now: float,
    recent: dict[str, float],
    notifier: Notifier | None = None,
) -> bool:
    """Разобрать одно нажатие. `True` — ролик уехал.

    Исключений наружу не выпускает: нажатие, уронившее службу, повторилось бы
    после перезапуска и уронило её снова.
    """
    callback_id = str(callback.get("id", ""))
    data = str(callback.get("data", ""))
    message = callback.get("message")
    message = message if isinstance(message, dict) else {}
    chat = message.get("chat")
    chat_id = str((chat or {}).get("id", ""))
    request = parse_clip_callback(data)
    if request is None:
        return False
    if not _allowed(client, chat_id):
        logger.warning("Нажатие из чужого чата %s пропущено", chat_id or "—")
        _answer_quietly(client, callback_id, "Не для вас")
        return False
    # Часики на кнопке Телеграм крутит до ответа и секунд через десять
    # показывает нажавшему ошибку. Ролик за это время не успевает, поэтому
    # кнопка отвечает сразу, а видео приходит отдельным сообщением.
    # «Не нажимали» — это отсутствие записи, а не нулевое время. Ноль здесь
    # означал бы «нажали в момент старта службы», а часы монотонные и после
    # запуска идут как раз от нуля: первые три минуты жизни службы каждая
    # кнопка отвечала бы «уже готовлю» и не слала ничего.
    pressed_at = recent.get(f"{chat_id}:{data}")
    fresh = pressed_at is not None and now - pressed_at < REPEAT_PRESS_SECONDS
    _answer_quietly(client, callback_id, "Уже готовлю" if fresh else "Готовлю видео…")
    if fresh:
        return False
    recent[f"{chat_id}:{data}"] = now
    reply_to = message.get("message_id")
    try:
        return _deliver(
            settings,
            client,
            factory,
            request,
            chat_id=chat_id,
            reply_to=int(reply_to) if isinstance(reply_to, int) else None,
            notifier=notifier,
        )
    except ClipUnavailable as exc:
        # Отказ виден тому, кто нажал: молчащая кнопка неотличима от сломанной,
        # и разбираться с ней пришлось бы через журнал на узле.
        logger.info("Нажатие %s без ролика: %s", data, exc)
        recent.pop(f"{chat_id}:{data}", None)
        _tell_quietly(client, chat_id, str(exc))
        return False


def _deliver(
    settings: Settings,
    client: TelegramClient,
    factory: Any,
    request: ClipRequest,
    *,
    chat_id: str,
    reply_to: int | None,
    notifier: Notifier | None,
) -> bool:
    # Сторож кормится нитью на весь путь ролика, а не только на отправку:
    # скачивание, сжатие и отдача идут минутами, и systemd убивал бы работу
    # посреди неё — та же беда, что покупалась на облачной копии 14 августа.
    # Сжатие добавило к этому пути самую долгую его часть.
    with feed_watchdog(notifier) if notifier is not None else _nothing():
        return _deliver_now(
            settings, client, factory, request, chat_id=chat_id, reply_to=reply_to
        )


def _deliver_now(
    settings: Settings,
    client: TelegramClient,
    factory: Any,
    request: ClipRequest,
    *,
    chat_id: str,
    reply_to: int | None,
) -> bool:
    data_dir = settings.data_dir
    name = _safe_name(request.event_id)
    # Временные файлы этой отправки: их и только их убирает `finally`. Ролик
    # прохода лежит на томе данных и принадлежит не нам.
    temporary: list[Path] = []
    if request.kind == GATE_KIND:
        downloaded = clip_directory(data_dir) / f"gate-{name}.mp4"
        temporary.append(downloaded)
        path = fetch_gate_clip(
            settings.gate_api_url,
            request.event_id,
            downloaded,
            timeout=settings.telegram_clip_timeout_seconds,
            ceiling_bytes=settings.telegram_clip_source_limit_mb * 1024 * 1024,
        )
        caption = "Калитка · видео события"
    elif request.kind == PASS_KIND:
        path, caption = pass_clip_path(
            factory,
            data_dir,
            request.event_id,
            VIEW_NAMES[request.view],
        )
        name = f"{name}-{request.view}"
    else:  # pragma: no cover — `parse_clip_callback` других видов не отдаёт
        return False
    try:
        shrunk = clip_directory(data_dir) / f"fit-{request.kind}-{name}.mp4"
        path, caption = _fit_for_telegram(
            settings, client, path, shrunk, caption=caption, chat_id=chat_id
        )
        if path == shrunk:
            temporary.append(shrunk)
        client.send_video(path, caption, chat_id=chat_id, reply_to=reply_to)
    except TelegramError as exc:
        raise ClipUnavailable(f"Телеграм не принял ролик: {exc}") from exc
    finally:
        for leftover in temporary:
            leftover.unlink(missing_ok=True)
    logger.info("Ролик %s отправлен в чат %s", request.event_id, chat_id)
    return True


def _fit_for_telegram(
    settings: Settings,
    client: TelegramClient,
    source: Path,
    destination: Path,
    *,
    caption: str,
    chat_id: str,
) -> tuple[Path, str]:
    """Ролик, который Телеграм примет: сам исходник или его сжатая копия.

    Пятьдесят мегабайт — предел Bot API, и раньше тяжёлый ролик просто не
    уезжал. Отказывал этот предел ровно на длинных событиях, то есть на тех,
    ради которых кнопку и жмут, поэтому узел пережимает такой ролик сам.

    О сжатии человеку говорят заранее: оно занимает минуты, а кнопка к этому
    времени уже ответила «готовлю видео…» — молчащий чат неотличим от
    сломанного. В подписи остаётся исходный вес: присланное хуже того, что
    лежит в админке, и знать об этом полезно до того, как вглядываться в
    номер машины.
    """
    size = source.stat().st_size
    if size <= UPLOAD_LIMIT_BYTES:
        return source, caption
    _tell_quietly(
        client,
        chat_id,
        f"Ролик весит {_megabytes(size)} — Телеграм столько от бота не принимает. "
        "Сжимаю, это займёт минуту-другую.",
    )
    logger.info("Ролик %s весит %s — сжимаю", source.name, _megabytes(size))
    try:
        shrink_to_fit(
            settings.ffmpeg_path,
            source,
            destination,
            limit_bytes=UPLOAD_LIMIT_BYTES,
        )
    except ShrinkFailed as exc:
        destination.unlink(missing_ok=True)
        raise ClipUnavailable(str(exc)) from exc
    return destination, f"{caption} · сжат с {_megabytes(size)}"


class _nothing:
    """Заглушка на случай запуска без systemd: кормить некого."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: object) -> None:
        return None


def _answer_quietly(client: TelegramClient, callback_id: str, text: str) -> None:
    """Погасить часики. Отказ здесь ничего не меняет: ролик едет своим ходом."""
    if not callback_id:
        return
    try:
        client.answer_callback(callback_id, text)
    except (TelegramError, TelegramUnreachable) as exc:
        logger.warning("Кнопка не ответила: %s", exc)


def _tell_quietly(client: TelegramClient, chat_id: str, text: str) -> None:
    try:
        client.send_text_to(chat_id, text)
    except (TelegramError, TelegramUnreachable) as exc:
        logger.warning("Объяснение не ушло: %s", exc)


def forget_old_presses(recent: dict[str, float], *, now: float) -> dict[str, float]:
    return {key: at for key, at in recent.items() if now - at < REPEAT_PRESS_SECONDS}


def run(*, once: bool = False) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    settings = Settings()
    validate_runtime_paths(settings)
    client = TelegramClient(
        settings.telegram_bot_token,
        settings.telegram_chat_id,
        api_base=settings.telegram_api_base,
    )
    if not client.configured:
        raise ConfigurationError(
            "Кнопка «Видео» включена, но бот не настроен: нет токена или адреса чата"
        )
    sweep_abandoned_clips(settings.data_dir)
    factory = create_session_factory(settings.resolved_database_path)
    path = offset_path(settings.data_dir)
    offset = load_offset(path)
    recent: dict[str, float] = {}
    notifier = Notifier.from_environment()
    notifier.ready("Кнопка «Видео» на связи")
    while True:
        try:
            updates = client.poll_updates(
                offset,
                timeout_seconds=settings.telegram_bot_poll_seconds,
            )
        except TelegramUnreachable as exc:
            logger.warning("Телеграм недоступен: %s", exc)
            notifier.watchdog()
            if once:
                return
            time.sleep(UNREACHABLE_PAUSE_SECONDS)
            continue
        except TelegramError as exc:
            # Сюда же приходит `Conflict`: обновления этого бота уже читает
            # кто-то другой. Второй читающий не безобиден — он забирает
            # нажатия себе, и кнопки молчат через раз.
            logger.warning("Обновления не прочитались: %s", exc)
            notifier.watchdog()
            if once:
                return
            time.sleep(UNREACHABLE_PAUSE_SECONDS)
            continue
        notifier.watchdog()
        for update in updates:
            update_id = update.get("update_id")
            if isinstance(update_id, int):
                # Смещение сохраняется ДО разбора, а не после: нажатие, от
                # которого служба падает, иначе читалось бы снова после
                # каждого перезапуска — и служба падала бы вечно.
                offset = update_id + 1
                save_offset(path, offset)
            callback = update.get("callback_query")
            if isinstance(callback, dict):
                handle_press(
                    settings,
                    client,
                    factory,
                    callback,
                    now=time.monotonic(),
                    recent=recent,
                    notifier=notifier,
                )
            notifier.watchdog()
        recent = forget_old_presses(recent, now=time.monotonic())
        if once:
            return


def main() -> None:
    try:
        run()
    except ConfigurationError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
