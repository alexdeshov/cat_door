"""Отправка сообщений в Телеграм — единственный канал уведомлений наружу.

Новой зависимости здесь намеренно нет: узел живёт в домашней сети без
интернета для чего бы то ни было, кроме этих запросов, и библиотека ради
одного POST — лишний повод однажды чинить окружение вместо дверцы.

Токен бота попадает в адрес запроса, а адрес попадает в текст ошибки urllib.
Поэтому наружу из этого модуля не уходит ни одна строка, не прошедшая через
маскировку: журнал читают люди, а журнал уезжает в systemd.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

TELEGRAM_API = "https://api.telegram.org"

# Телеграм принимает от бота файл не больше 50 МБ — это предел Bot API, а не
# нашей настройки: обойти его можно только своим сервером Bot API, чего на узле
# нет. Клип, не влезший сюда, честнее назвать по имени, чем уронить отправку
# невнятным «Request Entity Too Large».
UPLOAD_LIMIT_BYTES = 50 * 1024 * 1024

# Нагрузка кнопки ограничена 64 байтами. Предел жёсткий: сообщение с более
# длинной `callback_data` Телеграм не примет целиком, то есть уведомление
# пропадёт вместе с кнопкой.
CALLBACK_DATA_LIMIT = 64


logger = logging.getLogger("catdoor.telegram")


class TelegramError(RuntimeError):
    """Сообщение не ушло. Текст уже без токена."""


class TelegramUnreachable(TelegramError):
    """До Телеграма не достучаться: сеть, блокировка, обрыв.

    Отдельный тип нужен не ради красоты. Отказ по одному сообщению — частное
    дело этого сообщения, и следующее может уйти. Недоступность — общая беда:
    следующая попытка займёт те же пятнадцать секунд и кончится тем же. Кто
    рассылает пачкой, обязан различать эти два случая, иначе на каждой пачке
    тратит `число событий × таймаут`.
    """



# Отказы Телеграма формально точны и совершенно непонятны тому, кто настраивает
# бота впервые. Каждая подсказка здесь — из реальной попытки, а не из
# документации.
_HINTS = (
    (
        "can't send messages to the bot",
        (
            "адрес чата — это ваш личный идентификатор, а не идентификатор бота: "
            "бот не может писать сам себе"
        ),
    ),
    (
        "chat not found",
        (
            "напишите боту первым: пока вы не нажали «Start», он не имеет права "
            "начать разговор"
        ),
    ),
    (
        "bot was blocked by the user",
        "бот заблокирован в телеграме — разблокируйте его в переписке",
    ),
    ("unauthorized", "токен не принят: проверьте, что скопирован он целиком, вместе с двоеточием"),
)


def hint_for(reason: str) -> str | None:
    """Человеческое объяснение к отказу Телеграма, если оно известно."""
    lowered = reason.lower()
    return next((hint for marker, hint in _HINTS if marker in lowered), None)


# Кнопка под сообщением: подпись и нагрузка, которая вернётся нажатием.
Buttons = Sequence[tuple[str, str]]


def _markup(buttons: Buttons) -> dict[str, str]:
    """Поле `reply_markup` для сообщения с кнопками. Без кнопок — пусто.

    Каждая кнопка своей строкой: подписи у нас длинные («Видео с дверцы»), и
    в один ряд телефон их обрезает до многоточия.
    """
    if not buttons:
        return {}
    too_long = [data for _, data in buttons if len(data.encode()) > CALLBACK_DATA_LIMIT]
    if too_long:
        # Молча отправить сообщение без кнопки было бы хуже: кнопка исчезла бы
        # ровно там, где она нужна, и никто бы не узнал почему.
        raise TelegramError(f"нагрузка кнопки длиннее {CALLBACK_DATA_LIMIT} байт: {too_long[0]}")
    keyboard = [[{"text": text, "callback_data": data}] for text, data in buttons]
    return {"reply_markup": json.dumps({"inline_keyboard": keyboard}, ensure_ascii=False)}


def _multipart(fields: dict[str, str], file_field: str, file_path: Path) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
            f"{value}\r\n".encode()
        )
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; "
        f"filename=\"{file_path.name}\"\r\n"
        "Content-Type: application/octet-stream\r\n\r\n".encode()
    )
    parts.append(file_path.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


@dataclass(frozen=True)
class TelegramClient:
    token: str
    chat_id: str
    timeout_seconds: float = 15.0
    # Отдельный таймаут на отправку файла. Пятнадцати секунд хватает сообщению
    # и снимку, но ролик в сорок мегабайт уезжает через домашний канал минутами
    # — с общим таймаутом он обрывался бы ровно на середине, и каждая попытка
    # стоила бы полной перезаливки.
    upload_timeout_seconds: float = 600.0
    api_base: str = TELEGRAM_API

    @property
    def chat_ids(self) -> list[str]:
        """Получатели: один chat id или несколько через запятую.

        Каждому пишется лично, а не в общую группу: у людей в семье разные
        чаты, и переезд уведомлений в группу отнял бы их у того, кто настроил.
        """
        return [part.strip() for part in self.chat_id.split(",") if part.strip()]

    @property
    def configured(self) -> bool:
        return bool(self.token and self.chat_ids)

    def mask(self, text: str) -> str:
        return text.replace(self.token, "***") if self.token else text

    def send_message(self, text: str, buttons: Buttons = ()) -> None:
        self._broadcast("sendMessage", {"text": text, **_markup(buttons)})

    def send_photo(self, photo: Path, caption: str, buttons: Buttons = ()) -> None:
        """Снимок с подписью. Подпись Телеграм обрезает на 1024 символах."""
        self._broadcast(
            "sendPhoto",
            {"caption": caption[:1024], **_markup(buttons)},
            file=("photo", photo),
        )

    def send_text_to(self, chat_id: str, text: str) -> None:
        """Ответ одному собеседнику, а не всей семье.

        Нажал кнопку один — ему и отвечаем. Рассылка «клип не готов» всем
        получателям превратила бы чужое любопытство в общий шум.
        """
        self._call("sendMessage", {"chat_id": chat_id, "text": text})

    def send_video(
        self,
        video: Path,
        caption: str,
        *,
        chat_id: str,
        reply_to: int | None = None,
    ) -> None:
        """Ролик тому, кто его попросил.

        `supports_streaming` — не украшение: без него Телеграм отдаёт файл
        только целиком, и на телефоне через мобильную сеть ролик открывается
        после полной загрузки.

        `reply_to` подвешивает ролик к тому самому уведомлению, под которым
        нажали кнопку: в чате за день набирается десяток проходов, и видео
        само по себе не отвечает на вопрос «чьё оно». Промах ответа Телеграм
        прощает (`allow_sending_without_reply`) — исходное сообщение могли
        удалить, и терять из-за этого ролик незачем.

        Тело запроса собирается в памяти целиком: файл ограничен пятьюдесятью
        мегабайтами, и потоковая отправка стоила бы своего `http.client`
        вместо трёх строк на `urllib`.
        """
        fields = {
            "chat_id": chat_id,
            "caption": caption[:1024],
            "supports_streaming": "true",
        }
        if reply_to is not None:
            fields["reply_to_message_id"] = str(reply_to)
            fields["allow_sending_without_reply"] = "true"
        self._call("sendVideo", fields, file=("video", video), timeout=self.upload_timeout_seconds)

    def answer_callback(self, callback_id: str, text: str = "") -> None:
        """Погасить часики на кнопке.

        Телеграм крутит их до ответа и примерно через десять секунд показывает
        нажавшему ошибку. Клип за это время не успевает ни скачаться, ни
        уехать, поэтому ответ уходит сразу, а ролик — следом отдельным
        сообщением.
        """
        self._call("answerCallbackQuery", {"callback_query_id": callback_id, "text": text[:200]})

    def poll_updates(self, offset: int, *, timeout_seconds: float) -> list[dict[str, object]]:
        """Длинный опрос обновлений: нажатия кнопок.

        Вебхук на узле невозможен — снаружи в домашнюю сеть не войти, — поэтому
        единственный способ узнать о нажатии это спросить самим. Соединение
        держится `timeout_seconds` и возвращает пустой список, если за это время
        никто ничего не нажал; сетевой таймаут даётся с запасом, иначе штатное
        молчание Телеграма выглядело бы обрывом связи.

        `allowed_updates` сужает поток до нажатий: чужие сообщения боту
        заставляли бы службу вычитывать и подтверждать переписку, которая ей
        не нужна.
        """
        payload = self._call(
            "getUpdates",
            {
                "offset": str(offset),
                "timeout": str(int(timeout_seconds)),
                "allowed_updates": json.dumps(["callback_query"]),
            },
            timeout=timeout_seconds + self.timeout_seconds,
        )
        result = payload.get("result")
        if not isinstance(result, list):
            return []
        return [item for item in result if isinstance(item, dict)]

    def _broadcast(
        self,
        method: str,
        fields: dict[str, str],
        file: tuple[str, Path] | None = None,
    ) -> None:
        """Отправка всем получателям по очереди.

        Доставлено — если дошло хоть до одного: иначе один заблокировавший бота
        получатель ронял бы отправку целиком, а сторож здоровья повторял бы
        сообщение о перезагрузке остальным до бесконечности. Недоставленные
        адреса уходят в журнал; отказ всем подряд — ошибка, как и раньше.
        """
        if not self.configured:
            raise TelegramError("бот не настроен: нет токена или адреса чата")
        failures: list[Exception] = []
        for chat_id in self.chat_ids:
            try:
                self._call(method, {"chat_id": chat_id, **fields}, file=file)
            except (TelegramError, TelegramUnreachable) as exc:
                failures.append(exc)
                if len(self.chat_ids) > 1:
                    logger.warning("Получателю %s не доставлено: %s", chat_id, exc)
        if failures and len(failures) == len(self.chat_ids):
            raise failures[0]

    def _call(
        self,
        method: str,
        fields: dict[str, str],
        file: tuple[str, Path] | None = None,
        timeout: float | None = None,
    ) -> dict[str, object]:
        if not self.configured:
            raise TelegramError("бот не настроен: нет токена или адреса чата")
        url = f"{self.api_base}/bot{self.token}/{method}"
        if file is None:
            body = urllib.parse.urlencode(fields).encode()
            content_type = "application/x-www-form-urlencoded"
        else:
            body, content_type = _multipart(fields, file[0], file[1])
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds if timeout is None else timeout
            ) as response:
                payload = json.loads(response.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            # Телеграм объясняет отказ в теле ответа: «chat not found», «Unauthorized».
            # Без этого текста настройка бота превращается в гадание.
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise TelegramError(self.mask(f"Телеграм отказал ({exc.code}): {detail}")) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise TelegramUnreachable(self.mask(f"Телеграм недоступен: {exc}")) from exc
        except json.JSONDecodeError as exc:
            raise TelegramError("Телеграм ответил не JSON") from exc
        if not payload.get("ok"):
            raise TelegramError(self.mask(f"Телеграм отказал: {payload}"))
        return payload
