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
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

TELEGRAM_API = "https://api.telegram.org"


class TelegramError(RuntimeError):
    """Сообщение не ушло. Текст уже без токена."""


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
    api_base: str = TELEGRAM_API

    @property
    def configured(self) -> bool:
        return bool(self.token and self.chat_id)

    def mask(self, text: str) -> str:
        return text.replace(self.token, "***") if self.token else text

    def send_message(self, text: str) -> None:
        self._call("sendMessage", {"chat_id": self.chat_id, "text": text})

    def send_photo(self, photo: Path, caption: str) -> None:
        """Снимок с подписью. Подпись Телеграм обрезает на 1024 символах."""
        self._call(
            "sendPhoto",
            {"chat_id": self.chat_id, "caption": caption[:1024]},
            file=("photo", photo),
        )

    def _call(
        self,
        method: str,
        fields: dict[str, str],
        file: tuple[str, Path] | None = None,
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
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            # Телеграм объясняет отказ в теле ответа: «chat not found», «Unauthorized».
            # Без этого текста настройка бота превращается в гадание.
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise TelegramError(self.mask(f"Телеграм отказал ({exc.code}): {detail}")) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise TelegramError(self.mask(f"Телеграм недоступен: {exc}")) from exc
        except json.JSONDecodeError as exc:
            raise TelegramError("Телеграм ответил не JSON") from exc
        if not payload.get("ok"):
            raise TelegramError(self.mask(f"Телеграм отказал: {payload}"))
        return payload
