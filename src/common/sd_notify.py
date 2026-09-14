"""Minimal sd_notify client for systemd services of `Type=notify`.

systemd passes the path of a unix datagram socket in ``NOTIFY_SOCKET`` and reads
newline separated ``key=value`` messages from it. Outside systemd the variable is
absent, every method here turns into a no-op, and both services keep starting
from a shell exactly as they did before.

Written by hand rather than pulled from ``python-systemd``: that package needs a
compiler and system headers on the runtime host, which is a steep price for
three datagrams.
"""

from __future__ import annotations

import os
import socket
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager


def _resolve_address(address: str) -> str:
    """Return the socket path, translating the abstract namespace form.

    systemd may hand out an abstract socket, written with a leading ``@`` that
    stands for a leading NUL byte.
    """
    if not address:
        return ""
    if address.startswith("@"):
        return "\0" + address[1:]
    return address


def _watchdog_usec(source: Mapping[str, str]) -> int:
    """Read the watchdog period, honouring the pid guard systemd sets.

    ``WATCHDOG_USEC`` is inherited by children. systemd sets ``WATCHDOG_PID`` so
    that only the process the timeout belongs to acts on it; anything else must
    ignore the period rather than ping on someone else's behalf.
    """
    raw = source.get("WATCHDOG_USEC")
    if raw is None:
        return 0
    owner = source.get("WATCHDOG_PID")
    if owner is not None and owner != str(os.getpid()):
        return 0
    try:
        value = int(raw)
    except ValueError:
        return 0
    return max(0, value)


class Notifier:
    """Sends readiness, status and liveness datagrams to systemd."""

    def __init__(self, address: str = "", watchdog_usec: int = 0) -> None:
        self._address = _resolve_address(address)
        self._watchdog_usec = max(watchdog_usec, 0)
        self._socket: socket.socket | None = None
        # Пинговать могут двое: главный поток и нить-кормилец на время долгой
        # передачи (облачная копия). Без замка они делят один сокет — двойное
        # создание с утечкой дескриптора или sendto по закрытому соседом.
        self._lock = threading.Lock()

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> Notifier:
        source = os.environ if environ is None else environ
        return cls(source.get("NOTIFY_SOCKET", ""), _watchdog_usec(source))

    @property
    def enabled(self) -> bool:
        return bool(self._address)

    @property
    def watchdog_interval_seconds(self) -> float | None:
        """Half the configured period, or None when no watchdog is configured.

        systemd documents half the timeout as the interval to ping at, which
        leaves room for one missed cycle before the service is declared hung.
        """
        if not self.enabled or self._watchdog_usec <= 0:
            return None
        return self._watchdog_usec / 2_000_000

    def ready(self, status: str | None = None) -> None:
        message = "READY=1"
        if status is not None:
            message += f"\nSTATUS={_single_line(status)}"
        self._send(message)

    def watchdog(self) -> None:
        self._send("WATCHDOG=1")

    def status(self, text: str) -> None:
        self._send(f"STATUS={_single_line(text)}")

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def _send(self, message: str) -> None:
        """Deliver one datagram, swallowing every socket failure.

        A supervisor that stopped listening is not a reason to take the recorder
        down: losing the ping costs a restart at worst, raising here would cost
        the recording.
        """
        if not self.enabled:
            return
        with self._lock:
            try:
                if self._socket is None:
                    self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
                self._socket.sendto(message.encode("utf-8"), self._address)
            except OSError:
                self._close_locked()


def _single_line(text: str) -> str:
    """Collapse newlines: they separate fields in the protocol."""
    return " ".join(text.split())


# Сколько нить-кормилец живёт максимум. Таймауты передач по-операционные:
# полуживое TCP-соединение, отдающее байт раз в сто секунд, держит
# `_fetch_bytes`/`put_object` вечно — и нить, пингующая без предела, делала бы
# такое зависание невидимым НАВСЕГДА: сторож сыт, тревога живёт в том же
# заблокированном потоке (находка ревью 15.08). Честная передача самого
# крупного клипа (~150 МБ на 300 КБ/с в обе стороны) укладывается в
# двадцать минут; после часа нить замолкает, и systemd перезапускает
# застрявшую службу штатно.
HEARTBEAT_LIFETIME_SECONDS = 3600.0


@contextmanager
def feed_watchdog(notifier: Notifier) -> Iterator[None]:
    """Кормить сторожа во время передачи, а не только между событиями.

    Клип в десять минут — это около 150 МБ, и один `put_object` идёт минутами.
    (Число «300 КБ/с», стоявшее здесь до 31.08, было замером одного неудачного
    вечера: настоящая полоса узла — около 20 Мбит/с, проверено на догоне
    кошачьей копии. Порядок величин важнее точности: передача длиннее
    `WatchdogSec` бывает при любой из них.) 14 августа сторож
    (тогда 300 с) трижды убил выгрузчик посреди заливки, и каждый SIGABRT стоил
    потерянных отметок и повторной заливки уже залитого окна — оплаченных
    лишних версий в бакете.

    Нить пингует от имени того же процесса, поэтому `NotifyAccess=main` не
    против: проверка идёт по pid. Без сторожа (`watchdog_interval_seconds` —
    `None`) нить не заводится вовсе. Жизнь нити ограничена
    `HEARTBEAT_LIFETIME_SECONDS`: бесконечный пинг превращал бы зависшую
    навсегда передачу в невидимую.
    """
    interval = notifier.watchdog_interval_seconds
    if interval is None:
        yield
        return
    done = threading.Event()

    def beat() -> None:
        beats = 0
        while not done.wait(interval) and beats * interval < HEARTBEAT_LIFETIME_SECONDS:
            notifier.watchdog()
            beats += 1

    thread = threading.Thread(target=beat, name="watchdog-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join(timeout=1.0)
