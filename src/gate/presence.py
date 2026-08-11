"""Есть ли телефон человека в домашней сети. Второй фактор, пока выключенный.

Спрашивается не ping, а ARP: телефоны часто игнорируют ICMP в спящем режиме, а
на «кто держит этот адрес» отвечает сама прошивка Wi-Fi. Прямой `arping`
требует привилегий, которых у службы нет и не должно быть, поэтому делается то
же самое чужими руками: пакет по адресу заставляет ядро разрешить MAC, а
результат разрешения виден в таблице соседей.

Что эта проверка **не** доказывает, и знать это надо до включения:

1. Это защита от фотографии, а не от подготовленного человека: MAC подделывается
   одной командой.
2. Присутствие в сети — не присутствие у калитки. Телефон, лежащий дома на
   зарядке, отвечает так же, как из-за калитки. Схема защищает случай «человека
   нет дома».
3. Возвращающийся домой телефон присоединяется к Wi-Fi с задержкой, и у калитки
   покрытие хуже, чем в доме. Поэтому фактор по умолчанию необязателен: сперва
   наблюдение, сколько раз он отвечал в момент прихода.

Недоступность сети даёт «не знаю», а не «за»: молчание никогда не открывает.
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from src.gate.decision import PHONE_FACTOR, Factor

logger = logging.getLogger("catdoor.gate-presence")

# `ip neigh show <адрес>` отвечает строкой вида
# «192.168.2.50 dev wlp2s0 lladdr aa:bb:cc:dd:ee:ff REACHABLE».
_NEIGHBOUR = re.compile(r"lladdr\s+([0-9a-f:]{17})\s+(\w+)", re.IGNORECASE)
# Состояния, при которых ядро считает, что сосед на связи. `FAILED` — ядро
# спросило и не получило ответа: это «против», а не «не знаю».
_PRESENT_STATES = {"REACHABLE", "STALE", "DELAY", "PROBE", "PERMANENT"}
COMMAND_TIMEOUT_SECONDS = 3.0
# Пауза между настоящими опросами одного адреса. Служба смотрит события раз в
# секунду, и без кэша один человек у калитки давал бы шквал процессов.
CACHE_SECONDS = 10.0

Runner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
        check=False,
    )


def phone_factor(
    address: str,
    expected_mac: str,
    *,
    run: Runner | None = None,
) -> Factor:
    """Фактор телефона по одному адресу. Никогда не бросает исключений."""
    runner = run or _run
    if not address or not expected_mac:
        return Factor(PHONE_FACTOR, "unknown", "телефон не задан")
    try:
        # Ответ не важен: он нужен лишь затем, чтобы ядро разрешило MAC.
        runner(["ping", "-c", "1", "-W", "1", address])
        neighbours = runner(["ip", "neigh", "show", address])
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Присутствие телефона не проверено: %s", exc)
        return Factor(PHONE_FACTOR, "unknown", "проверка не выполнилась")
    match = _NEIGHBOUR.search(neighbours.stdout or "")
    if match is None:
        if "FAILED" in (neighbours.stdout or ""):
            return Factor(PHONE_FACTOR, "against", "телефона нет в сети")
        # Записи нет вовсе — ядро ещё не спрашивало или сеть недоступна.
        return Factor(PHONE_FACTOR, "unknown", "адрес не разрешён")
    mac, state = match.group(1).lower(), match.group(2).upper()
    if mac != expected_mac.lower():
        # Молчаливая потеря привязки — худший исход: система просто перестала
        # бы открывать, и никто не понял бы почему. iOS и Android умеют менять
        # приватный MAC периодически, и это ровно тот случай.
        logger.warning(
            "MAC по адресу %s не совпал с ожидаемым — привязка телефона устарела", address
        )
        return Factor(PHONE_FACTOR, "against", "MAC не совпал с ожидаемым")
    if state not in _PRESENT_STATES:
        return Factor(PHONE_FACTOR, "against", "телефона нет в сети")
    return Factor(PHONE_FACTOR, "for", "телефон в домашней сети")


@dataclass
class PhonePresence:
    """Проверка с короткой памятью: тот же адрес не опрашивается чаще, чем раз в паузу."""

    run: Runner | None = None
    cache_seconds: float = CACHE_SECONDS
    clock: Callable[[], float] = time.monotonic
    _cache: dict[str, tuple[float, Factor]] = field(default_factory=dict, repr=False)

    def factor(self, address: str, expected_mac: str) -> Factor:
        now = self.clock()
        cached = self._cache.get(address)
        if cached is not None and now - cached[0] < self.cache_seconds:
            return cached[1]
        result = phone_factor(address, expected_mac, run=self.run)
        self._cache[address] = (now, result)
        return result
