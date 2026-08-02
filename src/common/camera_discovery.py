"""Поиск камер в изолированной сети.

У камеры нет ни экрана, ни известного адреса: после перевключения в PoE-порт
она берёт у dnsmasq любой свободный адрес из пула. Найти её можно четырьмя
независимыми способами, и ни один из них не полон:

- аренды dnsmasq — единственный источник, который знает MAC вместе с адресом,
  но появляется только у камер, которые действительно спросили DHCP;
- ARP-таблица ядра — знает MAC тех, с кем сервер уже разговаривал;
- ONVIF WS-Discovery — отвечают камеры, у которых ONVIF включён, зато они
  сообщают своё имя;
- обход подсети по TCP — находит камеру, которая молчит на всё остальное.

Поэтому источники объединяются, а не выбираются. Ничего из найденного не
применяется само: обнаружение только предлагает, решение остаётся за
администратором.
"""

from __future__ import annotations

import re
import socket
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, replace
from ipaddress import IPv4Address, IPv4Network, ip_network
from pathlib import Path
from uuid import uuid4

from src.common.camera_registry import CameraRecord, CameraRegistry

# Пути, по которым камеры отдают основной поток. Порядок — от того, что стоит
# у владельца, к общим вариантам: первый ответивший и выигрывает, поэтому
# каждая лишняя позиция в списке стоит секунд ожидания на неотвечающем адресе.
RTSP_PATH_CANDIDATES: tuple[str, ...] = (
    "/Streaming/Channels/101",
    "/Streaming/Channels/102",
    "/cam/realmonitor?channel=1&subtype=0",
    "/stream1",
    "/live/ch0",
    "/onvif1",
    "/media/video1",
    "/h264",
)

RTSP_PORT = 554

WS_DISCOVERY_ADDRESS = ("239.255.255.250", 3702)

_MAC = re.compile(r"^[0-9a-f]{2}(:[0-9a-f]{2}){5}$")
_XADDRS = re.compile(r"<[^>]*XAddrs[^>]*>(.*?)</[^>]*XAddrs>", re.DOTALL)
_SCOPES = re.compile(r"<[^>]*Scopes[^>]*>(.*?)</[^>]*Scopes>", re.DOTALL)
_ONVIF_NAME = re.compile(r"onvif://www\.onvif\.org/name/(\S+)")
_HOST_IN_URL = re.compile(r"//(?:[^/@]*@)?([^/:]+)")

_EMPTY_MAC = "00:00:00:00:00:00"


# Насколько источник ручается, что устройство живо прямо сейчас. Открытый
# RTSP-порт — самое сильное свидетельство для камеры; аренда DHCP самое слабое:
# её могли взять один раз при сбросе и уйти на заводской адрес.
_SOURCE_LIVENESS = {"scan": 4, "onvif": 3, "arp": 2, "dhcp": 1}


@dataclass(frozen=True)
class DiscoveredDevice:
    """Устройство, замеченное в сети. Ещё не камера проекта.

    `other_hosts` — прочие адреса того же MAC. Камера после заводского сброса
    успевает засветиться сразу на двух: разово спрашивает DHCP и продолжает
    отвечать на заводском статическом. Прятать это нельзя, но и предлагать
    завести одну железку дважды тоже.
    """

    host: str
    mac: str = ""
    name: str = ""
    sources: tuple[str, ...] = ()
    other_hosts: tuple[str, ...] = ()

    @property
    def liveness(self) -> int:
        return max((_SOURCE_LIVENESS.get(source, 0) for source in self.sources), default=0)

    def merged_with(self, other: DiscoveredDevice) -> DiscoveredDevice:
        return replace(
            self,
            mac=self.mac or other.mac,
            name=self.name or other.name,
            sources=tuple(dict.fromkeys(self.sources + other.sources)),
            other_hosts=tuple(dict.fromkeys(self.other_hosts + other.other_hosts)),
        )


def _normalize_mac(value: str) -> str:
    mac = value.strip().lower().replace("-", ":")
    if not _MAC.match(mac) or mac == _EMPTY_MAC:
        return ""
    return mac


def read_dhcp_leases(path: Path, *, now: float | None = None) -> list[DiscoveredDevice]:
    """Разобрать `dnsmasq.leases`: срок, MAC, адрес, имя, client-id.

    Битая строка пропускается молча. Файл пишет чужая служба, и одна её
    странность не должна отменять поиск целиком.

    С заданным `now` истёкшие аренды отбрасываются. Это важно для автопочинки
    адресов: лежалая запись со старым адресом камеры иначе перезаписала бы
    рабочий адрес мёртвым. Для поиска в админке фильтр не включается — там
    устаревшая строка тоже подсказка, и решение принимает человек.
    """
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    devices: list[DiscoveredDevice] = []
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        if now is not None:
            try:
                if float(parts[0]) < now:
                    continue
            except ValueError:
                continue
        mac = _normalize_mac(parts[1])
        host = parts[2]
        if not mac or not host:
            continue
        name = parts[3] if len(parts) > 3 and parts[3] != "*" else ""
        devices.append(DiscoveredDevice(host=host, mac=mac, name=name, sources=("dhcp",)))
    return devices


def read_arp_table(path: Path) -> list[DiscoveredDevice]:
    """Разобрать `/proc/net/arp`, пропустив незавершённые записи.

    Флаг `0x0` означает, что ядро адрес спрашивало, но ответа не получило:
    такая строка говорит о попытке связаться, а не о найденном устройстве.
    """
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    devices: list[DiscoveredDevice] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 4 or parts[2] == "0x0":
            continue
        mac = _normalize_mac(parts[3])
        if not mac:
            continue
        devices.append(DiscoveredDevice(host=parts[0], mac=mac, sources=("arp",)))
    return devices


def port_is_open(host: str, port: int, timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def local_address_for(host: str) -> str:
    """Адрес сервера в той же сети, что и `host`, без единого отправленного пакета.

    Нужно для WS-Discovery: сокет, оставленный на маршруте по умолчанию, уйдёт
    в сеть управления, а камеры живут в изолированном сегменте на другой карте.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect((host, 9))
        return str(probe.getsockname()[0])
    except OSError:
        return ""
    finally:
        probe.close()


def parse_subnet(value: str) -> IPv4Network | None:
    try:
        network = ip_network(value.strip(), strict=False)
    except ValueError:
        return None
    return network if isinstance(network, IPv4Network) else None


def scan_subnet(
    subnet: IPv4Network,
    *,
    port: int = RTSP_PORT,
    timeout: float = 0.4,
    workers: int = 64,
    max_hosts: int = 1024,
) -> list[DiscoveredDevice]:
    """Постучаться в RTSP-порт каждого адреса подсети.

    Находит камеру, которая не отвечает ни на ONVIF, ни на DHCP — например,
    ту, которой адрес задали руками. Ограничение по числу адресов защищает от
    случайно указанной /8: обойти её не получится всё равно, а ждать придётся
    часами.
    """
    hosts = [str(host) for host in subnet.hosts()][:max_hosts]
    if not hosts:
        return []
    with ThreadPoolExecutor(max_workers=min(workers, len(hosts))) as executor:
        reachable = executor.map(lambda host: port_is_open(host, port, timeout), hosts)
        return [
            DiscoveredDevice(host=host, sources=("scan",))
            for host, is_open in zip(hosts, reachable, strict=True)
            if is_open
        ]


def _probe_message() -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"'
        ' xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"'
        ' xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"'
        ' xmlns:dn="http://www.onvif.org/ver10/network/wsdl">'
        f"<e:Header><w:MessageID>uuid:{uuid4()}</w:MessageID>"
        '<w:To e:mustUnderstand="true">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>'
        '<w:Action e:mustUnderstand="true">'
        "http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action></e:Header>"
        "<e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body>"
        "</e:Envelope>"
    ).encode()


def parse_probe_match(payload: str, sender: str) -> DiscoveredDevice | None:
    """Достать адрес и имя камеры из ответа WS-Discovery.

    Разбор регулярным выражением, а не XML-парсером, намеренно: производители
    расходятся в префиксах пространств имён и порядке элементов, а нужны отсюда
    ровно два поля. Адрес отправителя остаётся запасным вариантом — в XAddrs
    камеры иногда пишут адрес, по которому до них не достучаться.
    """
    hosts = [
        match.group(1)
        for address in _XADDRS.findall(payload)
        for url in address.split()
        if (match := _HOST_IN_URL.search(url))
    ]
    host = next((item for item in hosts if item), sender)
    if not host:
        return None
    scopes = " ".join(_SCOPES.findall(payload))
    name_match = _ONVIF_NAME.search(scopes)
    name = name_match.group(1).replace("%20", " ") if name_match else ""
    return DiscoveredDevice(host=host, name=name, sources=("onvif",))


def ws_discovery(
    *,
    timeout: float = 3.0,
    bind_address: str = "",
    attempts: int = 2,
) -> list[DiscoveredDevice]:
    """Разослать ONVIF-пробу и собрать ответы до истечения таймаута.

    Проба отправляется несколько раз: это UDP на групповой адрес, и потеря
    единственного пакета выглядела бы как отсутствие камер в сети.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except OSError:
        return []
    devices: dict[str, DiscoveredDevice] = {}
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        if bind_address:
            sock.bind((bind_address, 0))
            # bind задаёт адрес источника, но исходящий интерфейс мультикаста
            # на Linux выбирает IP_MULTICAST_IF. Без него проба уходила по
            # маршруту по умолчанию — в домашнюю сеть, и камеры в изолированном
            # сегменте её попросту не видели.
            with suppress(OSError):
                sock.setsockopt(
                    socket.IPPROTO_IP,
                    socket.IP_MULTICAST_IF,
                    socket.inet_aton(bind_address),
                )
        sock.settimeout(min(0.5, max(timeout, 0.05)))
        for _ in range(max(1, attempts)):
            try:
                sock.sendto(_probe_message(), WS_DISCOVERY_ADDRESS)
            except OSError:
                return []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                payload, sender = sock.recvfrom(65535)
            except TimeoutError:
                continue
            except OSError:
                break
            device = parse_probe_match(payload.decode("utf-8", "replace"), sender[0])
            if device is None:
                continue
            devices[device.host] = (
                devices[device.host].merged_with(device)
                if device.host in devices
                else device
            )
    finally:
        sock.close()
    return list(devices.values())


def merge_devices(*groups: list[DiscoveredDevice]) -> list[DiscoveredDevice]:
    """Свести источники в один список: сначала по адресу, затем по MAC.

    Один и тот же адрес приходит из аренд, ARP и обхода подсети; MAC знают
    только первые два, имя — только ONVIF, поэтому поля объединяются, а не
    выбирается победитель.

    Второй проход нужен потому, что одна железка бывает видна сразу на двух
    адресах, и показывать её двумя строками — значит предлагать завести её
    дважды. Из адресов остаётся тот, что живее по источнику: открытый
    RTSP-порт говорит больше, чем разовая аренда.
    """
    by_host: dict[str, DiscoveredDevice] = {}
    for group in groups:
        for device in group:
            existing = by_host.get(device.host)
            by_host[device.host] = (
                existing.merged_with(device) if existing is not None else device
            )
    return sorted(_collapse_by_mac(by_host.values()), key=lambda device: _sort_key(device.host))


def _collapse_by_mac(devices: Iterable[DiscoveredDevice]) -> list[DiscoveredDevice]:
    without_mac = [device for device in devices if not device.mac]
    by_mac: dict[str, list[DiscoveredDevice]] = {}
    for device in devices:
        if device.mac:
            by_mac.setdefault(device.mac, []).append(device)
    collapsed: list[DiscoveredDevice] = []
    for found in by_mac.values():
        # Ключ сортировки с минусом: сначала самый живой, при равенстве —
        # меньший адрес, чтобы порядок не зависел от порядка источников.
        primary, *rest = sorted(found, key=lambda item: (-item.liveness, _sort_key(item.host)))
        for other in rest:
            primary = primary.merged_with(other)
        collapsed.append(
            replace(
                primary,
                other_hosts=tuple(
                    dict.fromkeys(
                        [*primary.other_hosts, *(item.host for item in rest)]
                    )
                ),
            )
        )
    return [*collapsed, *without_mac]


def _sort_key(host: str) -> tuple[int, ...]:
    parts = host.split(".")
    if len(parts) == 4 and all(part.isdigit() for part in parts):
        return tuple(int(part) for part in parts)
    return (256,)


def discover_devices(
    *,
    leases_path: Path,
    arp_path: Path,
    subnet: str = "",
    onvif_timeout: float = 3.0,
    port_timeout: float = 0.4,
) -> list[DiscoveredDevice]:
    """Собрать всё, что видно в сети камер, из всех доступных источников."""
    network = parse_subnet(subnet)
    bind_address = ""
    if network is not None:
        bind_address = local_address_for(str(next(network.hosts(), network.network_address)))
    groups = [
        read_dhcp_leases(leases_path),
        read_arp_table(arp_path),
        ws_discovery(timeout=onvif_timeout, bind_address=bind_address),
    ]
    if network is not None:
        groups.append(scan_subnet(network, timeout=port_timeout))
    devices = merge_devices(*groups)
    if network is not None:
        devices = [device for device in devices if _in_network(device.host, network)]
    return [device for device in devices if device.host != bind_address]


def _in_network(host: str, network: IPv4Network) -> bool:
    try:
        return IPv4Address(host) in network
    except ValueError:
        return False


def match_camera(registry: CameraRegistry, device: DiscoveredDevice) -> CameraRecord | None:
    """Узнать в найденном устройстве уже заведённую камеру.

    Сначала по MAC: он и есть то, что не меняется при перестановке, ради чего
    поиск затевался. Адрес — запасной признак: он совпадает у камеры, которая
    никуда не переезжала, но у переставленной он уже чужой, и полагаться на
    него одного нельзя.
    """
    if device.mac:
        by_mac = next(
            (camera for camera in registry.cameras if camera.mac == device.mac),
            None,
        )
        if by_mac is not None:
            return by_mac
    if not device.host:
        return None
    return next(
        (camera for camera in registry.cameras if camera.host == device.host),
        None,
    )


def addresses_from_leases(
    registry: CameraRegistry,
    devices: Sequence[DiscoveredDevice],
    camera_ids: Iterable[str],
) -> dict[str, str]:
    """Новые адреса для перечисленных камер: `{идентификатор: адрес}`.

    Камера с известным MAC, обнаруженная на другом адресе, — это она же после
    новой аренды, других толкований нет. Возвращается только адрес: пароль,
    роль и путь потока к смене аренды отношения не имеют, и трогать их значило
    бы чинить не то, что сломалось.
    """
    by_mac = {device.mac: device.host for device in devices if device.mac and device.host}
    updates: dict[str, str] = {}
    for camera_id in camera_ids:
        camera = registry.get(camera_id)
        if camera is None or not camera.mac:
            continue
        host = by_mac.get(camera.mac)
        if host and host != camera.host:
            updates[camera.id] = host
    return updates


def is_authentication_failure(reason: str) -> bool:
    lowered = reason.lower()
    return "401" in lowered or "unauthorized" in lowered or "authoriz" in lowered


def candidate_paths(known: str = "") -> tuple[str, ...]:
    """Путь известной камеры пробуется первым, остальные — следом.

    Переставленная камера почти всегда та же самая модель с тем же путём, а
    каждая неудачная попытка стоит секунд ожидания ответа.
    """
    if not known:
        return RTSP_PATH_CANDIDATES
    return (known, *(path for path in RTSP_PATH_CANDIDATES if path != known))


@dataclass(frozen=True)
class StreamGuess:
    stream_path: str = ""
    reason: str = ""
    needs_credentials: bool = False

    @property
    def found(self) -> bool:
        return bool(self.stream_path)


def identify_stream_path(
    host: str,
    *,
    probe: Callable[[str], str | None],
    port: int = RTSP_PORT,
    username: str = "",
    password: str = "",
    paths: Sequence[str] = RTSP_PATH_CANDIDATES,
) -> StreamGuess:
    """Подобрать рабочий путь потока, перебирая варианты по одному.

    Перебор прекращается на отказе авторизации: пароль от пути не зависит, и
    остальные попытки только потратят время администратора, который ждёт ответ
    страницы.
    """
    last_reason = ""
    for path in paths:
        candidate = CameraRecord(
            id="discovery-probe",
            host=host,
            port=port,
            stream_path=path,
            username=username,
            password=password,
        )
        reason = probe(candidate.rtsp_url())
        if reason is None:
            return StreamGuess(stream_path=path)
        last_reason = reason
        if is_authentication_failure(reason):
            return StreamGuess(reason=reason, needs_credentials=True)
    return StreamGuess(reason=last_reason)
