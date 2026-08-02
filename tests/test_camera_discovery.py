from pathlib import Path

from src.common.camera_discovery import (
    RTSP_PATH_CANDIDATES,
    DiscoveredDevice,
    addresses_from_leases,
    candidate_paths,
    identify_stream_path,
    is_authentication_failure,
    match_camera,
    merge_devices,
    parse_probe_match,
    parse_subnet,
    read_arp_table,
    read_dhcp_leases,
)
from src.common.camera_registry import CameraRecord, CameraRegistry

PROBE_MATCH = """<?xml version="1.0" encoding="UTF-8"?>
<SOAP-ENV:Envelope xmlns:SOAP-ENV="http://www.w3.org/2003/05/soap-envelope"
  xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery">
  <SOAP-ENV:Body><d:ProbeMatches><d:ProbeMatch>
    <d:Scopes>onvif://www.onvif.org/type/video_encoder
      onvif://www.onvif.org/name/HiWatch%20DS-T200
      onvif://www.onvif.org/location/porch</d:Scopes>
    <d:XAddrs>http://192.168.1.117/onvif/device_service</d:XAddrs>
  </d:ProbeMatch></d:ProbeMatches></SOAP-ENV:Body>
</SOAP-ENV:Envelope>"""


def test_dhcp_leases_give_mac_and_address(runtime_path: Path) -> None:
    path = runtime_path / "dnsmasq.leases"
    path.write_text(
        "1785600000 44:47:cc:11:22:33 192.168.1.117 HIWATCH 01:44:47:cc:11:22:33\n"
        "1785600000 aa:bb:cc:dd:ee:ff 192.168.1.118 * *\n"
        "мусор\n",
        encoding="utf-8",
    )

    devices = read_dhcp_leases(path)

    assert [device.host for device in devices] == ["192.168.1.117", "192.168.1.118"]
    assert devices[0].mac == "44:47:cc:11:22:33"
    assert devices[0].name == "HIWATCH"
    assert devices[1].name == ""
    assert devices[0].sources == ("dhcp",)


def test_missing_leases_file_is_not_an_error(runtime_path: Path) -> None:
    """Отсутствие dnsmasq — обычное состояние, а не отказ поиска."""
    assert read_dhcp_leases(runtime_path / "нет-такого") == []
    assert read_arp_table(runtime_path / "нет-такого") == []


def test_arp_table_skips_unanswered_entries(runtime_path: Path) -> None:
    path = runtime_path / "arp"
    path.write_text(
        "IP address       HW type     Flags       HW address            Mask     Device\n"
        "192.168.1.117    0x1         0x2         44:47:cc:11:22:33     *        eno1\n"
        "192.168.1.150    0x1         0x0         00:00:00:00:00:00     *        eno1\n",
        encoding="utf-8",
    )

    devices = read_arp_table(path)

    assert [device.host for device in devices] == ["192.168.1.117"]
    assert devices[0].mac == "44:47:cc:11:22:33"


def test_sources_are_merged_not_chosen() -> None:
    """У каждого источника своя половина сведений: MAC у DHCP, имя у ONVIF."""
    merged = merge_devices(
        [DiscoveredDevice(host="192.168.1.117", mac="44:47:cc:11:22:33", sources=("dhcp",))],
        [DiscoveredDevice(host="192.168.1.117", name="HiWatch", sources=("onvif",))],
        [DiscoveredDevice(host="192.168.1.9", sources=("scan",))],
    )

    assert [device.host for device in merged] == ["192.168.1.9", "192.168.1.117"]
    camera = merged[1]
    assert camera.mac == "44:47:cc:11:22:33"
    assert camera.name == "HiWatch"
    assert camera.sources == ("dhcp", "onvif")


def test_probe_match_yields_address_and_name() -> None:
    device = parse_probe_match(PROBE_MATCH, "192.168.1.117")

    assert device is not None
    assert device.host == "192.168.1.117"
    assert device.name == "HiWatch DS-T200"
    assert device.sources == ("onvif",)


def test_probe_match_falls_back_to_the_sender() -> None:
    """В XAddrs камеры пишут и адреса, по которым до них не достучаться."""
    device = parse_probe_match("<d:ProbeMatches></d:ProbeMatches>", "192.168.1.130")

    assert device is not None
    assert device.host == "192.168.1.130"


def test_subnet_is_parsed_leniently_and_rejects_nonsense() -> None:
    assert parse_subnet("192.168.1.0/24") is not None
    assert parse_subnet(" 192.168.1.10/24 ") is not None
    assert parse_subnet("не подсеть") is None
    assert parse_subnet("") is None


def test_camera_is_recognised_by_mac_after_it_changed_address() -> None:
    """Ради этого и заводился MAC: адрес у переставленной камеры уже чужой."""
    registry = CameraRegistry(
        cameras=[
            CameraRecord(id="hiwatch", host="192.168.1.101", mac="44:47:cc:11:22:33"),
            CameraRecord(id="ursa", host="192.168.1.117"),
        ]
    )

    matched = match_camera(
        registry,
        DiscoveredDevice(host="192.168.1.117", mac="44:47:cc:11:22:33"),
    )

    assert matched is not None
    assert matched.id == "hiwatch"


def test_camera_without_mac_is_recognised_by_address() -> None:
    registry = CameraRegistry(cameras=[CameraRecord(id="ursa", host="192.168.1.117")])

    matched = match_camera(registry, DiscoveredDevice(host="192.168.1.117"))

    assert matched is not None
    assert matched.id == "ursa"
    assert match_camera(registry, DiscoveredDevice(host="192.168.1.200")) is None


def test_known_path_is_tried_first() -> None:
    paths = candidate_paths("/stream1")

    assert paths[0] == "/stream1"
    assert sorted(paths) == sorted(RTSP_PATH_CANDIDATES)
    assert candidate_paths() == RTSP_PATH_CANDIDATES


def test_stream_path_is_picked_by_the_first_answering_candidate() -> None:
    tried: list[str] = []

    def probe(url: str) -> str | None:
        tried.append(url)
        return None if url.endswith("/stream1") else "404 Not Found"

    guess = identify_stream_path(
        "192.168.1.117",
        probe=probe,
        username="admin",
        password="secret",
        paths=("/one", "/stream1", "/two"),
    )

    assert guess.stream_path == "/stream1"
    assert guess.found is True
    assert len(tried) == 2
    assert tried[0] == "rtsp://admin:secret@192.168.1.117:554/one"


def test_wrong_password_stops_the_search_instead_of_trying_every_path() -> None:
    """Пароль от пути не зависит, а администратор ждёт ответ страницы."""
    tried: list[str] = []

    def probe(url: str) -> str | None:
        tried.append(url)
        return "401 Unauthorized"

    guess = identify_stream_path("192.168.1.117", probe=probe, paths=("/one", "/two", "/three"))

    assert guess.found is False
    assert guess.needs_credentials is True
    assert len(tried) == 1
    assert is_authentication_failure("401 Unauthorized") is True
    assert is_authentication_failure("Connection timed out") is False


def test_unreachable_camera_reports_the_last_reason() -> None:
    guess = identify_stream_path(
        "192.168.1.117",
        probe=lambda _url: "камера не ответила за 6 с",
        paths=("/one", "/two"),
    )

    assert guess.found is False
    assert guess.needs_credentials is False
    assert guess.reason == "камера не ответила за 6 с"


def test_moved_camera_is_matched_to_its_new_lease() -> None:
    registry = CameraRegistry(
        cameras=[
            CameraRecord(id="hiwatch", host="192.168.1.101", mac="44:47:cc:11:22:33"),
            CameraRecord(id="ursa", host="192.168.1.102", mac="aa:bb:cc:dd:ee:ff"),
        ]
    )
    devices = [
        DiscoveredDevice(host="192.168.1.137", mac="44:47:cc:11:22:33"),
        DiscoveredDevice(host="192.168.1.102", mac="aa:bb:cc:dd:ee:ff"),
    ]

    updates = addresses_from_leases(registry, devices, ["hiwatch", "ursa"])

    # Только уехавшая: камера на прежнем адресе в правке не нуждается.
    assert updates == {"hiwatch": "192.168.1.137"}


def test_only_the_named_cameras_are_considered() -> None:
    """Правится адрес только у того, кто молчит, а не у всех разом."""
    registry = CameraRegistry(
        cameras=[CameraRecord(id="hiwatch", host="192.168.1.101", mac="44:47:cc:11:22:33")]
    )
    devices = [DiscoveredDevice(host="192.168.1.137", mac="44:47:cc:11:22:33")]

    assert addresses_from_leases(registry, devices, []) == {}
    assert addresses_from_leases(registry, devices, ["ursa"]) == {}


def test_one_camera_on_two_addresses_becomes_one_row() -> None:
    """После заводского сброса камера видна и по аренде, и по заводскому адресу.

    Двумя строками она предлагала бы завести одну железку дважды.
    """
    merged = merge_devices(
        [DiscoveredDevice(host="192.168.1.106", mac="9c:14:63:41:de:15", sources=("dhcp",))],
        [DiscoveredDevice(host="192.168.1.108", mac="9c:14:63:41:de:15", sources=("arp", "scan"))],
    )

    assert len(merged) == 1
    # Остаётся адрес, на котором открыт RTSP, а не тот, что однажды выдал DHCP.
    assert merged[0].host == "192.168.1.108"
    assert merged[0].other_hosts == ("192.168.1.106",)
    assert set(merged[0].sources) == {"dhcp", "arp", "scan"}


def test_the_liveliest_source_decides_which_address_survives() -> None:
    order = [
        DiscoveredDevice(host="192.168.1.10", mac="aa:bb:cc:dd:ee:ff", sources=("dhcp",)),
        DiscoveredDevice(host="192.168.1.11", mac="aa:bb:cc:dd:ee:ff", sources=("arp",)),
        DiscoveredDevice(host="192.168.1.12", mac="aa:bb:cc:dd:ee:ff", sources=("onvif",)),
    ]

    merged = merge_devices(order)

    assert merged[0].host == "192.168.1.12"
    assert sorted(merged[0].other_hosts) == ["192.168.1.10", "192.168.1.11"]


def test_devices_without_a_mac_keep_their_own_rows() -> None:
    """Без MAC совпадение недоказуемо: два адреса могут быть двумя железками."""
    merged = merge_devices(
        [DiscoveredDevice(host="192.168.1.20", sources=("scan",))],
        [DiscoveredDevice(host="192.168.1.21", sources=("scan",))],
    )

    assert [device.host for device in merged] == ["192.168.1.20", "192.168.1.21"]


def test_the_same_address_from_several_sources_is_still_one_row() -> None:
    merged = merge_devices(
        [DiscoveredDevice(host="192.168.1.30", mac="aa:bb:cc:dd:ee:ff", sources=("dhcp",))],
        [DiscoveredDevice(host="192.168.1.30", name="HiWatch", sources=("onvif",))],
    )

    assert len(merged) == 1
    assert merged[0].name == "HiWatch"
    assert merged[0].other_hosts == ()


def test_expired_leases_are_filtered_when_now_is_given(runtime_path: Path) -> None:
    """Для автопочинки лежалая аренда — не подсказка, а способ навредить."""
    path = runtime_path / "dnsmasq.leases"
    path.write_text(
        "1000000000 44:47:cc:11:22:33 192.168.1.137 OLD *\n"
        "9999999999 aa:bb:cc:dd:ee:ff 192.168.1.140 FRESH *\n",
        encoding="utf-8",
    )

    unfiltered = read_dhcp_leases(path)
    filtered = read_dhcp_leases(path, now=2_000_000_000)

    assert [device.host for device in unfiltered] == ["192.168.1.137", "192.168.1.140"]
    assert [device.host for device in filtered] == ["192.168.1.140"]
