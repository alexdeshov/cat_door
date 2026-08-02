import json
import stat
from pathlib import Path

import pytest

from src.common.camera_registry import (
    CameraRecord,
    CameraRegistry,
    camera_from_url,
    camera_registry_path,
    ensure_registry,
    load_registry,
    registry_from_settings,
    save_registry,
    slugify_camera_id,
)
from src.common.settings import Settings


def test_rtsp_url_escapes_credentials() -> None:
    camera = CameraRecord(
        id="hiwatch",
        host="192.168.1.101",
        port=554,
        stream_path="/Streaming/Channels/101",
        username="admin",
        password="p@ss/word",
    )

    assert camera.rtsp_url() == (
        "rtsp://admin:p%40ss%2Fword@192.168.1.101:554/Streaming/Channels/101"
    )


def test_camera_without_host_has_no_url_but_stays_a_record() -> None:
    camera = CameraRecord(id="spare", title="Запасная")

    assert camera.rtsp_url() == ""
    assert camera.is_addressable is False
    assert camera.display_title == "Запасная"


def test_camera_from_url_keeps_query_string() -> None:
    camera = camera_from_url(
        "side",
        "rtsp://user:secret@camera.local:8554/cam/realmonitor?subtype=0",
        role="side",
    )

    assert camera.host == "camera.local"
    assert camera.port == 8554
    assert camera.stream_path == "/cam/realmonitor?subtype=0"
    assert camera.username == "user"
    assert camera.password == "secret"
    assert camera.rtsp_url() == "rtsp://user:secret@camera.local:8554/cam/realmonitor?subtype=0"


def test_camera_id_and_mac_are_validated() -> None:
    with pytest.raises(ValueError):
        CameraRecord(id="Камера")
    with pytest.raises(ValueError):
        CameraRecord(id="cam", mac="not-a-mac")

    assert CameraRecord(id="cam", mac="AA-BB-CC-DD-EE-FF").mac == "aa:bb:cc:dd:ee:ff"
    assert slugify_camera_id("HiWatch у столба #2") == "hiwatch-2"


def test_resolve_prefers_role_over_matching_id() -> None:
    registry = CameraRegistry(
        cameras=[
            CameraRecord(id="face", host="old.local"),
            CameraRecord(id="hiwatch", host="new.local", role="face"),
        ]
    )

    assert registry.resolve("face").id == "hiwatch"
    assert registry.resolve("hiwatch").id == "hiwatch"
    assert registry.resolve("unknown") is None


def test_assigning_a_taken_role_frees_the_previous_owner() -> None:
    registry = CameraRegistry(
        cameras=[
            CameraRecord(id="ursa", host="a.local", role="face"),
            CameraRecord(id="hiwatch", host="b.local", role="side"),
        ]
    )

    updated = registry.with_camera(
        registry.get("hiwatch").model_copy(update={"role": "face"})
    )

    assert updated.by_role("face").id == "hiwatch"
    assert updated.get("ursa").role is None
    assert [camera.id for camera in updated.cameras] == ["ursa", "hiwatch"]


def test_swap_roles_exchanges_face_and_side() -> None:
    registry = CameraRegistry(
        cameras=[
            CameraRecord(id="ursa", host="a.local", role="face"),
            CameraRecord(id="hiwatch", host="b.local", role="side"),
            CameraRecord(id="yard", host="c.local"),
        ]
    )

    swapped = registry.with_swapped_roles()

    assert swapped.by_role("face").id == "hiwatch"
    assert swapped.by_role("side").id == "ursa"
    assert swapped.get("yard").role is None


def test_duplicate_role_in_a_hand_edited_file_is_defused_not_fatal() -> None:
    registry = CameraRegistry(
        cameras=[
            CameraRecord(id="one", role="face"),
            CameraRecord(id="two", role="face"),
            CameraRecord(id="one", role="side"),
        ]
    )

    assert [camera.id for camera in registry.cameras] == ["one", "two"]
    assert registry.get("two").role is None


def test_registry_seeds_from_environment_including_unconfigured_roles() -> None:
    settings = Settings(
        camera_face_rtsp="rtsp://admin:secret@10.0.0.5:554/Streaming/Channels/101",
        camera_side_rtsp="",
        camera_face_rotate_180=True,
        camera_side_rotate_180=False,
        cameras=[{"id": "yard", "rtsp": "rtsp://10.0.0.9:554/stream1", "title": "Двор"}],
    )

    registry = registry_from_settings(settings)

    face = registry.by_role("face")
    assert face.host == "10.0.0.5"
    assert face.password == "secret"
    assert face.rotate_180 is True
    side = registry.by_role("side")
    assert side is not None
    assert side.host == ""
    assert registry.get("yard").host == "10.0.0.9"


def test_extra_camera_with_a_role_id_overrides_that_address() -> None:
    settings = Settings(
        camera_face_rtsp="rtsp://admin:secret@10.0.0.5:554/old",
        cameras=[{"id": "face", "rtsp": "rtsp://admin:secret@10.0.0.6:554/new"}],
    )

    registry = registry_from_settings(settings)

    assert registry.by_role("face").host == "10.0.0.6"
    assert len(registry.cameras) == 2


def test_saved_registry_is_owner_only_and_round_trips(runtime_path: Path) -> None:
    path = camera_registry_path(runtime_path)
    registry = CameraRegistry(
        cameras=[CameraRecord(id="hiwatch", host="10.0.0.5", password="secret", role="face")]
    )

    save_registry(path, registry)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert load_registry(path) == registry


def test_ensure_registry_seeds_once_and_keeps_later_edits(runtime_path: Path) -> None:
    settings = Settings(
        camera_face_rtsp="rtsp://admin:secret@10.0.0.5:554/one",
        data_dir=runtime_path,
    )
    path = camera_registry_path(runtime_path)

    first = ensure_registry(path, settings)
    save_registry(path, first.with_camera(first.by_role("face").model_copy(update={"host": "10.0.0.7"})))
    second = ensure_registry(path, settings)

    assert second.by_role("face").host == "10.0.0.7"


def test_broken_registry_is_moved_aside_instead_of_blocking_startup(runtime_path: Path) -> None:
    settings = Settings(camera_face_rtsp="rtsp://10.0.0.5:554/one", data_dir=runtime_path)
    path = camera_registry_path(runtime_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ это не json", encoding="utf-8")

    registry = ensure_registry(path, settings)

    assert registry.by_role("face").host == "10.0.0.5"
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1
    assert list(path.parent.glob("cameras.json.invalid-*"))


def test_role_belongs_to_the_cat_circuit_and_says_so() -> None:
    """`face` и `side` — понятия кошачьего контура, не общие.

    Контур подставляется, а не требуется отдельно: иначе список позволял бы
    завести камеру калитки с ролью детектора котов.
    """
    assert CameraRecord(id="hiwatch", role="face").circuit == "cats"
    assert CameraRecord(id="gate", circuit="gate", role="side").circuit == "cats"


def test_a_camera_without_a_role_keeps_the_circuit_it_was_given() -> None:
    assert CameraRecord(id="gate", circuit="gate").circuit == "gate"
    assert CameraRecord(id="spare").circuit is None


def test_seeded_roles_belong_to_the_cat_circuit() -> None:
    settings = Settings(camera_face_rtsp="rtsp://10.0.0.5:554/one")

    registry = registry_from_settings(settings)

    assert {camera.circuit for camera in registry.cameras} == {"cats"}
