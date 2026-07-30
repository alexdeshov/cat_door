import numpy as np

from src.inference.cat_detector import CatDetection
from src.inference.preprocessing import (
    crop_strongest_cat,
    map_detector_box_to_frame,
    prepare_identity_tensor,
)


def test_detector_box_maps_out_of_letterbox_coordinates() -> None:
    coordinates = map_detector_box_to_frame(
        (80.0, 115.0, 240.0, 205.0),
        640,
        360,
    )

    assert coordinates == (160.0, 90.0, 480.0, 270.0)


def test_identity_crop_uses_strongest_box_and_twenty_percent_expansion() -> None:
    frame = np.full((360, 640, 3), 100, dtype=np.uint8)
    detections = (
        CatDetection(box=(120.0, 135.0, 200.0, 185.0), confidence=0.60),
        CatDetection(box=(80.0, 115.0, 240.0, 205.0), confidence=0.80),
    )

    crop = crop_strongest_cat(frame, detections, min_confidence=0.55)

    assert crop is not None
    assert crop.shape == (252, 448, 3)
    assert np.all(crop == 100)


def test_identity_tensor_uses_shared_resize_and_normalization_contract() -> None:
    frame = np.full((20, 40, 3), 255, dtype=np.uint8)

    tensor = prepare_identity_tensor(frame, 16)

    assert tensor.shape == (1, 3, 16, 16)
    assert tensor.dtype == np.float32
    assert float(tensor[0, 0, 0, 0]) > 2.2
