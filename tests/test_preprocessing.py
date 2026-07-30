import numpy as np

from src.common.detection_config import DetectorRoi
from src.inference.cat_detector import CatDetection
from src.inference.preprocessing import (
    crop_normalized_roi,
    crop_strongest_cat,
    map_detector_box_to_frame,
    map_detector_box_to_source,
    prepare_detector_frame,
    prepare_identity_tensor,
)


def test_detector_box_maps_out_of_letterbox_coordinates() -> None:
    coordinates = map_detector_box_to_frame(
        (80.0, 115.0, 240.0, 205.0),
        640,
        360,
    )

    assert coordinates == (160.0, 90.0, 480.0, 270.0)


def test_detector_preprocessing_and_mapping_support_416_input() -> None:
    frame = np.full((360, 640, 3), 100, dtype=np.uint8)

    detector_frame = prepare_detector_frame(frame, 416)
    coordinates = map_detector_box_to_frame(
        (104.0, 149.5, 312.0, 266.5),
        640,
        360,
        416,
    )

    assert detector_frame.shape == (416, 416, 3)
    assert coordinates == (160.0, 90.0, 480.0, 270.0)


def test_detector_preprocessing_uses_model_letterbox_value() -> None:
    frame = np.full((100, 200, 3), 50, dtype=np.uint8)

    detector_frame = prepare_detector_frame(frame, 416, 114)

    assert np.all(detector_frame[0] == 114)
    assert np.all(detector_frame[208, 208] == 50)


def test_top_left_letterbox_mapping_matches_yolox_preprocessing() -> None:
    frame = np.full((360, 640, 3), 50, dtype=np.uint8)

    detector_frame = prepare_detector_frame(
        frame,
        416,
        114,
        "letterbox",
        "top_left",
    )
    coordinates = map_detector_box_to_frame(
        (104.0, 58.5, 312.0, 175.5),
        640,
        360,
        416,
        "letterbox",
        "top_left",
    )

    assert np.all(detector_frame[0, 0] == 50)
    assert np.all(detector_frame[-1] == 114)
    assert coordinates == (160.0, 90.0, 480.0, 270.0)


def test_stretch_mapping_matches_nanodet_preprocessing() -> None:
    frame = np.full((360, 640, 3), 50, dtype=np.uint8)

    detector_frame = prepare_detector_frame(frame, 416, 0, "stretch")
    coordinates = map_detector_box_to_frame(
        (104.0, 104.0, 312.0, 312.0),
        640,
        360,
        416,
        "stretch",
    )

    assert detector_frame.shape == (416, 416, 3)
    assert np.all(detector_frame == 50)
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


def test_roi_detection_maps_back_to_full_frame_and_identity_crop() -> None:
    roi = DetectorRoi(x=0.25, y=0.25, width=0.5, height=0.5)
    coordinates = map_detector_box_to_source(
        (80.0, 115.0, 240.0, 205.0),
        640,
        360,
        640,
        360,
        roi=roi,
    )
    frame = np.full((360, 640, 3), 100, dtype=np.uint8)
    detector_frame = crop_normalized_roi(frame, roi)
    crop = crop_strongest_cat(
        frame,
        (CatDetection(box=(80.0, 115.0, 240.0, 205.0), confidence=0.8),),
        min_confidence=0.55,
        detector_frame=detector_frame,
        roi=roi,
    )

    assert coordinates == (240.0, 135.0, 400.0, 225.0)
    assert detector_frame.shape == (180, 320, 3)
    assert crop is not None
    assert crop.shape == (126, 224, 3)
