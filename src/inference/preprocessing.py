from __future__ import annotations

import math
from typing import Protocol

import numpy as np
from PIL import Image

from src.inference.cat_detector import CatDetection

DETECTOR_INPUT_SIZE = 320
IDENTITY_CROP_EXPANSION = 0.20
IDENTITY_NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
IDENTITY_NORMALIZATION_STD = (0.229, 0.224, 0.225)


class NormalizedRoi(Protocol):
    x: float
    y: float
    width: float
    height: float


def crop_frame_padding(frame: np.ndarray) -> np.ndarray:
    active_rows = np.flatnonzero(np.any(frame != 0, axis=(1, 2)))
    active_columns = np.flatnonzero(np.any(frame != 0, axis=(0, 2)))
    if active_rows.size == 0 or active_columns.size == 0:
        return frame
    return frame[
        active_rows[0] : active_rows[-1] + 1,
        active_columns[0] : active_columns[-1] + 1,
    ]


def crop_normalized_roi(
    frame: np.ndarray,
    roi: NormalizedRoi | None,
) -> np.ndarray:
    source = crop_frame_padding(frame)
    if roi is None:
        return source
    left = max(0, math.floor(source.shape[1] * roi.x))
    top = max(0, math.floor(source.shape[0] * roi.y))
    right = min(
        source.shape[1],
        math.ceil(source.shape[1] * (roi.x + roi.width)),
    )
    bottom = min(
        source.shape[0],
        math.ceil(source.shape[0] * (roi.y + roi.height)),
    )
    return np.ascontiguousarray(source[top:bottom, left:right])


def prepare_detector_frame(
    frame: np.ndarray,
    input_size: int = DETECTOR_INPUT_SIZE,
    letterbox_value: int = 0,
    resize_mode: str = "letterbox",
    letterbox_alignment: str = "center",
) -> np.ndarray:
    if input_size <= 0:
        raise ValueError("detector input size must be positive")
    if not 0 <= letterbox_value <= 255:
        raise ValueError("detector letterbox value must be between 0 and 255")
    image = Image.fromarray(crop_frame_padding(frame), mode="RGB")
    if resize_mode == "stretch":
        return np.asarray(
            image.resize(
                (input_size, input_size),
                Image.Resampling.BILINEAR,
            ),
            dtype=np.uint8,
        )
    if resize_mode != "letterbox":
        raise ValueError("detector resize mode must be letterbox or stretch")
    if letterbox_alignment not in {"center", "top_left"}:
        raise ValueError("detector letterbox alignment must be center or top_left")
    scale = min(input_size / image.width, input_size / image.height)
    resized_width = max(1, min(input_size, int(image.width * scale)))
    resized_height = max(1, min(input_size, int(image.height * scale)))
    image = image.resize(
        (resized_width, resized_height),
        Image.Resampling.BILINEAR,
    )
    detector_image = Image.new(
        "RGB",
        (input_size, input_size),
        color=(letterbox_value,) * 3,
    )
    offset = (
        (0, 0)
        if letterbox_alignment == "top_left"
        else (
            (input_size - image.width) // 2,
            (input_size - image.height) // 2,
        )
    )
    detector_image.paste(image, offset)
    return np.asarray(detector_image, dtype=np.uint8)


def map_detector_box_to_frame(
    box: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
    input_size: int = DETECTOR_INPUT_SIZE,
    resize_mode: str = "letterbox",
    letterbox_alignment: str = "center",
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    if resize_mode == "stretch":
        return (
            max(0.0, min(float(frame_width), x1 / input_size * frame_width)),
            max(0.0, min(float(frame_height), y1 / input_size * frame_height)),
            max(0.0, min(float(frame_width), x2 / input_size * frame_width)),
            max(0.0, min(float(frame_height), y2 / input_size * frame_height)),
        )
    if resize_mode != "letterbox":
        raise ValueError("detector resize mode must be letterbox or stretch")
    if letterbox_alignment not in {"center", "top_left"}:
        raise ValueError("detector letterbox alignment must be center or top_left")
    scale = min(
        input_size / frame_width,
        input_size / frame_height,
    )
    offset_x = (
        0.0
        if letterbox_alignment == "top_left"
        else (input_size - frame_width * scale) / 2
    )
    offset_y = (
        0.0
        if letterbox_alignment == "top_left"
        else (input_size - frame_height * scale) / 2
    )
    return (
        max(0.0, min(float(frame_width), (x1 - offset_x) / scale)),
        max(0.0, min(float(frame_height), (y1 - offset_y) / scale)),
        max(0.0, min(float(frame_width), (x2 - offset_x) / scale)),
        max(0.0, min(float(frame_height), (y2 - offset_y) / scale)),
    )


def map_detector_box_to_source(
    box: tuple[float, float, float, float],
    source_width: int,
    source_height: int,
    detector_frame_width: int,
    detector_frame_height: int,
    *,
    roi: NormalizedRoi | None,
    input_size: int = DETECTOR_INPUT_SIZE,
    resize_mode: str = "letterbox",
    letterbox_alignment: str = "center",
) -> tuple[float, float, float, float]:
    detector_coordinates = map_detector_box_to_frame(
        box,
        detector_frame_width,
        detector_frame_height,
        input_size,
        resize_mode,
        letterbox_alignment,
    )
    roi_x = roi.x if roi is not None else 0.0
    roi_y = roi.y if roi is not None else 0.0
    roi_width = roi.width if roi is not None else 1.0
    roi_height = roi.height if roi is not None else 1.0
    return (
        source_width
        * (roi_x + detector_coordinates[0] / detector_frame_width * roi_width),
        source_height
        * (roi_y + detector_coordinates[1] / detector_frame_height * roi_height),
        source_width
        * (roi_x + detector_coordinates[2] / detector_frame_width * roi_width),
        source_height
        * (roi_y + detector_coordinates[3] / detector_frame_height * roi_height),
    )


def crop_strongest_cat(
    frame: np.ndarray,
    detections: tuple[CatDetection, ...],
    *,
    min_confidence: float,
    expansion: float = IDENTITY_CROP_EXPANSION,
    detector_frame: np.ndarray | None = None,
    roi: NormalizedRoi | None = None,
    input_size: int = DETECTOR_INPUT_SIZE,
    resize_mode: str = "letterbox",
    letterbox_alignment: str = "center",
) -> np.ndarray | None:
    eligible = tuple(
        detection
        for detection in detections
        if detection.confidence >= min_confidence
    )
    if not eligible:
        return None
    strongest = max(eligible, key=lambda detection: detection.confidence)
    source = crop_frame_padding(frame)
    detector_source = crop_frame_padding(
        detector_frame if detector_frame is not None else frame
    )
    x1, y1, x2, y2 = map_detector_box_to_source(
        strongest.box,
        source.shape[1],
        source.shape[0],
        detector_source.shape[1],
        detector_source.shape[0],
        roi=roi,
        input_size=input_size,
        resize_mode=resize_mode,
        letterbox_alignment=letterbox_alignment,
    )
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return None
    left = max(0, math.floor(x1 - width * expansion))
    top = max(0, math.floor(y1 - height * expansion))
    right = min(source.shape[1], math.ceil(x2 + width * expansion))
    bottom = min(source.shape[0], math.ceil(y2 + height * expansion))
    if right <= left or bottom <= top:
        return None
    return np.ascontiguousarray(source[top:bottom, left:right])


def prepare_identity_tensor(
    frame: np.ndarray,
    image_size: int,
    *,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> np.ndarray:
    image = Image.fromarray(frame, mode="RGB").resize(
        (image_size, image_size),
        Image.Resampling.BILINEAR,
    )
    tensor = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None, ...] / 255.0
    normalization_mean = (
        mean
        if mean is not None
        else np.asarray(IDENTITY_NORMALIZATION_MEAN, dtype=np.float32).reshape(
            1, 3, 1, 1
        )
    )
    normalization_std = (
        std
        if std is not None
        else np.asarray(IDENTITY_NORMALIZATION_STD, dtype=np.float32).reshape(
            1, 3, 1, 1
        )
    )
    return (tensor - normalization_mean) / normalization_std
