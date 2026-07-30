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


def prepare_detector_frame(frame: np.ndarray) -> np.ndarray:
    image = Image.fromarray(crop_frame_padding(frame), mode="RGB")
    image.thumbnail(
        (DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE),
        Image.Resampling.BILINEAR,
    )
    detector_image = Image.new("RGB", (DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE))
    detector_image.paste(
        image,
        (
            (DETECTOR_INPUT_SIZE - image.width) // 2,
            (DETECTOR_INPUT_SIZE - image.height) // 2,
        ),
    )
    return np.asarray(detector_image, dtype=np.uint8)


def map_detector_box_to_frame(
    box: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
) -> tuple[float, float, float, float]:
    scale = min(
        DETECTOR_INPUT_SIZE / frame_width,
        DETECTOR_INPUT_SIZE / frame_height,
    )
    offset_x = (DETECTOR_INPUT_SIZE - frame_width * scale) / 2
    offset_y = (DETECTOR_INPUT_SIZE - frame_height * scale) / 2
    x1, y1, x2, y2 = box
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
) -> tuple[float, float, float, float]:
    detector_coordinates = map_detector_box_to_frame(
        box,
        detector_frame_width,
        detector_frame_height,
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
