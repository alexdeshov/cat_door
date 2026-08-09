from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class DetectorOutputError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedDetection:
    box: tuple[float, float, float, float]
    confidence: float


class DetectorOutputAdapter(Protocol):
    def parse(
        self,
        outputs: Mapping[str, np.ndarray],
    ) -> tuple[ParsedDetection, ...]: ...


def _output(outputs: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    try:
        return np.asarray(outputs[name])
    except KeyError as exc:
        raise DetectorOutputError(f"detector output {name!r} is missing") from exc


def _rows(array: np.ndarray, columns: int | None = None) -> np.ndarray:
    rows = np.asarray(array)
    if rows.ndim == 3 and rows.shape[0] == 1:
        rows = rows[0]
    if rows.ndim != 2 or (columns is not None and rows.shape[1] != columns):
        expected = f"Nx{columns}" if columns is not None else "NxC"
        raise DetectorOutputError(f"detector output must have shape {expected}")
    return rows


def _clip_box(
    box: Sequence[float],
    input_size: int,
) -> tuple[float, float, float, float] | None:
    coordinates = np.asarray(box, dtype=np.float64)
    if coordinates.shape != (4,) or not np.all(np.isfinite(coordinates)):
        return None
    x1, y1, x2, y2 = np.clip(coordinates, 0.0, float(input_size))
    if x2 <= x1 or y2 <= y1:
        return None
    return float(x1), float(y1), float(x2), float(y2)


def _nms(
    detections: Sequence[ParsedDetection],
    iou_threshold: float,
    max_detections: int,
) -> tuple[ParsedDetection, ...]:
    if not detections:
        return ()
    boxes = np.asarray([detection.box for detection in detections], dtype=np.float64)
    scores = np.asarray(
        [detection.confidence for detection in detections],
        dtype=np.float64,
    )
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    remaining = np.argsort(scores)[::-1]
    kept: list[int] = []
    while remaining.size and len(kept) < max_detections:
        index = int(remaining[0])
        kept.append(index)
        if remaining.size == 1:
            break
        candidates = remaining[1:]
        intersection_x1 = np.maximum(boxes[index, 0], boxes[candidates, 0])
        intersection_y1 = np.maximum(boxes[index, 1], boxes[candidates, 1])
        intersection_x2 = np.minimum(boxes[index, 2], boxes[candidates, 2])
        intersection_y2 = np.minimum(boxes[index, 3], boxes[candidates, 3])
        intersection = np.maximum(0.0, intersection_x2 - intersection_x1) * np.maximum(
            0.0,
            intersection_y2 - intersection_y1,
        )
        union = areas[index] + areas[candidates] - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        remaining = candidates[iou <= iou_threshold]
    return tuple(detections[index] for index in kept)


@dataclass(frozen=True)
class SsdLiteOutputAdapter:
    cat_class_id: int
    input_size: int
    output_names: tuple[str, str, str]
    score_threshold: float

    def parse(
        self,
        outputs: Mapping[str, np.ndarray],
    ) -> tuple[ParsedDetection, ...]:
        boxes = _rows(_output(outputs, self.output_names[0]), 4)
        labels = _output(outputs, self.output_names[1]).reshape(-1)
        scores = _output(outputs, self.output_names[2]).reshape(-1)
        if not (boxes.shape[0] == labels.size == scores.size):
            raise DetectorOutputError("SSDLite output lengths do not match")
        detections: list[ParsedDetection] = []
        for box, label, score in zip(boxes, labels, scores, strict=True):
            confidence = float(score)
            clipped = _clip_box(box, self.input_size)
            if (
                int(label) == self.cat_class_id
                and math.isfinite(confidence)
                and confidence >= self.score_threshold
                and clipped is not None
            ):
                detections.append(ParsedDetection(clipped, confidence))
        return tuple(detections)


@dataclass(frozen=True)
class YoloXOutputAdapter:
    cat_class_id: int
    input_size: int
    output_name: str
    score_threshold: float
    nms_iou_threshold: float
    max_detections: int

    def parse(
        self,
        outputs: Mapping[str, np.ndarray],
    ) -> tuple[ParsedDetection, ...]:
        predictions = _rows(_output(outputs, self.output_name))
        cat_score_column = 5 + self.cat_class_id
        if predictions.shape[1] <= cat_score_column:
            raise DetectorOutputError("YOLOX output has no declared cat class")
        detections: list[ParsedDetection] = []
        for prediction in predictions:
            confidence = float(prediction[4] * prediction[cat_score_column])
            if not math.isfinite(confidence) or confidence < self.score_threshold:
                continue
            center_x, center_y, width, height = prediction[:4]
            clipped = _clip_box(
                (
                    center_x - width / 2,
                    center_y - height / 2,
                    center_x + width / 2,
                    center_y + height / 2,
                ),
                self.input_size,
            )
            if clipped is not None:
                detections.append(ParsedDetection(clipped, confidence))
        return _nms(detections, self.nms_iou_threshold, self.max_detections)


@dataclass(frozen=True)
class Yolo11OutputAdapter:
    cat_class_id: int
    input_size: int
    output_name: str
    score_threshold: float

    def parse(
        self,
        outputs: Mapping[str, np.ndarray],
    ) -> tuple[ParsedDetection, ...]:
        predictions = _rows(_output(outputs, self.output_name), 6)
        detections: list[ParsedDetection] = []
        for prediction in predictions:
            confidence = float(prediction[4])
            clipped = _clip_box(prediction[:4], self.input_size)
            if (
                int(prediction[5]) == self.cat_class_id
                and math.isfinite(confidence)
                and confidence >= self.score_threshold
                and clipped is not None
            ):
                detections.append(ParsedDetection(clipped, confidence))
        return tuple(sorted(detections, key=lambda item: item.confidence, reverse=True))


@dataclass(frozen=True)
class NanoDetOutputAdapter:
    cat_class_id: int
    input_size: int
    output_name: str
    score_threshold: float
    nms_iou_threshold: float
    max_detections: int
    strides: tuple[int, ...]
    reg_max: int
    num_classes: int

    def parse(
        self,
        outputs: Mapping[str, np.ndarray],
    ) -> tuple[ParsedDetection, ...]:
        predictions = _rows(_output(outputs, self.output_name))
        regression_width = 4 * (self.reg_max + 1)
        if predictions.shape[1] != self.num_classes + regression_width:
            raise DetectorOutputError("NanoDet output width does not match manifest")
        if not 0 <= self.cat_class_id < self.num_classes:
            raise DetectorOutputError("NanoDet cat class is outside output range")
        centers: list[tuple[float, float, float]] = []
        for feature_stride in self.strides:
            side = math.ceil(self.input_size / feature_stride)
            centers.extend(
                (
                    float(x * feature_stride),
                    float(y * feature_stride),
                    float(feature_stride),
                )
                for y in range(side)
                for x in range(side)
            )
        if predictions.shape[0] != len(centers):
            raise DetectorOutputError("NanoDet output point count does not match input")
        distributions = predictions[:, self.num_classes :].reshape(
            -1,
            4,
            self.reg_max + 1,
        )
        stable = distributions - np.max(distributions, axis=2, keepdims=True)
        probabilities = np.exp(stable)
        probabilities /= np.sum(probabilities, axis=2, keepdims=True)
        bins = np.arange(self.reg_max + 1, dtype=np.float64)
        distances = np.sum(probabilities * bins, axis=2)
        detections: list[ParsedDetection] = []
        for prediction, distance, (center_x, center_y, point_stride) in zip(
            predictions,
            distances,
            centers,
            strict=True,
        ):
            confidence = float(prediction[self.cat_class_id])
            if not math.isfinite(confidence) or confidence < self.score_threshold:
                continue
            left, top, right, bottom = distance * point_stride
            clipped = _clip_box(
                (
                    center_x - left,
                    center_y - top,
                    center_x + right,
                    center_y + bottom,
                ),
                self.input_size,
            )
            if clipped is not None:
                detections.append(ParsedDetection(clipped, confidence))
        return _nms(detections, self.nms_iou_threshold, self.max_detections)


def create_output_adapter(
    manifest: Mapping[str, object],
    input_size: int,
) -> DetectorOutputAdapter:
    architecture = str(manifest["architecture"])
    raw_cat_class_id = manifest["cat_class_id"]
    if not isinstance(raw_cat_class_id, (int, str)) or isinstance(
        raw_cat_class_id,
        bool,
    ):
        raise DetectorOutputError("detector cat class id must be an integer")
    cat_class_id = int(raw_cat_class_id)
    raw_output_names = manifest.get("outputs")
    if not isinstance(raw_output_names, list) or not all(
        isinstance(name, str) for name in raw_output_names
    ):
        raise DetectorOutputError("detector manifest outputs must be a list of names")
    output_names = tuple(raw_output_names)
    postprocessing = manifest.get("postprocessing", {})
    if not isinstance(postprocessing, dict):
        raise DetectorOutputError("detector postprocessing manifest must be an object")
    score_threshold = float(postprocessing.get("score_threshold", 0.01))
    nms_iou_threshold = float(postprocessing.get("nms_iou_threshold", 0.6))
    max_detections = int(postprocessing.get("max_detections", 100))
    if not 0.0 <= score_threshold <= 1.0:
        raise DetectorOutputError("detector score threshold must be between 0 and 1")
    if not 0.0 < nms_iou_threshold <= 1.0:
        raise DetectorOutputError("detector NMS threshold must be between 0 and 1")
    if max_detections <= 0:
        raise DetectorOutputError("detector max detections must be positive")

    if architecture == "ssdlite320_mobilenet_v3_large":
        if len(output_names) != 3:
            raise DetectorOutputError("SSDLite requires boxes, labels and scores outputs")
        return SsdLiteOutputAdapter(
            cat_class_id,
            input_size,
            (output_names[0], output_names[1], output_names[2]),
            score_threshold,
        )
    if architecture == "yolox_nano":
        if len(output_names) != 1:
            raise DetectorOutputError("YOLOX requires one decoded output")
        return YoloXOutputAdapter(
            cat_class_id,
            input_size,
            output_names[0],
            score_threshold,
            nms_iou_threshold,
            max_detections,
        )
    # Экспорт Ultralytics даёт один и тот же end-to-end выход для любого
    # размера модели, поэтому yolo11n и yolo11s разбираются одинаково.
    if architecture.startswith(("yolo11", "yolo26")):
        if len(output_names) != 1:
            raise DetectorOutputError("YOLO11 requires one end-to-end NMS output")
        return Yolo11OutputAdapter(
            cat_class_id,
            input_size,
            output_names[0],
            score_threshold,
        )
    if architecture == "nanodet_plus_m":
        if len(output_names) != 1:
            raise DetectorOutputError("NanoDet requires one raw prediction output")
        raw_strides = postprocessing.get("strides", [8, 16, 32, 64])
        if not isinstance(raw_strides, list) or not all(
            isinstance(stride, int) and not isinstance(stride, bool) and stride > 0
            for stride in raw_strides
        ):
            raise DetectorOutputError("NanoDet strides must be positive integers")
        return NanoDetOutputAdapter(
            cat_class_id,
            input_size,
            output_names[0],
            score_threshold,
            nms_iou_threshold,
            max_detections,
            tuple(raw_strides),
            int(postprocessing.get("reg_max", 7)),
            int(postprocessing.get("num_classes", 80)),
        )
    raise DetectorOutputError(f"unsupported detector architecture: {architecture}")
