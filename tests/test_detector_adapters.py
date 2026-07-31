import numpy as np
import pytest

from src.inference.detector_adapters import (
    DetectorOutputError,
    NanoDetOutputAdapter,
    SsdLiteOutputAdapter,
    Yolo11OutputAdapter,
    YoloXOutputAdapter,
    create_output_adapter,
)


def test_ssdlite_adapter_selects_cat_outputs() -> None:
    adapter = SsdLiteOutputAdapter(
        cat_class_id=17,
        input_size=320,
        output_names=("boxes", "labels", "scores"),
        score_threshold=0.05,
    )

    detections = adapter.parse(
        {
            "boxes": np.asarray(
                [[10, 20, 100, 120], [1, 2, 3, 4], [30, 40, 400, 420]]
            ),
            "labels": np.asarray([17, 1, 17]),
            "scores": np.asarray([0.4, 0.99, 0.8]),
        }
    )

    assert len(detections) == 2
    assert detections[1].box == (30.0, 40.0, 320.0, 320.0)


def test_yolox_adapter_decodes_cat_box_and_applies_nms() -> None:
    adapter = YoloXOutputAdapter(
        cat_class_id=1,
        input_size=416,
        output_name="output",
        score_threshold=0.1,
        nms_iou_threshold=0.5,
        max_detections=100,
    )
    predictions = np.asarray(
        [
            [100, 120, 80, 40, 0.9, 0.1, 0.8],
            [101, 120, 80, 40, 0.8, 0.1, 0.7],
            [300, 300, 40, 40, 0.2, 0.9, 0.1],
        ],
        dtype=np.float32,
    )[None]

    detections = adapter.parse({"output": predictions})

    assert len(detections) == 1
    assert detections[0].box == pytest.approx((60.0, 100.0, 140.0, 140.0))
    assert detections[0].confidence == pytest.approx(0.72)


def test_yolo11_adapter_reads_end_to_end_nms_output() -> None:
    adapter = Yolo11OutputAdapter(
        cat_class_id=15,
        input_size=416,
        output_name="output0",
        score_threshold=0.05,
    )

    detections = adapter.parse(
        {
            "output0": np.asarray(
                [[[10, 20, 100, 120, 0.8, 15], [20, 30, 40, 50, 0.9, 0]]],
                dtype=np.float32,
            )
        }
    )

    assert len(detections) == 1
    assert detections[0].confidence == pytest.approx(0.8)


def test_yolo26_alias_reads_nms_free_end_to_end_output() -> None:
    adapter = create_output_adapter(
        {
            "architecture": "yolo26",
            "cat_class_id": 15,
            "outputs": ["output0"],
            "postprocessing": {"score_threshold": 0.05},
        },
        416,
    )

    detections = adapter.parse(
        {
            "output0": np.asarray(
                [[[15, 25, 110, 130, 0.87, 15], [20, 30, 40, 50, 0.95, 0]]],
                dtype=np.float32,
            )
        }
    )

    assert len(detections) == 1
    assert detections[0].box == (15.0, 25.0, 110.0, 130.0)
    assert detections[0].confidence == pytest.approx(0.87)


def test_nanodet_adapter_decodes_distribution_output() -> None:
    adapter = NanoDetOutputAdapter(
        cat_class_id=1,
        input_size=8,
        output_name="output",
        score_threshold=0.05,
        nms_iou_threshold=0.6,
        max_detections=100,
        strides=(8,),
        reg_max=1,
        num_classes=2,
    )
    prediction = np.asarray(
        [[0.1, 0.8, -10, 10, -10, 10, -10, 10, -10, 10]],
        dtype=np.float32,
    )[None]

    detections = adapter.parse({"output": prediction})

    assert len(detections) == 1
    assert detections[0].box == pytest.approx((0.0, 0.0, 8.0, 8.0), abs=0.001)
    assert detections[0].confidence == pytest.approx(0.8)


def test_adapter_factory_rejects_unknown_architecture() -> None:
    with pytest.raises(DetectorOutputError, match="unsupported"):
        create_output_adapter(
            {
                "architecture": "unknown",
                "cat_class_id": 15,
                "outputs": ["output"],
            },
            416,
        )
