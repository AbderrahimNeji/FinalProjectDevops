"""CPU-only tests for label, GPS, and metric helpers. No GPU or weights."""

from pathlib import Path

import pytest

from src.evaluation.validation import (
    LabelValidationError,
    compile_box_metrics,
    resolve_model_path,
    validate_detection_coordinates,
    validate_yolo_label_file,
    validate_yolo_label_line,
    xyxy_to_yolo_bbox,
)


def test_xyxy_to_yolo_center_box():
    x, y, w, h = xyxy_to_yolo_bbox(25, 40, 75, 80, img_width=100, img_height=100)
    assert x == pytest.approx(0.5)
    assert y == pytest.approx(0.6)
    assert w == pytest.approx(0.5)
    assert h == pytest.approx(0.4)


def test_xyxy_clips_inverted_and_out_of_frame_boxes():
    x, y, w, h = xyxy_to_yolo_bbox(120, 10, 80, 50, img_width=100, img_height=100)
    assert 0 <= x <= 1 and 0 <= w <= 1
    assert w == pytest.approx(0.2)
    assert h == pytest.approx(0.4)


def test_xyxy_rejects_empty_or_zero_image():
    with pytest.raises(LabelValidationError, match="positive"):
        xyxy_to_yolo_bbox(0, 0, 10, 10, img_width=0, img_height=100)
    with pytest.raises(LabelValidationError, match="no area"):
        xyxy_to_yolo_bbox(5, 5, 5, 12, img_width=100, img_height=100)


def test_yolo_label_line_accepts_valid_box():
    parsed = validate_yolo_label_line("1 0.5 0.5 0.2 0.3")
    assert parsed == (1, 0.5, 0.5, 0.2, 0.3)
    assert validate_yolo_label_line("   ") is None


def test_yolo_label_line_rejects_bad_class_and_range():
    with pytest.raises(LabelValidationError, match="Class id"):
        validate_yolo_label_line("9 0.5 0.5 0.2 0.2")
    with pytest.raises(LabelValidationError, match="outside"):
        validate_yolo_label_line("0 1.5 0.5 0.2 0.2")
    with pytest.raises(LabelValidationError, match="5 fields"):
        validate_yolo_label_line("0 0.5 0.5")


def test_yolo_label_file(tmp_path: Path):
    labels = tmp_path / "img.txt"
    labels.write_text("0 0.4 0.4 0.1 0.1\n\n2 0.7 0.7 0.2 0.2\n", encoding="utf-8")
    boxes = validate_yolo_label_file(labels, num_classes=4)
    assert len(boxes) == 2
    assert boxes[1][0] == 2

    bad = tmp_path / "bad.txt"
    bad.write_text("0 0.4 0.4 0.1 2.0\n", encoding="utf-8")
    with pytest.raises(LabelValidationError, match="bad.txt:1"):
        validate_yolo_label_file(bad)


def test_geojson_coordinates_report_invalid_points():
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [2.35, 48.85]},
                "properties": {},
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [10.0, 120.0]},
                "properties": {},
            },
        ],
    }
    report = validate_detection_coordinates(payload)
    assert report["count"] == 2
    assert report["valid"] == 1
    assert report["invalid"] == 1
    assert report["all_valid"] is False


def test_compile_box_metrics_computes_f1_without_inventing_map():
    metrics = compile_box_metrics(precision=0.8, recall=0.5)
    assert metrics["f1"] == pytest.approx(0.615384, rel=1e-3)
    assert "mAP50" not in metrics

    with_map = compile_box_metrics(
        precision=1.0, recall=1.0, map50=0.42, map50_95=0.21
    )
    assert with_map["mAP50"] == 0.42
    assert with_map["mAP50_95"] == 0.21

    with pytest.raises(ValueError):
        compile_box_metrics(precision=float("nan"), recall=0.5)


def test_resolve_model_path_uses_existing_file_and_fails_honestly(tmp_path: Path):
    weights = tmp_path / "runs" / "detect" / "rdd2022_model" / "weights"
    weights.mkdir(parents=True)
    model = weights / "best.pt"
    model.write_bytes(b"not-a-real-weight")

    found = resolve_model_path(base_dir=tmp_path)
    assert found == model

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="Train a model"):
        resolve_model_path(base_dir=empty)
