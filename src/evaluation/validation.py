"""
CPU-only helpers for YOLO labels, GeoJSON coordinates, and metric compilation.

These functions do not load neural-network weights or invent scores.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_NUM_CLASSES = 4
DEFAULT_MODEL_CANDIDATES = (
    Path("runs/detect/rdd2022_model/weights/best.pt"),
    Path("runs/detect/simple_model/weights/best.pt"),
    Path("models/rdd2022_best.pt"),
    Path("models/best.pt"),
)


class LabelValidationError(ValueError):
    """Invalid YOLO box or label content."""


def xyxy_to_yolo_bbox(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    img_width: float,
    img_height: float,
) -> tuple[float, float, float, float]:
    """Convert a pixel xyxy box to normalized YOLO xywh.

    Boxes that extend past the image are clipped. Empty or inverted boxes
    and non-positive image sizes raise LabelValidationError.
    """
    if img_width <= 0 or img_height <= 0:
        raise LabelValidationError("Image width and height must be positive")

    left = max(0.0, min(float(xmin), float(xmax)))
    right = min(float(img_width), max(float(xmin), float(xmax)))
    top = max(0.0, min(float(ymin), float(ymax)))
    bottom = min(float(img_height), max(float(ymin), float(ymax)))

    if right - left < 1e-6 or bottom - top < 1e-6:
        raise LabelValidationError("Bounding box has no area after clipping")

    x_center = ((left + right) / 2.0) / img_width
    y_center = ((top + bottom) / 2.0) / img_height
    width = (right - left) / img_width
    height = (bottom - top) / img_height

    return (
        _clip01(x_center),
        _clip01(y_center),
        _clip01(width),
        _clip01(height),
    )


def validate_yolo_label_line(
    line: str,
    num_classes: int = DEFAULT_NUM_CLASSES,
) -> tuple[int, float, float, float, float] | None:
    """Parse one YOLO label line. Blank lines return None."""
    text = line.strip()
    if not text or text.startswith("#"):
        return None

    parts = text.split()
    if len(parts) != 5:
        raise LabelValidationError(
            f"Expected 5 fields (class x y w h), got {len(parts)}: {text!r}"
        )

    try:
        class_id = int(float(parts[0]))
        x, y, w, h = (float(p) for p in parts[1:])
    except ValueError as exc:
        raise LabelValidationError(f"Non-numeric YOLO fields: {text!r}") from exc

    if not 0 <= class_id < num_classes:
        raise LabelValidationError(
            f"Class id {class_id} is outside 0..{num_classes - 1}"
        )

    for name, value in (("x", x), ("y", y), ("w", w), ("h", h)):
        if not math.isfinite(value):
            raise LabelValidationError(f"{name} is not finite")
        if value < 0.0 or value > 1.0:
            raise LabelValidationError(f"{name}={value} is outside [0, 1]")

    if w <= 0.0 or h <= 0.0:
        raise LabelValidationError("Box width and height must be positive")

    return class_id, x, y, w, h


def validate_yolo_label_file(
    path: str | Path,
    num_classes: int = DEFAULT_NUM_CLASSES,
) -> list[tuple[int, float, float, float, float]]:
    """Validate every line in a YOLO .txt label file."""
    label_path = Path(path)
    if not label_path.is_file():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    boxes: list[tuple[int, float, float, float, float]] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                parsed = validate_yolo_label_line(line, num_classes=num_classes)
            except LabelValidationError as exc:
                raise LabelValidationError(
                    f"{label_path}:{line_no}: {exc}"
                ) from exc
            if parsed is not None:
                boxes.append(parsed)
    return boxes


def validate_detection_coordinates(payload: Any) -> dict[str, Any]:
    """Check GPS points in a GeoJSON FeatureCollection or a list of detections.

    Does not invent a mean geolocation error when ground truth is absent.
    """
    points = list(_iter_coordinates(payload))
    valid: list[dict[str, float]] = []
    invalid: list[dict[str, Any]] = []

    for point in points:
        lat = point.get("latitude")
        lon = point.get("longitude")
        if not _is_valid_lat_lon(lat, lon):
            invalid.append(point)
        else:
            valid.append({"latitude": float(lat), "longitude": float(lon)})

    return {
        "count": len(points),
        "valid": len(valid),
        "invalid": len(invalid),
        "all_valid": len(points) > 0 and len(invalid) == 0,
        "invalid_points": invalid,
    }


def compile_box_metrics(
    *,
    precision: float,
    recall: float,
    map50: float | None = None,
    map50_95: float | None = None,
) -> dict[str, float]:
    """Derive F1 from measured precision/recall. Optional mAP values must be supplied."""
    for name, value in (
        ("precision", precision),
        ("recall", recall),
        ("map50", map50),
        ("map50_95", map50_95),
    ):
        if value is not None and not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite number")

    p = float(precision)
    r = float(recall)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    results = {
        "precision": p,
        "recall": r,
        "f1": f1,
    }
    if map50 is not None:
        results["mAP50"] = float(map50)
    if map50_95 is not None:
        results["mAP50_95"] = float(map50_95)
    return results


def resolve_model_path(
    explicit: str | Path | None = None,
    candidates: Sequence[str | Path] = DEFAULT_MODEL_CANDIDATES,
    *,
    base_dir: str | Path | None = None,
) -> Path:
    """Return the first existing model path. Does not download or train."""
    root = Path(base_dir) if base_dir else Path.cwd()
    search: list[Path] = []
    if explicit:
        search.append(Path(explicit))
    search.extend(Path(item) for item in candidates)

    for path in search:
        resolved = path if path.is_absolute() else root / path
        if resolved.is_file():
            return resolved

    tried = ", ".join(str(p) for p in search)
    raise FileNotFoundError(
        "No trained weights found. "
        f"Looked for: {tried}. Train a model before running evaluation."
    )


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _is_valid_lat_lon(lat: Any, lon: Any) -> bool:
    try:
        latitude = float(lat)
        longitude = float(lon)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(latitude)
        and math.isfinite(longitude)
        and -90.0 <= latitude <= 90.0
        and -180.0 <= longitude <= 180.0
    )


def _iter_coordinates(payload: Any) -> Iterable[dict[str, Any]]:
    if payload is None:
        return []

    if isinstance(payload, Mapping):
        if payload.get("type") == "FeatureCollection":
            for feature in payload.get("features") or []:
                yield from _iter_coordinates(feature)
            return
        if payload.get("type") == "Feature":
            geometry = payload.get("geometry") or {}
            props = payload.get("properties") or {}
            coords = geometry.get("coordinates")
            if geometry.get("type") == "Point" and isinstance(coords, (list, tuple)) and len(coords) >= 2:
                yield {
                    "longitude": coords[0],
                    "latitude": coords[1],
                    "properties": props,
                }
                return
            if "latitude" in props or "longitude" in props:
                yield {
                    "latitude": props.get("latitude"),
                    "longitude": props.get("longitude"),
                    "properties": props,
                }
            return
        if "latitude" in payload or "longitude" in payload:
            yield {
                "latitude": payload.get("latitude"),
                "longitude": payload.get("longitude"),
            }
            return
        if "lat" in payload or "lon" in payload:
            yield {
                "latitude": payload.get("lat"),
                "longitude": payload.get("lon"),
            }
            return

    if isinstance(payload, list):
        for item in payload:
            yield from _iter_coordinates(item)
        return

    return []
