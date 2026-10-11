"""
Evaluate a trained detector without inventing scores.

Requires a weights file produced by training. Unit tests for metrics and
label checks live in src/evaluation/validation.py and do not need a GPU.
Usage: python evaluate_model.py [--model PATH] [--data PATH] [video.mp4]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from src.evaluation.validation import (
    compile_box_metrics,
    resolve_model_path,
    validate_detection_coordinates,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO model")
    parser.add_argument(
        "video",
        nargs="?",
        default=None,
        help="Optional video for throughput measurement",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to trained weights (defaults to known training output paths)",
    )
    parser.add_argument(
        "--data",
        default="data/yolo_dataset/dataset.yaml",
        help="YOLO dataset YAML",
    )
    parser.add_argument(
        "--geojson",
        default="resultats/geojson/test_detections.geojson",
        help="Optional GeoJSON of detections to validate coordinates",
    )
    parser.add_argument(
        "--output",
        default="resultats/evaluation/metrics.json",
        help="Where to write measured metrics JSON",
    )
    return parser.parse_args(argv)


def evaluate(args):
    print("=" * 70)
    print("  EVALUATION - PROJECT 4")
    print("=" * 70)

    try:
        model_path = resolve_model_path(args.model)
    except FileNotFoundError as exc:
        print(f"\n{exc}")
        return 1

    data_yaml = Path(args.data)
    if not data_yaml.is_file():
        print(f"\nDataset YAML not found: {data_yaml}")
        print("Prepare a dataset before running model.val().")
        return 1

    print(f"\nModel: {model_path}")
    print(f"Dataset: {data_yaml}")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("\nultralytics is not installed. Install requirements.txt to run model.val().")
        print("CPU unit tests do not need it: pytest tests/test_evaluation_validation.py")
        return 1

    model = YOLO(str(model_path))
    metrics = model.val(data=str(data_yaml), plots=True)
    compiled = compile_box_metrics(
        precision=float(metrics.box.mp),
        recall=float(metrics.box.mr),
        map50=float(metrics.box.map50),
        map50_95=float(metrics.box.map),
    )

    print("\nMeasured detection metrics:")
    for key, value in compiled.items():
        print(f"  {key}: {value:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"model": str(model_path), "data": str(data_yaml), **compiled}, handle, indent=2)
    print(f"\nWrote {output_path}")

    print("\nGeolocation coordinate check")
    geojson_path = Path(args.geojson)
    if geojson_path.is_file():
        with geojson_path.open("r", encoding="utf-8") as handle:
            geojson = json.load(handle)
        report = validate_detection_coordinates(geojson)
        print(
            f"  points={report['count']} valid={report['valid']} invalid={report['invalid']}"
        )
        if report["count"] == 0:
            print("  No coordinates found in GeoJSON.")
        elif not report["all_valid"]:
            print("  Some coordinates are outside valid lat/lon ranges.")
        print("  Mean geolocation error was not computed (no GPS ground truth).")
    else:
        print(f"  No GeoJSON at {geojson_path}")

    print("\nThroughput")
    if args.video and os.path.exists(args.video):
        try:
            import cv2
        except ImportError:
            print("  opencv-python is not installed; skipped FPS measurement.")
            return 0

        cap = cv2.VideoCapture(args.video)
        frames_processed = 0
        start_time = time.time()
        while frames_processed < 100:
            ret, frame = cap.read()
            if not ret:
                break
            model.predict(frame, verbose=False)
            frames_processed += 1
        elapsed = time.time() - start_time
        cap.release()
        fps = frames_processed / elapsed if elapsed > 0 else 0.0
        print(f"  frames={frames_processed} seconds={elapsed:.2f} fps={fps:.2f}")
    else:
        print("  No video provided. Pass a path to measure FPS.")

    print("\nDone. Metrics above come from this run only.")
    return 0


if __name__ == "__main__":
    sys.exit(evaluate(parse_args()))
