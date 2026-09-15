#!/usr/bin/env python3
"""Freeze and measure the untouched pretrained RITnet baseline.

This evaluator deliberately keeps three layers separate:

1. raw RITnet class-mask quality;
2. fitted-circle measurement quality; and
3. production geometry-gate acceptance/failure.

The hand-drawn Labelme circles are the source of truth.  Because the iris labels
reconstruct the full circle under the lids, raw iris-mask overlap is diagnostic;
the fitted-circle metrics are the more faithful iris measurement score.

The script never trains and never writes to the labeling tree.  Each run emits a
manifest plus frame-, sequence-, and subject-level CSVs so future experiments can
be compared against exactly the same evaluation contract.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch

import ritnet_segment as pipeline


ROOT = Path(__file__).resolve().parent
FRAMES_DIR = ROOT / "labeling" / "frames_v2"
GT_DIR = ROOT / "labeling" / "gt"
GT_CSV = ROOT / "labeling" / "gt.csv"
MANIFEST_CSV = ROOT / "labeling" / "manifest_v2.csv"
DEFAULT_CHECKPOINT = ROOT / "RITnet" / "best_model.pkl"
IRIS_MM_ASSUMPTION = 11.7
BOOTSTRAP_SEED = 20260915
CLASS_NAMES = {0: "background", 1: "sclera", 2: "iris", 3: "pupil"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combined_hash(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths), key=lambda p: p.as_posix()):
        relative = path.resolve().relative_to(ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def run_text(*args: str, cwd: Path = ROOT) -> str | None:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def circle_from_labelme(shape: dict[str, Any]) -> tuple[float, float, float]:
    points = shape.get("points", [])
    if shape.get("shape_type") != "circle" or len(points) != 2:
        raise ValueError(f"{shape.get('label')} is not a two-point Labelme circle")
    (cx, cy), (ex, ey) = points
    return float(cx), float(cy), float(math.hypot(ex - cx, ey - cy))


def render_labelme_mask(document: dict[str, Any]) -> tuple[np.ndarray | None, dict[str, tuple[float, float, float]]]:
    height, width = int(document["imageHeight"]), int(document["imageWidth"])
    by_label: dict[str, dict[str, Any]] = {}
    for shape in document.get("shapes", []):
        label = shape.get("label")
        if label not in {"iris", "pupil", "sclera", "reject"}:
            raise ValueError(f"unknown label {label!r}")
        if label in by_label:
            raise ValueError(f"duplicate label {label!r}")
        by_label[label] = shape
    if "reject" in by_label:
        if len(by_label) != 1:
            raise ValueError("reject must be the only label")
        return None, {}
    if "iris" not in by_label:
        raise ValueError("non-reject label requires an iris circle")

    mask = np.zeros((height, width), np.uint8)
    if "sclera" in by_label:
        sclera = by_label["sclera"]
        if sclera.get("shape_type") != "polygon":
            raise ValueError("sclera must be a polygon")
        cv2.fillPoly(mask, [np.asarray(sclera["points"], np.int32)], pipeline.CLS_SCLERA)
    circles = {"iris": circle_from_labelme(by_label["iris"])}
    cx, cy, radius = circles["iris"]
    cv2.circle(mask, (round(cx), round(cy)), round(radius), pipeline.CLS_IRIS, -1)
    if "pupil" in by_label:
        circles["pupil"] = circle_from_labelme(by_label["pupil"])
        cx, cy, radius = circles["pupil"]
        cv2.circle(mask, (round(cx), round(cy)), round(radius), pipeline.CLS_PUPIL, -1)
    return mask, circles


def nearly_equal(value: str, expected: float, tolerance: float = 0.011) -> bool:
    return value != "" and abs(float(value) - expected) <= tolerance


def label_matches_csv(row: dict[str, str], circles: dict[str, tuple[float, float, float]]) -> bool:
    expected_category = "full" if "pupil" in circles else "iris_only"
    if row["category"] != expected_category:
        return False
    iris = circles["iris"]
    if not all(nearly_equal(row[key], value) for key, value in zip(("iris_cx", "iris_cy", "iris_r"), iris)):
        return False
    if expected_category == "full":
        pupil = circles["pupil"]
        return all(nearly_equal(row[key], value) for key, value in zip(("pupil_cx", "pupil_cy", "pupil_r"), pupil))
    return all(row[key] == "" for key in ("pupil_cx", "pupil_cy", "pupil_r"))


def is_edge_clipped(manifest_row: dict[str, str], fraction: float = 0.01) -> bool:
    x, y, width, height = map(int, manifest_row["box"].split("_"))
    original_h, original_w = int(manifest_row["orig_h"]), int(manifest_row["orig_w"])
    margin_x, margin_y = fraction * original_w, fraction * original_h
    return (
        x <= margin_x
        or y <= margin_y
        or x + width >= original_w - margin_x
        or y + height >= original_h - margin_y
    )


def audit_dataset() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate derived GT against Labelme JSON without changing any files."""
    gt_rows = list(csv.DictReader(GT_CSV.open()))
    manifest_rows = list(csv.DictReader(MANIFEST_CSV.open()))
    manifest_by_path = {Path(row["path"]).as_posix(): row for row in manifest_rows}
    all_pngs = sorted(FRAMES_DIR.glob("*/*.png"))
    all_jsons = sorted(FRAMES_DIR.glob("*/*.json"))
    png_relatives = {path.relative_to(ROOT).as_posix() for path in all_pngs}
    labeled_png_relatives = {
        path.with_suffix(".png").relative_to(ROOT).as_posix() for path in all_jsons
    }

    issues: list[dict[str, str]] = []
    records: list[dict[str, Any]] = []
    gt_keys: set[tuple[str, str]] = set()
    categories = Counter()

    for row in gt_rows:
        key = (row["seq_id"], row["frame"])
        categories[row["category"]] += 1
        if key in gt_keys:
            issues.append({"kind": "duplicate_gt_row", "record": "/".join(key)})
            continue
        gt_keys.add(key)
        frame_path = FRAMES_DIR / row["seq_id"] / row["frame"]
        label_path = frame_path.with_suffix(".json")
        mask_path = GT_DIR / row["seq_id"] / f"{frame_path.stem}_mask.png"
        relative_frame = frame_path.relative_to(ROOT).as_posix()
        manifest_row = manifest_by_path.get(relative_frame)
        record_issues: list[str] = []

        for kind, path in (("missing_frame", frame_path), ("missing_label_json", label_path)):
            if not path.exists():
                record_issues.append(kind)
        if manifest_row is None:
            record_issues.append("missing_manifest_row")
        if row["category"] != "reject" and not mask_path.exists():
            record_issues.append("missing_derived_mask")

        if label_path.exists():
            try:
                document = json.loads(label_path.read_text())
                rendered_mask, circles = render_labelme_mask(document)
                label_category = "reject" if rendered_mask is None else ("full" if "pupil" in circles else "iris_only")
                if label_category != row["category"]:
                    record_issues.append("label_csv_category_mismatch")
                elif rendered_mask is not None and not label_matches_csv(row, circles):
                    record_issues.append("label_csv_geometry_mismatch")
                if frame_path.exists():
                    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                    if image is None:
                        record_issues.append("unreadable_frame")
                    elif image.shape[:2] != (pipeline.IN_H, pipeline.IN_W):
                        record_issues.append("unexpected_frame_dimensions")
                    elif (int(document["imageHeight"]), int(document["imageWidth"])) != image.shape[:2]:
                        record_issues.append("label_image_dimensions_mismatch")
                if rendered_mask is not None and mask_path.exists():
                    stored_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if stored_mask is None or not np.array_equal(stored_mask, rendered_mask):
                        record_issues.append("derived_mask_is_stale")
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                record_issues.append(f"invalid_label:{exc}")

        if record_issues:
            for kind in record_issues:
                issues.append({"kind": kind, "record": relative_frame})
            continue
        if row["category"] == "reject":
            continue

        records.append(
            {
                **row,
                "frame_path": frame_path,
                "label_path": label_path,
                "mask_path": mask_path,
                "edge_clip": is_edge_clipped(manifest_row),
            }
        )

    pending = sorted(png_relatives - labeled_png_relatives)
    json_without_gt = []
    for label_path in all_jsons:
        seq_id, frame = label_path.parent.name, label_path.with_suffix(".png").name
        if (seq_id, frame) not in gt_keys:
            json_without_gt.append(label_path.relative_to(ROOT).as_posix())

    eligible_files = [GT_CSV, MANIFEST_CSV]
    for record in records:
        eligible_files.extend((record["frame_path"], record["label_path"], record["mask_path"]))
    audit = {
        "status": "clean" if not issues and not pending and not json_without_gt else "warning",
        "gt_rows": len(gt_rows),
        "eligible_rows": len(records),
        "excluded_gt_rows": len(gt_rows) - len(records),
        "categories_in_gt": dict(sorted(categories.items())),
        "eligible_categories": dict(sorted(Counter(r["category"] for r in records).items())),
        "eligible_subjects": dict(sorted(Counter(r["person"] for r in records).items())),
        "eligible_sequences": len({r["seq_id"] for r in records}),
        "edge_clipped_eligible_rows": sum(bool(r["edge_clip"]) for r in records),
        "frame_pngs": len(all_pngs),
        "label_jsons": len(all_jsons),
        "pending_pngs_without_json": pending,
        "label_jsons_without_gt_row": json_without_gt,
        "issues": issues,
        "dataset_sha256": combined_hash(eligible_files),
    }
    return records, audit


def binary_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = truth.astype(bool)
    prediction = prediction.astype(bool)
    intersection = int(np.logical_and(truth, prediction).sum())
    truth_n, prediction_n = int(truth.sum()), int(prediction.sum())
    union = truth_n + prediction_n - intersection
    return {
        "iou": intersection / union if union else 1.0,
        "dice": 2 * intersection / (truth_n + prediction_n) if truth_n + prediction_n else 1.0,
        "precision": intersection / prediction_n if prediction_n else 0.0,
        "recall": intersection / truth_n if truth_n else 0.0,
    }


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(np.uint8)
    eroded = cv2.erode(binary, np.ones((3, 3), np.uint8))
    return binary.astype(bool) & ~eroded.astype(bool)


def boundary_assd(truth: np.ndarray, prediction: np.ndarray) -> float | None:
    """Symmetric mean boundary distance in model-input pixels."""
    truth_boundary, prediction_boundary = mask_boundary(truth), mask_boundary(prediction)
    if not truth_boundary.any() or not prediction_boundary.any():
        return None
    distance_to_truth = cv2.distanceTransform((~truth_boundary).astype(np.uint8), cv2.DIST_L2, 3)
    distance_to_prediction = cv2.distanceTransform((~prediction_boundary).astype(np.uint8), cv2.DIST_L2, 3)
    return float(
        0.5
        * (
            distance_to_truth[prediction_boundary].mean()
            + distance_to_prediction[truth_boundary].mean()
        )
    )


def circle_mask(circle: tuple[float, float, float] | None) -> np.ndarray | None:
    if circle is None:
        return None
    return pipeline.disk_mask(circle, (pipeline.IN_H, pipeline.IN_W))


def gt_circle(row: dict[str, Any], prefix: str) -> tuple[float, float, float] | None:
    if row[f"{prefix}_cx"] == "":
        return None
    return tuple(float(row[f"{prefix}_{part}"]) for part in ("cx", "cy", "r"))  # type: ignore[return-value]


def gate_reason(iris: tuple[float, float, float] | None, pupil: tuple[float, float, float] | None) -> str:
    if iris is None:
        return "iris_fit_failed"
    if pupil is None:
        return "pupil_fit_failed"
    ratio = pupil[2] / iris[2]
    if ratio < pipeline.GATE_RATIO_MIN:
        return "pupil_ratio_too_small"
    if ratio > pipeline.GATE_RATIO_MAX:
        return "pupil_ratio_too_large"
    offset = math.hypot(pupil[0] - iris[0], pupil[1] - iris[1])
    if offset > pipeline.GATE_CONCENTRIC * iris[2]:
        return "pupil_not_concentric"
    return "accepted"


def finite_or_none(value: float | np.floating[Any] | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def synchronize() -> None:
    if pipeline.DEVICE == "mps":
        torch.mps.synchronize()
    elif str(pipeline.DEVICE).startswith("cuda"):
        torch.cuda.synchronize()


def evaluate_frame(
    record: dict[str, Any],
    channel: str,
    image: np.ndarray,
    prediction: np.ndarray,
    model_ms_per_frame: float,
) -> dict[str, Any]:
    ground_truth = cv2.imread(str(record["mask_path"]), cv2.IMREAD_GRAYSCALE)
    measurement = pipeline.measure(prediction, image)
    iris_prediction = measurement["iris_circle"]
    pupil_prediction = measurement["pupil_circle"]
    gt_iris = gt_circle(record, "iris")
    gt_pupil = gt_circle(record, "pupil")
    reason = gate_reason(iris_prediction, pupil_prediction)
    accepted = reason == "accepted"
    raw_iris = (prediction == pipeline.CLS_IRIS) | (prediction == pipeline.CLS_PUPIL)
    raw_pupil = prediction == pipeline.CLS_PUPIL
    gt_iris_mask = (ground_truth == pipeline.CLS_IRIS) | (ground_truth == pipeline.CLS_PUPIL)
    gt_pupil_mask = ground_truth == pipeline.CLS_PUPIL
    raw_iris_metrics = binary_metrics(gt_iris_mask, raw_iris)

    row: dict[str, Any] = {
        "channel": channel,
        "person": record["person"],
        "seq_id": record["seq_id"],
        "frame": record["frame"],
        "category": record["category"],
        "edge_clip": bool(record["edge_clip"]),
        "model_ms": model_ms_per_frame,
        "raw_iris_iou": raw_iris_metrics["iou"],
        "raw_iris_dice": raw_iris_metrics["dice"],
        "raw_pupil_pixels": int(raw_pupil.sum()),
        "iris_fit": iris_prediction is not None,
        "pupil_fit": pupil_prediction is not None,
        "gate_accepted": accepted,
        "failure_reason": reason,
    }

    if gt_iris is not None:
        predicted_iris_mask = circle_mask(iris_prediction)
        row["iris_circle_iou_all"] = 0.0 if predicted_iris_mask is None else binary_metrics(gt_iris_mask, predicted_iris_mask)["iou"]
        if iris_prediction is not None:
            row["iris_center_error_px"] = math.hypot(iris_prediction[0] - gt_iris[0], iris_prediction[1] - gt_iris[1])
            row["iris_diameter_abs_error_px"] = abs(2 * iris_prediction[2] - 2 * gt_iris[2])
            row["iris_diameter_abs_rel_error"] = row["iris_diameter_abs_error_px"] / (2 * gt_iris[2])

    if record["category"] == "full" and gt_pupil is not None:
        raw_pupil_metrics = binary_metrics(gt_pupil_mask, raw_pupil)
        for name, value in raw_pupil_metrics.items():
            row[f"raw_pupil_{name}"] = value
        row["raw_pupil_boundary_assd_px"] = boundary_assd(gt_pupil_mask, raw_pupil)
        predicted_pupil_mask = circle_mask(pupil_prediction)
        ungated_iou = 0.0 if predicted_pupil_mask is None else binary_metrics(gt_pupil_mask, predicted_pupil_mask)["iou"]
        row["pupil_circle_iou_ungated_all"] = ungated_iou
        row["pupil_circle_iou_gated_all"] = ungated_iou if accepted else 0.0
        if pupil_prediction is not None:
            row["pupil_center_error_px_ungated"] = math.hypot(
                pupil_prediction[0] - gt_pupil[0], pupil_prediction[1] - gt_pupil[1]
            )
            row["pupil_center_error_gt_diameter_ungated"] = row["pupil_center_error_px_ungated"] / (2 * gt_pupil[2])
        if accepted and iris_prediction is not None:
            diameter_error = abs(2 * pupil_prediction[2] - 2 * gt_pupil[2])
            predicted_ratio = pupil_prediction[2] / iris_prediction[2]
            gt_ratio = gt_pupil[2] / gt_iris[2]
            row["pupil_circle_iou_accepted"] = ungated_iou
            row["pupil_center_error_px_accepted"] = row["pupil_center_error_px_ungated"]
            row["pupil_center_error_gt_diameter_accepted"] = row["pupil_center_error_gt_diameter_ungated"]
            row["pupil_diameter_abs_error_px_accepted"] = diameter_error
            row["pupil_diameter_abs_rel_error_accepted"] = diameter_error / (2 * gt_pupil[2])
            row["pupil_iris_ratio_abs_error_accepted"] = abs(predicted_ratio - gt_ratio)
            row["pupil_mm_equivalent_abs_error_accepted"] = abs(predicted_ratio - gt_ratio) * IRIS_MM_ASSUMPTION
    else:
        # There is no defensible pupil GT on iris_only frames.  Acceptance here is
        # reported as unscorable output, never mislabeled as a false positive.
        row["unscorable_gate_accept"] = accepted
    return row


def evaluate(records: list[dict[str, Any]], channels: list[str], batch_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pipeline.model.eval()
    for channel in channels:
        pipeline.USE_RED_CHANNEL = channel == "red"
        print(f"Evaluating pretrained RITnet: channel={channel}, frames={len(records)}", flush=True)
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            images = [cv2.imread(str(record["frame_path"]), cv2.IMREAD_COLOR) for record in batch_records]
            if any(image is None for image in images):
                raise RuntimeError("an audited image became unreadable during evaluation")
            tensors = torch.cat(
                [pipeline.preprocess(pipeline.ritnet_input(image)) for image in images], dim=0
            ).to(pipeline.DEVICE)
            synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                logits = pipeline.model(tensors)
            synchronize()
            model_ms = (time.perf_counter() - started) * 1000 / len(batch_records)
            masks = logits.argmax(1).cpu().numpy().astype(np.uint8)
            for record, image, mask in zip(batch_records, images, masks):
                rows.append(evaluate_frame(record, channel, image, mask, model_ms))
            if start == 0 or (start + batch_size) % 80 == 0:
                print(f"  {min(start + batch_size, len(records))}/{len(records)}", flush=True)
    return rows


def values(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [float(row[field]) for row in rows if finite_or_none(row.get(field)) is not None]


def mean(rows: list[dict[str, Any]], field: str) -> float | None:
    series = values(rows, field)
    return float(np.mean(series)) if series else None


def rate(rows: list[dict[str, Any]], field: str) -> float | None:
    eligible = [row for row in rows if field in row and row[field] is not None]
    return float(np.mean([bool(row[field]) for row in eligible])) if eligible else None


def describe(series: list[float]) -> dict[str, float | int | None]:
    if not series:
        return {"n": 0, "mean": None, "median": None, "q1": None, "q3": None, "p95": None, "max": None}
    array = np.asarray(series, float)
    return {
        "n": len(series),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q1": float(np.percentile(array, 25)),
        "q3": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def sequence_summaries(frame_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        grouped[(row["channel"], row["person"], row["seq_id"])].append(row)
    output = []
    for (channel, person, seq_id), rows in sorted(grouped.items()):
        full = [row for row in rows if row["category"] == "full"]
        accepted = sum(bool(row["gate_accepted"]) for row in full)
        output.append(
            {
                "channel": channel,
                "person": person,
                "seq_id": seq_id,
                "labeled_frames": len(rows),
                "full_frames": len(full),
                "accepted_full_frames": accepted,
                "full_frame_accept_rate": accepted / len(full) if full else None,
                "zero_accept_failure": bool(full) and accepted == 0,
                "under_50pct_accept_failure": bool(full) and accepted / len(full) < 0.5,
                "edge_clip_frames": sum(bool(row["edge_clip"]) for row in rows),
            }
        )
    return output


def subject_summaries(
    frame_rows: list[dict[str, Any]], sequence_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seq_grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        grouped[(row["channel"], row["person"])].append(row)
    for row in sequence_rows:
        seq_grouped[(row["channel"], row["person"])].append(row)
    output = []
    for (channel, person), rows in sorted(grouped.items()):
        full = [row for row in rows if row["category"] == "full"]
        hidden = [row for row in rows if row["category"] == "iris_only"]
        sequences = [row for row in seq_grouped[(channel, person)] if row["full_frames"]]
        output.append(
            {
                "channel": channel,
                "person": person,
                "labeled_frames": len(rows),
                "full_frames": len(full),
                "iris_only_frames": len(hidden),
                "edge_clip_frames": sum(bool(row["edge_clip"]) for row in rows),
                "iris_fit_rate": rate(rows, "iris_fit"),
                "raw_iris_iou_mean": mean(rows, "raw_iris_iou"),
                "iris_circle_iou_all_mean": mean(rows, "iris_circle_iou_all"),
                "raw_pupil_iou_mean": mean(full, "raw_pupil_iou"),
                "raw_pupil_dice_mean": mean(full, "raw_pupil_dice"),
                "raw_pupil_recall_mean": mean(full, "raw_pupil_recall"),
                "raw_pupil_boundary_assd_px_mean": mean(full, "raw_pupil_boundary_assd_px"),
                "pupil_fit_rate": rate(full, "pupil_fit"),
                "pupil_gate_accept_rate": rate(full, "gate_accepted"),
                "pupil_circle_iou_ungated_all_mean": mean(full, "pupil_circle_iou_ungated_all"),
                "pupil_circle_iou_gated_all_mean": mean(full, "pupil_circle_iou_gated_all"),
                "pupil_circle_iou_accepted_mean": mean(full, "pupil_circle_iou_accepted"),
                "pupil_diameter_abs_rel_error_accepted_mean": mean(full, "pupil_diameter_abs_rel_error_accepted"),
                "pupil_mm_equivalent_abs_error_accepted_mean": mean(full, "pupil_mm_equivalent_abs_error_accepted"),
                "pupil_center_error_gt_diameter_accepted_mean": mean(full, "pupil_center_error_gt_diameter_accepted"),
                "unscorable_gate_accept_rate": rate(hidden, "unscorable_gate_accept"),
                "sequence_zero_accept_failure_rate": rate(sequences, "zero_accept_failure"),
                "sequence_under_50pct_accept_failure_rate": rate(sequences, "under_50pct_accept_failure"),
            }
        )
    return output


MACRO_METRICS = [
    "iris_fit_rate",
    "raw_iris_iou_mean",
    "iris_circle_iou_all_mean",
    "raw_pupil_iou_mean",
    "raw_pupil_dice_mean",
    "raw_pupil_recall_mean",
    "raw_pupil_boundary_assd_px_mean",
    "pupil_fit_rate",
    "pupil_gate_accept_rate",
    "pupil_circle_iou_ungated_all_mean",
    "pupil_circle_iou_gated_all_mean",
    "pupil_circle_iou_accepted_mean",
    "pupil_diameter_abs_rel_error_accepted_mean",
    "pupil_mm_equivalent_abs_error_accepted_mean",
    "pupil_center_error_gt_diameter_accepted_mean",
    "unscorable_gate_accept_rate",
    "sequence_zero_accept_failure_rate",
    "sequence_under_50pct_accept_failure_rate",
]


def bootstrap_subject_mean(series: list[float], samples: int, seed: int) -> dict[str, Any]:
    if not series:
        return {"n_subjects": 0, "mean": None, "std": None, "ci95": [None, None]}
    array = np.asarray(series, float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return {
        "n_subjects": len(series),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else None,
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
    }


def make_summary(
    frame_rows: list[dict[str, Any]],
    sequence_rows: list[dict[str, Any]],
    subject_rows: list[dict[str, Any]],
    channels: list[str],
    bootstrap_samples: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for channel_index, channel in enumerate(channels):
        channel_frames = [row for row in frame_rows if row["channel"] == channel]
        channel_subjects = [row for row in subject_rows if row["channel"] == channel]
        channel_sequences = [row for row in sequence_rows if row["channel"] == channel]
        macro = {}
        for metric_index, metric in enumerate(MACRO_METRICS):
            series = [float(row[metric]) for row in channel_subjects if finite_or_none(row.get(metric)) is not None]
            macro[metric] = bootstrap_subject_mean(
                series,
                bootstrap_samples,
                BOOTSTRAP_SEED + channel_index * 100 + metric_index,
            )
        full = [row for row in channel_frames if row["category"] == "full"]
        hidden = [row for row in channel_frames if row["category"] == "iris_only"]
        summary[channel] = {
            "counts": {
                "frames": len(channel_frames),
                "full_frames": len(full),
                "iris_only_frames": len(hidden),
                "subjects": len(channel_subjects),
                "sequences": len(channel_sequences),
                "full_sequences": sum(bool(row["full_frames"]) for row in channel_sequences),
            },
            "subject_macro": macro,
            "frame_descriptive": {
                "raw_pupil_iou": describe(values(full, "raw_pupil_iou")),
                "pupil_circle_iou_gated_all": describe(values(full, "pupil_circle_iou_gated_all")),
                "pupil_diameter_abs_rel_error_accepted": describe(values(full, "pupil_diameter_abs_rel_error_accepted")),
                "pupil_mm_equivalent_abs_error_accepted": describe(values(full, "pupil_mm_equivalent_abs_error_accepted")),
                "model_ms": describe(values(channel_frames, "model_ms")),
            },
            "failure_reasons_full_frames": dict(sorted(Counter(row["failure_reason"] for row in full).items())),
            "unscorable_iris_only_outputs": dict(sorted(Counter(row["failure_reason"] for row in hidden).items())),
        }
    return summary


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_clean(item) for item in value]
    if isinstance(value, tuple):
        return [json_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return finite_or_none(value)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    preferred = [
        field
        for field in ("channel", "person", "seq_id", "frame", "category", "edge_clip")
        if field in fields
    ]
    fields = preferred + [field for field in fields if field not in preferred]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field, "") for field in fields})


def format_ci(metric: dict[str, Any], percent: bool = False) -> str:
    if metric["mean"] is None:
        return "n/a"
    multiplier = 100 if percent else 1
    low, high = metric["ci95"]
    suffix = "%" if percent else ""
    return f"{metric['mean'] * multiplier:.1f}{suffix} [{low * multiplier:.1f}, {high * multiplier:.1f}]"


def build_report(manifest: dict[str, Any], audit: dict[str, Any], summary: dict[str, Any], subject_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Frozen pretrained RITnet baseline",
        "",
        f"Run: `{manifest['run_id']}`  ",
        f"Checkpoint SHA-256: `{manifest['checkpoint']['sha256']}`  ",
        f"Dataset SHA-256: `{audit['dataset_sha256']}`  ",
        f"Git commit: `{manifest['git']['commit']}` ({'dirty working tree' if manifest['git']['dirty'] else 'clean'})",
        "",
        "This run performs inference only. No subject is used for training, no checkpoint is selected, and no threshold is tuned.",
        "",
        "## Dataset integrity",
        "",
        f"- Status: **{audit['status']}**",
        f"- {audit['eligible_rows']} of {audit['gt_rows']} GT rows were eligible; {audit['excluded_gt_rows']} were excluded for explicit integrity issues.",
        f"- Eligible categories: `{audit['eligible_categories']}` across {audit['eligible_sequences']} eye sequences.",
        f"- {len(audit['pending_pngs_without_json'])} PNGs have no Labelme JSON and remain pending.",
    ]
    if audit["issues"]:
        lines.extend(("", "Integrity issues:"))
        for issue in audit["issues"]:
            lines.append(f"- `{issue['kind']}`: `{issue['record']}`")

    lines.extend(
        (
            "",
            "## Subject-macro results",
            "",
            "Confidence intervals resample subjects, not frames. Values are macro means across subjects; conditional errors use only gate-accepted frames and must be read with acceptance rate.",
            "",
            "| Metric | " + " | ".join(channel for channel in summary) + " |",
            "|---|" + "---|" * len(summary),
        )
    )
    report_metrics = [
        ("Raw pupil IoU", "raw_pupil_iou_mean", False),
        ("Raw pupil recall", "raw_pupil_recall_mean", True),
        ("Gate acceptance on visible-pupil frames", "pupil_gate_accept_rate", True),
        ("Pupil circle IoU, misses = 0", "pupil_circle_iou_gated_all_mean", False),
        ("Pupil circle IoU, accepted only", "pupil_circle_iou_accepted_mean", False),
        ("Pupil diameter absolute relative error, accepted only", "pupil_diameter_abs_rel_error_accepted_mean", True),
        ("Iris-normalized pupil error, mm-equivalent, accepted only", "pupil_mm_equivalent_abs_error_accepted_mean", False),
        ("Pupil center error / GT diameter, accepted only", "pupil_center_error_gt_diameter_accepted_mean", False),
        ("Sequences with zero accepted visible-pupil frames", "sequence_zero_accept_failure_rate", True),
        ("Unscorable acceptance on iris-only frames", "unscorable_gate_accept_rate", True),
    ]
    for label, metric_name, percent in report_metrics:
        lines.append(
            f"| {label} | "
            + " | ".join(format_ci(summary[channel]["subject_macro"][metric_name], percent) for channel in summary)
            + " |"
        )

    lines.extend(("", "## Per-subject production-candidate view (red channel)", ""))
    lines.extend(
        (
            "| Subject | Visible-pupil frames | Gate acceptance | Raw pupil IoU | Circle IoU, misses = 0 | Diameter abs. relative error, accepted |",
            "|---|---:|---:|---:|---:|---:|",
        )
    )
    for row in subject_rows:
        if row["channel"] != "red":
            continue
        def cell(name: str, percent: bool = False) -> str:
            value = finite_or_none(row.get(name))
            if value is None:
                return "n/a"
            return f"{value * 100:.1f}%" if percent else f"{value:.3f}"
        lines.append(
            f"| {row['person']} | {row['full_frames']} | {cell('pupil_gate_accept_rate', True)} | "
            f"{cell('raw_pupil_iou_mean')} | {cell('pupil_circle_iou_gated_all_mean')} | "
            f"{cell('pupil_diameter_abs_rel_error_accepted_mean', True)} |"
        )

    lines.extend(
        (
            "",
            "## Interpretation boundaries",
            "",
            "- Raw pupil scores use only `full` frames where a human labeled a visible pupil boundary.",
            "- `iris_only` frames have no pupil ground truth. An emitted measurement is therefore **unscorable**, not automatically a false positive.",
            "- Iris labels reconstruct a circle under eyelids. Raw iris-mask overlap is diagnostic; fitted-circle error is the measurement-relevant result.",
            "- The mm-equivalent error compares prediction and annotation through the same 11.7 mm iris assumption. It is not physical-mm validation against a reference pupillometer.",
            "- `edge_clip` remains in the primary result and is available as a column for stratified analysis; hard cases are not silently removed.",
            "- Host model timing is a development diagnostic, not an iPhone/Core ML latency claim.",
            "",
            "## Artifacts",
            "",
            "- `run_manifest.json`: checkpoint, code, data, environment, and evaluation contract identity",
            "- `summary.json`: subject-macro confidence intervals plus frame-level distributions",
            "- `subject_metrics.csv`: every fold/subject separately",
            "- `sequence_metrics.csv`: video/eye-sequence failure rates",
            "- `frame_metrics.csv`: auditable predictions, errors, and gate failure reasons",
        )
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze and evaluate the untouched pretrained RITnet baseline")
    parser.add_argument("--channels", nargs="+", choices=("gray", "red"), default=("gray", "red"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict-integrity", action="store_true", help="abort if the label/manifest audit has warnings")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if checkpoint != DEFAULT_CHECKPOINT.resolve():
        raise SystemExit(
            "This command is intentionally restricted to RITnet/best_model.pkl. "
            "Use a separate experiment runner for candidate weights."
        )
    records, audit = audit_dataset()
    if args.strict_integrity and audit["status"] != "clean":
        raise SystemExit(json.dumps(audit, indent=2))
    if not records:
        raise SystemExit("no integrity-eligible labeled records")

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"ritnet_pretrained_{timestamp}"
    output = (args.output or (ROOT / "evaluation" / "runs" / run_id)).resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    # ritnet_segment loads this same checkpoint at import. Assert that identity rather
    # than quietly evaluating a different global model.
    checkpoint_hash = sha256_file(checkpoint)
    git_status = run_text("git", "status", "--porcelain") or ""
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "untouched pretrained RITnet baseline; inference only",
        "checkpoint": {
            "path": checkpoint.relative_to(ROOT).as_posix(),
            "sha256": checkpoint_hash,
            "parameter_count": sum(parameter.numel() for parameter in pipeline.model.parameters()),
            "architecture": "RITnet DenseNet2D, dropout=True, probability=0.2, eval mode",
        },
        "git": {
            "commit": run_text("git", "rev-parse", "HEAD"),
            "dirty": bool(git_status),
            "changed_path_count": len(git_status.splitlines()) if git_status else 0,
            "ritnet_commit": run_text("git", "rev-parse", "HEAD", cwd=ROOT / "RITnet"),
        },
        "source_sha256": {
            "evaluate_pretrained.py": sha256_file(Path(__file__)),
            "ritnet_segment.py": sha256_file(ROOT / "ritnet_segment.py"),
            "densenet.py": sha256_file(ROOT / "RITnet" / "densenet.py"),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "device": str(pipeline.DEVICE),
        },
        "evaluation": {
            "channels": list(args.channels),
            "input_width": pipeline.IN_W,
            "input_height": pipeline.IN_H,
            "batch_size": args.batch_size,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "primary_universe": "all integrity-eligible labeled frames; edge-clipped frames included",
            "pupil_scoring_universe": "category=full only",
            "iris_mm_assumption": IRIS_MM_ASSUMPTION,
            "preprocessing": "selected RGB channel -> gamma 0.8 -> CLAHE(1.5, 8x8) -> [0,1] -> [-1,1]",
            "postprocessing": "current ritnet_segment.py circle fit plus documented geometry gate",
            "temporal_smoothing": "none; independent labeled frames",
            "class_ids": CLASS_NAMES,
        },
        "dataset_audit": audit,
    }
    (output / "run_manifest.json").write_text(json.dumps(json_clean(manifest), indent=2) + "\n")

    frame_rows = evaluate(records, list(args.channels), args.batch_size)
    sequence_rows = sequence_summaries(frame_rows)
    subject_rows = subject_summaries(frame_rows, sequence_rows)
    summary = make_summary(frame_rows, sequence_rows, subject_rows, list(args.channels), args.bootstrap_samples)

    write_csv(output / "frame_metrics.csv", frame_rows)
    write_csv(output / "sequence_metrics.csv", sequence_rows)
    write_csv(output / "subject_metrics.csv", subject_rows)
    (output / "summary.json").write_text(json.dumps(json_clean(summary), indent=2) + "\n")
    (output / "report.md").write_text(build_report(manifest, audit, summary, subject_rows))
    print(f"Baseline artifacts written to {output}", flush=True)


if __name__ == "__main__":
    main()
