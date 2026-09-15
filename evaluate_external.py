"""Evaluate the untouched RITnet checkpoint on licensed external RGB eye masks.

This is a read-only, raw-segmentation benchmark. Ground-truth masks define an oracle eye box,
which is expanded to the same 400x640 OpenEDS-style framing used by the phone pipeline. The
oracle box deliberately removes eye localization from the question; no circle fit or product
gate is scored because these datasets annotate visible semantic regions rather than full circles.

Only dataset-relative sample IDs and metrics are written. Source images/masks stay outside Git.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import platform
import re
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
DEFAULT_DATA_ROOT = ROOT.parent / "datasets"
DEFAULT_CHECKPOINT = ROOT / "RITnet" / "best_model.pkl"
BOOTSTRAP_SEED = 20260915
CLASS_NAMES = {0: "background", 1: "sclera", 2: "iris", 3: "pupil"}


def run_text(*command: str, cwd: Path = ROOT) -> str | None:
    try:
        return subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_hash(paths: Iterable[Path], base: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(path.relative_to(base).as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def synchronize() -> None:
    if str(pipeline.DEVICE) == "mps":
        torch.mps.synchronize()
    elif str(pipeline.DEVICE).startswith("cuda"):
        torch.cuda.synchronize()


def foreground(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"unreadable mask: {path}")
    return mask > 127


def load_sbvpi(data_root: Path) -> tuple[list[dict[str, Any]], list[Path], dict[str, Any]]:
    root = data_root / "SBVPI"
    metadata_path = root / "SBVPI_Gender_Age_Colour.txt"
    if not metadata_path.exists():
        raise SystemExit(f"SBVPI not found under {root}")
    colours = {
        str(index): line.split()[2].strip().lower()
        for index, line in enumerate(metadata_path.read_text().splitlines(), 1)
        if line.strip()
    }
    by_kind = {
        kind: {path.name.removesuffix(f"_{kind}.png"): path for path in root.glob(f"*/*_{kind}.png")}
        for kind in ("pupil", "iris", "sclera")
    }
    stems = sorted(set.intersection(*(set(paths) for paths in by_kind.values())))
    pattern = re.compile(r"^(\d+)([LR])_([lrsu])_(\d+)$")
    records, files = [], [metadata_path]
    for stem in stems:
        match = pattern.fullmatch(stem)
        if not match:
            raise ValueError(f"unexpected SBVPI sample name: {stem}")
        subject, eye, gaze, take = match.groups()
        pupil_path = by_kind["pupil"][stem]
        image_path = pupil_path.with_name(stem + ".jpg")
        if not image_path.exists():
            raise ValueError(f"missing SBVPI image: {image_path}")
        mask_paths = {kind: paths[stem] for kind, paths in by_kind.items()}
        records.append(
            {
                "dataset": "sbvpi",
                "sample_id": f"{subject}/{stem}",
                "subject": subject,
                "eye": eye,
                "gaze": gaze,
                "take": take,
                "eye_colour": colours.get(subject, "unknown"),
                "image_path": image_path,
                "mask_paths": mask_paths,
            }
        )
        files.extend((image_path, *mask_paths.values()))
    return records, files, {
        "available_rgb_images": len(list(root.glob("*/*.jpg"))),
        "pupil_masks": len(by_kind["pupil"]),
        "iris_masks": len(by_kind["iris"]),
        "sclera_masks": len(by_kind["sclera"]),
        "complete_evaluation_samples": len(records),
        "subjects": len({record["subject"] for record in records}),
    }


def load_mobius(data_root: Path) -> tuple[list[dict[str, Any]], list[Path], dict[str, Any]]:
    root = data_root / "MOBIUS (Segmentation)"
    metadata_path = root / "data.csv"
    if not metadata_path.exists():
        raise SystemExit(f"MOBIUS segmentation subset not found under {root}")
    with metadata_path.open(newline="") as source:
        colours = {row["ID"]: row["colour"].strip().lower() for row in csv.DictReader(source)}
    pattern = re.compile(r"^(\d+)_(\d)([inp])_([LR])([lrsu])_(\w+)$")
    records, files = [], [metadata_path]
    for image_path in sorted((root / "Images").glob("*.jpg")):
        match = pattern.fullmatch(image_path.stem)
        if not match:
            raise ValueError(f"unexpected MOBIUS sample name: {image_path.name}")
        subject, phone, light, eye, gaze, take = match.groups()
        mask_path = root / "Masks" / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise ValueError(f"missing MOBIUS mask: {mask_path}")
        records.append(
            {
                "dataset": "mobius",
                "sample_id": image_path.stem,
                "subject": subject,
                "eye": eye,
                "gaze": gaze,
                "take": take,
                "phone": phone,
                "lighting": light,
                "deliberately_unusable": take == "bad",
                "eye_colour": colours.get(subject, "unknown"),
                "image_path": image_path,
                "mask_paths": {"multiclass": mask_path},
            }
        )
        files.extend((image_path, mask_path))
    return records, files, {
        "paired_samples": len(records),
        "subjects": len({record["subject"] for record in records}),
        "phones": dict(sorted(Counter(record["phone"] for record in records).items())),
        "lighting": dict(sorted(Counter(record["lighting"] for record in records).items())),
        "eye_colours": dict(sorted(Counter(record["eye_colour"] for record in records).items())),
        "deliberately_unusable": sum(record["deliberately_unusable"] for record in records),
    }


def semantic_target(record: dict[str, Any]) -> np.ndarray:
    if record["dataset"] == "mobius":
        bgr = cv2.imread(str(record["mask_paths"]["multiclass"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"unreadable mask: {record['mask_paths']['multiclass']}")
        target = np.zeros(bgr.shape[:2], np.uint8)
        target[np.all(bgr == (0, 0, 255), axis=2)] = pipeline.CLS_SCLERA
        target[np.all(bgr == (0, 255, 0), axis=2)] = pipeline.CLS_IRIS
        target[np.all(bgr == (255, 0, 0), axis=2)] = pipeline.CLS_PUPIL
        return target

    pupil = foreground(record["mask_paths"]["pupil"])
    iris_disk = foreground(record["mask_paths"]["iris"])
    sclera = foreground(record["mask_paths"]["sclera"])
    if not (pupil.shape == iris_disk.shape == sclera.shape):
        raise ValueError(f"SBVPI mask dimensions disagree for {record['sample_id']}")
    target = np.zeros(pupil.shape, np.uint8)
    target[sclera] = pipeline.CLS_SCLERA
    target[iris_disk] = pipeline.CLS_IRIS  # SBVPI iris masks include the pupil disk.
    target[pupil] = pipeline.CLS_PUPIL
    return target


def oracle_reframe(image: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    visible_eye = target != 0
    ys, xs = np.where(visible_eye)
    if not len(xs):
        raise ValueError("ground-truth mask has no visible eye pixels")
    xmin, xmax, ymin, ymax = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    cx, cy = (xmin + xmax + 1) / 2, (ymin + ymax + 1) / 2
    width = max(2, int(math.ceil((xmax - xmin + 1) * 1.05)))
    height = max(2, int(math.ceil(width * pipeline.IN_H / pipeline.IN_W)))
    left, top = int(math.floor(cx - width / 2)), int(math.floor(cy - height / 2))
    right, bottom = left + width, top + height
    source_h, source_w = image.shape[:2]
    pad_left, pad_top = max(0, -left), max(0, -top)
    pad_right, pad_bottom = max(0, right - source_w), max(0, bottom - source_h)
    image_padded = cv2.copyMakeBorder(
        image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE
    )
    target_padded = cv2.copyMakeBorder(
        target, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0
    )
    left += pad_left; right += pad_left; top += pad_top; bottom += pad_top
    image_crop = image_padded[top:bottom, left:right]
    target_crop = target_padded[top:bottom, left:right]
    if image_crop.shape[:2] != (height, width):
        raise ValueError("oracle crop geometry is inconsistent")
    return (
        cv2.resize(image_crop, (pipeline.IN_W, pipeline.IN_H), interpolation=cv2.INTER_AREA),
        cv2.resize(target_crop, (pipeline.IN_W, pipeline.IN_H), interpolation=cv2.INTER_NEAREST),
        (xmin, ymin, xmax - xmin + 1, ymax - ymin + 1),
    )


def binary_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth, prediction = truth.astype(bool), prediction.astype(bool)
    intersection = int(np.logical_and(truth, prediction).sum())
    truth_n, prediction_n = int(truth.sum()), int(prediction.sum())
    union = truth_n + prediction_n - intersection
    return {
        "iou": intersection / union if union else 1.0,
        "dice": 2 * intersection / (truth_n + prediction_n) if truth_n + prediction_n else 1.0,
        "precision": intersection / prediction_n if prediction_n else 0.0,
        "recall": intersection / truth_n if truth_n else 0.0,
    }


def evaluate_frame(record: dict[str, Any], channel: str, target: np.ndarray, prediction: np.ndarray,
                   model_ms: float, oracle_box: tuple[int, int, int, int]) -> dict[str, Any]:
    row = {
        key: value for key, value in record.items()
        if key not in {"image_path", "mask_paths"}
    }
    row.update(channel=channel, model_ms=model_ms, oracle_box="_".join(map(str, oracle_box)))
    regions = {
        "pupil": target == pipeline.CLS_PUPIL,
        "iris_ring": target == pipeline.CLS_IRIS,
        "iris_disk": (target == pipeline.CLS_IRIS) | (target == pipeline.CLS_PUPIL),
        "sclera": target == pipeline.CLS_SCLERA,
    }
    predicted = {
        "pupil": prediction == pipeline.CLS_PUPIL,
        "iris_ring": prediction == pipeline.CLS_IRIS,
        "iris_disk": (prediction == pipeline.CLS_IRIS) | (prediction == pipeline.CLS_PUPIL),
        "sclera": prediction == pipeline.CLS_SCLERA,
    }
    for region in regions:
        for metric, value in binary_metrics(regions[region], predicted[region]).items():
            row[f"{region}_{metric}"] = value
    return row


def evaluate(records: list[dict[str, Any]], channels: list[str], batch_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pipeline.model.eval()
    print(
        f"Streaming {len(records)} samples; channels={','.join(channels)}; batch={batch_size}",
        flush=True,
    )
    # Prepare only one batch at a time. The full MOBIUS set would require several GB if all
    # resized RGB crops were retained. Each source image/mask is decoded once, both channel
    # ablations run while that batch is resident, then the pixels are released.
    for start in range(0, len(records), batch_size):
        batch = []
        for record in records[start:start + batch_size]:
            image = cv2.imread(str(record["image_path"]), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"unreadable image: {record['image_path']}")
            target = semantic_target(record)
            if target.shape != image.shape[:2]:
                raise ValueError(f"image/mask dimensions disagree for {record['sample_id']}")
            image, target, oracle_box = oracle_reframe(image, target)
            batch.append((record, image, target, oracle_box))

        for channel in channels:
            pipeline.USE_RED_CHANNEL = channel == "red"
            tensors = torch.cat(
                [pipeline.preprocess(pipeline.ritnet_input(image)) for _, image, _, _ in batch], dim=0
            ).to(pipeline.DEVICE)
            synchronize(); started = time.perf_counter()
            with torch.inference_mode():
                logits = pipeline.model(tensors)
            synchronize()
            model_ms = (time.perf_counter() - started) * 1000 / len(batch)
            masks = logits.argmax(1).cpu().numpy().astype(np.uint8)
            for (record, _, target, oracle_box), prediction in zip(batch, masks):
                rows.append(evaluate_frame(record, channel, target, prediction, model_ms, oracle_box))
        if start == 0 or (start + batch_size) % 80 == 0 or start + batch_size >= len(records):
            print(f"  {min(start + batch_size, len(records))}/{len(records)}", flush=True)
    return rows


METRICS = [
    f"{region}_{metric}"
    for region in ("pupil", "iris_disk", "iris_ring", "sclera")
    for metric in ("iou", "dice", "precision", "recall")
]


def mean_ci(values: list[float], samples: int = 10_000) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"n": 0, "mean": None, "ci95": [None, None]}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return {
        "n": len(array),
        "mean": float(array.mean()),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
    }


def summarize(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary, subject_rows = {}, []
    for channel in sorted({row["channel"] for row in rows}):
        channel_rows = [row for row in rows if row["channel"] == channel]
        subjects = sorted({row["subject"] for row in channel_rows}, key=lambda value: int(value))
        for subject in subjects:
            group = [row for row in channel_rows if row["subject"] == subject]
            result = {"channel": channel, "subject": subject, "frames": len(group)}
            result.update({metric: float(np.mean([row[metric] for row in group])) for metric in METRICS})
            subject_rows.append(result)
        subject_channel = [row for row in subject_rows if row["channel"] == channel]
        summary[channel] = {
            "counts": {"frames": len(channel_rows), "subjects": len(subjects)},
            "subject_macro": {
                metric: mean_ci([row[metric] for row in subject_channel]) for metric in METRICS
            },
            "frame_mean": {
                metric: float(np.mean([row[metric] for row in channel_rows])) for metric in METRICS
            },
            "by_eye_colour": {
                colour: {
                    "frames": len(group),
                    **{metric: float(np.mean([row[metric] for row in group])) for metric in METRICS},
                }
                for colour in sorted({row["eye_colour"] for row in channel_rows})
                for group in [[row for row in channel_rows if row["eye_colour"] == colour]]
            },
        }
    return summary, subject_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def metric_cell(value: dict[str, Any]) -> str:
    return f"{value['mean']:.3f} [{value['ci95'][0]:.3f}, {value['ci95'][1]:.3f}]"


def report(dataset: str, manifest: dict[str, Any], summary: dict[str, Any]) -> str:
    lines = [
        f"# Frozen pretrained RITnet external baseline: {dataset.upper()}", "",
        f"Run: `{manifest['run_id']}`  ",
        f"Checkpoint SHA-256: `{manifest['checkpoint_sha256']}`  ",
        f"Dataset inventory SHA-256: `{manifest['dataset_inventory_sha256']}`", "",
        "This is raw semantic segmentation with an oracle ground-truth eye box. It does not test eye localization, temporal tracking, circle measurement, or PLR dynamics.", "",
        "| Metric | " + " | ".join(summary) + " |",
        "|---|" + "---|" * len(summary),
    ]
    for label, metric in (
        ("Pupil IoU", "pupil_iou"),
        ("Pupil recall", "pupil_recall"),
        ("Pupil precision", "pupil_precision"),
        ("Iris-disk IoU", "iris_disk_iou"),
        ("Sclera IoU", "sclera_iou"),
    ):
        lines.append(
            f"| {label} | " + " | ".join(metric_cell(summary[ch]["subject_macro"][metric]) for ch in summary) + " |"
        )
    lines.extend(("", "## Contract", "", 
        "- Source weights are untouched `RITnet/best_model.pkl`; channels are preprocessing ablations only.",
        "- The visible-eye ground-truth union defines a tight box, expanded to the app's 400x640 OpenEDS-style aspect without geometric stretching.",
        "- Confidence intervals bootstrap subject-level means. Frame means remain available in `summary.json`.",
        "- SBVPI evaluation includes only samples with pupil, iris, and sclera masks; missing class annotations are never treated as negatives.",
        "- No source image, source mask, or rendered preview is copied into the run artifacts.",
    ))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("sbvpi", "mobius"), required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--channels", nargs="+", choices=("gray", "red"), default=("gray", "red"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    loader = load_sbvpi if args.dataset == "sbvpi" else load_mobius
    records, source_files, inventory = loader(args.data_root.resolve())
    if not records:
        raise SystemExit("no complete evaluation samples")
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"ritnet_external_{args.dataset}_{timestamp}"
    output = (args.output or ROOT / "evaluation" / "runs" / run_id).resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset": args.dataset,
        "dataset_inventory": inventory,
        "dataset_source_file_count": len(set(source_files)),
        "dataset_inventory_sha256": inventory_hash(source_files, args.data_root.resolve()),
        "checkpoint_sha256": sha256_file(DEFAULT_CHECKPOINT),
        "parameter_count": sum(parameter.numel() for parameter in pipeline.model.parameters()),
        "source_sha256": {
            "evaluate_external.py": sha256_file(Path(__file__)),
            "ritnet_segment.py": sha256_file(ROOT / "ritnet_segment.py"),
            "densenet.py": sha256_file(ROOT / "RITnet" / "densenet.py"),
        },
        "git_commit": run_text("git", "rev-parse", "HEAD"),
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
            "framing": "oracle visible-eye union; width_fill=1.05; aspect-preserving crop with replicated image padding",
            "preprocessing": "selected RGB channel -> gamma 0.8 -> CLAHE(1.5, 8x8) -> [0,1] -> [-1,1]",
            "postprocessing": "none; argmax class mask only",
            "class_ids": CLASS_NAMES,
        },
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = evaluate(records, list(args.channels), args.batch_size)
    summary, subject_rows = summarize(rows)
    write_csv(output / "frame_metrics.csv", rows)
    write_csv(output / "subject_metrics.csv", subject_rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "report.md").write_text(report(args.dataset, manifest, summary))
    print(f"External baseline artifacts written to {output}", flush=True)


if __name__ == "__main__":
    main()
