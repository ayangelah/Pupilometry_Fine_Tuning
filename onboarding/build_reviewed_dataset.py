"""Build label-ready eye crops from the hand-reviewed, per-frame eyeboxes.

The Labelme rectangles in ``labeling/eyebox_review`` are the source of truth; no detector box
participates. ``--check`` validates and compares with the current manifest without writing.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2

try:
    from .extract_frames import sample_positions
except ImportError:  # direct script execution: onboarding/ is already on sys.path
    from extract_frames import sample_positions

REPO = Path(__file__).resolve().parent.parent
CLIPS = REPO / "data" / "clips"
REVIEW = REPO / "labeling" / "eyebox_review"
FRAMES = REPO / "labeling" / "frames_v2"
MANIFEST = REPO / "labeling" / "manifest_v2.csv"
INVENTORY = REPO / "onboarding" / "eye_inventory.json"
TRACKING_ROIS = REPO / "new_subject_rois.json"
IN_W, IN_H, WIDTH_FILL = 400, 640, 1.05
VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv", ".m4v"}


def split_sample(name: str) -> tuple[str, str]:
    clip, tag = Path(name).stem.rsplit("_", 1)
    if tag != "full" and not (len(tag) == 3 and tag.startswith("f") and tag[1:].isdigit()):
        raise ValueError(f"unexpected review filename: {name}")
    return clip, tag


def rectangle(shape: dict) -> tuple[int, int, int, int]:
    if shape.get("label") != "eyebox" or shape.get("shape_type") != "rectangle":
        raise ValueError("annotations must be Labelme rectangles labelled 'eyebox'")
    if len(shape.get("points", [])) != 2:
        raise ValueError("eyebox rectangle must have exactly two points")
    (x0, y0), (x1, y1) = shape["points"]
    left, right = sorted((x0, x1)); top, bottom = sorted((y0, y1))
    # Match the original rebuild exactly: quantize corners before taking dimensions.
    left, right, top, bottom = int(left), int(right), int(top), int(bottom)
    return left, top, right - left, bottom - top


def side_boxes(boxes: list[tuple[int, int, int, int]], eye_class: str, width: int):
    """Attach anatomical eye names to boxes from an unmirrored, patient-facing image.

    The capture app uses the rear camera.  Therefore the patient's left eye appears on the
    right side of the saved image (and the patient's right appears image-left).
    """
    boxes = sorted(boxes, key=lambda b: b[0] + b[2] / 2)
    if eye_class == "L":
        if len(boxes) != 1: raise ValueError(f"L clip needs one eyebox, found {len(boxes)}")
        return [("left", boxes[0])]
    if eye_class == "R":
        if len(boxes) != 1: raise ValueError(f"R clip needs one eyebox, found {len(boxes)}")
        return [("right", boxes[0])]
    if eye_class != "B" or not 1 <= len(boxes) <= 2:
        raise ValueError(f"B clip needs one or two visible eyeboxes, found {len(boxes)}")
    if len(boxes) == 2:
        return [("right", boxes[0]), ("left", boxes[1])]
    box = boxes[0]
    return [("right" if box[0] + box[2] / 2 < width / 2 else "left", box)]


def framed_roi(box, frame_shape):
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    new_w, new_h = w * WIDTH_FILL, w * WIDTH_FILL * (IN_H / IN_W)
    nx, ny = cx - new_w / 2.0, cy - new_h / 2.0
    height, width = frame_shape[:2]
    nx = max(0, min(nx, width - 1)); ny = max(0, min(ny, height - 1))
    return int(nx), int(ny), int(min(new_w, width - nx)), int(min(new_h, height - ny))


def existing_rows() -> dict[tuple[str, int], dict]:
    if not MANIFEST.exists(): return {}
    return {(r["clip"], int(r["frame"])): r for r in csv.DictReader(MANIFEST.open())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--clips",
        nargs="+",
        metavar="CLIP",
        help=(
            "rewrite crop PNGs only for these source clip stems while still rebuilding the "
            "complete manifest (useful after correcting one clip's eye identity)"
        ),
    )
    args = parser.parse_args()
    selected_clips = set(args.clips or [])
    inventory, old = json.loads(INVENTORY.read_text()), existing_rows()
    rows, seed_rois, mismatches, errors, video_meta, captures = [], {}, [], [], {}, {}
    clips = {p.stem: p for p in CLIPS.iterdir() if p.suffix.lower() in VIDEO_EXTS}
    unknown_clips = selected_clips - clips.keys()
    if unknown_clips:
        raise SystemExit("unknown --clips value(s): " + ", ".join(sorted(unknown_clips)))

    for label_path in sorted(REVIEW.glob("*.json")):
        try:
            clip_name, tag = split_sample(label_path.name)
            if clip_name not in inventory: raise ValueError(f"{clip_name} missing from eye_inventory.json")
            if clip_name not in clips: raise ValueError(f"video for {clip_name} not found")
            data = json.loads(label_path.read_text())
            boxes = [rectangle(shape) for shape in data.get("shapes", [])]
            if not boxes: raise ValueError("no eyebox rectangles")
            if clip_name not in video_meta:
                probe = cv2.VideoCapture(str(clips[clip_name]))
                video_meta[clip_name] = (int(probe.get(cv2.CAP_PROP_FRAME_COUNT)),
                                         float(probe.get(cv2.CAP_PROP_FPS)))
                if args.check or (selected_clips and clip_name not in selected_clips):
                    probe.release()
                else:
                    captures[clip_name] = probe
            n, fps = video_meta[clip_name]; frame_index = sample_positions(n)[tag]
            frame, time_s = None, frame_index / fps
            write_crop = not args.check and (not selected_clips or clip_name in selected_clips)
            if write_crop:
                cap = captures[clip_name]
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index); ok, frame = cap.read()
                time_s = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                if not ok: raise ValueError(f"could not read frame {frame_index}")
                if not math.isfinite(time_s) or (frame_index and time_s <= 0): time_s = frame_index / fps
            height, width = int(data["imageHeight"]), int(data["imageWidth"])
            for side, box in side_boxes(boxes, inventory[clip_name], width):
                seq, name = f"{clip_name}_{side}_eye", f"{clip_name}_{side}_eye_{tag}.png"
                path = f"labeling/frames_v2/{seq}/{name}"
                row = dict(clip=seq, person=clip_name.split("_")[1].upper(), frame=frame_index,
                           time_s=f"{time_s:.6f}", box="_".join(map(str, box)),
                           orig_h=height, orig_w=width, path=path)
                rows.append(row)
                previous = old.get((seq, frame_index))
                if previous and previous["box"] != row["box"]:
                    mismatches.append(f"{seq} frame {frame_index}: {previous['box']} != {row['box']}")
                if tag == "full": seed_rois[seq] = list(box)
                if write_crop:
                    assert frame is not None
                    x, y, w, h = framed_roi(box, frame.shape)
                    eye = cv2.resize(frame[y:y+h, x:x+w], (IN_W, IN_H), interpolation=cv2.INTER_AREA)
                    destination = REPO / path; destination.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(destination), eye)
        except Exception as exc:
            errors.append(f"{label_path.name}: {exc}")

    for capture in captures.values():
        capture.release()
    if errors: raise SystemExit("\n".join(errors))
    if not args.check:
        fields = ["clip", "person", "frame", "time_s", "box", "orig_h", "orig_w", "path"]
        with MANIFEST.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
        prior = json.loads(TRACKING_ROIS.read_text()) if TRACKING_ROIS.exists() else {}
        prior.update(seed_rois); TRACKING_ROIS.write_text(json.dumps(prior, indent=2) + "\n")
    print(f"validated {len(rows)} eye crops from {len(list(REVIEW.glob('*.json')))} reviewed frames")
    print(f"box mismatches versus current manifest: {len(mismatches)}")
    for mismatch in mismatches[:10]: print("  " + mismatch)
    if args.check:
        print("check only: no files written")
    elif selected_clips:
        print(f"wrote {MANIFEST}; rewrote crops only for {', '.join(sorted(selected_clips))}")
    else:
        print(f"wrote {MANIFEST} and {FRAMES}")


if __name__ == "__main__":
    main()
