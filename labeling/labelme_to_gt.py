"""Convert Labelme JSON -> ground-truth artifacts, per the labeling convention:
  - iris  : circle (center, edge)          -> always present
  - pupil : circle (center, edge)          -> ABSENT means "boundary not human-observable"
  - sclera: polygon                        -> sparse/optional
  - reject: any small rectangle, by itself -> unusable blink/blur/occlusion
  - no JSON for a frame                    -> PENDING (not included in gt.csv)

Reads the hand labels next to their crops in labeling/frames_v2/<seq_id>/, where seq_id is the
eye sequence (subject_x_n_left_eye / _right_eye). Frames are keyed by eye, not by clip, because
each eye of a two-eye clip is its own measurement sequence.

Outputs (native crop resolution, so they align to predictions by one resize):
  labeling/gt/<seq_id>/<name>_mask.png   4-class mask (0 bg, 1 sclera, 2 iris, 3 pupil)
  labeling/gt.csv                        per-frame table: category + circle params (px)

Both outputs are derived: this script is the only thing that should write them, so gt.csv stays
reproducible from the JSONs. Rerun it after any labeling change.
"""
import argparse, json, glob, os, csv, math
from pathlib import Path
import numpy as np, cv2

BG, SCLERA, IRIS, PUPIL = 0, 1, 2, 3   # RITnet class ids
GT_DIR = "labeling/gt"
FRAMES = "labeling/frames_v2"
MANIFEST = "labeling/manifest_v2.csv"

def circle_of(shape):
    if shape.get("shape_type") != "circle" or len(shape.get("points", [])) != 2:
        raise ValueError(f"{shape.get('label')} must be a two-point Labelme circle")
    (cx, cy), (ex, ey) = shape["points"]
    return cx, cy, math.hypot(ex - cx, ey - cy)

def convert(strict=False):
    manifest_rows = list(csv.DictReader(open(MANIFEST)))
    expected = sorted((Path(r["path"]), r) for r in manifest_rows)
    manifest_paths = {p.as_posix() for p, _ in expected}
    orphan_pngs = sorted(
        p for p in glob.glob(f"{FRAMES}/*/*.png") if Path(p).as_posix() not in manifest_paths
    )
    orphan_jsons = [str(Path(p).with_suffix(".json")) for p in orphan_pngs
                    if Path(p).with_suffix(".json").exists()]
    rows, pending, errors = [], [], []
    # The reviewed manifest defines the evaluation universe. Globbing every directory here used
    # to let stale/orphan crops re-enter gt.csv even after the canonical eye assignment changed.
    for png_path, manifest_row in expected:
        png = png_path.as_posix()
        name = png_path.name; seq_id = png_path.parent.name
        person = manifest_row["person"]; jp = str(png_path.with_suffix(".json"))
        rec = dict(seq_id=seq_id, person=person, frame=name, visibility="",
                   iris_cx="", iris_cy="", iris_r="", pupil_cx="", pupil_cy="", pupil_r="")
        if not png_path.exists():
            errors.append(f"{png}: manifest frame is missing")
            continue
        if not os.path.exists(jp):
            pending.append(png); continue
        try:
            d = json.load(open(jp)); H, W = d["imageHeight"], d["imageWidth"]
            by_label = {}
            for shape in d.get("shapes", []):
                label = shape.get("label")
                if label not in {"iris", "pupil", "sclera", "reject"}:
                    raise ValueError(f"unknown label {label!r}")
                if label in by_label: raise ValueError(f"duplicate {label!r} label")
                by_label[label] = shape
            if "reject" in by_label:
                if len(by_label) != 1: raise ValueError("reject must be the only label on the frame")
                rows.append({**rec, "category": "reject", "visibility": "not_applicable"})
                continue
            if "iris" not in by_label: raise ValueError("non-reject frame requires an iris circle")

            mask = np.zeros((H, W), np.uint8)
            if "sclera" in by_label:
                if by_label["sclera"].get("shape_type") != "polygon":
                    raise ValueError("sclera must be a polygon")
                cv2.fillPoly(mask, [np.array(by_label["sclera"]["points"], np.int32)], SCLERA)
            cx, cy, r = circle_of(by_label["iris"])
            cv2.circle(mask, (round(cx), round(cy)), round(r), IRIS, -1)
            rec.update(iris_cx=round(cx, 2), iris_cy=round(cy, 2), iris_r=round(r, 2))
            if "pupil" in by_label:
                cx, cy, r = circle_of(by_label["pupil"])
                cv2.circle(mask, (round(cx), round(cy)), round(r), PUPIL, -1)
                rec.update(pupil_cx=round(cx, 2), pupil_cy=round(cy, 2), pupil_r=round(r, 2))
                category, visibility = "full", "visible"
            else:
                category, visibility = "iris_only", "not_visible"
            outdir = os.path.join(GT_DIR, seq_id); os.makedirs(outdir, exist_ok=True)
            cv2.imwrite(os.path.join(outdir, name[:-4] + "_mask.png"), mask)
            rows.append({**rec, "category": category, "visibility": visibility})
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    if errors or (strict and (pending or orphan_jsons)):
        detail = errors
        if strict and pending:
            detail.append(f"{len(pending)} manifest frame(s) have no JSON")
        if strict and orphan_jsons:
            detail.append(f"{len(orphan_jsons)} labeled orphan frame(s) are outside the manifest")
        raise SystemExit("\n".join(detail))
    fields = ["seq_id", "person", "frame", "category", "visibility",
              "iris_cx", "iris_cy", "iris_r", "pupil_cx", "pupil_cy", "pupil_r"]
    with open("labeling/gt.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n"); w.writeheader(); w.writerows(rows)
    from collections import Counter
    counts = Counter(r["category"] for r in rows)
    print(f"{len(rows)} labeled frames -> gt.csv | " +
          ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"pending (no JSON, excluded)={len(pending)} | masks under {GT_DIR}/")
    print(f"orphan PNGs outside manifest (ignored)={len(orphan_pngs)} | "
          f"with JSON={len(orphan_jsons)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="fail if any crop is still unlabeled")
    convert(strict=parser.parse_args().strict)
