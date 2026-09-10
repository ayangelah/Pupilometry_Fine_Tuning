"""Convert Labelme JSON -> ground-truth artifacts, per the labeling convention:
  - iris  : circle (center, edge)          -> always present
  - pupil : circle (center, edge)          -> ABSENT means "pupil too dark to see"
  - sclera: polygon                        -> sparse/optional
  - no JSON for a frame                    -> REJECT (unusable: blink/blur)

Reads the hand labels next to their crops in labeling/frames_v2/<seq_id>/, where seq_id is the
eye sequence (subject_x_n_left_eye / _right_eye). Frames are keyed by eye, not by clip, because
each eye of a two-eye clip is its own measurement sequence.

Outputs (native crop resolution, so they align to predictions by one resize):
  labeling/gt/<seq_id>/<name>_mask.png   4-class mask (0 bg, 1 sclera, 2 iris, 3 pupil)
  labeling/gt.csv                        per-frame table: category + circle params (px)

Both outputs are derived: this script is the only thing that should write them, so gt.csv stays
reproducible from the JSONs. Rerun it after any labeling change.
"""
import json, glob, os, csv, math
import numpy as np, cv2

BG, SCLERA, IRIS, PUPIL = 0, 1, 2, 3   # RITnet class ids
GT_DIR = "labeling/gt"
FRAMES = "labeling/frames_v2"
MANIFEST = "labeling/manifest_v2.csv"

# Frames to force to 'reject' despite having a JSON (blink/occlusion: iris only a sliver behind
# the lid, so a weak label and not meaningful to score). Non-destructive: keeps the JSON on disk.
REJECT_OVERRIDE = set()

def circle_of(shape):
    (cx, cy), (ex, ey) = shape["points"]
    return cx, cy, math.hypot(ex - cx, ey - cy)

manifest = {}
with open(MANIFEST) as f:
    for r in csv.DictReader(f):
        manifest[os.path.basename(r["path"])] = r

rows = []
for png in sorted(glob.glob(f"{FRAMES}/*/*.png")):
    name = os.path.basename(png)
    seq_id = os.path.basename(os.path.dirname(png))
    person = manifest.get(name, {}).get("person", "?")
    jp = png[:-4] + ".json"
    rec = dict(seq_id=seq_id, person=person, frame=name,
               iris_cx="", iris_cy="", iris_r="", pupil_cx="", pupil_cy="", pupil_r="")

    if not os.path.exists(jp) or name in REJECT_OVERRIDE:   # no file, or forced = reject
        rows.append({**rec, "category": "reject"}); continue

    d = json.load(open(jp))
    H, W = d["imageHeight"], d["imageWidth"]
    mask = np.zeros((H, W), np.uint8)
    shapes = {s["label"]: s for s in d["shapes"]}   # one of each label per frame

    if "sclera" in shapes:
        pts = np.array(shapes["sclera"]["points"], np.int32)
        cv2.fillPoly(mask, [pts], SCLERA)
    if "iris" in shapes:
        cx, cy, r = circle_of(shapes["iris"])
        cv2.circle(mask, (round(cx), round(cy)), round(r), IRIS, -1)
        rec.update(iris_cx=round(cx, 2), iris_cy=round(cy, 2), iris_r=round(r, 2))
    if "pupil" in shapes:                            # absent = pupil too dark
        cx, cy, r = circle_of(shapes["pupil"])
        cv2.circle(mask, (round(cx), round(cy)), round(r), PUPIL, -1)
        rec.update(pupil_cx=round(cx, 2), pupil_cy=round(cy, 2), pupil_r=round(r, 2))

    category = ("iris_only" if "pupil" not in shapes else "full")
    outdir = os.path.join(GT_DIR, seq_id); os.makedirs(outdir, exist_ok=True)
    cv2.imwrite(os.path.join(outdir, name[:-4] + "_mask.png"), mask)
    rows.append({**rec, "category": category})

fields = ["seq_id", "person", "frame", "category",
          "iris_cx", "iris_cy", "iris_r", "pupil_cx", "pupil_cy", "pupil_r"]
with open("labeling/gt.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

from collections import Counter
c = Counter(r["category"] for r in rows)
print(f"{len(rows)} frames -> gt.csv | " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))
print(f"masks (4-class) written under {GT_DIR}/")
