"""Step 1: auto-detect + clean both-eye ROIs for new full-face clips.

Usage:
    python onboarding/detect_rois.py j          # all subject_j_* clips (a subject letter)
    python onboarding/detect_rois.py subject_j_1 subject_j_2   # specific clip stems
    python onboarding/detect_rois.py            # no args: list un-onboarded subjects, then stop

Explicit by design: it never auto-processes every clip (that would re-detect dropped or
legacy-protocol clips). Name the new subject(s) you just added.

Detects both eyes per clip (Haar), then cleans: drops a box much smaller than its
partner or far off its partner's height (false detection), and mirror-fills a missing
eye across the face midline. Writes/updates new_subject_rois.json (keys <stem>_L/<stem>_R)
and an annotated frame per clip to debug_masks/onboarding/. Review with confirm_rois.py;
hand-patch or drop any the contact sheet shows are still wrong.

Note: 4K random-seek is slow (~30-45s/clip). Haar is imperfect on dim/gaze-off frames,
so a manual QC pass on the contact sheet is expected, not optional.
"""
import sys, json, re
from pathlib import Path
import numpy as np, cv2

REPO = Path(__file__).resolve().parent.parent
CLIPS = REPO / "data" / "clips"
ROIS = REPO / "new_subject_rois.json"
WORK = REPO / "debug_masks" / "onboarding"; WORK.mkdir(parents=True, exist_ok=True)
EYE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def detect(stem, n_sample=15, detect_w=640):
    """Sample frames across the clip, Haar-detect eyes, return median L/R tight boxes + W."""
    clip = next(p for p in CLIPS.glob("*") if p.stem == stem)
    cap = cv2.VideoCapture(str(clip)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = W / detect_w
    boxes, frame = [], None
    for fi in np.linspace(n * 0.1, n * 0.9, n_sample, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, fr = cap.read()
        if not ok: continue
        if frame is None: frame = fr.copy()
        g = cv2.cvtColor(cv2.resize(fr, (detect_w, int(H / scale))), cv2.COLOR_BGR2GRAY)
        g = cv2.createCLAHE(2.0, (8, 8)).apply(g)                 # lift dim clips
        for (x, y, w, h) in EYE.detectMultiScale(g, 1.1, 5, minSize=(24, 24)):
            if (y + h / 2) / (H / scale) < 0.55:                  # eyes sit in the upper face
                boxes.append([x * scale, y * scale, w * scale, h * scale, (x + w / 2) * scale])
    cap.release()
    if len(boxes) < 1: return {}, frame, W
    boxes = np.array(boxes)
    out = {}
    for side, grp in (("L", boxes[boxes[:, 4] < W / 2]), ("R", boxes[boxes[:, 4] >= W / 2])):
        if len(grp): out[side] = [int(v) for v in np.median(grp[:, :4], axis=0)]
    return out, frame, W


def clean(sides, W):
    """Reject a box much smaller / far off its partner's height, then mirror-fill a missing eye."""
    L, R = sides.get("L"), sides.get("R")
    if L and R:                                                   # size symmetry
        if   L[2] < 0.55 * R[2]: L = None
        elif R[2] < 0.55 * L[2]: R = None
    if L and R:                                                   # height symmetry (a low box = cheek)
        if abs((L[1] + L[3] / 2) - (R[1] + R[3] / 2)) > 0.5 * min(L[3], R[3]):
            if L[1] > R[1]: L = None
            else:           R = None
    if L and not R: R = [W - L[0] - L[2], L[1], L[2], L[3]]       # mirror across face midline
    if R and not L: L = [W - R[0] - R[2], R[1], R[2], R[3]]
    return {k: v for k, v in (("L", L), ("R", R)) if v}


def main():
    rois = json.load(open(ROIS)) if ROIS.exists() else {}
    done = {k.rsplit("_", 1)[0] for k in rois}
    old = REPO / "eye_rois.json"                                  # legacy single-eye clips are onboarded too
    if old.exists():
        done |= set(json.load(open(old)))
    all_stems = sorted(p.stem for p in CLIPS.glob("*") if re.match(r"^subject_[a-z]+_\d+$", p.stem))
    if len(sys.argv) > 1:                                          # explicit: letters and/or full stems
        want = set(sys.argv[1:])
        stems = [s for s in all_stems if s in want or s.split("_")[1] in want]
    else:                                                          # no args: just report, do not process
        todo = sorted({s.split("_")[1] for s in all_stems if s not in done})
        print("un-onboarded subject letters:", todo or "(none)")
        print("run e.g.  python onboarding/detect_rois.py " + (todo[0] if todo else "<letter>")); return
    if not stems:
        print("no matching clips in data/clips."); return
    for stem in stems:
        print(f"detecting {stem} ...", flush=True)
        try:
            raw, frame, W = detect(stem)
        except StopIteration:
            print(f"  {stem}: no clip file, skipped"); continue
        boxes = clean(raw, W)
        for side, box in boxes.items():
            rois[f"{stem}_{side}"] = box
        print(f"  {stem}: {boxes}" + ("  <-- missing an eye, check" if len(boxes) < 2 else ""))
        if frame is not None and boxes:
            vis = frame.copy()
            for side, (x, y, w, h) in boxes.items():
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 6)
                cv2.putText(vis, side, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6)
            cv2.imwrite(str(WORK / f"roi_{stem}.png"), cv2.resize(vis, (W // 4, vis.shape[0] // 4)))
    json.dump(rois, open(ROIS, "w"), indent=1)
    print(f"\n{len(rois)} eye-ROIs / {len(set(k.rsplit('_',1)[0] for k in rois))} clips -> {ROIS.name}. "
          f"Run confirm_rois.py to review.")


if __name__ == "__main__":
    main()
