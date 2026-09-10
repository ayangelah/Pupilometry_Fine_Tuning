"""Step 3: reframe + sample label frames from the confirmed ROIs.

Usage: python onboarding/extract_frames.py [N]     # N frames per eye-sequence, default 3

For each eye-sequence in new_subject_rois.json, samples N frames evenly across the clip
(spans the PLR: dilated dark-adapted -> constricted lit), reframes each to OpenEDS 400x640
(exactly what the model sees, so labels align), and writes to labeling/frames/<seq>/. Appends
manifest.csv (person = subject letter, for the LOSO split). Idempotent: sequences already in
manifest.csv are skipped, so it is safe to re-run after adding subjects.

Next: hand-label the crops in Labelme, convert to gt.csv/masks, add the new subject letters
to SUBJECTS in finetune_ritnet.py, re-run LOSO.
"""
import sys, json, csv
from pathlib import Path
import numpy as np, cv2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from ritnet_segment import framed_roi, crop, IN_W, IN_H

ROIS = REPO / "new_subject_rois.json"
FRAMES = REPO / "labeling" / "frames"
MANIFEST = REPO / "labeling" / "manifest.csv"


def extracted_frames():
    """clip -> set of frame indices already in the manifest (so densify never re-extracts them)."""
    import collections
    m = collections.defaultdict(set)
    if MANIFEST.exists():
        for r in csv.DictReader(open(MANIFEST)):
            m[r["clip"]].add(int(r["frame"]))
    return m


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    GAP = 15                                                     # skip a sample within this many frames of one already extracted
    rois = json.load(open(ROIS))
    have = extracted_frames()
    rows = []
    for seq_id, box in sorted(rois.items()):
        existing = set(have.get(seq_id, set()))                 # frames already extracted (preserve them)
        # Extract clip stem from seq_id (strip _left_eye or _right_eye suffix)
        if seq_id.endswith("_left_eye"):
            stem = seq_id[:-9]
        elif seq_id.endswith("_right_eye"):
            stem = seq_id[:-10]
        else:
            stem = seq_id.rsplit("_", 1)[0]  # fallback for old naming
        person = seq_id.split("_")[1].upper()                    # E / F / ...  (LOSO fold)
        try:
            clip = next(p for p in (REPO / "data" / "clips").glob("*") if p.stem == stem)
        except StopIteration:
            print(f"Warning: clip {stem} not found, skipping {seq_id}")
            continue
        cap = cv2.VideoCapture(str(clip)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 120.0
        outdir = FRAMES / seq_id; outdir.mkdir(parents=True, exist_ok=True)
        kept, had = 0, len(existing)
        for fi in np.linspace(0.08 * n, 0.92 * n, N, dtype=int):
            fi = int(fi)
            if any(abs(fi - e) < GAP for e in existing):        # already have a near-identical frame
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ok, fr = cap.read()
            if not ok: continue
            eye = cv2.resize(crop(fr, framed_roi(tuple(box), fr.shape)), (IN_W, IN_H),
                             interpolation=cv2.INTER_AREA)
            name = f"{seq_id}_f{fi:06d}.png"
            cv2.imwrite(str(outdir / name), eye); existing.add(fi); kept += 1
            rows.append([seq_id, person, fi, round(fi / fps, 3), "_".join(map(str, box)),
                         fr.shape[0], fr.shape[1], f"labeling/frames/{seq_id}/{name}"])
        cap.release()
        print(f"{seq_id} (person {person}): +{kept} frames (had {had})", flush=True)

    if not rows:
        print("\nnothing new to extract."); return
    write_header = not MANIFEST.exists()
    with open(MANIFEST, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["clip", "person", "frame", "time_s", "box", "orig_h", "orig_w", "path"])
        w.writerows(rows)
    print(f"\nextracted {len(rows)} frames / {len({r[0] for r in rows})} new eye-sequences / "
          f"{len({r[1] for r in rows})} subjects -> labeling/frames/, appended to manifest.csv")


if __name__ == "__main__":
    main()
