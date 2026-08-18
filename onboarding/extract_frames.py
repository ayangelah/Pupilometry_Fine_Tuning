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


def already_extracted():
    if not MANIFEST.exists(): return set()
    return {r["clip"] for r in csv.DictReader(open(MANIFEST))}


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    rois = json.load(open(ROIS))
    done = already_extracted()
    rows = []
    for seq_id, box in sorted(rois.items()):
        if seq_id in done:
            print(f"{seq_id}: already extracted, skip"); continue
        stem = seq_id.rsplit("_", 1)[0]
        person = seq_id.split("_")[1].upper()                    # E / F / ...  (LOSO fold)
        clip = next(p for p in (REPO / "data" / "clips").glob("*") if p.stem == stem)
        cap = cv2.VideoCapture(str(clip)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 120.0
        outdir = FRAMES / seq_id; outdir.mkdir(parents=True, exist_ok=True)
        kept = 0
        for fi in np.linspace(0.08 * n, 0.92 * n, N, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, fr = cap.read()
            if not ok: continue
            eye = cv2.resize(crop(fr, framed_roi(tuple(box), fr.shape)), (IN_W, IN_H),
                             interpolation=cv2.INTER_AREA)
            name = f"{seq_id}_f{int(fi):06d}.png"
            cv2.imwrite(str(outdir / name), eye); kept += 1
            rows.append([seq_id, person, int(fi), round(fi / fps, 3), "_".join(map(str, box)),
                         fr.shape[0], fr.shape[1], f"labeling/frames/{seq_id}/{name}"])
        cap.release()
        print(f"{seq_id} (person {person}): {kept} frames", flush=True)

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
