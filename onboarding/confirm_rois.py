"""Step 2: build a contact sheet of every eye crop for a one-glance QC.

Usage: python onboarding/confirm_rois.py

Crops each eye in new_subject_rois.json from a representative frame and tiles them into
debug_masks/onboarding/roi_confirm.png. Scan it: any cell not centred on an eye (eyebrow,
cheek, edge) is a bad ROI. Fix by hand-editing new_subject_rois.json (mirror a good eye,
or set the box), or drop the sequence, then re-run this. Both eyes of a clip must be good.
"""
import json, collections, math
from pathlib import Path
import numpy as np, cv2

REPO = Path(__file__).resolve().parent.parent
ROIS = REPO / "new_subject_rois.json"
OUT = REPO / "debug_masks" / "onboarding" / "roi_confirm.png"
TH, LAB, COLS, GAP = 190, 22, 6, 6


def main():
    rois = json.load(open(ROIS))
    byclip = collections.defaultdict(dict)
    for k, box in rois.items():
        stem, side = k.rsplit("_", 1); byclip[stem][side] = box

    cells = []
    for stem in sorted(byclip):
        clip = next(p for p in (REPO / "data" / "clips").glob("*") if p.stem == stem)
        c = cv2.VideoCapture(str(clip)); n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
        c.set(cv2.CAP_PROP_POS_FRAMES, int(0.35 * n)); ok, fr = c.read(); c.release()
        if not ok: continue
        H, W = fr.shape[:2]
        for side in ("L", "R"):
            if side not in byclip[stem]: continue
            x, y, w, h = byclip[stem][side]
            crop = fr[max(0, y):min(H, y + h), max(0, x):min(W, x + w)]
            thumb = cv2.resize(crop, (TH, TH)) if crop.size else np.zeros((TH, TH, 3), np.uint8)
            cell = np.full((TH + LAB, TH, 3), 255, np.uint8); cell[LAB:] = thumb
            cv2.rectangle(cell, (0, 0), (TH - 1, TH + LAB - 1), (60, 60, 60), 1)
            cv2.putText(cell, f"{stem.replace('subject_','')} {side}", (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
            cells.append(cell)

    rows = math.ceil(len(cells) / COLS)
    sheet = np.full((rows * (TH + LAB) + (rows + 1) * GAP, COLS * TH + (COLS + 1) * GAP, 3), 245, np.uint8)
    for i, cell in enumerate(cells):
        r, cc = divmod(i, COLS)
        yy, xx = GAP + r * (TH + LAB + GAP), GAP + cc * (TH + GAP)
        sheet[yy:yy + TH + LAB, xx:xx + TH] = cell
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), sheet)
    print(f"{len(cells)} eye crops -> {OUT.relative_to(REPO)}  ({rows}x{COLS} grid)")


if __name__ == "__main__":
    main()
