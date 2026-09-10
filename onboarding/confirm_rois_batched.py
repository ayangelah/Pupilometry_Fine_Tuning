#!/usr/bin/env python3
"""
Generate 5 contact sheets (batched) to verify eye ROIs across all frames.
Each sheet has ~8 clips so they're readable on screen.
"""
import json, collections, math
from pathlib import Path
import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
ROIS_FILE = REPO / "new_subject_rois.json"
CLIPS = REPO / "data" / "clips"
DEBUG_MASKS = REPO / "debug_masks" / "onboarding"
DEBUG_MASKS.mkdir(parents=True, exist_ok=True)

def main():
    rois = json.load(open(ROIS_FILE))

    # Group by clip stem
    byclip = collections.defaultdict(dict)
    for k, box in rois.items():
        # Strip eye label to get clip stem
        if k.endswith("_left_eye"):
            stem, side = k[:-9], "L"
        elif k.endswith("_right_eye"):
            stem, side = k[:-10], "R"
        else:
            stem, side = k.rsplit("_", 1)
        byclip[stem][side] = box

    clips_sorted = sorted(byclip.keys())
    clips_per_sheet = 8
    num_sheets = math.ceil(len(clips_sorted) / clips_per_sheet)

    print(f"Generating {num_sheets} confirmation sheets for {len(clips_sorted)} clips...")

    for sheet_num in range(num_sheets):
        start_idx = sheet_num * clips_per_sheet
        end_idx = min(start_idx + clips_per_sheet, len(clips_sorted))
        sheet_clips = clips_sorted[start_idx:end_idx]

        print(f"\nSheet {sheet_num + 1}/{num_sheets}: clips {start_idx + 1}-{end_idx}")

        cells = []
        for stem in sheet_clips:
            try:
                clip = next(p for p in CLIPS.glob("*") if p.stem == stem)
            except StopIteration:
                print(f"  Warning: clip {stem} not found")
                continue

            c = cv2.VideoCapture(str(clip))
            n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
            c.set(cv2.CAP_PROP_POS_FRAMES, int(0.35 * n))
            ok, fr = c.read()
            c.release()
            if not ok:
                continue

            H, W = fr.shape[:2]

            # Resize for sheet first
            cell_h = int(H * 0.15)
            cell_w = int(W * 0.15)
            cell = cv2.resize(fr, (cell_w, cell_h))

            # Draw boxes on resized frame (scaled coordinates)
            scale_h = cell_h / H
            scale_w = cell_w / W
            for side in ("L", "R"):
                if side not in byclip[stem]:
                    continue
                x, y, w, h = byclip[stem][side]
                # Scale coordinates to resized frame
                x_s, y_s = int(x * scale_w), int(y * scale_h)
                w_s, h_s = int(w * scale_w), int(h * scale_h)
                color = (0, 255, 0) if side == "L" else (0, 0, 255)  # green for L, red for R
                cv2.rectangle(cell, (x_s, y_s), (x_s + w_s, y_s + h_s), color, 3)  # thicker lines

            # Add label
            label = f"{stem}"
            cv2.putText(cell, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
            cells.append(cell)

        # Assemble into a grid (2 columns)
        cols = 2
        rows = math.ceil(len(cells) / cols)
        if cells:
            # Find max dimensions and pad all cells to same size
            max_h = max(c.shape[0] for c in cells)
            max_w = max(c.shape[1] for c in cells)
            padded_cells = []
            for cell in cells:
                # Pad to max size
                pad_h = max_h - cell.shape[0]
                pad_w = max_w - cell.shape[1]
                padded = cv2.copyMakeBorder(cell, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                padded_cells.append(padded)

            grid = np.zeros((rows * max_h, cols * max_w, 3), dtype=np.uint8)
            for i, cell in enumerate(padded_cells):
                r, c = i // cols, i % cols
                grid[r * max_h:(r + 1) * max_h, c * max_w:(c + 1) * max_w] = cell

            sheet_path = DEBUG_MASKS / f"roi_confirm_sheet_{sheet_num + 1}.png"
            cv2.imwrite(str(sheet_path), grid)
            print(f"  ✓ saved {sheet_path}")

    print(f"\n✓ Generated {num_sheets} sheets to {DEBUG_MASKS}/roi_confirm_sheet_*.png")
    print("  Green boxes = left eye, Red boxes = right eye")

if __name__ == "__main__":
    main()
