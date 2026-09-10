#!/usr/bin/env python3
"""
Simple click-drag eye boxing - no tagging needed, auto-labels based on known eye classifications.

Usage: python onboarding/box_eyes.py subject_a_1 subject_a_2 subject_b_1

For each clip: matplotlib window shows the first frame.
  1. Drag a box around each eye visible
  2. Close window when done (or press Q)
  3. Boxes auto-labeled based on known classification (L/R/B)
  4. Moves to next clip

Boxes saved to new_subject_rois.json.
"""
import sys, json
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

REPO = Path(__file__).resolve().parent.parent
ROIS = REPO / "new_subject_rois.json"
CLIPS = REPO / "data" / "clips"

# Eye classifications (from user's tagging: 1=B, 2=B, ..., 40=B)
EYE_CLASSES = {
    "subject_a_1": "B", "subject_a_2": "B", "subject_a_4": "B", "subject_a_5": "B",
    "subject_a_6": "B", "subject_a_7": "L", "subject_a_8": "B",
    "subject_b_1": "L", "subject_b_2": "L", "subject_b_3": "B", "subject_b_4": "B",
    "subject_c_1": "R",
    "subject_d_1": "R", "subject_d_2": "B",
    "subject_e_1": "B", "subject_e_2": "B", "subject_e_3": "L", "subject_e_4": "R",
    "subject_f_1": "B", "subject_f_2": "B", "subject_f_3": "L", "subject_f_4": "R",
    "subject_g_1": "L", "subject_g_2": "B",
    "subject_h_1": "R", "subject_h_2": "B", "subject_h_3": "L", "subject_h_4": "B",
    "subject_i_1": "B", "subject_i_2": "B", "subject_i_3": "L", "subject_i_4": "R",
    "subject_j_1": "R", "subject_j_2": "L", "subject_j_3": "B", "subject_j_4": "B",
    "subject_k_1": "L", "subject_k_2": "R", "subject_k_3": "B", "subject_k_4": "B",
}

def get_first_frame(clip_stem):
    """Load the first frame of a clip."""
    try:
        clip_path = next(CLIPS.glob(f"{clip_stem}.*"))
    except StopIteration:
        print(f"Error: no clip found for {clip_stem}")
        return None
    cap = cv2.VideoCapture(str(clip_path))
    ok, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None

def box_one_clip(clip_stem):
    """Interactive boxing for one clip."""
    if clip_stem not in EYE_CLASSES:
        print(f"Skipping {clip_stem}: not in eye classifications")
        return False

    frame = get_first_frame(clip_stem)
    if frame is None:
        print(f"Skipping {clip_stem}: frame load failed")
        return False

    eye_class = EYE_CLASSES[clip_stem]
    boxes = []
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(frame)
    ax.set_title(f"{clip_stem} ({eye_class}): drag box(es), close when done")

    def on_select(eclick, erelease):
        x0, x1 = sorted([eclick.xdata, erelease.xdata])
        y0, y1 = sorted([eclick.ydata, erelease.ydata])
        box = [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]
        boxes.append(box)

        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="lime", linewidth=2)
        ax.add_patch(rect)
        ax.text(x0, y0 - 8, str(len(boxes)), color="lime", fontsize=10)
        fig.canvas.draw_idle()
        print(f"  Box {len(boxes)}: {box}")

    def on_key(event):
        if event.key.upper() == 'Q':
            plt.close(fig)

    rs = RectangleSelector(
        ax, on_select, useblit=True, button=[1],
        minspanx=20, minspany=20, interactive=True
    )
    fig.canvas.mpl_connect('key_press_event', on_key)

    print(f"\n{clip_stem} ({eye_class}): draw {eye_class} eye box(es), close window when done")
    plt.show()

    if not boxes:
        print(f"  → no boxes drawn, skipping")
        return False

    # Save with auto-labels based on eye_class
    rois = json.load(open(ROIS)) if ROIS.exists() else {}

    # Clear old entries for this clip
    for k in list(rois.keys()):
        if k.startswith(clip_stem):
            del rois[k]

    # Label boxes based on known eye_class
    if eye_class == 'L':
        # Left eye only: keep first (only) box
        rois[f"{clip_stem}_left_eye"] = boxes[0]
        print(f"  ✓ {clip_stem}_left_eye: {boxes[0]}")
    elif eye_class == 'R':
        # Right eye only: keep first (only) box
        rois[f"{clip_stem}_right_eye"] = boxes[0]
        print(f"  ✓ {clip_stem}_right_eye: {boxes[0]}")
    else:  # 'B' for both
        # Both eyes: sort by x position (left, then right)
        sorted_boxes = sorted(boxes, key=lambda b: b[0])
        rois[f"{clip_stem}_left_eye"] = sorted_boxes[0]
        print(f"  ✓ {clip_stem}_left_eye: {sorted_boxes[0]}")
        if len(sorted_boxes) > 1:
            rois[f"{clip_stem}_right_eye"] = sorted_boxes[1]
            print(f"  ✓ {clip_stem}_right_eye: {sorted_boxes[1]}")

    json.dump(rois, open(ROIS, "w"), indent=1)
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python onboarding/box_eyes.py subject_a_1 subject_a_2 subject_b_1")
        print("       (list clips you want to box)")
        sys.exit(1)

    clips = sys.argv[1:]
    for clip_stem in clips:
        box_one_clip(clip_stem)

    print("\nDone. Boxes saved to new_subject_rois.json")

if __name__ == "__main__":
    main()
