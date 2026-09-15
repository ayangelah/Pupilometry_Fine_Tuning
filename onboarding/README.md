# Canonical data onboarding

This pipeline turns a new consented PLR clip into the reviewed eye crops used by training and
validation. Hand-drawn, per-frame eyeboxes are the source of truth. The older Haar/static-ROI
path remains available for experiments, but it does not write the canonical dataset.

## Flow

1. Put clips in `data/clips/`, named `subject_<letter>_<n>.MOV`.
2. Add each clip to `onboarding/eye_inventory.json`: `L`, `R`, or `B` for the visible eye(s).
3. Run `python onboarding/extract_frames.py [clip_stem ...]`.
4. Open `labeling/eyebox_review` in Labelme. Draw one `eyebox` rectangle around every visible
   eye in each sampled full-resolution frame.
5. Run `python onboarding/build_reviewed_dataset.py --check`. Resolve every reported error.
6. Run `python onboarding/build_reviewed_dataset.py`. This writes `frames_v2`,
   `manifest_v2.csv`, and the `_full` tracking seeds in `new_subject_rois.json`.
7. Label iris/pupil circles according to `labeling/HOW_TO_LABEL.md`.
8. Run `python labeling/labelme_to_gt.py --strict` to rebuild `gt.csv` and masks.

The sample set is 10/30/50/70/90% of each clip plus a 35% `full` frame. The builder uses the
same deterministic positions and OpenEDS reframing as the current 351-frame dataset. Its
`--check` mode reconstructs the crop metadata without writing files and compares it to the
existing manifest.

`left_eye` and `right_eye` are always the patient's anatomical sides. The capture app uses
unmirrored rear-camera video, so in a two-eye frame the patient's right eye is image-left and
the patient's left eye is image-right. Normalize mirrored imports before onboarding; `L`/`R`
inventory entries are authoritative for clips containing only one eye. Do not mark a clip `B`
merely because the eye crosses the image midpoint: `B` means that both anatomical eyes can be
independently labeled in at least one reviewed frame.

## Timing

`manifest_v2.csv` records each source frame's decoded video time. Average FPS is never treated
as a universal 120-fps clock. The capture app's sidecar will be authoritative for future torch
onset; QuickTime-era clips retain manually marked video-time onset values.

## Legacy utilities

`detect_rois.py`, `confirm_rois.py`, and `box_eyes.py` predate the per-frame reviewed dataset.
They can help explore a new clip or make an initial tracking seed, but their boxes must not
silently replace the hand-reviewed boxes in `eyebox_review`.
