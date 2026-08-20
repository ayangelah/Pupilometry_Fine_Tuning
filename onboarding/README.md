# Onboarding pipeline (new subjects -> label-ready frames)

Turns raw full-face PLR clips into OpenEDS-reframed eye crops ready to hand-label for the
detector fine-tune. Both eyes per clip (each an independent training sequence, same LOSO
subject). Run from the repo root.

## Flow

```
1. drop new clips in data/clips/  named  subject_<letter>_<n>.MOV   (both eyes, full face)
2. python onboarding/detect_rois.py       # auto both-eye ROIs + clean -> new_subject_rois.json
3. python onboarding/confirm_rois.py       # contact sheet -> debug_masks/onboarding/roi_confirm.png
4.   ... eyeball the sheet; hand-fix any bad box in new_subject_rois.json (see below), re-run step 3
5. python onboarding/extract_frames.py [N] # N frames/eye reframed to 400x640 -> labeling/frames/ + manifest.csv
6.   ... hand-label the crops in Labelme -> convert to gt.csv/masks
7.   ... add the new subject letters to SUBJECTS in finetune_ritnet.py, re-run LOSO
```

Steps 2 and 5 are idempotent: `detect_rois` skips clips already in `new_subject_rois.json`,
`extract_frames` skips sequences already in `manifest.csv`. Safe to re-run as subjects arrive.

**Manual boxing (`box_eyes.py`) — the reliable alternative to steps 2-4.** `detect_rois` (Haar)
assumes both eyes are visible and stumbles on single-eye / angled clips. `box_eyes.py` is an
interactive click-drag tool (VS Code `# %%` cells, needs `pip install ipympl`): for each clip you
drag a box around **each visible eye** — one for a single-eye/angled clip, two for full-face — and
it writes them straight to `new_subject_rois.json`. This handles the 1-eye AND 2-eye cases
explicitly, and mirrors the deployed alignment overlay (you place the box the nurse aligns to).
Use it instead of `detect_rois` when footage is mixed, or to fix any box the contact sheet flags.

## Fixing a bad ROI (step 4)

The contact sheet is the QC. Any cell not centred on an eye is wrong. Edit
`new_subject_rois.json` (keys `<stem>_L` / `<stem>_R`, boxes `[x, y, w, h]` in original pixels):

- **Mirror a good eye** onto its missing/bad partner: `R = [W - Lx - Lw, Ly, Lw, Lh]` (W = frame width).
- **Drop a clip** (e.g. low-res / undetectable): delete both its `_L` and `_R` entries.

Then re-run `confirm_rois.py`.

## Notes

- 4K random-seek makes detection slow (~30-45s/clip); expect a few minutes per batch.
- Haar is imperfect on dim / gaze-off / low-res footage, so the manual QC pass is expected.
- `detect_rois` treats a box much smaller than, or far off the height of, its partner as a
  false detection and mirror-fills; it can still miss, hence the contact sheet.
- Outputs: `new_subject_rois.json` (gitignored, tied to private clips), overlays +
  contact sheet in `debug_masks/onboarding/` (gitignored), crops in `labeling/frames/`
  (gitignored), rows in `labeling/manifest.csv` (tracked, coords only).
