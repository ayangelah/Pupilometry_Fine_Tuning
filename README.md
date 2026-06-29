# Pupillometry from a phone camera

Offline, on-device pupil measurement from short eye-video clips. The goal is to recover the
kind of readout a handheld pupillometer gives (pupil diameter over time and the pupillary
light reflex metrics derived from it) using only a phone camera and no dedicated hardware.

This is a work in progress and a **triage / screening aid, not an FDA-cleared diagnostic.**
No equivalence is claimed to cleared devices.

## Why it can report millimeters without hardware

The iris diameter is population-stable (about 11.7 to 12.0 mm) and does not change with
neurological state. Measuring the iris in pixels alongside the pupil in the same frame turns
the iris into a built-in ruler that cancels camera distance, zoom, and angle:

```
pupil_mm = (pupil_px / iris_px) * IRIS_MM
```

The 11.7 mm assumption is a known, stated source of error.

## Pipeline

```
Stage 1a  RITnet 4-class eye segmentation per frame (on an eye crop)
Stage 1b  measurement layer:
            - iris circle fit from the limbus (the iris/sclera boundary)
            - pupil localized inside the iris (local threshold in color, glints removed)
            - geometry / quality gate -> reject bad frames
          -> pupil_px and iris_px per frame
Stage 2   iris-grounded mm calibration + PLR metrics (baseline, min, % constriction,
          latency, constriction/dilation velocity)
Stage 3   (later) learned scorer: normal vs abnormal reflex
```

The measurement is formula-driven first. Segmentation provides structure; the diameters and
metrics come from explicit geometry, not a black box.

## Detector

Eye segmentation uses [RITnet](https://github.com/AayushKrChaudhary/RITnet), a small
pretrained network (background / sclera / iris / pupil). It expects a near-eye, OpenEDS-style
framing, so each clip is cropped to the eye and reframed before inference. RITnet handles the
iris well; the pupil on visible-light (non-infrared) brown eyes is refined by a classical
step inside the iris.

## Setup

```bash
# 1. environment (conda example)
pip install -r requirements.txt

# 2. clone the detector into ./RITnet (not vendored here)
git clone https://github.com/AayushKrChaudhary/RITnet.git
```

## Run

The scripts use VS Code `# %%` interactive cells.

- `extract_pupil_diameter.py` - classical OpenCV baseline (diameter-vs-time curve, auto light-onset detection).
- `ritnet_segment.py` - RITnet integration plus the measurement layer.

Input clips are eye-video files placed in `data/clips/`. Sample data is not included
(it contains identifiable faces).

## Status

Active development. Current focus: the Stage 1b measurement layer (robust per-frame iris and
pupil diameters) ahead of mm calibration and the PLR metric readout.
