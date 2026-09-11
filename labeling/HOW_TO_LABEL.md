# Ground-truth labeling guide (brown-eye pupillometry)

Purpose: hand-trace pupil + iris (+ sclera) on sampled frames to get a **true** correctness
score for the detector, held out by subject (LOSO). These masks are also the fine-tune targets.

## Launch

```bash
conda activate labelme-env
labelme labeling/frames --labels pupil,iris,sclera,reject --nodata --autosave
```

- `--nodata` keeps each JSON small (does not embed the image).
- `--autosave` writes a `<image>.json` next to each PNG as you go.
- Open the folder, arrow-key through images.

## What to draw (one eye per frame)

| Region | Labelme tool | How | Label (exact) |
|---|---|---|---|
| **Iris** | **Circle** | Click the iris center, drag to the limbus (iris/sclera edge). Draw the **full disk even under the eyelids** - irises are circles at our granularity, and we grade the model on the full circle. | `iris` |
| **Pupil** | **Polygon** | Trace the actual pupil boundary point by point. Do **not** assume it is round; follow what you see. | `pupil` |
| **Sclera** | Polygon | (Optional) Outline the visible white. Skip if short on time. | `sclera` |
| Background | (nothing) | Leave everything else untraced. | - |

## Rules that keep the metric honest

1. **Labels must be exactly** `pupil`, `iris`, `sclera` (lowercase). The converter keys on these.
2. **Draw from scratch** - trace what the pixels show, do not trace-over any model overlay. (This is the *test* set; anchoring to the model's guess would flatter its score.)
3. **One iris and one pupil per frame.**
4. **Unlabelable frame** (blink, heavy motion blur, pupil truly invisible): draw a small **rectangle labeled `reject`** and nothing else. This marks it as a *true* reject, distinct from "not labeled yet" - we need that distinction.
5. Save happens automatically; a frame with no JSON = not done yet.

## Notes

- You are labeling the **reframed eye crop the model sees**, so your masks line up with predictions with no coordinate guesswork.
- Subject grouping for the LOSO split is in `labeling/manifest.csv` (person A/B/C per clip). Do not move frames between clips.
- Once a few are done, run the converter (Labelme JSON -> class masks + derived diameters) on them and check the output before labeling the rest.
