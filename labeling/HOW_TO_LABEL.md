# Ground-truth labeling guide (brown-eye pupillometry)

Purpose: hand-trace pupil + iris (+ sclera) on sampled frames to get a **true** correctness
score for the detector, held out by subject (LOSO). These masks are also the fine-tune targets.

## Launch

```bash
conda activate labelme-env
labelme labeling/frames_v2/<sequence> --labels iris,pupil,sclera,reject --nodata --autosave
```

- `--nodata` keeps each JSON small (does not embed the image).
- `--autosave` writes a `<image>.json` next to each PNG as you go.
- Open the folder, arrow-key through images.

## What to draw (one eye per frame)

| Region | Labelme tool | How | Label (exact) |
|---|---|---|---|
| **Iris** | **Circle** | Click the iris center, drag to the limbus (iris/sclera edge). Draw the **full disk even under the eyelids** - irises are circles at our granularity, and we grade the model on the full circle. | `iris` |
| **Pupil** | **Circle** | Click the pupil center and drag to its visible edge. This is measurement ground truth for pupil diameter, matching the circular output used by the app. | `pupil` |
| **Sclera** | Polygon | (Optional) Outline the visible white. Skip if short on time. | `sclera` |
| Background | (nothing) | Leave everything else untraced. | - |

## Rules that keep the metric honest

1. **Labels must be exactly** `pupil`, `iris`, `sclera` (lowercase). The converter keys on these.
2. **Draw from scratch** - trace what the pixels show, do not trace-over any model overlay. (This is the *test* set; anchoring to the model's guess would flatter its score.)
3. **One iris per usable frame and at most one pupil.** Omit the pupil only under rule 4.
4. **Pupil boundary invisible but iris traceable:** draw the iris circle and omit the pupil. This becomes `iris_only` / `not_visible`; it teaches measurement confidence, not a fake “no pupil” segmentation target.
5. **Whole frame unusable** (blink, heavy blur, occlusion, no trustworthy iris): select Labelme's rectangle tool, draw a small rectangle anywhere, name it **`reject`**, and draw nothing else. The rectangle location and size have no meaning; its presence is an image-level status marker.
6. **No JSON means pending**, not rejected. Save an explicit `reject` annotation when review is complete but the frame is unusable.

## Notes

- You are labeling the **reframed eye crop the model sees**, so your masks line up with predictions with no coordinate guesswork.
- Subject grouping and reviewed per-frame eyeboxes are in `labeling/manifest_v2.csv`. Do not move frames between sequences.
- Once a few are done, run the converter (Labelme JSON -> class masks + derived diameters) on them and check the output before labeling the rest.

Run `python labeling/labelme_to_gt.py --strict` for the final build. Strict mode fails if any
crop still has no JSON or any label violates the schema.
