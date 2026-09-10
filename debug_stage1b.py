"""Debug Stage 1b: why is each frame being rejected?

Usage:
  python debug_stage1b.py <clip_stem> [--eye left|right] [--n-images 12]

Repeats build_curve's pass exactly (same CURVE_STRIDE, same per-frame eye tracking, same
iris lock from LIT frames) and then, instead of collapsing to a curve, reports the REASON
each frame failed the geometry gate and writes overlay images so the fits can be eyeballed.

The gate compares every frame against the ONE locked iris radius (the median LIT radius),
not that frame's own radius - so a bad lock rejects the whole clip. That is the failure
mode this script exists to make visible.
"""
import os, argparse, json
from collections import Counter
from pathlib import Path

import numpy as np
import cv2

from ritnet_segment import (segment, measure, overlay, ritnet_input, crop, framed_roi,
                            lock_iris_radius, light_onset_frame, reject_temporal_outliers,
                            passes_gate, EyeTracker, TRACK_EYE, CURVE_STRIDE, FPS,
                            IN_W, IN_H, CLIPS_DIR,
                            GATE_RATIO_MIN, GATE_RATIO_MAX, GATE_CONCENTRIC)
from stage2_demo import pick_eyebox

DEBUG_DIR = Path("debug_masks/stage1b")


def failure_reason(iris_circle, locked_r, pupil):
    """Mirror passes_gate's branches, but say WHICH one fired."""
    if iris_circle is None:
        return "no iris fit"
    if pupil is None:
        return "no pupil blob"
    if not np.isfinite(locked_r) or locked_r <= 0:
        return "bad iris lock"
    ratio = pupil[2] / locked_r
    if ratio < GATE_RATIO_MIN:
        return f"pupil too small (ratio {ratio:.2f} < {GATE_RATIO_MIN})"
    if ratio > GATE_RATIO_MAX:
        return f"pupil too large (ratio {ratio:.2f} > {GATE_RATIO_MAX})"
    off = np.hypot(pupil[0] - iris_circle[0], pupil[1] - iris_circle[1]) / locked_r
    if off > GATE_CONCENTRIC:
        return f"off-center ({off:.2f}r > {GATE_CONCENTRIC}r)"
    return "pass"


def debug_clip(clip_stem, want_eye=None, n_images=12, roi_file="new_subject_rois.json"):
    try:
        clip_path = next(Path(CLIPS_DIR).glob(f"{clip_stem}.*"))
    except StopIteration:
        print(f"ERROR: clip not found: {clip_stem}")
        return

    cap = cv2.VideoCapture(str(clip_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame0 = cap.read()
    if not ok:
        print("ERROR: could not read the first frame")
        cap.release()
        return

    rois = json.loads(Path(roi_file).read_text())
    eye, seed_box = pick_eyebox(clip_stem, rois, frame0.shape, want_eye)
    if seed_box is None:
        cap.release()
        return

    n_measured = len(range(0, n, CURVE_STRIDE))
    render_at = set(np.linspace(0, n_measured - 1, min(n_images, n_measured)).astype(int))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    tracker, tracking = None, (TRACK_EYE and seed_box is not None)
    kept, iris_c, iris_r, pupils, renders = [], [], [], [], []
    fi = -1
    print(f"\npass over {clip_path.name}: {n} frames, stride {CURVE_STRIDE}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        if tracking:                                        # track EVERY frame (continuity)
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if tracker is None:
                tracker = EyeTracker(g, seed_box); tight = tuple(seed_box)
            else:
                tight = tracker.update(g)
            frame_roi = framed_roi(tight, frame.shape)
        else:
            frame_roi = framed_roi(seed_box, frame.shape)
        if fi % CURVE_STRIDE:
            continue
        fr = crop(frame, frame_roi)
        crop_bgr = cv2.resize(fr, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
        m = measure(segment(ritnet_input(fr)), crop_bgr)
        ic = m["iris_circle"]
        iris_c.append((ic[0], ic[1]) if ic else None)
        iris_r.append(ic[2] if ic else np.nan)
        pupils.append(m["pupil_circle"])
        if (len(kept)) in render_at:
            renders.append((fi, overlay(crop_bgr, m), m))
        kept.append(fi)
    cap.release()
    print(f"  {len(kept)} frames measured")

    # --- iris lock, exactly as build_curve does it -------------------------------------
    onset = light_onset_frame(clip_stem)
    lit = np.array(kept) >= onset if onset is not None else np.ones(len(kept), bool)
    lit_radii = [r for r, L in zip(iris_r, lit) if L]
    locked_r = lock_iris_radius(lit_radii if any(lit) else iris_r)
    all_r = np.asarray(iris_r, float)

    print(f"\niris lock")
    print(f"  onset             {'unmarked' if onset is None else f'frame {onset} ({onset / FPS:.2f}s)'}")
    print(f"  lit frames        {int(lit.sum())} / {len(kept)}")
    print(f"  locked radius     {locked_r:.1f} px  (diameter {2 * locked_r:.1f} px)")
    if np.isfinite(all_r).any():
        print(f"  per-frame radius  median {np.nanmedian(all_r):.1f}  "
              f"p10 {np.nanpercentile(all_r[np.isfinite(all_r)], 10):.1f}  "
              f"p90 {np.nanpercentile(all_r[np.isfinite(all_r)], 90):.1f}  "
              f"({int(np.isfinite(all_r).sum())} fits)")

    # --- gate, with reasons ------------------------------------------------------------
    reasons = [failure_reason(c, locked_r, p) for c, p in zip(iris_c, pupils)]
    bucket = Counter(r.split(" (")[0] for r in reasons)
    n_pass = bucket.get("pass", 0)

    print(f"\ngeometry gate: {n_pass} / {len(kept)} pass ({100 * n_pass / max(len(kept), 1):.1f}%)")
    for reason, count in bucket.most_common():
        if reason == "pass":
            continue
        print(f"  {count:5d}  {reason}")

    # temporal spike rejection runs after the gate in build_curve; report its extra cost
    pupil_px = np.array([2.0 * p[2] if p else np.nan for p in pupils], float)
    pupil_px[[r != "pass" for r in reasons]] = np.nan
    after_temporal = reject_temporal_outliers(pupil_px)
    extra = int(np.isfinite(pupil_px).sum() - np.isfinite(after_temporal).sum())
    print(f"  {extra:5d}  additionally killed by temporal spike rejection")
    print(f"  -> {int(np.isfinite(after_temporal).sum())} frames survive into the curve")

    post = np.array(kept) >= (onset if onset is not None else 0)
    print(f"  -> {int(np.isfinite(after_temporal)[post].sum())} of those are post-onset "
          f"(velocity fit needs >=11)")

    # --- overlay images ---------------------------------------------------------------
    out_dir = DEBUG_DIR / f"{clip_stem}_{eye}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.glob("*.png"):
        path.unlink()
    idx_of = {f: i for i, f in enumerate(kept)}
    for fi_r, vis, m in renders:
        i = idx_of[fi_r]
        why = reasons[i]
        tag = "PASS" if why == "pass" else "FAIL"
        band = np.full((44, IN_W, 3), 20, np.uint8)
        cv2.putText(band, f"f{fi_r} {fi_r / FPS:.2f}s [{tag}]", (6, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (120, 255, 120) if tag == "PASS" else (120, 120, 255), 1, cv2.LINE_AA)
        cv2.putText(band, why[:58], (6, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (220, 220, 220), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"{i:04d}_f{fi_r:06d}_{tag}.png"), np.vstack([band, vis]))

    print(f"\n{len(renders)} overlays -> {out_dir}/")
    print("  blue = fitted iris, green = fitted pupil, yellow = limbus points kept, red = trimmed")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("clip")
    p.add_argument("--eye", choices=["left", "right"], default=None)
    p.add_argument("--n-images", type=int, default=12)
    p.add_argument("--roi-file", default="new_subject_rois.json")
    a = p.parse_args()
    debug_clip(a.clip, want_eye=a.eye, n_images=a.n_images, roi_file=a.roi_file)
