"""Stage 2 demo: one clip -> mm curve -> PLR readout.

Usage:
  python stage2_demo.py <clip_stem> [--eye left|right] [--roi-file FILE]

Wires the existing pieces together; it owns no measurement or metric math:
  build_curve (ritnet_segment)  ->  calibrate_mm / compute_metrics (plr_metrics)

Eye choice matters. A clip labelled "both eyes" has two boxes, and one of them is often
the eye at the frame edge with the iris cut off. That box measures nothing (the limbus
arc is missing, so the circle fit is unconstrained), so it is skipped by default and the
choice is always printed.
"""
import os, json, argparse
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

from ritnet_segment import build_curve, framed_roi, CLIPS_DIR
from plr_metrics import (assess_validity, calibrate_mm, compute_metrics,
                         baseline_window_mask)

EDGE_FRAC = 0.01          # a box within this fraction of a frame edge is treated as clipped


def eyebox_candidates(clip_stem, rois, frame_shape):
    """[(eye, box, clipped)] for every ROI key belonging to clip_stem.

    `eye` is 'left'/'right' (or '' for an unsuffixed key). `clipped` is True when the box
    touches a frame edge, which means the eye is cut off and the iris arc is incomplete.
    The threshold is a FRACTION of the frame so it behaves the same on 4K and 720p clips.
    """
    H, W = frame_shape[:2]
    mx, my = EDGE_FRAC * W, EDGE_FRAC * H
    out = []
    for key, box in rois.items():
        if key == clip_stem:
            eye = ""
        elif key == f"{clip_stem}_left_eye":
            eye = "left"
        elif key == f"{clip_stem}_right_eye":
            eye = "right"
        else:
            continue                                     # a different clip (or a longer stem)
        x, y, w, h = box
        clipped = (x <= mx) or (y <= my) or (x + w >= W - mx) or (y + h >= H - my)
        out.append((eye, list(box), clipped))
    return sorted(out, key=lambda c: c[0])


def pick_eyebox(clip_stem, rois, frame_shape, want_eye=None):
    """Choose one eyebox and say why. Returns (eye, box) or (None, None)."""
    cands = eyebox_candidates(clip_stem, rois, frame_shape)
    if not cands:
        print(f"ERROR: no eyebox for {clip_stem}")
        return None, None

    print("eyeboxes available:")
    for eye, box, clipped in cands:
        print(f"  {eye or '(unnamed)':<10} {box}" + ("   <- touches frame edge, iris cut off" if clipped else ""))

    if want_eye is not None:
        for eye, box, clipped in cands:
            if eye == want_eye:
                if clipped:
                    print(f"WARNING: --eye {want_eye} is edge-clipped; measurements will be unreliable")
                print(f"using: {eye} (requested)")
                return eye, box
        print(f"ERROR: no '{want_eye}' eyebox for {clip_stem}")
        return None, None

    usable = [c for c in cands if not c[2]]
    if not usable:
        eye, box, _ = max(cands, key=lambda c: c[1][2])
        print(f"WARNING: every eyebox is edge-clipped; falling back to widest ({eye})")
        return eye, box
    eye, box, _ = max(usable, key=lambda c: c[1][2])      # widest usable box
    print(f"using: {eye} (widest box clear of the frame edge)")
    return eye, box


def run_demo(clip_stem, want_eye=None, roi_file="new_subject_rois.json"):
    try:
        clip_path = next(Path(CLIPS_DIR).glob(f"{clip_stem}.*"))
    except StopIteration:
        print(f"ERROR: clip not found: {clip_stem}")
        return None

    cap = cv2.VideoCapture(str(clip_path))
    ok, frame0 = cap.read()
    cap.release()
    if not ok:
        print("ERROR: could not read the first frame")
        return None

    rois = json.loads(Path(roi_file).read_text())
    eye, seed_box = pick_eyebox(clip_stem, rois, frame0.shape, want_eye)
    if seed_box is None:
        return None

    print(f"\nStage 1b: RITnet + measurement layer ({clip_path.name})")
    curve = build_curve(clip_path, framed_roi(seed_box, frame0.shape), seed_box=seed_box)
    if curve.get("onset_t") is None:
        print(f"ERROR: {clip_stem} has no onset in light_onset.json")
        return None

    v = assess_validity(curve)
    t = np.asarray(curve["t"], float)
    onset_t = float(curve["onset_t"])
    post = t >= onset_t
    n_post_valid = int(np.isfinite(np.asarray(curve["pupil_px"], float))[post].sum())

    print(f"\ngate pass rate    {v['valid_frac']:.1%} of measured frames")
    print(f"baseline frames   {v['n_baseline']}")
    print(f"post-onset valid  {n_post_valid} / {int(post.sum())}")
    for flag, msg in (("clip_ok", "curve too sparse to trust"),
                      ("baseline_ok", "no baseline denominator: %constriction and latency unreliable")):
        if not v[flag]:
            print(f"WARNING: {flag}=False - {msg}")
    if v["constriction_ok"] is False:
        print(f"WARNING: constriction_ok=False - depth {v['constriction_depth_pct']:.1f}%, "
              f"post-onset dilation {v['post_onset_dilation']:+.1%} (check the onset mark)")

    if n_post_valid < 11:
        print(f"\nSTOP: only {n_post_valid} valid post-onset frames. The velocity fit needs >=11.")
        print("      Metrics from this curve would be fiction. Fix the ROI or the footage first.")
        return None

    pupil_mm = calibrate_mm(curve)
    m = compute_metrics(curve, pupil_mm)

    print("\n" + "=" * 62)
    print(f"PLR READOUT  {clip_stem} ({eye} eye)")
    print("=" * 62)
    print(f"  baseline diameter   {m.baseline_mm:7.2f} mm     NPi ~4.38")
    print(f"  minimum diameter    {m.min_mm:7.2f} mm     NPi ~2.87")
    print(f"  constriction        {m.pct_constriction:7.1f} %      NPi ~34.7")
    print(f"  latency             {m.latency_s:7.3f} s      NPi ~0.23")
    print(f"  constriction vel    {m.cv_mean_mm_s:7.2f} mm/s   NPi ~2.74")
    print(f"    peak (MCV)        {m.cv_peak_mm_s:7.2f} mm/s   NPi ~4.16")
    print(f"  dilation vel        {m.dv_mm_s:7.2f} mm/s   NPi ~1.13")
    print("=" * 62)

    _plot(clip_stem, eye, t, pupil_mm, onset_t)
    return m


def _plot(clip_stem, eye, t, pupil_mm, onset_t):
    """Curve plot. Valid samples only as markers, so gated gaps read as gaps and adjacent
    survivors are never joined into a spike that looks like signal."""
    Path("debug_masks").mkdir(exist_ok=True)
    good = np.isfinite(pupil_mm)
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(t[good], pupil_mm[good], ".", ms=4, color="#2b6cb0", label=f"pupil mm ({good.sum()} valid)")
    ax.axvline(onset_t, color="#dd6b20", ls="--", lw=1.5, label="light onset")

    bw = baseline_window_mask(t, onset_t)
    if bw.any():
        ax.axvspan(t[bw][0], t[bw][-1], color="#3182ce", alpha=0.15, label="baseline window")

    post = good & (t >= onset_t)
    if post.any():
        i = np.argmin(pupil_mm[post])
        ax.plot(t[post][i], pupil_mm[post][i], "*", ms=14, color="#c53030", label="minimum")

    ax.set_xlabel("time (s)")
    ax.set_ylabel("pupil diameter (mm)")
    ax.set_title(f"{clip_stem} - {eye} eye")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    out = f"debug_masks/stage2_{clip_stem}_{eye}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nplot -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("clip", help="clip stem, e.g. subject_a_1")
    p.add_argument("--eye", choices=["left", "right"], default=None,
                   help="which eyebox to measure (default: widest box clear of the frame edge)")
    p.add_argument("--roi-file", default="new_subject_rois.json")
    a = p.parse_args()
    run_demo(a.clip, want_eye=a.eye, roi_file=a.roi_file)
