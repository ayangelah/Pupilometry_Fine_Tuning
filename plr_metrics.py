# %% [markdown]
# # Stage 2: iris-grounded mm calibration + PLR metric formulas (the Derrick readout)
#
# Consumes the per-frame curve from `build_curve` in ritnet_segment.py
# (dict: t, iris_px, pupil_px, ratio, valid, onset_t) and turns it into a
# clinician-style readout: 1c mm calibration, 1d the six PLR metrics, 1e the readout.
#
# Unimplemented metrics raise NotImplementedError with the formula stated in words
# plus the NPi healthy-adult normative, so the contract is explicit before coding it.

# %% [imports]
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

# %% [config]
# 1c calibration constant. The iris is a per-frame ruler: its diameter is
# population-stable and neuro-invariant, so pupil_px/iris_px * IRIS_MM cancels
# distance, zoom, and angle. 11.7mm is a stated, known error source (real horizontal
# visible-iris diameter varies ~11.7-12.0mm across people).
IRIS_MM = 11.7

# Snap-baseline window (see CLAUDE.md). PLR latency is ~180-230ms: after light-on
# the pupil holds its dark-adapted resting size for that long before it starts to
# constrict. So the resting/initial diameter is read from the FIRST LIT, PRE-
# CONSTRICTION frames, not the noisy dark pre-flash. These two constants bound that
# window relative to onset_t; tune them if the exposure ramp or the latency
# estimate for the rig differs.
BASELINE_SKIP_MS = 30.0    # skip the first frames after onset (AE ramp / blink / saturation)
BASELINE_END_MS  = 150.0   # end the window before constriction can begin (< ~180ms latency)

# Clip-level validity gate thresholds (tune these; they are the knobs).
MIN_VALID_FRAC      = 0.40   # below this gate pass-rate the whole curve is too sparse to trust
MIN_BASELINE_FRAMES = 3      # baseline-dependent metrics (%constriction, latency) need at least
                             # this many valid frames in the snap-baseline window (else no denominator)


# %% [1c calibration] one calibration line
def calibrate_mm(curve: dict, iris_mm: float = IRIS_MM) -> np.ndarray:
    """Convert the per-frame pupil measurement to millimeters via the iris ruler.

    Input: the `curve` dict from build_curve (has 'ratio' = pupil_px / iris_px,
    already scale-invariant to the 400x640 resize and locked to the LIT iris).
    Output: pupil_mm, a float array aligned to curve['t'] (NaN where the frame was
    gated out, i.e. where ratio is NaN).
    """
    ratio = np.asarray(curve["ratio"], float)
    pupil_mm = ratio * iris_mm
    return pupil_mm


# %% [1d metrics]  results container + windowing helper + per-formula stubs

@dataclass
class PLRMetrics:
    """One clip's PLR readout, all diameters in mm, velocities in mm/s, times in s.
    NPi healthy-adult normatives shown for sanity-checking the outputs."""
    baseline_mm:      float   # resting/initial diameter (snap-baseline). NPi ~4.38
    min_mm:           float   # minimum diameter after onset.            NPi ~2.87
    pct_constriction: float   # (baseline - min) / baseline * 100.       NPi ~34.7
    latency_s:        float   # onset -> start of constriction.          NPi ~0.23
    cv_peak_mm_s:     float   # peak (max) constriction velocity (MCV).  NPi ~4.16
    cv_mean_mm_s:     float   # mean constriction velocity (CV).         NPi ~2.74
    dv_mm_s:          float   # dilation (recovery) velocity.            NPi ~1.13


def baseline_window_mask(t: np.ndarray, onset_t: float) -> np.ndarray:
    """Boolean mask selecting the snap-baseline frames,
    [onset_t + BASELINE_SKIP_MS, onset_t + BASELINE_END_MS]."""
    t = np.asarray(t, float)
    lo = onset_t + BASELINE_SKIP_MS / 1000.0
    hi = onset_t + BASELINE_END_MS / 1000.0
    return (t >= lo) & (t <= hi)


def assess_validity(curve: dict) -> dict:
    """Clip-level validity check. Mechanical counts against the thresholds above; returns
    the counts plus two verdicts the readout/compute layer acts on:
      clip_ok      - enough of the curve passed the geometry gate to trust anything at all
      baseline_ok  - the snap-baseline window holds enough valid frames for a denominator
    A clip can be clip_ok but NOT baseline_ok (dead baseline, e.g. subject_b_1): size and
    velocity may still hold while %constriction and latency do not. Policy (what to do on a
    failed verdict) is decided by the caller."""
    t = np.asarray(curve["t"], float)
    valid = np.asarray(curve.get("valid", np.isfinite(curve.get("ratio", t))), bool)
    onset_t = curve.get("onset_t")
    valid_frac = float(valid.mean()) if valid.size else 0.0
    n_baseline = 0 if onset_t is None else int((baseline_window_mask(t, float(onset_t)) & valid).sum())
    return dict(valid_frac=valid_frac, n_baseline=n_baseline,
                clip_ok=valid_frac >= MIN_VALID_FRAC,
                baseline_ok=n_baseline >= MIN_BASELINE_FRAMES)


def baseline_diameter(t: np.ndarray, pupil_mm: np.ndarray, onset_t: float) -> float:
    """Resting diameter = a robust average (mean or median) of pupil_mm over
    baseline_window_mask, ignoring NaNs. A few clean frames suffice (windowed median).
    This is the denominator for %constriction and the reference for latency."""
    mask = baseline_window_mask(t, onset_t)   # boolean array over frames
    window = pupil_mm[mask]
    return np.nanmedian(window)



def minimum_diameter(t: np.ndarray, pupil_mm: np.ndarray, onset_t: float) -> float:
    """Minimum pupil_mm over the post-onset region (t >= onset_t), NaN-safe."""
    post_mask = t >= onset_t
    pupil_after_onset = pupil_mm[post_mask]
    return np.nanmin(pupil_after_onset)


def percent_constriction(baseline_mm: float, min_mm: float) -> float:
    """(baseline - min) / baseline * 100."""
    return (baseline_mm - min_mm) / baseline_mm * 100.


def latency(t: np.ndarray, pupil_mm: np.ndarray, onset_t: float, baseline_mm: float) -> float:
    """Onset -> start of constriction, in seconds. 'start' e.g. the first post-onset
    time the diameter drops below baseline by a small threshold, or where the smoothed
    velocity first crosses a negative threshold. Returns t_start - onset_t."""
    constriction_percent = 0.05
    post_mask = t >= onset_t
    t_post = t[post_mask]
    pupil_after_onset = pupil_mm[post_mask]
    for i in range(pupil_after_onset.size):
        diameter = pupil_after_onset[i]
        if (baseline_mm - diameter) / baseline_mm >= constriction_percent:
            return t_post[i] - onset_t
    return np.nan


def smooth_diameter(y: np.ndarray, window: int = 5) -> np.ndarray:
    """Median-filter a 1-D diameter signal so single-frame outliers don't blow up its
    derivative. NaN-safe: linearly interpolates short gated gaps onto the valid samples
    before filtering, so the phase stays contiguous. `window` (forced odd) trades noise
    rejection against flattening the true peak, so start small. Median, not mean/savgol,
    because it rejects a lone spike outright instead of being dragged toward it."""
    y = np.asarray(y, float)
    good = np.isfinite(y)
    if good.sum() < 3:
        return y.copy()
    k = window if window % 2 else window + 1          # odd kernel
    idx = np.arange(y.size)
    filled = np.interp(idx, idx[good], y[good])        # bridge gaps so the filter is continuous
    return median_filter(filled, size=k, mode="nearest")   # 'nearest' holds edges (no zero-pull slope)


def constriction_velocity(t: np.ndarray, pupil_mm: np.ndarray, onset_t: float) -> tuple[float, float]:
    """Constriction velocities in mm/s over the constriction phase (onset -> min).
    Returns (peak, mean) of the negative slope magnitude. dt from np.diff(t) (stride-aware).
    Light smoothing before differentiating helps; slopes amplify per-frame noise."""
    constriction_mask = t >= onset_t
    t_phase     = t[constriction_mask]        # build the phase from the mask
    pupil_phase = smooth_diameter(pupil_mm[constriction_mask], window=5)
    floor = np.nanmin(pupil_phase)
    tol = 0.05 * (pupil_phase[0] - floor)
    iend  = np.argmax(pupil_phase <= floor + tol)
    t_phase     = t_phase[:iend + 1]
    pupil_phase = pupil_phase[:iend + 1]
    dt = np.median(np.diff(t_phase))
    vel = -savgol_filter(pupil_phase, window_length=11, polyorder=2, deriv=1, delta=dt)
    peak = np.nanmax(vel)
    mean = np.nanmean(vel)
    return (peak, mean)



def dilation_velocity(t: np.ndarray, pupil_mm: np.ndarray, onset_t: float) -> float:
    """Recovery velocity in mm/s after the minimum (re-dilation slope).
    The only metric that lives in the dim post-flash regime, so it is the noisiest;
    the snap-baseline does not help it (see CLAUDE.md dilation caveat)."""
    dilation_mask = t >= onset_t
    t_phase     = t[dilation_mask]        # build the phase from the mask
    pupil_phase = pupil_mm[dilation_mask]
    imin = np.nanargmin(pupil_phase)          # then cut it at the minimum
    t_phase     = t_phase[imin:]
    pupil_phase = pupil_phase[imin:]
    vel = np.diff(pupil_phase) / np.diff(t_phase)   # one copy only
    mean = np.nanmean(vel)
    return (mean)


def compute_metrics(curve: dict, pupil_mm: np.ndarray | None = None) -> PLRMetrics:
    """Calibrate if needed, then run the six metric formulas and pack the dataclass.
    onset_t is a known input carried on the curve (light_onset.json)."""
    if curve.get("onset_t") is None:
        raise ValueError("curve has no onset_t; mark this clip in light_onset.json first.")
    t = np.asarray(curve["t"], float)
    onset_t = float(curve["onset_t"])
    if pupil_mm is None:
        pupil_mm = calibrate_mm(curve)
    pupil_mm = np.asarray(pupil_mm, float)

    base = baseline_diameter(t, pupil_mm, onset_t)
    mn   = minimum_diameter(t, pupil_mm, onset_t)
    cv_peak, cv_mean = constriction_velocity(t, pupil_mm, onset_t)
    return PLRMetrics(
        baseline_mm=base,
        min_mm=mn,
        pct_constriction=percent_constriction(base, mn),
        latency_s=latency(t, pupil_mm, onset_t, base),
        cv_peak_mm_s=cv_peak,
        cv_mean_mm_s=cv_mean,
        dv_mm_s=dilation_velocity(t, pupil_mm, onset_t),
    )


# %% [run]  wire a real clip through Stage 2
# from pathlib import Path
# import cv2
# from ritnet_segment import build_curve, _load_rois, get_roi, framed_roi, CLIPS_DIR
#
# CLIP = "IMG_4599 2"
# _cl = next(p for p in Path(CLIPS_DIR).glob("*") if p.stem == CLIP)
# _c0 = cv2.VideoCapture(str(_cl)); _c0.set(cv2.CAP_PROP_POS_FRAMES, 0); _ok, _f0 = _c0.read(); _c0.release()
# _seed = get_roi(CLIP, _f0, _load_rois())
# curve = build_curve(_cl, framed_roi(_seed, _f0.shape), seed_box=_seed)
#
# pupil_mm = calibrate_mm(curve)
# m = compute_metrics(curve, pupil_mm)
# print(m)   # readout table + annotated curve = 1e (readout.py), once these numbers look right
