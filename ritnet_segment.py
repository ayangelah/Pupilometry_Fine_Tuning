# %% [markdown]
# Phase 1b: run RITnet on the real close-up clips and VALIDATE the pupil+iris masks.
# RITnet = small pretrained 4-class eye-segmentation U-Net (bg / sclera / iris / pupil).
#
# SPINE:
#   - the detector CHOICE (RITnet) and, later, its fine-tune train/eval
#   - the diameter DEFINITION: diameter = mean of the two fitted-ellipse axes
#     (kept identical to the classical baseline so the two detectors are comparable)
#   - mm calibration + PLR metrics live downstream, NOT in this file
# DELEGABLE (this file): RITnet load/inference boilerplate, the exact RITnet
#   preprocessing, mask->contour->ellipse plumbing, overlay/debug dump.
#
# This file ONLY produces masks, pixel diameters, and overlays to look at.
# It does not calibrate to mm and does not compute any PLR metric.

# %% [imports]
import sys
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# RITnet lives in a cloned subrepo; put it on the path so we can import its model.
RITNET_DIR = Path("RITnet")
sys.path.insert(0, str(RITNET_DIR))
from densenet import DenseNet2D  # noqa: E402

# %% [config]
CLIPS_DIR   = "data/clips"
WEIGHTS     = RITNET_DIR / "best_model.pkl"
IN_W, IN_H  = 400, 640        # RITnet/OpenEDS native (W x H), portrait. Divisible by 16.
N_SAMPLE    = 9               # frames sampled evenly across each clip, for visual validation
# Apple-Silicon GPU (mps) when available, else CPU. On this laptop mps is ~4.5x faster on
# the RITnet forward pass (472ms -> 103ms/frame), which is the bottleneck for full-clip runs.
# Override to "cpu" if an mps op ever misbehaves. Weights load on CPU then move to DEVICE.
DEVICE      = "mps" if torch.backends.mps.is_available() else "cpu"
MIN_PTS     = 5               # fitEllipse needs >= 5 contour points
LIMBUS_BUFFER = 6             # px; a limbus point farther than this from the fitted circle is trimmed

# Part 2 pupil detection: specular-glint removal (bright AND colorless == a reflection)
GLINT_V      = 220            # value/brightness (0-255) above which a pixel may be glint
GLINT_S      = 60             # saturation (0-255) below which it is colorless enough to be glint
GLINT_DILATE = 3             # grow the glint mask this many px before inpainting (halo bleeds out)

# --- RGB robustness (optimizations_brainstorm.md, Group B) -------------------
# B1: RITnet is IR-native. The RED channel penetrates melanin better than a
# luminance average, so it is a closer visible-light proxy for the pupil/iris
# contrast RITnet learned on IR. Expected to help most on dark irises. This flag
# is the entire red-vs-gray ablation: flip it, re-run, compare the overlays.
USE_RED_CHANNEL = True
# B3: inpaint saturated glint on the MODEL INPUT before gamma/CLAHE, so CLAHE
# cannot amplify the blowout into the predicted mask. Distinct from the pupil-
# stage remove_glint(). Default OFF so the red-vs-gray effect is isolated first;
# turn on for a second, independent comparison (the doc says ablate each change).
GLINT_PRESEG = False

# capture / curve
FPS             = 120.0      # clip frame rate (iPhone). Used for the time axis of the curve.

# Part 3: iris-lock + geometry/temporal quality gate (Angela tunes these thresholds)
CURVE_STRIDE    = 3          # build the curve from every Nth frame (120fps -> 40fps at 3); speed vs resolution
GATE_RATIO_MIN  = 0.15       # pupil/iris ratio below this is implausible -> reject that frame
GATE_RATIO_MAX  = 0.75       # ...or above this
GATE_CONCENTRIC = 0.25       # pupil center must sit within this fraction of the iris radius of the iris center
TEMPORAL_WIN    = 7          # rolling-median window (frames) for single-frame spike rejection
TEMPORAL_K      = 4.0        # reject a point that is > K local-MADs from its rolling median
# Iris-lock uses LIT frames only: dark pre-flash iris fits are inflated and poison the clip-median
# ruler (measured: clip 4569 locks +85.6% -> -5.0% once dark frames are excluded). The light-on
# moment is a KNOWN, not learned: marked per clip now, reported by the phone's torch in the app.
LIGHT_ONSET_FILE = Path("light_onset.json")   # {clip_stem: onset_time_seconds}; see light_onset_frame()

# RITnet needs the EYE TO FILL THE FRAME (it was trained on near-eye OpenEDS images).
# Our clips show the whole face, so we crop to an eye ROI first. The camera is static
# within a clip, so one box per clip is reused for all its frames and cached to disk.
USE_ROI    = True
ROI_GUI    = False                  # cv2.selectROI hangs the Jupyter kernel on macOS; keep False
ROI_FILE   = Path("eye_rois.json")  # {clip_stem: [x, y, w, h]} = TIGHT eye box, original pixels

# RITnet/OpenEDS frames the eye a specific way: eye spans the FULL width but only the
# central ~third vertically, with brow above and cheek below (the rest is background).
# So we don't feed the tight eye box directly; we EXPAND it to that framing. You keep
# drawing the natural tight box; this reproduces the training distribution.
EXPAND_ROI = True
WIDTH_FILL = 1.05                   # eye spans ~full frame width (OpenEDS sclera touches edges)

# RITnet class ids (from the repo)
CLS_PUPIL  = 3
CLS_IRIS   = 2                # iris RING only; full iris disk = iris ring + pupil
CLS_SCLERA = 1               # the white of the eye; its border with iris is the true limbus

# %% [load model]  <-- delegable boilerplate
def load_ritnet(weights, device):
    model = DenseNet2D(dropout=True, prob=0.2)   # must match models.py instantiation
    # weights_only=True refuses to execute arbitrary pickle code; a pure state_dict still loads.
    state = torch.load(str(weights), map_location="cpu", weights_only=True)
    # weights may have been saved under DataParallel ("module." prefix); strip if so.
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model

model = load_ritnet(WEIGHTS, DEVICE)
print(f"RITnet loaded on {DEVICE}, params={sum(p.numel() for p in model.parameters()):,}")

# %% [preprocess]  <-- delegable: RITnet's EXACT test-time pipeline
# gamma 0.8 LUT -> CLAHE(1.5, 8x8) -> normalize to [-1, 1]. Input must be grayscale.
_GAMMA_LUT = (255.0 * (np.linspace(0, 1, 256) ** 0.8)).astype(np.uint8)
_CLAHE = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

def preprocess(gray):
    """One grayscale frame (any size) -> (1,1,IN_H,IN_W) float tensor, RITnet-style."""
    g = cv2.resize(gray, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
    g = cv2.LUT(g, _GAMMA_LUT)
    g = _CLAHE.apply(g)
    t = torch.from_numpy(g).float().div(255.0)        # [0,1]
    t = (t - 0.5) / 0.5                               # [-1,1], matches Normalize([0.5],[0.5])
    return t.unsqueeze(0).unsqueeze(0)                # add batch + channel dims

# %% [model input]  <-- delegable: choose the single channel RITnet sees (B1/B3)
def _inpaint_glint_1ch(chan):
    """Near-saturated pixels are specular glint; dilate + inpaint BEFORE gamma/CLAHE
    so CLAHE cannot amplify the blowout into the mask (B3). Single-channel analogue of
    remove_glint(); no saturation test here, so it can over-inpaint bright skin on the
    red channel. Ablate it (GLINT_PRESEG) rather than trusting it."""
    glint = (chan >= GLINT_V).astype(np.uint8)
    if not glint.any():
        return chan
    glint = cv2.dilate(glint, np.ones((GLINT_DILATE, GLINT_DILATE), np.uint8))
    return cv2.inpaint(chan, glint, 3, cv2.INPAINT_TELEA)

def ritnet_input(bgr):
    """BGR eye crop (any size) -> single-channel uint8 for RITnet. Red channel (B1) or
    luminance, with optional pre-segmentation glint inpaint (B3). This is the ONLY place
    the model's input channel is chosen, so red-vs-gray is a one-flag ablation."""
    chan = bgr[:, :, 2] if USE_RED_CHANNEL else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)  # OpenCV is BGR: [...,2]=red
    if GLINT_PRESEG:
        chan = _inpaint_glint_1ch(chan)
    return np.ascontiguousarray(chan)

# %% [segment]  <-- delegable: forward pass -> per-pixel class mask
@torch.no_grad()
def segment(gray):
    """single-channel frame (red or gray, see ritnet_input) -> (IN_H, IN_W) uint8 mask, class ids 0..3."""
    x = preprocess(gray).to(DEVICE)
    out = model(x)                       # (1, 4, IN_H, IN_W)
    mask = out.argmax(dim=1).squeeze(0)  # (IN_H, IN_W)
    return mask.cpu().numpy().astype(np.uint8)

# %% [measurement layer - PART 1: iris circle from the limbus]  <-- SPINE (Angela writes the math)
# Why: the eyelid covers the top/bottom of the iris, so a plain ellipse fit (5 DOF) wobbles
# on the occluded vertical axis. We instead keep only the iris boundary points that touch the
# SCLERA (the true limbus), and fit a CIRCLE (3 DOF), which the visible left/right arcs pin
# reliably. Off-axis foreshortening is a stated small error and mostly cancels in the ratio.

def fit_circle(xs, ys):
    """Least-squares circle through points (xs, ys) -> (cx, cy, r) algebraic (Kasa) least-squares fit.
    Hint: rearrange (x-cx)^2 + (y-cy)^2 = r^2 into the LINEAR form
          x^2 + y^2 = 2*cx*x + 2*cy*y + (r^2 - cx^2 - cy^2).
    Build A = [2x, 2y, 1], b = x^2 + y^2, solve A @ [cx, cy, c] = b (np.linalg.lstsq),
    then r = sqrt(c + cx^2 + cy^2). Think about why this is linear in the unknowns even
    though a circle is 'nonlinear' (that is the lesson)."""
    n = xs.size
    A = np.empty((n, 3))
    b = np.empty(n)
    for i in range(n):
        A[i][0] = 2 * xs[i]
        A[i][1] = 2 * ys[i]
        A[i][2] = 1
        b[i] = xs[i]**2 + ys[i]**2
    [cx, cy, c], *_ = np.linalg.lstsq(A, b, rcond=None)
    r = np.sqrt(c + cx**2 + cy**2)
    
    return (cx, cy, r)


def fit_circle_robust(xs, ys):
    """Fit a circle, drop points more than `buffer` px off it, refit. Trimmed least-squares.
    NOTE: the loop is fine here; RITnet inference dominates the per-frame cost. If a full-curve
    profile ever flags this, vectorize with a boolean mask (dist = np.hypot(...); keep = ...)."""
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)   # so .size / xs[i] work if given lists
    cx, cy, r = fit_circle(xs, ys)                           # first pass, all points
    buffer = LIMBUS_BUFFER
    passes = 3
    for _ in range(passes):
        new_xs, new_ys = [], []
        for i in range(xs.size):
            dist_from_center = np.hypot(xs[i] - cx, ys[i] - cy)
            if r - buffer < dist_from_center < r + buffer:
                new_xs.append(xs[i])
                new_ys.append(ys[i])
        if len(new_xs) < MIN_PTS:                                # trimmed too hard -> keep first fit
            break
        
        cx, cy, r = fit_circle(np.array(new_xs, float), np.array(new_ys, float))
    return cx, cy, r


def iris_limbus_points(mask):
    """4-class mask -> (xs, ys) arrays of iris-boundary pixels that border the SCLERA.

    Suggested steps (you decide the exact ops):
      1. iris_disk = (mask == CLS_IRIS) | (mask == CLS_PUPIL)        # full disk, ring + pupil
      2. boundary  = iris_disk AND NOT erode(iris_disk)              # 1-px rim of the disk
      3. near_sclera = dilate(mask == CLS_SCLERA, a few px)          # where sclera is adjacent
      4. limbus = boundary AND near_sclera                          # drop lid/lash-side points
      5. ys, xs = np.where(limbus); return xs, ys
    The point of step 3-4: use the class RITnet is GOOD at (sclera) to qualify which iris
    boundary points are trustworthy. Lid-occluded boundary touches background, not sclera."""
    iris_disk = (mask == CLS_IRIS) | (mask == CLS_PUPIL)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(iris_disk.astype(np.uint8))
    if num > 1:                                              # >1 means at least one real blob
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]) # skip bg (row 0), +1 to re-index
        iris_disk = (labels == largest)
    k = np.ones((3, 3), np.uint8)
    eroded  = cv2.erode(iris_disk.astype(np.uint8), k).astype(bool)
    boundary = iris_disk & ~(eroded)
    near_sclera = cv2.dilate((mask == CLS_SCLERA).astype(np.uint8), k, iterations = 2).astype(bool)
    limbus = boundary & near_sclera
    ys, xs = np.where(limbus)
    return xs, ys


def iris_circle(mask):
    """Mask -> (cx, cy, r) iris circle in RITnet-input pixels, or None. Wiring only."""
    xs, ys = iris_limbus_points(mask)
    if xs is None or len(xs) < MIN_PTS:
        return None
    cx, cy, r = fit_circle_robust(np.asarray(xs, float), np.asarray(ys, float))
    if not np.isfinite([cx, cy, r]).all() or r <= 0:
        return None
    return (float(cx), float(cy), float(r))

# %% [measurement layer - PART 2: pupil inside the iris]  glint/masking = plumbing; threshold+blob+fit = SPINE (Angela)
def disk_mask(circle, shape):
    """Boolean (H, W) mask, True for pixels inside the (cx, cy, r) circle. Plumbing."""
    cx, cy, r = circle
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2

def remove_glint(bgr):
    """Inpaint specular highlights (bright AND low-saturation) so a glint sitting on the pupil
    doesn't punch a bright hole in it or bias the fit. Plumbing."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    glint = ((v >= GLINT_V) & (s <= GLINT_S)).astype(np.uint8) * 255
    glint = cv2.dilate(glint, np.ones((GLINT_DILATE, GLINT_DILATE), np.uint8))
    return cv2.inpaint(bgr, glint, 3, cv2.INPAINT_TELEA)

def pupil_in_iris(mask, iris):
    """Color crop (BGR, IN_H x IN_W) + iris circle (cx, cy, r) -> pupil circle (cx, cy, r) or None.
    steps:
      1. inside = disk_mask(iris, bgr.shape[:2])     # only look within the iris        [helper above]
      2. clean  = remove_glint(bgr)                  # kill specular spots first        [helper above]
      3. Pick a channel where the pupil pops from the iris. Convert `clean` to HSV (or Lab) and
         choose one channel. Try saturation S and value V; keep whichever splits best on the
         dark-iris clip (the 'desaturated-dark pupil vs colored-dark iris' idea).
      4. LOCAL Otsu on ONLY the iris pixels of that channel (not the whole image):
             thr, _ = cv2.threshold(channel[inside], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
             pupil_mask = inside & (channel < thr)   # pupil = the low side; flip '<' if needed
         (cv2.threshold runs Otsu on whatever array you hand it, so channel[inside] computes the
          threshold from the iris histogram alone. Then you apply it to the full channel yourself.)
      5. Central blob: connected components on pupil_mask; keep the blob containing / nearest the
         iris center (cx, cy). Same tool as iris_limbus_points.
      6. Fit: the pupil is fully visible (not lid-occluded), so fit directly. Reuse fit_circle on
         the blob boundary, or cv2.minEnclosingCircle. Return (cx, cy, r), or None if none found.
    """
    pupil_mask = (mask == CLS_PUPIL)    # RITnet's pupil pixels
    pupil_mask = cv2.morphologyEx(pupil_mask.astype(np.uint8),
                                  cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)  # smooth raggedness

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(pupil_mask.astype(np.uint8))
    if num <= 1:
        return None
    icx, icy = iris[0], iris[1]
    d = np.hypot(centroids[1:, 0] - icx, centroids[1:, 1] - icy)   # each blob's distance to iris center
    lbl = 1 + int(np.argmin(d))
    blob = (labels == lbl)
    k = np.ones((3, 3), np.uint8)
    boundary = blob & ~cv2.erode(blob.astype(np.uint8), k).astype(bool)   # the pupil rim
    ys, xs = np.where(boundary)
    if len(xs) < MIN_PTS:
        return None
    cx, cy, r = fit_circle_robust(xs, ys)
    return (float(cx), float(cy), float(r))


# %% [measure]  combine iris (Part 1) + pupil (Part 2) into one per-frame result
def measure(mask, bgr):
    """4-class mask + aligned color crop (both IN_H x IN_W) -> per-frame measurement dict."""
    xs, ys = iris_limbus_points(mask)                # points that feed the iris fit
    ic = None
    limbus_in = limbus_out = None
    if xs is not None and len(xs) >= MIN_PTS:
        cx, cy, r = fit_circle_robust(np.asarray(xs, float), np.asarray(ys, float))
        if np.isfinite([cx, cy, r]).all() and r > 0:
            ic = (float(cx), float(cy), float(r))
            keep = np.abs(np.hypot(xs - cx, ys - cy) - r) < LIMBUS_BUFFER   # for the overlay
            limbus_in  = (xs[keep],  ys[keep])
            limbus_out = (xs[~keep], ys[~keep])
    pupil = pupil_in_iris(mask, ic) if ic is not None else None
    iris_px  = 2.0 * ic[2]    if ic    else np.nan   # diameter = 2r
    pupil_px = 2.0 * pupil[2] if pupil else np.nan
    ratio = pupil_px / iris_px if (np.isfinite(pupil_px) and iris_px > 0) else np.nan
    return dict(iris_circle=ic, iris_px=iris_px, pupil_circle=pupil, pupil_px=pupil_px, ratio=ratio,
                limbus_xy=(xs, ys), limbus_in=limbus_in, limbus_out=limbus_out)

# %% [measurement layer - PART 3: iris-lock + geometry/temporal gate]
def lock_iris_radius(radii):
    """The iris is a fixed ruler within a clip, so its true radius is one constant. Take the
    median of the finite per-frame radii; that median is robust to the odd bad frame."""
    radii = np.asarray(radii, float)
    return float(np.nanmedian(radii)) if np.isfinite(radii).any() else np.nan

def light_onset_frame(clip_stem, fps=FPS):
    """Frame index at/after which the flashlight is ON for this clip -> those frames are LIT.
    Read from a manual mark now (LIGHT_ONSET_FILE, onset time in seconds). IN THE APP: replace
    this lookup with the phone's own torch-on timestamp - a KNOWN signal, so onset is never
    inferred or learned. Returns None if unmarked (build_curve then locks from ALL frames)."""
    if not LIGHT_ONSET_FILE.exists():
        return None
    onset_s = json.loads(LIGHT_ONSET_FILE.read_text()).get(clip_stem)
    return None if onset_s is None else int(round(float(onset_s) * fps))

def passes_gate(iris_center, iris_r, pupil):
    """Geometry sanity for one frame's pupil read. Reject if the pupil is not plausibly sized,
    or not roughly concentric with / contained in the iris. Thresholds are config (GATE_*)."""
    if pupil is None or iris_center is None or not np.isfinite(iris_r) or iris_r <= 0:
        return False
    pcx, pcy, pr = pupil
    ratio = pr / iris_r
    if not (GATE_RATIO_MIN <= ratio <= GATE_RATIO_MAX):                  # implausible size
        return False
    if np.hypot(pcx - iris_center[0], pcy - iris_center[1]) > GATE_CONCENTRIC * iris_r:  # off-center
        return False
    return True

def reject_temporal_outliers(y, window=TEMPORAL_WIN, k=TEMPORAL_K):
    """Set to NaN any point > k local-MADs from a ROLLING median. Local (not global) so the real
    constriction, which is a smooth trend, is preserved while single-frame spikes are killed."""
    y = np.asarray(y, float)
    out = y.copy()
    half = window // 2
    for i in range(len(y)):
        if not np.isfinite(y[i]):
            continue
        w = y[max(0, i - half): i + half + 1]
        w = w[np.isfinite(w)]
        if len(w) < 3:
            continue
        med = np.median(w)
        mad = np.median(np.abs(w - med)) + 1e-6
        if abs(y[i] - med) > k * mad:
            out[i] = np.nan
    return out

def build_curve(clip_path, roi, seed_box=None):
    """One SEQUENTIAL pass over a clip -> pupil-vs-time curve, iris locked and gated.
    If TRACK_EYE and seed_box (the tight eye_rois box) is given, the eye is FOLLOWED
    per-frame so handheld drift never pushes it out of the crop (A5); otherwise the fixed
    `roi` is used for every frame. Reads frames in order (no per-frame seek, which was the
    slow path on 4K clips) and measures every CURVE_STRIDE-th; tracking still runs every
    frame so continuity is preserved. Returns dict(t, iris_px, pupil_px, ratio, valid, onset_t)."""
    cap = cv2.VideoCapture(str(clip_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tracker, tracking = None, (TRACK_EYE and seed_box is not None)
    iris_c, iris_r, pupil, kept = [], [], [], []
    fi = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        if tracking:                                                    # follow the eye EVERY frame (continuity)
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if tracker is None:
                tracker = EyeTracker(g, seed_box); tight = tuple(seed_box)
            else:
                tight = tracker.update(g)
            frame_roi = framed_roi(tight, frame.shape)
        else:
            frame_roi = roi
        if fi % CURVE_STRIDE:                                           # measure only every Nth frame
            continue
        fr = crop(frame, frame_roi)
        crop_bgr = cv2.resize(fr, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
        m = measure(segment(ritnet_input(fr)), crop_bgr)
        ic = m["iris_circle"]
        iris_c.append((ic[0], ic[1]) if ic else None)
        iris_r.append(ic[2] if ic else np.nan)
        pupil.append(m["pupil_circle"])
        kept.append(fi)
        if len(kept) % 25 == 0:
            print(f"  ...{fi}/{n} frames processed", end="\r")
    cap.release()
    print(f"  {len(kept)} frames measured / {n} total" + (" (eye-tracked)" if tracking else ""))

    onset = light_onset_frame(Path(clip_path).stem)                     # KNOWN light-on (manual now, phone later)
    lit = np.array(kept) >= onset if onset is not None else np.ones(len(kept), bool)
    lit_radii = [r for r, L in zip(iris_r, lit) if L]                   # lock the ruler from LIT frames only
    locked_r = lock_iris_radius(lit_radii if any(lit) else iris_r)      # fall back to all if none flagged lit
    pupil_px = np.array([2.0 * p[2] if p else np.nan for p in pupil], float)
    for j, (c, p) in enumerate(zip(iris_c, pupil)):                      # gate each frame vs the LOCKED iris
        if not passes_gate(c, locked_r, p):
            pupil_px[j] = np.nan
    pupil_px = reject_temporal_outliers(pupil_px)                        # kill single-frame spikes
    iris_px = np.full(len(pupil_px), 2.0 * locked_r)
    t = np.array(kept, float) / FPS
    onset_t = onset / FPS if onset is not None else None                 # KNOWN light-on, in curve-t units (for Stage 2 windowing)
    return dict(t=t, iris_px=iris_px, pupil_px=pupil_px, ratio=pupil_px / iris_px,
                valid=np.isfinite(pupil_px), onset_t=onset_t)

# %% [test] fit_circle in isolation  <-- delegable: validates YOUR circle math, no RITnet needed
# Run after implementing fit_circle, BEFORE the full pipeline. Run cells: imports -> Part 1 -> this.
# The key case is #3: only the left/right arcs survive (lid covers top/bottom), and a 3-DOF
# circle still recovers the radius. That is exactly why we chose circle over ellipse.
if __name__ == "__main__":   # test cell: runs interactively / as a script, skipped on import
    def _circle_points(cx, cy, r, angles, noise=0.0, seed=0):
        rng = np.random.default_rng(seed)
        x = cx + r * np.cos(angles) + rng.normal(0, noise, angles.shape)
        y = cy + r * np.sin(angles) + rng.normal(0, noise, angles.shape)
        return x, y

    def _check(name, cx, cy, r, x, y, tol_c, tol_r, assert_ok=True):
        fcx, fcy, fr = fit_circle(x, y)
        ec, er = float(np.hypot(fcx - cx, fcy - cy)), float(abs(fr - r))
        tag = "PASS" if (ec <= tol_c and er <= tol_r) else ("FAIL" if assert_ok else "info")
        print(f"[{tag:4s}] {name:22s} n={len(x):3d}  center_err={ec:5.2f} (tol {tol_c})  radius_err={er:5.2f} (tol {tol_r})")
        if assert_ok:
            assert ec <= tol_c and er <= tol_r, f"{name}: center off {ec:.2f}, radius off {er:.2f}"

    CX, CY, R = 200.0, 320.0, 90.0           # a plausible iris circle in 400x640 space
    ang = np.linspace(0, 2 * np.pi, 200, endpoint=False)

    _check("full clean",  CX, CY, R, *_circle_points(CX, CY, R, ang),                 0.5, 0.5)
    _check("full + noise", CX, CY, R, *_circle_points(CX, CY, R, ang, noise=1.5),     2.0, 2.0)

    # lid occlusion: keep only side arcs (|cos| large == near-horizontal points)
    side = np.abs(np.cos(ang)) > 0.34        # ~+-70 deg arcs on left and right; tighten to test harder
    _check("side arcs (lid)", CX, CY, R, *_circle_points(CX, CY, R, ang[side], noise=1.5, seed=1), 3.0, 3.0)

    # one side only: ill-conditioned, shown to mark the limit (no assert)
    one = np.cos(ang) > 0.5
    _check("one side (limit)", CX, CY, R, *_circle_points(CX, CY, R, ang[one], noise=1.0, seed=2), 8.0, 8.0, assert_ok=False)

    print("\nfit_circle: asserted cases passed")

# %% [test] fit_circle_robust convergence  <-- delegable: proves the iteration rejects outliers
# Plant a known circle (side arcs, like a real lidded iris), then inject the two junk types
# we actually see: a CENTER BLOB (reflection mislabeled as sclera) and a LID STRAND (iris label
# smeared along the lid). The naive one-shot fit gets dragged off; the iterated robust fit should
# recover the true circle. Run cells: imports -> Part 1 -> this.
if __name__ == "__main__":   # test cell: runs interactively / as a script, skipped on import
    rng = np.random.default_rng(7)

    # good limbus: left/right arcs of the true circle, with noise (this is the signal)
    gx, gy = _circle_points(CX, CY, R, ang[np.abs(np.cos(ang)) > 0.34], noise=1.5, seed=3)

    # outlier 1: center blob (~90px inside the true circle -> should be trimmed)
    bx = CX + rng.uniform(-12, 12, 25)
    by = CY + rng.uniform(-12, 12, 25)

    # outlier 2: lid strand, a short off-circle arc ~18px outside the top (-> should be trimmed)
    sx = CX + rng.uniform(-30, 30, 15)
    sy = (CY - (R + 18)) + rng.uniform(-3, 3, 15)

    x_all = np.concatenate([gx, bx, sx])
    y_all = np.concatenate([gy, by, sy])

    ncx, ncy, nr = fit_circle(x_all, y_all)                 # naive, one shot (all points)
    rcx, rcy, rr = fit_circle_robust(x_all, y_all)          # iterated, trims outliers
    naive_err  = abs(nr - R)
    robust_c   = float(np.hypot(rcx - CX, rcy - CY))
    robust_r   = float(abs(rr - R))
    print(f"naive  fit: r={nr:5.1f}  radius_err={naive_err:5.2f}")
    print(f"robust fit: r={rr:5.1f}  center_err={robust_c:5.2f}  radius_err={robust_r:5.2f}")

    assert robust_c < 3.0 and robust_r < 3.0, "robust fit did not recover the true circle"
    assert robust_r < naive_err, "robust fit should beat the naive fit on the contaminated set"
    print("fit_circle_robust: converged and beat the naive fit")

# %% [overlay helper]  <-- delegable debug dump
def _draw_circle(img, circ, color):
    if circ is None:
        return
    cx, cy, r = circ
    cv2.circle(img, (int(round(cx)), int(round(cy))), int(round(r)), color, 2)
    cv2.circle(img, (int(round(cx)), int(round(cy))), 2, color, -1)   # center dot

def overlay(frame_bgr, m):
    """Resize to RITnet input size, draw the fitted iris (blue), and scatter limbus points:
    green = kept by the fit, red = trimmed. If the split isn't available, all points in red."""
    vis = cv2.resize(frame_bgr, (IN_W, IN_H), interpolation=cv2.INTER_AREA)

    def _scatter(xy, color):
        if xy is not None and xy[0] is not None and len(xy[0]) > 0:
            xs, ys = xy
            vis[np.clip(ys, 0, IN_H - 1), np.clip(xs, 0, IN_W - 1)] = color

    if m.get("limbus_in") is not None:
        _scatter(m.get("limbus_out"), (0, 0, 255))     # red    = trimmed (ignored by the fit)
        _scatter(m.get("limbus_in"),  (0, 255, 255))   # yellow = kept limbus points
    else:
        _scatter(m.get("limbus_xy"),  (0, 0, 255))     # fallback: raw points, all red
    _draw_circle(vis, m.get("iris_circle"),  (255, 0, 0))   # blue  = fitted iris circle
    _draw_circle(vis, m.get("pupil_circle"), (0, 255, 0))   # green = fitted pupil circle
    return vis

# %% [eye ROI]  <-- delegable: crop the eye so it fills the frame, RITnet-style
def _load_rois():
    return json.loads(ROI_FILE.read_text()) if ROI_FILE.exists() else {}

def _save_rois(rois):
    ROI_FILE.write_text(json.dumps(rois, indent=2))

def get_roi(clip_stem, first_frame, rois):
    """Return (x,y,w,h) eye box in ORIGINAL pixels, or None if not set yet.
    No-GUI by default: boxes come from eye_rois.json (use the preview cell below to
    read coordinates). ROI_GUI=True falls back to cv2.selectROI, which can hang on macOS."""
    if clip_stem in rois:
        return tuple(rois[clip_stem])
    if not ROI_GUI:
        return None                                   # caller warns; do NOT hang
    box = cv2.selectROI(f"eye box: {clip_stem}", first_frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(f"eye box: {clip_stem}")
    if box[2] == 0 or box[3] == 0:
        return None
    rois[clip_stem] = [int(v) for v in box]
    _save_rois(rois)
    return tuple(rois[clip_stem])

def expand_to_openeds(box, frame_shape):
    """Tight eye box -> OpenEDS-style framing box. Eye fills the width; the box is grown
    to portrait aspect (IN_H/IN_W) around the eye center, adding brow above / cheek below
    so the eye lands in the central vertical third. Clamped to the frame."""
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    new_w = w * WIDTH_FILL
    new_h = new_w * (IN_H / IN_W)              # portrait: ~1.6x the width
    nx, ny = cx - new_w / 2.0, cy - new_h / 2.0
    H, W = frame_shape[:2]
    nx = max(0, min(nx, W - 1)); ny = max(0, min(ny, H - 1))
    nw = min(new_w, W - nx);     nh = min(new_h, H - ny)
    return (int(nx), int(ny), int(nw), int(nh))

def framed_roi(box, frame_shape):
    return expand_to_openeds(box, frame_shape) if EXPAND_ROI else tuple(box)

def crop(frame, roi):
    x, y, w, h = roi
    return frame[y:y + h, x:x + w]

# %% [eye tracking]  <-- delegable (A5): follow the eye per-frame so handheld drift
# does not push it out of the crop. A single fixed box only works when the camera is
# static within a clip; on drifting clips (e.g. IMG_4569, ~570px of travel) the eye
# leaves the frame and the masks fail. This tracks it.
TRACK_EYE     = True     # per-frame ROI via template tracking; False = one fixed box per clip
SEARCH_MARGIN = 140      # px searched around the previous eye center (frame-to-frame motion is small at 120fps)
MATCH_MIN     = 0.30     # normalized-correlation floor; below it we HOLD the last center (blink/blur/eye gone)

class EyeTracker:
    """Follows the eye across a handheld clip. Seeded from the clip's eye_rois.json box
    (frame 0 becomes the template). Each frame, normalized cross-correlation (TM_CCOEFF_NORMED,
    illumination-robust so it survives the flashlight onset) relocates the eye within a window
    around the last center. The box SIZE stays fixed (the seed size); only its CENTER moves,
    the same 'fixed ruler, moving center' idea as the iris lock. update() returns a per-frame
    TIGHT box for framed_roi() to expand to OpenEDS framing. Handles translation, not zoom/roll."""
    def __init__(self, seed_frame_gray, seed_box):
        x, y, w, h = seed_box
        self.tpl = seed_frame_gray[y:y + h, x:x + w].copy()   # frame-0 appearance = the target
        self.w, self.h = w, h
        self.cx, self.cy = x + w / 2.0, y + h / 2.0
        self.last_score = 1.0

    def update(self, frame_gray):
        H, W = frame_gray.shape
        x0 = int(max(0, self.cx - self.w / 2 - SEARCH_MARGIN)); y0 = int(max(0, self.cy - self.h / 2 - SEARCH_MARGIN))
        x1 = int(min(W, self.cx + self.w / 2 + SEARCH_MARGIN)); y1 = int(min(H, self.cy + self.h / 2 + SEARCH_MARGIN))
        region = frame_gray[y0:y1, x0:x1]
        if region.shape[0] >= self.h and region.shape[1] >= self.w:
            res = cv2.matchTemplate(region, self.tpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(res)
            self.last_score = float(score)
            if score >= MATCH_MIN:                            # else: hold last center (occlusion/blur)
                self.cx = x0 + loc[0] + self.w / 2.0
                self.cy = y0 + loc[1] + self.h / 2.0
        return (int(self.cx - self.w / 2), int(self.cy - self.h / 2), self.w, self.h)

# %% [pick ROIs, no GUI]  show each clip's first frame with a pixel grid; fill eye_rois.json
# HOW TO USE:
#   1. Run this cell. For each clip you'll see frame 0 with gridlines labeled in pixels.
#   2. Read off the eye box: x,y = TOP-LEFT corner, w,h = width,height (add a little margin).
#   3. Put it in eye_rois.json, e.g.  "IMG_4598 2": [820, 560, 520, 360]
#   4. Re-run this cell to confirm the drawn red box hugs the eye, then run validation.
# A starter eye_rois.json with every clip is written below; just replace the zeros.
def preview_rois():
    clips = sorted(p for p in Path(CLIPS_DIR).glob("*") if p.suffix.lower() in {".mov", ".mp4", ".avi", ".mkv"})
    rois = _load_rois()
    rois = {**{c.stem: rois.get(c.stem, [0, 0, 0, 0]) for c in clips}}   # ensure every clip has an entry
    _save_rois(rois)
    for c in clips:
        cap = cv2.VideoCapture(str(c)); cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read(); cap.release()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        _, ax = plt.subplots(figsize=(5, 5 * h / w))
        ax.imshow(rgb)
        ax.set_xticks(np.arange(0, w, 200)); ax.set_yticks(np.arange(0, h, 200))
        ax.grid(color="cyan", lw=0.5, alpha=0.6)
        box = rois[c.stem]
        if box[2] > 0 and box[3] > 0:           # draw tight box (red) + expanded framing (yellow)
            x, y, bw, bh = box
            ax.add_patch(plt.Rectangle((x, y), bw, bh, fill=False, ec="red", lw=2))
            if EXPAND_ROI:
                ex, ey, ew, eh = expand_to_openeds(box, rgb.shape)
                ax.add_patch(plt.Rectangle((ex, ey), ew, eh, fill=False, ec="yellow", lw=2, ls="--"))
            ax.set_title(f"{c.stem}  tight(red)={box}", fontsize=8)
        else:
            ax.set_title(f"{c.stem}  -- NO ROI YET, read x,y,w,h and edit eye_rois.json", fontsize=8)
        plt.tight_layout(); plt.show()

if __name__ == "__main__":   # demo cell: runs interactively / as a script, skipped on import
    preview_rois()
    print(f"edit {ROI_FILE} (replace the [0,0,0,0] boxes), then re-run this cell to verify the red box")

# %% [run validation]  sample frames per clip, segment, measure, dump overlays
if __name__ == "__main__":   # demo cell: runs interactively / as a script, skipped on import
    clips = sorted(p for p in Path(CLIPS_DIR).glob("*") if p.suffix.lower() in {".mov", ".mp4", ".avi", ".mkv"})
    assert clips, f"no videos in {CLIPS_DIR}/"
    out_root = Path("debug_masks"); out_root.mkdir(exist_ok=True)
    rois = _load_rois()
    # tag overlays by the model-input mode so red-vs-gray runs persist side by side (B1 ablation)
    INPUT_TAG = ("red" if USE_RED_CHANNEL else "gray") + ("_deglint" if GLINT_PRESEG else "")
    print(f"{len(clips)} clip(s); sampling {N_SAMPLE} frames each; USE_ROI={USE_ROI}; input={INPUT_TAG}\n")

    for clip in clips:
        cap = cv2.VideoCapture(str(clip))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, max(n - 1, 0), N_SAMPLE, dtype=int)   # evenly spaced across the clip
        sub = out_root / f"{clip.stem}__{INPUT_TAG}"; sub.mkdir(exist_ok=True)

        roi = None
        if USE_ROI:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok0, frame0 = cap.read()
            tight = get_roi(clip.stem, frame0, rois) if ok0 else None
            if tight is None or tight[2] == 0 or tight[3] == 0:
                print(f"-- skip {clip.name}: no eye ROI set. Run the preview cell and fill eye_rois.json")
                cap.release(); continue
            roi = framed_roi(tight, frame0.shape)   # expand tight eye box to OpenEDS framing

        print(f"== {clip.name} ({n} frames)  roi={roi} ==")
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                continue
            if roi is not None:
                frame = crop(frame, roi)
            crop_bgr = cv2.resize(frame, (IN_W, IN_H), interpolation=cv2.INTER_AREA)  # color, aligned to mask
            mask = segment(ritnet_input(frame))
            m = measure(mask, crop_bgr)
            cv2.imwrite(str(sub / f"frame_{int(i):05d}.png"), overlay(crop_bgr, m))
            print(f"  f{int(i):5d}  iris={m['iris_px']:6.1f}px  pupil={m['pupil_px']:6.1f}px  ratio={m['ratio']:.3f}")
        cap.release()
    print(f"\nwrote overlays to {out_root.resolve()}/<clip>/")
    print("legend: blue=iris circle, green=pupil circle, yellow dots=kept limbus, red dots=trimmed")

# %% [run curve]  build the pupil-vs-time curve for ONE clip (iris locked + gated + spikes removed)
if __name__ == "__main__":   # demo cell: runs interactively / as a script, skipped on import
    CURVE_CLIP = "subject_b_2"     # <-- clip stem to plot
    _cl = next(p for p in Path(CLIPS_DIR).glob("*") if p.stem == CURVE_CLIP)
    _rois = _load_rois()
    _c0 = cv2.VideoCapture(str(_cl)); _c0.set(cv2.CAP_PROP_POS_FRAMES, 0); _ok0, _f0 = _c0.read(); _c0.release()
    _seed = get_roi(CURVE_CLIP, _f0, _rois)            # TIGHT eye box; seeds the per-frame tracker
    _roi = framed_roi(_seed, _f0.shape)               # fixed-box fallback if TRACK_EYE is off
    cur = build_curve(_cl, _roi, seed_box=_seed)       # eye-tracked when TRACK_EYE; sequential read (faster)
    print(f"{CURVE_CLIP}: {cur['valid'].mean() * 100:.0f}% frames valid, locked iris = {cur['iris_px'][0]:.1f}px")

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True, height_ratios=[3, 1])
    ax[0].plot(cur['t'], cur['pupil_px'], '.-', ms=3, lw=1)
    ax[0].set_ylabel("pupil diameter (px)")
    ax[0].set_title(f"pupil vs time, {CURVE_CLIP}  (iris locked + geometry/temporal gate)")
    ax[1].plot(cur['t'], cur['ratio'], '.-', ms=3, lw=1, color="purple")
    ax[1].set_ylabel("pupil / iris"); ax[1].set_xlabel("time (s)")
    plt.tight_layout(); plt.show()

# %% [eval harness]  <-- delegable eval scaffolding (valid% + C5 reporting), NOT spine math.
# Runs every clip through build_curve, RED vs GRAY, and reports the gate PASS RATE plus the
# quality stats the doc's C5 asks for: frames measured, frames rejected, and the LONGEST
# contiguous rejected run. A clip can be 70% valid overall yet have its entire pre-flash
# baseline dropped, which makes it useless; valid% alone hides that. Onset t0 is YOUR spine,
# so this stays t0-free and reports WHERE the longest gap sits (start-end time) instead, so a
# baseline-clustered failure is still visible. This is the before/after metric for the fine-tune.
EVAL_STRIDE = 9          # coarser than CURVE_STRIDE for a faster sweep; lower = more detail, slower
EVAL_MODES  = ["gray", "red"]   # channels to compare; drop one to halve the runtime

def longest_reject_run(valid, t):
    """Longest run of consecutive INVALID (gated-out) frames -> (n_frames, t_start, t_end)."""
    best_n, best_ts, best_te = 0, None, None
    run, start = 0, 0
    for i, ok in enumerate(valid):
        if not ok:
            if run == 0:
                start = i
            run += 1
            if run > best_n:
                best_n, best_ts, best_te = run, t[start], t[i]
        else:
            run = 0
    return best_n, best_ts, best_te

def clip_stats(cur):
    v = np.asarray(cur["valid"]); t = cur["t"]
    n, nv = len(v), int(v.sum())
    gn, gts, gte = longest_reject_run(v, t)
    span = f"{gts:.2f}-{gte:.2f}s" if gts is not None else "-"
    return dict(measured=n, valid=nv, pct=100.0 * nv / max(n, 1), rejected=n - nv, gap=gn, span=span)

def eval_all(stride=EVAL_STRIDE, modes=EVAL_MODES):
    """Sweep every clip x every mode, print a valid%/reject/longest-gap table, return the curves.
    Toggles USE_RED_CHANNEL per mode and restores it after (so the notebook state is unchanged)."""
    global USE_RED_CHANNEL, CURVE_STRIDE
    clips = sorted(p for p in Path(CLIPS_DIR).glob("*") if p.suffix.lower() in {".mov", ".mp4", ".avi", ".mkv"})
    rois = _load_rois()
    saved_red, saved_stride = USE_RED_CHANNEL, CURVE_STRIDE
    CURVE_STRIDE = stride
    results = {}
    try:
        for clip in clips:
            c0 = cv2.VideoCapture(str(clip)); c0.set(cv2.CAP_PROP_POS_FRAMES, 0); ok, f0 = c0.read(); c0.release()
            if not ok:
                print(f"skip {clip.stem}: unreadable"); continue
            seed = get_roi(clip.stem, f0, rois)
            if seed is None:
                print(f"skip {clip.stem}: no ROI in eye_rois.json"); continue
            roi = framed_roi(seed, f0.shape)
            for mode in modes:
                USE_RED_CHANNEL = (mode == "red")
                print(f"[{clip.stem} | {mode}]")
                results[(clip.stem, mode)] = clip_stats(build_curve(clip, roi, seed_box=seed))
    finally:
        USE_RED_CHANNEL, CURVE_STRIDE = saved_red, saved_stride   # never leave the globals flipped

    print(f"\n{'clip':16s} {'mode':5s} {'valid%':>7s} {'meas':>5s} {'rej':>4s} {'longest reject run':>22s}")
    print("-" * 64)
    for (stem, mode), s in results.items():
        print(f"{stem:16s} {mode:5s} {s['pct']:6.1f}% {s['measured']:5d} {s['rejected']:4d}   {s['gap']:4d}f @ {s['span']}")
    return results

if __name__ == "__main__":   # demo cell: runs interactively / as a script, skipped on import
    eval_results = eval_all()

# %% [optional] inspect the raw 4-class mask for one frame (sanity-check the classes)
# Uncomment to see what RITnet actually labels. Useful if a brown iris gets merged into pupil.
# cap = cv2.VideoCapture(str(clips[0])); cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
# ok, frame = cap.read(); cap.release()
# mask = segment(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
# plt.figure(figsize=(4, 6)); plt.imshow(mask, cmap="viridis"); plt.colorbar(); plt.title("RITnet classes 0..3"); plt.show()

# %% [diagnose Part 2]  see glint removal + which channel actually isolates the pupil
# Set DIAG_CLIP to the clip you want to inspect (stem, no extension). Shows, for one frame:
# original vs glint-removed, what glint removal changed, and the pupil_mask from V vs from S.
if __name__ == "__main__":   # demo cell: runs interactively / as a script, skipped on import
    DIAG_CLIP  = "subject_b_2"     # <-- change to your darkest-iris clip stem
    DIAG_FRAME = None             # None = middle frame, or an int index

    _clip = next(p for p in Path(CLIPS_DIR).glob("*") if p.stem == DIAG_CLIP)
    _rois = _load_rois()
    _cap = cv2.VideoCapture(str(_clip)); _nf = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _fi = _nf // 2 if DIAG_FRAME is None else DIAG_FRAME
    _cap.set(cv2.CAP_PROP_POS_FRAMES, _fi); _ok, _fr = _cap.read(); _cap.release()
    _fr = crop(_fr, framed_roi(get_roi(DIAG_CLIP, _fr, _rois), _fr.shape))
    diag_bgr = cv2.resize(_fr, (IN_W, IN_H), interpolation=cv2.INTER_AREA)

    diag_mask = segment(ritnet_input(diag_bgr))
    diag_ic   = iris_circle(diag_mask)
    diag_in   = disk_mask(diag_ic, diag_bgr.shape[:2])
    diag_clean = remove_glint(diag_bgr)
    _hsv = cv2.cvtColor(diag_clean, cv2.COLOR_BGR2HSV)
    diag_s, diag_v = _hsv[..., 1], _hsv[..., 2]

    ritnet_pupil = (diag_mask == CLS_PUPIL)           # RITnet's OWN pupil class, on the framed crop
    diag_pupil = pupil_in_iris(diag_mask, diag_ic)    # the circle the pipeline now returns (from RITnet pupil)

    # draw the fitted iris (blue) + pupil (green) on RITnet's pupil mask, so mask and circle line up
    mask_vis = cv2.cvtColor(ritnet_pupil.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    _draw_circle(mask_vis, diag_ic, (255, 0, 0))
    _draw_circle(mask_vis, diag_pupil, (0, 255, 0))

    print(f"iris_r={diag_ic[2]:.1f}  pupil_r={diag_pupil[2] if diag_pupil else float('nan'):.1f}"
          f"  ratio={(diag_pupil[2]/diag_ic[2]) if diag_pupil else float('nan'):.3f}"
          f"  RITnet pupil px={int(ritnet_pupil.sum())}")

    _f, _ax = plt.subplots(1, 4, figsize=(15, 6))
    _ax[0].imshow(cv2.cvtColor(diag_bgr, cv2.COLOR_BGR2RGB)); _ax[0].set_title("original crop")
    _ax[1].imshow(diag_s, cmap="gray");                       _ax[1].set_title("S (saturation), reference")
    _ax[2].imshow(ritnet_pupil, cmap="gray");                _ax[2].set_title("RITnet pupil class (raw)")
    _ax[3].imshow(cv2.cvtColor(mask_vis, cv2.COLOR_BGR2RGB)); _ax[3].set_title("RITnet pupil + fitted circles")
    for a in _ax:
        a.axis("off")
    plt.tight_layout(); plt.show()

# %%
