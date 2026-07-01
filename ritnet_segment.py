# %% [markdown]
# Phase 1b: run RITnet on the real close-up clips and VALIDATE the pupil+iris masks.
# RITnet = small pretrained 4-class eye-segmentation U-Net (bg / sclera / iris / pupil).
#
# SPINE (Angela owns, interviewer will probe):
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
DEVICE      = "cpu"           # validation is a handful of frames; CPU is plenty. "mps" later for the full curve.
MIN_PTS     = 5               # fitEllipse needs >= 5 contour points
LIMBUS_BUFFER = 6             # px; a limbus point farther than this from the fitted circle is trimmed

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

# %% [segment]  <-- delegable: forward pass -> per-pixel class mask
@torch.no_grad()
def segment(gray):
    """grayscale frame -> (IN_H, IN_W) uint8 mask with class ids 0..3."""
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

def measure(mask):
    """Class mask -> per-frame measurement dict. PART 1 only for now: iris circle.
    Also returns the limbus points used, for debugging the selection."""
    xs, ys = iris_limbus_points(mask)                # the points that feed the fit
    ic = None
    limbus_in = limbus_out = None
    if xs is not None and len(xs) >= MIN_PTS:
        cx, cy, r = fit_circle_robust(np.asarray(xs, float), np.asarray(ys, float))
        if np.isfinite([cx, cy, r]).all() and r > 0:
            ic = (float(cx), float(cy), float(r))
            # split the raw points by agreement with the FINAL circle, for the overlay
            keep = np.abs(np.hypot(xs - cx, ys - cy) - r) < LIMBUS_BUFFER
            limbus_in  = (xs[keep],  ys[keep])       # green: used by the fit
            limbus_out = (xs[~keep], ys[~keep])      # red: trimmed / ignored
    iris_px = 2.0 * ic[2] if ic else np.nan          # diameter = 2r
    return dict(iris_circle=ic, iris_px=iris_px, limbus_xy=(xs, ys),
                limbus_in=limbus_in, limbus_out=limbus_out,
                pupil_circle=None, pupil_px=np.nan, ratio=np.nan)  # pupil filled in Part 2

# %% [test] fit_circle in isolation  <-- delegable: validates YOUR circle math, no RITnet needed
# Run after implementing fit_circle, BEFORE the full pipeline. Run cells: imports -> Part 1 -> this.
# The key case is #3: only the left/right arcs survive (lid covers top/bottom), and a 3-DOF
# circle still recovers the radius. That is exactly why we chose circle over ellipse.
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
        _scatter(m.get("limbus_out"), (0, 0, 255))   # red   = trimmed (ignored by the fit)
        _scatter(m.get("limbus_in"),  (0, 255, 0))   # green = kept (used by the fit)
    else:
        _scatter(m.get("limbus_xy"),  (0, 0, 255))   # fallback: raw points, all red
    _draw_circle(vis, m.get("iris_circle"), (255, 0, 0))   # blue = fitted iris circle
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

preview_rois()
print(f"edit {ROI_FILE} (replace the [0,0,0,0] boxes), then re-run this cell to verify the red box")

# %% [run validation]  sample frames per clip, segment, measure, dump overlays
clips = sorted(p for p in Path(CLIPS_DIR).glob("*") if p.suffix.lower() in {".mov", ".mp4", ".avi", ".mkv"})
assert clips, f"no videos in {CLIPS_DIR}/"
out_root = Path("debug_masks"); out_root.mkdir(exist_ok=True)
rois = _load_rois()
print(f"{len(clips)} clip(s); sampling {N_SAMPLE} frames each; USE_ROI={USE_ROI}\n")

for clip in clips:
    cap = cv2.VideoCapture(str(clip))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(n - 1, 0), N_SAMPLE, dtype=int)   # evenly spaced across the clip
    sub = out_root / clip.stem; sub.mkdir(exist_ok=True)

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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = segment(gray)
        m = measure(mask)
        cv2.imwrite(str(sub / f"frame_{int(i):05d}.png"), overlay(frame, m))
        nlp = 0 if m["limbus_xy"][0] is None else len(m["limbus_xy"][0])
        print(f"  f{int(i):5d}  iris={m['iris_px']:6.1f}px  limbus_pts={nlp:4d}")
    cap.release()
print(f"\nwrote overlays to {out_root.resolve()}/<clip>/")
print("red dots = limbus points used; blue = fitted iris circle. Red should hug the iris/SCLERA edge (sides).")

# %% [optional] inspect the raw 4-class mask for one frame (sanity-check the classes)
# Uncomment to see what RITnet actually labels. Useful if a brown iris gets merged into pupil.
# cap = cv2.VideoCapture(str(clips[0])); cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
# ok, frame = cap.read(); cap.release()
# mask = segment(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
# plt.figure(figsize=(4, 6)); plt.imshow(mask, cmap="viridis"); plt.colorbar(); plt.title("RITnet classes 0..3"); plt.show()
