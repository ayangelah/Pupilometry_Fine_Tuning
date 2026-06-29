# %% [markdown]
# PLR MVP, step 1: extract a pupil-diameter-vs-time curve from ONE clip.
# No MediaPipe. Direct pupil segmentation, which is the right tool for
# eye-only close-ups (your clips and the RAPD-HD footage).
#
# SPINE (own this, it's what an interviewer will probe):
#   - estimate_pupil_diameter(): how a frame becomes a single diameter number
#   - the brightness-based stimulus onset detection
# DELEGABLE (fine to lean on libraries/boilerplate):
#   - video I/O, plotting, the debug-frame dump

# %% [imports]
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# %% [config]
CLIPS_DIR  = "data/clips"   # process every video in here
FPS        = 120.0          # your capture rate
DOWNSCALE  = 0.5            # 0.5 = process at half the 4K resolution for speed; set to 1.0 for full res
MIN_AREA   = 200            # contour smaller than this (in processed px) is treated as noise/blink
DEBUG_FRAMES = [0, 60, 180, 360, 600]  # frame indices to save annotated, so you can SEE the detection

# %% [detector]  <-- SPINE
def estimate_pupil_diameter(gray):
    """One grayscale eye frame -> (diameter_in_px, ellipse or None).

    The pupil is the darkest large blob. Otsu on the inverted image isolates it,
    a morphological close fills the specular glint from the light, and we fit an
    ellipse to the largest contour. Diameter = mean of the two ellipse axes,
    which is more stable than a min-enclosing circle when the pupil is slightly
    off-axis or partially lidded.
    """
    g = cv2.GaussianBlur(gray, (7, 7), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.nan, None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA or len(c) < 5:
        return np.nan, None

    (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(c)
    diameter = (axis_a + axis_b) / 2.0
    return diameter, ((cx, cy), (axis_a, axis_b), angle)

# %% [process one clip]  read frames, build the time series, find onset
# DELEGABLE wrapper around the SPINE: estimate_pupil_diameter() and the
# brightness-onset math below are unchanged, just moved here so we can loop.
def process_clip(video_path):
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"could not open {video_path}"

    diam, bright, dbg = [], [], {}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if DOWNSCALE != 1.0:
            frame = cv2.resize(frame, None, fx=DOWNSCALE, fy=DOWNSCALE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        d, ell = estimate_pupil_diameter(gray)
        diam.append(d)
        bright.append(float(gray.mean()))  # scene luminance, used to find the light onset

        if idx in DEBUG_FRAMES:
            vis = frame.copy()
            if ell is not None:
                (cx, cy), (ax_a, ax_b), ang = ell
                box = ((float(cx), float(cy)), (float(ax_a), float(ax_b)), float(ang))
                cv2.ellipse(vis, box, (0, 255, 0), 2)
            dbg[idx] = vis
        idx += 1

    cap.release()
    diam = np.array(diam, dtype=float)
    bright = np.array(bright, dtype=float)
    t = np.arange(len(diam)) / FPS

    # SPINE: stimulus onset = first big positive jump in scene brightness.
    db = np.diff(bright, prepend=bright[0])
    onset_idx = int(np.argmax(db))      # frame of the steepest brightening
    onset_t = onset_idx / FPS

    return dict(t=t, diam=diam, bright=bright, onset_idx=onset_idx, onset_t=onset_t, dbg=dbg)

# %% [run all clips]  Phase 1b validation pass over data/clips/
clips = sorted(p for p in Path(CLIPS_DIR).glob("*") if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"})
assert clips, f"no videos found in {CLIPS_DIR}/ -- drop your clips there"
print(f"found {len(clips)} clip(s) in {CLIPS_DIR}/")

results = {}
for clip in clips:
    r = process_clip(clip)
    results[clip.name] = r
    found = np.isfinite(r["diam"]).mean() * 100
    print(f"{clip.name:24s} {len(r['diam'])} frames, {len(r['diam'])/FPS:5.1f}s, "
          f"pupil found {found:3.0f}%, onset @ {r['onset_t']:.2f}s")

# %% [plot]  one curve per clip, eyeball before trusting any metric
for name, r in results.items():
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True, height_ratios=[3, 1])
    ax[0].plot(r["t"], r["diam"], lw=1)
    ax[0].axvline(r["onset_t"], color="orange", ls="--", label="light onset")
    ax[0].set_ylabel("pupil diameter (px)")
    ax[0].legend(loc="upper right")
    ax[0].set_title(f"PLR curve, {name}")
    ax[1].plot(r["t"], r["bright"], lw=1, color="gray")
    ax[1].axvline(r["onset_t"], color="orange", ls="--")
    ax[1].set_ylabel("scene\nbrightness")
    ax[1].set_xlabel("time (s)")
    plt.tight_layout()
    plt.show()

# %% [verify detection]  save annotated frames per clip and LOOK at them
out = Path("debug_frames"); out.mkdir(exist_ok=True)
for name, r in results.items():
    sub = out / Path(name).stem; sub.mkdir(exist_ok=True)
    for i, im in r["dbg"].items():
        cv2.imwrite(str(sub / f"frame_{i:05d}.png"), im)
print(f"wrote annotated frames to {out}/<clip>/  -- open them and confirm the green ellipse is on the pupil")