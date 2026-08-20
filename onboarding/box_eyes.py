"""Interactive click-drag eye boxing for onboarding (handles 1-eye AND 2-eye clips).

Run as VS Code `# %%` cells. Needs the interactive backend ipympl:  pip install ipympl
(install BEFORE going offline; the boxing itself is fully offline, like the labelme labeling).

For each un-boxed clip: set CLIP, DRAG a box around EACH visible eye - one box for a single-eye
or angled clip, two for a full-face clip - then run the SAVE cell. Boxes are written to
new_subject_rois.json (_L/_R assigned left-to-right); the rest of the onboarding pipeline
(confirm_rois -> extract_frames) then runs unchanged.

This mirrors the deployed alignment overlay: you place the box the nurse's overlay will align
to, and the 1-vs-2-eye count is whatever you draw.
"""
# %% [setup]  <-- run once
# %matplotlib widget          # <-- UNCOMMENT this line in VS Code (removes the leading '# '); enables click-drag
import json, re
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

REPO  = Path("/Users/angela/Desktop/AI_Portfolio/Pupilometry_App_Fine_Tuning")
ROIS  = REPO / "new_subject_rois.json"
CLIPS = REPO / "data" / "clips"

def unboxed():
    have = set(k.rsplit("_", 1)[0] for k in json.load(open(ROIS))) if ROIS.exists() else set()
    if (REPO / "eye_rois.json").exists():
        have |= set(json.load(open(REPO / "eye_rois.json")))
    return sorted(p.stem for p in CLIPS.glob("*")
                  if re.match(r"^subject_[a-z]+_\d+$", p.stem) and p.stem not in have)
print("un-boxed clips:", unboxed())

# %% [box ONE clip]  <-- set CLIP, run, then DRAG a box around each visible eye (re-run to start over)
CLIP = "subject_j_1"
_clip = next(p for p in CLIPS.glob("*") if p.stem == CLIP)
_cap = cv2.VideoCapture(str(_clip)); _n = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
_cap.set(cv2.CAP_PROP_POS_FRAMES, int(0.35 * _n)); _ok, _fr = _cap.read(); _cap.release()
_frame = cv2.cvtColor(_fr, cv2.COLOR_BGR2RGB)

boxes = []                                                  # reset each time this cell runs
_fig, _ax = plt.subplots(figsize=(6, 10)); _ax.imshow(_frame)
_ax.set_title(f"{CLIP}: drag a box around EACH visible eye (1 or 2), then run SAVE")
def _onselect(ec, er):
    x0, x1 = sorted([ec.xdata, er.xdata]); y0, y1 = sorted([ec.ydata, er.ydata])
    boxes.append([int(x0), int(y0), int(x1 - x0), int(y1 - y0)])
    _ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="lime", lw=2))
    _ax.text(x0, y0 - 8, str(len(boxes)), color="lime", fontsize=12)
    _fig.canvas.draw_idle(); print(f"box {len(boxes)}: {boxes[-1]}")
_rs = RectangleSelector(_ax, _onselect, useblit=True, button=[1],
                        minspanx=20, minspany=20, interactive=True)   # keep _rs referenced or it dies
plt.show()

# %% [SAVE]  <-- run after drawing; assigns _L/_R left-to-right, overwrites any old boxes for this clip
_rois = json.load(open(ROIS)) if ROIS.exists() else {}
for _k in (f"{CLIP}_L", f"{CLIP}_R"):
    _rois.pop(_k, None)                                     # clear old so re-boxing overwrites
_bs = sorted(boxes, key=lambda b: b[0] + b[2] / 2)         # order eyes left-to-right
if len(_bs) == 1:
    _rois[f"{CLIP}_L"] = _bs[0]
elif len(_bs) >= 2:
    _rois[f"{CLIP}_L"], _rois[f"{CLIP}_R"] = _bs[0], _bs[1]
if _bs:
    json.dump(_rois, open(ROIS, "w"), indent=1)
    print(f"saved for {CLIP}: {[k for k in _rois if k.startswith(CLIP)]}")
else:
    print("no boxes drawn - nothing saved (draw a box in the cell above first)")
