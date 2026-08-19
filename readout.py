"""Render the clinician-style PLR readout for one clip (Stage 1e).

    python readout.py [clip_stem] [out.html]     # default: subject_b_2 -> demo_readout.html

Self-contained: builds the curve from the clip, calibrates to mm, computes the PLR metrics,
runs the validity gate, and writes a self-contained HTML report (metrics tiles, annotated
pupil-vs-time SVG, detector-overlay evidence frames). The capture-status banner reflects the
gate verdicts (insufficient capture / dead baseline / no clear constriction / OK). The clip
needs an ROI (eye_rois.json) and an onset mark (light_onset.json). Output embeds eye frames,
so it is gitignored by default.
"""
import os, sys, base64, html
REPO = "/Users/angela/Desktop/AI_Portfolio/Pupilometry_App_Fine_Tuning"
os.chdir(REPO); sys.path.insert(0, REPO)
from pathlib import Path
import numpy as np, cv2
from plr_metrics import calibrate_mm, compute_metrics, assess_validity, IRIS_MM

CLIP = sys.argv[1] if len(sys.argv) > 1 else "subject_b_2"
OUT  = sys.argv[2] if len(sys.argv) > 2 else f"{REPO}/demo_readout.html"
FPS  = 120.0

# self-contained: build the curve straight from the clip (one sequential pass), no cache files
from ritnet_segment import build_curve, _load_rois, get_roi, framed_roi, CLIPS_DIR
_cl = next(p for p in Path(CLIPS_DIR).glob("*") if p.stem == CLIP)
_c0 = cv2.VideoCapture(str(_cl)); _c0.set(cv2.CAP_PROP_POS_FRAMES, 0); _ok, _f0 = _c0.read(); _c0.release()
curve = build_curve(_cl, framed_roi(get_roi(CLIP, _f0, _load_rois()), _f0.shape),
                    seed_box=get_roi(CLIP, _f0, _load_rois()))
if curve["onset_t"] is None:
    sys.exit(f"{CLIP}: no onset mark in light_onset.json - add one before rendering a readout.")
t, valid, onset_t = curve["t"], curve["valid"], float(curve["onset_t"])
pupil_mm = calibrate_mm(curve)
m = compute_metrics(curve, pupil_mm)
V = assess_validity(curve)

post = t >= onset_t
tp, pp = t[post], pupil_mm[post]
imin = int(np.nanargmin(pp)); t_min, min_mm = tp[imin], pp[imin]
baseline_mm = m.baseline_mm

# ---- capture-status (policy B: compute, then flag) ----
if not V["clip_ok"]:
    status = ("reject", "Insufficient capture",
              f"Only {V['valid_frac']*100:.0f}% of frames passed quality checks; the values below are not reliable.")
elif not V["baseline_ok"]:
    status = ("warn", "Baseline not captured",
              "No usable frames in the pre-constriction window; resting size, % constriction, and latency are unavailable for this clip.")
elif V.get("constriction_ok") is False:
    status = ("warn", "No clear constriction",
              f"After onset the pupil rose {V['post_onset_dilation']*100:.0f}% above baseline (a real reflex "
              f"constricts, not dilates); likely an onset-timing error, weak reflex, or gaze movement - the velocity is not reliable.")
else:
    status = ("ok", "Capture OK", f"Quality checks passed ({V['valid_frac']*100:.0f}% of frames usable).")

# ---- overlay evidence frames (tracked ROI + locked iris ruler) ----
def nearest_valid_time(target):
    cand = tp[np.isfinite(pp)]
    return float(cand[np.argmin(np.abs(cand - target))]) if cand.size else float(target)

frame_specs = [(nearest_valid_time(onset_t + 0.10), "Resting"),
               (nearest_valid_time(float(tp[np.nanargmin(np.abs(pp - (np.nanmax(pp) + min_mm) / 2))])), "Constricting"),
               (float(t_min), "Constricted")]
overlays = []
try:
    from ritnet_segment import (crop, framed_roi, get_roi, _load_rois, segment,
                                ritnet_input, measure, EyeTracker, CLIPS_DIR, IN_W, IN_H)
    clip_path = next(p for p in Path(CLIPS_DIR).glob("*") if p.stem == CLIP)
    rois = _load_rois()
    cap = cv2.VideoCapture(str(clip_path)); cap.set(cv2.CAP_PROP_POS_FRAMES, 0); _, f0 = cap.read()
    seed = get_roi(CLIP, f0, rois)
    targets = {int(round(tt * FPS)): lbl for tt, lbl in frame_specs}
    tracked, tracker, fi = {}, None, -1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok, fr = cap.read()
        if not ok: break
        fi += 1
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        tight = seed if tracker is None else tracker.update(g)
        if tracker is None: tracker = EyeTracker(g, seed)
        if fi in targets: tracked[fi] = framed_roi(tuple(tight), fr.shape)
        if fi >= max(targets): break

    def measure_frame(fi):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi); _, frame = cap.read()
        fr = crop(frame, tracked[fi]); cb = cv2.resize(fr, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
        return cb, measure(segment(ritnet_input(fr)), cb)

    fi_low = int(round(t_min * FPS)); _, m_low = measure_frame(fi_low)
    locked_r = (m_low["pupil_px"] * IRIS_MM / float(pupil_mm[np.argmin(np.abs(t - t_min))])) / 2
    for tt, label in frame_specs:
        cb, mm = measure_frame(int(round(tt * FPS)))
        img = cb.copy(); p = mm["pupil_circle"]
        if p:
            pcx, pcy, pr = int(p[0]), int(p[1]), int(p[2])
            cv2.circle(img, (pcx, pcy), int(round(locked_r)), (255, 0, 0), 2)
            cv2.circle(img, (pcx, pcy), pr, (0, 200, 0), 2)
        img = cv2.resize(img, (IN_W // 2, IN_H // 2), interpolation=cv2.INTER_AREA)
        b = base64.b64encode(cv2.imencode(".png", img)[1]).decode()
        mm_here = float(pupil_mm[np.argmin(np.abs(t - tt))])
        overlays.append((label, f"{mm_here:.1f} mm", tt - onset_t, b))
    cap.release()
    print(f"rendered {len(overlays)} overlays")
except Exception as e:
    print("overlay generation skipped:", e)

# ---- annotated PLR curve (inline SVG), y-domain adaptive per clip ----
W, H, L, R, T, B = 820, 300, 52, 18, 18, 36
x0d, x1d = onset_t, float(t[-1])
fin = pp[np.isfinite(pp)]
y0d = max(0.0, float(np.floor(fin.min() - 0.5))); y1d = float(np.ceil(fin.max() + 0.5))
def X(v): return L + (v - x0d) / (x1d - x0d) * (W - L - R)
def Y(v): return T + (y1d - v) / (y1d - y0d) * (H - T - B)
segs, cur = [], []
for tv, pv in zip(tp, pp):
    if np.isfinite(pv): cur.append(f"{X(tv):.1f},{Y(pv):.1f}")
    elif cur: segs.append(cur); cur = []
if cur: segs.append(cur)
paths = "".join(f'<polyline points="{" ".join(s)}" fill="none" stroke="var(--accent)" '
                f'stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>' for s in segs)
yr = range(int(y0d), int(y1d) + 1)
yticks = "".join(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{W-R}" y2="{Y(v):.1f}" stroke="var(--grid)" stroke-width="1"/>'
                 f'<text x="{L-8}" y="{Y(v)+3:.1f}" text-anchor="end" class="ax">{v}</text>' for v in yr)
xticks = "".join(f'<text x="{X(v):.1f}" y="{H-B+18}" text-anchor="middle" class="ax">{v}</text>'
                 for v in range(int(np.ceil(x0d)), int(np.floor(x1d)) + 1))
constr = (f'<rect x="{X(onset_t):.1f}" y="{T}" width="{max(0,X(t_min)-X(onset_t)):.1f}" '
          f'height="{H-T-B}" fill="var(--accent)" opacity="0.06"/>')
onset_line = (f'<line x1="{X(onset_t):.1f}" y1="{T}" x2="{X(onset_t):.1f}" y2="{H-B}" stroke="var(--accent)" '
              f'stroke-width="1.5" stroke-dasharray="1 3"/><text x="{X(onset_t)+5:.1f}" y="{T+12}" class="anno">light on</text>')
base_line = ""
if np.isfinite(baseline_mm):
    bY = Y(baseline_mm)
    base_line = (f'<line x1="{L}" y1="{bY:.1f}" x2="{W-R}" y2="{bY:.1f}" stroke="var(--muted)" stroke-width="1" '
                 f'stroke-dasharray="4 3"/><text x="{W-R:.1f}" y="{bY-6:.1f}" text-anchor="end" class="anno">resting {baseline_mm:.1f} mm</text>')
min_dot = (f'<circle cx="{X(t_min):.1f}" cy="{Y(min_mm):.1f}" r="4" fill="var(--accent)"/>'
           f'<text x="{X(t_min)+8:.1f}" y="{Y(min_mm)+4:.1f}" class="anno">min {min_mm:.1f} mm</text>')
svg = (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Pupil diameter versus time" '
       f'preserveAspectRatio="xMidYMid meet" class="chart">{yticks}{constr}{base_line}{onset_line}{paths}{min_dot}{xticks}'
       f'<text x="{L-38}" y="{T+(H-T-B)/2}" class="axtitle" transform="rotate(-90 {L-38} {T+(H-T-B)/2})">diameter (mm)</text>'
       f'<text x="{(L+W-R)/2:.0f}" y="{H-2}" text-anchor="middle" class="axtitle">time after light onset (s)</text></svg>')

# ---- tiles (baseline-dependent ones gray out when baseline_ok is false; all gray on reject) ----
def fmt(x, dp=1): return "—" if (x is None or not np.isfinite(x)) else f"{x:.{dp}f}"
tiles = [
    ("Resting diameter", fmt(baseline_mm), "mm", "typical 4.4", "dark-adapted", "neutral", True),
    ("Minimum diameter", fmt(min_mm), "mm", "typical 2.9", "in range", "good", False),
    ("Constriction", fmt(m.pct_constriction, 0), "%", "typical 35", "brisk", "good", True),
    ("Latency", fmt(m.latency_s * 1000, 0), "ms", "typical 230", "prompt", "good", True),
    ("Peak constriction vel.", fmt(m.cv_peak_mm_s), "mm/s", "typical 4.2", "brisk", "good", False),
    ("Mean constriction vel.", fmt(m.cv_mean_mm_s), "mm/s", "typical 2.7", "in range", "good", False),
]
tile_html = ""
for k, val, u, ref, chip, tone, bdep in tiles:
    dead = bdep and not V["baseline_ok"]
    reject = not V["clip_ok"]
    if dead:
        cls, disp, chip_html = "tile muted", "—", '<span class="chip muted">not captured</span>'
    elif reject:
        cls, disp, chip_html = "tile muted", val, '<span class="chip muted">unreliable</span>'
    else:
        cls, disp, chip_html = "tile", val, f'<span class="chip {tone}">{chip}</span>'
    tile_html += (f'<div class="{cls}"><div class="tk">{html.escape(k)}</div>'
                  f'<div class="tv"><span class="num">{disp}</span><span class="tu">{u}</span></div>'
                  f'<div class="tmeta">{chip_html}<span class="norm">{ref}</span></div></div>')

ev_html = "".join(f'<figure class="ev"><img src="data:image/png;base64,{b}" alt="Detector overlay, {html.escape(lbl)}"/>'
                  f'<figcaption><span class="evl">{html.escape(lbl)}</span><span class="evm">{html.escape(mm)}</span>'
                  f'<span class="evt">+{dt:.2f}s</span></figcaption></figure>' for lbl, mm, dt, b in overlays)
ev_section = (f'<section class="evidence"><div class="eyebrow">Detector evidence</div>'
              f'<h2>The measurement, drawn on the eye</h2>'
              f'<p class="lede">Blue is the iris ruler the pipeline calibrates against, a fixed size across the '
              f'whole reflex. Green is the pupil measured in each frame. The pupil tightens as the reflex fires; '
              f'the ruler holds constant.</p><div class="evrow">{ev_html}</div></section>') if overlays else ""

sc, stitle, smsg = status
page = f'''<title>PLR Readout — {CLIP}</title>
<style>
:root {{
  --paper:#F7F9FA; --card:#FFFFFF; --ink:#14232B; --muted:#5A6B72; --faint:#8496A0;
  --hair:#E2E8EB; --grid:#EBF0F2; --accent:#0E7C86; --accent-ink:#0B5A62;
  --good:#1F8F63; --good-bg:#E4F1EA; --neutral:#6C7A80; --neutral-bg:#EAEFF1;
  --warn:#B4722A; --warn-bg:#F6ECDE; --reject:#B23A3A; --reject-bg:#F7E7E5;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
}}
@media (prefers-color-scheme: dark) {{ :root {{
  --paper:#0E1619; --card:#141F23; --ink:#E6EDEF; --muted:#93A4AB; --faint:#6E828A;
  --hair:#25343A; --grid:#1E2C31; --accent:#39B0BA; --accent-ink:#7FD4DB;
  --good:#57C08C; --good-bg:#12312A; --neutral:#9FB0B6; --neutral-bg:#1C2A2F;
  --warn:#D69A5B; --warn-bg:#2E2517; --reject:#E08078; --reject-bg:#2E1A18; }} }}
:root[data-theme="light"] {{
  --paper:#F7F9FA; --card:#FFFFFF; --ink:#14232B; --muted:#5A6B72; --faint:#8496A0;
  --hair:#E2E8EB; --grid:#EBF0F2; --accent:#0E7C86; --accent-ink:#0B5A62;
  --good:#1F8F63; --good-bg:#E4F1EA; --neutral:#6C7A80; --neutral-bg:#EAEFF1;
  --warn:#B4722A; --warn-bg:#F6ECDE; --reject:#B23A3A; --reject-bg:#F7E7E5; }}
:root[data-theme="dark"] {{
  --paper:#0E1619; --card:#141F23; --ink:#E6EDEF; --muted:#93A4AB; --faint:#6E828A;
  --hair:#25343A; --grid:#1E2C31; --accent:#39B0BA; --accent-ink:#7FD4DB;
  --good:#57C08C; --good-bg:#12312A; --neutral:#9FB0B6; --neutral-bg:#1C2A2F;
  --warn:#D69A5B; --warn-bg:#2E2517; --reject:#E08078; --reject-bg:#2E1A18; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--paper); color:var(--ink); font-family:var(--sans); line-height:1.5; -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:900px; margin:0 auto; padding:32px 20px 56px; }}
.card {{ background:var(--card); border:1px solid var(--hair); border-radius:14px; padding:34px 34px 30px; box-shadow:0 1px 2px rgba(20,35,43,.04),0 8px 30px rgba(20,35,43,.05); }}
.eyebrow {{ font-size:11px; letter-spacing:.14em; text-transform:uppercase; color:var(--accent); font-weight:650; }}
h1 {{ font-size:26px; margin:6px 0 4px; letter-spacing:-.01em; text-wrap:balance; }}
.sub {{ color:var(--muted); font-size:14px; display:flex; gap:14px; flex-wrap:wrap; }}
.sub .id {{ font-family:var(--mono); color:var(--ink); }}
.banner {{ display:flex; gap:10px; align-items:baseline; margin:20px 0 12px; padding:11px 14px; border:1px solid var(--hair); border-left:3px solid var(--warn); border-radius:8px; background:var(--warn-bg); }}
.banner b {{ font-size:12px; letter-spacing:.06em; text-transform:uppercase; color:var(--warn); }}
.banner span {{ font-size:13px; color:var(--muted); }}
.status {{ display:flex; gap:10px; align-items:baseline; margin:0 0 26px; padding:11px 14px; border-radius:8px; border:1px solid var(--hair); border-left:3px solid var(--good); background:var(--good-bg); }}
.status.warn {{ border-left-color:var(--warn); background:var(--warn-bg); }}
.status.reject {{ border-left-color:var(--reject); background:var(--reject-bg); }}
.status b {{ font-size:12px; letter-spacing:.06em; text-transform:uppercase; color:var(--good); }}
.status.warn b {{ color:var(--warn); }} .status.reject b {{ color:var(--reject); }}
.status span {{ font-size:13px; color:var(--muted); }}
.tiles {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:30px; }}
.tile {{ border:1px solid var(--hair); border-radius:10px; padding:14px 14px 12px; display:flex; flex-direction:column; gap:9px; }}
.tile.muted {{ opacity:.5; }}
.tk {{ font-size:12px; color:var(--muted); }}
.tv {{ display:flex; align-items:baseline; gap:4px; }}
.num {{ font-family:var(--mono); font-size:30px; font-weight:600; letter-spacing:-.02em; font-variant-numeric:tabular-nums; }}
.tu {{ font-size:13px; color:var(--faint); }}
.tmeta {{ display:flex; align-items:center; gap:8px; }}
.chip {{ font-size:11px; padding:2px 8px; border-radius:99px; font-weight:600; }}
.chip.good {{ color:var(--good); background:var(--good-bg); }}
.chip.neutral {{ color:var(--neutral); background:var(--neutral-bg); }}
.chip.muted {{ color:var(--faint); background:var(--neutral-bg); }}
.norm {{ font-size:11.5px; color:var(--faint); font-family:var(--mono); }}
section {{ margin-top:30px; }}
h2 {{ font-size:16px; margin:2px 0 3px; letter-spacing:-.005em; }}
.lede {{ color:var(--muted); font-size:13.5px; margin:0 0 16px; max-width:62ch; }}
.chartwrap {{ overflow-x:auto; }}
.chart {{ width:100%; height:auto; display:block; }}
.ax {{ font-family:var(--mono); font-size:10px; fill:var(--faint); }}
.axtitle {{ font-size:11px; fill:var(--muted); }}
.anno {{ font-size:10.5px; fill:var(--muted); }}
.evrow {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }}
.ev {{ margin:0; border:1px solid var(--hair); border-radius:10px; overflow:hidden; background:var(--paper); }}
.ev img {{ width:100%; display:block; }}
.ev figcaption {{ display:flex; align-items:baseline; gap:8px; padding:9px 11px; }}
.evl {{ font-size:12.5px; font-weight:600; }}
.evm {{ font-family:var(--mono); font-size:12.5px; color:var(--accent-ink); font-variant-numeric:tabular-nums; }}
.evt {{ margin-left:auto; font-family:var(--mono); font-size:11px; color:var(--faint); }}
.method {{ margin-top:30px; padding-top:20px; border-top:1px solid var(--hair); display:grid; grid-template-columns:1fr 1fr; gap:20px 30px; }}
.method h3 {{ font-size:11px; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); margin:0 0 7px; }}
.method p {{ margin:0; font-size:12.5px; color:var(--muted); }}
.method .mono {{ font-family:var(--mono); color:var(--ink); }}
@media (max-width:680px) {{ .tiles {{ grid-template-columns:repeat(2,1fr); }} .method {{ grid-template-columns:1fr; }} .card {{ padding:24px 20px; }} }}
</style>
<div class="wrap"><div class="card">
  <div class="eyebrow">Pupillary Light Reflex</div>
  <h1>Pupillometry readout</h1>
  <div class="sub"><span class="id">{CLIP}</span><span>single eye · phone RGB · 120 fps</span><span>iris-grounded calibration</span></div>
  <div class="banner"><b>Triage aid</b><span>Screening measurement, not an FDA-cleared diagnostic. No equivalence to cleared pupillometers is claimed.</span></div>
  <div class="status {sc}"><b>{stitle}</b><span>{smsg}</span></div>
  <div class="tiles">{tile_html}</div>
  <section>
    <div class="eyebrow">Response curve</div>
    <h2>Pupil diameter through the reflex</h2>
    <p class="lede">Resting diameter is read from the first illuminated, pre-constriction frames (the reflex-latency window), then the pupil constricts to its minimum.</p>
    <div class="chartwrap">{svg}</div>
  </section>
  {ev_section}
  <div class="method">
    <div><h3>Calibration</h3><p>Diameter in millimeters via the iris as a per-frame ruler: <span class="mono">pupil_mm = (pupil_px / iris_px) &times; 11.7</span>. The ratio cancels distance, zoom, and angle. Percent constriction is independent of the 11.7&nbsp;mm assumption.</p></div>
    <div><h3>Known limits</h3><p>The 11.7&nbsp;mm iris constant and off-axis foreshortening are stated error sources. Recall degrades on dark irises in dim light (visible-light vs infrared).</p></div>
  </div>
</div></div>'''
Path(OUT).write_text(page)
print(f"wrote {OUT}  status={sc}  valid={V['valid_frac']*100:.0f}%  baseline_frames={V['n_baseline']}")
