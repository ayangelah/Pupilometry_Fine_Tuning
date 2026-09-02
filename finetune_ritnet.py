# %% [markdown]
# Phase 2: fine-tune RITnet to fix PUPIL RECALL on dim/brown eyes (the isolated failure).
#
# SPINE (Angela writes by hand): the training loop - loss, optimizer, LR, which layers to
#   fine-tune, epochs, early stopping - and the interpretation of the before/after eval.
# DELEGABLE (this file, below): LOSO data loaders, GT->target construction, model load/freeze
#   boilerplate, checkpoint I/O, and the before/after eval wired through the labeled harness.
#
# TARGET DESIGN (agreed): supervise ONLY inside the iris disk - pupil pixels -> pupil, the rest
# of the iris -> iris, everything outside the iris -> IGNORE (no loss). This targets exactly the
# pupil-vs-iris split, leaves sclera/background untouched, and on dark iris_only frames teaches
# "disk is all iris, no pupil" (suppresses false pupils in the dark).

# %% [imports + load ritnet_segment defs (config, preprocess, model, eval helpers)]
import os, csv, math, copy
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np, cv2, torch
from torch.utils.data import Dataset, DataLoader

_src = open("ritnet_segment.py").read().split("# %% [run validation]")[0]  # defs only, no run cells
_ns = {"__name__": "__ft__"}; exec(compile(_src, "ritnet_segment.py", "exec"), _ns)
IN_W, IN_H, DEVICE = _ns["IN_W"], _ns["IN_H"], _ns["DEVICE"]
CLS_IRIS, CLS_PUPIL = _ns["CLS_IRIS"], _ns["CLS_PUPIL"]
preprocess, ritnet_input, segment, measure = _ns["preprocess"], _ns["ritnet_input"], _ns["segment"], _ns["measure"]
load_ritnet, WEIGHTS, disk_mask, passes_gate = _ns["load_ritnet"], _ns["WEIGHTS"], _ns["disk_mask"], _ns["passes_gate"]
IGNORE = 255

# %% [config]  <-- Angela tunes the training hyperparameters
SUBJECTS   = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]     # LOSO folds: each is held out as the test subject once
BATCH      = 4
EPOCHS     = 5
CKPT_DIR   = "checkpoints"; os.makedirs(CKPT_DIR, exist_ok=True)
# --- TODO(Angela), the meat -------------------------------------------------
# LR, EPOCHS, WEIGHT_DECAY, which layers to freeze, loss + class weights (pupil is a MINORITY
# class inside the disk, so plain CE will under-weight it - consider weighting/dice/focal),
# early-stopping criterion. These live in train_fold() below, left as hooks.
# ----------------------------------------------------------------------------

# %% [LOSO fold records]  <-- plumbing: build (frame, mask, category) lists per subject, cull applied
def _cull():
    try:
        return {r["frame"] for r in csv.DictReader(open("labeling/quality.csv")) if r["flag"] == "CLIP"}
    except FileNotFoundError:
        return set()

def load_records():
    """One record per usable labeled frame: dict(person, clip, frame, category, frame_path, mask_path).
    Excludes reject frames and edge-clipped (CLIP) frames. iris_only frames are KEPT (they supervise
    'no pupil in the dark')."""
    cull, recs = _cull(), []
    for r in csv.DictReader(open("labeling/gt.csv")):
        if r["category"] == "reject" or r["frame"] in cull:
            continue
        stem = r["frame"][:-4]
        recs.append(dict(person=r["person"], clip=r["clip"], frame=r["frame"], category=r["category"],
                         frame_path=f"labeling/frames/{r['clip']}/{r['frame']}",
                         mask_path=f"labeling/gt/{r['clip']}/{stem}_mask.png"))
    return recs

def fold_split(held_out, recs):
    train = [r for r in recs if r["person"] != held_out]
    test  = [r for r in recs if r["person"] == held_out]
    return train, test

def pick_val(held_out, subjects=None):
    """Validation-subject strategy for best-epoch selection (ROTATION): the subject AFTER the
    held-out test subject in SUBJECTS (wrapping). Deterministic, disjoint from the test subject,
    so selecting an epoch on it does NOT leak into the held-out score. <- Angela owns this policy."""
    subjects = subjects or SUBJECTS
    i = subjects.index(held_out)
    return subjects[(i + 1) % len(subjects)]

# %% [dataset]  <-- plumbing: frame -> RITnet input tensor; GT -> ANCHORED 4-class target
class IrisPupilDataset(Dataset):
    """Target construction (fixed): OUTSIDE the iris disk, keep the BASE model's own prediction
    (bg/sclera/outer-iris) as a pseudo-label; INSIDE the disk, use your hand-drawn GT (iris ring
    vs pupil). This anchors everything outside the disk so fine-tuning cannot collapse to
    'predict iris everywhere' (which is what ignoring the outside let it do). The base predictions
    are precomputed once at construction (the base net is fixed)."""
    def __init__(self, records, use_red=True):
        self.records, self.use_red = records, use_red
        _ns["USE_RED_CHANNEL"] = use_red
        base = load_ritnet(WEIGHTS, DEVICE).eval()                  # the ORIGINAL pretrained net
        self.base_pred = []
        with torch.no_grad():
            for r in records:                                       # one forward per frame, cached
                logits = base(preprocess(ritnet_input(cv2.imread(r["frame_path"]))).to(DEVICE))
                self.base_pred.append(logits.argmax(1)[0].cpu().numpy().astype(np.uint8))  # (IN_H,IN_W)

    def __len__(self):
        return len(self.records)

    def _target(self, gt_native, base_pred):
        """(IN_H,IN_W) long target = base prediction, with the iris disk overwritten by GT."""
        gt = cv2.resize(gt_native, (IN_W, IN_H), interpolation=cv2.INTER_NEAREST)
        t = base_pred.copy()                                       # OUTSIDE disk: keep base bg/sclera/iris
        t[gt == CLS_IRIS]  = CLS_IRIS                              # INSIDE disk: your GT iris ring
        t[gt == CLS_PUPIL] = CLS_PUPIL                             # ...and your GT pupil
        return torch.from_numpy(t).long()

    def __getitem__(self, i):
        r = self.records[i]
        _ns["USE_RED_CHANNEL"] = self.use_red                       # channel choice mirrors the pipeline
        x = preprocess(ritnet_input(cv2.imread(r["frame_path"]))).squeeze(0)   # (1,IN_H,IN_W)
        gt = cv2.imread(r["mask_path"], cv2.IMREAD_GRAYSCALE)
        return x, self._target(gt, self.base_pred[i])

# %% [model load + freeze helper]  <-- boilerplate
def load_model():
    """Fresh RITnet initialized from the pretrained weights, ready to fine-tune."""
    return load_ritnet(WEIGHTS, DEVICE)

def set_trainable(model, train_substrings=None):
    """Freeze everything, then unfreeze params whose name contains any given substring. Pass
    train_substrings=None to train ALL params. Inspect model.named_parameters() to pick names."""
    if train_substrings is None:
        for p in model.parameters():
            p.requires_grad = True
        return model
    for name, p in model.named_parameters():
        p.requires_grad = any(s in name for s in train_substrings)
    return model

def freeze_bn(model):
    """Put BatchNorm layers in eval mode so their running mean/var stay FROZEN during training
    (requires_grad=False does NOT stop BN stats updating in train() mode - they drift the whole
    net toward the fine-tune data). Gradients still flow to trainable params; only BN stats are
    held to their pretrained values. Re-apply at the START of each epoch, because eval_subject()
    flips the model to eval() and back to train(), which re-enables BN's stat updates."""
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
    return model

def save_ckpt(model, held_out, tag="best"):
    path = os.path.join(CKPT_DIR, f"ritnet_ft_holdout_{held_out}_{tag}.pkl")
    torch.save(model.state_dict(), path); return path

# %% [before/after eval]  <-- plumbing: labeled accuracy on ONE subject, with a given model
def eval_subject(model, held_out, recs, use_red=True):
    """Run `model` over held-out subject's frames -> pupil/iris detection + IoU + diameter error.
    Uses the SAME recipe as eval_labeled: iris over all non-reject, pupil over 'full' frames only."""
    _ns["model"], _ns["USE_RED_CHANNEL"] = model, use_red          # segment() reads these module globals
    prev = model.training; model.eval()
    test = [r for r in recs if r["person"] == held_out]
    def _iou(a, b):
        if a is None or b is None: return None
        u = np.logical_or(a, b).sum(); return float(np.logical_and(a, b).sum()) / u if u else 0.0
    P = dict(iris_n=0, iris_det=0, iris_iou=[], pup_n=0, pup_det=0, pup_iou=[], pup_de=[])
    gt = {r["frame"]: r for r in csv.DictReader(open("labeling/gt.csv"))}
    for r in test:
        g = gt[r["frame"]]; im = cv2.imread(r["frame_path"]); h, w = im.shape[:2]
        bgr = cv2.resize(im, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
        with torch.no_grad():
            m = measure(segment(ritnet_input(bgr)), bgr)
        s = 0.5 * (IN_W / w + IN_H / h)
        if g["iris_cx"] != "":                                        # skip frames with no GT iris (e.g. pupil drawn, iris missed)
            P["iris_n"] += 1
            if m["iris_circle"] is not None:
                P["iris_det"] += 1
                gd = np.zeros((h, w), np.uint8); cv2.circle(gd, (round(float(g["iris_cx"])), round(float(g["iris_cy"]))), round(float(g["iris_r"])), 1, -1)
                P["iris_iou"].append(_iou(cv2.resize(gd, (IN_W, IN_H), interpolation=cv2.INTER_NEAREST).astype(bool), disk_mask(m["iris_circle"], (IN_H, IN_W))))
        if g["category"] == "full":
            P["pup_n"] += 1
            gated = m["iris_circle"] is not None and passes_gate(m["iris_circle"][:2], m["iris_circle"][2], m["pupil_circle"])
            if gated:
                P["pup_det"] += 1
                gp = np.zeros((h, w), np.uint8); cv2.circle(gp, (round(float(g["pupil_cx"])), round(float(g["pupil_cy"]))), round(float(g["pupil_r"])), 1, -1)
                P["pup_iou"].append(_iou(cv2.resize(gp, (IN_W, IN_H), interpolation=cv2.INTER_NEAREST).astype(bool), disk_mask(m["pupil_circle"], (IN_H, IN_W))))
                P["pup_de"].append(abs(m["pupil_px"] - 2 * float(g["pupil_r"]) * s) / (2 * float(g["pupil_r"]) * s))
    model.train(prev)
    f = lambda l: round(float(np.mean(l)), 3) if l else None
    return dict(iris_det=P["iris_det"] / max(P["iris_n"], 1), iris_iou=f(P["iris_iou"]),
                pup_det=P["pup_det"] / max(P["pup_n"], 1), pup_iou=f(P["pup_iou"]), pup_de=f(P["pup_de"]))

# %% [train one fold]
def train_fold(held_out, recs, val_subj=None):
    exclude = {held_out} | ({val_subj} if val_subj else set())        # val subj is disjoint from train+test
    train_recs = [r for r in recs if r["person"] not in exclude and r["category"] == "full"]  # pupil-visible only
    loader = DataLoader(IrisPupilDataset(train_recs), batch_size=BATCH, shuffle=True)
    model = load_model()

    # ---- TODO: the training loop (the meat) --------------------------
    # set_trainable(model, train_substrings=[...])          # choose what to fine-tune
    # criterion = ... (CrossEntropyLoss(ignore_index=IGNORE, weight=...); pupil is minority)
    # optimizer = ...                                        # LR / weight decay
    # for epoch in range(EPOCHS):
    #     for x, y in loader:
    #         x, y = x.to(DEVICE), y.to(DEVICE)
    #         logits = model(x)                              # (B, 4, IN_H, IN_W)
    #         loss = criterion(logits, y); loss.backward(); optimizer.step(); optimizer.zero_grad()
    #     # track eval_subject(model, held_out, recs) on the held-out set; checkpoint the best
    # return model
    set_trainable(model, train_substrings=["out_conv", "up_block4"])          # choose what to fine-tune
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE, weight=torch.tensor([1, 1, 1, 6], dtype=torch.float).to(DEVICE)) # bg, sclera, iris, pupil
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=1e-4
    )                                        # LR / weight decay
    best = {"score": -1.0, "epoch": -1, "state": None}     # best-epoch snapshot (plumbing)
    for epoch in range(EPOCHS): # each epoch
        model.train()
        freeze_bn(model)
        total_loss = 0
        for x, y in loader: # each batch
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)                              # (B, 4, IN_H, IN_W)
            loss = criterion(logits, y); loss.backward(); optimizer.step(); optimizer.zero_grad()
            total_loss += loss.item()
        # validate on the VAL subject (disjoint from held_out -> no leak into the reported score)
        v = eval_subject(model, val_subj, recs) if val_subj else None
        print(f"epoch {epoch}: avg loss {total_loss/len(loader):.4f}" + (f"  val[{val_subj}] {v}" if v else ""))
        if v is not None:
            score = v["pup_det"]                           # <-- Angela: SELECTION CRITERION (what to maximize)
            if score > best["score"]:
                best = {"score": score, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
    if best["state"] is not None:                          # <-- Angela: restore best-on-val epoch (vs keep last)
        model.load_state_dict(best["state"])
        print(f"  -> restored best epoch {best['epoch']} (val pup_det {best['score']:.3f})")
    return model

# Best-epoch selection (val_subj=pick_val(...)) is PARKED: at n=11 a single val subject is too
# noisy a selector - it matched fixed-5-epoch where it picked the last epoch and hurt fold D where
# it picked ep0 (mean Δ +0.020 vs +0.023 fixed). Pass val_subj to re-enable once a POOLED multi-
# subject val set makes the signal less noisy. Default (None) reproduces the best config.

# %% [run one fold: before vs after]  <-- Angela drives; prints the delta the fine-tune bought
HELD_OUT = "A"
recs = load_records()
base = load_model()
print("BEFORE:", eval_subject(base, HELD_OUT, recs))
ft = train_fold(HELD_OUT, recs)               # <- your loop  (add val_subj=pick_val(HELD_OUT) for best-epoch)
print("AFTER :", eval_subject(ft, HELD_OUT, recs))
save_ckpt(ft, HELD_OUT)

# %% [run LOSO: hold out every subject once]  <-- the full sweep (a few min per fold)
recs = load_records()
loso = {}
for held_out in SUBJECTS:
    print(f"\n===== hold out {held_out} =====", flush=True)
    before = eval_subject(load_model(), held_out, recs)          # base model on the unseen subject
    ft = train_fold(held_out, recs)                              # fine-tune on the other 10 (val_subj=pick_val(held_out) to select best-epoch)
    after = eval_subject(ft, held_out, recs)                     # fine-tuned model on the unseen subject
    save_ckpt(ft, held_out)
    loso[held_out] = (before, after)
    print(f"[{held_out}] BEFORE {before}  ->  AFTER {after}")

print("\n===== LOSO summary (before -> after per held-out subject) =====")
for s, (b, a) in loso.items():
    print(f"{s}:  before {b}\n     after  {a}")

# %%
