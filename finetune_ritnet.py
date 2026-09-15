# %% [markdown]
# Phase 2: fine-tune RITnet to fix PUPIL RECALL on dim/brown eyes (the isolated failure).
#
# TARGET DESIGN: supervise ONLY inside the iris disk - pupil pixels -> pupil, the rest
# of the iris -> iris, everything outside the iris -> IGNORE (no loss). This targets exactly the
# pupil-vs-iris split and leaves sclera/background untouched. ``iris_only`` means a human could
# not resolve the pupil boundary: those frames supervise a visibility head, never a fabricated
# "all iris / no pupil" segmentation target.

# %% [imports + load ritnet_segment defs (config, preprocess, model, eval helpers)]
import argparse, os, csv, math, copy
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np, cv2, torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

_src = open("ritnet_segment.py").read().split("# %% [run validation]")[0]  # defs only, no run cells
_ns = {"__name__": "__ft__"}; exec(compile(_src, "ritnet_segment.py", "exec"), _ns)
IN_W, IN_H, DEVICE = _ns["IN_W"], _ns["IN_H"], _ns["DEVICE"]
CLS_IRIS, CLS_PUPIL = _ns["CLS_IRIS"], _ns["CLS_PUPIL"]
preprocess, ritnet_input, segment, measure = _ns["preprocess"], _ns["ritnet_input"], _ns["segment"], _ns["measure"]
load_ritnet, WEIGHTS, disk_mask, passes_gate = _ns["load_ritnet"], _ns["WEIGHTS"], _ns["disk_mask"], _ns["passes_gate"]
IGNORE = 255

# %% [config]  <-- training hyperparameters
SUBJECTS   = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]     # LOSO folds: each is held out as the test subject once
BATCH      = 4
EPOCHS     = 5
VISIBILITY_LOSS_WEIGHT = 0.5
SEED       = 17
CKPT_DIR   = "checkpoints"; os.makedirs(CKPT_DIR, exist_ok=True)
# LR, weight decay, trainable layers, and class weights are set in train_fold() below. Pupil is a
# MINORITY class inside the disk, so plain CE would under-weight it; it gets extra loss weight.

# %% [LOSO fold records]  <-- build (frame, mask, category) lists per subject
def _edge_clipped(frac=0.01):
    """Frame filenames whose eyebox touches a frame edge, within `frac` of the frame size.
    The eye is cut off there, so the iris arc is incomplete and NO circle can be fitted to it -
    which makes iris_iou / pup_de on those frames score noise regardless of the prediction.
    They are still valid TRAINING data (the labels are right for the visible sliver), so only
    eval_subject drops them.

    PER-FRAME, not per-sequence: each frame in frames_v2 was cropped with the box drawn on THAT
    frame (manifest_v2's 'box' column), and a handheld clip drifts, so one frame can hug the edge
    while the rest of its sequence is clear. Box and frame size both come from manifest_v2.csv,
    so this costs no video I/O. Fraction, not pixels, so 4K and 720p are judged the same way."""
    out = set()
    try:
        rows = list(csv.DictReader(open("labeling/manifest_v2.csv")))
    except FileNotFoundError:
        return out
    for r in rows:
        x, y, w, h = map(int, r["box"].split("_"))
        H, W = int(r["orig_h"]), int(r["orig_w"])
        mx, my = frac * W, frac * H
        if x <= mx or y <= my or x + w >= W - mx or y + h >= H - my:
            out.add(os.path.basename(r["path"]))
    return out


def load_records():
    """One record per usable labeled frame: dict(person, seq_id, frame, category, frame_path,
    mask_path, edge_clip). Excludes reject frames. iris_only frames are KEPT to supervise
    pupil-boundary visibility. edge_clip marks frames whose eyebox is cut off by the frame
    boundary: kept for training, dropped by eval_subject."""
    edge, recs = _edge_clipped(), []
    for r in csv.DictReader(open("labeling/gt.csv")):
        if r["category"] == "reject":
            continue
        stem = r["frame"][:-4]
        recs.append(dict(person=r["person"], seq_id=r["seq_id"], frame=r["frame"], category=r["category"],
                         frame_path=f"labeling/frames_v2/{r['seq_id']}/{r['frame']}",
                         mask_path=f"labeling/gt/{r['seq_id']}/{stem}_mask.png",
                         edge_clip=r["frame"] in edge))
    n_edge = sum(r["edge_clip"] for r in recs)
    print(f"{len(recs)} labeled frames | {n_edge} edge-clipped: "
          f"trained on, excluded from eval", flush=True)
    return recs

def fold_split(held_out, recs):
    train = [r for r in recs if r["person"] != held_out]
    test  = [r for r in recs if r["person"] == held_out]
    return train, test

def pick_val(held_out, subjects=None):
    """Validation-subject strategy for best-epoch selection (ROTATION): the subject AFTER the
    held-out test subject in SUBJECTS (wrapping). Deterministic, disjoint from the test subject,
    so selecting an epoch on it does NOT leak into the held-out score."""
    subjects = subjects or SUBJECTS
    i = subjects.index(held_out)
    return subjects[(i + 1) % len(subjects)]

# %% [dataset]  <-- frame -> RITnet input tensor; GT -> ANCHORED 4-class target
_BASE_PRED_CACHE = {}


class IrisPupilDataset(Dataset):
    """Target construction (fixed): OUTSIDE the iris disk, keep the BASE model's own prediction
    (bg/sclera/outer-iris) as a pseudo-label; INSIDE the disk, use the hand-drawn GT (iris ring
    vs pupil). This anchors everything outside the disk so fine-tuning cannot collapse to
    'predict iris everywhere' (which is what ignoring the outside let it do). The base predictions
    are precomputed once at construction (the base net is fixed)."""
    def __init__(self, records, use_red=True):
        self.records, self.use_red = records, use_red
        _ns["USE_RED_CHANNEL"] = use_red
        base = load_ritnet(WEIGHTS, DEVICE).eval()                  # the ORIGINAL pretrained net
        self.base_pred = []
        with torch.no_grad():
            for r in records:
                if r["category"] != "full":
                    # Segmentation loss is masked for human-unresolvable frames; no pseudo-label
                    # forward is needed just to manufacture a target that will not be used.
                    self.base_pred.append(np.zeros((IN_H, IN_W), np.uint8))
                    continue
                key = (r["frame_path"], use_red)
                if key not in _BASE_PRED_CACHE:
                    logits = base(preprocess(ritnet_input(cv2.imread(r["frame_path"]))).to(DEVICE))
                    _BASE_PRED_CACHE[key] = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
                self.base_pred.append(_BASE_PRED_CACHE[key])

    def __len__(self):
        return len(self.records)

    def _target(self, gt_native, base_pred):
        """(IN_H,IN_W) long target = base prediction, with the iris disk overwritten by GT."""
        gt = cv2.resize(gt_native, (IN_W, IN_H), interpolation=cv2.INTER_NEAREST)
        t = base_pred.copy()                                       # OUTSIDE disk: keep base bg/sclera/iris
        t[gt == CLS_IRIS]  = CLS_IRIS                              # INSIDE disk: GT iris ring
        t[gt == CLS_PUPIL] = CLS_PUPIL                             # ...and GT pupil
        return torch.from_numpy(t).long()

    def __getitem__(self, i):
        r = self.records[i]
        _ns["USE_RED_CHANNEL"] = self.use_red                       # channel choice mirrors the pipeline
        x = preprocess(ritnet_input(cv2.imread(r["frame_path"]))).squeeze(0)   # (1,IN_H,IN_W)
        gt = cv2.imread(r["mask_path"], cv2.IMREAD_GRAYSCALE)
        visible = float(r["category"] == "full")
        segmentation_supervised = visible  # no invented pupil mask when a human cannot see it
        return x, self._target(gt, self.base_pred[i]), visible, segmentation_supervised

# %% [model load + freeze helper]  <-- boilerplate
class RITnetWithVisibility(torch.nn.Module):
    """RITnet segmentation plus a tiny image-level pupil-boundary visibility head.

    The head pools the bottleneck feature map (32 channels), adding only 33 parameters. Its
    output is confidence that a human-resolvable pupil boundary is present in this frame.
    ``forward`` remains segmentation-only so the existing inference helpers still work.
    """
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.visibility_head = torch.nn.Linear(32, 1)

    def forward_with_visibility(self, x):
        logits = self.base(x)
        pooled = torch.nn.functional.adaptive_avg_pool2d(self.base.x5, 1).flatten(1)
        return logits, self.visibility_head(pooled).squeeze(1)

    def forward(self, x):
        return self.base(x)


def load_model():
    """Fresh RITnet initialized from the pretrained weights, ready to fine-tune."""
    return RITnetWithVisibility(load_ritnet(WEIGHTS, DEVICE)).to(DEVICE)

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
    torch.save(dict(segmentation_state=model.base.state_dict(),
                    visibility_state=model.visibility_head.state_dict(),
                    held_out=held_out, visibility_threshold=0.5), path)
    return path


def balanced_sampler(records, seed=SEED):
    """Give each subject × visibility group equal expected weight per epoch."""
    from collections import Counter
    counts = Counter((r["person"], r["category"]) for r in records)
    weights = torch.tensor([1.0 / counts[(r["person"], r["category"])] for r in records],
                           dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, len(records), replacement=True, generator=generator)

# %% [before/after eval]  <-- labeled accuracy on ONE subject, with a given model
def eval_subject(model, held_out, recs, use_red=True, eval_visibility=False):
    """Run `model` over held-out subject's frames -> pupil/iris detection + IoU + diameter error.
    Uses the SAME recipe as eval_labeled: iris over all non-reject, pupil over 'full' frames only."""
    _ns["model"], _ns["USE_RED_CHANNEL"] = model, use_red          # segment() reads these module globals
    prev = model.training; model.eval()
    # edge-clipped frames are excluded here only: their GT circle is un-fittable, so scoring a
    # circle fit against it measures noise. They remain in train_fold's training set.
    test = [r for r in recs if r["person"] == held_out and not r.get("edge_clip")]
    def _iou(a, b):
        if a is None or b is None: return None
        u = np.logical_or(a, b).sum(); return float(np.logical_and(a, b).sum()) / u if u else 0.0
    P = dict(iris_n=0, iris_det=0, iris_iou=[], pup_n=0, pup_det=0,
             pup_iou=[], pup_iou_all=[], pup_de=[], vis_n=0, vis_ok=0,
             visible_n=0, visible_ok=0, hidden_n=0, hidden_ok=0)
    gt = {r["frame"]: r for r in csv.DictReader(open("labeling/gt.csv"))}
    for r in test:
        g = gt[r["frame"]]; im = cv2.imread(r["frame_path"]); h, w = im.shape[:2]
        bgr = cv2.resize(im, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
        with torch.no_grad():
            m = measure(segment(ritnet_input(bgr)), bgr)
            if eval_visibility and hasattr(model, "forward_with_visibility"):
                x = preprocess(ritnet_input(bgr)).to(DEVICE)
                _, vis_logit = model.forward_with_visibility(x)
                predicted_visible = bool(torch.sigmoid(vis_logit)[0].item() >= 0.5)
                actual_visible = g["category"] == "full"
                P["vis_n"] += 1; P["vis_ok"] += int(predicted_visible == actual_visible)
                key = "visible" if actual_visible else "hidden"
                P[f"{key}_n"] += 1; P[f"{key}_ok"] += int(predicted_visible == actual_visible)
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
                pupil_iou = _iou(cv2.resize(gp, (IN_W, IN_H), interpolation=cv2.INTER_NEAREST).astype(bool), disk_mask(m["pupil_circle"], (IN_H, IN_W)))
                P["pup_iou"].append(pupil_iou); P["pup_iou_all"].append(pupil_iou)
                P["pup_de"].append(abs(m["pupil_px"] - 2 * float(g["pupil_r"]) * s) / (2 * float(g["pupil_r"]) * s))
            else:
                P["pup_iou_all"].append(0.0)  # a miss is zero overlap, not omitted from the mean
    model.train(prev)
    f = lambda l: round(float(np.mean(l)), 3) if l else None
    out = dict(iris_det=P["iris_det"] / max(P["iris_n"], 1), iris_iou=f(P["iris_iou"]),
               pup_det=P["pup_det"] / max(P["pup_n"], 1), pup_iou_all=f(P["pup_iou_all"]),
               pup_iou_detected=f(P["pup_iou"]), pup_de_detected=f(P["pup_de"]))
    if eval_visibility:
        out.update(visibility_acc=P["vis_ok"] / max(P["vis_n"], 1),
                   visibility_tpr=P["visible_ok"] / max(P["visible_n"], 1),
                   visibility_tnr=P["hidden_ok"] / max(P["hidden_n"], 1))
    return out

# %% [train one fold]
def train_fold(held_out, recs, val_subj=None):
    exclude = {held_out} | ({val_subj} if val_subj else set())        # val subj is disjoint from train+test
    train_recs = [r for r in recs if r["person"] not in exclude]
    fold_seed = SEED + SUBJECTS.index(held_out)
    np.random.seed(fold_seed); torch.manual_seed(fold_seed)
    loader = DataLoader(IrisPupilDataset(train_recs), batch_size=BATCH,
                        sampler=balanced_sampler(train_recs, fold_seed))
    # IrisPupilDataset constructs a fully loaded/frozen base model, whose discarded random
    # initialization advances PyTorch's RNG. Reset here so the visibility head is reproducible.
    torch.manual_seed(fold_seed)
    model = load_model()

    set_trainable(model, train_substrings=["out_conv", "up_block4", "visibility_head"])
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE, reduction="none",
                                           weight=torch.tensor([1, 1, 1, 6], dtype=torch.float).to(DEVICE))
    visibility_criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=1e-4
    )                                        # LR / weight decay
    best = {"score": -1.0, "epoch": -1, "state": None}     # best-epoch snapshot
    for epoch in range(EPOCHS): # each epoch
        model.train()
        freeze_bn(model)
        total_loss = 0
        for x, y, visible, segmentation_supervised in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            visible = visible.float().to(DEVICE)
            segmentation_supervised = segmentation_supervised.float().to(DEVICE)
            logits, visibility_logits = model.forward_with_visibility(x)
            per_frame_seg = criterion(logits, y).mean(dim=(1, 2))
            seg_loss = (per_frame_seg * segmentation_supervised).sum() / segmentation_supervised.sum().clamp_min(1)
            visibility_loss = visibility_criterion(visibility_logits, visible)
            loss = seg_loss + VISIBILITY_LOSS_WEIGHT * visibility_loss
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            total_loss += loss.item()
        # validate on the VAL subject (disjoint from held_out -> no leak into the reported score)
        v = eval_subject(model, val_subj, recs, eval_visibility=True) if val_subj else None
        print(f"epoch {epoch}: avg loss {total_loss/len(loader):.4f}" + (f"  val[{val_subj}] {v}" if v else ""))
        if v is not None:
            score = v["pup_det"]                           # selection criterion (what to maximize)
            if score > best["score"]:
                best = {"score": score, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
    if best["state"] is not None:                          # restore the best-on-val epoch (vs keep last)
        model.load_state_dict(best["state"])
        print(f"  -> restored best epoch {best['epoch']} (val pup_det {best['score']:.3f})")
    return model

# Best-epoch selection (val_subj=pick_val(...)) is PARKED: at n=11 a single val subject is too
# noisy a selector - it matched fixed-5-epoch where it picked the last epoch and hurt fold D where
# it picked ep0 (mean Δ +0.020 vs +0.023 fixed). Pass val_subj to re-enable once a POOLED multi-
# subject val set makes the signal less noisy. Default (None) reproduces the best config.

def run_fold(held_out, recs):
    base = load_model()
    print("BEFORE:", eval_subject(base, held_out, recs))
    ft = train_fold(held_out, recs)
    after = eval_subject(ft, held_out, recs, eval_visibility=True)
    print("AFTER :", after)
    save_ckpt(ft, held_out)
    return after


def main():
    parser = argparse.ArgumentParser(description="Visibility-aware RITnet fine-tuning")
    parser.add_argument("--held-out", choices=SUBJECTS, default="A",
                        help="run one subject-held-out fold (default: A)")
    parser.add_argument("--loso", action="store_true", help="run every held-out subject")
    args = parser.parse_args()
    recs = load_records()
    targets = SUBJECTS if args.loso else [args.held_out]
    results = {}
    for held_out in targets:
        print(f"\n===== hold out {held_out} =====", flush=True)
        results[held_out] = run_fold(held_out, recs)
    if args.loso:
        print("\n===== LOSO summary =====")
        for subject, result in results.items(): print(f"{subject}: {result}")


if __name__ == "__main__":
    main()

# %%
