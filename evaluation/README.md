# Evaluation-first course

The pretrained RITnet checkpoint is the control.  No new fine-tuning run is worth keeping until
the control can be reproduced from a named checkpoint, a named dataset state, and one metric
contract.

See `EXTERNAL_DATASETS.md` for the audited roles and predeclared evaluation/training protocol for
the locally available MOBIUS and SBVPI datasets.

Run the frozen external raw-segmentation controls with:

```bash
python evaluate_external.py --dataset sbvpi
python evaluate_external.py --dataset mobius
```

The runner reads licensed data from `../datasets`, streams one batch at a time, and writes only
dataset-relative IDs and metrics under `evaluation/runs/`.

## Phase 0: freeze the control

Run:

```bash
python evaluate_pretrained.py
```

The command evaluates both controlled visible-light input choices:

- `gray`: the closest visible-light analogue to RITnet's original single-channel input;
- `red`: the current production candidate, intended to preserve dark-iris contrast.

This is a preprocessing ablation, not two different models.  Both use the untouched
`RITnet/best_model.pkl` weights.  Every run records checkpoint, data, and source hashes.

The score layers must stay separate:

1. **Raw model:** class-mask IoU/Dice/recall and boundary distance.
2. **Measurement reconstruction:** fitted-circle overlap, center error, and diameter error.
3. **Product behavior:** geometry-gate frame failures and sequence failures.

Conditional diameter error is never reported without gate acceptance.  A model that returns one
excellent measurement and rejects the rest is not strong.

## Phase 1: close evaluation gaps

Before calling the baseline final:

1. Reconcile every label/manifest mismatch and pending frame reported by the dataset audit.
2. Compare PyTorch and Core ML logits or class masks on the exact same exported crops.
   This is complete for the bundled export: minimum class agreement was 99.9793% across three
   crops. See `runs/coreml_parity_20260915/report.md`.
3. Measure model latency, end-to-end latency, sustained FPS, and thermal behavior on the target
   iPhone.  Host timing is not a substitute.
4. Add frame-level failure tags for blur, illumination, reflection, eyelid/lash occlusion, gaze,
   and clipping.  Automated pipeline failures are already emitted; visual causes still require
   review.

## Phase 2: one hypothesis per experiment

The next experiment is selected from baseline failures, not intuition.  Examples:

| Finding | First controlled experiment |
|---|---|
| Red wins only in lit/dark-iris frames | illumination-conditioned red vs gray selection |
| Raw mask is good but circle/gate fails | postprocessing ablation; do not retrain |
| Raw pupil recall collapses under reflection | reflection augmentation only |
| Good still frames but unstable video | temporal filter/tracker only |
| Boundary error persists across capture conditions | label audit, then one loss/resolution change |

For each experiment, record the hypothesis before running it and change one material variable.
Use the same subject-held-out universe and generate the same artifacts as the control.

## Promotion rule

A candidate does not replace pretrained RITnet because its demo looks better or because one
aggregate number rises.  It must:

- improve the predeclared primary metric (pupil circle IoU with failures scored as zero);
- improve or preserve visible-pupil frame acceptance;
- avoid a material regression in subject-macro diameter error and its 95th percentile;
- improve across subjects rather than being carried by one large subject;
- preserve performance on edge-clipped and other named failure subsets; and
- satisfy the on-device latency budget.

Prefer a paired subject-level confidence interval for the candidate-minus-control delta.  With
only 11 subjects, report the uncertainty and per-subject direction even when the interval remains
wide.

## Holdout policy

The existing 11 subjects are a development cohort: they have already informed preprocessing,
postprocessing, and fine-tuning choices.  LOSO is still the correct development estimate, but it
is not an untouched final test.  Freeze a new consented subject-level holdout before the clinician
MVP review and consult it only after the experiment is selected.
