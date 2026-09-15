# Frozen pretrained RITnet baseline

Run: `ritnet_pretrained_20260915T183427Z`
Checkpoint SHA-256: `02de259b8fe2e26e825101bb5d3f54b48a13a1a264eb0631c367db4596d0defa`
Dataset SHA-256: `ae5d5445aa766996c8382e465bcfa8b9457cb2c4e2549c519a90e0792507ff24`
Git commit: `3f8eb8e342bf7344b609c52ea222997a911dadad` (dirty working tree)

This run performs inference only. No subject is used for training, no checkpoint is selected, and no threshold is tuned.

## Dataset integrity

- Status: **clean**
- 351 of 351 GT rows were eligible; 0 were excluded for explicit integrity issues.
- Eligible categories: `{'full': 192, 'iris_only': 159}` across 61 eye sequences.
- 0 PNGs have no Labelme JSON and remain pending.

## Subject-macro results

Confidence intervals resample subjects, not frames. Values are macro means across subjects; conditional errors use only gate-accepted frames and must be read with acceptance rate.

| Metric | gray | red |
|---|---|---|
| Raw pupil IoU | 0.6 [0.5, 0.6] | 0.7 [0.6, 0.8] |
| Raw pupil recall | 83.0% [74.6, 90.0] | 88.7% [84.4, 92.7] |
| Gate acceptance on visible-pupil frames | 59.2% [48.3, 70.1] | 67.7% [56.4, 78.6] |
| Pupil circle IoU, misses = 0 | 0.4 [0.3, 0.5] | 0.5 [0.4, 0.6] |
| Pupil circle IoU, accepted only | 0.7 [0.6, 0.8] | 0.8 [0.7, 0.8] |
| Pupil diameter absolute relative error, accepted only | 18.2% [10.9, 26.3] | 9.0% [6.9, 11.1] |
| Iris-normalized pupil error, mm-equivalent, accepted only | 0.7 [0.6, 0.9] | 0.5 [0.4, 0.7] |
| Pupil center error / GT diameter, accepted only | 0.1 [0.1, 0.2] | 0.1 [0.1, 0.1] |
| Sequences with zero accepted visible-pupil frames | 18.2% [7.6, 30.3] | 12.9% [3.8, 23.5] |
| Unscorable acceptance on iris-only frames | 19.0% [9.3, 29.5] | 28.9% [17.1, 43.7] |

## Per-subject production-candidate view (red channel)

| Subject | Visible-pupil frames | Gate acceptance | Raw pupil IoU | Circle IoU, misses = 0 | Diameter abs. relative error, accepted |
|---|---:|---:|---:|---:|---:|
| A | 47 | 80.9% | 0.640 | 0.604 | 13.2% |
| B | 16 | 68.8% | 0.680 | 0.573 | 6.2% |
| C | 5 | 100.0% | 0.755 | 0.755 | 11.2% |
| D | 12 | 58.3% | 0.897 | 0.522 | 4.4% |
| E | 22 | 45.5% | 0.627 | 0.351 | 13.7% |
| F | 19 | 89.5% | 0.780 | 0.700 | 9.6% |
| G | 6 | 33.3% | 0.636 | 0.296 | 6.4% |
| H | 13 | 53.8% | 0.518 | 0.337 | 14.7% |
| I | 12 | 75.0% | 0.649 | 0.603 | 6.0% |
| J | 30 | 60.0% | 0.673 | 0.463 | 7.1% |
| K | 10 | 80.0% | 0.782 | 0.655 | 6.1% |

## Interpretation boundaries

- Raw pupil scores use only `full` frames where a human labeled a visible pupil boundary.
- `iris_only` frames have no pupil ground truth. An emitted measurement is therefore **unscorable**, not automatically a false positive.
- Iris labels reconstruct a circle under eyelids. Raw iris-mask overlap is diagnostic; fitted-circle error is the measurement-relevant result.
- The mm-equivalent error compares prediction and annotation through the same 11.7 mm iris assumption. It is not physical-mm validation against a reference pupillometer.
- `edge_clip` remains in the primary result and is available as a column for stratified analysis; hard cases are not silently removed.
- Host model timing is a development diagnostic, not an iPhone/Core ML latency claim.

## Artifacts

- `run_manifest.json`: checkpoint, code, data, environment, and evaluation contract identity
- `summary.json`: subject-macro confidence intervals plus frame-level distributions
- `subject_metrics.csv`: every fold/subject separately
- `sequence_metrics.csv`: video/eye-sequence failure rates
- `frame_metrics.csv`: auditable predictions, errors, and gate failure reasons
