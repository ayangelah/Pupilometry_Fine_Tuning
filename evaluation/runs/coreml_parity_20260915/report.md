# Core ML parity check — 2026-09-15

This check compares the class selected at every pixel by the bundled FP16 Core ML model
against the untouched PyTorch RITnet checkpoint. It verifies export fidelity; it is not a
measurement-accuracy or iPhone-speed benchmark.

| Check | Result |
|---|---:|
| Validation crops | 3 |
| Minimum pixel-class agreement | 99.9793% |
| Mean pixel-class agreement | 99.9888% |
| Maximum absolute logit error | 2.628643 |
| Median macOS prediction latency | 15.4 ms |

## Inputs and identity

- Checkpoint: `RITnet/best_model.pkl`
- Checkpoint SHA-256: `02de259b8fe2e26e825101bb5d3f54b48a13a1a264eb0631c367db4596d0defa`
- Core ML specification SHA-256: `bc7f8bad960b9f540a1c765eb9a41cec432fbdfba4f037587d6235504bf38e31`
- Core ML weights SHA-256: `4e2069e8d768401da1fe04ac4f9001e997f1493e9f448633531356a6069e4e53`
- Crops: `subject_a_1_left_eye_f00.png`, `subject_a_1_left_eye_f01.png`, and
  `subject_a_1_left_eye_f02.png`
- Command: `python export_coreml.py --verify-only --validation-frames 3`

The minimum agreement exceeds the export guardrail of 99.9%, so the bundled model is a
faithful mobile representation of the frozen checkpoint for these verification crops.
