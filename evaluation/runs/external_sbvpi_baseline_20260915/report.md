# Frozen pretrained RITnet external baseline: SBVPI

Run: `ritnet_external_sbvpi_20260915T190055Z`
Checkpoint SHA-256: `02de259b8fe2e26e825101bb5d3f54b48a13a1a264eb0631c367db4596d0defa`
Dataset inventory SHA-256: `cb051075113b92104154d1e809af0406f8106e31273ad60bf50d51e6b280fb87`

This is raw semantic segmentation with an oracle ground-truth eye box. It does not test eye localization, temporal tracking, circle measurement, or PLR dynamics.

| Metric | gray | red |
|---|---|---|
| Pupil IoU | 0.876 [0.841, 0.906] | 0.914 [0.886, 0.934] |
| Pupil recall | 0.935 [0.900, 0.964] | 0.946 [0.916, 0.965] |
| Pupil precision | 0.925 [0.903, 0.943] | 0.958 [0.942, 0.970] |
| Iris-disk IoU | 0.837 [0.797, 0.875] | 0.884 [0.849, 0.911] |
| Sclera IoU | 0.332 [0.281, 0.388] | 0.446 [0.398, 0.497] |

## Contract

- Source weights are untouched `RITnet/best_model.pkl`; channels are preprocessing ablations only.
- The visible-eye ground-truth union defines a tight box, expanded to the app's 400x640 OpenEDS-style aspect without geometric stretching.
- Confidence intervals bootstrap subject-level means. Frame means remain available in `summary.json`.
- SBVPI evaluation includes only samples with pupil, iris, and sclera masks; missing class annotations are never treated as negatives.
- No source image, source mask, or rendered preview is copied into the run artifacts.
