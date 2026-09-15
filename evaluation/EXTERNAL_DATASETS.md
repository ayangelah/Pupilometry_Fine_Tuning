# External RGB eye datasets

The datasets live at `../datasets/` and are intentionally excluded from Git. They contain
licensed biometric imagery. Commit adapters, split manifests containing dataset-relative IDs,
and aggregate metrics only—never images, masks, or identifying metadata.

## Local inventory (2026-09-15)

| Dataset | What is locally available | Best role here | Main limitation |
|---|---|---|---|
| MOBIUS segmentation subset | 3,559 paired RGB images and four-class masks; 35 subjects; 1,667 brown-eye images; 1,018 iPhone 6s images; 1,179 poor-light images | First external segmentation benchmark, then external pretraining/domain-mix experiment | Static biometric images, not 120 fps PLR video; no torch onset or pupil-dynamics target |
| SBVPI | 1,856 RGB images; 1,840 sclera masks, 128 iris masks, and 121 pupil masks; 120 samples have pupil+iris+sclera together | Clean-domain ceiling/control for the untouched checkpoint | DSLR macro capture, deliberately high quality, sparse pupil/iris labels, and a large phone-domain gap |

MOBIUS is the dataset most likely to improve the deployed segmenter. It uses multiple phones,
includes controlled poor-light captures, and maps directly to RITnet's background/sclera/iris/
pupil classes. Its 1,667 brown-eye images are much larger than the current development cohort.

SBVPI is useful, but initially as a diagnostic. Of its 121 pupil-labeled images, 47 are brown-eye
images from six brown-eye subjects (plus six mixed brown/grey images from one subject). A strong
SBVPI result with a weak MOBIUS/phone result would implicate capture/domain shift; a weak result
on both would implicate model or preprocessing. Do not use SBVPI's many sclera-only images as if
they had negative pupil labels—the pupil is unannotated, not absent.

## Frozen SBVPI result

`evaluation/runs/external_sbvpi_baseline_20260915/` evaluates the 120 samples that have pupil,
iris, and sclera masks together (19 subjects). Subject-macro results:

| Metric | gray | red |
|---|---:|---:|
| Pupil IoU | 0.876 | **0.914** |
| Pupil recall | 0.935 | **0.946** |
| Pupil precision | 0.925 | **0.958** |
| Iris-disk IoU | 0.837 | **0.884** |

The 46 complete brown-eye samples had frame-mean pupil IoU 0.880 with gray and 0.933 with red.
This validates the clean-image capability of the checkpoint and makes MOBIUS the decisive next
test: it can separate a general RGB-domain issue from the app's specific phone/capture regime.
SBVPI sclera IoU was much lower (0.332 gray / 0.446 red), so the result does not support a claim
that every class transfers equally well.

## Label semantics: keep two targets separate

MOBIUS/SBVPI masks describe **visible semantic pixels**. The project's hand-drawn iris and pupil
circles describe **measurement geometry**, including the iris continuation behind the eyelids.
These are both useful ground truths, but they are not interchangeable:

- train/evaluate raw RITnet class logits against visible semantic masks;
- evaluate the downstream circle fit and diameter against the local circle annotations;
- never fill occluded iris pixels as class `iris` in a semantic loss;
- use an ignore region when a local pixel's semantic class is unknown;
- keep `iris_only` as “pupil boundary not human-observable,” not “no pupil exists.”

This separation is a plausible explanation for prior fine-tuning failing to improve the base
model: the local circle-derived masks ask a semantic segmenter to label invisible iris beneath
skin/eyelid as iris.

## Predeclared external-baseline protocol

1. Freeze `RITnet/best_model.pkl`, both `gray` and `red` preprocessing, and the current 400x640
   input size. No thresholds are tuned on the external test results.
2. Derive a tight eye box from the ground-truth visible-eye mask, expand it with the same
   OpenEDS-style framing used by the app, and apply the identical transform to image and mask.
   Report this explicitly as **oracle framing**; it tests segmentation, not eye localization.
3. Score raw per-class IoU, Dice, precision, recall, and boundary distance. Do not use the app's
   circle gate as the primary external metric because these datasets annotate visible regions,
   not full circles.
4. Report subject-macro results and strata:
   - MOBIUS: phone, lighting, eye colour, gaze, and deliberately unusable flag;
   - SBVPI: eye colour and gaze, with unannotated classes excluded rather than counted negative.
5. Save dataset-relative sample IDs, the dataset inventory hash, checkpoint/source hashes, and
   aggregate/frame metrics. Do not save or commit source imagery or rendered previews.

## First training experiment after the benchmarks

Use MOBIUS as external pretraining, then fine-tune within each local LOSO fold. Split MOBIUS by
subject—not image—and keep one fixed MOBIUS subject holdout to detect catastrophic forgetting.
Compare against the untouched checkpoint under the existing 351-frame local contract. The first
run should change only the training data/label semantics; do not simultaneously change the
architecture, resolution, geometry gate, or temporal filter.

Promotion still depends on local measurement behavior: visible-pupil acceptance and pupil-circle
IoU with misses scored zero must improve across subjects without materially worsening diameter
error or iris-only output behavior.

## Licensing boundary

The local repository policy records both datasets as non-commercial research data obtained under
signed agreements. Before distributing model weights trained on either dataset, publishing sample
images, or using them in a commercial product, verify that the exact signed agreements permit
that use and permit distribution of derivative weights. Dataset access is not by itself proof of
those rights.
