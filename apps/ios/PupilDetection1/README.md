# Visualyze iOS experimental capture build

This native iOS 16+ app bundles the FP16 Core ML export of the pretrained RITnet model. For each
aligned eye it crops the alignment region, applies the same preprocessing used by Python
(red channel, gamma 0.8, CLAHE 1.5 with an 8x8 grid), runs inference away from the capture queue,
and draws fitted iris and pupil outlines over the preview. Blue is the iris; green is an accepted
pupil fit; orange is a fit rejected by the current nested-circle geometry gate.

The app displays preliminary iris-ratio diameter and, after a PLR stimulus, constriction, latency,
and mean constriction velocity. These outputs are for model/capture feedback, not clinical use.

## Capture modes

- **120 PLR** requests the best available 120 fps back-camera format, preferring 4K and HDR. It
  automatically retries at 1080p if the phone cannot run the 4K movie-plus-live-model pipeline.
  Recording remains available regardless of the current model/quality advisory and is intentionally
  limited to one eye.
- **4K60 DATA** requests 4K 60 fps HDR for single-eye training footage and deliberately does not
  require model acceptance, so sharp, well-exposed model failures can be retained for labeling. It
  also falls back to 1080p60 if necessary.

The selected device is allowed to switch constituent lenses during alignment when the phone
supports a virtual dual/triple camera. Lens switching is locked for the movie recording. The app
uses standard stabilization, or low-latency stabilization where the OS and format support it.
The saved movie uses HEVC when available. Model buffers are preview-sized 8-bit YUV proxies; this
keeps live inference from forcing the recorded movie down to model resolution and lets AVFoundation
respond more quickly to focus, exposure, and white-balance changes. The one-eye inference adapter
implements Device RGB conversion and the intended independent X/Y resize. Do not replace that
resize with a single aspect-preserving scale: the model requires a fully populated 400x640 tensor.
Exact Swift-produced input-tensor parity with Python has not yet been measured.

## Alignment and image-quality guidance

Before recording, the app continuously focuses and meters on the eye alignment region, restricts
autofocus toward the near range, applies a modest -0.7 EV exposure bias, and checks:

- focus and exposure have settled;
- eye-region sharpness and brightness;
- clipped highlights and crushed shadows;
- in 120 PLR mode, iris presence, useful scale, and accepted nested pupil geometry.

The orange alignment box turns green after five consecutive passing samples, but recording is
available as soon as the camera starts. The checks are advisory so a brief model failure cannot
block capture. Recording begins under the existing continuous focus/metering state; after the movie
output confirms it has started, the app waits up to 1.2 seconds for focus, exposure, and white
balance to settle, locks their current values, and then enables the PLR torch. The numerical
thresholds are provisional starting points and must be calibrated on the target iPhone models using
representative eyes, distances, and lighting.

Inference is attempted every fourth input frame (at most 30 Hz in 120 PLR and 15 Hz in 4K60 DATA),
and a new request is dropped while inference is already running. The single-eye crop, inference,
overlay, and circle extraction path intentionally matches the best-performing prototype checkpoint;
the nested-circle gate is unchanged. Movie writing uses `AVCaptureMovieFileOutput` independently of
the model path.

## Torch timing and exported diagnostics

The torch uses a repeatable provisional level of 0.20 rather than full power. This is not a
clinically calibrated illuminance. In 120 PLR mode, tapping the torch before recording explains why
the app requires recording to start first: the movie needs a pre-light baseline and a timestampable
stimulus. The torch can be activated after the movie-output delegate confirms recording started.

The original clean movie is a temporary input while the app renders a second video with the
iris/pupil circles and preliminary measurements burned in. The annotated copy remains in the app
for user playback. Both clean and annotated movies are imported into Photos in one transaction;
only after Photos confirms the import does the app remove its temporary clean movie. If Photos
permission or import fails, the clean movie is retained in Documents to prevent data loss. The
schema-v6 `.capture.json` remains available through Files/Finder and identifies the
`single_eye_known_good_v1` inference path in addition to requested
and active cadence, capture and model-buffer dimensions, HDR and camera metadata, stabilization and
torch level, frame PTS and luma samples, eye-region quality samples, dropped-frame count, live fits,
per-stage preprocessing/model/postprocessing latency, and torch request plus first-post-request-frame
timestamps. The app estimates physical onset from a sustained luma rise
after the torch request and uses it for the preliminary PLR readout; the request timestamp alone only
identifies when software asked the torch to change.

## On-device acceptance check

The first iPhone 17 Pro smoke test on September 15, 2026 produced qualitatively strong live pupil
and iris tracking. That is encouraging integration feedback, not a quantitative accuracy or speed
result. The checklist below remains open until its evidence is saved from representative captures.

Run a clean install on each target iPhone and verify:

1. Both modes show the expected active resolution, cadence, HDR state, device, and stabilization in
   the Xcode console and sidecar.
2. The single-eye circles match the live preview and the burned-in annotated export.
3. The camera focuses at the intended working distance without repeatedly switching lenses.
4. The readiness prompts reject real blur and clipped torch reflections without blocking usable
   dark-brown-eye frames.
5. Preview latency remains usable and the sidecar reports few dropped data frames.
6. Photos contains both the clean movie at its requested cadence and the annotated movie, while the
   in-app library plays the annotated copy.
7. Torch request timing agrees with the first post-request frame and the visible luma rise.

For a useful comparison, capture the same eye at a marked distance in the Apple Camera app, 120 PLR,
and 4K60 DATA, then compare a still from each recording rather than comparing only their live
previews. Camera.app applies private computational processing that third-party capture pipelines may
not reproduce exactly.

The bundled model can be reproduced with `pip install -r requirements-coreml.txt` followed by
`python export_coreml.py`. Core ML export dependencies are development-only and are not required by
the iOS app at runtime. The exporter refuses to overwrite an existing package and verifies
pixel-class parity against real labeled-eye crops before accepting a new export. The checked-in
package reached 99.9793% minimum pixel-class agreement with PyTorch on three crops; that comparison
starts from an already-preprocessed tensor and does not verify the Swift camera-to-tensor path.
