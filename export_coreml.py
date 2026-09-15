"""Export the pretrained RITnet segmentation network to Core ML and verify parity.

The Core ML boundary is deliberately the already-preprocessed 1x1x640x400 tensor. The iOS
client must reproduce ``ritnet_segment.preprocess`` (red channel, gamma 0.8, CLAHE 1.5/8x8,
then [0, 255] -> [-1, 1]). Keeping that contract explicit prevents a convenient ImageType
conversion from silently changing the model that LOSO evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import coremltools as ct
import cv2
import numpy as np
import torch


REPO = Path(__file__).resolve().parent
RITNET = REPO / "RITnet"
DEFAULT_OUTPUT = (
    REPO / "apps/ios/PupilDetection1/PupilDetection1/Resources/Models/RITnet.mlpackage"
)
IN_W, IN_H = 400, 640

sys.path.insert(0, str(RITNET))
from densenet import DenseNet2D  # noqa: E402


def load_model() -> torch.nn.Module:
    model = DenseNet2D(dropout=True, prob=0.2)
    state = torch.load(RITNET / "best_model.pkl", map_location="cpu", weights_only=True)
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)
    return model.eval()


def preprocess_bgr(image: np.ndarray) -> np.ndarray:
    """Exact Python inference preprocessing, returned as NCHW float32."""
    red = cv2.resize(image[:, :, 2], (IN_W, IN_H), interpolation=cv2.INTER_AREA)
    lut = (255.0 * (np.linspace(0, 1, 256) ** 0.8)).astype(np.uint8)
    red = cv2.LUT(red, lut)
    red = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(red)
    return ((red.astype(np.float32) / 255.0 - 0.5) / 0.5)[None, None]


def validation_inputs(limit: int = 3) -> list[np.ndarray]:
    paths = sorted((REPO / "labeling/frames_v2").glob("*/*.png"))[:limit]
    if not paths:
        return [np.zeros((1, 1, IN_H, IN_W), dtype=np.float32)]
    return [preprocess_bgr(cv2.imread(str(path))) for path in paths]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_coreml(
    converted: ct.models.MLModel,
    model: torch.nn.Module,
    validation_frames: int,
) -> None:
    agreements, absolute_errors, timings = [], [], []
    for array in validation_inputs(validation_frames):
        with torch.no_grad():
            expected = model(torch.from_numpy(array)).numpy()
        started = time.perf_counter()
        actual = converted.predict({"eye": array})["logits"]
        timings.append(time.perf_counter() - started)
        agreements.append(float(np.mean(expected.argmax(1) == actual.argmax(1))))
        absolute_errors.append(float(np.max(np.abs(expected - actual))))

    print(f"checkpoint sha256: {sha256(RITNET / 'best_model.pkl')}")
    print(f"pixel-class agreement: min={min(agreements):.6f}, mean={np.mean(agreements):.6f}")
    print(f"max absolute logit error: {max(absolute_errors):.6f}")
    print(f"Mac prediction latency: median={1000*np.median(timings):.1f} ms (not an iPhone benchmark)")
    if min(agreements) < 0.999:
        raise SystemExit("Core ML class agreement below 99.9%; do not integrate this export")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-frames", type=int, default=3)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify an existing Core ML package without overwriting it",
    )
    args = parser.parse_args()

    torch.manual_seed(17)
    model = load_model()
    if args.verify_only:
        if not args.output.exists():
            raise SystemExit(f"Core ML package not found: {args.output}")
        verify_coreml(ct.models.MLModel(str(args.output)), model, args.validation_frames)
        return

    example = torch.zeros(1, 1, IN_H, IN_W)
    traced = torch.jit.trace(model, example)
    with torch.no_grad():
        torch.jit.freeze(traced)

    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16,
        inputs=[ct.TensorType(name="eye", shape=example.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
    )
    converted.author = "Visualyze pupilometry prototype"
    converted.short_description = "Pretrained RITnet 4-class eye segmentation"
    converted.input_description["eye"] = (
        "Float32 NCHW eye tensor, 1x1x640x400; red + gamma 0.8 + CLAHE 1.5/8x8, normalized [-1,1]"
    )
    converted.output_description["logits"] = (
        "Float32 NCHW class logits: background, sclera, iris, pupil"
    )
    converted.user_defined_metadata["source_checkpoint"] = "RITnet/best_model.pkl"
    converted.user_defined_metadata["input_width"] = str(IN_W)
    converted.user_defined_metadata["input_height"] = str(IN_H)
    converted.user_defined_metadata["preprocessing"] = "red,gamma0.8,clahe1.5_8x8,normalize-1to1"
    converted.user_defined_metadata["checkpoint_sha256"] = sha256(RITNET / "best_model.pkl")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}; remove it explicitly first")
    converted.save(str(args.output))

    print(f"wrote {args.output}")
    verify_coreml(converted, model, args.validation_frames)


if __name__ == "__main__":
    main()
