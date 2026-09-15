"""Import a Visualyze iOS capture sidecar into ``light_onset.json``.

The torch API time is a request bound. The confirmed onset is the first sustained luma step in
the video-data samples after that request. If a step cannot be confirmed, the first delivered
frame after the request is stored as explicitly provisional.

Usage: ``python import_capture_sidecar.py clip.capture.json [clip_stem]``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ONSETS = Path("light_onset.json")


def detect_onset(sidecar: dict) -> dict:
    events = [event for event in sidecar.get("torch_events", []) if event.get("state") == "on"]
    if not events: raise ValueError("sidecar has no torch-on event")
    event = events[0]
    bound = float(event["first_frame_after_request_video_time_s"])
    frames = sidecar.get("frames", [])
    t = np.asarray([frame["video_time_s"] for frame in frames], float)
    luma = np.asarray([frame["mean_luma"] for frame in frames], float)
    before = np.isfinite(luma) & (t >= bound - 0.20) & (t < bound)
    after = np.isfinite(luma) & (t >= bound) & (t <= bound + 0.30)
    confirmed = None
    if before.sum() >= 3 and after.sum() >= 2:
        baseline = float(np.median(luma[before]))
        mad = float(np.median(np.abs(luma[before] - baseline)))
        threshold = baseline + max(5.0, 6.0 * mad)
        indices = np.flatnonzero(after)
        for first, second in zip(indices[:-1], indices[1:]):
            if second == first + 1 and luma[first] >= threshold and luma[second] >= threshold:
                confirmed = float(t[first]); break
    result = {
        "source": "capture_sidecar_luma_step" if confirmed is not None else "first_frame_after_torch_request",
        "confirmed_video_time_s": confirmed,
        "first_frame_after_request_video_time_s": bound,
        "torch_request_host_time_s": event.get("requested_host_time_s"),
    }
    return result


def main() -> None:
    if len(sys.argv) not in (2, 3): raise SystemExit(__doc__)
    sidecar_path = Path(sys.argv[1]); clip_stem = sys.argv[2] if len(sys.argv) == 3 else sidecar_path.name.removesuffix(".capture.json")
    result = detect_onset(json.loads(sidecar_path.read_text()))
    onsets = json.loads(ONSETS.read_text()) if ONSETS.exists() else {}
    onsets[clip_stem] = result
    ONSETS.write_text(json.dumps(onsets, indent=2, sort_keys=True) + "\n")
    status = "confirmed" if result["confirmed_video_time_s"] is not None else "PROVISIONAL"
    print(f"{clip_stem}: {status} onset -> {result}")


if __name__ == "__main__":
    main()
