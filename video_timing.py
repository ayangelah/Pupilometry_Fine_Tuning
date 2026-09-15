"""Video timing helpers shared by ingestion, onset marking, and curve building.

OpenCV's reported average FPS is metadata, not a time axis. Variable-frame-rate QuickTime
files can contain repeated or skipped nominal frame slots. ``CAP_PROP_POS_MSEC`` is the
decoded frame's container timestamp and is preferred whenever the backend exposes it.
"""
from __future__ import annotations

import math

import cv2


DEFAULT_FPS = 120.0


def capture_fps(cap: cv2.VideoCapture, fallback: float = DEFAULT_FPS) -> float:
    """Return a finite positive average FPS, or ``fallback`` if metadata is unavailable."""
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    return fps if math.isfinite(fps) and fps > 0 else float(fallback)


def decoded_frame_time(
    cap: cv2.VideoCapture,
    frame_index: int,
    fps: float,
    previous: float | None = None,
) -> tuple[float, bool]:
    """Return ``(time_s, timestamp_from_container)`` for the frame just read.

    Some backends return zero or a stale timestamp for every frame. We accept zero for frame
    zero but require later timestamps to increase. The fallback uses this file's average FPS.
    """
    pts = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
    valid = math.isfinite(pts) and pts >= 0 and (
        frame_index == 0 or previous is None or pts > previous
    )
    if valid:
        return pts, True
    fallback = frame_index / fps
    if previous is not None and fallback <= previous:
        fallback = previous + 1.0 / fps
    return fallback, False


def nearest_frame_index(times_s, target_s: float) -> int:
    """Index of the decoded-frame timestamp nearest ``target_s``."""
    import numpy as np

    times = np.asarray(times_s, dtype=float)
    if times.size == 0:
        raise ValueError("cannot select a frame from an empty timestamp sequence")
    return int(np.nanargmin(np.abs(times - float(target_s))))
