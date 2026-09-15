"""Extract full-resolution frames for the canonical hand-eyebox review.

Usage: ``python onboarding/extract_frames.py [clip_stem ...]``

Writes 10/30/50/70/90% frames plus a 35% ``full`` frame to
``labeling/eyebox_review``. Label every visible eye with an ``eyebox`` rectangle in Labelme,
then run ``python onboarding/build_reviewed_dataset.py``. Existing images and JSON labels are
never overwritten.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
CLIPS = REPO / "data" / "clips"
REVIEW = REPO / "labeling" / "eyebox_review"
VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv", ".m4v"}


def sample_positions(frame_count: int) -> dict[str, int]:
    regular = np.linspace(frame_count * 0.1, frame_count * 0.9, 5, dtype=int)
    out = {f"f{i:02d}": int(frame) for i, frame in enumerate(regular)}
    out["full"] = int(frame_count * 0.35)
    return out


def clip_paths(requested: list[str]) -> list[Path]:
    videos = sorted(p for p in CLIPS.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not requested:
        return videos
    wanted = set(requested)
    missing = wanted - {p.stem for p in videos}
    if missing:
        raise SystemExit("clip(s) not found: " + ", ".join(sorted(missing)))
    return [p for p in videos if p.stem in wanted]


def main() -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    for clip in clip_paths(sys.argv[1:]):
        cap = cv2.VideoCapture(str(clip))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            print(f"warning: could not inspect {clip.name}")
            continue
        for tag, frame_index in sample_positions(n).items():
            destination = REVIEW / f"{clip.stem}_{tag}.png"
            if destination.exists():
                skipped += 1
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                print(f"warning: could not read {clip.stem} frame {frame_index}")
                continue
            cv2.imwrite(str(destination), frame)
            written += 1
        cap.release()
    print(f"eyebox review: wrote {written}, preserved {skipped} existing frame(s) in {REVIEW}")
    print("next: label every visible eye with an 'eyebox' rectangle, then run build_reviewed_dataset.py")


if __name__ == "__main__":
    main()
