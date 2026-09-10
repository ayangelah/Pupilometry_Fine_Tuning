"""Frame-by-frame onset finder for all clips.

Usage:
  python find_onset.py              # mark all clips
  python find_onset.py <clip_stem>  # mark one clip

Controls:
  SPACE  - play/pause
  LEFT   - previous frame
  RIGHT  - next frame
  d      - mark this frame as onset
  s      - skip this clip (don't mark)
  q      - quit all
"""
import sys, cv2, json
from pathlib import Path
from collections import OrderedDict

CLIPS_DIR = Path("data/clips")

def find_onset(clip_stem):
    """Interactive frame viewer to find light onset. Returns True if marked, False if skipped."""
    try:
        clip_path = next(CLIPS_DIR.glob(f"{clip_stem}.*"))
    except StopIteration:
        print(f"Clip not found: {clip_stem}")
        return None

    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 120.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = 0
    playing = False

    print(f"\n{'='*60}")
    print(f"Loaded: {clip_path.name}")
    print(f"Frames: {n_frames}, FPS: {fps}")
    print("SPACE=play/pause, LEFT/RIGHT=frame, d=mark onset, s=skip, q=quit")
    print(f"{'='*60}")

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break

        time_s = frame_idx / fps
        title = f"{clip_stem} | Frame {frame_idx}/{n_frames} | Time {time_s:.3f}s"

        cv2.imshow(title, frame)
        key = cv2.waitKey(30 if playing else 0) & 0xFF

        if key == ord(' '):
            playing = not playing
        elif key == ord('d'):
            print(f"\n✓ Onset marked: frame {frame_idx}, time {time_s:.3f}s")
            with open("light_onset.json") as f:
                data = json.load(f)
            data[clip_stem] = round(time_s, 2)
            with open("light_onset.json", "w") as f:
                json.dump(data, f, indent=2)
            print(f"Updated light_onset.json")
            cap.release()
            cv2.destroyAllWindows()
            return True
        elif key == ord('s'):
            print(f"Skipped {clip_stem}")
            cap.release()
            cv2.destroyAllWindows()
            return False
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return None  # quit all
        elif key == 81:  # LEFT arrow
            frame_idx = max(0, frame_idx - 1)
        elif key == 83:  # RIGHT arrow
            frame_idx = min(n_frames - 1, frame_idx + 1)
        else:
            if playing:
                frame_idx = min(n_frames - 1, frame_idx + 1)

    cap.release()
    cv2.destroyAllWindows()
    return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single clip
        find_onset(sys.argv[1])
    else:
        # All clips (exclude .DS_Store and other non-video files)
        video_exts = {".mov", ".mp4", ".avi", ".mkv", ".MOV", ".MP4"}
        clips = sorted(set(p.stem for p in CLIPS_DIR.glob("*") if p.suffix in video_exts))
        print(f"\nMarking onset for {len(clips)} clips...")
        marked = 0
        for clip_stem in clips:
            result = find_onset(clip_stem)
            if result is None:
                print("\nQuit.")
                break
            elif result:
                marked += 1
        print(f"\n✓ Marked {marked}/{len(clips)} clips")
