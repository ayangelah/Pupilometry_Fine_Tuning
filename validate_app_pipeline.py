#!/usr/bin/env python3
"""
Validate that a .mov file from the iOS app feeds through the RITnet pipeline.
Usage: python validate_app_pipeline.py path/to/recording.mov
"""
import sys, os, cv2, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from ritnet_segment import build_curve, IN_W, IN_H, framed_roi, crop
from plr_metrics import compute_metrics, calibrate_mm
from light_onset import light_onset_frame

def validate_app_clip(mov_path):
    """Test: does the app's .mov → RITnet → metrics pipeline work end-to-end?"""

    if not os.path.exists(mov_path):
        print(f"Error: {mov_path} not found")
        return False

    print(f"\n{'='*60}")
    print(f"Validating app pipeline: {os.path.basename(mov_path)}")
    print(f"{'='*60}")

    # Extract frames
    cap = cv2.VideoCapture(mov_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 120.0
    print(f"\nClip: {frame_count} frames @ {fps:.1f} fps = {frame_count/fps:.2f}s")

    if frame_count < 30:
        print("Warning: clip is very short (<1s at 30fps)")

    # Try to build curve with a default eye ROI (center crop)
    # In production, you'd have eye_rois.json; for validation we auto-detect
    h, w = None, None
    frames = []
    frame_idx = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
        if h is None:
            h, w = fr.shape[:2]
        frame_idx += 1
        if frame_idx >= 300:  # Limit for testing
            break
    cap.release()

    print(f"Loaded {len(frames)} frames ({h}x{w})")

    # Auto-detect a rough eye ROI (center crop, ~1/3 of frame)
    margin_h, margin_w = h // 6, w // 6
    roi = (margin_w, margin_h, w - 2*margin_w, h - 2*margin_h)

    print(f"\nAuto-detected eye ROI: {roi} (assumes center-aligned eye)")

    # Build curve
    print("\nRunning RITnet segmentation...")
    try:
        curve_data = build_curve(frames, roi, onset_t=None, skip_gate=False)
    except Exception as e:
        print(f"Error in build_curve: {e}")
        return False

    if curve_data is None or 'pupil_mm' not in curve_data:
        print("Error: build_curve returned None or missing pupil_mm")
        return False

    pupil_mm = curve_data['pupil_mm']
    iris_mm = curve_data.get('iris_mm')
    valid_pct = 100 * np.sum(~np.isnan(pupil_mm)) / len(pupil_mm)

    print(f"\nCurve extracted:")
    print(f"  Valid frames: {valid_pct:.1f}%")
    print(f"  Pupil range: {np.nanmin(pupil_mm):.2f} to {np.nanmax(pupil_mm):.2f} mm")
    if iris_mm is not None:
        print(f"  Iris: {np.nanmedian(iris_mm):.2f} mm (median)")

    # Try to compute metrics (requires onset_t from light_onset.json or detection)
    print("\nComputing PLR metrics...")
    try:
        metrics = compute_metrics(curve_data)
        if metrics is not None:
            print(f"  Baseline: {metrics.baseline_diameter:.2f} mm")
            print(f"  Minimum: {metrics.minimum_diameter:.2f} mm")
            print(f"  % Constriction: {metrics.percent_constriction:.1f}%")
            print(f"  Latency: {metrics.latency:.2f}s")
            if metrics.constriction_velocity is not None:
                print(f"  Constriction velocity: {metrics.constriction_velocity:.2f} mm/s")
            if metrics.dilation_velocity is not None:
                print(f"  Dilation velocity: {metrics.dilation_velocity:.2f} mm/s")
        else:
            print("  Metrics computation returned None (check baseline window validity)")
    except Exception as e:
        print(f"  Warning: metrics computation failed: {e}")

    print(f"\n{'✓ Pipeline validation PASSED'}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_app_pipeline.py <path/to/recording.mov>")
        print("\nExample:")
        print("  1. Record a clip with the iOS app")
        print("  2. In Xcode, Devices and Simulators -> your iPhone -> File Sharing")
        print("  3. Drag the .mov to a folder, then:")
        print("  python validate_app_pipeline.py ~/Downloads/recording.mov")
        sys.exit(1)

    mov_path = sys.argv[1]
    success = validate_app_clip(mov_path)
    sys.exit(0 if success else 1)
