import unittest

import cv2
import numpy as np

from import_capture_sidecar import detect_onset
from plr_metrics import dilation_velocity, minimum_diameter
from video_timing import decoded_frame_time, nearest_frame_index
from onboarding.build_reviewed_dataset import rectangle, sample_positions, side_boxes


class FakeCapture:
    def __init__(self, milliseconds): self.milliseconds = milliseconds
    def get(self, prop):
        self.asserted_prop = prop
        return self.milliseconds


class TimingTests(unittest.TestCase):
    def test_container_timestamp_wins(self):
        time_s, from_container = decoded_frame_time(FakeCapture(125.0), 5, 30.0, 0.1)
        self.assertEqual(time_s, 0.125)
        self.assertTrue(from_container)

    def test_stale_timestamp_falls_back_monotonically(self):
        time_s, from_container = decoded_frame_time(FakeCapture(0.0), 5, 30.0, 0.15)
        self.assertAlmostEqual(time_s, 5 / 30)
        self.assertFalse(from_container)

    def test_nearest_timestamp(self):
        self.assertEqual(nearest_frame_index([0.0, 0.033, 0.1], 0.08), 2)


class CaptureSidecarTests(unittest.TestCase):
    def test_luma_step_confirms_optical_onset_after_request(self):
        frames = [
            {"video_time_s": i / 120, "mean_luma": 40 if i < 36 else 70}
            for i in range(80)
        ]
        result = detect_onset({
            "frames": frames,
            "torch_events": [{
                "state": "on",
                "first_frame_after_request_video_time_s": 0.275,
                "requested_host_time_s": 10.0,
            }],
        })
        self.assertEqual(result["source"], "capture_sidecar_luma_step")
        self.assertAlmostEqual(result["confirmed_video_time_s"], 0.3)


class LabelPipelineTests(unittest.TestCase):
    def test_review_positions_match_existing_protocol(self):
        self.assertEqual(sample_positions(1808), {
            "f00": 180, "f01": 542, "f02": 904,
            "f03": 1265, "f04": 1627, "full": 632,
        })

    def test_rectangle_quantizes_corners_before_dimensions(self):
        shape = {"label": "eyebox", "shape_type": "rectangle",
                 "points": [[9.4, 1002.8], [327.9, 1214.1]]}
        self.assertEqual(rectangle(shape), (9, 1002, 318, 212))

    def test_both_eye_boxes_use_patient_anatomical_sides(self):
        image_left = (100, 200, 80, 60)
        image_right = (700, 200, 80, 60)
        self.assertEqual(side_boxes([image_right, image_left], "B", 1000), [
            ("right", image_left), ("left", image_right),
        ])

    def test_single_visible_box_in_both_eye_clip_uses_anatomical_side(self):
        self.assertEqual(side_boxes([(700, 200, 80, 60)], "B", 1000), [
            ("left", (700, 200, 80, 60)),
        ])


class MetricGuardTests(unittest.TestCase):
    def test_nadir_at_clip_end_has_no_dilation_velocity(self):
        t = np.arange(20) / 40
        pupil = np.linspace(5, 3, 20)
        self.assertTrue(np.isnan(dilation_velocity(t, pupil, 0.0)))

    def test_minimum_ignores_pre_latency_outlier(self):
        t = np.arange(40) / 40
        pupil = np.full(40, 4.0); pupil[1] = 1.0; pupil[20:24] = 3.0
        self.assertAlmostEqual(minimum_diameter(t, pupil, 0.0), 3.0)


if __name__ == "__main__":
    unittest.main()
