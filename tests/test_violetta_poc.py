import unittest

import cv2
import numpy as np

from src.detection.violetta_poc import (
    MultiCopyIdentityTracker,
    _draw,
    _target_snapshot,
    detect_instances,
    makeup_scores,
    unique_makeup_index,
)


class ViolettaPocTests(unittest.TestCase):
    def _lineup(self, target_slot=2):
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        pink = (180, 80, 190)
        for index, center_x in enumerate((120, 310, 500, 690), start=1):
            cv2.ellipse(
                frame,
                (center_x, 290),
                (58, 76),
                0,
                0,
                360,
                pink,
                -1,
            )
            if index == target_slot:
                cv2.circle(
                    frame,
                    (center_x - 8, 290 + 42),
                    9,
                    (255, 200, 30),
                    -1,
                )
        return frame

    def test_segments_all_copies_and_finds_unique_makeup(self):
        frame = self._lineup()
        detections = detect_instances(frame)
        self.assertEqual(len(detections), 4)

        scores = makeup_scores(frame, detections)
        self.assertEqual(unique_makeup_index(scores), 1)

    def test_rejects_ambiguous_makeup_evidence(self):
        self.assertIsNone(unique_makeup_index([0.03, 0.05, 0.04, 0.02]))

    def test_tracks_ids_through_a_visible_crossing(self):
        tracker = MultiCopyIdentityTracker(800)

        def detections_at(centers):
            return [
                {"center": np.array((x, 200), dtype=np.float32)}
                for x in centers
            ]

        initial = detections_at((100, 250, 450, 600))
        assignments, ambiguous = tracker.update(initial, 0, makeup_index=1)
        self.assertFalse(ambiguous)
        self.assertEqual(assignments[2], 1)

        moved = detections_at((140, 210, 490, 560))
        assignments, ambiguous = tracker.update(moved, 1, makeup_index=1)
        self.assertFalse(ambiguous)

        crossed = detections_at((180, 170, 530, 520))
        assignments, ambiguous = tracker.update(crossed, 2, makeup_index=1)
        self.assertFalse(ambiguous)
        self.assertEqual(assignments[1], 0)
        self.assertEqual(assignments[2], 1)

    def test_uses_motion_only_after_initial_target_identification(self):
        tracker = MultiCopyIdentityTracker(800)
        initial = [
            {"center": np.array((x, 200), dtype=np.float32)}
            for x in (100, 250, 450, 600)
        ]
        tracker.update(initial, 0, makeup_index=1)

        moved = [
            {"center": np.array((x, 200), dtype=np.float32)}
            for x in (140, 210, 490, 560)
        ]
        assignments, ambiguous = tracker.update(moved, 1)
        self.assertFalse(ambiguous)
        self.assertEqual(assignments[2], 1)

    def test_does_not_assign_target_after_long_unobserved_gap(self):
        tracker = MultiCopyIdentityTracker(800)
        initial = [
            {"center": np.array((x, 200), dtype=np.float32)}
            for x in (100, 250, 400, 550)
        ]
        tracker.update(initial, 0, makeup_index=1)

        final = [
            {"center": np.array((x, 200), dtype=np.float32)}
            for x in (100, 250, 400, 550)
        ]
        assignments, ambiguous = tracker.update(final, 100)
        self.assertFalse(ambiguous)
        self.assertEqual(assignments, {})

    def test_draws_target_box_only_when_identity_is_supported(self):
        frame = self._lineup()
        detections = detect_instances(frame)
        assignments = {index + 1: index for index in range(4)}

        tracked = _draw(
            frame, detections, assignments, 2, "TARGET BOXED", None
        )
        unknown = _draw(
            frame, detections, {}, 2, "TARGET UNKNOWN", None
        )
        x, y, width, height = detections[1]["bbox"]
        self.assertTrue(np.any(np.all(tracked[y, x:x + width] == (0, 255, 255), axis=1)))
        self.assertFalse(np.any(np.all(unknown[y, x:x + width] == (0, 255, 255), axis=1)))

    def test_motion_snapshot_highlights_target_box(self):
        frame = self._lineup()
        detection = detect_instances(frame)[1]
        snapshot = _target_snapshot(frame, detection, "TARGET ID 2 - START")
        x, y, width, _ = detection["bbox"]
        self.assertTrue(
            np.any(
                np.all(
                    snapshot[y, x : x + width] == (0, 255, 255),
                    axis=1,
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
