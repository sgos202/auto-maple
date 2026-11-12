import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.detection.violetta_yolo_poc import (
    MotionMaskTracker,
    _mask_features,
    export_dataset,
    prepare_frames,
)


class ViolettaYoloPocTests(unittest.TestCase):
    @staticmethod
    def _detections(centers, widths=None):
        widths = widths or [36] * len(centers)
        result = []
        for center_x, box_width in zip(centers, widths):
            mask = np.zeros((160, 800), dtype=np.uint8)
            cv2.ellipse(mask, (int(center_x), 80), (box_width // 2, 28), 0, 0, 360, 255, -1)
            result.append(_mask_features(mask))
        return result

    def test_motion_and_mask_shape_keep_identity_through_crossing(self):
        tracker = MotionMaskTracker(800)
        self.assertTrue(tracker.initialize(self._detections((100, 160, 450, 600))))
        assignments, ambiguous = tracker.update(
            self._detections((115, 145, 465, 585)), 1
        )
        self.assertFalse(ambiguous)
        self.assertEqual(assignments[2], 1)
        assignments, ambiguous = tracker.update(
            self._detections((125, 135, 480, 570)), 2
        )
        self.assertTrue(ambiguous)
        self.assertEqual(assignments, {})
        assignments, ambiguous = tracker.update(
            self._detections((145, 115, 495, 555)), 3
        )
        self.assertFalse(ambiguous)
        self.assertEqual(assignments[2], 0)

    def test_missing_instances_are_unknown_not_guessed(self):
        tracker = MotionMaskTracker(800)
        tracker.initialize(self._detections((100, 250, 450, 600)))
        assignments, ambiguous = tracker.update(self._detections((140, 210, 490)), 1)
        self.assertEqual(assignments, {})
        self.assertFalse(ambiguous)

    def test_export_uses_temporally_separate_train_and_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames_dir = root / "frames"
            frames_dir.mkdir()
            frames = [
                {"image": "frame_00000.jpg", "frame": 0, "time_seconds": 0.0},
                {"image": "frame_00001.jpg", "frame": 90, "time_seconds": 3.0},
            ]
            for frame in frames:
                cv2.imwrite(
                    str(frames_dir / frame["image"]),
                    np.zeros((100, 120, 3), dtype=np.uint8),
                )
            (root / "manifest.json").write_text(
                json.dumps({"source_video": "video.mp4", "frames": frames}),
                encoding="utf-8",
            )
            polygon = [[10, 10], [50, 10], [50, 60], [10, 60]]
            (root / "annotations.json").write_text(
                json.dumps(
                    {
                        "source_video": "video.mp4",
                        "annotations": [
                            {**frame, "polygons": [polygon]} for frame in frames
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config, splits = export_dataset(root)
            self.assertEqual(splits, {"train": 1, "val": 1})
            self.assertTrue(config.exists())
            train_label = (root / "labels/train/frame_00000.txt").read_text(
                encoding="utf-8"
            )
            self.assertTrue(train_label.startswith("0 0.083333 0.100000"))

    def test_prepare_frames_writes_manifest_dimensions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video_path = root / "sample.avi"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                10,
                (120, 80),
            )
            for value in range(10):
                writer.write(np.full((80, 120, 3), value * 10, dtype=np.uint8))
            writer.release()
            dataset = root / "dataset"
            self.assertEqual(prepare_frames(video_path, dataset, 0.5, 0.6), 2)
            manifest = json.loads(
                (dataset / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["frame_size"], [120, 80])
            self.assertLessEqual(
                max(frame["time_seconds"] for frame in manifest["frames"]), 0.6
            )


if __name__ == "__main__":
    unittest.main()
