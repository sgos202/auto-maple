import glob
import unittest

import cv2
import numpy as np

from src.detection.mini_game_alert import (
    MiniGamePromptDetector,
    PopUpAnchorDetector,
    RisingEdgeAlert,
)


class MiniGamePromptDetectorTests(unittest.TestCase):
    def setUp(self):
        self.template = np.random.default_rng(42).integers(
            0, 256, (110, 285), dtype=np.uint8
        )
        self.detector = MiniGamePromptDetector(self.template)

    def test_detects_prompt_in_bottom_right_search_area(self):
        frame = np.zeros((340, 996), dtype=np.uint8)
        frame[70:180, 700:985] = self.template
        self.assertTrue(self.detector.detect(frame))

    def test_detects_scaled_prompt_in_bgra_screen_capture(self):
        frame = np.zeros((510, 1494, 4), dtype=np.uint8)
        scaled = cv2.resize(self.template, (428, 165), interpolation=cv2.INTER_CUBIC)
        frame[105:270, 1040:1468, :3] = scaled[:, :, None]
        frame[:, :, 3] = 255
        self.assertTrue(self.detector.detect(frame))

    def test_does_not_match_prompt_outside_search_area(self):
        frame = np.zeros((340, 996), dtype=np.uint8)
        frame[70:180, 100:385] = self.template
        self.assertFalse(self.detector.detect(frame))

    def test_alert_triggers_once_until_prompt_disappears(self):
        alert = RisingEdgeAlert(release_frames=3)
        self.assertFalse(alert.update(False))
        self.assertTrue(alert.update(True))
        self.assertFalse(alert.update(True))
        self.assertFalse(alert.update(False))
        self.assertFalse(alert.update(True))
        self.assertFalse(alert.update(False))
        self.assertFalse(alert.update(False))
        self.assertFalse(alert.update(False))
        self.assertTrue(alert.update(True))


class PopUpAnchorDetectorTests(unittest.TestCase):
    SCALES = tuple(round(0.8 + 0.05 * i, 2) for i in range(10))

    def _detector(self, prefix):
        anchors = [cv2.imread(p) for p in sorted(glob.glob(f"assets/pop_up_anchors/{prefix}_*.png"))]
        self.assertTrue(anchors)
        return PopUpAnchorDetector(anchors, threshold=0.83, scales=self.SCALES, downscale=0.5)

    def _frame_with(self, path, scale=1.0, wipe_bottom=False, x=420, y=230):
        popup = cv2.resize(cv2.imread(path), None, fx=scale, fy=scale)
        if wipe_bottom:     # Simulate different body text
            half = popup.shape[0] // 2
            popup[half:] = popup[half:].mean(axis=(0, 1))
        frame = cv2.GaussianBlur(
            np.random.default_rng(0).integers(0, 256, (768, 1366, 3), dtype=np.uint8), (7, 7), 0
        )
        h, w = popup.shape[:2]
        frame[y:y + h, x:x + w] = popup
        return frame

    def test_detects_each_pop_up_at_different_sizes(self):
        for prefix in ("pop-up-1", "pop-up-2"):
            detector = self._detector(prefix)
            for scale in (0.85, 1.0, 1.2):
                with self.subTest(prefix=prefix, scale=scale):
                    self.assertTrue(detector.detect(self._frame_with(f"assets/{prefix}.png", scale)))

    def test_detects_pop_up_when_body_text_differs(self):
        for prefix in ("pop-up-1", "pop-up-2"):
            with self.subTest(prefix=prefix):
                frame = self._frame_with(f"assets/{prefix}.png", wipe_bottom=prefix == "pop-up-1")
                self.assertTrue(self._detector(prefix).detect(frame))

    def test_no_detection_on_background_or_other_pop_up(self):
        background = self._frame_with("assets/pop-up-1.png", scale=0.01)
        for prefix, other in (("pop-up-1", "pop-up-2"), ("pop-up-2", "pop-up-1")):
            detector = self._detector(prefix)
            with self.subTest(prefix=prefix):
                self.assertFalse(detector.detect(background))
                self.assertFalse(detector.detect(self._frame_with(f"assets/{other}.png")))


if __name__ == "__main__":
    unittest.main()
