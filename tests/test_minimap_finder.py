"""The minimap is found on its own from its white outline: at start, after the
bot restarts the game, when the window moves, and on a new map."""

import os
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from src.modules import minimap_finder
from src.modules.minimap_finder import find_minimap_area, find_minimap_areas


def outlined(frame, left, top, width, height, fill=(90, 60, 30)):
    """Draws a minimap map area the way the game does: a white line 2 px thick
    above and below and 3 px at the sides. Returns the area inside it."""
    frame[top:top + height, left:left + width] = 255
    x1, y1, x2, y2 = left + 3, top + 2, left + width - 3, top + height - 2
    frame[y1:y2, x1:x2] = fill
    return (x1, y1), (x2, y2)


def scene(height=768, width=1024):
    frame = np.zeros((height, width, 3), np.uint8)
    frame[:] = (200, 150, 60)                       # sky
    frame[300:420, 200:700] = 255                   # a cloud: white, no outline
    return frame


class OutlineTests(unittest.TestCase):

    def test_finds_the_area_inside_the_outline(self):
        frame = scene()
        area = outlined(frame, 3, 70, 217, 87)
        self.assertEqual(find_minimap_areas(frame), [area])

    def test_any_size_and_place(self):
        for left, top, width, height in ((1, 65, 249, 89), (400, 300, 120, 60),
                                         (16, 91, 217, 88), (700, 20, 300, 200)):
            frame = scene()
            frame[300:420, 200:700] = (200, 150, 60)
            area = outlined(frame, left, top, width, height)
            self.assertEqual(find_minimap_areas(frame), [area], (left, top))

    def test_a_four_channel_screenshot(self):
        frame = scene()
        area = outlined(frame, 3, 70, 217, 87)
        bgra = np.dstack([frame, np.full(frame.shape[:2], 255, np.uint8)])
        self.assertEqual(find_minimap_area(bgra), area)

    def test_nothing_without_a_minimap(self):
        self.assertEqual(find_minimap_areas(scene()), [])
        self.assertIsNone(find_minimap_area(scene()))
        self.assertIsNone(find_minimap_area(None))

    def test_a_plain_white_box_is_not_a_map(self):
        frame = scene()
        frame[70:157, 3:220] = 255
        self.assertEqual(find_minimap_areas(frame), [])

    def test_something_over_the_outline_hides_it(self):
        frame = scene()
        outlined(frame, 3, 70, 217, 87)
        frame[60:100, 100:130] = (20, 20, 20)       # the mouse pointer on the top line
        self.assertEqual(find_minimap_areas(frame), [])

    def test_pale_rows_at_the_bottom_of_the_map(self):
        """Perion: the map's last rows are white and touch the bottom line."""
        frame = scene()
        (x1, y1), (x2, y2) = outlined(frame, 1, 65, 249, 91)
        frame[y2 - 2:y2, x1:x2] = 255
        self.assertEqual(find_minimap_areas(frame), [((x1, y1), (x2, y2 - 2))])

    def test_the_marker_picks_among_several(self):
        frame = scene()
        first = outlined(frame, 3, 70, 217, 87)
        second = outlined(frame, 500, 500, 200, 100)
        self.assertEqual(find_minimap_areas(frame), [first, second])
        self.assertEqual(find_minimap_area(frame), first)
        self.assertEqual(find_minimap_area(frame, accept=lambda tl, br: tl == second[0]), second)
        none = lambda tl, br: False
        self.assertEqual(find_minimap_area(frame, accept=none), first)
        self.assertIsNone(find_minimap_area(frame, accept=none, strict=True))


class RecordedScreenTests(unittest.TestCase):
    """Real screenshots of the game."""

    CASES = (
        ('assets/perion_street_corner.png', ((4, 67), (247, 152))),
        ('assets/restart/1-map.jpg', ((19, 93), (231, 177))),
        ('assets/restart/4-login.jpg', None),
        ('assets/restart/5-world.jpg', None),
        ('assets/restart/6-channels.jpg', None),
        ('assets/restart/7-character.jpg', None),
    )

    def test_recorded_screens(self):
        seen = 0
        for path, expected in self.CASES:
            if not os.path.exists(path):
                continue
            seen += 1
            self.assertEqual(find_minimap_area(cv2.imread(path)), expected, path)
        if not seen:
            self.skipTest('no recorded screens')

    def test_the_capture_finds_it_with_the_marker(self):
        from src.modules import capture as capture_module
        seen = 0
        stub = SimpleNamespace(capture=SimpleNamespace(minimap_ratio=1.0))
        fake = SimpleNamespace(_report_error=lambda *a: None)
        for path, expected in self.CASES:
            if not os.path.exists(path):
                continue
            seen += 1
            with patch.object(capture_module.utils, 'config', stub):
                found = capture_module.Capture.find_outline(fake, cv2.imread(path))
            self.assertEqual(found, expected, path)
        if not seen:
            self.skipTest('no recorded screens')


class RecalibrationTests(unittest.TestCase):

    def needs(self, now_rect, old_rect=(0, 0, 1024, 768), outline=None,
              area=((6, 72), (217, 155)), by_hand=False, marker_lost=False):
        from src.modules.capture import Capture
        fake = SimpleNamespace(
            game_window_rect=lambda: (1, now_rect),
            find_outline=lambda frame: outline,
            frame=np.zeros((10, 10, 3), np.uint8))
        return Capture._needs_recalibration(fake, old_rect, area, by_hand, marker_lost)

    def test_same_window_and_marker_seen(self):
        self.assertFalse(self.needs((0, 0, 1024, 768), outline=((6, 72), (217, 155))))
        self.assertFalse(self.needs((0, 0, 1024, 768), outline=None))   # outline hidden
        self.assertFalse(self.needs((0, 0, 1024, 768), outline=((7, 72), (219, 156))))  # 1-2 px

    def test_minimap_shrank_with_the_marker_still_inside(self):
        # run-20261118-170231: kept a 262x120 area while the map was 211x83
        self.assertTrue(self.needs((0, 0, 1024, 768), area=((14, 103), (276, 223)),
                                   outline=((14, 103), (225, 186))))

    def test_the_window_moved_or_is_new(self):
        self.assertTrue(self.needs((40, 30, 1064, 798)))
        self.assertTrue(self.needs((0, 0, 0, 0)))           # the game closed
        self.assertTrue(self.needs((40, 30, 1064, 798), by_hand=True))

    def test_a_new_map_with_another_minimap(self):
        new = ((4, 67), (247, 152))
        self.assertTrue(self.needs((0, 0, 1024, 768), outline=new, marker_lost=True))

    def test_marker_lost_but_the_minimap_is_the_same(self):
        same = ((6, 72), (217, 155))
        self.assertFalse(self.needs((0, 0, 1024, 768), outline=same, marker_lost=True))
        self.assertFalse(self.needs((0, 0, 1024, 768), outline=None, marker_lost=True))

    def test_an_area_drawn_by_hand_is_kept(self):
        new = ((4, 67), (247, 152))
        self.assertFalse(self.needs((0, 0, 1024, 768), outline=new,
                                    by_hand=True, marker_lost=True))


class AfterRestartTests(unittest.TestCase):

    def test_the_minimap_is_found_again_after_a_restart(self):
        from src.modules.game_restart import RestartRunner
        capture = SimpleNamespace(calibrated=True)
        dropped = []

        def capture_loop():                 # notices, then finds it again
            deadline = time.monotonic() + 5
            while capture.calibrated and time.monotonic() < deadline:
                time.sleep(0.01)
            dropped.append(not capture.calibrated)
            capture.calibrated = True

        thread = threading.Thread(target=capture_loop, daemon=True)
        thread.start()
        real_sleep = time.sleep
        with patch('src.modules.game_restart.time.sleep', lambda s: real_sleep(min(s, 0.05))):
            found = RestartRunner._await_minimap(SimpleNamespace(capture=capture), seconds=5)
        thread.join(2)
        self.assertEqual(dropped, [True])
        self.assertTrue(found)

    def test_finder_limits_are_sane(self):
        self.assertGreaterEqual(minimap_finder.WHITE_MIN, 200)


if __name__ == '__main__':
    unittest.main()
