import os
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.modules import health
from src.modules.health import (
    ManaPotionMonitor, estimate_mp_percent, find_hp_bar_region, find_mp_bar_region)


def hud(hp=0.7, mp=0.5, width=104):
    """A bottom HUD like the game's: HP (red) then MP (blue) bars on grey tracks."""
    frame = np.full((640, 1366, 3), 40, np.uint8)     # the game window's width
    y1, y2 = 596, 604
    for x1, fill, colour in ((506, hp, (0, 0, 200)), (614, mp, (220, 140, 0))):
        frame[y1:y2, x1:x1 + width] = (190, 190, 190)          # the track
        filled = int(round(width * fill))
        if filled:
            frame[y1:y2, x1:x1 + filled] = colour
        for tick in range(x1 + filled + 13, x1 + width, 13):   # tick marks on the track
            frame[y1:y2, tick] = (120, 120, 120)
    return frame


class MonitorHarness:
    def __init__(self, enabled=True, key='t', threshold=20):
        with patch.object(health.ManaPotionMonitor, 'save_config', lambda self: None, create=True):
            self.monitor = ManaPotionMonitor.__new__(ManaPotionMonitor)
            ManaPotionMonitor.__init__(self.monitor)
        self.monitor.config.update({'Enabled': enabled, 'Key': key, 'Threshold': threshold})
        self.pressed = []
        self.clock = 100.0

    def frame(self, frame):
        with patch.object(health, 'press', lambda key, n, **k: self.pressed.append(key)), \
                patch.object(health, 'is_game_focused', return_value=True), \
                patch.object(health.time, 'monotonic', lambda: self.clock), \
                patch('builtins.print'):
            self.monitor.process_frame(frame, find_hp_bar_region(frame))
        self.clock += 0.1
        return self.monitor.current_percent()


class ManaBarTests(unittest.TestCase):
    def test_reads_the_mp_bar_next_to_hp(self):
        frame = hud(mp=0.5)
        region = find_mp_bar_region(frame, find_hp_bar_region(frame))
        self.assertIsNotNone(region)
        self.assertAlmostEqual(estimate_mp_percent(frame, region), 50, delta=3)

    def test_defaults_are_20_percent_and_t(self):
        self.assertEqual(ManaPotionMonitor.DEFAULT_CONFIG['Threshold'], 20)
        self.assertEqual(ManaPotionMonitor.DEFAULT_CONFIG['Key'], 't')
        self.assertTrue(ManaPotionMonitor.DEFAULT_CONFIG['Enabled'])

    def test_uses_t_at_20_percent_not_above(self):
        harness = MonitorHarness()
        harness.frame(hud(mp=0.5))
        harness.frame(hud(mp=0.25))
        self.assertEqual(harness.pressed, [])
        harness.frame(hud(mp=0.18))
        self.assertEqual(harness.pressed, ['t'])

    def test_repeats_after_the_cooldown_while_still_low(self):
        harness = MonitorHarness()
        for _ in range(20):         # 2 s at 15% MP
            harness.frame(hud(mp=0.15))
        self.assertEqual(harness.pressed, ['t', 't'])

    def test_empty_mp_bar_still_counts(self):
        harness = MonitorHarness()
        harness.frame(hud(mp=0.5))              # learns where the bar is
        self.assertEqual(harness.frame(hud(mp=0.0)), 0.0)
        self.assertEqual(harness.pressed, ['t'])

    def test_off_means_no_potions(self):
        harness = MonitorHarness(enabled=False)
        harness.frame(hud(mp=0.1))
        self.assertEqual(harness.pressed, [])

    @unittest.skipUnless(os.path.exists(os.path.join('assets', 'spawn_issue.mp4')),
                         'recording not available')
    def test_reads_the_recording(self):
        # spawn_issue.mp4: MP [185/371] on the HUD (game at x >= 1076)
        capture = cv2.VideoCapture(os.path.join('assets', 'spawn_issue.mp4'))
        capture.set(cv2.CAP_PROP_POS_FRAMES, 30)
        ok, frame = capture.read()
        frame = frame[:, 1076:]
        region = find_mp_bar_region(frame, find_hp_bar_region(frame))
        self.assertAlmostEqual(estimate_mp_percent(frame, region), 50, delta=5)


if __name__ == '__main__':
    unittest.main()
