import unittest
from unittest.mock import patch

import numpy as np

from src.modules import health


def hud(exp_fill):
    """A bottom HUD row like the game's: HP, MP and EXP bars on a dark strip
    (colours measured on violetta-20261120-100529-uncalibrated/00.jpg)."""
    frame = np.full((40, 400, 3), (51, 56, 60), np.uint8)       # dark HUD
    y1, y2 = 20, 31
    frame[y1:y2, 10:117] = (190, 190, 190)
    frame[y1:y2, 10:60] = (2, 8, 227)                          # HP red
    frame[y1:y2, 117:229] = (190, 190, 190)
    frame[y1:y2, 117:210] = (232, 156, 0)                       # MP blue
    frame[y1:y2, 229:230] = (60, 56, 51)                        # MP edge
    frame[y1:y2, 230:344] = (188, 190, 189)                     # EXP track
    frame[y1:y2, 230:230 + exp_fill] = (0, 229, 212)            # EXP yellow-green
    frame[y1:y2, 344:345] = (255, 253, 248)                     # white rim
    return frame, (10, y1, 117, y2)


class ExpBarTests(unittest.TestCase):
    def test_reads_the_fill_of_the_bar(self):
        frame, hp = hud(106)
        mp = health.find_mp_bar_region(frame, hp)
        region = health.find_exp_bar_region(frame, hp, mp)
        self.assertEqual(region, (230, 20, 344, 31))
        self.assertAlmostEqual(health.estimate_exp_percent(frame, region), 93.0, delta=0.5)

    def test_the_dark_hud_past_the_bar_is_not_track(self):
        frame, hp = hud(57)
        region = health.find_exp_bar_region(frame, hp, None)
        self.assertEqual(region[2], 344)
        self.assertAlmostEqual(health.estimate_exp_percent(frame, region), 50.0, delta=0.5)

    def test_no_hp_bar_no_reading(self):
        frame, _ = hud(50)
        self.assertIsNone(health.find_exp_bar_region(frame, None))


class ExpMonitorTests(unittest.TestCase):
    def feed(self, monitor, readings):
        clock = [1000.0]
        for seconds, fill in readings:
            clock[0] += seconds
            frame, hp = hud(fill)
            with patch('src.modules.health.time.monotonic', return_value=clock[0]):
                monitor.process_frame(frame, hp)

    def test_rate_and_time_to_level(self):
        monitor = health.ExpMonitor()
        self.feed(monitor, [(0, 57), (60, 63), (60, 68)])      # 50 % -> ~59.6 % in 2 min
        rate = monitor.rate_per_hour()
        self.assertGreater(rate, 200)
        with patch('src.modules.health.time.monotonic', return_value=1120.0):
            self.assertAlmostEqual(monitor.seconds_to_level(),
                                   (100 - monitor.current_percent()) * 3600 / rate, places=3)

    def test_rate_needs_a_minute(self):
        monitor = health.ExpMonitor()
        self.feed(monitor, [(0, 57), (20, 60)])
        self.assertIsNone(monitor.rate_per_hour())

    def test_level_up_counts_and_keeps_the_rate_positive(self):
        monitor = health.ExpMonitor()
        self.feed(monitor, [(0, 108), (60, 112), (60, 6)])     # ~95 % -> ~98 % -> ~5 %
        self.assertEqual(monitor.levels_gained(), 1)
        self.assertGreater(monitor.rate_per_hour(), 0)


if __name__ == '__main__':
    unittest.main()
