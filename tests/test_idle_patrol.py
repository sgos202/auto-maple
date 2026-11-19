"""Edge loot jumps (hay bales, msg-boc.png), idle variety, and the chat watch."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from src.modules import bot as bot_module
from src.modules import chat_watch as chat_module
from tests.test_sweep import FakeTracker, _bot


class Tracker(FakeTracker):
    def __init__(self, edge=False, variety=False, **kw):
        super().__init__(sweep=False, **kw)
        self.edge, self.variety = edge, variety

    def edge_loot_enabled(self):
        return self.edge

    def idle_variety_enabled(self):
        return self.variety


class Harness:
    def __init__(self, **options):
        self.now = 100.0
        self.keys = []
        self.presses = []
        self.taps = []
        self.tracker = Tracker(clock=lambda: self.now, **options)
        self.bot = _bot()
        self.bot._tap_loot = lambda key, count=1: self.taps.append((key, count))
        self.bot._release_mob_attack = lambda: None
        self.position = [0.5, 0.5]
        self.config = SimpleNamespace(
            capture=SimpleNamespace(mob_tracker=self.tracker,
                                    loot_tracker=SimpleNamespace(LOOT_KEY='z')),
            enabled=True, player_pos=self.position, player_pos_time=None)
        self._patches = [
            patch.object(bot_module, 'config', self.config),
            patch.object(bot_module, 'key_down', lambda key: self.keys.append(('down', key))),
            patch.object(bot_module, 'key_up', lambda key: self.keys.append(('up', key))),
            patch.object(bot_module, 'press', lambda key, n=1, **k: self.presses.append(key)),
            patch.object(bot_module.time, 'monotonic', lambda: self.now),
            patch.object(bot_module.time, 'sleep', lambda seconds: None),
            patch('builtins.print', lambda *a, **k: None),
        ]

    def run(self, seconds, step=0.0, wall=None, tick=0.02):
        """Patrol for SECONDS, the character moving STEP per tick, stopped by
        a wall at minimap x WALL (on the right)."""
        end = self.now + seconds
        while self.now < end:
            if self.bot._auto_attack_movement == 'right':
                self.position[0] += step
            elif self.bot._auto_attack_movement == 'left':
                self.position[0] -= step
            if wall is not None:
                self.position[0] = min(wall, self.position[0])
            self.bot._patrol_for_mob()
            self.now += tick

    def __enter__(self):
        for item in self._patches:
            item.start()
        return self

    def __exit__(self, *exc):
        for item in reversed(self._patches):
            item.stop()


class EdgeLootTests(unittest.TestCase):
    def test_jumps_for_loot_at_the_end_then_turns(self):
        with Harness(edge=True) as h:
            h.run(4.0, step=0.002, wall=0.6)
            self.assertIn('alt', h.presses)
            self.assertTrue(h.taps and all(key == 'z' for key, _ in h.taps))
            self.assertEqual(h.bot._patrol_direction, 'left')

    def test_without_the_option_it_just_turns(self):
        with Harness(edge=False) as h:
            h.run(4.0, step=0.002, wall=0.6)
            self.assertNotIn('alt', h.presses)
            self.assertEqual(h.taps, [])
            self.assertEqual(h.bot._patrol_direction, 'left')

    def test_same_end_not_jumped_again_right_away(self):
        with Harness(edge=True) as h:
            h.run(4.0, step=0.002, wall=0.6)
            first = h.presses.count('alt')
            h.bot._patrol_direction = 'right'       # pushed back at it at once
            h.run(3.0, step=0.002, wall=0.6)
            self.assertEqual(h.presses.count('alt'), first)


class IdleVarietyTests(unittest.TestCase):
    def test_hops_and_stops_while_patrolling(self):
        with Harness(variety=True) as h:
            h.run(60.0, step=0.0002)
            self.assertGreaterEqual(h.presses.count('alt'), 4)
            stops = [k for k in h.keys if k[0] == 'up']
            self.assertTrue(stops)
            # a stop is never taken for a wall
            self.assertEqual(h.bot.__dict__.get('_walls', {}), {})

    def test_off_means_the_old_patrol(self):
        with Harness(variety=False) as h:
            h.run(10.0, step=0.0002)
            self.assertEqual(h.presses, [])

    def test_no_hop_next_to_a_wall(self):
        with Harness(variety=True) as h:
            h.bot._walls = {'right': 0.51}
            h.bot._patrol_direction = 'right'
            h.bot._idle_hop_due = 0
            h.bot._idle_hop('right', 0.5, h.now)
            self.assertEqual(h.presses, [])

    def test_long_quiet_walk_jumps_more(self):
        with Harness(variety=True) as h, patch.object(bot_module.random, 'random', lambda: 0.99):
            h.run(15.0, step=0.0002)            # before the quiet threshold
            early = h.presses.count('alt')
            h.run(15.0, step=0.0002)            # quiet: hops every 1.2-4 s
            self.assertGreater(h.presses.count('alt') - early, early)

    def test_quiet_walk_can_hold_the_jump_key(self):
        with Harness(variety=True) as h:
            h.bot._quiet_since = h.now - 60
            h.bot._idle_hop_due = 0
            with patch.object(bot_module.random, 'random', lambda: 0.0):
                h.bot._idle_hop('right', 0.5, h.now)
            self.assertIn(('down', 'alt'), h.keys)
            self.assertIn(('up', 'alt'), h.keys)
            self.assertEqual(h.presses, [])

    def test_no_held_jump_without_room(self):
        with Harness(variety=True) as h:
            h.bot._quiet_since = h.now - 60
            h.bot._walls = {'right': 0.55}      # room for a hop, not a held jump
            h.bot._idle_hop_due = 0
            with patch.object(bot_module.random, 'random', lambda: 0.0):
                h.bot._idle_hop('right', 0.5, h.now)
            self.assertNotIn(('down', 'alt'), h.keys)
            self.assertEqual(h.presses, ['alt'])

    def test_fight_resets_the_quiet_clock(self):
        with Harness(variety=True) as h:
            h.run(20.0, step=0.0002)
            h.now += 5.0                        # a fight had the controls
            h.run(0.1, step=0.0002)
            self.assertLess(h.now - h.bot._quiet_since, 1.0)


def _frame_with_line(text_rows):
    """A dark game picture with the chat box (All bar + white field) and white
    text lines in the log at TEXT_ROWS: [(y, x, width)]."""
    frame = np.full((600, 800, 3), 40, np.uint8)
    frame[550:562, 100:160] = chat_module.ALL_BAR_BGR
    frame[548:564, 162:500] = 254
    for y, x, width in text_rows:
        for i in range(0, width, 6):
            frame[y:y + 7, x + i:x + i + 3] = 250
            frame[y + 2:y + 4, x + i + 3:x + i + 5] = 250
    return frame


class ChatWatchTests(unittest.TestCase):
    def test_finds_the_say_line_in_the_screenshot(self):
        frame = cv2.imread('assets/msg-boc.png')
        bar = chat_module.find_all_bar(frame)
        lines = chat_module.chat_lines(frame, chat_module.log_region(frame, bar))
        self.assertEqual([kind for kind, _, _ in lines], ['say'])

    def test_megaphones_and_notices_are_ignored(self):
        for name in ('assets/henny.png', 'assets/perion_street_corner.png'):
            frame = cv2.imread(name)
            bar = chat_module.find_all_bar(frame)
            self.assertIsNotNone(bar, name)
            self.assertEqual(chat_module.chat_lines(frame, chat_module.log_region(frame, bar)), [])

    def _watch(self, frames):
        sent = []
        clock = [0.0]
        watch = chat_module.ChatWatch(
            frame_source=lambda: frames[0], send=lambda text, photo: sent.append(text),
            wants=lambda: True, running=lambda: True, clock=lambda: clock[0])
        return watch, sent, clock

    def test_alerts_only_on_a_new_line_not_a_scrolled_one(self):
        old = _frame_with_line([(530, 110, 120)])
        frames = [old]
        watch, sent, clock = self._watch(frames)
        watch.check()                                   # what was there: remembered
        self.assertEqual(sent, [])
        frames[0] = _frame_with_line([(515, 110, 120)])     # same line pushed up
        clock[0] += 30
        watch.check()
        self.assertEqual(sent, [])
        frames[0] = _frame_with_line([(500, 110, 120), (530, 110, 60)])   # someone talks
        clock[0] += 30
        watch.check()
        self.assertEqual(len(sent), 1)


if __name__ == '__main__':
    unittest.main()
