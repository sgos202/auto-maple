"""Sweep patrol: long maps with ramps (assets/perion_street_corner.png, stumps)."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from src.modules import bot as bot_module
from src.modules import mob_tracker as tracker_module
from src.modules.bot import Bot

SCREENSHOT = 'assets/perion_street_corner.png'
BADGE = (511, 507, 565, 521)            # the "EIDIABLO" name tag
DARK_STUMP = (554, 521, 616, 576)       # 47 px right of the character
STUMP = (744, 521, 792, 576)            # 230 px right; another stands at 180
# (In the screenshot the character hangs on a rope above the ground: the
# stumps' middles are this far below the name tag's.)
ALIGNMENT = 35


def _tracker(sweep, direction='right'):
    frame = cv2.imread(SCREENSHOT)      # samples always from the untouched picture
    tracker = tracker_module.MobTracker()
    tracker.save_config = lambda: None
    x1, y1, x2, y2 = BADGE
    tracker._player_template = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    tracker._mob_templates = [
        (frame[y1:y2, x1:x2].copy(), ALIGNMENT) for x1, y1, x2, y2 in (DARK_STUMP, STUMP)]
    tracker._mask_cache = {}
    tracker.config.update({
        'Enabled': True, 'Range': 200, 'Threshold': 78, 'Enemy count': 2,
        'Enemy alignments': f'{ALIGNMENT},{ALIGNMENT}', 'Enemy speeds': 'normal,normal',
        'Focus configured': False, 'Focused enemies': '', 'Group skill': False,
        'Magic missile': False, 'Sweep patrol': sweep,
    })
    tracker._ignored_spots = []
    tracker.set_sweep_direction(direction)
    return tracker


def _scan(tracker, frame, scans=6):
    # (a second or so of the same picture at 30 frames a second: a still
    # picture never shows an HP bar, so a longer look ends in "scenery")
    clock = [1000.0]
    with patch.object(tracker_module, 'is_game_focused', return_value=True),             patch.object(tracker_module.time, 'monotonic', lambda: clock[0]):
        tracker.statuses = set()
        for _ in range(scans * 6):
            clock[0] += 1 / 30.0
            tracker.process_frame(frame, True)
            tracker.statuses.add(tracker.status())
    return tracker.target()


NEAR_COLUMNS = (566, 625)               # the dark stump (clear of the name tag)
FAR_COLUMNS = (686, 800)                # the two stumps further right


def _ramp(frame, slope):
    """The picture with each stump moved up SLOPE px per px it stands to the
    right of the character (down for a negative slope), as on a ramp."""
    pivot = (BADGE[0] + BADGE[2]) // 2
    out = frame.copy()
    for x1, x2 in (NEAR_COLUMNS, FAR_COLUMNS):
        rise = int(round(slope * ((x1 + x2) / 2 - pivot)))
        out[:, x1:x2] = np.roll(frame[:, x1:x2], -rise, axis=0)
    return out


def _without_the_near_stump(frame):
    out = frame.copy()
    x1, x2 = NEAR_COLUMNS
    out[:, x1:x2] = frame[:, x1 + 64:x2 + 64]       # rock wall instead
    return out


class SlopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = cv2.imread(SCREENSHOT)

    def test_stumps_on_flat_ground_are_found_either_way(self):
        for sweep in (False, True):
            target = _scan(_tracker(sweep), self.frame)
            self.assertIsNotNone(target, sweep)
            self.assertTrue(30 <= target[0] <= 65, target)     # the dark stump
            self.assertTrue(target[1], target)                 # in swing range

    def test_a_stump_just_down_a_ramp_is_swung_at_only_with_the_sweep_on(self):
        frame = _ramp(self.frame, -0.45)
        # (the near one is 47 px away: ~21 px lower, more than the 16 px a
        # flat map allows before a swing)
        off = _scan(_tracker(False), frame)
        self.assertFalse(off is not None and off[1], off)
        on = _scan(_tracker(True), frame)
        self.assertTrue(on is not None and 30 <= on[0] <= 65 and on[1], on)

    def test_stumps_up_a_ramp_are_walked_to_not_taken_for_a_ledge(self):
        frame = _without_the_near_stump(_ramp(self.frame, 0.3))
        # (the two at 180 / 230 px stand ~55-70 px higher)
        self.assertIsNone(_scan(_tracker(False), frame))
        target = _scan(_tracker(True), frame)
        self.assertIsNotNone(target)
        self.assertTrue(150 <= target[0] <= 250 and not target[1], target)

    def test_far_stumps_down_a_ramp_are_seen_at_all(self):
        frame = _without_the_near_stump(_ramp(self.frame, -0.3))
        self.assertIsNone(_scan(_tracker(False), frame))    # ~55 px below: unseen
        target = _scan(_tracker(True), frame)
        self.assertIsNotNone(target)
        self.assertTrue(150 <= target[0] <= 250, target)

    def test_enemies_well_behind_the_sweep_wait_for_the_way_back(self):
        frame = _without_the_near_stump(self.frame)     # only the two at 180 / 230 px
        self.assertIsNotNone(_scan(_tracker(True, 'right'), frame))
        tracker = _tracker(True, 'left')
        self.assertIsNone(_scan(tracker, frame))
        self.assertTrue(any('behind the sweep' in status for status in tracker.statuses),
                        tracker.statuses)
        # ...but one close behind still bites, so it is fought
        near = _scan(_tracker(True, 'left'), self.frame)
        self.assertTrue(near is not None and 30 <= near[0] <= 65, near)

    def test_the_option_is_off_unless_ticked_and_never_with_jumpers(self):
        tracker = _tracker('')          # a settings file from before
        self.assertFalse(tracker.sweep_patrol_enabled())
        tracker.config['Sweep patrol'] = True
        self.assertTrue(tracker.sweep_patrol_enabled())
        tracker.config['Enemy speeds'] = 'jumper,normal'
        self.assertFalse(tracker.sweep_patrol_enabled())


def _bot():
    bot = Bot.__new__(Bot)
    bot._mob_attack_keys = ()
    bot._auto_attack_movement = None
    bot._patrol_direction = None
    bot._patrol_until = 0
    bot.command_book = SimpleNamespace(
        dict={'cruelstab': object()},
        module=SimpleNamespace(Key=SimpleNamespace(JUMP='alt', CRUEL_STAB='ctrl')))
    return bot


class FakeTracker:
    def __init__(self, sweep=True, offset=80, closing=0.0, clock=None):
        self.sweep, self.offset, self.closing = sweep, offset, closing
        self.clock = clock or (lambda: 0.0)
        self.directions = []

    def sweep_patrol_enabled(self):
        return self.sweep

    def level_farming_enabled(self):
        return False

    def set_sweep_direction(self, direction):
        self.directions.append(direction)

    def target_details(self):
        return {'time': self.clock(), 'offset': self.offset, 'on_level': True,
                'closing': self.closing}

    def status(self):
        return ''

    def seconds_since_bar(self):
        return 0.0


class Harness:
    """A bot with a fake clock, keyboard and tracker."""

    def __init__(self, **tracker):
        self.now = 100.0
        self.keys = []
        self.tracker = FakeTracker(clock=lambda: self.now, **tracker)
        self.bot = _bot()
        self.position = [0.5, 0.5]
        hand = SimpleNamespace(key=None, hold=lambda key: None, release=lambda: None)
        self.bot._attack_hand = lambda: hand
        self.config = SimpleNamespace(
            capture=SimpleNamespace(mob_tracker=self.tracker), enabled=True,
            player_pos=self.position, player_pos_time=None)
        self._patches = [
            patch.object(bot_module, 'config', self.config),
            patch.object(bot_module, 'key_down', lambda key: self.keys.append(('down', key))),
            patch.object(bot_module, 'key_up', lambda key: self.keys.append(('up', key))),
            patch.object(bot_module.time, 'monotonic', lambda: self.now),
            patch.object(bot_module.time, 'sleep', lambda seconds: None),
            patch('builtins.print', lambda *a, **k: None),
        ]

    def __enter__(self):
        for item in self._patches:
            item.start()
        return self

    def __exit__(self, *exc):
        for item in reversed(self._patches):
            item.stop()


class StepInTests(unittest.TestCase):
    def _hold(self, h, seconds, tick=0.02):
        end = h.now + seconds
        while h.now < end:
            h.bot._hold_attack('right', 'ctrl', h.tracker)
            h.now += tick

    def test_knocked_back_past_the_reach_steps_in_at_once(self):
        with Harness(offset=95) as h:
            h.bot._facing = 'right'
            self._hold(h, 0.1)
            self.assertIn(('down', 'right'), h.keys)
            self._hold(h, 0.5)
            # (95 - 54) px at 125 px/s: the arrow comes up again ~0.33 s later
            self.assertIn(('up', 'right'), h.keys)
            self.assertEqual(h.bot._mob_attack_keys, ('ctrl',))     # attack never let go

    def test_in_reach_it_steps_only_after_a_few_swings(self):
        with Harness(offset=64) as h:
            h.bot._facing = 'right'
            self._hold(h, 1.5)
            self.assertNotIn(('down', 'right'), h.keys)
            self._hold(h, 0.6)
            self.assertEqual(h.keys.count(('down', 'right')), 1)
            self._hold(h, 1.0)                  # not again straight away
            self.assertEqual(h.keys.count(('down', 'right')), 1)
            self._hold(h, 1.5)                  # ...but after 2-3 more swings
            self.assertEqual(h.keys.count(('down', 'right')), 2)

    def test_no_step_when_close_coming_or_not_a_sweep_map(self):
        for kwargs in ({'offset': 50}, {'offset': 95, 'closing': 40.0},
                       {'offset': 95, 'sweep': False}):
            with Harness(**kwargs) as h:
                h.bot._facing = 'right'
                self._hold(h, 3.0)
                self.assertNotIn(('down', 'right'), h.keys, kwargs)

    def test_letting_go_of_the_attack_ends_the_step(self):
        with Harness(offset=110) as h:
            h.bot._facing = 'right'
            self._hold(h, 0.06)
            self.assertIn(('down', 'right'), h.keys)
            h.bot._release_mob_attack()
            self.assertIn(('up', 'right'), h.keys)
            self.assertIsNone(h.bot._step_in)


class WalkTests(unittest.TestCase):
    def test_a_walk_right_after_a_swing_is_not_a_wall(self):
        """11:58 log: CTRL up, D down, "Blocked walking right" 0.8 s later."""
        with Harness() as h:
            h.bot._mob_attack_keys = ('ctrl',)
            h.bot._release_mob_attack()
            blocked = []
            for _ in range(45):                 # 0.9 s, the position never changes
                blocked.append(h.bot._pursuit_blocked('right', 0.124))
                h.now += 0.02
            self.assertFalse(any(blocked))
            for _ in range(45):                 # still not moving after that: a wall
                blocked.append(h.bot._pursuit_blocked('right', 0.124))
                h.now += 0.02
            self.assertTrue(blocked[-1])

    def test_the_sweep_crosses_the_whole_map_and_turns_only_at_its_end(self):
        with Harness() as h:
            bot = h.bot
            for _ in range(int(30 / 0.02)):     # 30 s: far longer than PATROL_MAX_LEG
                h.position[0] = min(0.97, h.position[0] + 0.0004)
                bot._patrol_for_mob()
                h.now += 0.02
                if bot._patrol_direction == 'left':
                    break
            self.assertEqual(bot._patrol_direction, 'left')
            self.assertGreaterEqual(h.position[0], 0.97)        # only at the end
            self.assertEqual(h.tracker.directions[-1], 'left')

    def test_without_the_option_the_timed_turn_is_as_before(self):
        with Harness(sweep=False) as h:
            bot = h.bot
            for _ in range(int(13 / 0.02)):
                h.position[0] += 0.0002
                bot._patrol_for_mob()
                h.now += 0.02
            self.assertEqual(bot._patrol_direction, 'left')
            self.assertLess(h.position[0], 0.7)

    def test_no_instant_about_turn_on_a_sweep(self):
        with Harness() as h:
            h.bot._set_auto_attack_movement('right')
            h.now += 0.1
            h.bot._set_auto_attack_movement('left')
            self.assertIsNone(h.bot._auto_attack_movement)      # stands instead
            h.now += 0.5
            h.bot._set_auto_attack_movement('left')
            self.assertEqual(h.bot._auto_attack_movement, 'left')

    def test_drop_spots_behind_the_sweep_are_left(self):
        with Harness() as h:
            given_up = []
            h.bot._loot_planner = SimpleNamespace(
                collect_direction=lambda now, x, level: 'left',
                give_up=lambda now, x, level, direction: given_up.append(direction))
            h.bot._loot_enabled = lambda: True
            h.bot._map_limit = lambda direction: None
            h.bot._patrol_direction = 'right'
            self.assertFalse(h.bot._collect_loot(0.5))
            self.assertEqual(given_up, ['left'])
            h.bot._patrol_direction = 'left'
            self.assertTrue(h.bot._collect_loot(0.5))


if __name__ == '__main__':
    unittest.main()
