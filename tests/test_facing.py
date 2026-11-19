"""Facing read from the head, and scenery by portals (attack1.mp4 / attack2.mp4)."""

import os
import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from src.detection.facing import FacingReader
from src.modules import bot as bot_module
from src.modules import mob_tracker as tracker_module
from src.modules.bot import Bot

PROFILE = os.path.join('.settings', 'profiles', 'henesys-hg1-eidiablo')
GAME_X = (1216, 2240)           # the game window inside the two recordings


def _head(seed=3):
    """A made-up head, 52 x 50, clearly different left to right: 'hair' over
    the top and down the right side, 'face' with two dark eyes on the left."""
    rng = np.random.RandomState(seed)
    head = np.zeros((50, 52, 3), np.uint8)
    head[:] = (60, 110, 170)                        # hair
    head[18:46, 4:34] = (190, 215, 250)             # face, toward the left
    head[26:31, 8:13] = head[26:31, 22:27] = (20, 20, 20)       # eyes
    head[8:14, 10:44] = (200, 60, 30)               # a headband
    return np.clip(head.astype(np.int16) + rng.randint(-6, 7, head.shape), 0, 255).astype(np.uint8)


def _scene(head, facing, seed, shift=(0, 0)):
    """A frame with the head above a name tag at (150, 200), on random scenery."""
    rng = np.random.RandomState(seed)
    frame = rng.randint(0, 255, (300, 300, 3)).astype(np.uint8)
    frame = cv2.GaussianBlur(frame, (9, 9), 0)
    picture = head if facing == 'left' else head[:, ::-1]
    x, y = 150 - 26 + shift[0], 200 - 80 + shift[1]
    frame[y:y + 50, x:x + 52] = picture
    return frame


class FacingReaderTests(unittest.TestCase):
    CENTRE = (150, 200)

    def _trained(self, head):
        reader = FacingReader()
        for i, facing in enumerate(('left', 'left', 'right', 'left', 'right')):
            reader.learn(_scene(head, facing, i), self.CENTRE, facing, now=i * 1.0)
        return reader

    def test_says_nothing_before_it_has_seen_the_character_walk(self):
        reader = FacingReader()
        self.assertIsNone(reader.read(_scene(_head(), 'left', 1), self.CENTRE, 0.0))
        reader.learn(_scene(_head(), 'left', 1), self.CENTRE, 'left', 0.0)
        self.assertFalse(reader.ready)      # one vote names no side yet

    def test_reads_either_way_for_whatever_head_it_was_shown(self):
        for seed in (3, 11):                # two different "characters"
            head = _head(seed) if seed == 3 else np.ascontiguousarray(_head(seed)[::-1])
            reader = self._trained(head)
            self.assertTrue(reader.ready)
            for facing in ('left', 'right'):
                for shift in ((0, 0), (7, -3), (-9, 4)):        # leaning into a swing
                    frame = _scene(head, facing, 50 + shift[0], shift)
                    self.assertEqual(reader.read(frame, self.CENTRE, 10.0), facing,
                                     (seed, facing, shift))

    def test_a_vote_at_a_bad_moment_is_outvoted(self):
        head = _head()
        reader = FacingReader()
        labels = [('left', 'left'), ('left', 'left'), ('right', 'left'),    # wrong label
                  ('right', 'right'), ('left', 'left'), ('right', 'right')]
        for i, (shown, said) in enumerate(labels):
            reader.learn(_scene(head, shown, i), self.CENTRE, said, now=i * 1.0)
        self.assertEqual(reader.read(_scene(head, 'right', 77), self.CENTRE, 9.0), 'right')
        self.assertEqual(reader.read(_scene(head, 'left', 78), self.CENTRE, 9.1), 'left')

    def test_steady_needs_two_readings_in_a_row_and_a_fresh_one(self):
        head = _head()
        reader = self._trained(head)
        reader.read(_scene(head, 'left', 60), self.CENTRE, 10.0)
        reader.read(_scene(head, 'right', 61), self.CENTRE, 10.1)
        self.assertIsNone(reader.steady(10.1))
        reader.read(_scene(head, 'right', 62), self.CENTRE, 10.2)
        self.assertEqual(reader.steady(10.2), 'right')
        self.assertIsNone(reader.steady(11.5))      # stale

    def test_something_else_on_screen_is_not_a_reading(self):
        reader = self._trained(_head())
        rng = np.random.RandomState(5)
        noise = cv2.GaussianBlur(rng.randint(0, 255, (300, 300, 3)).astype(np.uint8), (9, 9), 0)
        self.assertIsNone(reader.read(noise, self.CENTRE, 10.0))


def _clip_tracker():
    """A tracker with the samples the two clips were recorded with."""
    settings = os.path.join(PROFILE, 'mob_tracker')
    if not os.path.isfile(settings):
        raise unittest.SkipTest('profile henesys-hg1-eidiablo not saved here')
    with open(settings, 'rb') as file:
        loaded = pickle.load(file)
    tracker = tracker_module.MobTracker()
    tracker._player_template = cv2.imread(os.path.join(PROFILE, 'mob_player_badge.png'), 0)
    alignments = [int(v) for v in loaded['Enemy alignments'].split(',')]
    tracker._mob_templates = [
        (cv2.imread(os.path.join(PROFILE, os.path.basename(
            tracker._mob_template_path(index)))), alignment)
        for index, alignment in enumerate(alignments)]
    tracker._mask_cache = {}
    tracker.config = dict(tracker.DEFAULT_CONFIG, **loaded)
    tracker.config.update({'Enabled': True, 'Magic missile': False, 'Sweep patrol': False})
    tracker._ignored_spots = []
    return tracker


def _replay(video_path, seconds, each_scan):
    from src.common import config, utils
    dot = cv2.imread(os.path.join('assets', 'player_template.png'), 0)
    clock = [0.0]
    video = cv2.VideoCapture(video_path)
    saved = (config.player_pos, getattr(config, 'player_pos_time', None))
    try:
        with patch.object(tracker_module, 'is_game_focused', return_value=True), \
                patch.object(tracker_module.time, 'monotonic', lambda: clock[0]), \
                patch.object(tracker_module.MobTracker, 'save_config', lambda self: None):
            tracker = _clip_tracker()
            index = 0
            while True:
                ok, frame = video.read()
                if not ok or clock[0] > seconds:
                    break
                clock[0] = index / 30.0
                index += 1
                game = frame[:, GAME_X[0]:GAME_X[1]]
                minimap = game[60:200, 5:205]       # as live: the marker's map spot
                found = utils.multi_match(minimap, dot, threshold=0.8)
                if found:
                    config.player_pos = (found[0][0] / minimap.shape[1],
                                         found[0][1] / minimap.shape[0])
                    config.player_pos_time = clock[0]
                tracker.process_frame(game, True)
                if tracker._frame_count % 3 == 0:
                    each_scan(clock[0], tracker)
    finally:
        config.player_pos, config.player_pos_time = saved
        video.release()


class PortalClipTests(unittest.TestCase):
    ATTACK1 = os.path.join('assets', 'attack1.mp4')
    ATTACK2 = os.path.join('assets', 'attack2.mp4')

    @unittest.skipUnless(os.path.exists(ATTACK1), 'clip not available')
    def test_after_the_kill_in_the_portal_the_swinging_ends(self):
        """attack1.mp4: the red snail dies at ~2.7 s; the lock stayed on the
        portal behind it and the character swung at nothing until 8.3 s."""
        swings, far = [], []

        def scan(at, tracker):
            target = tracker.target()
            if target is not None and target[1] and abs(target[0]) < 100:
                swings.append(at)
            if target is not None and not target[1] and target[0] > 300:
                far.append(at)

        _replay(self.ATTACK1, 9.0, scan)
        self.assertTrue(any(at < 2.7 for at in swings), swings)    # the real fight
        self.assertFalse([at for at in swings if 4.0 <= at <= 9.0])
        # ...and the snails coming from the right are the target instead
        self.assertTrue(any(at < 5.0 for at in far), far)

    @unittest.skipUnless(os.path.exists(ATTACK2), 'clip not available')
    def test_by_the_edge_portal_the_real_enemies_further_off_are_the_target(self):
        """attack2.mp4: at the right edge the lock went from the portal to the
        decor mushrooms (63-67 %) while snails and shrooms stood 250-700 px
        to the left, never looked at."""
        scenery, real = [], []

        def scan(at, tracker):
            target = tracker.target()
            if at < 2.0 or at > 7.0 or target is None:
                return
            # (on screen the portal and the decor stand right of x=680, the
            # snails and shrooms left of it)
            where = tracker._player_position[0] + target[0]
            (real if where < 680 else scenery).append((at, target))

        _replay(self.ATTACK2, 7.2, scan)
        self.assertGreater(len(real), 25, real)
        self.assertFalse(scenery)


def _bot():
    bot = Bot.__new__(Bot)
    bot._mob_attack_keys = ()
    bot._auto_attack_movement = None
    bot._patrol_direction = None
    return bot


class FakeTracker:
    def __init__(self, facing=None, offsets=()):
        self._facing, self.offsets = facing, list(offsets)
        self.notes = []

    def facing(self):
        return self._facing

    def facing_seen(self):
        return self._facing

    def enemy_offsets(self):
        return self.offsets

    def note_walking(self, direction):
        self.notes.append(direction)

    def sweep_patrol_enabled(self):
        return False

    def level_farming_enabled(self):
        return False

    def status(self):
        return ''

    def seconds_since_bar(self):
        return 0.0


class Harness:
    def __init__(self, tracker):
        self.now = 50.0
        self.keys, self.lines = [], []
        self.tracker = tracker
        self.bot = _bot()
        hand = SimpleNamespace(key=None, hold=lambda key: self.keys.append(('hold', key)),
                               release=lambda: None)
        self.bot._attack_hand = lambda: hand
        config = SimpleNamespace(capture=SimpleNamespace(mob_tracker=tracker), enabled=True)

        def sleep(seconds):
            self.now += seconds

        self._patches = [
            patch.object(bot_module, 'config', config),
            patch.object(bot_module, 'key_down', lambda key: self.keys.append(('down', key))),
            patch.object(bot_module, 'key_up', lambda key: self.keys.append(('up', key))),
            patch.object(bot_module.time, 'monotonic', lambda: self.now),
            patch.object(bot_module.time, 'sleep', sleep),
            patch('builtins.print', lambda *a, **k: self.lines.append(' '.join(map(str, a)))),
        ]

    def __enter__(self):
        for item in self._patches:
            item.start()
        return self

    def __exit__(self, *exc):
        for item in reversed(self._patches):
            item.stop()


class BotFacingTests(unittest.TestCase):
    def test_seen_facing_the_wrong_way_it_lets_go_and_turns_again(self):
        """attack1.mp4: believed left, was swinging right."""
        tracker = FakeTracker(facing='right')
        with Harness(tracker) as h:
            h.bot._facing = 'left'
            h.bot._mob_attack_keys = ('ctrl',)

            def turned():
                return 'left' if h.now >= 50.2 else 'right'
            tracker.facing_seen = turned
            h.bot._hold_attack('left', 'ctrl', tracker)
            self.assertEqual([k for k in h.keys if k[1] == 'left'],
                             [('down', 'left'), ('up', 'left')])
            self.assertLess(h.now, 50.3)                    # let go once seen turned
            self.assertEqual(h.keys[-1], ('hold', 'ctrl'))  # ...and swings again
            self.assertTrue(any('turning it again' in line for line in h.lines))
            # not again straight away
            before = len(h.keys)
            h.bot._hold_attack('left', 'ctrl', tracker)
            self.assertEqual([k for k in h.keys[before:] if k[1] == 'left'], [])

    def test_facing_the_right_way_or_unread_nothing_is_touched(self):
        for facing in ('left', None):
            tracker = FakeTracker(facing=facing)
            with Harness(tracker) as h:
                h.bot._facing = 'left'
                h.bot._mob_attack_keys = ('ctrl',)
                h.bot._hold_attack('left', 'ctrl', tracker)
                self.assertEqual(h.keys, [('hold', 'ctrl')], facing)

    def test_the_log_says_which_way_it_faces_and_how_many_enemies_are_in_front(self):
        tracker = FakeTracker(facing='right', offsets=[64, 180, -120])
        with Harness(tracker) as h:
            h.bot._facing = 'right'
            h.bot._log_surroundings(tracker)
            self.assertEqual(len(h.lines), 1)
            self.assertIn('Facing right (seen from the head)', h.lines[0])
            self.assertIn('in front: 2 (nearest 64 px)', h.lines[0])
            self.assertIn('behind: 1 (nearest 120 px)', h.lines[0])
            h.now += 2.0
            h.bot._log_surroundings(tracker)            # nothing changed: no new line
            self.assertEqual(len(h.lines), 1)
            tracker.offsets = [-120]
            h.bot._log_surroundings(tracker)
            self.assertIn('in front: 0, behind: 1', h.lines[-1])
            # the head not read: it says so, and goes by the keys
            tracker._facing = None
            h.bot._facing = 'left'
            h.now += 2.0
            h.bot._log_surroundings(tracker)
            self.assertIn('Facing left (by the keys; head not read)', h.lines[-1])
            self.assertIn('in front: 1', h.lines[-1])

    def test_walking_teaches_the_reader_only_while_really_moving_that_way(self):
        tracker = FakeTracker()
        with Harness(tracker) as h:
            h.bot._auto_attack_movement = 'right'
            x = 0.5
            for _ in range(30):                 # 0.6 s walking right
                h.bot._note_walk(tracker, x)
                h.now += 0.02
                x += 0.001
            self.assertIn('right', tracker.notes)
            tracker.notes.clear()
            h.bot._walk_watch = None
            for _ in range(30):                 # arrow held, knocked back the other way
                h.bot._note_walk(tracker, x)
                h.now += 0.02
                x -= 0.001
            self.assertEqual(tracker.notes, [])


class DropFlipTests(unittest.TestCase):
    """attack2.mp4: left for the target, right for a "drop", every 0.5 s."""

    def _collect(self, h, drops, x=0.8):
        h.bot._visible_drops = lambda: drops
        h.bot._map_limit = lambda direction: None
        for name in ('_drop_banned', '_scenery_drop'):
            if name not in h.bot.__dict__:
                setattr(h.bot, name, lambda *a: False)
        moves = []
        h.bot._set_auto_attack_movement = moves.append
        result = h.bot._collect_visible_drop(x)
        return result, moves

    def test_a_drop_across_from_the_last_target_waits_a_moment(self):
        tracker = FakeTracker()
        with Harness(tracker) as h:
            bot_module.config.capture.loot_tracker = SimpleNamespace(LOOT_KEY='x')
            h.bot._last_target = ('left', h.now - 0.3)
            self.assertEqual(self._collect(h, [(120, 0, 0.9)]), (False, []))
            # the same side as the target, or later on: fine
            self.assertTrue(self._collect(h, [(-120, 0, 0.9)])[0])
            h.bot._drop_chase = None
            h.now += 2.0
            self.assertTrue(self._collect(h, [(120, 0, 0.9)])[0])

    def test_a_drop_the_walk_cannot_reach_twice_is_scenery(self):
        tracker = FakeTracker()
        with Harness(tracker) as h:
            h.bot._drop_past_wall('right', 0.97, 180, h.now)
            self.assertFalse(h.bot.__dict__.get('_scenery_drops'))
            h.now += 9.0
            h.bot._drop_past_wall('right', 0.972, 175, h.now)
            self.assertTrue(h.bot._scenery_drops)
            self.assertTrue(h.bot._scenery_drop(0.97, 178, h.now))
            self.assertTrue(any('scenery past the edge' in line for line in h.lines))


if __name__ == '__main__':
    unittest.main()
