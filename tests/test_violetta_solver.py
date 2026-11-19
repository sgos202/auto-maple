"""
End-to-end tests for the Violetta solver on synthetic games rendered from the
game's own sprites: random window scale and position, random shuffles that
follow the game's rules, and colour-cycling stage lights over the characters.
"""

import random
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.detection import violetta_solver as vs
from src.detection import violetta_tracker as vt


SPRITES = vs.Sprites()


def _group(name):
    return [(cv2.imread(f'{vs.ASSET_DIR}/{name}.{i}.png', cv2.IMREAD_UNCHANGED),
             SPRITES.pivots[f'{name}.{i}']) for i in range(8)]


WALK = _group('lieDetector.npcCommon.move')
STAND = _group('lieDetector.npc.stand')
SPECIAL = _group('lieDetector.npcSpecial.stand')
BACKDROP = cv2.imread(f'{vs.ASSET_DIR}/lieDetector.back.png')


def _paste(canvas, sprite, pivot, anchor, scale):
    image = cv2.resize(sprite, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    x0 = int(round(anchor[0] - pivot[0] * scale))
    y0 = int(round(anchor[1] - pivot[1] * scale))
    h, w = image.shape[:2]
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(canvas.shape[1], x0 + w), min(canvas.shape[0], y0 + h)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    part = image[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    alpha = part[:, :, 3:4] / 255.0
    region = canvas[cy0:cy1, cx0:cx1]
    canvas[cy0:cy1, cx0:cx1] = (part[:, :, :3] * alpha + region * (1 - alpha)).astype(np.uint8)


class FakeGame:
    """Renders the frames of one game the way the real one plays it."""

    FPS = 30
    LIGHTS = ((255, 80, 200), (80, 255, 80), (255, 140, 60), (90, 120, 255))

    def __init__(self, seed, scale, origin, frame_size):
        self.rng = random.Random(seed)
        self.scale, self.origin, self.frame_size = scale, origin, frame_size
        self.slots = list(range(4))                 # character -> slot
        self.rng.shuffle(self.slots)
        self.special = self.rng.randrange(4)
        self.offsets = [(self.rng.uniform(-6, 6), self.rng.uniform(-3, 3)) for _ in range(4)]

    def _window(self, placements, lit, tick):
        canvas = np.full((318, 441, 3), 40, np.uint8)
        canvas[65:65 + BACKDROP.shape[0], 13:13 + BACKDROP.shape[1]] = BACKDROP
        for group, x, y, scale in sorted(placements, key=lambda p: p[3]):   # back first
            sprite, pivot = group[tick % len(group)]
            _paste(canvas, sprite, pivot, (x, y), scale)
        if lit:     # coloured stage light over everything, cycling twice a second
            colour = np.array(self.LIGHTS[(tick // (self.FPS // 2)) % len(self.LIGHTS)], np.float32)
            canvas = (canvas * 0.45 + colour[::-1] * 0.55).astype(np.uint8)
        return canvas

    def _frame(self, window):
        big = cv2.resize(window, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        frame = np.full((self.frame_size[1], self.frame_size[0], 3), 20, np.uint8)
        x, y = self.origin
        h = min(big.shape[0], frame.shape[0] - y)
        w = min(big.shape[1], frame.shape[1] - x)
        frame[y:y + h, x:x + w] = big[:h, :w]
        return frame

    def _standing(self, special_visible):
        out = []
        for c in range(4):
            group = SPECIAL if (special_visible and c == self.special) else STAND
            out.append((group, vs.slot_x(self.slots[c]) + self.offsets[c][0],
                        vs.STAND_PIVOT_Y + self.offsets[c][1], 1.0))
        return out

    def shuffle_paths(self):
        """One shuffle as the game does it: per-character spots (slot, row)."""
        standing = self.rng.randrange(4)
        moving = [c for c in range(4) if c != standing]
        while True:
            targets = [self.slots[c] for c in moving]
            self.rng.shuffle(targets)
            if all(t != self.slots[c] for t, c in zip(targets, moving)):
                break
        destination = dict(zip(moving, targets))
        destination[standing] = self.slots[standing]
        rows = list(vs.WALK_ROWS)
        self.rng.shuffle(rows)
        paths = []
        for c in range(4):
            start, end, row = self.slots[c], destination[c], rows[c]
            if c == standing:
                waypoints = [start, start, start]
            else:
                first = self.rng.choice([s for s in range(4) if s not in (start, end)])
                second = self.rng.choice([s for s in range(4) if s not in (first, end)])
                waypoints = [first, second, end]
            paths.append([(start, vs.STAND_ROW), (start, row)] +
                         [(w, row) for w in waypoints] + [(end, vs.STAND_ROW)])
        return paths

    def geometry(self, row):
        return vs.row_geometry(row)

    def frames(self, rounds=1, leg_frames=30, mark_rounds=False):
        tick = 0
        for _ in range(20):                         # the reveal
            yield self._frame(self._window(self._standing(True), False, tick))
            tick += 1
        for _ in range(15):                         # masks on, lights up
            yield self._frame(self._window(self._standing(False), True, tick))
            tick += 1
        for _ in range(rounds):
            paths = self.shuffle_paths()
            legs = len(paths[0]) - 1
            for step in range(legs * leg_frames):
                leg, part = divmod(step, leg_frames)
                part /= leg_frames
                placements = []
                for c in range(4):
                    (s0, r0), (s1, r1) = paths[c][leg], paths[c][leg + 1]
                    sc0, y0 = self.geometry(r0)
                    sc1, y1 = self.geometry(r1)
                    x = vs.slot_x(s0) + (vs.slot_x(s1) - vs.slot_x(s0)) * part + self.offsets[c][0]
                    y = y0 + (y1 - y0) * part + self.offsets[c][1]
                    placements.append((WALK, x, y, sc0 + (sc1 - sc0) * part))
                yield self._frame(self._window(placements, True, tick))
                tick += 1
            self.slots = [p[-1][0] for p in paths]
            for _ in range(30):                     # waiting for the pick
                yield self._frame(self._window(self._standing(False), True, tick))
                tick += 1
            if mark_rounds:
                yield None                          # the answer buttons light up here

    @property
    def answer(self):
        return self.slots[self.special]


class RandomRowsGame(FakeGame):
    """
    The variant from logs/violetta/violetta-20261114-213630.mp4: every round
    picks new sizes and heights for the walking rows, and any number of the
    Violettas may stay in their slots (stepping into a row in place) -- or none.
    """

    def __init__(self, seed, scale, origin, frame_size, stay_counts=(0, 1, 2)):
        super().__init__(seed, scale, origin, frame_size)
        self.stay_counts = stay_counts
        self.rows = {}

    def geometry(self, row):
        return self.rows.get(row) or vs.row_geometry(row)

    def _standing(self, special_visible):
        # The real Violettas stand still between rounds (no motion until the walk)
        return [([group[0]], x, y, s) for group, x, y, s in super()._standing(special_visible)]

    def shuffle_paths(self):
        while True:     # four sizes far enough apart to tell apart, at random heights
            scales = sorted(self.rng.uniform(0.5, 1.55) for _ in range(4))
            if all(b / a > 1.14 for a, b in zip(scales, scales[1:])):
                break
        self.rng.shuffle(scales)
        ids = [100 + len(self.rows) + n for n in range(4)]
        for row, scale in zip(ids, scales):
            self.rows[row] = (scale, self.rng.uniform(112, 138))
        staying = self.rng.sample(range(4), self.rng.choice(self.stay_counts))
        moving = [c for c in range(4) if c not in staying]
        while True:
            targets = [self.slots[c] for c in moving]
            self.rng.shuffle(targets)
            if all(t != self.slots[c] for t, c in zip(targets, moving)):
                break
        destination = {c: self.slots[c] for c in staying}
        destination.update(zip(moving, targets))
        paths = []
        for c in range(4):
            start, end, row = self.slots[c], destination[c], ids[c]
            if c in staying:
                waypoints = [start, start, start]
            else:
                middle = self.rng.choice([s for s in range(4) if s not in (start, end)])
                waypoints = [middle, end, end]  # the real ones keep moving, then stop at the end
            paths.append([(start, vs.STAND_ROW), (start, row)] +
                         [(w, row) for w in waypoints] + [(end, vs.STAND_ROW)])
        return paths


class ViolettaSolverTests(unittest.TestCase):
    def test_nearest_slots_assigns_distinct_slots(self):
        slots, _ = vs.nearest_slots([150, 160, 352, 40])
        self.assertEqual(sorted(slots), [0, 1, 2, 3])
        self.assertEqual(slots[2], 3)
        self.assertEqual(slots[3], 0)

    def test_calibration_recovers_window_scale_and_position(self):
        for scale, origin in ((1.0, (130, 70)), (2.0, (40, 25)), (3.0, (10, 5))):
            with self.subTest(scale=scale):
                game = FakeGame(0, scale, origin, (1400, 1000))
                calibration = vs.Calibration.find(next(game.frames()), SPRITES)
                self.assertIsNotNone(calibration)
                self.assertAlmostEqual(calibration.scale, scale, delta=0.03 * scale)
                self.assertAlmostEqual(calibration.origin[0], origin[0], delta=2 * scale)
                self.assertAlmostEqual(calibration.origin[1], origin[1], delta=2 * scale)

    def test_follows_the_make_up_violetta_through_shuffles(self):
        for seed, scale in zip(range(6), (1.0, 2.0, 1.0, 1.5, 2.0, 1.0)):
            with self.subTest(seed=seed, scale=scale):
                size = (int(441 * scale) + 60, int(318 * scale) + 40)
                game = FakeGame(seed, scale, (30, 20), size)
                solver = vs.ViolettaSolver(SPRITES, rounds=2)
                answers = [a for a in map(solver.update, game.frames(rounds=2)) if a is not None]
                self.assertEqual(len(answers), 2, solver.events)
                self.assertEqual(answers[-1], game.answer, solver.events)
                self.assertTrue(solver.confident, solver.events)

    def test_unreadable_shuffle_still_answers_with_last_known_slot(self):
        solver = vs.ViolettaSolver(SPRITES)
        solver.state, solver.target_slot = 'shuffle', 2
        solver.calibration = vs.Calibration(1.0, (0, 0), 1.0)
        solver._recorder = vs.ShuffleRecorder()
        blank = np.zeros((318, 441, 3), np.uint8)
        answers = [solver.update(blank) for _ in range(10)]
        self.assertIn(2, answers)
        self.assertFalse(solver.confident)

    def test_force_answer_guesses_last_known_slot_when_no_shuffle_was_seen(self):
        solver = vs.ViolettaSolver(SPRITES, rounds=2)
        self.assertIsNone(solver.force_answer())        # nothing to answer before the reveal
        solver.state, solver.target_slot = 'wait_shuffle', 3
        solver.calibration = vs.Calibration(1.0, (0, 0), 1.0)
        self.assertEqual(solver.force_answer(), 3)
        self.assertFalse(solver.confident)
        self.assertEqual((solver.state, solver.rounds_left), ('wait_shuffle', 1))
        boxes, labels = solver.overlay_shapes()
        self.assertFalse(any(kind == 'target' for *_, kind, _ in boxes))
        self.assertEqual(solver.force_answer(), 3)
        self.assertEqual(solver.state, 'done')

    def test_stable_target_flow_answers_when_shuffle_track_is_unavailable(self):
        from types import SimpleNamespace

        solver = vs.ViolettaSolver(SPRITES, rounds=1)
        solver.state, solver.target_slot = 'wait_shuffle', 2
        solver.calibration = vs.Calibration(1.0, (0, 0), 1.0)
        scale = 0.8
        x = vs.slot_x(0) + vt.SLOT_OFFSET
        solver._target_flow = SimpleNamespace(
            visible=True,
            confidence=0.85,
            center=np.array((x + 23.5 * scale,
                             vs.STAND_PIVOT_Y + 48 * scale), dtype=np.float32),
            width=85 * scale,
            height=96 * scale,
        )
        solver._target_flow_history = [
            (frame, x, 0.85) for frame in range(vs.FLOW_MIN_SAMPLES)
        ]
        solver._resolve_best = lambda: (None, 'unreadable rows', None)

        answer = solver.force_answer(fast=True)

        self.assertEqual(answer, 0)
        self.assertTrue(solver.confident)
        self.assertEqual(solver.last_report['layout'], 'target optical flow')
        self.assertTrue(any('target optical flow tracked slot 1' in message
                            for _, message in solver.events))

    def test_conflicting_target_flow_cannot_override_motion_track_endpoint(self):
        tracked = {0: 0, 1: 3, 2: 1, 3: 2}
        report = {
            'frames': 57,
            'consistent': False,
            'seen': [0.63, 0.81, 0.37, 0.60],
            'max_step': [32.1, 102.0, 106.0, 113.1],
            'nearest_end_slot': [0, 3, 0, 1],
            'nearest_end_error': [3.4, 6.0, 6.5, 29.1],
            'terminal_confirmed': [False, False, False, False],
        }
        solver = vs.ViolettaSolver(SPRITES, rounds=1)
        solver.state, solver.target_slot = 'wait_shuffle', 2
        solver._resolve_best = lambda: (
            None, {'consistent': False, 'layout': 'classic',
                   'worst_slot_error': 133.1}, None)
        solver._tracked = lambda: (tracked, report)
        solver._target_flow_reading = lambda: {
            'slot': 1, 'error': 9.5, 'confidence': 1.0,
            'frames': 100, 'max_step': 12.0,
        }
        solver._read_unknown_rows = lambda: (None, 'no walking rows found')

        answer = solver._finish_shuffle()

        self.assertEqual(answer, 0)
        self.assertFalse(solver.confident)
        self.assertTrue(any('not trusting the conflicting flow result' in message
                            for _, message in solver.events))

    def test_consistent_row_mapping_wins_over_conflicting_motion_tracking(self):
        solver = vs.ViolettaSolver(SPRITES, rounds=1)
        solver.state, solver.target_slot = 'wait_shuffle', 0
        solver._resolve_best = lambda: (
            {0: (0, 3), 1: (1, 0), 2: (2, 1), 3: (3, 2)},
            {'consistent': True, 'layout': 'classic'}, 'classic')
        solver._tracked = lambda: (
            {0: 1, 1: 0, 2: 2, 3: 3},
            {'consistent': True, 'seen': [1.0] * 4, 'max_step': [0.0] * 4})

        answer = solver._finish_shuffle()

        self.assertEqual(answer, 3)
        self.assertEqual(solver.last_report['layout'], 'classic')

    def test_fast_force_answer_uses_target_nearest_slot_without_slow_row_search(self):
        class FakeTracker:
            def result(self):
                return ({0: 2, 1: 1, 2: 0, 3: 3}, {
                    'frames': 43, 'consistent': False, 'seen': [0.8, 0.2, 0.8, 0.8],
                    'max_step': [17.2, 2.1, 64.8, 66.8],
                    'end_error': [103.6, 5.8, 1.4, 97.4],
                    'nearest_end_slot': [1, 1, 0, 3],
                    'nearest_end_error': [1.4, 5.8, 1.4, 2.0],
                    'worst_slot_error': 103.6,
                })

        solver = vs.ViolettaSolver(SPRITES, rounds=1)
        solver.state, solver.target_slot = 'wait_shuffle', 0
        solver._tracker = FakeTracker()
        solver._read_unknown_rows = lambda: self.fail('fast answer must not run row search')

        answer = solver.force_answer(fast=True)

        self.assertEqual(answer, 1)
        self.assertTrue(solver.confident)
        self.assertTrue(any('her track is clean' in message for _, message in solver.events))

    def test_fast_force_answer_ignores_an_untrusted_tracker(self):
        class FakeTracker:
            def result(self):
                return ({0: 2, 1: 1, 2: 0, 3: 3}, {
                    'frames': 12, 'consistent': False, 'seen': [0.2, 0.8, 0.8, 0.8],
                    'max_step': [10.0, 10.0, 10.0, 10.0],
                    'end_error': [103.6, 5.8, 1.4, 97.4],
                    'nearest_end_slot': [1, 1, 0, 3],
                    'nearest_end_error': [1.4, 5.8, 1.4, 2.0],
                    'worst_slot_error': 103.6,
                })

        solver = vs.ViolettaSolver(SPRITES, rounds=1)
        solver.state, solver.target_slot = 'wait_shuffle', 0
        solver._tracker = FakeTracker()
        solver._read_unknown_rows = lambda: self.fail('fast answer must not run row search')

        answer = solver.force_answer(fast=True)

        self.assertEqual(answer, 0)
        self.assertFalse(solver.confident)
        self.assertTrue(any('tracker was not trustworthy' in message
                            for _, message in solver.events))

    def test_fast_force_answer_uses_a_well_seen_untrusted_track(self):
        # 20261119-195802 round 1: a decoy passed her at the same size, so her
        # track was untrusted; it still ended on the right slot, unlike her
        # start slot, where the decoy finished
        class FakeTracker:
            def result(self):
                return ({0: 0, 1: 3, 2: 1, 3: 2}, {
                    'frames': 40, 'consistent': True, 'seen': [0.96, 0.88, 0.9, 0.9],
                    'max_step': [10.0, 10.0, 10.0, 10.0],
                    'end_error': [2.0, 4.0, 3.0, 2.0],
                    'nearest_end_slot': [0, 3, 1, 2],
                    'nearest_end_error': [2.0, 4.0, 3.0, 2.0],
                    'ambiguous': [False, True, False, True],
                    'lost': [False] * 4,
                    'worst_slot_error': 4.0,
                })

        solver = vs.ViolettaSolver(SPRITES, rounds=1)
        solver.state, solver.target_slot = 'wait_shuffle', 3
        solver._tracker = FakeTracker()
        solver._resolve_best = lambda: (None, 'no fixed rows', None)
        solver._read_unknown_rows = lambda: self.fail('fast answer must not run row search')

        answer = solver.force_answer(fast=True)

        self.assertEqual(answer, 2)
        self.assertFalse(solver.confident)

    def test_motion_overlay_mode_keeps_row_reader_as_answer_source(self):
        from types import SimpleNamespace

        solver = vs.ViolettaSolver(SPRITES, rounds=1)
        solver.state, solver.target_slot = 'wait_shuffle', 0
        solver.overlay_motion_only = True
        solver._row_finder = SimpleNamespace(last_walks={})
        mapping = {
            'row_a': (0, 1), 'row_b': (1, 2),
            'row_c': (2, 3), 'row_d': (3, 0),
        }
        solver._resolve_best = lambda: (None, 'no fixed rows', None)
        solver._read_unknown_rows = lambda: (
            mapping, {'consistent': True, 'worst_slot_error': 0.0})
        solver._tracked = lambda: self.fail('overlay-only tracking must not choose answers')
        solver._target_flow_reading = lambda: self.fail(
            'overlay-only tracking must not choose answers')

        self.assertEqual(solver._finish_shuffle(), 1)

    def test_fast_answer_uses_a_trusted_overlay_target_without_advancing_other_ids(self):
        ends = {0: 2, 1: 0, 2: 1, 3: 3}

        class FakeTracker:
            def snapshot(self):
                return {
                    'frames': 12,
                    'last_frame': 20,
                    'tracks': [
                        {
                            'slot': slot,
                            'x': vs.slot_x(ends[slot]) + vt.SLOT_OFFSET,
                            'seen': 0.8 if slot == 0 else 0.2,
                            'max_step': 10.0,
                            'terminal_confirmed': False,
                        }
                        for slot in range(vs.SLOT_COUNT)
                    ],
                }

        solver = vs.ViolettaSolver(SPRITES, rounds=1)
        solver.state, solver.target_slot = 'wait_shuffle', 0
        solver.overlay_motion_only = True
        solver._tracker = FakeTracker()
        solver._tracker_started = 0
        solver.frame_index = 21
        solver._resolve_best = lambda: (None, 'no fixed rows', None)
        solver._read_unknown_rows = lambda: self.fail('fast answer must skip row search')

        answer = solver.force_answer(fast=True)

        self.assertEqual(answer, 2)
        self.assertTrue(solver.confident)
        self.assertEqual(solver.last_report['layout'], 'trusted target track')
        self.assertEqual(solver._slot_identity, list(range(vs.SLOT_COUNT)))

    def test_low_confidence_live_answer_still_clicks_when_game_is_focused(self):
        from types import SimpleNamespace

        solver = SimpleNamespace(
            confident=False,
            button_point=lambda slot: (12.0, 18.0),
        )
        with patch('ctypes.windll.user32.GetForegroundWindow', return_value=7), \
                patch('time.sleep'), patch.object(vs, '_click') as click:
            point = vs._send_answer_click(solver, 2, {'left': 100, 'top': 200}, 7)

        self.assertEqual(point, (112.0, 218.0))
        click.assert_called_once_with(112.0, 218.0)

    def test_live_answer_click_still_requires_the_game_to_be_focused(self):
        from types import SimpleNamespace

        solver = SimpleNamespace(button_point=lambda slot: (12.0, 18.0))
        with patch('ctypes.windll.user32.GetForegroundWindow', return_value=8), \
                patch.object(vs, '_click') as click:
            point = vs._send_answer_click(solver, 2, {'left': 100, 'top': 200}, 7)

        self.assertIsNone(point)
        click.assert_not_called()

    def test_consistent_tracker_is_rejected_when_target_path_jumps(self):
        tracked = {0: 0, 1: 2, 2: 1, 3: 3}
        report = {
            'frames': 59,
            'consistent': True,
            'seen': [0.81, 0.71, 0.66, 0.69],
            'max_step': [81.9, 54.3, 23.3, 57.5],
            'nearest_end_error': [5.6, 5.9, 13.7, 18.0],
            'terminal_confirmed': [False, False, False, False],
        }
        solver = vs.ViolettaSolver(SPRITES, rounds=1)
        solver.state, solver.target_slot = 'wait_shuffle', 3
        solver._resolve_best = lambda: (
            None, {'consistent': False, 'layout': 'classic',
                   'worst_slot_error': 100.0}, None)
        solver._tracked = lambda: (tracked, report)

        answer = solver.force_answer(fast=True)

        self.assertEqual(answer, 3)
        self.assertFalse(solver.confident)
        self.assertTrue(any('tracker was not trustworthy' in message
                            for _, message in solver.events))

    def test_uncertain_tracker_fallback_uses_target_nearest_slot(self):
        tracked = {0: 0, 1: 3, 2: 1, 3: 2}
        report = {
            'frames': 74,
            'consistent': False,
            'seen': [0.69, 0.74, 0.28, 0.76],
            'max_step': [281.2, 113.1, 151.9, 123.7],
            'end_error': [3.4, 6.0, 87.3, 4.0],
            'nearest_end_slot': [0, 3, 0, 2],
            'nearest_end_error': [3.4, 6.0, 16.7, 4.0],
            'terminal_confirmed': [True, False, False, False],
            'worst_slot_error': 87.3,
        }
        solver = vs.ViolettaSolver(SPRITES, rounds=1)
        solver.state, solver.target_slot = 'wait_shuffle', 2
        solver._resolve_best = lambda: (
            None, {'consistent': False, 'layout': 'classic',
                   'worst_slot_error': 100.0}, None)
        solver._tracked = lambda: (tracked, report)
        solver._read_unknown_rows = lambda: (None, 'no walking rows found')

        answer = solver.force_answer()

        self.assertEqual(answer, 0)
        self.assertFalse(solver.confident)
        self.assertTrue(any("target track's nearest slot 1" in message
                            for _, message in solver.events))

    def test_uncertain_row_mapping_uses_target_nearest_slot(self):
        tracked = {0: 0, 1: 3, 2: 1, 3: 2}
        report = {
            'frames': 74,
            'consistent': False,
            'seen': [0.69, 0.74, 0.28, 0.76],
            'max_step': [281.2, 113.1, 151.9, 123.7],
            'end_error': [3.4, 6.0, 87.3, 4.0],
            'nearest_end_slot': [0, 3, 0, 2],
            'nearest_end_error': [3.4, 6.0, 16.7, 4.0],
            'terminal_confirmed': [True, False, False, False],
            'worst_slot_error': 87.3,
        }
        mapping = {'r0': (0, 3), 'r1': (1, 2), 'r2': (2, 1), 'r3': (3, 0)}
        solver = vs.ViolettaSolver(SPRITES, rounds=1)
        solver.state, solver.target_slot = 'wait_shuffle', 2
        solver._resolve_best = lambda: (
            mapping, {'consistent': False, 'layout': 'classic',
                      'worst_slot_error': 100.0}, 'classic')
        solver._tracked = lambda: (tracked, report)
        solver._read_unknown_rows = lambda: (None, 'no walking rows found')

        answer = solver.force_answer()

        self.assertEqual(answer, 0)
        self.assertFalse(solver.confident)
        self.assertTrue(any("target track's nearest slot 1" in message
                            for _, message in solver.events))


class TrackerRecoveryTests(unittest.TestCase):
    def test_snapshot_normalizes_long_moves_and_marks_close_crossings(self):
        tracker = vt.ViolettaTracker(SPRITES)
        tracker.start(0)
        tracker.tracks[0].take(6, (0.8, 1.0, 120.0, 121.0))
        tracker.frames = 2
        tracker.last_frame = 6
        snapshot = tracker.snapshot()

        self.assertAlmostEqual(snapshot['tracks'][0]['max_speed'], 80.0 / 6)
        self.assertGreater(snapshot['tracks'][0]['max_step'], 25.0)

        tracker.tracks[1].take(6, (0.8, 1.0, 144.0, 114.0))
        tracker.tracks[0].take(9, (0.8, 1.0, 140.0, 110.0))
        tracker.last_frame = 9
        snapshot = tracker.snapshot()
        self.assertTrue(snapshot['tracks'][0]['identity_ambiguous'])
        self.assertTrue(snapshot['tracks'][1]['identity_ambiguous'])

    def test_refine_returns_the_pose_that_matched(self):
        from src.detection import violetta_rows as vr

        templates = [
            np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32),
        ]
        mask = np.ones((3, 3), np.uint8)

        class FakeBank:
            def get(self, scale):
                return [(template, mask, 1.0, 1.0) for template in templates]

        canvas = np.zeros((40, 40), np.float32)
        canvas[15:18, 15:18] = templates[1]
        hit = vr._refine(canvas, FakeBank(), (0.8, 1.0, 16, 16))

        self.assertIsNotNone(hit)
        self.assertEqual(hit[4], 1)

    def test_track_retains_matched_pose_for_tight_overlay_bounds(self):
        track = vt.Track(0, 0)
        track.take(1, (0.8, 0.9, track.x + 2, track.y, 5))

        self.assertEqual(track.pose, 5)

    def test_ambiguous_detection_does_not_switch_between_overlapping_tracks(self):
        target = vt.Track(0, 0)
        target.x = 100.0
        nearby = vt.Track(1, 0)
        nearby.x = 108.0
        hit = (0.8, 1.0, 100.0, vt.STAND_PIVOT_Y)

        costs = [vt._cost(track, 1, hit) for track in (target, nearby)]
        self.assertTrue(all(cost is not None for cost in costs))
        self.assertLess(abs(costs[0] - costs[1]), vt.ASSOCIATION_AMBIGUITY_MARGIN)
        self.assertEqual(vt.associate([target, nearby], 1, [hit]), {})

    def test_missing_track_reacquires_the_unique_detection_left_by_three_known_ids(self):
        tracks = [vt.Track(slot, 25) for slot in range(vs.SLOT_COUNT)]
        for track, (x, scale, y) in zip(tracks[:3], (
                (40.0, 1.1, 120.0), (250.0, 0.84, 103.0), (357.0, 1.49, 133.0))):
            track.x, track.scale, track.y = x, scale, y
            track.matched = 6
        missing = tracks[3]
        missing.x, missing.scale, missing.y = 346.9, 0.60, 95.2
        missing.seen_at = missing.last_update_frame = 10
        missing.vx, missing.direction, missing.matched = -3.0, -1, 8
        missing.wait(25)
        hits = [
            (0.70, 1.10, 40.0, 120.0),
            (0.72, 0.84, 250.0, 103.0),
            (0.68, 1.49, 357.0, 133.0),
            (0.61, 0.62, 209.5, 95.2),
            (0.40, 0.64, 209.0, 95.5),
        ]

        assignment = vt.associate(tracks, 25, hits)

        self.assertEqual(assignment, {0: 0, 1: 1, 2: 2, 3: 3})
        for track_index, hit_index in assignment.items():
            tracks[track_index].take(25, hits[hit_index])

        follow_up = [
            (0.70, 1.10, 40.0, 120.0),
            (0.72, 0.84, 250.0, 103.0),
            (0.68, 1.49, 357.0, 133.0),
            (0.55, 0.60, 183.9, 95.2),
        ]
        next_assignment = vt.associate(tracks, 28, follow_up)
        self.assertEqual(next_assignment.get(3), 3)

    def test_missing_track_is_not_guessed_when_two_residual_paths_are_plausible(self):
        tracks = [vt.Track(slot, 25) for slot in range(vs.SLOT_COUNT)]
        for track, (x, scale, y) in zip(tracks[:3], (
                (40.0, 1.1, 120.0), (250.0, 0.84, 103.0), (357.0, 1.49, 133.0))):
            track.x, track.scale, track.y = x, scale, y
            track.matched = 6
        missing = tracks[3]
        missing.x, missing.scale, missing.y = 346.9, 0.60, 95.2
        missing.seen_at = missing.last_update_frame = 10
        missing.vx, missing.direction, missing.matched = -3.0, -1, 8
        missing.wait(25)
        hits = [
            (0.70, 1.10, 40.0, 120.0),
            (0.72, 0.84, 250.0, 103.0),
            (0.68, 1.49, 357.0, 133.0),
            (0.61, 0.62, 209.5, 95.2),
            (0.60, 0.62, 270.0, 95.2),
        ]

        assignment = vt.associate(tracks, 25, hits)

        self.assertNotIn(3, assignment)

    def test_one_frame_miss_preserves_motion_and_tracks_missed_duration(self):
        track = vt.Track(0, 10)
        track.vx = 10.0

        track.wait(11)
        self.assertEqual(track.missed_frames, 1)
        self.assertAlmostEqual(track.vx, 8.0)
        self.assertAlmostEqual(track.predict(11), track.x + 10.0)

        track.wait(13)
        self.assertEqual(track.missed_frames, 3)
        self.assertAlmostEqual(track.vx, 8.0 * 0.8 ** 2)
        self.assertAlmostEqual(track.predict(13), track.x + 24.4)

        track.take(14, (0.8, 1.1, track.x + 5, track.y))
        self.assertEqual(track.missed_frames, 0)
        self.assertAlmostEqual(track.predict(14), track.x)

    def test_prediction_coasts_vertical_and_scale_motion_during_miss(self):
        track = vt.Track(0, 10)
        track.take(11, (0.8, 0.9, track.x + 6, track.y + 6))
        observed_y, observed_scale = track.y, track.scale

        track.wait(12)
        _, predicted_y, predicted_scale = track.predict_state(12)

        self.assertGreater(predicted_y, observed_y)
        self.assertLess(predicted_scale, observed_scale)

    def test_reacquisition_rejects_a_large_scale_jump_while_moving(self):
        track = vt.Track(0, 0)
        track.vx = 10.0
        hit = (0.8, 1.2, track.x + 20, track.y)

        # In place she may change size fast (002959 at ~9 fps: 0.15 a frame)...
        self.assertAlmostEqual(vt._size_gate(track, 2), 2 * vt.GATE_SCALE_PER_FRAME)
        self.assertAlmostEqual(vt._size_gate(track, 2, reacquire=True),
                               2 * vt.GATE_SCALE_PER_FRAME)
        # ...but not while moving
        self.assertIsNone(vt._cost(track, 2, hit))
        self.assertIsNone(vt._cost(track, 2, hit, reacquire=True))

    def test_walker_hidden_mid_walk_does_not_take_a_size_jump_in_place(self):
        # 20261119-221411 round 3: walking left at 0.62, hidden by a front
        # walker; a 0.35 match where she vanished must not become her
        track = vt.Track(0, 0)
        track.x, track.scale, track.y = 184.0, 0.62, 106.0
        track.vx, track.walk_matches, track.direction = -11.0, 5, -1
        track.seen_at = track.last_update_frame = 175
        track.matched = 10
        self.assertIsNone(vt._cost(track, 179, (0.6, 0.35, 185.0, 100.0)))
        # ...while one standing still may still step into a row in place
        track.vx, track.walk_matches = 0.0, 0
        self.assertIsNotNone(vt._cost(track, 177, (0.6, 0.52, 184.0, 100.0)))

    def test_one_frame_step_into_a_row_may_shrink_quickly(self):
        # 20261119-195802 at ~10 fps: 0.69 -> 0.59 in one frame (0.157 of log size)
        track = vt.Track(0, 0)
        self.assertAlmostEqual(vt._size_gate(track, 1), vt.GATE_SCALE_ONE_FRAME)
        self.assertGreater(vt._size_gate(track, 1), abs(np.log(0.59 / 0.69)))
        # ...without widening longer gaps beyond 0.15 a frame
        self.assertAlmostEqual(vt._size_gate(track, 3), 3 * vt.GATE_SCALE_PER_FRAME)

    def test_missed_track_searches_larger_scales_near_last_position(self):
        class ImmediatePool:
            @staticmethod
            def map(function, values):
                return list(map(function, values))

        track = vt.Track(0, 0)
        track.vx = 10.0
        track.wait(1)
        track.wait(2)
        calls = []

        def record_search(canvas, bank, scale, x0, y0, x_window, y_window, poses=None):
            calls.append((scale, x0, x_window))
            return None

        with patch.object(vt, '_search_one', side_effect=record_search):
            vt.detect_near(np.zeros((200, 200, 3), np.uint8), object(),
                           ImmediatePool(), [track], 3)

        predicted_x = track.predict(3) + vt.vr.PAD
        last_x = track.x + vt.vr.PAD
        self.assertTrue(any(scale > 1.15 and x0 == last_x and x_window <= 20
                            for scale, x0, x_window in calls))
        self.assertFalse(any(scale > 1.15 and x0 == predicted_x and x_window > 20
                             for scale, x0, x_window in calls))

    def test_local_tracking_keeps_low_score_match_when_motion_and_scale_fit(self):
        class ImmediatePool:
            @staticmethod
            def map(function, values):
                return list(map(function, values))

        track = vt.Track(0, 0)

        def find_match(canvas, bank, scale, x0, y0, x_window, y_window, poses=None):
            if x_window > vt.IN_PLACE_PX + 4 and scale == vt.snap(track.scale):
                return 0.35, scale, x0, y0, 0
            if x_window <= 3:
                return 0.43, scale, x0, y0, 0
            return None

        with patch.object(vt, '_search_one', side_effect=find_match):
            hits = vt.detect_near(np.zeros((200, 200, 3), np.uint8), object(),
                                  ImmediatePool(), [track], 1)

        self.assertEqual(len(hits), 1)
        self.assertAlmostEqual(hits[0][0], 0.43)
        self.assertIsNotNone(vt._cost(track, 1, hits[0]))

    def test_long_gap_search_covers_observed_scale_handoff_displacement(self):
        class ImmediatePool:
            @staticmethod
            def map(function, values):
                return list(map(function, values))

        track = vt.Track(0, 89)
        track.x, track.scale, track.y = 22.4, 1.49, 136.7
        track.wait(104)
        calls = []

        def record_search(canvas, bank, scale, x0, y0, x_window, y_window, poses=None):
            calls.append((scale, x0, x_window))
            return None

        with patch.object(vt, '_search_one', side_effect=record_search):
            vt.detect_near(np.zeros((200, 500, 3), np.uint8), object(),
                           ImmediatePool(), [track], 104)

        predicted_x = track.predict(104) + vt.vr.PAD
        self.assertTrue(any(x0 == predicted_x and x_window >= 212
                            for _, x0, x_window in calls))

    def test_long_gap_does_not_assign_distant_hit_by_matching_scale(self):
        target = vt.Track(0, 89)
        target.x, target.scale, target.y = 22.4, 1.49, 136.7
        target.wait(104)

        nearby = vt.Track(2, 98)
        nearby.x, nearby.scale, nearby.y, nearby.vx = 218.3, 1.11, 125.0, -5.3
        nearby.wait(101)
        nearby.wait(104)

        hits = [(0.61, 1.11, 181.8, 129.0), (0.60, 1.54, 234.5, 136.7)]
        assignment = vt.associate([target, nearby], 104, hits)

        self.assertEqual(assignment, {1: 0})

    def test_tracker_searches_adjacent_terminal_slot(self):
        class ImmediatePool:
            @staticmethod
            def map(function, values):
                return list(map(function, values))

        track = vt.Track(0, 107)
        track.x, track.scale, track.y, track.vx, track.matched = 235.3, 1.53, 136.7, 3.5, 6
        calls = []

        def record_search(canvas, bank, scale, x0, y0, x_window, y_window, poses=None):
            calls.append((scale, x0, x_window))
            return None

        with patch.object(vt, '_search_one', side_effect=record_search):
            vt.detect_near(np.zeros((200, 500, 3), np.uint8), object(),
                           ImmediatePool(), [track], 110)

        terminal_anchor = vt.slot_x(3) + vt.SLOT_OFFSET + vt.vr.PAD
        self.assertTrue(any(abs(x0 - terminal_anchor) < 1 and x_window == vt.TERMINAL_X_WINDOW
                            for _, x0, x_window in calls))

    def test_terminal_handoff_requires_slot_repeat_before_trust(self):
        track = vt.Track(0, 107)
        track.x, track.scale, track.y, track.vx, track.matched = 320.0, 1.53, 136.7, 0.0, 6
        track.direction = 1
        hit = (0.63, 1.41, 352.0, 131.8)
        terminal_slot = vt._terminal_handoff(track, 110, hit)

        self.assertEqual(terminal_slot, 3)
        self.assertIsNotNone(vt._cost(track, 110, hit))
        track.take(110, hit, terminal_slot)
        for frame, scale in ((113, 1.23), (116, 1.03), (119, 1.0)):
            track.take(frame, (0.69, scale, 352.0, 123.0))
        self.assertTrue(track.terminal_confirmed)

    def test_terminal_handoff_rejects_fast_jump_without_motion_support(self):
        track = vt.Track(2, 77)
        track.x, track.scale, track.y = 244.0, 0.79, 110.0
        track.matched = 6
        hit = (0.62, 0.78, 138.0, 118.0)

        self.assertIsNone(vt._terminal_handoff(track, 80, hit))
        self.assertIsNone(vt._cost(track, 80, hit))

    def test_weak_match_cannot_reverse_a_motion_path(self):
        track = vt.Track(0, 95)
        track.x, track.scale, track.y = 142.0, 0.568, 109.0
        track.direction = -1
        track.matched = 10
        hit = (0.536, 0.585, 208.6, 114.1)

        self.assertIsNone(vt._cost(track, 107, hit))

    def test_motion_match_cannot_exceed_physical_speed_gate(self):
        track = vt.Track(0, 10)
        track.x, track.matched = 100.0, 4
        # 20 px/frame on her own row is a walk (002959 round 4: ~25 px/frame)...
        self.assertIsNotNone(vt._cost(track, 13, (0.8, 1.0, 160.0, track.y)))
        # ...but that fast with a size change is a hop onto another Violetta
        self.assertIsNone(vt._cost(track, 13, (0.8, 1.1, 160.0, track.y)))
        # ...and beyond the fastest walk, never
        too_far = 100.0 + (vt.FAST_WALK_SPEED + vt.MOTION_SPEED_TOLERANCE) * 3 + vt.GATE_X + 1
        self.assertIsNone(vt._cost(track, 13, (0.8, 1.0, too_far, track.y)))

    def _lost_pair(self):
        """002959 round 4: two Violettas lost together on the far-back row."""
        left, right = vt.Track(0, 250), vt.Track(1, 250)
        for track, x, y in ((left, 234.0, 103.0), (right, 336.0, 108.0)):
            track.x, track.scale, track.y = x, 0.6, y
            track.matched, track.lost = 8, True
        return [left, right]

    def test_same_row_lost_pair_waits_until_both_are_visible(self):
        tracks = self._lost_pair()
        hits = [(0.6, 0.55, 205.0, 103.0)]
        self.assertFalse(vt.order_kept(tracks, hits, (0, None)))
        self.assertFalse(vt.order_kept(tracks, hits, (None, 0)))
        self.assertTrue(vt.order_kept(tracks, hits, (None, None)))

    def test_same_row_lost_pair_comes_back_in_its_old_order(self):
        tracks = self._lost_pair()
        hits = [(0.6, 0.55, 41.0, 95.0), (0.6, 0.55, 150.0, 103.0)]
        self.assertTrue(vt.order_kept(tracks, hits, (0, 1)))
        self.assertFalse(vt.order_kept(tracks, hits, (1, 0)))
        found = vt.associate(tracks, 270, hits, cost=vt._lost_cost,
                             valid=lambda pick: vt.order_kept(tracks, hits, pick),
                             miss_cost=vt.LOST_MISS_COST)
        self.assertEqual(found, {0: 0, 1: 1})

    def test_lost_track_does_not_block_a_walker_arriving_at_its_place(self):
        # 190834 round 4: a track lost on slot 1 made the yellow walker
        # arriving there ambiguous, and she lost her match
        tracker = vt.ViolettaTracker.__new__(vt.ViolettaTracker)
        stale, walker = vt.Track(0, 290), vt.Track(1, 290)
        stale.x, stale.scale, stale.y, stale.lost = 40.0, 1.0, 121.0, True
        stale.seen_at = stale.last_update_frame = 292
        walker.x, walker.scale, walker.y, walker.matched = 49.0, 1.18, 132.0, 8
        walker.direction, walker.vx, walker.walk_matches = -1, -5.0, 3
        walker.seen_at = walker.last_update_frame = 325
        tracker.tracks = [stale, walker]
        hits = [(0.72, 1.14, 37.0, 127.0)]
        self.assertEqual(tracker._associate_with_lost(331, hits, False), {1: 0})

    def test_lost_tracks_fill_the_slots_left_in_their_order(self):
        xs = [234.0 - vt.SLOT_OFFSET, 342.0, 246.0, 336.0 - vt.SLOT_OFFSET]
        slots, _ = vs.end_slots_with_lost(xs, [True, False, False, True])
        self.assertEqual(slots[1:3], [3, 2])
        self.assertEqual((slots[0], slots[3]), (0, 1))

    def test_close_pass_at_different_sizes_is_not_ambiguous(self):
        front, stepping = vt.Track(0, 0), vt.Track(1, 0)
        front.path = [(240, 132.0, 1.54, 154.0, 0.7)]
        stepping.path = [(240, 133.0, 1.37, 145.0, 0.7)]
        self.assertEqual(vt._identity_ambiguities([front, stepping]), [False, False])
        stepping.path = [(240, 133.0, 1.50, 150.0, 0.7)]
        self.assertEqual(vt._identity_ambiguities([front, stepping]), [True, True])

    def test_shrinking_in_place_moves_the_pivot_up_not_down(self):
        track = vt.Track(0, 270)
        track.x, track.scale, track.y, track.matched = 142.0, 0.59, 109.0, 8
        self.assertIsNone(vt._cost(track, 279, (0.5, 0.43, 143.0, 136.0)))
        self.assertIsNotNone(vt._cost(track, 279, (0.5, 0.43, 143.0, 92.0)))

    def test_motion_match_does_not_jump_to_a_distant_track_after_occlusion(self):
        track = vt.Track(2, 77)
        track.x, track.scale, track.y, track.vx = 162.5, 0.6, 105.2, -0.2
        track.direction, track.matched = -1, 6
        hit = (0.701, 0.535, 293.4, 108.7)

        self.assertIsNone(vt._cost(track, 89, hit))

    def test_terminal_confirmation_allows_trusting_the_target_track(self):
        from src.detection.violetta_solver import trusted_track
        report = {
            'frames': 20, 'seen': [0.31, 0.69, 0.41, 0.34],
            'max_step': [30.0, 123.0, 60.0, 8.0],
            'nearest_end_error': [2.0, 8.4, 40.0, 3.0],
            'terminal_confirmed': [False, True, False, False],
        }

        self.assertTrue(trusted_track({0: 0, 1: 1, 2: 2, 3: 3}, report, 1))

    def test_isolated_track_reacquires_scale_after_two_misses(self):
        track = vt.Track(0, 0)
        track.vx = 10.0
        track.wait(1)
        track.wait(2)
        hit = (0.8, 1.0, track.predict(3), track.y)

        self.assertIsNotNone(vt._cost(track, 3, hit))

    def test_reacquisition_allows_large_scale_change_in_place(self):
        track = vt.Track(0, 0)
        track.wait(1)
        track.wait(2)
        hit = (0.8, 1.3, track.x, track.y)

        self.assertIsNotNone(vt._cost(track, 3, hit))

    def test_sampled_frame_gap_reacquires_scale_at_the_same_position(self):
        class ImmediatePool:
            @staticmethod
            def map(function, values):
                return list(map(function, values))

        track = vt.Track(0, 122)
        track.x, track.scale, track.y = 33.5, 0.603, 119.0
        hit = (0.61, 0.765, 33.6, 119.0)
        self.assertIsNotNone(vt._cost(track, 125, hit))

        calls = []

        def record_search(canvas, bank, scale, x0, y0, x_window, y_window, poses=None):
            calls.append((scale, x0, x_window))
            return None

        with patch.object(vt, '_search_one', side_effect=record_search):
            vt.detect_near(np.zeros((200, 200, 3), np.uint8), object(),
                           ImmediatePool(), [track], 125)

        last_x = track.x + vt.vr.PAD
        self.assertTrue(any(0.74 <= scale <= 0.80 and x0 == last_x
                            and x_window < vt.GATE_X
                            for scale, x0, x_window in calls))

    def test_long_gap_motion_reacquisition_allows_depth_scale_change(self):
        track = vt.Track(0, 0)
        track.x, track.scale, track.vx = 100.0, 0.8, 0.0
        track.wait(3)
        hit = (0.8, 0.6, 160.0, track.y)

        self.assertIsNotNone(vt._cost(track, 15, hit))

    def test_initial_overlay_scans_and_draws_all_four_slots(self):
        solver = vs.ViolettaSolver(SPRITES)
        solver.calibration = vs.Calibration(2.0, (10, 20), 1.0)
        solver.state, solver.target_slot = 'reveal', 2
        solver._standing_scan = [0.81, 0.76, 0.69, 0.72]

        boxes, labels = solver.overlay_shapes()

        self.assertEqual([box[4] for box in boxes if box[4].startswith('id')],
                         ['id1', 'id2', 'id3', 'id4'])
        self.assertEqual(len([l for l in labels if l[3].startswith('id')]), vs.SLOT_COUNT)
        self.assertIn('ID1 scan 0.81', labels[0][2])
        left, top, right, bottom = solver._standing_bounds
        expected_top_left = solver.calibration.to_frame(
            (vs.slot_x(0) + left, vs.STAND_PIVOT_Y + top))
        expected_bottom_right = solver.calibration.to_frame(
            (vs.slot_x(0) + right, vs.STAND_PIVOT_Y + bottom))
        self.assertEqual(boxes[0][:4], (*expected_top_left, *expected_bottom_right))

    def test_shuffle_overlay_holds_missing_rows_and_keeps_four_boxes(self):
        solver = vs.ViolettaSolver(SPRITES)
        solver.calibration = vs.Calibration(1.0, (0, 0), 1.0)
        solver.state, solver.target_slot = 'shuffle', 2
        solver._overlay_layout = 'classic'
        solver._update_overlay_rows('classic', {0: (vs.slot_x(0), 0.75)})
        solver._update_overlay_rows('classic', {0: (vs.slot_x(0), 0.2)})
        solver._tracker = object()

        boxes, labels = solver.overlay_shapes()

        self.assertEqual([box[4] for box in boxes if box[4].startswith('id')],
                         ['id1', 'id2', 'id3', 'id4'])
        self.assertIn('row 0 0.75 hold 1', labels[0][2])
        scale, pivot_y = vs.row_geometry(0)
        self.assertEqual(boxes[0][:4], (vs.slot_x(0) - 19 * scale, pivot_y,
                                        vs.slot_x(0) + 66 * scale, pivot_y + 96 * scale))

    def test_motion_snapshot_moves_overlay_before_row_shuffle_is_detected(self):
        class FakeTracker:
            def snapshot(self):
                return {
                    'last_frame': 8,
                    'tracks': [{
                        'slot': 0, 'matched': 4, 'score': 0.8, 'age': 1,
                        'predicted_x': vs.slot_x(2), 'predicted_y': 105.0,
                        'predicted_scale': 0.7, 'vx': 5.0, 'vy': 0.0,
                        'vlog_scale': 0.0, 'pose': 1,
                    }],
                }

        solver = vs.ViolettaSolver(SPRITES)
        solver.calibration = vs.Calibration(1.0, (0, 0), 1.0)
        solver.state, solver.target_slot = 'wait_shuffle', 0
        solver.frame_index, solver._tracker_started = 11, 0
        solver._tracker = FakeTracker()

        boxes, labels = solver.overlay_shapes()

        predicted_x = vs.slot_x(2) + 5.0 * 1.8
        left, top, right, bottom = solver._walk_union_bounds
        self.assertIn('ID1* PREDICTED 3f', labels[0][2])
        self.assertAlmostEqual(boxes[0][0], predicted_x + left * 0.7)
        self.assertAlmostEqual(boxes[0][1], 105.0 + top * 0.7)
        self.assertAlmostEqual(boxes[0][2], predicted_x + right * 0.7)
        self.assertAlmostEqual(boxes[0][3], 105.0 + bottom * 0.7)

    def test_motion_overlay_uses_matched_pose_bounds(self):
        class FakeTracker:
            def snapshot(self):
                return {
                    'last_frame': 8,
                    'tracks': [{
                        'slot': 0, 'matched': 4, 'score': 0.8, 'age': 0,
                        'predicted_x': vs.slot_x(2), 'predicted_y': 105.0,
                        'predicted_scale': 0.7, 'vx': 0.0, 'vy': 0.0,
                        'vlog_scale': 0.0, 'pose': 1,
                    }],
                }

        solver = vs.ViolettaSolver(SPRITES)
        solver.calibration = vs.Calibration(1.0, (0, 0), 1.0)
        solver.state, solver.target_slot = 'wait_shuffle', 0
        solver.frame_index, solver._tracker_started = 9, 0
        solver._tracker = FakeTracker()

        boxes, _ = solver.overlay_shapes()

        left, top, right, bottom = solver._walk_pose_bounds[1]
        self.assertEqual(boxes[0][:4], (
            vs.slot_x(2) + left * 0.7,
            105.0 + top * 0.7,
            vs.slot_x(2) + right * 0.7,
            105.0 + bottom * 0.7,
        ))

    def test_color_identity_follows_the_violetta_to_her_new_slot(self):
        solver = vs.ViolettaSolver(SPRITES)
        solver.calibration = vs.Calibration(1.0, (0, 0), 1.0)
        solver.state, solver.target_slot = 'shuffle', 1
        mapping = {
            'row_a': (0, 1), 'row_b': (1, 0),
            'row_c': (2, 2), 'row_d': (3, 3),
        }

        self.assertTrue(solver._advance_slot_identities(mapping))
        solver._overlay_layout = 'classic'
        solver._update_overlay_rows('classic', {4: (vs.slot_x(1), 0.8)})
        boxes, labels = solver.overlay_shapes()

        self.assertEqual(solver._slot_identity, [1, 0, 2, 3])
        self.assertEqual(boxes[0][4], 'id1')
        self.assertEqual(labels[0][2], 'ID1* row 4 0.80')
        scale, pivot_y = vs.row_geometry(4)
        self.assertEqual(boxes[0][:4], (vs.slot_x(1) - 19 * scale, pivot_y,
                                        vs.slot_x(1) + 66 * scale,
                                        pivot_y + 96 * scale))

    def test_incomplete_mapping_cannot_swap_color_identities(self):
        solver = vs.ViolettaSolver(SPRITES)

        self.assertFalse(solver._advance_slot_identities({'row_a': (0, 1)}))
        self.assertEqual(solver._slot_identity, [0, 1, 2, 3])

    def test_low_confidence_mapping_does_not_swap_color_ids_across_rounds(self):
        solver = vs.ViolettaSolver(SPRITES, rounds=1)
        solver.overlay_motion_only = True
        solver.target_slot = 0
        solver._target_identity = 0
        solver.calibration = vs.Calibration(1.0, (0, 0), 1.0)
        mapping = {
            'row_a': (0, 3), 'row_b': (1, 0),
            'row_c': (2, 1), 'row_d': (3, 2),
        }
        solver._resolve_best = lambda: (
            mapping, {'consistent': False, 'layout': 'found'}, 'found')

        answer = solver._finish_shuffle()

        self.assertEqual(answer, 3)
        self.assertEqual(solver._slot_identity, [0, 1, 2, 3])
        self.assertEqual(solver._target_identity, 0)
        self.assertTrue(any('keeping color IDs' in message
                            for _, message in solver.events))
        with patch.object(solver, 'button_point', wraps=solver.button_point) as point, \
                patch('src.detection.violetta_solver.cv2.putText') as put_text:
            solver.draw(np.zeros((368, 492, 3), np.uint8))
        point.assert_called_once_with(0)
        self.assertTrue(any('ID1 held slot 1; guess 4' in call.args[1]
                            for call in put_text.call_args_list))

    def test_consistent_motion_mapping_keeps_colors_without_changing_row_answer(self):
        class FakeTracker:
            def snapshot(self):
                return {
                    'last_frame': 50,
                    'frames': 10,
                    'tracks': [
                        {'slot': 0, 'x': vs.slot_x(3) + vt.SLOT_OFFSET, 'seen': 0.8,
                         'max_step': 20.0, 'terminal_confirmed': False},
                        {'slot': 1, 'x': vs.slot_x(0) + vt.SLOT_OFFSET, 'seen': 0.8,
                         'max_step': 20.0, 'terminal_confirmed': False},
                        {'slot': 2, 'x': vs.slot_x(1) + vt.SLOT_OFFSET, 'seen': 0.8,
                         'max_step': 20.0, 'terminal_confirmed': False},
                        {'slot': 3, 'x': vs.slot_x(2) + vt.SLOT_OFFSET, 'seen': 0.8,
                         'max_step': 20.0, 'terminal_confirmed': False},
                    ],
                }

        solver = vs.ViolettaSolver(SPRITES, rounds=1)
        solver.overlay_motion_only = True
        solver.state, solver.target_slot = 'wait_shuffle', 0
        solver.frame_index, solver._tracker_started = 51, 40
        solver._tracker = FakeTracker()
        row_mapping = {
            'row_a': (0, 3), 'row_b': (1, 0),
            'row_c': (2, 2), 'row_d': (3, 1),
        }
        solver._resolve_best = lambda: (
            row_mapping, {'consistent': False, 'layout': 'found'}, 'found')

        answer = solver._finish_shuffle(fast=True)

        self.assertEqual(answer, 3)
        self.assertTrue(solver.confident)
        self.assertEqual(solver._slot_identity, [1, 2, 3, 0])

    def test_trusted_motion_mapping_infers_one_missing_identity_by_exclusion(self):
        mapping = {0: 1, 1: 2, 2: 3, 3: 0}
        report = {
            'frames': 10,
            'seen': [0.8, 0.8, 0.8, 0.1],
            'max_step': [10.0, 12.0, 8.0, 50.0],
            'nearest_end_error': [2.0, 4.0, 3.0, 80.0],
            'terminal_confirmed': [False] * vs.SLOT_COUNT,
        }

        result = vs.trusted_mapping_by_exclusion(mapping, report)

        self.assertEqual(result, (mapping, 3))

    def test_trusted_motion_mapping_does_not_infer_when_two_ids_are_missing(self):
        mapping = {0: 1, 1: 2, 2: 3, 3: 0}
        report = {
            'frames': 10,
            'seen': [0.8, 0.8, 0.1, 0.1],
            'max_step': [10.0, 12.0, 50.0, 50.0],
            'nearest_end_error': [2.0, 4.0, 80.0, 70.0],
            'terminal_confirmed': [False] * vs.SLOT_COUNT,
        }

        self.assertIsNone(vs.trusted_mapping_by_exclusion(mapping, report))

    def test_row_colors_use_the_earliest_start_slot_not_detection_order(self):
        solver = vs.ViolettaSolver(SPRITES)
        solver.state = 'shuffle'
        rows = [row for row, _, _ in vs.LAYOUTS['classic']]
        starts = {0: vs.slot_x(1), 2: vs.slot_x(2),
                  3: vs.slot_x(3), 4: vs.slot_x(0)}
        recorder = vs.ShuffleRecorder(rows)
        for frame in range(2):
            recorder.add(frame, {row: (x, 0.8) for row, x in starts.items()})
        solver._recorders = {'classic': recorder}
        current = {
            4: (vs.slot_x(0) + 32, 0.8),
            3: (vs.slot_x(3) - 15, 0.8),
            0: (vs.slot_x(1) + 20, 0.8),
            2: (vs.slot_x(2) + 10, 0.8),
        }

        solver._start_overlay_tracking('classic', current)

        self.assertEqual({row: track['slot'] for row, track in solver._overlay_rows.items()},
                         {0: 1, 2: 2, 3: 3, 4: 0})


class PassThroughAmbiguityTests(unittest.TestCase):
    """A close pass is ambiguous only if the two could have swapped there."""

    @staticmethod
    def _track(slot, xs, scale=1.13, y=140.0, first=0):
        track = vt.Track(slot, first)
        track.path = [(first + i, float(x), scale, y, 0.7) for i, x in enumerate(xs)]
        return track

    def test_decoy_walking_straight_through_is_not_ambiguous(self):
        # 20261119-195802 round 1: she walks left, the decoy walks right past her
        her = self._track(3, [340 - 8 * i for i in range(16)])
        decoy = self._track(1, [210 + 8 * i for i in range(16)])
        self.assertEqual(vt._identity_ambiguities([her, decoy]), [False, False])

    def test_walker_passing_one_standing_still_is_not_ambiguous(self):
        walker = self._track(0, [180 + 8 * i for i in range(16)])
        stander = self._track(1, [240] * 16)
        self.assertEqual(vt._identity_ambiguities([walker, stander]), [False, False])

    def test_both_turning_round_where_they_meet_is_ambiguous(self):
        # A swapped pass looks like two walkers bouncing off each other
        left = self._track(0, [200 + 8 * i for i in range(8)] + [256 - 8 * i for i in range(8)])
        right = self._track(1, [320 - 8 * i for i in range(8)] + [264 + 8 * i for i in range(8)])
        self.assertEqual(vt._identity_ambiguities([left, right]), [True, True])

    def test_pace_handed_from_one_to_the_other_is_ambiguous(self):
        # One stops as the other, standing there, sets off the same way
        walker = self._track(0, [180 + 8 * i for i in range(8)] + [240] * 8)
        stander = self._track(1, [240] * 8 + [248 + 8 * i for i in range(8)])
        self.assertEqual(vt._identity_ambiguities([walker, stander]), [True, True])


class OpticalFlowSeedTests(unittest.TestCase):
    def test_new_round_resets_the_previous_target_flow(self):
        class FakeTracker:
            def start(self, frame):
                self.frame = frame

        solver = vs.ViolettaSolver(SPRITES)
        solver.tracker_mode = 'synchronous'
        solver._tracker = FakeTracker()
        solver._target_flow = object()
        solver._target_flow_frame = 5
        solver._target_flow_history = [(4, 120.0, 0.9)]

        solver._start_tracker(10)

        self.assertIsNone(solver._target_flow)
        self.assertIsNone(solver._target_flow_frame)
        self.assertEqual(solver._target_flow_history, [])

    def test_explicit_box_seeds_flow_without_opening_reveal(self):
        from src.detection.violetta_challenge import ViolettaTracker

        frame = np.random.RandomState(7).randint(
            0, 256, (200, 200, 3), dtype=np.uint8)
        tracker = ViolettaTracker.from_box(frame, (50, 40, 130, 160))

        self.assertTrue(tracker.visible)
        np.testing.assert_allclose(tracker.center, (90, 100))

    def test_lost_flow_reacquires_strong_match_at_a_slot_after_a_jump(self):
        from types import SimpleNamespace

        class FakeTracker:
            def snapshot(self):
                return {
                    'threshold': 0.5,
                    'tracks': [{
                        'slot': 2, 'age': 0, 'score': 0.58, 'matched': 21,
                        'x': vs.slot_x(0) + vt.SLOT_OFFSET + 16.7,
                        'y': 100.0, 'scale': 0.5,
                    }],
                }

        solver = vs.ViolettaSolver(SPRITES)
        solver.state, solver.target_slot = 'wait_shuffle', 2
        solver.calibration = vs.Calibration(1.0, (0, 0), 1.0)
        solver._tracker, solver._tracker_started = FakeTracker(), 5
        solver._target_flow = SimpleNamespace(
            visible=False, confidence=0.0, lost_frames=1,
            center=np.array((280, 150), dtype=np.float32),
            width=45.0, height=50.0,
        )
        solver._target_flow_history = [(1, 200.0, 0.9)]
        frame = np.random.RandomState(8).randint(
            0, 256, (318, 441, 3), dtype=np.uint8)

        solver._seed_target_flow(10, frame)

        self.assertTrue(solver._target_flow.visible)
        self.assertEqual(len(solver._target_flow_history), 1)
        self.assertEqual(solver._flow_seed_key, (5, 21))

    def test_visible_flow_is_not_reseeded_from_appearance_matches(self):
        from types import SimpleNamespace

        class FakeTracker:
            def snapshot(self):
                return {
                    'threshold': 0.5,
                    'tracks': [{
                        'slot': 2, 'age': 0, 'score': 0.9, 'matched': 21,
                        'x': vs.slot_x(3) + vt.SLOT_OFFSET,
                        'y': 100.0, 'scale': 0.5,
                    }],
                }

        solver = vs.ViolettaSolver(SPRITES)
        solver.state, solver.target_slot = 'wait_shuffle', 2
        solver._tracker, solver._tracker_started = FakeTracker(), 5
        flow = SimpleNamespace(visible=True)
        solver._target_flow = flow
        solver._target_flow_history = [(1, 200.0, 0.9)]

        solver._seed_target_flow(10, np.zeros((318, 441, 3), dtype=np.uint8))

        self.assertIs(solver._target_flow, flow)
        self.assertEqual(solver._target_flow_history, [(1, 200.0, 0.9)])
        self.assertIsNone(solver._flow_seed_key)

    def test_lost_flow_does_not_reacquire_a_weak_match(self):
        from types import SimpleNamespace

        class FakeTracker:
            def snapshot(self):
                return {
                    'threshold': 0.5,
                    'tracks': [{
                        'slot': 2, 'age': 0, 'score': 0.55, 'matched': 21,
                        'x': vs.slot_x(0) + vt.SLOT_OFFSET + 16.7,
                        'y': 100.0, 'scale': 0.5,
                    }],
                }

        solver = vs.ViolettaSolver(SPRITES)
        solver.state, solver.target_slot = 'wait_shuffle', 2
        solver.calibration = vs.Calibration(1.0, (0, 0), 1.0)
        solver._tracker, solver._tracker_started = FakeTracker(), 5
        previous_flow = SimpleNamespace(
            visible=False, confidence=0.0, lost_frames=1,
            center=np.array((280, 150), dtype=np.float32),
            width=45.0, height=50.0,
        )
        solver._target_flow = previous_flow
        solver._target_flow_history = [(1, 200.0, 0.9)]
        frame = np.random.RandomState(9).randint(
            0, 256, (318, 441, 3), dtype=np.uint8)

        solver._seed_target_flow(10, frame)

        self.assertIs(solver._target_flow, previous_flow)
        self.assertEqual(solver._target_flow_history, [(1, 200.0, 0.9)])


class RevealArrowTests(unittest.TestCase):
    @staticmethod
    def _draw_arrow(frame, slot):
        x = int(round(vs.slot_x(slot) + 18))
        points = np.array([
            [x - 12, 65], [x + 12, 65], [x, 84],
            [x + 5, 84], [x + 5, 105], [x - 5, 105], [x - 5, 84], [x, 84],
        ], np.int32)
        cv2.fillPoly(frame, [points], (0, 150, 255))

    def test_finds_the_unique_reveal_arrow(self):
        frame = np.zeros((318, 441, 3), np.uint8)
        self._draw_arrow(frame, 2)

        slot = vs.find_reveal_arrow(frame, vs.Calibration(1.0, (0, 0), 1.0))

        self.assertEqual(slot, 2)

    def test_rejects_ambiguous_reveal_arrows(self):
        frame = np.zeros((318, 441, 3), np.uint8)
        self._draw_arrow(frame, 1)
        self._draw_arrow(frame, 2)

        slot = vs.find_reveal_arrow(frame, vs.Calibration(1.0, (0, 0), 1.0))

        self.assertIsNone(slot)


class LayoutTests(unittest.TestCase):
    """The classic (simulator) and light (live game) walking-row layouts."""

    def test_row_ids_are_unique_and_each_layout_has_four_walkers(self):
        rows = [row for layout in vs.LAYOUTS.values() for row, _, _ in layout]
        self.assertEqual(len(rows), len(set(rows)))
        for layout in vs.LAYOUTS.values():
            self.assertEqual(len(layout), vs.SLOT_COUNT)

    def test_front_row_is_the_biggest(self):
        self.assertEqual(vs.front_row([row for row, _, _ in vs.LAYOUTS['classic']]), 0)
        self.assertEqual(vs.front_row([row for row, _, _ in vs.LAYOUTS['light']]), 10)

    def test_light_layout_shuffle_resolves(self):
        # The live game's shuffle as measured: A stays in slot 1, B 2->3, C 3->4, D 4->2
        rows = [row for row, _, _ in vs.LAYOUTS['light']]
        starts, ends = [56, 159, 241, 332], [56, 241, 345, 142]
        recorder = vs.ShuffleRecorder(rows)
        for step in range(40):
            t = step / 39
            recorder.add(step, {row: (round(a + (b - a) * t), 0.65)
                                for row, a, b in zip(rows, starts, ends)})
        mapping, report = recorder.resolve()
        self.assertTrue(report['consistent'])
        self.assertEqual(mapping[11], (1, 2))       # the target's walk: slot 2 -> slot 3
        self.assertEqual(mapping[10], (0, 0))       # the one that stays


class UnknownRowsTests(unittest.TestCase):
    """Shuffles whose walking rows are not in LAYOUTS (src/detection/violetta_rows.py)."""

    def test_follows_her_when_every_round_has_new_rows(self):
        for seed, scale in zip(range(4), (1.0, 2.0, 1.0, 1.5)):
            with self.subTest(seed=seed, scale=scale):
                size = (int(441 * scale) + 60, int(318 * scale) + 40)
                game = RandomRowsGame(seed, scale, (30, 20), size)
                solver = vs.ViolettaSolver(SPRITES, rounds=2)
                answers, truth = [], []
                for frame in game.frames(rounds=2, mark_rounds=True):
                    if frame is None:   # the buttons lit up: answer like play_live does
                        truth.append(game.answer)
                        answer = solver.force_answer()
                        if answer is not None:
                            answers.append(answer)
                        continue
                    answer = solver.update(frame)
                    if answer is not None:
                        answers.append(answer)
                self.assertEqual(answers, truth, solver.events)

    def test_path_skips_stray_matches_and_bridges_hidden_stretches(self):
        from src.detection.violetta_rows import longest_path
        walk = [(i, 40 + 5 * i) for i in range(10, 20)] + [(i, 40 + 5 * i) for i in range(32, 40)]
        strays = [(2, 350), (5, 30), (45, 250)]
        path = longest_path(sorted(walk + strays))
        self.assertEqual(path, walk)

    def test_unseen_violettas_take_the_slots_left_over_and_prefer_staying(self):
        from src.detection.violetta_rows import _resolve
        # Row 0 walks slot 3 -> slot 2, row 1 walks slot 4 -> slot 3; the two
        # never seen walking: slot 1 stays, slot 2 must have gone to slot 4
        tracks = [{i: (252 - 104 * i / 19, 0.7) for i in range(20)},
                  {i: (356 - 104 * i / 19, 0.7) for i in range(20)}]
        mapping, report, _ = _resolve(tracks, [0, 1])
        self.assertTrue(report['consistent'])
        self.assertEqual(mapping['r0'], (2, 1))
        self.assertEqual(mapping['r1'], (3, 2))
        self.assertEqual(sorted(v for k, v in mapping.items() if k.startswith('unseen')),
                         [(0, 0), (1, 3)])


class ParkPointTests(unittest.TestCase):
    """Where the idle cursor goes: off the game window, never over the stage."""

    def _inside_window(self, calibration, point):
        x0, y0 = calibration.to_frame((0, 0))
        x1, y1 = calibration.to_frame(vs.WINDOW_SIZE)
        return x0 <= point[0] <= x1 and y0 <= point[1] <= y1

    def test_parks_right_of_the_window(self):
        calibration = vs.Calibration(1.0, (100, 50), 1.0)
        point = vs.park_point(calibration, {'width': 1366, 'height': 768})
        self.assertIsNotNone(point)
        self.assertGreater(point[0], 100 + vs.WINDOW_SIZE[0])
        self.assertFalse(self._inside_window(calibration, point))

    def test_parks_left_when_the_window_touches_the_right_edge(self):
        calibration = vs.Calibration(1.5, (700, 100), 1.0)
        region = {'width': int(700 + 1.5 * vs.WINDOW_SIZE[0]) + 10, 'height': 768}
        point = vs.park_point(calibration, region)
        self.assertIsNotNone(point)
        self.assertLess(point[0], 700)
        self.assertFalse(self._inside_window(calibration, point))

    def test_parks_below_when_the_window_fills_the_width(self):
        calibration = vs.Calibration(1.0, (0, 0), 1.0)
        point = vs.park_point(calibration, {'width': vs.WINDOW_SIZE[0], 'height': 768})
        self.assertIsNotNone(point)
        self.assertGreater(point[1], vs.WINDOW_SIZE[1])

    def test_no_spot_when_the_window_fills_the_client(self):
        calibration = vs.Calibration(1.0, (0, 0), 1.0)
        region = {'width': vs.WINDOW_SIZE[0], 'height': vs.WINDOW_SIZE[1]}
        self.assertIsNone(vs.park_point(calibration, region))


class RealLabelTests(unittest.TestCase):
    """The Violetta the bot believes is real gets a REAL tag (GUESS when unsure)."""

    def _solver(self, confident):
        solver = vs.ViolettaSolver(SPRITES, rounds=2)
        solver.calibration = vs.Calibration(1.0, (0, 0), 1.0)
        solver.state, solver.target_slot, solver.confident = 'wait_shuffle', 2, confident
        return solver

    def test_trusted_pick_is_tagged_real(self):
        boxes, labels = self._solver(True).overlay_shapes()
        self.assertEqual(boxes[-1][4], 'target')
        self.assertEqual(labels[-1][2:], (vs.REAL_LABEL, 'target'))
        # around her own box (ID3: slot 3 at the start)
        own = next(box for box in boxes if box[4] == 'id3')
        self.assertLess(boxes[-1][0], own[0])
        self.assertGreater(boxes[-1][2], own[2])

    def test_unsure_pick_is_tagged_guess(self):
        boxes, labels = self._solver(False).overlay_shapes()
        self.assertEqual(boxes[-1][4], 'guess')
        self.assertEqual(labels[-1][2:], (vs.GUESS_LABEL, 'guess'))

    def test_replay_drawing_accepts_the_tag(self):
        frame = np.zeros((360, 480, 3), np.uint8)
        drawn = self._solver(True).draw(frame)
        self.assertTrue((drawn == vs.TARGET_BGR).all(axis=2).any())


if __name__ == '__main__':
    unittest.main()


class TrustedTrackTests(unittest.TestCase):
    """Trusting the make-up Violetta's own track when the others aren't all clean."""

    REPORT = {'frames': 20, 'seen': [0.31, 0.69, 0.41, 0.34],
              'max_step': [30.0, 4.9, 60.0, 8.0],
              'end_error': [2.0, 8.4, 40.0, 3.0]}       # violetta-20261115-140537 round 2

    def test_her_clean_track_is_trusted_even_if_others_are_lost(self):
        from src.detection.violetta_solver import trusted_track
        self.assertTrue(trusted_track({0: 0, 1: 1, 2: 2, 3: 3}, self.REPORT, 1))

    def test_rarely_seen_track_is_not(self):
        from src.detection.violetta_solver import trusted_track
        self.assertFalse(trusted_track({0: 0, 1: 1, 2: 2, 3: 3}, self.REPORT, 0))

    def test_a_jump_onto_another_violetta_is_not(self):
        from src.detection.violetta_solver import trusted_track
        report = dict(self.REPORT, max_step=[30.0, 74.2, 60.0, 8.0])   # 213630 round 3
        self.assertFalse(trusted_track({0: 0, 1: 1, 2: 2, 3: 3}, report, 1))

    def test_long_gap_uses_speed_but_a_close_crossing_stays_ambiguous(self):
        from src.detection.violetta_solver import trusted_track
        mapping = {0: 0, 1: 1, 2: 2, 3: 3}
        report = dict(
            self.REPORT,
            max_step=[30.0, 39.8, 60.0, 8.0],
            max_speed=[10.0, 13.3, 10.0, 8.0],
            ambiguous=[False] * vs.SLOT_COUNT,
        )
        self.assertTrue(trusted_track(mapping, report, 1))

        report['ambiguous'][1] = True
        self.assertFalse(trusted_track(mapping, report, 1))

    def test_ending_between_slots_is_not(self):
        from src.detection.violetta_solver import trusted_track
        report = dict(self.REPORT, end_error=[2.0, 111.4, 40.0, 3.0])  # 140537 round 1, sparse
        self.assertFalse(trusted_track({0: 0, 1: 1, 2: 2, 3: 3}, report, 1))

    def test_two_frame_track_is_not_trusted(self):
        from src.detection.violetta_solver import trusted_track
        report = dict(self.REPORT, frames=2)
        self.assertFalse(trusted_track({0: 0, 1: 1, 2: 2, 3: 3}, report, 1))

    def test_two_frame_tracker_result_is_marked_inconsistent(self):
        class ShortTracker:
            def result(self):
                return ({0: 0, 1: 1, 2: 2, 3: 3}, {'frames': 2, 'consistent': True})

        solver = vs.ViolettaSolver(SPRITES)
        solver._tracker = ShortTracker()
        _, report = solver._tracked()

        self.assertFalse(report['consistent'])
        self.assertTrue(any('only 2 frames' in message for _, message in solver.events))

    def test_missing_numbers_are_not(self):
        from src.detection.violetta_solver import trusted_track
        self.assertFalse(trusted_track(None, self.REPORT, 1))
        self.assertFalse(trusted_track({0: 0}, {'seen': [0.9]}, 0))


class ReplayOverlayTests(unittest.TestCase):
    def test_click_ripple_is_drawn_then_expires(self):
        frame = np.zeros((318, 441, 3), np.uint8)

        vs._draw_click_ripple(frame, (220, 270), 3, 0)

        magenta = (frame[:, :, 0] > 0) & (frame[:, :, 1] == 0) & (frame[:, :, 2] > 0)
        self.assertGreater(np.count_nonzero(magenta), 0)
        expired = np.zeros_like(frame)
        vs._draw_click_ripple(expired, (220, 270), 3, vs.CLICK_RIPPLE_FRAMES)
        self.assertFalse(np.any(expired))


class UnsureLostTrackTests(unittest.TestCase):
    """20261120-100529 round 1: her track went lost 2.6 px from slot 4 as the
    buttons lit; the old fallback ignored it and guessed her start slot."""

    def report(self, lost, seen=0.86, error=2.6, nearest=3):
        return {'seen': [seen, 0.88, 0.87, 0.93], 'lost': [lost, False, True, False],
                'nearest_end_error': [error, 3.4, 2.9, 8.4],
                'nearest_end_slot': [nearest, 1, 0, 2]}

    def test_lost_on_her_slot_is_usable(self):
        self.assertTrue(vs._usable_unsure_track({0: 3}, self.report(True), 0))

    def test_lost_mid_walk_is_not(self):
        self.assertFalse(vs._usable_unsure_track({0: 3}, self.report(True, error=40), 0))

    def test_lost_poorly_seen_is_not(self):
        self.assertFalse(vs._usable_unsure_track({0: 3}, self.report(True, seen=0.6), 0))

    def test_lost_mapped_off_her_slot_is_not(self):
        self.assertFalse(vs._usable_unsure_track({0: 1}, self.report(True), 0))

    def test_followed_track_unchanged(self):
        self.assertTrue(vs._usable_unsure_track({0: 3}, self.report(False, seen=0.6, error=30), 0))
