"""
End-to-end tests for the Violetta solver on synthetic games rendered from the
game's own sprites: random window scale and position, random shuffles that
follow the game's rules, and colour-cycling stage lights over the characters.
"""

import random
import unittest

import cv2
import numpy as np

from src.detection import violetta_solver as vs


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

    def frames(self, rounds=1, leg_frames=30):
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
                    sc0, y0 = vs.row_geometry(r0)
                    sc1, y1 = vs.row_geometry(r1)
                    x = vs.slot_x(s0) + (vs.slot_x(s1) - vs.slot_x(s0)) * part + self.offsets[c][0]
                    y = y0 + (y1 - y0) * part + self.offsets[c][1]
                    placements.append((WALK, x, y, sc0 + (sc1 - sc0) * part))
                yield self._frame(self._window(placements, True, tick))
                tick += 1
            self.slots = [p[-1][0] for p in paths]
            for _ in range(30):                     # waiting for the pick
                yield self._frame(self._window(self._standing(False), True, tick))
                tick += 1

    @property
    def answer(self):
        return self.slots[self.special]


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


if __name__ == '__main__':
    unittest.main()
