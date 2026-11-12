"""
Tests for the fading-figure tracker on synthetic games: a textured background
covered in decoy outlines scrolls as one layer; a white figure shows during the
countdown, fades, and then is only visible as a lens that warps the background
while it wanders on its own path.
"""

import unittest

import cv2
import numpy as np

from src.detection import fading_figure as ff


def _star(radius, angle=0.0, points=5):
    pts = []
    for i in range(points * 2):
        r = radius if i % 2 == 0 else radius * 0.45
        a = angle + i * np.pi / points
        pts.append((r * np.sin(a), -r * np.cos(a)))
    return np.array(pts, np.float32)


def _world(size, rng, figure_radius):
    """A large textured background with decoy outlines of the figure drawn on it."""
    noise = rng.normal(0, 1, (size // 8, size // 8)).astype(np.float32)
    texture = cv2.resize(noise, (size, size), interpolation=cv2.INTER_CUBIC)
    texture = cv2.GaussianBlur(texture, (0, 0), 3)
    texture = (texture - texture.min()) / (np.ptp(texture) + 1e-6)
    base = np.dstack([40 + 90 * texture, 80 + 90 * texture, 120 + 90 * texture]).astype(np.uint8)  # sand tones
    for _ in range(size * size // 9000):
        centre = rng.uniform(0, size, 2)
        poly = _star(figure_radius * rng.uniform(0.8, 1.2), rng.uniform(0, np.pi)) + centre
        cv2.polylines(base, [poly.astype(np.int32)], True, (60, 80, 100), 3, cv2.LINE_AA)
        cv2.polylines(base, [poly.astype(np.int32)], True, (150, 185, 215), 1, cv2.LINE_AA)
    return base


def _game(seed, frames=200, width=480, height=360, radius=26, white_frames=50, fade_frames=20):
    """Renders a game; returns (frames, true centres)."""
    rng = np.random.default_rng(seed)
    world = _world(1400, rng, radius)
    camera = np.array((450.0, 450.0))
    camera_velocity = rng.normal(0, 3, 2)
    figure = np.array((width / 2, height / 2), float)
    velocity = np.zeros(2)
    out, truth = [], []
    for i in range(frames):
        camera_velocity = 0.95 * camera_velocity + rng.normal(0, 0.6, 2)
        camera = np.clip(camera + camera_velocity, 50, 1400 - max(width, height) - 50)
        if i >= white_frames:  # the figure starts wandering when it fades
            velocity = 0.9 * velocity + rng.normal(0, 0.5, 2)
            velocity = np.clip(velocity, -4, 4)
            figure = figure + velocity
            for k, limit in ((0, width), (1, height)):
                if not 2 * radius < figure[k] < limit - 2 * radius:
                    velocity[k] = -velocity[k]
                    figure[k] = np.clip(figure[k], 2 * radius, limit - 2 * radius)
        x0, y0 = camera.astype(int)
        view = world[y0:y0 + height, x0:x0 + width].copy()
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [(_star(radius, 0.3 + 0.01 * i) + figure).astype(np.int32)], 255, cv2.LINE_AA)
        # lens: the background inside the figure is magnified about the figure's centre
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        map_x = (figure[0] + (xx - figure[0]) * 0.6).astype(np.float32)
        map_y = (figure[1] + (yy - figure[1]) * 0.6).astype(np.float32)
        lens = cv2.remap(view, map_x, map_y, cv2.INTER_LINEAR)
        alpha = (mask.astype(np.float32) / 255)[..., None]
        view = (view * (1 - alpha) + lens * alpha).astype(np.uint8)
        if i < white_frames + fade_frames:
            white = 1.0 if i < white_frames else 1 - (i - white_frames + 1) / fade_frames
            view = (view * (1 - alpha * white) + 250 * alpha * white).astype(np.uint8)
        out.append(view)
        truth.append(figure.copy())
    return out, truth


class TestFadingFigure(unittest.TestCase):
    def _run(self, seed):
        frames, truth = _game(seed)
        tracker = ff.FadingFigureTracker()
        errors = []
        for i, frame in enumerate(frames):
            position = tracker.update(frame, i / 30.0)
            if i >= 75 and position is not None:  # after the fade
                errors.append(np.linalg.norm(position - truth[i]))
        return tracker, np.array(errors)

    def test_finds_white_figure_and_its_size(self):
        frames, truth = _game(1, frames=30)
        tracker = ff.FadingFigureTracker()
        for i, frame in enumerate(frames):
            position = tracker.update(frame, i / 30.0)
        self.assertEqual(tracker.state, 'white')
        self.assertLess(np.linalg.norm(position - truth[-1]), 3)
        self.assertAlmostEqual(tracker.radius, 26 * 0.72, delta=4)  # equivalent radius of the star

    def test_follows_lens_after_fade(self):
        for seed in (2, 3, 4):
            with self.subTest(seed=seed):
                tracker, errors = self._run(seed)
                self.assertEqual(tracker.state, 'lens')
                radius = tracker.radius
                inside = np.mean(errors < 1.5 * radius)
                self.assertGreaterEqual(inside, 0.9, f'only {inside:.0%} within 1.5r; '
                                                     f'median error {np.median(errors):.1f}px')

    def test_human_mouse_is_smooth_and_close(self):
        mouse = ff.HumanMouse(seed=0)
        points = []
        for i in range(120):
            target = np.array((200 + 3 * i, 150 + 20 * np.sin(i / 10)))
            points.append((target, mouse.step(target, i / 30.0)))
        errors = [np.linalg.norm(t - p) for t, p in points[20:]]
        steps = [np.linalg.norm(points[i + 1][1] - points[i][1]) for i in range(len(points) - 1)]
        self.assertLess(np.median(errors), 10)
        self.assertLess(max(steps), 900 / 30.0 + 5)  # never teleports


if __name__ == '__main__':
    unittest.main()
