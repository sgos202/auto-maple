"""Tests for climbing back to the patrol platform, on a small simulated map."""

import unittest

from src.modules.recovery import LevelRecovery


class World:
    """Three levels joined by ropes; minimap units, y grows downwards."""

    PLATFORMS = [   # (y, x_low, x_high), top first
        (0.56, 0.20, 0.78),     # home
        (0.69, 0.15, 0.85),
        (0.86, 0.00, 1.00),     # ground
    ]
    ROPES = [       # (x, bottom_y, top_y)
        (0.37, 0.69, 0.56),
        (0.53, 0.86, 0.69),
    ]
    WALK = 0.10     # per second
    CLIMB = 0.08

    def __init__(self, x, y):
        self.x, self.y = x, y
        self.time = 0.0
        self.keys = set()
        self.rope = None
        self.presses = []

    def key_down(self, key):
        self.keys.add(key)

    def key_up(self, key):
        self.keys.discard(key)

    def press(self, key, count, **_):
        self.presses.append(key)
        if key == 'jump' and 'down' in self.keys and self.rope is None:
            lower = [p for p in self.PLATFORMS if p[0] > self.y + 0.01 and p[1] <= self.x <= p[2]]
            if lower:
                self.y = lower[0][0]

    def position(self):
        return (self.x, self.y)

    def clock(self):
        return self.time

    def sleep(self, seconds):
        step = 0.01
        for _ in range(max(1, round(seconds / step))):
            self.time += step
            if self.rope is not None:
                if 'up' in self.keys:
                    self.y = max(self.rope[2], self.y - self.CLIMB * step)
                    if self.y <= self.rope[2]:
                        self.rope = None
                continue
            if 'left' in self.keys:
                self.x -= self.WALK * step
            if 'right' in self.keys:
                self.x += self.WALK * step
            self.x = min(max(self.x, 0.0), 1.0)
            level = next(p for p in self.PLATFORMS if abs(p[0] - self.y) < 0.005)
            if not level[1] <= self.x <= level[2]:      # walked off the edge
                self.y = next(p[0] for p in self.PLATFORMS if p[0] > self.y + 0.01)
            if 'up' in self.keys:
                for rope in self.ROPES:
                    if abs(rope[0] - self.x) < 0.006 and abs(rope[1] - self.y) < 0.005:
                        self.rope = rope


def make(world, home=0.56, x_range=(0.215, 0.771)):
    logs = []
    recovery = LevelRecovery(world.key_down, world.key_up, world.press, world.position,
                             jump_key='jump', clock=world.clock, sleep=world.sleep,
                             log=logs.append)
    recovery.set_home(home, *x_range)
    return recovery, logs


class TestLevelRecovery(unittest.TestCase):
    def test_brief_dip_is_not_a_fall_but_a_long_one_is(self):
        world = World(0.6, 0.69)
        recovery, _ = make(world)
        self.assertFalse(recovery.fallen())
        world.time += 0.5
        self.assertFalse(recovery.fallen())
        world.time += 0.6
        self.assertTrue(recovery.fallen())
        world.y = 0.56                      # back on the platform
        self.assertFalse(recovery.fallen())
        world.y = 0.50                      # a jump (higher) never counts
        world.time += 5
        self.assertFalse(recovery.fallen())

    def test_climbs_back_from_the_platform_below(self):
        world = World(0.60, 0.69)
        recovery, logs = make(world)
        self.assertTrue(recovery.recover())
        self.assertAlmostEqual(world.y, 0.56, delta=0.01)
        self.assertEqual(world.keys, set())             # nothing left held
        self.assertEqual(len(recovery.ropes), 1)
        self.assertAlmostEqual(recovery.ropes[0][0], 0.37, delta=0.02)

    def test_climbs_two_levels_from_the_ground(self):
        world = World(0.30, 0.86)
        recovery, _ = make(world)
        self.assertTrue(recovery.recover())
        self.assertAlmostEqual(world.y, 0.56, delta=0.01)
        self.assertEqual(world.keys, set())
        self.assertEqual(len(recovery.ropes), 2)

    def test_stays_inside_the_home_range_away_from_portals(self):
        world = World(0.75, 0.86)
        recovery, _ = make(world)
        xs = []
        sleep = world.sleep

        def spy(seconds):
            sleep(seconds)
            xs.append(world.x)
        recovery._sleep = spy
        self.assertTrue(recovery.recover())
        self.assertGreater(min(xs), 0.15)
        self.assertLess(max(xs), 0.85)

    def test_remembered_rope_is_used_directly_next_time(self):
        world = World(0.60, 0.69)
        recovery, _ = make(world)
        recovery.recover()
        first = world.time
        world.x, world.y, world.time = 0.30, 0.69, 0.0
        recovery.recover()
        self.assertAlmostEqual(world.y, 0.56, delta=0.01)
        self.assertLess(world.time, first + 0.5)

    def test_gives_up_after_timeout_and_waits_before_retrying(self):
        world = World(0.60, 0.69)
        world.ROPES = []
        recovery, logs = make(world)
        self.assertFalse(recovery.recover())
        self.assertEqual(world.keys, set())
        self.assertFalse(recovery.fallen())             # cooling down
        world.time += recovery.RETRY_SECONDS + recovery.FALL_SECONDS + 1
        self.assertTrue(recovery.fallen())

    def test_learns_platform_range_while_patrolling(self):
        world = World(0.5, 0.56)
        recovery, _ = make(world, x_range=(None, None))
        recovery.set_home(0.56)
        for x in (0.3, 0.5, 0.7):
            recovery.observe((x, 0.56))
        recovery.observe((0.9, 0.86))                   # off the platform: ignored
        self.assertEqual(recovery.range, (0.3, 0.7))


if __name__ == '__main__':
    unittest.main()
