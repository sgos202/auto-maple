import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from src.common import config
from src.modules import channel as channel_module
from src.modules import map_events
from src.modules.channel import (
    ChangeChannel, FindQuietChannel, cell_centre, count_other_players,
    current_channel, find_ui, DIALOG_BUTTON, MENU_BUTTON, MENU_CHANGE_CHANNEL)
from src.modules.plan_steps import (
    DONE, FAILED, RUNNING, Entry, HideUntilGone, HoldOnRope, LeaveRope, MountRope, Plan, Step,
    Wait, WalkOff)

VIDEO = os.path.join('assets', 'iron_hog_issue.mp4')


def _frames(path, indices):
    video = cv2.VideoCapture(path)
    frames, index = {}, 0
    wanted = set(indices)
    while wanted:
        ok, frame = video.read()
        if not ok:
            break
        if index in wanted:
            frames[index] = frame
            wanted.discard(index)
        index += 1
    return frames


_CLIP = {}


def clip():
    """Frames of the recording: 0 hog by the rope (MENU visible), 42 game menu
    open, 90 channel list (CH.16 highlighted), 160 loading screen, 300 new
    channel with another player on the minimap."""
    if not _CLIP:
        _CLIP.update(_frames(VIDEO, (0, 30, 42, 90, 160, 300)))
    return _CLIP


def minimap_of(frame):
    return frame[31:114, 3:214]     # the minimap's map area in the recording


_stay_patch = None


def setUpModule():
    # Tests run with channel changes allowed, whatever .settings/channel says
    global _stay_patch
    _stay_patch = patch.object(map_events, 'stay_on_channel', return_value=False)
    _stay_patch.start()


def tearDownModule():
    _stay_patch.stop()


class FakeIO:
    """A tiny Pig Beach: walking, taps, jumps, a rope and knockbacks."""

    # Measured on iron_hog_escape.mp4 (the rope at 0.284, ground 0.256)
    WALK_SPEED = 0.045          # minimap units a second
    JUMP_SPEED = 0.08           # sideways in a running jump (catches came 0.1-0.33 s
                                # after take-offs 0.019-0.024 out)
    TAP = 0.003
    CLIMB_SPEED = 0.03
    GROUND_Y = 0.256
    AIR = 0.45
    JUMP_HEIGHT = 0.033         # top of a jump 0.222
    ROPE_BOTTOM = 0.022         # the rope's end hangs this high: only reachable near the top
    GRAB = 0.004
    AIR_ACCEL = 0.12            # an arrow held in the air bends the arc this much (units/s^2)
    AIR_HIT_PUSH = 0.008        # a pig's hit mid-jump throws it back this far

    def __init__(self, x=0.40, y=GROUND_Y, rope_x=0.284, knockbacks=0, air_hits=0):
        self.t = 100.0
        self.x, self.y = x, y
        self.rope_x = rope_x
        self.walking = None
        self.held = set()
        self.on_rope = False
        self.air_until = None
        self.air_dir = None
        self.jumped_at = None
        self.knockbacks = knockbacks
        self.air_hits = air_hits
        self.vx = 0.0
        self.hit_this_jump = False
        self.logs, self.presses, self.clicks = [], [], []
        self.jump_key = 'alt'
        self.frame_value = None
        self.minimap_value = None
        self.black_until = 0.0
        self.pos_time = self.t

    # ---- Controls API ----
    def now(self):
        return self.t

    def position(self):
        return (round(self.x, 4), round(self.y, 4))

    def position_time(self):
        return self.pos_time

    def walk(self, direction):
        self.walking = direction

    def key_down(self, key):
        self.held.add(key)

    def key_up(self, key):
        self.held.discard(key)

    def press(self, key, n=1, down_time=0.05, up_time=0.05):
        self.presses.append((round(self.t, 2), key))
        if key in ('left', 'right') and not self.on_rope:
            self.x += self.TAP if key == 'right' else -self.TAP
        if key == self.jump_key and self.on_rope and self.walking in ('left', 'right'):
            # jumping off a rope sideways (arrow held): fall from it to the ground
            self.on_rope = False
            self.jumped_at = self.t
            self.air_until = self.t + self.AIR
            self.air_dir = self.walking
            self.vx = self._speed(self.walking)
            self.hit_this_jump = True
            return
        if key == self.jump_key and self.air_until is None and not self.on_rope:
            if self.knockbacks:
                self.knockbacks -= 1
                # a pig hit it at take-off: thrown back against the run
                self.x += -0.03 if self.walking == 'right' else 0.03
                return
            # A real jump (iron_hog_issue-2.mp4): ~0.6 s in the air, top at
            # 0.3 s, running speed kept; the rope's tip is only reachable near
            # the top, within GRAB of its x
            self.jumped_at = self.t
            self.air_until = self.t + self.AIR
            self.air_dir = self.walking
            self.vx = self._speed(self.walking)
            self.hit_this_jump = False

    def _speed(self, direction, speed=None):
        speed = self.JUMP_SPEED if speed is None else speed
        return {'right': speed, 'left': -speed}.get(direction, 0.0)

    def frame(self):
        return self.frame_value

    def minimap(self):
        return self.minimap_value

    def click(self, x, y):
        self.clicks.append((round(x), round(y)))

    def expect_black_screen(self, seconds):
        self.black_until = self.t + seconds

    def log(self, message):
        self.logs.append(message)

    def release(self):
        self.walking = None
        self.held.clear()

    # ---- the world ----
    def advance(self, dt=0.02):
        self.t += dt
        self.pos_time = self.t
        if self.on_rope:
            if 'up' in self.held:
                self.y -= self.CLIMB_SPEED * dt
            return
        if self.air_until is not None:
            # In the air the run's speed carries on; a held arrow bends it a little
            if self.walking is not None:
                change = self._speed(self.walking) - self.vx
                limit = self.AIR_ACCEL * dt
                self.vx += max(-limit, min(limit, change))
            k = (self.t - self.jumped_at) / self.AIR       # 0..1 through the jump
            if self.air_hits and not self.hit_this_jump and k >= 0.2:
                # a pig's hit on the way up: thrown back, the top lands short
                self.air_hits -= 1
                self.hit_this_jump = True
                self.x -= self.AIR_HIT_PUSH if self.vx >= 0 else -self.AIR_HIT_PUSH
            self.x += self.vx * dt
        elif self.walking == 'right':
            self.x += self.WALK_SPEED * dt
        elif self.walking == 'left':
            self.x -= self.WALK_SPEED * dt
        if self.air_until is not None:
            self.y = self.GROUND_Y - self.JUMP_HEIGHT * 4 * k * (1 - k)
            if 'up' in self.held and self.GROUND_Y - self.y >= self.ROPE_BOTTOM \
                    and abs(self.x - self.rope_x) <= self.GRAB:
                self.on_rope = True
                self.air_until = None
                return
            if self.t >= self.air_until:
                self.air_until = None
                self.y = self.GROUND_Y


def run_step(step, io, seconds, dt=0.02):
    status = RUNNING
    while io.t < 100.0 + seconds:
        status = step.run(io)
        if status != RUNNING:
            step.end(io)
            return status
        io.advance(dt)
    return status


class PlanTests(unittest.TestCase):
    class Fails(Step):
        name = 'fails'

        def __init__(self, times):
            super().__init__()
            self.times = times

        def tick(self, io):
            if self.times:
                self.times -= 1
                return self.fail('nope')
            return DONE

    def test_steps_run_in_order(self):
        io = FakeIO()
        plan = Plan('p', [Entry(Wait(0.1)), Entry(Wait(0.1))])
        status = RUNNING
        while status == RUNNING:
            status = plan.tick(io)
            io.advance()
        self.assertEqual(status, DONE)
        self.assertEqual([name for name, _ in plan.log], ['wait', 'wait'])

    def test_failure_jumps_back_then_falls_through_when_exhausted(self):
        io = FakeIO()
        plan = Plan('p', [
            Entry(Wait(0), label='start'),
            Entry(self.Fails(5), on_fail='start', retries=2, exhausted='end'),
            Entry(self.Fails(99), on_fail='abort'),
            Entry(Wait(0), label='end'),
        ])
        status = RUNNING
        while status == RUNNING:
            status = plan.tick(io)
        self.assertEqual(status, DONE)
        outcomes = [outcome for name, outcome in plan.log if name == 'fails']
        self.assertEqual(len(outcomes), 3)  # two retries, then on to 'end'

    def test_abort_fails_the_plan(self):
        io = FakeIO()
        plan = Plan('p', [Entry(self.Fails(1), on_fail='abort'), Entry(Wait(0))])
        self.assertEqual(plan.tick(io), FAILED)
        self.assertIn('nope', plan.reason)


ROPE = dict(rope_x=0.284, mount_y=0.256, safe_y=0.213)
JUMPS = dict(jump_left=0.265, jump_right=0.303)


def pig_beach_rope(**kw):
    """MountRope for the Pig Beach rope as recorded in iron_hog_escape.mp4."""
    args = dict(ROPE, **JUMPS)
    args.update(kw)
    return MountRope(args.pop('rope_x'), args.pop('mount_y'), args.pop('safe_y'), **args)


class RopeTests(unittest.TestCase):
    def test_walks_to_the_rope_jumps_and_climbs_to_the_safe_spot(self):
        io = FakeIO(x=0.40)
        step = pig_beach_rope()
        self.assertEqual(run_step(step, io, 15), DONE)
        self.assertTrue(io.on_rope)
        self.assertLessEqual(io.y, 0.213 + MountRope.SAFE_TOLERANCE)
        self.assertNotIn('up', io.held)     # let go of up: hangs there
        self.assertIn('alt', [key for _, key in io.presses])

    def test_takes_off_at_the_recorded_jump_points(self):
        for start, takeoff in ((0.20, 0.265), (0.36, 0.303)):
            io = FakeIO(x=start)
            step = pig_beach_rope()
            while getattr(step, 'phase', None) != 'air':
                step.run(io)
                io.advance()
            self.assertAlmostEqual(step.takeoff_x, takeoff, delta=0.004)

    def test_a_plain_jump_top_is_not_mistaken_for_the_safe_spot(self):
        # 0.222 is the top of a jump there; the safe spot is 0.213
        self.assertLess(0.213 + MountRope.SAFE_TOLERANCE, 0.222)

    def test_knocked_back_tries_again(self):
        io = FakeIO(x=0.25, knockbacks=2)
        step = pig_beach_rope()
        self.assertEqual(run_step(step, io, 25), DONE)
        self.assertEqual(sum('Missed the rope' in line for line in io.logs), 2)

    def test_three_tries_then_gives_up(self):
        io = FakeIO(x=0.30, knockbacks=99)
        step = pig_beach_rope()
        self.assertEqual(run_step(step, io, 40), FAILED)
        self.assertIn('missed the rope 3 times', step.reason)

    def test_runs_at_the_rope_from_either_side_and_grabs_it_mid_jump(self):
        for start in (0.20, 0.36):
            io = FakeIO(x=start)
            step = pig_beach_rope()
            self.assertEqual(run_step(step, io, 20), DONE, start)
            self.assertTrue(io.on_rope)
            self.assertEqual(step.misses, 0, start)     # first try, like a player

    def test_without_recorded_jump_points_uses_the_measured_lead(self):
        io = FakeIO(x=0.20)
        step = MountRope(0.284, 0.256, 0.213)
        self.assertEqual(step.lead, MountRope.LEAD)
        self.assertEqual(run_step(step, io, 20), DONE)
        self.assertEqual(step.misses, 0)

    def test_right_under_the_rope_jumps_straight_up(self):
        io = FakeIO(x=0.285)
        step = pig_beach_rope()
        self.assertEqual(run_step(step, io, 20), DONE)
        self.assertEqual(step.misses, 0)
        self.assertLess(io.t, 102.0)        # no walk away for a run-up

    def test_inside_the_jump_point_backs_off_for_a_run_up(self):
        io = FakeIO(x=0.294)
        step = pig_beach_rope()
        run_step(step, io, 0.1)
        self.assertEqual(step.phase, 'backoff')
        self.assertEqual(io.walking, 'right')
        self.assertEqual(run_step(step, io, 20), DONE)
        self.assertEqual(step.misses, 0)

    def test_steers_at_the_rope_in_the_air(self):
        io = FakeIO(x=0.20)
        step = pig_beach_rope()
        while getattr(step, 'phase', None) != 'air':
            step.run(io)
            io.advance()
        # a pig's hit threw it past the rope: hold back toward it
        io.x = 0.290
        step.run(io)
        self.assertEqual(io.walking, 'left')
        io.x = 0.285                        # level with it: let go
        step.run(io)
        self.assertIsNone(io.walking)
        io.x = 0.277                        # short of it: keep going
        step.run(io)
        self.assertEqual(io.walking, 'right')

    def test_after_a_miss_pauses_then_goes_again(self):
        io = FakeIO(x=0.25, knockbacks=1)
        step = pig_beach_rope()
        phases = []
        status = RUNNING
        while status == RUNNING and io.t < 120:
            status = step.run(io)
            if not phases or phases[-1][0] != step.phase:
                phases.append((step.phase, io.t))
            io.advance()
        self.assertEqual(status, DONE)
        names = [p for p, _ in phases]
        paused = phases[names.index('pause') + 1][1] - phases[names.index('pause')][1]
        self.assertTrue(MountRope.RETRY_PAUSE[0] - 0.03 <= paused <= MountRope.RETRY_PAUSE[1] + 0.03,
                        paused)

    def test_a_hit_in_mid_air_still_reaches_the_rope(self):
        io = FakeIO(x=0.20, air_hits=1)
        step = pig_beach_rope()
        self.assertEqual(run_step(step, io, 20), DONE)
        self.assertTrue(io.on_rope)

    def test_learns_the_take_off_point_after_a_miss(self):
        # A slower character (slowed, heavy): the recorded take-off is too early
        io = FakeIO(x=0.40)
        io.JUMP_SPEED = 0.03
        step = pig_beach_rope()
        self.assertEqual(run_step(step, io, 30), DONE)
        self.assertGreaterEqual(step.misses, 1)
        self.assertLess(step.misses, 3)
        self.assertLess(step.leads['left'], 0.303 - 0.284)
        self.assertTrue(any('jumped too early' in line for line in io.logs))

    def test_waits_on_the_rope_and_notices_being_knocked_off(self):
        io = FakeIO(x=0.284, y=0.213)
        io.on_rope = True
        self.assertEqual(run_step(HoldOnRope(0.284, 0.213, 10), io, 12), DONE)
        self.assertGreaterEqual(io.t, 110.0)
        io = FakeIO(x=0.284, y=0.213)
        hold = HoldOnRope(0.284, 0.213, 10)
        hold.run(io)
        io.y = io.GROUND_Y          # knocked off
        self.assertEqual(run_step(hold, io, 3), FAILED)


class WalkOffTests(unittest.TestCase):
    def test_walks_off_the_spawn_cliff_and_lands(self):
        io = FakeIO(x=0.057, y=0.147)
        step = WalkOff('right', (0.057, 0.147))
        original = io.advance

        def advance(dt=0.02):
            original(dt)
            if io.x >= 0.133 and io.y < io.GROUND_Y:
                io.y = min(io.GROUND_Y, io.y + 0.3 * dt)     # falling
        io.advance = advance
        self.assertEqual(run_step(step, io, 12), DONE)
        self.assertAlmostEqual(io.y, io.GROUND_Y, places=3)
        self.assertIsNone(io.walking)

    def test_nothing_to_do_away_from_the_spawn(self):
        io = FakeIO(x=0.30, y=0.256)
        self.assertEqual(run_step(WalkOff('right', (0.057, 0.147)), io, 1), DONE)
        self.assertLess(io.t, 100.1)


@unittest.skipUnless(os.path.exists(VIDEO), 'recording not available')
class ChannelScreenTests(unittest.TestCase):
    """The UI crops and the grid geometry, on the recording's own frames."""

    def test_finds_each_screen(self):
        frames = clip()
        self.assertIsNotNone(find_ui(frames[0], MENU_BUTTON, region=(0.4, 0.75, 1.0, 1.0)))
        self.assertIsNone(find_ui(frames[0], MENU_CHANGE_CHANNEL, region=(0.4, 0.3, 1.0, 1.0)))
        self.assertIsNotNone(find_ui(frames[42], MENU_CHANGE_CHANNEL, region=(0.4, 0.3, 1.0, 1.0)))
        self.assertIsNone(find_ui(frames[0], DIALOG_BUTTON))
        self.assertIsNotNone(find_ui(frames[90], DIALOG_BUTTON))

    def test_reads_the_current_channel_and_places_the_next_one(self):
        frame = clip()[90]
        button = find_ui(frame, DIALOG_BUTTON)
        self.assertEqual(current_channel(frame, button), 16)
        x, y = cell_centre(button, 17)      # CH.17: first column, third row
        self.assertAlmostEqual(x, 434.5, delta=3)
        self.assertAlmostEqual(y, 356.5, delta=3)
        x, y = cell_centre(button, 40)
        self.assertAlmostEqual(x, 924.5, delta=3)
        self.assertAlmostEqual(y, 400.5, delta=3)

    def test_counts_other_players_on_the_minimap(self):
        frames = clip()
        self.assertEqual(count_other_players(minimap_of(frames[0])), 0)
        self.assertEqual(count_other_players(minimap_of(frames[300])), 1)

    def test_recognises_the_map_from_its_title(self):
        frames = clip()
        self.assertTrue(map_events.PIG_BEACH.visible(frames[0]))
        self.assertFalse(map_events.PIG_BEACH.visible(frames[160]))     # loading


@unittest.skipUnless(os.path.exists(VIDEO), 'recording not available')
class ChangeChannelTests(unittest.TestCase):
    def _screens(self, io, frames, fail_load=False):
        """The game: the screen shown follows the clicks the step makes."""
        state = {'screen': 0, 'black_from': None}
        menu = find_ui(frames[0], MENU_BUTTON, region=(0.4, 0.75, 1.0, 1.0))
        item = find_ui(frames[42], MENU_CHANGE_CHANNEL, region=(0.4, 0.3, 1.0, 1.0))
        button = find_ui(frames[90], DIALOG_BUTTON)

        def click(x, y):
            io.clicks.append((round(x), round(y)))
            if state['screen'] == 0 and abs(x - menu[0]) < 5 and abs(y - menu[1]) < 5:
                state['screen'] = 42
            elif state['screen'] == 42 and abs(x - item[0]) < 5 and abs(y - item[1]) < 5:
                state['screen'] = 90
            elif state['screen'] == 90 and abs(x - button[0]) < 5 and abs(y - button[1]) < 5:
                if not fail_load:
                    state['screen'], state['black_from'] = 160, io.t

        def frame():
            if state['screen'] == 160 and io.t - state['black_from'] > 1.5:
                state['screen'] = 300
            return frames[state['screen']]
        original_advance = io.advance

        def advance(dt=0.02):
            original_advance(dt)
            if state['screen'] == 160:
                io.pos_time = state['black_from']   # no minimap fix while loading
        io.click, io.frame, io.advance = click, frame, advance
        return state, button

    def test_menu_then_list_then_next_channel_then_loading(self):
        frames = clip()
        io = FakeIO()
        state, button = self._screens(io, frames)
        step = ChangeChannel()
        self.assertEqual(run_step(step, io, 20), DONE)
        self.assertEqual(step.channel, 17)
        cell = cell_centre(button, 17)
        self.assertIn((round(cell[0]), round(cell[1])), io.clicks)
        self.assertEqual(io.clicks[-1], (round(button[0]), round(button[1])))
        self.assertEqual(io.black_until, io.t)      # the notifier may watch again
        self.assertTrue(any('16 -> 17' in line for line in io.logs))

    def test_a_channel_that_does_not_load_is_skipped(self):
        frames = clip()
        io = FakeIO()
        state, button = self._screens(io, frames, fail_load=True)
        step = ChangeChannel()
        step.ATTEMPTS = 3       # (normally every channel is tried: 2026-11-19)
        self.assertEqual(run_step(step, io, 90), FAILED)
        self.assertTrue(any('16 -> 17' in line for line in io.logs))
        self.assertTrue(any('? -> 18' in line or '16 -> 18' in line for line in io.logs))
        self.assertIn('esc', [key for _, key in io.presses])


class FindQuietChannelTests(unittest.TestCase):
    class FakeChange(Step):
        name = 'change channel'
        counter = [0]

        def __init__(self):
            super().__init__()
            self.channel = None
            self.known_channel = None

        def tick(self, io):
            if io.now() - self.started < 1.0:
                return RUNNING
            self.counter[0] += 1
            self.channel = (self.known_channel or 16) + 1
            io.minimap_value = io.channel_minimaps[self.counter[0] - 1]
            return DONE

    def _run(self, players):
        empty = np.zeros((83, 211, 3), np.uint8)
        busy = empty.copy()
        cv2.rectangle(busy, (100, 50), (103, 53), (0, 0, 255), -1)      # a red dot
        io = FakeIO()
        io.channel_minimaps = [busy if p else empty for p in players]
        self.FakeChange.counter = [0]
        step = FindQuietChannel(max_channels=3, change_factory=self.FakeChange)
        self.assertEqual(run_step(step, io, 60), DONE)
        return self.FakeChange.counter[0], io

    def test_a_free_channel_is_kept(self):
        changes, io = self._run([False])
        self.assertEqual(changes, 1)

    def test_busy_channels_are_left_until_a_free_one(self):
        changes, io = self._run([True, False, True])
        self.assertEqual(changes, 2)
        self.assertTrue(any('Channel 18 is free' in line for line in io.logs))

    def test_after_three_busy_channels_it_stays(self):
        changes, io = self._run([True, True, True, False])
        self.assertEqual(changes, 3)
        self.assertTrue(any('staying here' in line for line in io.logs))


def iron_hog_watch():
    return next(e for e in map_events.PIG_BEACH.events if e.name == 'Iron Hog').watch


class SpriteWatchTests(unittest.TestCase):
    @unittest.skipUnless(os.path.exists(VIDEO), 'recording not available')
    def test_spots_the_iron_hog_and_confirms_it(self):
        watch = iron_hog_watch()
        watch.reset()
        frames = clip()
        self.assertTrue(watch.find(frames[0]))
        self.assertFalse(watch.update(frames[0], 1.0))      # one sighting is not enough
        self.assertTrue(watch.update(frames[30], 1.5))
        self.assertEqual(watch.find(frames[300]), [])       # the new channel: no hog
        watch.reset()

    @unittest.skipUnless(os.path.exists(VIDEO), 'recording not available')
    def test_iron_hog_fallback_works_when_map_title_is_unavailable(self):
        watcher = map_events.MapEventWatcher(maps=[])
        event = next(e for e in map_events.PIG_BEACH.events if e.name == 'Iron Hog')
        event.watch.reset()
        frames = clip()
        stub = SimpleNamespace(player_pos=None, capture=None)
        try:
            with patch.object(map_events, 'config', stub), \
                    patch.object(map_events, 'GLOBAL_EVENTS', []):
                watcher.scan(frames[0], 1.0)
                self.assertIsNone(watcher.take())
                watcher.scan(frames[30], 1.5)
                self.assertEqual(watcher.take(), (map_events.PIG_BEACH, event))
        finally:
            event.watch.reset()

    @unittest.skipUnless(os.path.exists(os.path.join('assets', 'pig_beach_training.mp4')),
                         'recording not available')
    def test_ordinary_pigs_are_not_the_iron_hog(self):
        watch = iron_hog_watch()
        frames = _frames(os.path.join('assets', 'pig_beach_training.mp4'), range(0, 600, 40))
        self.assertTrue(frames)
        self.assertEqual([i for i, f in frames.items() if watch.find(f)], [])


class ManagerTests(unittest.TestCase):
    class Watcher:
        def __init__(self, pending):
            self.pending = pending
            self.paused = False
            self.finished_events = []

        def take(self):
            pending, self.pending = self.pending, None
            return pending

        def finished(self, event, now):
            self.finished_events.append(event.name)

    class Bot:
        def __init__(self):
            self.calls = []

        def _prepare_for_plan(self):
            self.calls.append('prepare')

        def _resume_after_plan(self, new_channel=True):
            self.calls.append('resume')

    def _manager(self, plan):
        event = map_events.Event('Thing', lambda: None, lambda profile: plan)
        watcher = self.Watcher((None, event))
        bot = self.Bot()
        io = FakeIO()
        return map_events.MapEventManager(bot, watcher=watcher, controls=io), bot, io, watcher

    def test_runs_the_plan_then_hands_back(self):
        manager, bot, io, watcher = self._manager(Plan('p', [Entry(Wait(0.5))]))
        ticks = 0
        while manager.service():
            io.advance()
            ticks += 1
            if not manager.active:
                break
        self.assertGreater(ticks, 10)
        self.assertEqual(bot.calls, ['prepare', 'resume'])
        self.assertEqual(watcher.finished_events, ['Thing'])
        self.assertFalse(watcher.paused)
        self.assertFalse(manager.service())     # nothing pending: farming goes on

    def test_a_failed_plan_pauses_the_bot(self):
        plan = Plan('p', [Entry(PlanTests.Fails(1), on_fail='abort')])
        manager, bot, io, watcher = self._manager(plan)
        with patch.object(config, 'enabled', True):
            manager.service()
            self.assertFalse(config.enabled)


class BotHookTests(unittest.TestCase):
    def test_a_running_plan_has_the_controls_instead_of_farming(self):
        from src.modules import bot as bot_module
        from src.modules.bot import Bot
        bot = Bot.__new__(Bot)
        bot._auto_attack_movement = 'left'
        bot._mob_attack_keys = ()
        bot.map_events = SimpleNamespace(service=lambda: True, cancel=lambda: None)
        targeted = []
        tracker = SimpleNamespace(settings=lambda: {'Enabled': True},
                                  target=lambda: targeted.append(1))
        stub = SimpleNamespace(capture=SimpleNamespace(mob_tracker=tracker), enabled=True)
        with patch.object(bot_module, 'config', stub), \
                patch.object(bot_module, 'is_game_focused', return_value=True), \
                patch.object(bot_module.time, 'sleep'):
            bot._run_standalone_mob_attack()
        self.assertEqual(targeted, [])          # no farming this tick

    def test_back_to_farming_resets_channel_bound_state(self):
        from src.modules.bot import Bot
        bot = Bot.__new__(Bot)
        bot._level = 'step'
        bot._walls = {'left': 0.5}
        bot._level_walls = {'ground': {'right': 0.6}}
        bot._loot_planner = object()
        bot._patrol_direction = 'left'
        with patch.object(Bot, '_set_blocked_side', lambda self, side: None):
            bot._resume_after_plan()
        self.assertEqual(bot._level, 'ground')
        self.assertEqual(bot._walls, {'right': 0.6})
        self.assertIsNone(bot._loot_planner)
        self.assertIsNone(bot._patrol_direction)


class PigBeachPlanTests(unittest.TestCase):
    def test_the_pit_plan_and_limit(self):
        self.assertEqual(map_events.PIG_BEACH.limits, {'right': 0.657 - 0.012})
        pit = next(e for e in map_events.PIG_BEACH.events if e.name == 'Pit')
        plan = pit.plan_factory(map_events.PIG_BEACH)
        self.assertFalse(plan.new_channel)
        step = plan.entries[0].step
        self.assertEqual((step.direction, step.safe_x), ('left', 0.516))

    def test_the_iron_hog_plan(self):
        plan = map_events.iron_hog_escape(map_events.PIG_BEACH)
        names = [entry.step.name for entry in plan.entries]
        self.assertEqual(names, ['climb the rope', 'wait on the rope',
                                 'find a quiet channel', 'walk off the cliff'])
        rope = plan.entries[0].step
        # recorded in iron_hog_escape.mp4
        self.assertEqual((rope.rope_x, rope.mount_y, rope.safe_y), (0.284, 0.256, 0.213))
        self.assertAlmostEqual(rope.leads['right'], 0.284 - 0.265)
        self.assertAlmostEqual(rope.leads['left'], 0.303 - 0.284)
        self.assertEqual(rope.attempts, 3)
        self.assertEqual(plan.entries[0].on_fail, 'channel')    # 3 misses: change channel
        self.assertEqual(plan.entries[1].step.seconds, 10.0)
        self.assertEqual(plan.entries[2].step.max_channels, 3)


if __name__ == '__main__':
    unittest.main()


class PitTests(unittest.TestCase):
    """Pig Beach: the pig island spans 0.526-0.657; right of it a lower pit."""

    def _world(self, x):
        io = FakeIO(x=x, y=0.263)
        original = io.advance

        def advance(dt=0.02):
            before = io.x
            original(dt)
            airborne = io.air_until is not None
            # the island's right side is a wall from the pit, unless jumping
            if before > 0.657 >= io.x and not airborne:
                io.x = 0.658
            io.y = 0.249 if 0.526 <= io.x <= 0.657 else (0.263 if io.x > 0.657 else 0.256)
        io.advance = advance
        return io

    def test_jumps_out_of_the_pit_back_to_the_farming_ground(self):
        from src.modules.plan_steps import ReturnInside
        io = self._world(0.704)
        self.assertEqual(run_step(ReturnInside('left', 0.516), io, 25), DONE)
        self.assertLessEqual(io.x, 0.516)
        self.assertIsNone(io.walking)
        self.assertNotIn('down', io.held)
        self.assertGreaterEqual(sum(key == 'up' for _, key in io.presses), 3)
        self.assertGreaterEqual(sum(key == 'alt' for _, key in io.presses), 3)

    def test_jumps_off_the_pig_island_to_flat_ground(self):
        from src.modules.plan_steps import ReturnInside
        io = self._world(0.60)
        self.assertEqual(run_step(ReturnInside('left', 0.516), io, 25), DONE)
        self.assertLessEqual(io.x, 0.516)
        self.assertAlmostEqual(io.y, io.GROUND_Y)
        self.assertIsNone(io.walking)
        self.assertNotIn('down', io.held)

    def test_waits_for_a_stable_landing_after_up_and_jump_taps(self):
        from src.modules.plan_steps import ReturnInside
        io = FakeIO(x=0.53, y=0.20)
        step = ReturnInside('left', 0.516, timeout=3.0)
        self.assertEqual(step.run(io), RUNNING)
        self.assertEqual(io.walking, 'left')
        self.assertEqual(io.presses[:2], [(100.0, 'up'), (100.0, 'alt')])

        status = RUNNING
        while io.t < 103.0 and status == RUNNING:
            io.advance()
            if io.t < 100.8:
                io.y = 0.20 + (0.01 if int((io.t - 100.0) / 0.1) % 2 else 0.0)
            else:
                io.y = io.GROUND_Y
            status = step.run(io)

        self.assertEqual(status, DONE)
        self.assertGreaterEqual(io.t, 101.1)
        step.end(io)
        self.assertNotIn('down', io.held)

    def test_stops_moving_when_minimap_position_is_lost(self):
        from src.modules.plan_steps import ReturnInside
        io = FakeIO(x=0.704)
        step = ReturnInside('left', 0.516, timeout=5.0)
        self.assertEqual(step.run(io), RUNNING)
        self.assertEqual(io.walking, 'left')
        self.assertEqual(io.presses[:2], [(100.0, 'up'), (100.0, 'alt')])
        io.position = lambda: None
        self.assertEqual(run_step(step, io, 4), FAILED)
        self.assertIsNone(io.walking)
        self.assertNotIn('down', io.held)

    def test_walking_alone_would_stay_stuck(self):
        io = self._world(0.704)
        io.walking = 'left'
        for _ in range(500):
            io.advance()
        self.assertGreater(io.x, 0.657)     # (why the jumping is needed)

    def test_out_of_bounds_needs_a_moment_past_the_island_end(self):
        watch = map_events.OutOfBounds('right', 0.657)
        stub = SimpleNamespace(player_pos=(0.70, 0.263), player_pos_time=None)
        with patch.object(map_events, 'config', stub), \
                patch.object(map_events.time, 'monotonic', return_value=50.0):
            stub.player_pos_time = 50.0
            self.assertFalse(watch.update(None, 10.0))
            self.assertTrue(watch.update(None, 10.5))
            stub.player_pos = (0.65, 0.249)      # on the island: fine
            self.assertFalse(watch.update(None, 11.0))
            stub.player_pos = (0.66, 0.249)      # within the margin: fine
            self.assertFalse(watch.update(None, 12.0))
            self.assertFalse(watch.update(None, 13.0))


class PitFallbackTests(unittest.TestCase):
    def _pending(self, jumper, position):
        watcher = map_events.MapEventWatcher(maps=[])
        pit = next(event for event in map_events.PIG_BEACH.events
                   if event.name == 'Pit')
        pit.watch.reset()
        stub = SimpleNamespace(
            player_pos=position,
            player_pos_time=10.0,
            capture=SimpleNamespace(
                mob_tracker=SimpleNamespace(
                    level_farming_enabled=lambda: jumper,
                )
            ),
        )
        frame = np.zeros((120, 420, 3), dtype=np.uint8)
        with patch.object(map_events, 'config', stub), \
                patch.object(map_events, 'GLOBAL_EVENTS', []), \
                patch.object(map_events.time, 'monotonic', return_value=10.0):
            watcher.scan(frame, 10.0)
            watcher.scan(frame, 10.5)
            pending = watcher.take()
        pit.watch.reset()
        return pending, pit

    def test_unknown_map_uses_jumper_pit_coordinates_as_fallback(self):
        pending, pit = self._pending(True, (0.704, 0.263))
        self.assertEqual(pending, (map_events.PIG_BEACH, pit))

    def test_pit_fallback_requires_jumper_profile_and_lower_level(self):
        pending, _ = self._pending(False, (0.704, 0.263))
        self.assertIsNone(pending)
        pending, _ = self._pending(True, (0.704, 0.249))
        self.assertIsNone(pending)


class MapLimitBotTests(unittest.TestCase):
    def _bot(self):
        from src.modules.bot import Bot
        bot = Bot.__new__(Bot)
        bot._walls = {}
        bot.map_events = SimpleNamespace(limits=lambda: {'right': 0.645})
        return bot

    def test_farming_treats_the_island_end_as_a_wall(self):
        bot = self._bot()
        self.assertTrue(bot._wall_ahead('right', 0.640))
        self.assertFalse(bot._wall_ahead('right', 0.60))
        self.assertFalse(bot._wall_ahead('left', 0.640))

    def test_never_walks_down_off_the_pit_side_of_the_island(self):
        from src.modules import bot as bot_module
        bot = self._bot()
        bot._mob_attack_keys = ()
        bot._level = 'ground'           # level unknown to the climb logic
        with patch.object(bot_module, 'key_up'):
            op = bot._start_descend(2, 150, 0.60, 0.249, 10.0)    # pigs below, right
        self.assertEqual(op['direction'], 'left')

    def test_walls_learnt_in_the_pit_are_dropped_on_return(self):
        from src.modules.bot import Bot
        bot = self._bot()
        bot._walls = {'left': 0.66, 'right': 0.75}
        bot._level_walls = {'ground': {'left': 0.30}}
        with patch.object(Bot, '_set_blocked_side', lambda self, side: None):
            bot._resume_after_plan(new_channel=False)
        self.assertEqual(bot._walls, {})
        self.assertEqual(bot._level_walls['ground'], {'left': 0.30})


class EmptyMapTests(unittest.TestCase):
    """Patrols a 0.30-0.60 strip with walls at both ends; nothing to fight."""

    def _patrol(self, seconds, enemy_at=None):
        from src.modules import bot as bot_module
        from src.modules.bot import Bot
        bot = Bot.__new__(Bot)
        bot._patrol_direction = None
        bot._patrol_until = 0
        bot._auto_attack_movement = None
        bot._mob_attack_keys = ()
        bot._walls = {}
        requests = []
        bot.map_events = SimpleNamespace(service=lambda: False, cancel=lambda: None,
                                         limits=lambda: {},
                                         request_empty_map=lambda laps: requests.append(clock[0]))
        clock = [100.0]
        state = {'x': 0.45}
        tracker = SimpleNamespace(
            settings=lambda: {'Enabled': True}, set_sweep_direction=lambda d: None,
            set_blocked_side=lambda s: None, waiting=lambda: False,
            ledge_enemies=lambda: (0, None), lower_enemies=lambda: (0, None),
            consume_attack_action=lambda: (None, False, 0), group_in_range=lambda: False,
            consume_kills=lambda: [])
        loot = SimpleNamespace(settings=lambda: {'Enabled': False}, consume_pickup=lambda: None)
        stub = SimpleNamespace(player_pos=(0.45, 0.256), player_pos_time=100.0, enabled=True,
                               capture=SimpleNamespace(mob_tracker=tracker, loot_tracker=loot))
        with patch.object(bot_module, 'config', stub), \
                patch.object(bot_module, 'key_down'), patch.object(bot_module, 'key_up'), \
                patch.object(bot_module, 'press'), \
                patch.object(bot_module, 'is_game_focused', return_value=True), \
                patch.object(bot_module, 'FALL_RECOVERY', False), \
                patch.object(bot_module.time, 'sleep'), \
                patch.object(bot_module.time, 'monotonic', lambda: clock[0]):
            while clock[0] < 100.0 + seconds:
                now = clock[0]
                tracker.target = (lambda: (40, False)) if enemy_at and enemy_at[0] <= now < enemy_at[1] \
                    else (lambda: None)
                bot._run_standalone_mob_attack()
                move = {'right': 0.05, 'left': -0.05}.get(bot._auto_attack_movement, 0) * 0.05
                state['x'] = min(0.60, max(0.30, state['x'] + move))
                stub.player_pos, stub.player_pos_time = (round(state['x'], 4), 0.256), now
                clock[0] += 0.05
        return requests

    def test_three_empty_laps_ask_for_a_channel_change(self):
        # A lap: 0.30 -> 0.60 -> 0.30 at 0.05/s = 12 s plus stuck time
        requests = self._patrol(60)
        self.assertEqual(len(requests), 1)
        self.assertGreater(requests[0], 130.0)      # not before 3 laps

    def test_an_enemy_restarts_the_count(self):
        no_enemy = self._patrol(60)
        with_enemy = self._patrol(60, enemy_at=(120.0, 121.0))
        self.assertLess(no_enemy[0], with_enemy[0] if with_enemy else float('inf'))

    def test_the_plan_changes_channel_then_walks_off_the_spawn(self):
        plan = map_events.empty_map_hop(map_events.PIG_BEACH)
        self.assertTrue(plan.new_channel)
        self.assertEqual([e.step.name for e in plan.entries],
                         ['find a quiet channel', 'walk off the cliff'])
        self.assertEqual([e.step.name for e in map_events.empty_map_hop(None).entries],
                         ['find a quiet channel'])

    def test_the_manager_queues_it_once(self):
        watcher = map_events.MapEventWatcher(maps=[])
        manager = map_events.MapEventManager(SimpleNamespace(), watcher=watcher, controls=FakeIO())
        manager.request_empty_map(3)
        manager.request_empty_map(3)
        self.assertEqual(watcher.take()[1].name, 'Empty map')
        self.assertIsNone(watcher.take())


class PlayersNearbyTests(unittest.TestCase):
    """2+ other players on the character's platform for ~5 s: change channel."""

    def _minimap(self, dots):
        minimap = np.full((100, 200, 3), (90, 40, 20), np.uint8)    # dark blue
        for x, y in dots:                                           # pixel centres
            cv2.circle(minimap, (x, y), 3, (0, 0, 255), -1)          # red dot
        return minimap

    def _run(self, dots, me, seconds=7.0, flicker=False):
        minimap = self._minimap(dots)
        empty = self._minimap([])
        watch = map_events.PlayersNearby()
        fired = None
        with patch.object(config, 'capture', SimpleNamespace(minimap={'minimap': minimap},
                                                             minimap_ratio=2.0)), \
                patch.object(config, 'player_pos', me):
            for i in range(int(seconds / 0.5)):
                # every 4th look misses them (a jump, overlapping dots)
                config.capture.minimap['minimap'] = empty if flicker and i % 4 == 3 else minimap
                config.player_pos_time = map_events.time.monotonic()
                if watch.update(None, i * 0.5) and fired is None:
                    fired = i * 0.5
        return fired

    # minimap 200x100 px, ratio 2: x = px / 200, y = py / 2 / 100
    def test_two_players_beside_me_change_channel(self):
        self.assertIsNotNone(self._run([(90, 80), (120, 80)], (0.5, 0.4)))

    def test_waits_five_seconds(self):
        self.assertGreaterEqual(self._run([(90, 80), (120, 80)], (0.5, 0.4)), 5.0)

    def test_one_player_is_fine(self):
        self.assertIsNone(self._run([(90, 80)], (0.5, 0.4)))

    def test_players_on_another_platform_are_fine(self):
        self.assertIsNone(self._run([(90, 60), (120, 60)], (0.5, 0.4)))     # y 0.30

    def test_players_far_along_my_platform_are_fine(self):
        self.assertIsNone(self._run([(10, 80), (190, 80)], (0.5, 0.4)))     # x 0.05, 0.95

    def test_dots_missed_now_and_then_still_count(self):
        self.assertIsNotNone(self._run([(90, 80), (120, 80)], (0.5, 0.4), flicker=True))

    def test_watched_on_maps_without_a_profile(self):
        watcher = map_events.MapEventWatcher(maps=[])
        with patch.object(map_events.CROWDED.watch, 'update', return_value=True):
            watcher.scan(np.zeros((10, 10, 3), np.uint8), 100.0)
        self.assertIs(watcher.take()[1], map_events.CROWDED)

    def test_plan_changes_channel(self):
        plan = map_events.crowded_hop(None)
        self.assertTrue(plan.new_channel)
        self.assertIn('players nearby', plan.name)


class PigIslandLapsTests(unittest.TestCase):
    """Pig Beach (user 2026-11-19): pig island cleared, then 2 empty laps of the
    flat ground: change channel."""

    def _bot(self, level='ground'):
        from src.modules.bot import Bot
        bot = Bot.__new__(Bot)
        self.requests = []
        bot.map_events = SimpleNamespace(
            watcher=SimpleNamespace(map=map_events.PIG_BEACH),
            limits=lambda: dict(map_events.PIG_BEACH.limits),
            request_empty_map=lambda laps, message=None: self.requests.append((laps, message)))
        bot._level = level
        return bot

    @staticmethod
    def _tracker(up=0, down=0):
        return SimpleNamespace(ledge_enemies=lambda: (up, 40 if up else None),
                               lower_enemies=lambda: (down, 40 if down else None))

    def test_pig_beach_asks_for_one_lap(self):
        self.assertEqual(map_events.PIG_BEACH.spots['empty_map_laps'], 1)
        self.assertEqual(map_events.PIG_BEACH.spots['island_clear_laps'], 1)

    def test_one_empty_lap_changes_channel_on_pig_beach(self):
        bot = self._bot()
        bot._empty_turn()
        self.assertEqual(self.requests, [])
        bot._empty_turn()                       # the lap ends
        self.assertEqual(self.requests, [(1, None)])

    def test_one_empty_lap_on_the_island_too(self):
        bot = self._bot(level='step')
        bot._island_cleared = True
        bot._empty_turn()
        bot._empty_turn()
        self.assertEqual(self.requests, [(1, None)])

    def test_other_maps_keep_three_laps(self):
        bot = self._bot()
        bot.map_events.watcher = SimpleNamespace(map=SimpleNamespace(spots={}))
        for _ in range(5):
            bot._empty_turn()
        self.assertEqual(self.requests, [])
        bot._empty_turn()
        self.assertEqual(self.requests, [(3, None)])

    def test_a_pig_restarts_the_ground_laps(self):
        bot = self._bot()
        bot._island_cleared = True
        bot._empty_turn()
        bot._count_empty_laps(self._tracker(down=1), None)
        bot._empty_turn()
        self.assertEqual(self.requests, [])

    def test_island_counts_as_cleared_after_a_while_up_there_with_no_pigs(self):
        bot = self._bot(level='step')
        bot._track_island(self._tracker(), None, 5.0, 100.0)
        self.assertFalse(bot.__dict__.get('_island_cleared'))   # just arrived
        bot._track_island(self._tracker(), None, 5.0, 100.0 + bot.ISLAND_CLEAR_SECONDS)
        self.assertTrue(bot._island_cleared)
        bot._track_island(self._tracker(), object(), 0.0, 104.0)   # a pig up there again
        self.assertFalse(bot._island_cleared)

    def test_pigs_seen_on_the_island_from_below_uncheck_it(self):
        bot = self._bot()
        bot._island_cleared = True
        bot._track_island(self._tracker(up=2), None, 5.0, 100.0)
        self.assertFalse(bot._island_cleared)

    def test_cleared_island_and_nothing_below_walks_down(self):
        from src.modules import bot as bot_module
        bot = self._bot(level='step')
        bot._island_cleared = True
        bot._step = {'direction': 'right', 'edge_x': 0.526, 'ground_y': 0.256}
        bot._mob_attack_keys = ()
        with patch.object(bot_module, 'key_up'),                 patch.object(bot_module, 'config', SimpleNamespace(capture=None)):
            op = bot._choose_level_op(self._tracker(), None, 0.58, 0.240, 100.0, 5.0)
        self.assertIsNotNone(op)
        self.assertEqual((op['kind'], op['direction']), ('descend', 'left'))

    def test_a_new_channel_unchecks_the_island(self):
        from src.modules.bot import Bot
        bot = self._bot()
        bot._island_cleared = True
        bot._walls, bot._level_walls = {}, {}
        with patch.object(Bot, '_set_blocked_side', lambda self, side: None):
            bot._resume_after_plan(new_channel=True)
        self.assertFalse(bot._island_cleared)


if __name__ == '__main__':
    unittest.main()


class StayOnChannelTests(unittest.TestCase):
    """The GUI's "Stay on this channel": never change channel on its own."""

    def setUp(self):
        self.stay = patch.object(map_events, 'stay_on_channel', return_value=True)
        self.stay.start()

    def tearDown(self):
        self.stay.stop()

    def test_empty_map_does_not_queue_a_channel_change(self):
        watcher = map_events.MapEventWatcher(maps=[])
        manager = map_events.MapEventManager(SimpleNamespace(), watcher=watcher, controls=FakeIO())
        manager.request_empty_map(3)
        self.assertIsNone(watcher.take())

    def test_players_nearby_is_ignored(self):
        watcher = map_events.MapEventWatcher(maps=[])
        with patch.object(map_events.CROWDED.watch, 'update', return_value=True):
            watcher.scan(np.zeros((10, 10, 3), np.uint8), 100.0)
        self.assertIsNone(watcher.take())


    def test_the_iron_hog_is_still_watched(self):
        watcher = map_events.MapEventWatcher(maps=[map_events.PIG_BEACH])
        watcher.map = map_events.PIG_BEACH
        watcher._map_checked = 1e9
        hog = next(e for e in map_events.PIG_BEACH.events if e.name == 'Iron Hog')
        with patch.object(type(hog.watch), 'ready', True),                 patch.object(hog.watch, 'update', return_value=True):
            watcher.scan(np.zeros((10, 10, 3), np.uint8), 100.0)
        self.assertIs(watcher.take()[1], hog)

    def test_iron_hog_plan_hides_without_changing_channel(self):
        plan = map_events.iron_hog_escape(map_events.PIG_BEACH)
        self.assertFalse(plan.new_channel)
        self.assertEqual([e.step.name for e in plan.entries],
                         ['climb the rope', 'hide on the rope until it leaves', 'jump off the rope',
                          'find a quiet channel', 'walk off the cliff', 'wait'])
        names = self._route(plan, rope_ok=True)
        self.assertEqual(names, ['climb the rope', 'hide on the rope until it leaves',
                                 'jump off the rope', 'wait'])
        self.assertFalse(plan.new_channel)

    def test_iron_hog_hide_changes_channel_when_the_rope_fails(self):
        # 2026-11-18, user: "try 3 times to get on rope, if not possible change channel"
        plan = map_events.iron_hog_escape(map_events.PIG_BEACH)
        names = self._route(plan, rope_ok=False)
        self.assertEqual(names, ['climb the rope', 'find a quiet channel',
                                 'walk off the cliff', 'wait'])
        self.assertTrue(plan.new_channel)

    @staticmethod
    def _route(plan, rope_ok):
        """Runs PLAN with every step faked (the rope fails unless ROPE_OK);
        the names of the steps in the order they ran."""
        ran = []

        class Fake(Step):
            def __init__(self, real, ok):
                super().__init__()
                self.name, self.ok = real.name, ok
                self.changes_channel = getattr(real, 'changes_channel', False)

            def tick(self, io):
                ran.append(self.name)
                return DONE if self.ok else self.fail('faked')

        for entry in plan.entries:
            entry.step = Fake(entry.step, rope_ok or entry.step.name != 'climb the rope')
        io = FakeIO()
        status = RUNNING
        while status == RUNNING:
            status = plan.tick(io)
        assert status == DONE, plan.reason
        return ran

    class _Watch:
        def __init__(self, gone_at):
            self.gone_at, self.io = gone_at, None

        def find(self, frame):
            return [(1, 1, 0.9)] if self.io.t < self.gone_at else []

    def _hide(self, gone_at, max_seconds=180.0):
        io = FakeIO(x=0.286, y=0.200)
        io.on_rope = True
        watch = self._Watch(gone_at)
        watch.io = io
        step = HideUntilGone(0.286, 0.200, lambda: watch, min_seconds=10.0,
                             clear_seconds=8.0, max_seconds=max_seconds)
        return run_step(step, io, 400), io

    def test_waits_until_the_hog_has_been_gone_a_while(self):
        status, io = self._hide(gone_at=130.0)           # 30 s in
        self.assertEqual(status, DONE)
        self.assertGreaterEqual(io.t, 137.5)     # last seen <= 0.5 s before 130
        self.assertLess(io.t, 140.0)

    def test_waits_at_least_the_rope_wait(self):
        status, io = self._hide(gone_at=0.0)             # never seen
        self.assertEqual(status, DONE)
        self.assertGreaterEqual(io.t, 110.0)

    def test_gives_up_if_the_hog_never_leaves(self):
        status, _ = self._hide(gone_at=1e9, max_seconds=60.0)
        self.assertEqual(status, FAILED)

    def test_jumps_off_the_rope_and_lands(self):
        io = FakeIO(x=0.286, y=0.200)
        io.on_rope = True
        io.held.add('up')
        status = run_step(LeaveRope(0.200, 'right'), io, 5)
        self.assertEqual(status, DONE)
        self.assertFalse(io.on_rope)
        self.assertNotIn('up', io.held)
        self.assertIsNone(io.walking)


class ChannelSettingTests(unittest.TestCase):
    def test_saved_and_read_back(self):
        import tempfile
        from src.modules import channel_settings
        with tempfile.TemporaryDirectory() as folder,                 patch.object(channel_settings, '_settings', None),                 patch.object(channel_settings.ChannelSettings, '__init__',
                             lambda self: channel_settings.Configurable.__init__(
                                 self, 'channel', directory=folder)):
            self.assertFalse(channel_settings.stay_on_channel())
            channel_settings.set_stay_on_channel(True)
            self.assertTrue(channel_settings.stay_on_channel())
            channel_settings._settings = None           # a restart reads the file
            self.assertTrue(channel_settings.stay_on_channel())


class MapMobProfileTests(unittest.TestCase):
    def test_detected_map_loads_its_enemy_profile(self):
        loaded = []
        tracker = SimpleNamespace(
            load_profile=lambda name, **kwargs: loaded.append((name, kwargs)))
        watcher = map_events.MapEventWatcher(maps=[map_events.PIG_BEACH])
        with patch.object(config, 'enabled', True), \
                patch.object(config, 'capture', SimpleNamespace(mob_tracker=tracker)), \
                patch.object(map_events.PIG_BEACH, 'visible', return_value=True), \
                patch.object(map_events.PIG_BEACH, 'events', []), \
                patch.object(map_events, 'GLOBAL_EVENTS', []):
            watcher.scan(np.zeros((10, 10, 3), np.uint8), 100.0)
        self.assertEqual(loaded, [
            ('pig-profile', {'preserve_player_badge': True})])

    def test_missing_enemy_profile_pauses_before_map_scanning(self):
        def fail_to_load(name, **kwargs):
            raise ValueError(f'No saved profile named {name!r}.')

        watcher = map_events.MapEventWatcher(maps=[map_events.PIG_BEACH])
        with patch.object(config, 'enabled', True), \
                patch.object(config, 'capture',
                             SimpleNamespace(mob_tracker=SimpleNamespace(
                                 load_profile=fail_to_load))), \
                patch.object(map_events.PIG_BEACH, 'visible', return_value=True), \
                patch.object(map_events.PIG_BEACH, 'events', []), \
                patch.object(map_events, 'GLOBAL_EVENTS', []):
            watcher.scan(np.zeros((10, 10, 3), np.uint8), 100.0)
            self.assertFalse(config.enabled)
        self.assertIsNone(watcher.map)


class MapTitleTests(unittest.TestCase):
    """The map is recognised from its name in the minimap's title bar."""

    def _frame_with_title_at(self, x, y):
        title = map_events.PIG_BEACH.title
        frame = np.full((768, 1040, 3), 200, np.uint8)
        frame[y:y + title.shape[0], x:x + title.shape[1]] = title[:, :, None]
        return frame

    def test_title_in_the_client_area_is_recognised(self):
        self.assertTrue(map_events.PIG_BEACH.visible(self._frame_with_title_at(47, 40)))

    def test_title_in_a_whole_window_capture_is_recognised(self):
        # Captures include the Windows title bar and border (~39 px, 8 px):
        # 19 Nov, "The Pig Beach" at y 79-93 fell outside the old 90 px search,
        # so pig-profile never loaded and no pig was ever seen
        self.assertTrue(map_events.PIG_BEACH.visible(self._frame_with_title_at(55, 79)))
