"""A Split Road: channel hunt, chair rest and the 4 s wait after attacking (no game needed)."""

import unittest
from types import SimpleNamespace
from unittest import mock

from src.common import config
from src.modules import map_events
from src.modules.channel import ChangeChannel
from src.modules.plan_steps import (
    ClimbOnto, GoToSpot, ScoutPlatform, SitUntilHealed, RUNNING, DONE, FAILED)

SPOTS = map_events._SPLIT_ROAD_SPOTS
PLATFORM = map_events._platform(SPOTS)
SAFE = SPOTS['safe']


class RoadIO:
    """The platform (0.021-0.197 at y 0.176) and the step down to the safe ground (y 0.192)."""

    WALK = 0.045
    DT = 0.02

    def __init__(self, x=0.10, y=0.176, hp=15.0, enemies=0, heal=2.0, since_attack=99.0):
        self.t = 0.0
        self.x, self.y = x, y
        self.hp_value = hp
        self.heal = heal            # HP % a second while sitting
        self.sitting = False
        self.walking = None
        self.enemies = enemies
        self.air_until = None
        self.presses, self.logs = [], []
        self.jump_key = 'alt'
        self.since_attack = since_attack

    # Controls API
    def now(self):
        return self.t

    def position(self):
        return (round(self.x, 4), round(self.y, 4))

    def position_time(self):
        return self.t

    def walk(self, direction):
        self.walking = direction
        if direction is not None:
            self.sitting = False

    def key_down(self, key):
        pass

    def key_up(self, key):
        pass

    def press(self, key, n=1, down_time=0.05, up_time=0.05):
        self.presses.append(key)
        if key == 'c':
            self.sitting = not self.sitting
        elif key == self.jump_key:
            self.sitting = False
            if self.air_until is None:
                self.air_until = self.t + 0.45
        elif key in ('left', 'right'):
            self.sitting = False
            self.x += 0.003 if key == 'right' else -0.003

    def hp(self):
        return self.hp_value

    def enemy_count(self):
        return self.enemies

    def seconds_since_attack(self):
        return self.since_attack

    def frame(self):
        return None

    def minimap(self):
        return None

    def click(self, x, y):
        self.presses.append(('click', x, y))

    def expect_black_screen(self, seconds):
        pass

    def log(self, message):
        self.logs.append(message)

    def release(self):
        self.walking = None

    # The world
    LEDGE_END = 0.214           # the safe ledge (y 0.192) ends here; the bottom (0.209) beyond

    def level(self, x):
        """Ground height at X: platform, safe ledge, bottom (stuck.mp4)."""
        if x <= PLATFORM[1] + 0.004:
            return 0.176
        return 0.192 if x <= self.LEDGE_END else 0.209

    def step(self):
        self.t += self.DT
        self.since_attack += self.DT
        if self.walking:
            sign = 1 if self.walking == 'right' else -1
            new_x = self.x + sign * self.WALK * self.DT
            # Higher ground ahead is a wall unless jumping
            if self.air_until is None and self.level(new_x) < self.y - 0.001:
                new_x = self.x
            self.x = new_x
        if self.air_until is not None and self.t >= self.air_until:
            self.air_until = None
            self.y = min(self.y, self.level(self.x))    # landed one level up (or level)
        if self.air_until is None and self.level(self.x) > self.y:
            self.y = self.level(self.x)                 # walked off an edge: down a level
        if self.sitting:
            self.hp_value = min(100.0, self.hp_value + self.heal * self.DT)
        self.since_attack = self.since_attack


def run(step, io, seconds=60.0):
    status = RUNNING
    while status == RUNNING and io.t < seconds:
        status = step.run(io)
        io.step()
    return status


class SplitRoadStepTests(unittest.TestCase):
    def test_walks_off_the_platform_to_the_safe_spot(self):
        io = RoadIO(x=0.15)
        self.assertEqual(run(GoToSpot(*SAFE, what='the safe spot'), io), DONE)
        self.assertAlmostEqual(io.x, SAFE[0], delta=0.006)
        self.assertEqual(io.y, 0.192)

    def test_sits_until_hp_is_full(self):
        io = RoadIO(x=SAFE[0], y=SAFE[1], hp=18.0, heal=5.0)
        self.assertEqual(run(SitUntilHealed('c'), io, seconds=120), DONE)
        self.assertGreaterEqual(io.hp_value, 98.0)
        self.assertEqual(io.presses.count('c'), 1)

    def test_presses_the_chair_again_when_hp_does_not_rise(self):
        io = RoadIO(x=SAFE[0], y=SAFE[1], hp=18.0, heal=5.0)
        io.press = (lambda real: (lambda key, *a, **k: (
            io.presses.append(key) if key == 'c' and io.presses.count('c') == 0
            else real(key, *a, **k))))(io.press)      # the first press is lost
        self.assertEqual(run(SitUntilHealed('c'), io, seconds=200), DONE)
        self.assertGreaterEqual(io.presses.count('c'), 2)

    def test_jumps_back_up_left_onto_the_platform(self):
        io = RoadIO(x=SAFE[0], y=SAFE[1])
        self.assertEqual(run(ClimbOnto(PLATFORM, SAFE, 'left'), io), DONE)
        self.assertEqual(io.y, 0.176)
        self.assertIn('alt', io.presses)

    def test_already_on_the_platform(self):
        io = RoadIO(x=0.1)
        self.assertEqual(run(ClimbOnto(PLATFORM, SAFE, 'left'), io), DONE)
        self.assertNotIn('alt', io.presses)

    def test_unknown_spawn_fails_with_the_position(self):
        io = RoadIO(x=0.6, y=0.30)
        step = ClimbOnto(PLATFORM, SAFE, 'left')
        self.assertEqual(run(step, io), FAILED)
        self.assertIn('(0.600, 0.300)', step.reason)

    def test_scout_wants_two_enemies(self):
        self.assertEqual(run(ScoutPlatform(PLATFORM, 2), RoadIO(enemies=2)), DONE)
        step = ScoutPlatform(PLATFORM, 2)
        self.assertEqual(run(step, RoadIO(enemies=1)), FAILED)
        self.assertIn('only 1 enemy', step.reason)


class ChannelWaitTests(unittest.TestCase):
    def test_waits_four_seconds_after_an_attack(self):
        io = RoadIO(since_attack=0.5)
        change = ChangeChannel()
        with mock.patch('src.modules.channel.find_ui', return_value=None):
            while io.since_attack < 3.4:
                self.assertEqual(change.run(io), RUNNING)
                io.step()
            self.assertEqual(io.presses, [])        # nothing opened yet
            while io.since_attack < 4.6:
                change.run(io)
                io.step()
        self.assertTrue(io.presses)                 # now it opens the menu (Esc)
        self.assertTrue(any('after the last attack' in m for m in io.logs))


class ChannelPersistenceTests(unittest.TestCase):
    """12:58 log: channels 1, 2, 3 did not load, the hunt gave up and paused the bot."""

    def test_a_fourth_failed_channel_is_not_the_end(self):
        io = RoadIO()
        change = ChangeChannel()
        change.begin(io)
        change.target = 3
        change.attempt = 3
        self.assertEqual(change._retry(io, io.now(), 'channel 3 did not load'), RUNNING)
        self.assertEqual(change.attempt, 4)
        self.assertGreater(change.at, io.now())         # a pause before the next try
        self.assertIsNone(change.timeout)

    def test_gives_up_only_after_every_channel(self):
        io = RoadIO()
        change = ChangeChannel()
        change.begin(io)
        change.attempt = ChangeChannel.ATTEMPTS
        self.assertEqual(change._retry(io, io.now(), 'nope'), FAILED)
        self.assertEqual(ChangeChannel.ATTEMPTS, 40)

    def test_a_failed_round_starts_again_in_the_plans(self):
        with mock.patch.object(map_events, 'stay_on_channel', return_value=False):
            for plan in (map_events.platform_hunt(map_events.SPLIT_ROAD),
                         map_events.rest_on_chair(map_events.SPLIT_ROAD)):
                entry = next(e for e in plan.entries if e.label == 'channel')
                self.assertEqual(entry.on_fail, 'channel')
                self.assertEqual(entry.retries, map_events.SPLIT_ROAD_CHANNEL_ROUNDS - 1)


class SplitRoadPlanTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(map_events, 'stay_on_channel', return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_first_look_checks_this_channel_before_changing(self):
        plan = map_events.platform_hunt(map_events.SPLIT_ROAD, change_first=False)
        io = RoadIO(enemies=3)
        status = RUNNING
        while status == RUNNING and io.t < 30:
            status = plan.tick(io)
            io.step()
        self.assertEqual(status, DONE)
        self.assertFalse(any(isinstance(p, tuple) for p in io.presses))     # no clicks

    def test_too_few_enemies_goes_to_the_channel_change(self):
        plan = map_events.platform_hunt(map_events.SPLIT_ROAD, change_first=False)
        io = RoadIO(enemies=1)
        while plan.index != 0 and io.t < 30:
            plan.tick(io)
            io.step()
        self.assertEqual(plan.index, 0)
        self.assertIsInstance(plan.current, map_events.HopChannel)

    def test_rest_plan_order(self):
        plan = map_events.rest_on_chair(map_events.SPLIT_ROAD)
        names = [type(e.step).__name__ for e in plan.entries]
        self.assertEqual(names[:5], ['GoToSpot', 'Wait', 'SitUntilHealed', 'ClimbOnto',
                                     'ScoutPlatform'])
        io = RoadIO(x=0.12, hp=15.0, heal=5.0, enemies=1)
        status = RUNNING
        arrived = chair = None
        while status == RUNNING and io.t < 120:
            status = plan.tick(io)
            if arrived is None and plan.index == 1:
                arrived = io.t                  # standing on the safe spot
            if chair is None and 'c' in io.presses:
                chair = io.t
            io.step()
        self.assertEqual(status, DONE)
        self.assertGreaterEqual(chair - arrived, 4.0)
        self.assertGreaterEqual(io.hp_value, 98.0)
        self.assertEqual(io.y, 0.176)

    def test_healed_with_no_enemy_on_top_changes_channel(self):
        plan = map_events.rest_on_chair(map_events.SPLIT_ROAD)
        io = RoadIO(x=0.12, hp=15.0, heal=5.0, enemies=0)
        while not isinstance(plan.current, map_events.HopChannel) and io.t < 120:
            self.assertEqual(plan.tick(io), RUNNING)
            io.step()
        self.assertIsInstance(plan.current, map_events.HopChannel)
        self.assertGreaterEqual(io.hp_value, 98.0)
        self.assertEqual(io.y, 0.176)

    def test_stay_on_channel_rest_ends_on_top(self):
        with mock.patch.object(map_events, 'stay_on_channel', return_value=True):
            plan = map_events.rest_on_chair(map_events.SPLIT_ROAD)
        self.assertEqual(type(plan.entries[-1].step).__name__, 'ClimbOnto')

    def test_down_beside_the_platform_jumps_back_up(self):
        plan = map_events.back_onto_platform(map_events.SPLIT_ROAD)
        io = RoadIO(x=0.209, y=0.184)       # as read at the start of loop.mp4's run
        status = RUNNING
        while status == RUNNING and io.t < 30:
            status = plan.tick(io)
            io.step()
        self.assertEqual(status, DONE)
        self.assertEqual(io.y, 0.176)
        self.assertIn('alt', io.presses)

    def test_from_the_bottom_ground_up_two_levels(self):
        """stuck.mp4: stranded at (0.224, 0.209), a step below the safe ledge."""
        plan = map_events.back_onto_platform(map_events.SPLIT_ROAD)
        io = RoadIO(x=0.224, y=0.209)
        status = RUNNING
        heights = set()
        while status == RUNNING and io.t < 40:
            status = plan.tick(io)
            io.step()
            heights.add(io.y)
        self.assertEqual(status, DONE)
        self.assertEqual(io.y, 0.176)
        self.assertIn(0.192, heights)               # via the safe ledge
        self.assertGreaterEqual(io.presses.count('alt'), 2)

    def test_from_far_right_on_the_bottom_walks_to_the_step_first(self):
        plan = map_events.back_onto_platform(map_events.SPLIT_ROAD)
        io = RoadIO(x=0.262, y=0.209)
        status = RUNNING
        while status == RUNNING and io.t < 40:
            status = plan.tick(io)
            io.step()
        self.assertEqual(status, DONE)
        self.assertEqual(io.y, 0.176)

    def test_channel_change_lands_back_on_the_platform(self):
        plan = map_events.empty_map_hop(map_events.SPLIT_ROAD)
        self.assertEqual(type(plan.entries[-1].step).__name__, 'ClimbOnto')


class SplitRoadWatchTests(unittest.TestCase):
    def _tracker(self, offsets):
        return SimpleNamespace(target=lambda: None, enemy_offsets=lambda: offsets)

    def test_cleared_after_five_quiet_seconds_on_the_platform(self):
        watch = map_events.PlatformCleared(PLATFORM)
        capture = SimpleNamespace(mob_tracker=self._tracker([]))
        with mock.patch.object(config, 'capture', capture, create=True), \
                mock.patch.object(config, 'player_pos', (0.1, 0.176), create=True), \
                mock.patch.object(map_events.time, 'monotonic', return_value=0.0), \
                mock.patch.object(config, 'player_pos_time', 0.0, create=True):
            self.assertFalse(watch.update(None, 0.0))
            self.assertFalse(watch.update(None, 4.9))
            self.assertTrue(watch.update(None, 5.1))

    def test_an_enemy_in_sight_restarts_the_count(self):
        watch = map_events.PlatformCleared(PLATFORM)
        tracker = self._tracker([])
        capture = SimpleNamespace(mob_tracker=tracker)
        with mock.patch.object(config, 'capture', capture, create=True), \
                mock.patch.object(config, 'player_pos', (0.1, 0.176), create=True), \
                mock.patch.object(map_events.time, 'monotonic', return_value=0.0), \
                mock.patch.object(config, 'player_pos_time', 0.0, create=True):
            watch.update(None, 0.0)
            tracker.enemy_offsets = lambda: [120]
            self.assertFalse(watch.update(None, 4.0))
            tracker.enemy_offsets = lambda: []
            self.assertFalse(watch.update(None, 8.0))
            self.assertTrue(watch.update(None, 13.1))

    def _off(self, position, times):
        watch = map_events.OffThePlatform(PLATFORM, map_events._tiers(SPOTS))
        with mock.patch.object(config, 'player_pos', position, create=True), \
                mock.patch.object(config, 'player_pos_time', 0.0, create=True), \
                mock.patch.object(map_events.time, 'monotonic', return_value=0.0):
            return [watch.update(None, t) for t in times]

    def test_off_the_platform_after_a_second_on_the_low_ground(self):
        self.assertEqual(self._off((0.209, 0.184), [0.0, 0.5, 1.1]), [False, False, True])
        self.assertEqual(self._off((0.207, 0.192), [0.0, 1.1]), [False, True])

    def test_off_on_the_bottom_ground_too(self):
        """stuck.mp4: (0.224, 0.209) was missed by the old safe-height-only check."""
        self.assertEqual(self._off((0.224, 0.209), [0.0, 1.1]), [False, True])

    def test_far_away_on_the_hill_is_not_off(self):
        self.assertEqual(self._off((0.32, 0.209), [0.0, 2.0]), [False, False])

    def test_on_the_platform_end_is_not_off(self):
        self.assertEqual(self._off((0.197, 0.176), [0.0, 2.0]), [False, False])
        self.assertEqual(self._off((0.150, 0.176), [0.0, 2.0]), [False, False])

    def test_low_hp(self):
        watch = map_events.HealthBelow(20)
        capture = SimpleNamespace(health=SimpleNamespace(current_percent=lambda: 19.0))
        with mock.patch.object(config, 'capture', capture, create=True), \
                mock.patch('src.modules.health.is_game_focused', return_value=True):
            self.assertFalse(watch.update(None, 0.0))
            self.assertTrue(watch.update(None, 0.5))


class MapLimitFilterTests(unittest.TestCase):
    """2-point.mp4: the signpost past the right end (x ~0.207) was locked as an enemy."""

    def _filter(self, items, me_x=0.15, player_screen_x=500):
        import numpy as np
        from src.modules import mob_tracker as tm
        tracker = tm.MobTracker.__new__(tm.MobTracker)
        bot = SimpleNamespace(map_events=SimpleNamespace(limits=lambda: {'left': 0.021,
                                                                         'right': 0.197}))
        capture = SimpleNamespace(minimap={'minimap': np.zeros((68, 239, 3), np.uint8)})
        with mock.patch.object(tm._config, 'bot', bot, create=True), \
                mock.patch.object(tm._config, 'capture', capture, create=True), \
                mock.patch.object(tm._config, 'player_pos', (me_x, 0.176), create=True), \
                mock.patch.object(tm._config, 'player_pos_time', 5.0, create=True), \
                mock.patch.object(tm.time, 'monotonic', return_value=5.0):
            return tracker._inside_map_limits(items, player_screen_x)

    def test_signpost_past_the_right_end_is_dropped(self):
        scale = 16 * 239
        signpost = ('match', 500 + (0.207 - 0.15) * scale, 600, None)
        on_platform_end = ('match', 500 + (0.196 - 0.15) * scale, 600, None)
        kept = self._filter([signpost, on_platform_end])
        self.assertEqual(kept, [on_platform_end])

    def test_a_hurt_enemy_past_the_end_is_kept(self):
        hurt = ('match', 500 + 0.06 * 16 * 239, 600, 0.4)
        self.assertEqual(self._filter([hurt]), [hurt])


class DropPastLimitTests(unittest.TestCase):
    """stuck.mp4: a drop down past the platform's end was walked to, off the edge."""

    def _past(self, x, offset):
        import numpy as np
        from src.modules.bot import Bot
        from src.modules.mob_tracker import MobTracker
        bot = Bot.__new__(Bot)
        bot.map_events = SimpleNamespace(limits=lambda: {'left': 0.021, 'right': 0.197})
        tracker = MobTracker.__new__(MobTracker)
        capture = SimpleNamespace(mob_tracker=tracker,
                                  minimap={'minimap': np.zeros((68, 239, 3), np.uint8)})
        with mock.patch.object(config, 'capture', capture, create=True):
            return bot._drop_past_limit(x, offset)

    def test_drop_on_the_low_ground_is_skipped(self):
        self.assertTrue(self._past(0.185, (0.226 - 0.185) * 16 * 239))

    def test_drop_on_the_platform_is_kept(self):
        self.assertFalse(self._past(0.150, (0.190 - 0.150) * 16 * 239))
        self.assertFalse(self._past(0.150, -(0.150 - 0.030) * 16 * 239))


class ChatAlertTests(unittest.TestCase):
    """stuck.mp4: scenery above the chat box read as new white lines (9 at a time)."""

    def test_scenery_above_the_chat_box_is_not_chat(self):
        import cv2
        from src.modules import chat_watch as cw
        video = cv2.VideoCapture('assets/stuck.mp4')
        ok, frame = video.read()
        video.release()
        if not ok:
            self.skipTest('clip not available')
        game = frame[0:760, 70:1170]
        bar = cw.find_all_bar(game)
        self.assertIsNotNone(bar)
        self.assertEqual(cw.chat_lines(game, cw.log_region(game, bar)), [])

    def test_whispers_are_not_sent_unless_switched_on(self):
        from src.modules import chat_watch as cw
        sent = []
        clock = [0.0]
        lines = [[]]
        watch = cw.ChatWatch(frame_source=lambda: 'frame',
                             send=lambda text, photo: sent.append(text),
                             wants=lambda: True, running=lambda: True, clock=lambda: clock[0])
        with mock.patch.object(cw, 'find_all_bar', return_value=(0, 0, 10, 10)), \
                mock.patch.object(cw, 'log_region', return_value=(0, 0, 1, 1)), \
                mock.patch.object(cw, 'chat_lines', side_effect=lambda f, r: lines[0]):
            watch.check()                                   # primed
            lines[0] = [('whisper', 'w1', 5)]
            clock[0] += 30
            watch.check()
            self.assertEqual(sent, [])
            lines[0] = [('whisper', 'w1', 5), ('say', 's1', 10)]
            clock[0] += 30
            watch.check()
        self.assertEqual(len(sent), 1)
        self.assertIn('1 chat line from someone on this map', sent[0])
        self.assertNotIn('whisper', sent[0])


if __name__ == '__main__':
    unittest.main()
