import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2

from src.common import config, vkeys
from src.modules import bot as bot_module
from src.modules import loot_tracker as loot_module
from src.modules.bot import Bot
from src.modules.loot_planner import LootPlanner


class LootPlannerTests(unittest.TestCase):
    def _taps(self, planner, start, end, x, level='ground', attacking=False, step=0.05):
        times, now = [], start
        while now < end:
            taps, _ = planner.taps_due(now, x, level, attacking)
            if taps:
                times.append(round(now, 2))
            now += step
        return times

    def test_kill_taps_while_its_drops_land(self):
        planner = LootPlanner()
        planner.killed(10.0, 0.5, 'ground')
        taps = self._taps(planner, 10.0, 11.2, 0.5)
        # the burst: one kill's drops (3 + 2 per kill), which also sweeps the spot
        self.assertEqual(len(taps), LootPlanner.SWEEP_TAPS + LootPlanner.TAPS_PER_KILL)
        self.assertGreaterEqual(taps[0], 10.30)

    def test_a_pile_from_several_kills_gets_spammed_like_a_player(self):
        planner = LootPlanner()
        for kill in range(4):                   # four pigs died here
            planner.fought(10.0 + kill, 0.5, 'ground')
            planner.spots[0].kills += 1
        times = self._taps(planner, 20.0, 23.0, 0.5, step=0.02)
        self.assertEqual(len(times), LootPlanner.SWEEP_TAPS + 4 * LootPlanner.TAPS_PER_KILL)
        gaps = [b - a for a, b in zip(times, times[1:])]
        self.assertTrue(all(0.08 <= gap <= 0.17 for gap in gaps), gaps)
        self.assertGreater(len(set(round(g, 2) for g in gaps)), 1)  # not metronomic

    def test_spot_taps_are_capped(self):
        planner = LootPlanner()
        planner.fought(10.0, 0.5, 'ground')
        planner.spots[0].kills = 50
        times = self._taps(planner, 20.0, 30.0, 0.5, step=0.02)
        self.assertEqual(len(times), LootPlanner.MAX_SPOT_TAPS)

    def test_fighting_taps_now_and_then(self):
        planner = LootPlanner()
        now, taps = 10.0, 0
        while now < 15.0:
            planner.fought(now, 0.5, 'ground')
            taps += planner.taps_due(now, 0.5, 'ground', True)[0]
            now += 0.05
        self.assertEqual(taps, 5)               # one per second

    def test_walking_over_a_fight_spot_sweeps_it_once(self):
        planner = LootPlanner()
        planner.fought(10.0, 0.50, 'ground')
        # Walk left to right across it at 0.04 minimap/s, 20 ms bot ticks
        now, x, taps, swept = 12.0, 0.46, 0, []
        while x < 0.54:
            n, spot = planner.taps_due(now, x, 'ground', False)
            taps += n
            if spot:
                swept.append(spot)
            x += 0.0008
            now += 0.02
        self.assertEqual(taps, LootPlanner.SWEEP_TAPS)
        self.assertEqual(len(swept), 1)
        # walking back over it: already swept, no taps
        self.assertEqual(self._taps(planner, 20.0, 22.0, 0.50), [])

    def test_no_taps_away_from_spots_or_before_drops_settle(self):
        planner = LootPlanner()
        planner.fought(10.0, 0.50, 'ground')
        self.assertEqual(self._taps(planner, 12.0, 14.0, 0.70), [])
        self.assertEqual(self._taps(planner, 10.0, 10.25, 0.50), [])

    def test_a_spot_on_the_step_is_not_swept_from_the_ground(self):
        planner = LootPlanner()
        planner.fought(10.0, 0.65, 'step')
        self.assertEqual(self._taps(planner, 12.0, 14.0, 0.65, level='ground'), [])
        self.assertEqual(planner.pending(14.0, 'step'), 1)

    def test_collect_walks_to_the_nearest_spot_then_stays(self):
        planner = LootPlanner()
        planner.fought(10.0, 0.40, 'ground')
        planner.fought(10.0, 0.70, 'ground')
        self.assertEqual(planner.collect_direction(12.0, 0.45, 'ground'), 'left')
        self.assertEqual(planner.collect_direction(12.5, 0.402, 'ground'), 'stay')
        self.assertIsNone(planner.collect_direction(12.0, 0.95, 'ground'))  # too far

    def test_collect_gives_up_on_a_spot_it_cannot_reach(self):
        planner = LootPlanner()
        planner.fought(10.0, 0.40, 'ground')
        self.assertEqual(planner.collect_direction(12.0, 0.50, 'ground'), 'left')
        self.assertIsNone(planner.collect_direction(
            12.0 + LootPlanner.COLLECT_TIMEOUT, 0.50, 'ground'))

    def test_spots_expire(self):
        planner = LootPlanner()
        planner.fought(10.0, 0.40, 'ground')
        self.assertEqual(planner.pending(10.0 + LootPlanner.SPOT_TTL + 1, 'ground'), 0)


class SmartLootBotTests(unittest.TestCase):
    """The bot fights at 0.50, the pig dies, then nothing else is around."""

    def _run(self, ticks, target, positions, kills=lambda tick: [], loot_enabled=True):
        bot = Bot.__new__(Bot)
        bot._patrol_direction = 'right'
        bot._patrol_until = 0
        bot._auto_attack_movement = None
        bot._mob_attack_keys = ()
        bot._walls = {}
        bot.command_book = SimpleNamespace(
            dict={'cruelstab': object()},
            module=SimpleNamespace(Key=SimpleNamespace(JUMP='alt', CRUEL_STAB='a')))
        tracker = SimpleNamespace(
            settings=lambda: {'Enabled': True},
            set_sweep_direction=lambda direction: None,
            set_blocked_side=lambda side: None,
            consume_attack_action=lambda: (None, False, 0),
            group_in_range=lambda: False,
            waiting=lambda: False,
        )
        loot = SimpleNamespace(
            settings=lambda: {'Enabled': loot_enabled},
            consume_pickup=lambda: None, LOOT_KEY='x')
        config_stub = SimpleNamespace(
            player_pos=None, player_pos_time=None, enabled=True,
            capture=SimpleNamespace(mob_tracker=tracker, loot_tracker=loot),
            routine=SimpleNamespace(sequence=[], __len__=lambda: 0),
        )
        pressed, held = [], []
        clock = [100.0]
        state = {'x': positions(0)}
        with patch.object(bot_module, 'config', config_stub), \
                patch.object(bot_module, 'key_down'), patch.object(bot_module, 'key_up'), \
                patch.object(bot_module, 'press',
                             lambda key, n=1, *a, **k: pressed.extend([(round(clock[0], 2), key)] * n)), \
                patch.object(bot_module, 'is_game_focused', return_value=True), \
                patch.object(bot_module, 'FALL_RECOVERY', False), \
                patch.object(bot_module.time, 'sleep'), \
                patch.object(bot_module.time, 'monotonic', lambda: clock[0]):
            bot._set_patrol_home = lambda: None
            for tick in range(ticks):
                x = positions(tick, bot._auto_attack_movement, state)
                config_stub.player_pos = (x, 0.7)
                tracker.target = lambda tick=tick: target(tick)
                tracker.consume_kills = lambda tick=tick: kills(tick)
                bot._run_standalone_mob_attack()
                held.append(bot._auto_attack_movement)
                clock[0] += 0.1
        return bot, pressed, held

    @staticmethod
    def _walker(start):
        def positions(tick, moving=None, state=None):
            if state is None:
                return start
            if moving == 'right':
                state['x'] += 0.01
            elif moving == 'left':
                state['x'] -= 0.01
            return round(state['x'], 4)
        return positions

    def test_kill_burst_then_walks_back_to_sweep_the_fight_spot(self):
        # Fights at 0.50 for 2 s, the pig dies (tracker reports it), then
        # patrol carries the character off to the right
        target = lambda tick: (30, True) if tick < 20 else None
        kills = lambda tick: [(0, 30)] if tick == 20 else []
        bot, pressed, held = self._run(80, target, self._walker(0.50), kills)
        loot_taps = [t for t, key in pressed if key == 'x']
        self.assertTrue(any(100.0 <= t < 102.0 for t in loot_taps))    # while fighting
        after_kill = [t for t in loot_taps if 102.0 <= t <= 103.1]
        self.assertGreaterEqual(len(after_kill), 3)                      # the burst
        self.assertEqual(bot._loot_planner.pending(110.0, 'ground'), 0)  # swept

    def test_vanished_target_counts_as_a_kill_without_a_report(self):
        target = lambda tick: (30, True) if tick < 10 else None
        bot, pressed, _ = self._run(25, target, self._walker(0.50))
        after = [t for t, key in pressed if key == 'x' and t >= 101.0]
        self.assertGreaterEqual(len(after), 3)

    def test_nothing_to_fight_walks_to_an_unswept_spot(self):
        bot, _, held = self._run(1, lambda tick: None, self._walker(0.60))
        bot._loot_planner.fought(90.0, 0.50, 'ground')
        _, _, held = self._run_more(bot, 30)
        self.assertEqual(held[0], 'left')

    def _run_more(self, bot, ticks):
        planner = bot._loot_planner
        original_new = Bot.__new__
        with patch.object(Bot, '__new__', lambda cls: bot):
            bot2, pressed, held = self._run(ticks, lambda tick: None, self._walker(0.60))
        self.assertIs(bot2._loot_planner, planner)
        return bot2, pressed, held

    def test_looting_off_sends_no_loot_taps(self):
        target = lambda tick: (30, True) if tick < 20 else None
        _, pressed, _ = self._run(60, target, self._walker(0.50), loot_enabled=False)
        self.assertNotIn('x', [key for _, key in pressed])


class LootTrackerReplayTests(unittest.TestCase):
    """assets/loot_issue.mp4: the bot walked over a dozen drops, no pickups."""
    VIDEO = os.path.join('assets', 'loot_issue.mp4')
    GAME_LEFT = 1176        # the recording shows the log on the left

    @unittest.skipUnless(os.path.exists(VIDEO), 'recording not available')
    def test_pickups_keep_coming_at_the_live_scan_rate(self):
        tracker = loot_module.LootTracker.__new__(loot_module.LootTracker)
        loot_module.LootTracker.__init__(tracker)
        badge = cv2.imread(os.path.join('.settings', 'mob_player_badge.png'),
                           cv2.IMREAD_GRAYSCALE)
        if badge is None or not tracker._loot_templates:
            self.skipTest('player badge / loot samples not calibrated here')
        tracker.config['Enabled'] = True
        tracker._player_template = badge
        clock = [0.0]
        pickups = []
        capture = cv2.VideoCapture(self.VIDEO)
        with patch.object(config, 'capture', SimpleNamespace(mob_tracker=None)), \
                patch.object(loot_module, 'is_game_focused', return_value=True), \
                patch.object(loot_module.time, 'monotonic', lambda: clock[0]), \
                patch.object(tracker, 'save_config', lambda: None, create=True):
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                clock[0] = index / 30.0
                if index % 2 == 0:      # the live capture loop runs ~15 frames/s here
                    tracker.process_frame(frame[:, self.GAME_LEFT:], True)
                    if tracker.consume_pickup():
                        pickups.append(clock[0])
                index += 1
        # The old frame-count scans + screen-spot latch managed 4, with a 3.3 s block
        self.assertGreaterEqual(len(pickups), 7)
        self.assertLess(max(b - a for a, b in zip(pickups, pickups[1:])), 2.5)


class QuietKeyTests(unittest.TestCase):
    def test_quiet_keys_skip_the_key_log(self):
        with patch('builtins.print') as printed, \
                patch.object(vkeys, 'QUIET_KEYS', {'x'}), \
                patch.object(vkeys, 'LOG_SENDINPUT', True), \
                patch.object(vkeys, 'LOG_SENDINPUT_QUIET_KEYS', False):
            for key in ('x', 'z'):
                for up in (False, True):
                    x = vkeys._key_input(key, up)
                    vkeys._log_key_send(x, key, 'UP' if up else 'DOWN', None, 'ok')
        self.assertEqual(printed.call_count, 2)


class DropFinderTests(unittest.TestCase):
    """assets/loot_issue-v2.mp4: mesos left lying by the Pig Beach cliff."""
    VIDEO = os.path.join('assets', 'loot_issue-v2.mp4')

    def _frame(self, path, index):
        capture = cv2.VideoCapture(path)
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        return frame if ok else None

    def _badge(self, frame):
        badge = cv2.imread(os.path.join('.settings', 'mob_player_badge.png'))
        if badge is None:
            self.skipTest('player badge not calibrated here')
        _, score, _, loc = cv2.minMaxLoc(cv2.matchTemplate(frame[:, :, :3], badge, cv2.TM_CCOEFF_NORMED))
        if score < 0.6:
            return None
        return loc, (loc[0] + badge.shape[1], loc[1] + badge.shape[0])

    @unittest.skipUnless(os.path.exists(VIDEO), 'recording not available')
    def test_sees_the_mesos_left_by_the_cliff(self):
        from src.detection.drop_finder import DropFinder
        frame = self._frame(self.VIDEO, 100)
        bounds = self._badge(frame)
        self.assertIsNotNone(bounds)
        player_x = (bounds[0][0] + bounds[1][0]) / 2
        drops = DropFinder().find(frame, bounds)
        xs = sorted(round(player_x + d[0]) for d in drops)
        self.assertTrue(any(abs(x - 435) <= 8 for x in xs), xs)     # the gold meso
        self.assertTrue(any(abs(x - 534) <= 8 for x in xs), xs)     # a bronze meso

    @unittest.skipUnless(os.path.exists(os.path.join('assets', 'iron_hog_issue.mp4')),
                         'recording not available')
    def test_nothing_on_clean_ground(self):
        from src.detection.drop_finder import DropFinder
        finder = DropFinder()
        for index in (0, 300, 400):
            frame = self._frame(os.path.join('assets', 'iron_hog_issue.mp4'), index)
            bounds = self._badge(frame)
            if bounds is None:
                continue
            self.assertEqual(finder.find(frame, bounds), [], index)


class JumperSwingTests(unittest.TestCase):
    def _bot(self):
        bot = Bot.__new__(Bot)
        bot._mob_attack_keys = ()
        bot._auto_attack_movement = None
        bot.command_book = SimpleNamespace(
            dict={'cruelstab': object()},
            module=SimpleNamespace(Key=SimpleNamespace(CRUEL_STAB='a')))
        return bot

    def _attack(self, bot, offset, clock, group=False, group_action=(None, False, 0)):
        tracker = SimpleNamespace(
            consume_attack_action=lambda: group_action, target=lambda: (offset, True),
            group_in_range=lambda: group, level_farming_enabled=lambda: True)
        down, up = [], []
        with patch.object(bot_module, 'config', SimpleNamespace(
                capture=SimpleNamespace(mob_tracker=tracker), enabled=True)), \
                patch.object(bot_module, 'key_down', down.append), \
                patch.object(bot_module, 'key_up', up.append), \
                patch.object(bot_module, 'press'), \
                patch.object(bot_module, 'is_game_focused', return_value=True), \
                patch.object(bot_module.time, 'sleep'), \
                patch.object(bot_module.time, 'monotonic', lambda: clock):
            result = bot._service_mob_attack()
        return result, down, up

    def test_holds_arrow_and_attack_together(self):
        # spawn_issue.mp4: the attack key held alone swung only once
        bot = self._bot()
        for clock in (10.0, 10.3, 10.7, 11.5):
            self._attack(bot, -50, clock)
            self.assertEqual(bot._mob_attack_keys, ('left', 'a'))

    def test_keeps_swinging_at_a_group_between_q_casts(self):
        bot = self._bot()
        result, _, _ = self._attack(bot, 40, 10.0, group=True)
        self.assertTrue(result)
        self.assertEqual(bot._mob_attack_keys, ('right', 'a'))  # not standing idle


class VisibleDropTests(unittest.TestCase):
    def _run(self, bot, drops, x, clock, target=None):
        loot = SimpleNamespace(settings=lambda: {'Enabled': True}, LOOT_KEY='x',
                               ground_drops=lambda: drops)
        pressed = []
        with patch.object(bot_module, 'config', SimpleNamespace(
                capture=SimpleNamespace(loot_tracker=loot), enabled=True)), \
                patch.object(bot_module, 'key_down'), \
                patch.object(bot_module, 'key_up'), \
                patch.object(bot_module, 'press', lambda key, n=1, **k: pressed.append(key)), \
                patch('builtins.print'), \
                patch.object(bot_module.time, 'monotonic', lambda: clock):
            result = bot._collect_visible_drop(x, target)
        return result, pressed

    def _bot(self):
        bot = Bot.__new__(Bot)
        bot._auto_attack_movement = None
        bot._walls = {}
        return bot

    def test_walks_to_a_drop_then_picks_it_up(self):
        bot = self._bot()
        result, _ = self._run(bot, [(-80, 180, 0.9)], 0.50, 10.0)
        self.assertTrue(result)
        self.assertEqual(bot._auto_attack_movement, 'left')
        result, pressed = self._run(bot, [(-5, 180, 0.9)], 0.48, 10.5)
        self.assertTrue(result)
        self.assertIsNone(bot._auto_attack_movement)
        self.assertEqual(pressed, ['x'])

    def test_a_drop_behind_a_wall_is_left_alone(self):
        bot = self._bot()
        self._run(bot, [(-80, 180, 0.9)], 0.50, 10.0)
        result, _ = self._run(bot, [(-80, 180, 0.9)], 0.50, 11.0)     # not moving
        self.assertFalse(result)
        result, _ = self._run(bot, [(-80, 180, 0.9)], 0.50, 12.0)
        self.assertFalse(result)                                      # banned there

    def test_a_drop_it_cannot_pick_up_is_given_up(self):
        bot = self._bot()
        clock = 10.0
        for _ in range(Bot.DROP_MAX_TAPS + 2):
            self._run(bot, [(0, 180, 0.9)], 0.50, clock)
            clock += 0.3
        self.assertFalse(self._run(bot, [(0, 180, 0.9)], 0.50, clock)[0])

    def test_with_a_pig_coming_only_a_near_drop_is_worth_it(self):
        bot = self._bot()
        self.assertFalse(self._run(bot, [(-120, 180, 0.9)], 0.5, 10.0, target=(100, False))[0])
        # looting-strike-issue.mp4: a coin 90 px in front of a pig was "on the
        # way"; the bot stepped to it and was hit before it could swing
        self.assertFalse(self._run(bot, [(60, 180, 0.9)], 0.5, 10.0, target=(150, False))[0])
        self.assertTrue(self._run(bot, [(60, 180, 0.9)], 0.5, 10.0, target=(220, False))[0])
        bot = self._bot()   # behind the bot, pig still far: fine; pig close: no
        self.assertTrue(self._run(bot, [(-60, 180, 0.9)], 0.5, 10.0, target=(300, False))[0])
        bot = self._bot()
        self.assertFalse(self._run(bot, [(-60, 180, 0.9)], 0.5, 10.0, target=(180, False))[0])
        bot = self._bot()
        self.assertFalse(self._run(bot, [(300, 180, 0.9)], 0.5, 10.0, target=(350, False))[0])

    def test_scenery_that_looks_like_a_drop_is_remembered(self):
        # (the cut log by the Pig Beach tree matches a bronze coin)
        bot = self._bot()
        clock = 10.0
        for _ in range(bot.DROP_MAX_TAPS + 2):
            self._run(bot, [(0, 180, 0.9)], 0.50, clock)
            clock += 0.3
        clock += bot.DROP_BAN_SECONDS + 5       # long after the short ban
        self.assertFalse(self._run(bot, [(0, 180, 0.9)], 0.50, clock)[0])
        # seen again from 120 px to its left: not worth the walk
        away = 0.50 - 120 / bot.PX_PER_MINIMAP
        self.assertFalse(self._run(bot, [(120, 180, 0.9)], away, clock)[0])
        # a drop somewhere else is still collected
        self.assertTrue(self._run(bot, [(-150, 180, 0.9)], away, clock)[0])

    def test_never_into_the_pit(self):
        bot = self._bot()
        bot.map_events = SimpleNamespace(limits=lambda: {'right': 0.645})
        self.assertFalse(self._run(bot, [(80, 180, 0.9)], 0.640, 10.0)[0])


class SweepOrderTests(unittest.TestCase):
    def test_a_row_of_spots_is_swept_in_one_pass(self):
        # looting-strike-issue.mp4: 0.203 -> 0.189 -> 0.217 -> 0.231, a turn at each
        planner = LootPlanner()
        for spot_x in (0.189, 0.203, 0.217, 0.231):
            planner.fought(10.0, spot_x, 'ground')
        x, now, turns, last = 0.206, 12.0, 0, None
        for _ in range(3000):
            direction = planner.collect_direction(now, x, 'ground')
            if direction is None:
                break
            planner.taps_due(now, x, 'ground', False)
            if direction in ('left', 'right'):
                if last is not None and direction != last:
                    turns += 1
                last = direction
                x += 0.0008 if direction == 'right' else -0.0008
            now += 0.02
        self.assertEqual(planner.pending(now, 'ground'), 0)
        self.assertLessEqual(turns, 1)


class LootHandTests(unittest.TestCase):
    def test_taps_do_not_block_and_always_release(self):
        import threading
        import time as real_time
        from src.modules.loot_hand import LootHand
        events, done = [], threading.Event()

        def up(key):
            events.append(('up', key))
            if sum(1 for e in events if e[0] == 'up') >= 2:
                done.set()

        hand = LootHand(lambda key: events.append(('down', key)), up, lambda: True)
        started = real_time.monotonic()
        for _ in range(10):
            hand.tap('x')                       # asked far faster than a finger
        self.assertLess(real_time.monotonic() - started, 0.05)     # never waits
        self.assertTrue(done.wait(2.0))
        real_time.sleep(0.4)
        # (stale requests are dropped: at most the one in hand + MAX_PENDING)
        downs = [e for e in events if e[0] == 'down']
        self.assertLessEqual(len(downs), hand.MAX_PENDING + 1)
        self.assertEqual([e[0] for e in events], ['down', 'up'] * len(downs))

    def test_no_taps_while_paused_or_unfocused(self):
        import time as real_time
        from src.modules.loot_hand import LootHand
        events = []
        hand = LootHand(lambda key: events.append('down'), lambda key: events.append('up'),
                        lambda: False)
        hand.tap('x', 2)
        real_time.sleep(0.2)
        self.assertEqual(events, [])


class GroupSkillGuardTests(unittest.TestCase):
    def _ready(self, bot, mana, clock):
        capture = SimpleNamespace(mana=SimpleNamespace(current_percent=lambda: mana))
        with patch.object(bot_module, 'config', SimpleNamespace(capture=capture)), \
                patch.object(bot_module.time, 'monotonic', lambda: clock):
            return bot._group_q_ready()

    def test_no_second_q_right_after_one_that_fired(self):
        bot = Bot.__new__(Bot)
        self.assertTrue(self._ready(bot, 80.0, 10.0))
        bot._group_q = (10.0, 80.0)
        # the cast cost MP (the two Qs were 0.53 s apart in the clip)
        self.assertFalse(self._ready(bot, 76.0, 10.2))
        self.assertFalse(self._ready(bot, 76.0, 10.6))
        self.assertTrue(self._ready(bot, 76.0, 10.05 + bot.GROUP_Q_COOLDOWN))

    def test_a_q_the_game_dropped_is_pressed_again(self):
        bot = Bot.__new__(Bot)
        bot._group_q = (10.0, 80.0)
        self.assertFalse(self._ready(bot, 80.0, 10.1))
        self.assertTrue(self._ready(bot, 80.0, 10.0 + bot.GROUP_Q_RETRY + 0.05))    # no MP spent


class StrikeClipTests(unittest.TestCase):
    VIDEO = os.path.join('assets', 'looting-strike-issue.mp4')

    @unittest.skipUnless(os.path.exists(VIDEO), 'clip not available')
    def test_the_pig_by_the_character_stays_locked(self):
        """
        One pig walks and hops around the character for 15 s. Its HP bar used to
        teach the tracker a "raised step" 33 px up, after which the pig was
        dropped as out of reach on every hop: no swings while it bit.
        """
        from src.modules import mob_tracker as tracker_module
        if not os.path.exists(tracker_module.MobTracker.PLAYER_TEMPLATE_PATH):
            self.skipTest('no calibrated samples')
        clock = [0.0]
        video = cv2.VideoCapture(self.VIDEO)
        fps = video.get(cv2.CAP_PROP_FPS) or 30.0
        with patch.object(tracker_module, 'is_game_focused', return_value=True), \
                patch.object(tracker_module.time, 'monotonic', lambda: clock[0]):
            tracker = tracker_module.MobTracker()
            if 'jumper' not in tracker.enemy_speed_profiles():
                self.skipTest('the live samples are not the Pig Beach ones')
            index, out_of_reach, scans = 0, 0, 0
            while True:
                ok, frame = video.read()
                if not ok:
                    break
                clock[0] = index / fps
                index += 1
                tracker.process_frame(frame[:, 1026:], True)
                if 5.2 <= clock[0] and index % 3 == 0:
                    scans += 1
                    out_of_reach += 'out of reach' in tracker.status()
        video.release()
        self.assertGreater(scans, 100)
        self.assertEqual(out_of_reach, 0)
        self.assertIsNone(tracker.ledge_height())       # there is no step in view


if __name__ == '__main__':
    unittest.main()
