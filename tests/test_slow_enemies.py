"""Slow walkers (snails) and the optional group skill - knock-back-dmg.mp4."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2

from src.modules import bot as bot_module
from src.modules import mob_tracker as tracker_module
from src.modules.bot import Bot


def _bot():
    bot = Bot.__new__(Bot)
    bot._mob_attack_keys = ()
    bot._auto_attack_movement = None
    bot.command_book = SimpleNamespace(
        dict={'cruelstab': object()},
        module=SimpleNamespace(Key=SimpleNamespace(JUMP='alt', CRUEL_STAB='ctrl')))
    return bot


class FakeTracker:
    def __init__(self, group_skill=True, target=(-40, True), offsets=()):
        self.group_skill = group_skill
        self._target = target
        self._offsets = list(offsets)

    def group_skill_enabled(self):
        return self.group_skill

    def consume_attack_action(self):
        return ('left', True, 3)        # "a pile in range: use Q"

    def target(self):
        return self._target

    def group_in_range(self):
        return True

    def level_farming_enabled(self):
        return False

    def enemy_offsets(self):
        return self._offsets


class GroupSkillTests(unittest.TestCase):
    def _attack(self, tracker):
        bot = _bot()
        presses, downs = [], []
        config = SimpleNamespace(capture=SimpleNamespace(mob_tracker=tracker), enabled=True)
        with patch.object(bot_module, 'config', config), \
                patch.object(bot_module, 'is_game_focused', return_value=True), \
                patch.object(bot_module, 'press', lambda key, *a, **k: presses.append(key)), \
                patch.object(bot_module, 'key_down', downs.append), \
                patch.object(bot_module, 'key_up', lambda key: None), \
                patch.object(bot_module.time, 'sleep', lambda s: None), \
                patch.object(Bot, '_percent', lambda self, m: 100.0):
            bot._service_mob_attack()
            pressed_q = bot._press_q('left')
        return presses, downs, bot, pressed_q

    def test_q_is_used_on_a_pile_when_the_character_has_it(self):
        presses, _, _, pressed_q = self._attack(FakeTracker(group_skill=True))
        self.assertIn('q', presses)
        self.assertTrue(pressed_q)

    def test_no_q_without_the_skill_and_the_pile_is_still_attacked(self):
        presses, downs, bot, pressed_q = self._attack(FakeTracker(group_skill=False))
        self.assertNotIn('q', presses)
        self.assertFalse(pressed_q)
        # swinging instead: faced left with a tap, the attack key held alone
        self.assertEqual(bot._mob_attack_keys, ('ctrl',))
        self.assertEqual(bot._facing, 'left')

    def test_stationary_jumper_swing_faces_without_holding_the_arrow(self):
        bot = _bot()
        bot._facing = 'left'
        downs, ups, attacks = [], [], []
        hand = SimpleNamespace(key=None, hold=attacks.append)
        bot._attack_hand = lambda: hand
        with patch.object(bot_module, 'key_down', downs.append), \
                patch.object(bot_module, 'key_up', ups.append), \
                patch.object(bot_module.time, 'sleep'), \
                patch.object(bot_module.time, 'monotonic', return_value=10.0), \
                patch.object(bot, '_watch_hold', return_value=True):
            bot._hold_attack(
                'right', 'ctrl', FakeTracker(), stationary=True
            )

        self.assertEqual(downs, ['right'])
        self.assertEqual(ups, ['right'])
        self.assertEqual(attacks, ['ctrl'])
        self.assertEqual(bot._mob_attack_keys, ('ctrl',))

    def test_a_pile_in_range_between_qs_keeps_the_swing_going(self):
        """It used to let go of the attack and stand in the pile (snail map)."""
        bot = _bot()
        bot._group_q = (bot_module.time.monotonic(), 100.0)    # Q just fired
        config = SimpleNamespace(
            capture=SimpleNamespace(mob_tracker=FakeTracker()), enabled=True)
        with patch.object(bot_module, 'config', config), \
                patch.object(bot_module, 'is_game_focused', return_value=True), \
                patch.object(bot_module, 'press', lambda *a, **k: None), \
                patch.object(bot_module, 'key_down', lambda key: None), \
                patch.object(bot_module, 'key_up', lambda key: None):
            self.assertTrue(bot._service_mob_attack())
        self.assertEqual(bot._mob_attack_keys, ('ctrl',))
        self.assertEqual(bot._facing, 'left')

    def test_settings_from_before_the_option_count_as_having_q(self):
        tracker = tracker_module.MobTracker.__new__(tracker_module.MobTracker)
        tracker.config = {'Group skill': ''}
        self.assertTrue(tracker._group_skill_locked())
        tracker.config = {'Group skill': False}
        self.assertFalse(tracker._group_skill_locked())


class DropWalkTests(unittest.TestCase):
    def test_a_drop_past_another_enemy_is_not_walked_to(self):
        bot = _bot()
        for name in ('_drop_banned', '_scenery_drop'):
            setattr(bot, name, lambda *a: False)
        bot._map_limit = lambda direction: None
        bot._visible_drops = lambda: [(-100, 0, 0.9)]
        moves = []
        bot._set_auto_attack_movement = moves.append
        loot = SimpleNamespace(LOOT_KEY='x')
        for offsets, expected in (([-60], False), ([], True)):
            bot._drop_chase = None
            tracker = FakeTracker(offsets=offsets)
            config = SimpleNamespace(
                capture=SimpleNamespace(mob_tracker=tracker, loot_tracker=loot), enabled=True)
            with patch.object(bot_module, 'config', config):
                self.assertEqual(bot._collect_visible_drop(0.5), expected, offsets)


class SnailClipTests(unittest.TestCase):
    VIDEO = os.path.join('assets', 'knock-back-dmg.mp4')

    def _replay(self, group_skill):
        if not os.path.exists(tracker_module.MobTracker.PLAYER_TEMPLATE_PATH):
            self.skipTest('no calibrated samples')
        clock = [0.0]
        video = cv2.VideoCapture(self.VIDEO)
        result = {'scans': 0, 'swing': 0, 'walk_close': 0, 'lost': 0, 'q': 0}
        with patch.object(tracker_module, 'is_game_focused', return_value=True), \
                patch.object(tracker_module.time, 'monotonic', lambda: clock[0]), \
                patch.object(tracker_module.MobTracker, 'save_config', lambda self: None):
            tracker = tracker_module.MobTracker()
            if set(tracker.enemy_speed_profiles()) != {'normal'}:
                self.skipTest('the live samples are not the snail ones')
            tracker.config['Group skill'] = group_skill
            index = 0
            while True:
                ok, frame = video.read()
                if not ok:
                    break
                clock[0] = index / 30.0
                index += 1
                tracker.process_frame(frame[:, 1126:], True)
                if tracker._frame_count % 3 or clock[0] < 1.0:
                    continue
                result['scans'] += 1
                target = tracker.target()
                result['q'] += tracker.consume_attack_action()[1]
                if target is None:
                    result['lost'] += 1
                elif target[1]:
                    result['swing'] += 1
                elif abs(target[0]) < tracker.NORMAL_ATTACK_RANGE:
                    result['walk_close'] += 1
        video.release()
        return result

    @unittest.skipUnless(os.path.exists(VIDEO), 'clip not available')
    def test_snails_by_the_character_are_attacked_not_walked_into(self):
        """The old logic swung in ~half the scans with snails touching the
        character, walked in ~40 % and lost the target in ~8 %."""
        result = self._replay(True)
        self.assertGreater(result['scans'], 150)
        self.assertGreater(result['swing'] / result['scans'], 0.8)
        self.assertEqual(result['walk_close'], 0)
        self.assertLessEqual(result['lost'], 3)
        self.assertGreater(result['q'], 0)

    @unittest.skipUnless(os.path.exists(VIDEO), 'clip not available')
    def test_without_the_group_skill_no_q_is_asked_for(self):
        result = self._replay(False)
        self.assertEqual(result['q'], 0)
        self.assertGreater(result['swing'] / result['scans'], 0.8)


class SnailClip2Tests(unittest.TestCase):
    VIDEO = os.path.join('assets', 'knock-back-dmg-2.mp4')

    @unittest.skipUnless(os.path.exists(VIDEO), 'clip not available')
    def test_a_snail_that_crawls_into_the_character_is_never_ignored(self):
        """It was taken for scenery after missed swings and ignored for 8 s by
        a screen spot that moved with the camera; a 2% shroom hidden behind
        the character counted as killed."""
        import numpy as np
        from src.common import config
        if not os.path.exists(tracker_module.MobTracker.PLAYER_TEMPLATE_PATH):
            self.skipTest('no calibrated samples')
        clock = [0.0]
        video = cv2.VideoCapture(self.VIDEO)
        drops = blind = scans = 0
        saved = (config.player_pos, config.player_pos_time)
        try:
            with patch.object(tracker_module, 'is_game_focused', return_value=True), \
                    patch.object(tracker_module.time, 'monotonic', lambda: clock[0]), \
                    patch.object(tracker_module.MobTracker, 'save_config', lambda self: None):
                tracker = tracker_module.MobTracker()
                if set(tracker.enemy_speed_profiles()) != {'normal'}:
                    self.skipTest('the live samples are not the snail ones')
                index = 0
                while True:
                    ok, frame = video.read()
                    if not ok:
                        break
                    clock[0] = index / 30.0
                    index += 1
                    minimap = frame[60:140, 1040:1240].astype(int)
                    ys, xs = np.where((minimap[:, :, 2] > 200) & (minimap[:, :, 1] > 200)
                                      & (minimap[:, :, 0] < 90))
                    if len(xs):     # the yellow player dot, as the live bot reads it
                        config.player_pos = (xs.mean() / 200, ys.mean() / 200)
                        config.player_pos_time = clock[0]
                    tracker.process_frame(frame[:, 1038:], True)
                    if tracker._frame_count % 3:
                        continue
                    scans += 1
                    drops += 'no HP bar' in tracker.status()
                    blind += tracker.target() is None
        finally:
            config.player_pos, config.player_pos_time = saved
            video.release()
        self.assertGreater(scans, 200)
        self.assertEqual(drops, 0)
        self.assertLessEqual(blind, 30)     # was 43 (and 28 after the map-spot fix)


class SnailClip3Tests(unittest.TestCase):
    VIDEO = os.path.join('assets', 'knock-back-dmg-3.mp4')

    @unittest.skipUnless(os.path.exists(VIDEO), 'clip not available')
    def test_the_shroom_wins_over_the_far_portal_and_is_let_come(self):
        """knock-back-dmg-3.mp4: locked on the portal 500 px away while a
        shroom came from the other side; then walked into the shroom."""
        import numpy as np
        from src.common import config
        if not os.path.exists(tracker_module.MobTracker.PLAYER_TEMPLATE_PATH):
            self.skipTest('no calibrated samples')
        clock = [0.0]
        video = cv2.VideoCapture(self.VIDEO)
        saved = (config.player_pos, config.player_pos_time)
        offsets, waited_before_swing, swung, qs = [], False, False, 0
        try:
            with patch.object(tracker_module, 'is_game_focused', return_value=True), \
                    patch.object(tracker_module.time, 'monotonic', lambda: clock[0]), \
                    patch.object(tracker_module.MobTracker, 'save_config', lambda self: None):
                tracker = tracker_module.MobTracker()
                if set(tracker.enemy_speed_profiles()) != {'normal'}:
                    self.skipTest('the live samples are not the snail ones')
                index = 0
                while True:
                    ok, frame = video.read()
                    if not ok:
                        break
                    clock[0] = index / 30.0
                    index += 1
                    minimap = frame[45:125, 1078:1328].astype(int)
                    ys, xs = np.where((minimap[:, :, 2] > 200) & (minimap[:, :, 1] > 200)
                                      & (minimap[:, :, 0] < 90))
                    if len(xs):
                        config.player_pos = (xs.mean() / 250, ys.mean() / 250)
                        config.player_pos_time = clock[0]
                    tracker.process_frame(frame[:, 1078:], True)
                    if tracker._frame_count % 3:
                        continue
                    qs += tracker.consume_attack_action()[1]
                    target = tracker.target()
                    if 4.4 <= clock[0] <= 5.2 and target is not None:
                        offsets.append(target[0])
                    if 5.0 <= clock[0] <= 5.6 and target is not None:
                        if tracker.waiting() and not swung:
                            waited_before_swing = True
                        swung = swung or target[1]
        finally:
            config.player_pos, config.player_pos_time = saved
            video.release()
        self.assertTrue(offsets)
        self.assertTrue(all(offset < 0 for offset in offsets), offsets)   # the shroom
        self.assertTrue(waited_before_swing)
        self.assertTrue(swung)
        self.assertEqual(qs, 0)


class ScenerySpotTests(unittest.TestCase):
    def test_proven_scenery_is_kept_long_only_by_its_map_spot(self):
        tracker = tracker_module.MobTracker.__new__(tracker_module.MobTracker)
        tracker._ignored_spots = []
        tracker._player_position = (500, 300)
        fake = SimpleNamespace(player_pos=(0.9, 0.2), player_pos_time=10.0, bot=None)
        with patch.object(tracker_module, '_config', fake), \
                patch.object(tracker_module.time, 'monotonic', lambda: 10.0):
            tracker._ignore_spot(590, 300, 10.0, tracker.SCENERY_IGNORE_SECONDS)
            fake.player_pos = (0, 0)        # no minimap fix: screen px, briefly
            tracker._ignore_spot(590, 300, 10.0, tracker.SCENERY_IGNORE_SECONDS)
        self.assertEqual(tracker._ignored_spots[0][2], 10.0 + tracker.SCENERY_IGNORE_SECONDS)
        self.assertEqual(tracker._ignored_spots[1][2], 10.0 + tracker.IGNORE_SPOT_SECONDS)


class PortalClipTests(unittest.TestCase):
    VIDEO = os.path.join('assets', 'bug.mp4')

    @unittest.skipUnless(os.path.exists(VIDEO), 'clip not available')
    def test_swinging_left_and_right_in_the_portal_ends(self):
        """bug.mp4: standing in the portal, its matches on both sides took
        turns as the target and each switch restarted the scenery timer."""
        import numpy as np
        from src.common import config
        if not os.path.exists(tracker_module.MobTracker.PLAYER_TEMPLATE_PATH):
            self.skipTest('no calibrated samples')
        clock = [0.0]
        video = cv2.VideoCapture(self.VIDEO)
        saved = (config.player_pos, config.player_pos_time)
        dropped_at, scale = None, None
        try:
            with patch.object(tracker_module, 'is_game_focused', return_value=True), \
                    patch.object(tracker_module.time, 'monotonic', lambda: clock[0]), \
                    patch.object(tracker_module.MobTracker, 'save_config', lambda self: None):
                tracker = tracker_module.MobTracker()
                if set(tracker.enemy_speed_profiles()) != {'normal'}:
                    self.skipTest('the live samples are not the snail ones')
                index = 0
                while True:
                    ok, frame = video.read()
                    if not ok:
                        break
                    clock[0] = index / 30.0
                    index += 1
                    game = frame[0:745, :]
                    minimap = game[45:120, 0:245].astype(int)
                    ys, xs = np.where((minimap[:, :, 2] > 200) & (minimap[:, :, 1] > 200)
                                      & (minimap[:, :, 0] < 90))
                    if len(xs):
                        config.player_pos = (xs.mean() / 245, ys.mean() / 245)
                        config.player_pos_time = clock[0]
                    tracker.process_frame(game, True)
                    if dropped_at is None and tracker.status().startswith('Swung for'):
                        dropped_at = clock[0]
                scale = tracker.__dict__.get('_ppm_est')
        finally:
            config.player_pos, config.player_pos_time = saved
            video.release()
        self.assertIsNotNone(dropped_at)
        self.assertLess(dropped_at, 4.0)
        self.assertTrue(scale and 3000 <= scale <= 5000, scale)  # learnt at the edge


class HennyPortalClipTests(unittest.TestCase):
    VIDEO = os.path.join('assets', 'henny_bug-2.mp4')

    @unittest.skipUnless(os.path.exists(VIDEO), 'clip not available')
    def test_the_portal_at_the_left_edge_is_left_for_the_snails(self):
        """henny_bug-2.mp4: the portal's matches move about across its width;
        by the minimap (a dot step is ~18 px here) they missed the ignored
        spots, and the flowers by it matched at 66% for one scan now and then."""
        from src.common import config, utils
        if not os.path.exists(tracker_module.MobTracker.PLAYER_TEMPLATE_PATH):
            self.skipTest('no calibrated samples')
        dot = cv2.imread(os.path.join('assets', 'player_template.png'), 0)
        clock = [0.0]
        video = cv2.VideoCapture(self.VIDEO)
        saved = (config.player_pos, config.player_pos_time)
        near_after, scans_after, dropped_at = 0, 0, None
        try:
            with patch.object(tracker_module, 'is_game_focused', return_value=True), \
                    patch.object(tracker_module.time, 'monotonic', lambda: clock[0]), \
                    patch.object(tracker_module.MobTracker, 'save_config', lambda self: None):
                tracker = tracker_module.MobTracker()
                if set(tracker.enemy_speed_profiles()) != {'normal'}:
                    self.skipTest('the live samples are not the snail ones')
                index = 0
                while True:
                    ok, frame = video.read()
                    if not ok:
                        break
                    clock[0] = index / 30.0
                    index += 1
                    game = frame[0:755, :]
                    minimap = game[55:180, 30:175]
                    found = utils.multi_match(minimap, dot, threshold=0.8)  # as live
                    if found:
                        config.player_pos = (found[0][0] / minimap.shape[1],
                                             found[0][1] / minimap.shape[0])
                        config.player_pos_time = clock[0]
                    tracker.process_frame(game, True)
                    if tracker._frame_count % 3:
                        continue
                    if dropped_at is None and tracker.status().startswith('Swung for'):
                        dropped_at = clock[0]
                    if dropped_at is not None and clock[0] > dropped_at + 0.5:
                        scans_after += 1
                        target = tracker.target()
                        near_after += target is not None and abs(target[0]) < 150
        finally:
            config.player_pos, config.player_pos_time = saved
            video.release()
        self.assertIsNotNone(dropped_at)
        self.assertLess(dropped_at, 4.0)
        self.assertGreater(scans_after, 50)
        self.assertEqual(near_after, 0)     # never back to the portal or the flowers


class EdgePortalClipTests(unittest.TestCase):
    VIDEO = os.path.join('assets', 'edge-portal-issue.mp4')

    @unittest.skipUnless(os.path.exists(VIDEO), 'clip not available')
    def test_the_portal_and_the_mushroom_decor_are_both_dropped(self):
        """edge-portal-issue.mp4: at the right edge the bot walked between the
        portal and the decorative mushrooms, swinging at each; the walks (and
        hidden-lock swings) restarted the scenery count, so it never ended."""
        from src.common import config, utils
        if not os.path.exists(tracker_module.MobTracker.PLAYER_TEMPLATE_PATH):
            self.skipTest('no calibrated samples')
        dot = cv2.imread(os.path.join('assets', 'player_template.png'), 0)
        clock = [0.0]
        video = cv2.VideoCapture(self.VIDEO)
        saved = (config.player_pos, config.player_pos_time)
        drops = []
        try:
            with patch.object(tracker_module, 'is_game_focused', return_value=True),                     patch.object(tracker_module.time, 'monotonic', lambda: clock[0]),                     patch.object(tracker_module.MobTracker, 'save_config', lambda self: None):
                tracker = tracker_module.MobTracker()
                if set(tracker.enemy_speed_profiles()) != {'normal'}:
                    self.skipTest('the live samples are not the snail ones')
                index = 0
                while True:
                    ok, frame = video.read()
                    if not ok or clock[0] > 15.0:
                        break
                    clock[0] = index / 30.0
                    index += 1
                    game = frame[0:755, :]
                    minimap = game[55:180, 30:175]
                    found = utils.multi_match(minimap, dot, threshold=0.8)
                    if found:
                        config.player_pos = (found[0][0] / minimap.shape[1],
                                             found[0][1] / minimap.shape[0])
                        config.player_pos_time = clock[0]
                    tracker.process_frame(game, True)
                    if tracker.status().startswith('Swung for') and (
                            not drops or clock[0] - drops[-1][0] > 1.0):
                        drops.append((clock[0], tracker._player_position[0]))
        finally:
            config.player_pos, config.player_pos_time = saved
            video.release()
        # the portal (screen x ~1255) and then the mushrooms (~1070)
        self.assertTrue(any(at < 9.0 and x > 1180 for at, x in drops), drops)
        self.assertTrue(any(at < 15.0 and x < 1120 for at, x in drops), drops)


class BlankTotalTests(unittest.TestCase):
    def _tracker(self):
        return tracker_module.MobTracker.__new__(tracker_module.MobTracker)

    def test_walking_between_swings_drains_slowly_instead_of_restarting(self):
        tracker = self._tracker()
        now = 0.0
        tracker._blank_step(now, True, False)
        for _ in range(8):              # 1.6 s swinging
            now += 0.2
            tracker._blank_step(now, True, False)
        for _ in range(8):              # 1.6 s walking to the other match
            now += 0.2
            tracker._blank_step(now, False, False)
        for _ in range(7):              # 1.4 s swinging again
            now += 0.2
            total = tracker._blank_step(now, True, False)
        self.assertGreaterEqual(total, tracker.BLANK_AREA_SECONDS)

    def test_an_hp_bar_or_a_knock_back_clears_it(self):
        tracker = self._tracker()
        tracker._blank_step(0.0, True, False)
        tracker._blank_step(0.4, True, False)
        self.assertEqual(tracker._blank_step(0.6, True, True), 0.0)

    def test_a_lag_spike_adds_at_most_one_step(self):
        tracker = self._tracker()
        tracker._blank_step(0.0, True, False)
        self.assertAlmostEqual(tracker._blank_step(5.0, True, False),
                               tracker.BLANK_MAX_STEP)


class FutileBackstopTests(unittest.TestCase):
    """memory-leak.png: weak proof (a badge bump, portal animation) kept a
    scenery lock 'real' for good; only an HP bar clears the backstop."""

    def _tracker(self):
        tracker = tracker_module.MobTracker.__new__(tracker_module.MobTracker)
        tracker._ignored_spots = []
        tracker._player_position = (1200, 300)
        return tracker

    def test_swinging_with_no_bar_fills_the_backstop(self):
        tracker = self._tracker()
        now, total = 0.0, 0.0
        tracker._futile_step(now, True, False)
        for _ in range(31):
            now += 0.2
            total = tracker._futile_step(now, True, False)
        self.assertGreaterEqual(total, tracker.FUTILE_SECONDS)

    def test_only_an_hp_bar_clears_it(self):
        tracker = self._tracker()
        tracker._futile_step(0.0, True, False)
        tracker._futile_step(0.4, True, False)
        self.assertEqual(tracker._futile_step(0.6, True, True), 0.0)

    def test_the_same_scenery_caught_again_is_ignored_longer(self):
        tracker = self._tracker()
        fake = SimpleNamespace(player_pos=(0.9, 0.2), player_pos_time=10.0, bot=None)
        with patch.object(tracker_module, '_config', fake), \
                patch.object(tracker_module.time, 'monotonic', lambda: 10.0):
            tracker._ignore_spot(1300, 300, 10.0, 60.0)
            tracker._ignore_spot(1300, 300, 80.0, 60.0)
            tracker._ignore_spot(1300, 300, 200.0, 60.0)
        lengths = [spot[2] - start for spot, start in
                   zip(tracker._ignored_spots, (10.0, 80.0, 200.0))]
        self.assertEqual(lengths, [60.0, 120.0, 240.0])

    def test_a_knock_back_resets_the_longer_ignores(self):
        tracker = self._tracker()
        fake = SimpleNamespace(player_pos=(0.9, 0.2), player_pos_time=10.0, bot=None)
        with patch.object(tracker_module, '_config', fake), \
                patch.object(tracker_module.time, 'monotonic', lambda: 10.0):
            tracker._ignore_spot(1230, 300, 10.0, 60.0)
            tracker._unignore_near(1200, 20.0)
            tracker._ignore_spot(1230, 300, 30.0, 60.0)
        self.assertEqual(tracker._ignored_spots[-1][2] - 30.0, 60.0)


class MapSpotTests(unittest.TestCase):
    def _tracker(self):
        tracker = tracker_module.MobTracker.__new__(tracker_module.MobTracker)
        tracker._ignored_spots = []
        tracker._player_position = (1200, 300)
        return tracker

    def test_a_knock_back_forgets_ignored_spots_by_the_character(self):
        tracker = self._tracker()
        fake = SimpleNamespace(player_pos=(0.9, 0.2), player_pos_time=10.0, bot=None)
        with patch.object(tracker_module, '_config', fake), \
                patch.object(tracker_module.time, 'monotonic', lambda: 10.0):
            tracker._ignore_spot(1230, 300, 10.0, 60.0)     # 30 px off: by the character
            tracker._ignore_spot(1500, 300, 10.0, 60.0)     # 300 px off
            tracker._unignore_near(1200, 10.0)
        self.assertEqual([spot[0] for spot in tracker._ignored_spots], [1500])

    def test_the_map_scale_is_learnt_where_the_camera_stops(self):
        tracker = self._tracker()
        fake = SimpleNamespace(player_pos=(0.90, 0.2), player_pos_time=10.0, bot=None)
        with patch.object(tracker_module, '_config', fake):
            tracker._calibrate_map_scale(1200, 1356, 10.0)  # right of centre: clamped
            fake.player_pos = (0.92, 0.2)
            tracker._calibrate_map_scale(1282, 1356, 10.0)  # walked 82 px = 0.02 units
        self.assertAlmostEqual(tracker._ppm_est, 4100, delta=1)
        self.assertEqual(tracker._px_per_minimap(), tracker._ppm_est)


class FacingTests(unittest.TestCase):
    def test_waiting_for_a_coming_enemy_turns_to_face_it_once(self):
        bot = _bot()
        bot._facing = 'right'
        taps = []
        with patch.object(bot_module, 'key_down', lambda key: taps.append(('down', key))), \
                patch.object(bot_module, 'key_up', lambda key: taps.append(('up', key))), \
                patch.object(bot_module.time, 'sleep', lambda s: None):
            bot._face(-90)
            bot._face(-80)                  # already facing it: no second tap
        self.assertEqual(taps, [('down', 'left'), ('up', 'left')])
        self.assertEqual(bot._facing, 'left')


class StrikeFollowUpTests(unittest.TestCase):
    """strike-issue.mp4 / memory-issue.mp4: one hit landed, then the bot let go
    and walked (or turned to the snail on the other side) into a snail."""

    @staticmethod
    def _tracker():
        import threading
        tracker = tracker_module.MobTracker.__new__(tracker_module.MobTracker)
        tracker._lock = threading.Lock()
        return tracker

    @staticmethod
    def _item(x, y=500):
        import numpy as np
        sprite = np.zeros((40, 40), np.uint8)   # feet at y + 20
        return ((0.9, 0, sprite), x, y, None)

    def follow(self, tracker, attackable, delta_x, candidates, now, player_x=500):
        with tracker._lock:
            return tracker._normal_follow_up(
                attackable, False, delta_x, 'left' if delta_x < 0 else 'right',
                delta_x, [self._item(player_x + o) for o in candidates], player_x, 520, now)

    def test_keeps_swinging_the_same_way_after_a_swing(self):
        tracker = self._tracker()
        self.follow(tracker, True, 45, [45, -90], now=0.0)
        # Knocked back past the range, the snail on the left now the target
        attackable, waiting, offset, direction = self.follow(tracker, False, -90, [75, -90], 0.5)
        self.assertEqual((attackable, waiting, offset, direction), (True, False, 75, 'right'))
        # After the follow-up window it may move on
        attackable, _, offset, _ = self.follow(tracker, False, -90, [75, -90], 1.2)
        self.assertEqual((attackable, offset), (False, -90))

    def test_a_target_on_the_other_side_waits_unless_it_is_touching(self):
        tracker = self._tracker()
        self.follow(tracker, True, 45, [45, -60], now=0.0)
        attackable, _, offset, _ = self.follow(tracker, True, -60, [50, -60], 0.4)
        self.assertEqual((attackable, offset), (True, 50))     # finish the right one
        attackable, _, offset, _ = self.follow(tracker, True, -30, [50, -30], 0.6)
        self.assertEqual((attackable, offset), (True, -30))    # about to bite: turn

    def test_nothing_left_on_that_side_lets_it_go(self):
        tracker = self._tracker()
        self.follow(tracker, True, 45, [45], now=0.0)
        attackable, _, offset, _ = self.follow(tracker, False, -150, [-150], 0.3)
        self.assertEqual((attackable, offset), (False, -150))

    def test_follow_up_direction_for_the_bot(self):
        tracker = self._tracker()
        with patch.object(tracker_module.time, 'monotonic', lambda: 0.0):
            self.follow(tracker, True, -45, [-45], now=0.0)
            self.assertEqual(tracker.follow_up_direction(), 'left')
        with patch.object(tracker_module.time, 'monotonic', lambda: 2.0):
            self.assertIsNone(tracker.follow_up_direction())

    def test_bot_holds_the_swing_instead_of_walking(self):
        bot = _bot()
        pressed = []
        tracker = SimpleNamespace(follow_up_direction=lambda: 'right')
        with patch.object(bot_module, 'key_down', pressed.append),                 patch.object(bot_module, 'key_up', lambda key: None),                 patch.object(Bot, '_jumper_farming', lambda self, tracker: False):
            self.assertTrue(bot._follow_up_swing(tracker))
        self.assertEqual(bot._mob_attack_keys, ('ctrl',))
        self.assertEqual(pressed[0], 'right')     # faced right with a tap first
        self.assertIsNone(bot._auto_attack_movement)

    def test_swings_that_leave_the_hp_bar_alone_are_whiffs(self):
        tracker = self._tracker()
        whiffs = []
        with tracker._lock:
            for i in range(20):         # 2 s of swinging, bar stuck at ~70%
                now = i * 0.1
                whiffs.append(tracker._note_normal_hits(
                    True, 0.70 + (0.01 if i % 2 else 0.0), 65, True, now))
        self.assertFalse(any(whiffs[:15]))
        self.assertTrue(whiffs[-1])
        with tracker._lock:             # a hit lands: in reach after all
            self.assertFalse(tracker._note_normal_hits(True, 0.55, 65, True, 2.1))

    def test_other_side_threat_needs_to_be_touching(self):
        tracker = self._tracker()
        ahead, behind_far, behind_near = self._item(550), self._item(440), self._item(470)
        self.assertFalse(tracker._may_switch_side(ahead, behind_far, 500, 0.0))
        tracker._side_switch_at = float('-inf')
        far_ahead = self._item(580)     # out of reach in front, one touching behind
        self.assertTrue(tracker._may_switch_side(far_ahead, behind_near, 500, 0.0))
        tracker._side_switch_at = 0.0   # just turned: not straight back
        self.assertFalse(tracker._may_switch_side(far_ahead, behind_near, 500, 0.5))


if __name__ == '__main__':
    unittest.main()
