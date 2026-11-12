import unittest
import threading
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.modules import bot as bot_module
from src.modules.bot import Bot
from src.modules.mob_tracker import MobTracker


def positioned_candidate(index, x, y, width=20, height=20, score=0.9):
    template = np.zeros((height, width, 3), dtype=np.uint8)
    candidate = (score, index, template, 0, (0, 0), 0, 0)
    return candidate, x, y, None


class MobTrackingTests(unittest.TestCase):
    def test_candidate_deduplication_merges_overlapping_sample_matches(self):
        candidates = [
            positioned_candidate(0, 50, 50),
            positioned_candidate(1, 52, 50, width=22, height=20, score=0.8),
            positioned_candidate(0, 100, 50),
        ]

        deduplicated = MobTracker._deduplicate_positioned_candidates(candidates)

        self.assertEqual(len(deduplicated), 2)

    def test_candidate_deduplication_keeps_close_distinct_enemies(self):
        candidates = [
            positioned_candidate(0, 50, 50),
            positioned_candidate(1, 56, 50),
        ]

        deduplicated = MobTracker._deduplicate_positioned_candidates(candidates)

        self.assertEqual(len(deduplicated), 2)

    def test_group_requires_three_distinct_compact_enemy_candidates(self):
        candidates = [
            positioned_candidate(0, 100, 100),
            positioned_candidate(0, 130, 105),
            positioned_candidate(0, 160, 100),
        ]

        group = MobTracker._nearest_enemy_group(candidates, player_x=120)

        self.assertEqual(len(group), 3)

    def test_group_does_not_count_a_spread_out_chain(self):
        candidates = [
            positioned_candidate(0, 0, 100),
            positioned_candidate(0, 80, 100),
            positioned_candidate(0, 160, 100),
            positioned_candidate(0, 240, 100),
        ]

        group = MobTracker._nearest_enemy_group(candidates, player_x=120)

        self.assertEqual(group, [])

    def test_group_rejects_wide_cluster_even_when_it_has_a_center_anchor(self):
        candidates = [
            positioned_candidate(0, 0, 100),
            positioned_candidate(0, 60, 100),
            positioned_candidate(0, 120, 100),
        ]

        group = MobTracker._nearest_enemy_group(candidates, player_x=60)

        self.assertEqual(group, [])

    def test_sweep_candidate_prefers_the_current_scan_direction(self):
        candidates = [
            positioned_candidate(0, 80, 100),
            positioned_candidate(0, 120, 100),
            positioned_candidate(0, 180, 100),
        ]

        right = MobTracker._sweep_candidate(candidates, 100, 'right')
        left = MobTracker._sweep_candidate(candidates, 100, 'left')

        self.assertEqual(right[1], 120)
        self.assertEqual(left[1], 80)

    def test_group_skill_requires_three_enemies_in_attack_range(self):
        in_range = [
            positioned_candidate(0, 100, 100),
            positioned_candidate(0, 130, 100),
            positioned_candidate(0, 160, 100),
        ]
        only_two_in_range = [
            positioned_candidate(0, 100, 100),
            positioned_candidate(0, 130, 100),
            positioned_candidate(0, 170, 100),
        ]

        self.assertTrue(MobTracker._group_in_attack_range(
            in_range, player_x=100, player_y=100
        ))
        self.assertFalse(MobTracker._group_in_attack_range(
            only_two_in_range, player_x=100, player_y=100
        ))

    def test_fast_profile_is_persisted_per_enemy_sample(self):
        tracker = MobTracker.__new__(MobTracker)
        tracker._lock = threading.Lock()
        tracker._mob_templates = [
            (np.zeros((20, 20, 3), dtype=np.uint8), 0),
            (np.zeros((20, 20, 3), dtype=np.uint8), 0),
        ]
        tracker.config = {'Enemy speeds': 'normal,fast'}
        tracker._locked_target = (0, 10, 10)
        tracker._pending_target = None
        tracker.save_config = lambda: None

        self.assertEqual(
            tracker.enemy_speed_profiles(), ['normal', 'fast']
        )
        tracker.set_enemy_speed_profile(0, 'fast')
        self.assertEqual(
            tracker.enemy_speed_profiles(), ['fast', 'fast']
        )

    def test_health_bar_parser_captures_wide_bar_above_enemy(self):
        frame = np.full((200, 240, 3), 120, dtype=np.uint8)
        frame[82, 65:155] = 0
        frame[82, 67:112] = (0, 255, 0)

        ratio = MobTracker._health_bar_ratio(
            frame, ((100, 100), (120, 125))
        )

        self.assertIsNotNone(ratio)
        self.assertAlmostEqual(ratio, 0.5, delta=0.03)

    def test_bot_holds_command_book_attack_key_until_target_is_lost(self):
        class Tracker:
            target_value = (10, True)

            def consume_attack_action(self):
                return None, False, 0

            def target(self):
                return self.target_value

            def group_in_range(self):
                return False

        class Key:
            CRUEL_STAB = 'ctrl'

        tracker = Tracker()
        bot = Bot.__new__(Bot)
        bot._mob_attack_keys = ()
        bot._auto_attack_movement = None
        bot.command_book = SimpleNamespace(
            dict={'cruelstab': object()},
            module=SimpleNamespace(Key=Key),
        )
        events = []
        config_stub = SimpleNamespace(
            capture=SimpleNamespace(mob_tracker=tracker)
        )
        with patch.object(bot_module, 'config', config_stub), \
                patch.object(bot_module, 'is_game_focused', return_value=True), \
                patch.object(
                    bot_module, 'key_down',
                    side_effect=lambda key: events.append(('down', key))
                ), \
                patch.object(
                    bot_module, 'key_up',
                    side_effect=lambda key: events.append(('up', key))
                ):
            self.assertTrue(bot._service_mob_attack())
            self.assertTrue(bot._service_mob_attack())
            self.assertEqual(events, [('down', 'right'), ('down', 'ctrl')])

            tracker.target_value = None
            self.assertFalse(bot._service_mob_attack())

        self.assertEqual(events[-2:], [('up', 'ctrl'), ('up', 'right')])

    def test_bot_releases_attack_before_resuming_target_approach(self):
        class Tracker:
            def settings(self):
                return {'Enabled': True}

            def target(self):
                return (25, False)

        bot = Bot.__new__(Bot)
        bot._mob_attack_keys = ('left', 'ctrl')
        bot._auto_attack_movement = None
        events = []
        config_stub = SimpleNamespace(
            enabled=True,
            capture=SimpleNamespace(
                mob_tracker=Tracker(),
                loot_tracker=SimpleNamespace(consume_pickup=lambda: None),
            ),
        )
        with patch.object(bot_module, 'config', config_stub), \
                patch.object(bot_module, 'is_game_focused', return_value=True), \
                patch.object(
                    bot_module, 'key_down',
                    side_effect=lambda key: events.append(('down', key))
                ), \
                patch.object(
                    bot_module, 'key_up',
                    side_effect=lambda key: events.append(('up', key))
                ), \
                patch.object(bot_module.time, 'sleep'):
            bot._run_standalone_mob_attack()

        self.assertEqual(events, [
            ('up', 'ctrl'),
            ('up', 'left'),
            ('down', 'right'),
        ])

    def test_bot_uses_q_for_group_action_and_does_not_leave_keys_held(self):
        class Tracker:
            def consume_attack_action(self):
                return 'left', True, 3

            def target(self):
                return (-10, True)

            def group_in_range(self):
                return True

        bot = Bot.__new__(Bot)
        bot._mob_attack_keys = ()
        bot._auto_attack_movement = None
        bot.command_book = None
        events = []
        config_stub = SimpleNamespace(
            capture=SimpleNamespace(mob_tracker=Tracker())
        )
        with patch.object(bot_module, 'config', config_stub), \
                patch.object(bot_module, 'is_game_focused', return_value=True), \
                patch.object(
                    bot_module, 'key_down',
                    side_effect=lambda key: events.append(('down', key))
                ), \
                patch.object(
                    bot_module, 'key_up',
                    side_effect=lambda key: events.append(('up', key))
                ), \
                patch.object(bot_module.time, 'sleep'), \
                patch.object(
                    bot_module, 'press',
                    side_effect=lambda key, count, **kwargs:
                        events.append(('press', key, count))
                ):
            self.assertTrue(bot._service_mob_attack())

        self.assertEqual(events, [
            ('down', 'left'),
            ('up', 'left'),
            ('press', 'q', 1),
        ])


def _enemy_sample(background):
    """A round orange-and-cream 'mushroom' in a 40x44 box on BACKGROUND (BGR)."""
    import cv2
    sample = np.full((44, 40, 3), background, dtype=np.uint8)
    cv2.ellipse(sample, (20, 30), (13, 11), 0, 0, 360, (200, 235, 245), -1)   # body
    cv2.ellipse(sample, (20, 16), (17, 11), 0, 0, 360, (30, 150, 245), -1)    # cap
    cv2.circle(sample, (14, 14), 3, (60, 220, 250), -1)
    return sample


class MaskedMatchingTests(unittest.TestCase):
    def setUp(self):
        self.sample = _enemy_sample((235, 190, 120))        # sampled against sky
        self.mask = MobTracker._sample_mask(self.sample)

    def test_sample_mask_keeps_the_enemy_and_drops_the_scenery(self):
        self.assertIsNotNone(self.mask)
        self.assertEqual(self.mask[30, 20], 255)            # body
        self.assertEqual(self.mask[16, 20], 255)            # cap
        self.assertEqual(self.mask[3, 3], 0)                # sky corner

    def test_finds_enemy_on_different_scenery_but_not_a_same_shaped_cloud(self):
        import cv2
        scene = np.full((120, 300, 3), (60, 140, 70), dtype=np.uint8)    # grass, not sky
        scene[40:84, 50:90] = _enemy_sample((60, 140, 70))
        # A pale blob with the same outline and shading layout, wrong colours
        cloud = cv2.cvtColor(cv2.cvtColor(self.sample, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        scene[30:74, 200:240] = cv2.addWeighted(cloud, 0.4, np.full_like(cloud, 255), 0.6, 0)

        matches, best = MobTracker._masked_matching_locations(
            scene, self.sample, self.mask, 0.58
        )

        self.assertEqual(len(matches), 1)
        (x, y), score = matches[0]
        self.assertLessEqual(abs(x - 50), 2)
        self.assertLessEqual(abs(y - 40), 2)
        self.assertGreater(score, 0.8)

    def test_ground_line_ignores_a_sample_taken_mid_jump(self):
        standing = (np.zeros((60, 60, 3), np.uint8), -38)   # bottom edge at -8
        jumping = (np.zeros((70, 64, 3), np.uint8), -112)   # bottom edge at -77
        self.assertEqual(MobTracker._ground_offset([jumping, standing]), -8)

    def test_two_samples_matching_one_enemy_count_once(self):
        same_enemy = [
            positioned_candidate(1, 100, 100, width=64, height=60, score=0.9),
            positioned_candidate(2, 110, 102, width=64, height=60, score=0.8),
        ]
        neighbours = [
            positioned_candidate(1, 100, 100, width=64, height=60),
            positioned_candidate(1, 135, 100, width=64, height=60),
        ]
        self.assertEqual(len(MobTracker._merge_same_enemy(same_enemy)), 1)
        self.assertEqual(len(MobTracker._merge_same_enemy(neighbours)), 2)


class PatrolHomeTests(unittest.TestCase):
    def _bot(self, recovery):
        bot = Bot.__new__(Bot)
        bot._mob_attack_keys = ()
        bot._auto_attack_movement = None
        bot._patrol_direction = 'right'
        bot._home_key = None
        bot.command_book = SimpleNamespace(module=SimpleNamespace(Key=SimpleNamespace(JUMP='space')))
        bot.recovery = recovery
        return bot

    def test_routine_points_define_the_home_platform(self):
        from src.modules.recovery import LevelRecovery
        from src.routine.components import Point
        recovery = LevelRecovery(None, None, None, lambda: (0.5, 0.562))
        points = [Point(0.215, 0.562), Point(0.771, 0.562)]

        class Routine(list):
            sequence = points
        config_stub = SimpleNamespace(routine=Routine(points), player_pos=(0.5, 0.562))
        with patch.object(bot_module, 'config', config_stub):
            self._bot(recovery)._set_patrol_home()
        self.assertAlmostEqual(recovery.home_y, 0.562)
        self.assertEqual(recovery.range, (0.215, 0.771))

    def test_fallen_character_recovers_before_anything_else(self):
        calls = []
        recovery = SimpleNamespace(
            home_y=0.562, observe=lambda position: None,
            fallen=lambda position: True,
            recover=lambda: calls.append('recover'), jump_key=None,
        )
        bot = self._bot(recovery)
        config_stub = SimpleNamespace(player_pos=(0.5, 0.69))
        with patch.object(bot_module, 'config', config_stub), patch.object(bot_module, 'FALL_RECOVERY', True),                 patch.object(bot_module, 'is_game_focused', return_value=True),                 patch.object(bot_module, 'key_up'), patch.object(bot_module, 'key_down'):
            self.assertTrue(bot._recover_if_fallen())
        self.assertEqual(calls, ['recover'])
        self.assertEqual(recovery.jump_key, 'space')        # from the command book
        self.assertIsNone(bot._patrol_direction)            # patrol restarts cleanly

    def test_recovery_switched_off_does_nothing(self):
        recovery = SimpleNamespace(home_y=0.562, fallen=lambda position: True)
        with patch.object(bot_module, 'FALL_RECOVERY', False):
            self.assertFalse(self._bot(recovery)._recover_if_fallen())


if __name__ == '__main__':
    unittest.main()
