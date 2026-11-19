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
    def test_jumper_hit_feedback_waits_for_a_swing_then_recovers_on_a_hit(self):
        tracker = MobTracker.__new__(MobTracker)
        tracker._lock = threading.Lock()

        self.assertEqual(tracker._jumper_attack_reach(
            True, True, 0.8, None, 70, 10.0), (105, False))
        self.assertEqual(tracker._jumper_attack_reach(
            True, True, 0.8, None, 70, 11.19), (105, False))
        self.assertEqual(tracker._jumper_attack_reach(
            True, True, 0.8, None, 70, 11.21), (MobTracker.JUMPER_WHIFF_RANGE, True))

        self.assertEqual(tracker._jumper_attack_reach(
            True, True, 0.74, None, 70, 11.22), (105, False))

        no_bar = MobTracker.__new__(MobTracker)
        no_bar._lock = threading.Lock()
        self.assertEqual(no_bar._jumper_attack_reach(
            True, True, None, 10.0, 70, 11.21),
            (MobTracker.JUMPER_WHIFF_RANGE, True))

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
            positioned_candidate(0, 80, 100),
            positioned_candidate(0, 160, 100),
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
        only_two_in_range = [           # third enemy is beyond the skill's reach
            positioned_candidate(0, 100, 100),
            positioned_candidate(0, 130, 100),
            positioned_candidate(0, 300, 100),
        ]
        none_close_enough = [           # a pile, but the nearest is not in attack range yet
            positioned_candidate(0, 200, 100),
            positioned_candidate(0, 230, 100),
            positioned_candidate(0, 260, 100),
        ]

        self.assertTrue(MobTracker._group_in_attack_range(
            in_range, player_x=100, player_y=100
        ))
        self.assertFalse(MobTracker._group_in_attack_range(
            only_two_in_range, player_x=100, player_y=100
        ))
        self.assertFalse(MobTracker._group_in_attack_range(
            none_close_enough, player_x=100, player_y=100
        ))

    def test_group_found_inside_a_longer_row(self):
        candidates = [                  # a tight trio with a fourth enemy further along
            positioned_candidate(0, 495, 100),
            positioned_candidate(0, 537, 100),
            positioned_candidate(0, 593, 100),
            positioned_candidate(0, 660, 100),
        ]

        group = MobTracker._nearest_enemy_group(candidates, player_x=750)

        self.assertGreaterEqual(len(group), 3)

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
            # strike-issue-2.mp4: an arrow tap to face, then the attack key alone
            self.assertEqual(events, [('down', 'right'), ('up', 'right'), ('down', 'ctrl')])

            tracker.target_value = None
            self.assertFalse(bot._service_mob_attack())

        self.assertEqual(events[3:], [('up', 'ctrl')])     # the arrow is up already

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


class HuntingTests(unittest.TestCase):
    @staticmethod
    def _enemy(x, health=None):
        candidate, _, y, _ = positioned_candidate(0, x, 100)
        return candidate, x, y, health

    def test_hurt_enemy_is_chosen_over_a_nearer_healthy_one(self):
        healthy, hurt = self._enemy(160), self._enemy(260, health=0.4)
        chosen = MobTracker._choose_target([healthy, hurt], 100, 'right')
        self.assertIs(chosen, hurt)

    def test_far_hurt_enemy_does_not_beat_one_next_to_the_player(self):
        healthy, hurt = self._enemy(130), self._enemy(560, health=0.4)
        chosen = MobTracker._choose_target([healthy, hurt], 100, 'right')
        self.assertIs(chosen, healthy)

    def test_enemy_ahead_is_preferred_to_one_slightly_nearer_behind(self):
        behind, ahead = self._enemy(0), self._enemy(230)
        self.assertIs(MobTracker._choose_target([behind, ahead], 100, 'right'), ahead)
        self.assertIs(MobTracker._choose_target([behind, ahead], 100, 'left'), behind)

    @staticmethod
    def _draw_bar(image, x, y, width, health, lost_colour):
        image[y:y + 8, x:x + width] = 0
        image[y, x + 1:x + width - 1] = 255
        image[y + 7, x + 1:x + width - 1] = 255
        image[y + 1:y + 7, x + 1] = 255
        image[y + 1:y + 7, x + width - 2] = 255
        inner = width - 6
        filled = round(inner * health)
        image[y + 2:y + 6, x + 3:x + 3 + filled] = (0, 230, 20)
        image[y + 2:y + 6, x + 3 + filled:x + 3 + inner] = lost_colour

    def test_health_bars_are_found_with_their_hp(self):
        rng = np.random.default_rng(0)
        image = rng.integers(60, 200, (80, 300, 3), dtype=np.uint8)
        self._draw_bar(image, 40, 20, 52, 0.25, (0, 0, 210))       # red = HP lost
        self._draw_bar(image, 180, 30, 52, 0.75, (0, 0, 0))        # black = HP lost
        bars = sorted(MobTracker._find_health_bars(image))
        self.assertEqual(len(bars), 2)
        self.assertAlmostEqual(bars[0][0], 66, delta=3)
        self.assertAlmostEqual(bars[0][2], 0.25, delta=0.08)
        self.assertAlmostEqual(bars[1][0], 206, delta=3)
        self.assertAlmostEqual(bars[1][2], 0.75, delta=0.08)

    def test_scenery_without_bars_gives_no_health_bars(self):
        rng = np.random.default_rng(1)
        image = rng.integers(0, 255, (80, 300, 3), dtype=np.uint8)
        image[30:34, 50:200] = 255          # a white stripe is not a bar
        self.assertEqual(MobTracker._find_health_bars(image), [])

    def test_sample_is_also_matched_mirrored(self):
        tracker = MobTracker.__new__(MobTracker)
        sample = np.zeros((20, 30, 3), np.uint8)
        sample[:, :10] = 200
        mask = np.zeros((20, 30), np.uint8)
        mask[:, :12] = 255
        (as_saved, saved_mask), (mirrored, mirrored_mask) = tracker._variants_for(sample, mask)
        self.assertIs(as_saved, sample)
        self.assertTrue((mirrored[:, -10:] == 200).all())
        self.assertTrue((mirrored_mask[:, -12:] == 255).all())

    def test_tightly_boxed_sample_still_gets_a_cut_out(self):
        sample = np.full((40, 40, 3), (30, 120, 230), np.uint8)   # nothing but enemy
        mask = MobTracker._sample_mask(sample)
        self.assertIsNotNone(mask)
        self.assertEqual(mask[0, 0], 0)         # corners dropped
        self.assertEqual(mask[20, 20], 255)


class PatrolTests(unittest.TestCase):
    def _run(self, positions, step=0.25):
        bot = Bot.__new__(Bot)
        bot._patrol_direction = None
        bot._patrol_until = 0
        bot._auto_attack_movement = None
        tracker = SimpleNamespace(set_sweep_direction=lambda direction: None)
        config_stub = SimpleNamespace(
            player_pos=positions[0], enabled=True,
            capture=SimpleNamespace(mob_tracker=tracker),
        )
        directions = []
        clock = [100.0]
        with patch.object(bot_module, 'config', config_stub),                 patch.object(bot_module, 'key_down'), patch.object(bot_module, 'key_up'),                 patch.object(bot_module.time, 'monotonic', lambda: clock[0]):
            for position in positions:
                config_stub.player_pos = position
                bot._patrol_for_mob()
                directions.append(bot._patrol_direction)
                clock[0] += step
        return directions

    def test_stops_walking_when_the_minimap_position_stays_lost(self):
        # Log 09:51: capture failed (out of memory), position lost, and the bot
        # flipped A/D every 2.5 s indefinitely
        bot = Bot.__new__(Bot)
        bot._patrol_direction = None
        bot._patrol_until = 0
        bot._auto_attack_movement = None
        bot._mob_attack_keys = ()
        tracker = SimpleNamespace(set_sweep_direction=lambda direction: None)
        config_stub = SimpleNamespace(
            player_pos=(0, 0), enabled=True,
            capture=SimpleNamespace(mob_tracker=tracker))
        clock = [100.0]
        moves = []
        with patch.object(bot_module, 'config', config_stub),                 patch.object(bot_module, 'key_down'), patch.object(bot_module, 'key_up'),                 patch.object(bot_module.time, 'monotonic', lambda: clock[0]):
            for _ in range(int(Bot.BLIND_PATROL_SECONDS / 0.25) + 8):
                bot._patrol_for_mob()
                moves.append(bot._auto_attack_movement)
                clock[0] += 0.25
            self.assertIn('left', moves)            # timed turns at first
            self.assertIsNone(moves[-1])            # then it stands still
            config_stub.player_pos = (0.4, 0.5)     # position back: walks again
            bot._patrol_for_mob()
            self.assertIsNotNone(bot._auto_attack_movement)

    def test_lost_at_the_map_end_walks_away_instead_of_into_the_wall(self):
        # bug-2.mp4: the marker went on the minimap's border at the right end;
        # the 12 s patrol leg held D into the wall, then the bot stood forever
        bot = Bot.__new__(Bot)
        bot._patrol_direction = None
        bot._patrol_until = 0
        bot._auto_attack_movement = None
        bot._mob_attack_keys = ()
        tracker = SimpleNamespace(set_sweep_direction=lambda direction: None)
        clock = [100.0]
        config_stub = SimpleNamespace(
            player_pos=(0.90, 0.5), player_pos_time=100.0, enabled=True,
            capture=SimpleNamespace(mob_tracker=tracker))
        moves = []
        with patch.object(bot_module, 'config', config_stub),                 patch.object(bot_module, 'key_down'), patch.object(bot_module, 'key_up'),                 patch.object(bot_module.time, 'monotonic', lambda: clock[0]),                 patch('builtins.print', lambda *a, **k: None):
            for index in range(8):              # walking right to the end
                config_stub.player_pos = (0.90 + 0.01 * index, 0.5)
                config_stub.player_pos_time = clock[0]
                bot._patrol_for_mob()
                clock[0] += 0.25
            self.assertEqual(bot._auto_attack_movement, 'right')
            for _ in range(int(Bot.BLIND_PATROL_SECONDS / 0.25) + 40):
                bot._patrol_for_mob()           # (the fix at 0.97 goes stale)
                moves.append(bot._auto_attack_movement)
                clock[0] += 0.25
        fresh = int(Bot.POSITION_STALE_SECONDS / 0.25)    # the fix still counts
        self.assertNotIn('right', moves[fresh:])    # then never on into the wall
        self.assertEqual(moves[fresh], 'left')
        tail = moves[-40:]                      # stopped: short steps away
        self.assertIn('left', tail)
        self.assertIn(None, tail)

    def test_keeps_walking_while_the_character_moves(self):
        positions = [(0.2 + 0.01 * index, 0.5) for index in range(30)]
        self.assertEqual(set(self._run(positions)), {'right'})

    def test_turns_round_when_blocked(self):
        positions = [(0.2 + 0.01 * index, 0.5) for index in range(6)] + [(0.26, 0.5)] * 8
        directions = self._run(positions)
        self.assertEqual(directions[0], 'right')
        self.assertEqual(directions[-1], 'left')
        self.assertEqual(directions.count('left') + directions.count('right'), len(directions))

    def test_falls_back_to_timed_turns_without_a_minimap_position(self):
        directions = self._run([(0, 0)] * 24)
        self.assertIn('left', directions)
        self.assertEqual(directions[:8], ['right'] * 8)

    def _locked_tracker(self, health, x=120):
        from src.modules import mob_tracker as tracker_module
        tracker = MobTracker.__new__(MobTracker)
        tracker._lock = threading.Lock()
        tracker._locked_target = (0, x, 50)
        tracker._locked_health = health
        tracker._pending_target = None
        tracker._target_missed_scans = 0
        tracker._target_missed_since = None
        tracker._last_attack_time = 0
        tracker._group_in_range = True
        return tracker, tracker_module

    def test_vanished_target_is_not_attacked_for_the_whole_grace_period(self):
        # (50 px off: not hidden behind the character, see the overlap tests)
        tracker, tracker_module = self._locked_tracker(health=None, x=150)
        clock = [100.0]
        with patch.object(tracker_module.time, 'monotonic', lambda: clock[0]),                 patch.object(tracker_module, 'is_game_focused', return_value=True):
            self.assertTrue(tracker._hold_locked_target(100, 50, True, True))
            self.assertTrue(tracker._target_attackable)
            self.assertFalse(tracker._group_in_range)
            clock[0] += MobTracker.HIDDEN_ATTACK_SECONDS + 0.1
            self.assertTrue(tracker._hold_locked_target(100, 50, True, True))
            self.assertFalse(tracker._target_attackable)
            self.assertIsNone(tracker._attack_direction)

    def test_lost_pig_is_not_chased_and_let_go_quickly(self):
        # attack-issue-2.mp4: a pig walked through the character; the bot then
        # walked 1.5 s toward the spot where it had been
        tracker, tracker_module = self._locked_tracker(health=None)
        tracker._enemy_speed_profiles_locked = lambda: ('jumper',)
        tracker._locked_target = (0, 60, 50)     # 40 px to the left, then lost
        clock = [100.0]
        with patch.object(tracker_module.time, 'monotonic', lambda: clock[0]),                 patch.object(tracker_module, 'is_game_focused', return_value=True):
            self.assertTrue(tracker._hold_locked_target(100, 50, True, True))
            self.assertTrue(tracker._target_attackable)
            clock[0] += MobTracker.HIDDEN_ATTACK_SECONDS + 0.02
            self.assertTrue(tracker._hold_locked_target(100, 50, True, True))
            self.assertEqual(tracker._target_offset_x, 0)   # stand, don't walk there
            clock[0] = 100.0 + MobTracker.JUMPER_LOST_GRACE_SECONDS + 0.05
            self.assertFalse(tracker._hold_locked_target(100, 50, True, True))
        self.assertIsNone(tracker._locked_target)

    def test_low_health_target_that_vanishes_counts_as_killed(self):
        tracker, tracker_module = self._locked_tracker(health=0.1, x=150)
        with patch.object(tracker_module, 'is_game_focused', return_value=True):
            self.assertFalse(tracker._hold_locked_target(100, 50, True, True))
        self.assertIsNone(tracker._locked_target)
        self.assertFalse(tracker._target_attackable)

    def test_low_health_snail_hidden_behind_the_character_is_not_a_kill(self):
        # knock-back-dmg-2.mp4: a shroom at 2% walked behind the character
        tracker, tracker_module = self._locked_tracker(health=0.02, x=125)
        tracker._locked_rel = (25, 99.9)
        tracker._locked_closing = 100.0      # walking at (through) the character
        clock = [100.0]
        with patch.object(tracker_module.time, 'monotonic', lambda: clock[0]),                 patch.object(tracker_module, 'is_game_focused', return_value=True):
            self.assertTrue(tracker._hold_locked_target(100, 50, True, True))
            self.assertTrue(tracker._target_attackable)
            self.assertGreater(tracker._target_offset_x, 0)
            clock[0] += 0.5                  # it has come out on the other side
            self.assertTrue(tracker._hold_locked_target(100, 50, True, True))
            self.assertTrue(tracker._target_attackable)
            self.assertLess(tracker._target_offset_x, 0)
            clock[0] = 100.0 + MobTracker.NORMAL_HIDDEN_SECONDS + 0.1
            self.assertFalse(tracker._hold_locked_target(100, 50, True, True))

    def test_symmetric_sample_is_not_matched_twice(self):
        tracker = MobTracker.__new__(MobTracker)
        sample = np.full((20, 30, 3), 120, np.uint8)
        self.assertEqual(len(tracker._variants_for(sample, None)), 1)

    def test_enemy_attacked_without_its_hp_bar_showing_is_dropped(self):
        tracker = MobTracker.__new__(MobTracker)
        tracker._in_range_since = None
        self.assertFalse(tracker._takes_no_damage(True, None, 10.0))
        self.assertFalse(tracker._takes_no_damage(True, None, 11.0))
        self.assertTrue(tracker._takes_no_damage(
            True, None, 10.0 + MobTracker.NO_DAMAGE_SECONDS))

    def test_ignored_scenery_spot_stays_on_the_map_not_by_the_character(self):
        # knock-back-dmg-2.mp4: kept in screen px, the ignored spot moved with
        # the camera, so the enemy biting from that side was ignored for 8 s
        from src.modules import mob_tracker as tracker_module
        tracker = MobTracker.__new__(MobTracker)
        tracker._ignored_spots = []
        tracker._player_position = (500, 300)
        fake = SimpleNamespace(player_pos=(0.40, 0.2), player_pos_time=10.0, bot=None)
        with patch.object(tracker_module, '_config', fake),                 patch.object(tracker_module.time, 'monotonic', lambda: 10.0):
            tracker._ignore_spot(485, 300, 10.0)
            item = (None, 485, 300, None)
            self.assertTrue(tracker._on_ignored_spot(item, tracker._ignored_spots, 500))
            fake.player_pos = (0.45, 0.2)    # walked 300 px on: same screen spot,
            self.assertFalse(                # another place on the map
                tracker._on_ignored_spot(item, tracker._ignored_spots, 500))

    def test_hp_bar_or_leaving_range_restarts_the_no_damage_count(self):
        tracker = MobTracker.__new__(MobTracker)
        tracker._in_range_since = None
        tracker._takes_no_damage(True, None, 10.0)
        self.assertFalse(tracker._takes_no_damage(True, 0.6, 11.0))     # bar seen
        self.assertFalse(tracker._takes_no_damage(True, None, 12.0))
        self.assertFalse(tracker._takes_no_damage(True, None, 13.0))
        self.assertFalse(tracker._takes_no_damage(False, None, 14.0))   # walked off
        self.assertFalse(tracker._takes_no_damage(True, None, 15.0))
        self.assertTrue(tracker._takes_no_damage(
            True, None, 15.0 + MobTracker.NO_DAMAGE_SECONDS))

    def test_oval_fallback_mask_is_recognised_as_loose(self):
        tight = np.full((40, 40, 3), (30, 120, 230), np.uint8)
        self.assertTrue(MobTracker._is_loose_mask(MobTracker._sample_mask(tight)))
        self.assertFalse(MobTracker._is_loose_mask(None))
        cut_out = np.zeros((40, 40), np.uint8)
        cut_out[10:30, 10:30] = 255
        self.assertFalse(MobTracker._is_loose_mask(cut_out))

    def test_sample_mask_is_the_same_every_time(self):
        rng = np.random.default_rng(1)
        sample = rng.integers(0, 255, (36, 39, 3), dtype=np.uint8)
        sample[8:28, 8:30] = (200, 60, 20)
        first = MobTracker._sample_mask(sample)
        for _ in range(3):
            self.assertTrue(np.array_equal(first, MobTracker._sample_mask(sample)))


class PigBeachTests(unittest.TestCase):
    """Jumping enemies, the raised step of the ground floor, and profiles."""

    GROUND_Y = 500      # feet of an enemy standing on the player's level

    def _tracker(self):
        tracker = MobTracker.__new__(MobTracker)
        tracker._lock = threading.Lock()
        return tracker

    def _enemy(self, x, rise, height=40):
        # centre y for an enemy whose feet are RISE px above the ground line
        return positioned_candidate(0, x, self.GROUND_Y - rise - height / 2, height=height)

    def test_enemy_standing_on_a_raised_step_is_out_of_reach(self):
        tracker = self._tracker()
        found = []
        for step in range(8):                       # 0.8 s, walking along the step
            found = tracker._other_level_items(
                [self._enemy(900 + 8 * step, 65)], self.GROUND_Y, 10.0 + 0.1 * step
            )
        self.assertEqual(len(found), 1)

    def test_hopping_enemy_is_not_mistaken_for_one_on_a_step(self):
        tracker = self._tracker()
        # A hop: up to ~85 px and back down in 0.7 s, higher than the step
        arc = [10, 35, 60, 78, 85, 80, 62, 40, 15, 0]
        for step, rise in enumerate(arc):
            found = tracker._other_level_items(
                [self._enemy(400 + 9 * step, rise)], self.GROUND_Y, 10.0 + 0.07 * step
            )
            self.assertEqual(found, [], f'hop frame {step} counted as a step')

    def test_enemy_on_the_ground_is_never_a_step_enemy(self):
        tracker = self._tracker()
        for step in range(10):
            self.assertEqual(tracker._other_level_items(
                [self._enemy(300, 0)], self.GROUND_Y, 10.0 + 0.1 * step), [])

    def test_raised_enemy_is_chosen_after_one_on_the_ground(self):
        ground = self._enemy(700, 0)
        raised = self._enemy(560, 65)               # nearer, but up on the step
        chosen = MobTracker._choose_target([raised, ground], 500, 'right', self.GROUND_Y)
        self.assertIs(chosen, ground)

    def test_jumper_profile_is_accepted_and_saved(self):
        tracker = self._tracker()
        tracker._mob_templates = [(np.zeros((20, 20, 3), np.uint8), 0)] * 2
        tracker.config = {'Enemy speeds': 'normal,normal'}
        tracker._locked_target = None
        tracker._pending_target = None
        tracker.save_config = lambda: None
        tracker.set_enemy_speed_profile(1, 'jumper')
        self.assertEqual(tracker.enemy_speed_profiles(), ['normal', 'jumper'])
        self.assertEqual(tracker.config['Enemy speeds'], 'normal,jumper')
        with self.assertRaises(ValueError):
            tracker.set_enemy_speed_profile(0, 'teleports')

    def test_blocking_a_side_drops_a_target_past_the_wall(self):
        tracker = self._tracker()
        tracker._player_position = (500, 300)
        tracker._locked_target = (0, 700, 300)      # 200 px right, past the step
        tracker.__dict__.update(
            _pending_target=None, _target_missed_scans=0, _target_missed_since=None,
            _target_offset_x=200, _target_vertical_offset=0, _target_attackable=False,
            _locked_health=None, _attack_direction=None, _attack_is_group=False,
        )
        tracker.set_blocked_side('left')            # other side: target kept
        self.assertIsNotNone(tracker._locked_target)
        tracker.set_blocked_side('right')
        self.assertIsNone(tracker._locked_target)
        self.assertIsNone(tracker.target())

    def test_profiles_save_and_load_samples_and_settings(self):
        import os
        import tempfile
        import cv2
        with tempfile.TemporaryDirectory() as root:
            settings = os.path.join(root, '.settings')
            os.makedirs(settings)
            paths = {
                'TEMPLATE_DIRECTORY': settings,
                'PLAYER_TEMPLATE_PATH': os.path.join(settings, 'mob_player_badge.png'),
                'MOB_TEMPLATE_PATH': os.path.join(settings, 'mob_target.png'),
                'PROFILE_DIRECTORY': os.path.join(settings, 'profiles'),
            }
            with patch.multiple(MobTracker, **paths):
                tracker = self._tracker()
                tracker.config = dict(MobTracker.DEFAULT_CONFIG, **{
                    'Enabled': True, 'Enemy count': 2, 'Enemy alignments': '-30,-28',
                    'Enemy speeds': 'normal,normal', 'Focused enemies': '1',
                })
                tracker.save_config = lambda: None
                badge = np.full((20, 50), 90, np.uint8)
                snail = np.full((30, 30, 3), (200, 40, 40), np.uint8)
                spore = np.full((30, 30, 3), (20, 120, 240), np.uint8)
                cv2.imwrite(paths['PLAYER_TEMPLATE_PATH'], badge)
                cv2.imwrite(paths['MOB_TEMPLATE_PATH'], snail)
                cv2.imwrite(os.path.join(settings, 'mob_target_1.png'), spore)
                tracker._player_template = badge
                tracker._mob_templates = [(snail, -30), (spore, -28)]
                tracker.save_profile('snail-spore')
                self.assertEqual(MobTracker.profile_names(), ['snail-spore'])

                # Recalibrate for pigs (one sample, jumper), then go back
                pig = np.full((40, 60, 3), (150, 160, 240), np.uint8)
                cv2.imwrite(paths['MOB_TEMPLATE_PATH'], pig)
                tracker._mob_templates = [(pig, -25)]
                tracker.config.update({'Enemy count': 1, 'Enemy alignments': '-25',
                                       'Enemy speeds': 'jumper', 'Enabled': False})
                pig_badge = np.full((20, 50), 120, np.uint8)
                cv2.imwrite(paths['PLAYER_TEMPLATE_PATH'], pig_badge)
                tracker._player_template = pig_badge
                tracker.save_profile('pig-beach')
                tracker.load_profile('snail-spore', preserve_player_badge=True)

                self.assertEqual(tracker.config['Enemy count'], 2)
                self.assertEqual(tracker.config['Enemy speeds'], 'normal,normal')
                self.assertFalse(tracker.config['Enabled'])     # loading never enables
                self.assertEqual(len(tracker._mob_templates), 2)
                self.assertTrue(np.array_equal(tracker._mob_templates[0][0], snail))
                self.assertTrue(np.array_equal(tracker._player_template, pig_badge))
                # what was active before loading was kept in the autosave
                self.assertIn('_autosave', MobTracker.profile_names())
                current_badge = np.full((20, 50), 150, np.uint8)
                cv2.imwrite(paths['PLAYER_TEMPLATE_PATH'], current_badge)
                tracker._player_template = current_badge
                tracker.load_profile('pig-beach')
                self.assertEqual(tracker.enemy_speed_profiles(), ['jumper'])
                self.assertTrue(np.array_equal(tracker._player_template, pig_badge))
                with self.assertRaises(ValueError):
                    tracker.save_profile('../escape')


class WallTests(unittest.TestCase):
    """Turning back at walls and raised steps instead of pushing into them."""

    def _bot(self):
        bot = Bot.__new__(Bot)
        bot._patrol_direction = 'right'
        bot._patrol_until = 0
        bot._auto_attack_movement = None
        bot._mob_attack_keys = ()
        return bot

    def _drive(self, bot, ticks, target, positions, step=0.1):
        """Runs the standalone loop; TARGET(i) gives tracker.target() per tick."""
        blocked = []
        tracker = SimpleNamespace(
            settings=lambda: {'Enabled': True},
            set_sweep_direction=lambda direction: None,
            set_blocked_side=blocked.append,
            consume_attack_action=lambda: (None, False, 0),
            group_in_range=lambda: False,
            waiting=lambda: False,
        )
        config_stub = SimpleNamespace(
            player_pos=positions(0), enabled=True,
            capture=SimpleNamespace(
                mob_tracker=tracker,
                loot_tracker=SimpleNamespace(consume_pickup=lambda: None),
            ),
            routine=SimpleNamespace(sequence=[], __len__=lambda: 0),
        )
        held = []
        clock = [100.0]
        with patch.object(bot_module, 'config', config_stub), \
                patch.object(bot_module, 'key_down'), patch.object(bot_module, 'key_up'), \
                patch.object(bot_module, 'is_game_focused', return_value=True), \
                patch.object(bot_module, 'FALL_RECOVERY', False), \
                patch.object(bot_module.time, 'sleep'), \
                patch.object(bot_module.time, 'monotonic', lambda: clock[0]):
            bot._set_patrol_home = lambda: None
            for tick in range(ticks):
                config_stub.player_pos = positions(tick)
                tracker.target = lambda tick=tick: target(tick)
                bot._run_standalone_mob_attack()
                held.append(bot._auto_attack_movement)
                clock[0] += step
        return held, blocked

    def test_chasing_into_a_step_turns_back_and_remembers_it(self):
        bot = self._bot()
        # An enemy up on the step to the right; the character walks until the step
        # stops it at x=0.60 and stays there
        positions = lambda tick: (min(0.5 + 0.02 * tick, 0.60), 0.7)
        held, blocked = self._drive(
            bot, 25, target=lambda tick: (150, False), positions=positions)
        self.assertEqual(held[0], 'right')
        self.assertIn('left', held)                 # turned back
        self.assertEqual(blocked[0], 'right')       # enemies past the step dropped
        self.assertAlmostEqual(bot._walls['right'], 0.60)
        stuck_ticks = sum(
            1 for tick, h in enumerate(held) if h == 'right' and positions(tick)[0] >= 0.60
        )
        self.assertLessEqual(stuck_ticks, 10)       # under a second at the step

    def test_known_step_turns_back_at_once(self):
        bot = self._bot()
        bot._walls = {'right': 0.60}
        held, _ = self._drive(
            bot, 3, target=lambda tick: (150, False), positions=lambda tick: (0.598, 0.7))
        self.assertEqual(held[0], 'left')

    def test_attackable_target_at_wall_uses_stationary_attack(self):
        bot = self._bot()
        bot._walls = {'right': 0.60}
        tracker = SimpleNamespace(
            settings=lambda: {'Enabled': True},
            target=lambda: (70, True),
            waiting=lambda: False,
        )
        stub = SimpleNamespace(
            enabled=True,
            capture=SimpleNamespace(mob_tracker=tracker),
        )
        attack_modes, movements = [], []
        bot._minimap_x = lambda: 0.598
        bot._service_map_events = lambda: False
        bot._set_patrol_home = lambda: None
        bot._recover_if_fallen = lambda: False
        bot._service_loot_pickup = lambda: None
        bot._check_walls = lambda x: None
        bot._count_empty_laps = lambda tracker, target: None
        bot._service_smart_loot = lambda *args: None
        bot._service_levels = lambda *args: False
        bot._service_missile = lambda *args: False
        bot._service_mob_attack = lambda **kwargs: attack_modes.append(kwargs)
        bot._set_auto_attack_movement = movements.append

        with patch.object(bot_module, 'config', stub), \
                patch.object(bot_module, 'is_game_focused', return_value=True), \
                patch.object(bot_module.time, 'sleep'):
            bot._run_standalone_mob_attack()

        self.assertEqual(attack_modes, [{'stationary': True}])
        self.assertEqual(movements, [None])

    def test_flat_platform_still_chases_across_it(self):
        bot = self._bot()
        held, blocked = self._drive(
            bot, 30, target=lambda tick: (150, False),
            positions=lambda tick: (0.1 + 0.02 * tick, 0.7))
        self.assertEqual(set(held), {'right'})
        self.assertEqual(blocked, [])
        self.assertEqual(bot.__dict__.get('_walls', {}), {})

    def test_patrol_turns_at_a_wall_it_learnt(self):
        bot = self._bot()
        bot._walls = {'left': 0.20}
        held, _ = self._drive(
            bot, 3, target=lambda tick: None, positions=lambda tick: (0.205, 0.7))
        bot._patrol_direction = 'left'
        bot._patrol_seen = 0
        held, _ = self._drive(
            bot, 2, target=lambda tick: None, positions=lambda tick: (0.205, 0.7))
        self.assertEqual(held[-1], 'right')

    def test_a_wall_the_character_gets_past_is_forgotten(self):
        bot = self._bot()
        bot._walls = {'right': 0.40}
        self._drive(bot, 2, target=lambda tick: None, positions=lambda tick: (0.5, 0.7))
        self.assertNotIn('right', bot._walls)

    def test_walking_back_from_the_wall_lets_enemies_past_it_count_again(self):
        bot = self._bot()
        positions = lambda tick: (min(0.5 + 0.02 * tick, 0.60), 0.7) if tick < 20 \
            else (0.60 - 0.01 * (tick - 20), 0.7)
        held, blocked = self._drive(
            bot, 30, target=lambda tick: (150, False) if tick < 20 else None,
            positions=positions)
        self.assertEqual(blocked[0], 'right')
        self.assertIsNone(blocked[-1])

    def test_frozen_position_at_the_step_does_not_flip_every_tick(self):
        # The live failure: the game lagged at the step, the minimap position froze,
        # and walls ended up on both sides of one spot -> A, D, A, D every tick
        bot = self._bot()
        bot._walls = {'right': 0.60, 'left': 0.598}
        held, _ = self._drive(
            bot, 40, target=lambda tick: None, positions=lambda tick: (0.60, 0.7))
        swaps = sum(1 for a, b in zip(held, held[1:]) if a and b and a != b)
        self.assertLessEqual(swaps, 5)          # was 39: one swap per tick

    def test_a_wall_learnt_next_to_the_opposite_one_replaces_it(self):
        bot = self._bot()
        bot._walls = {'right': 0.60}
        bot._learn_wall('left', 0.59)
        self.assertEqual(bot._walls, {'left': 0.59})
        bot._learn_wall('right', 0.95)          # far apart: both kept
        self.assertEqual(bot._walls, {'left': 0.59, 'right': 0.95})

    def test_stale_minimap_position_learns_no_walls(self):
        bot = self._bot()
        stub = SimpleNamespace(player_pos=(0.6, 0.7), player_pos_time=10.0)
        with patch.object(bot_module, 'config', stub), \
                patch.object(bot_module.time, 'monotonic', lambda: 10.2):
            self.assertEqual(bot._minimap_x(), 0.6)
        with patch.object(bot_module, 'config', stub), \
                patch.object(bot_module.time, 'monotonic', lambda: 11.0):
            self.assertIsNone(bot._minimap_x())

    def test_watchdog_forgets_walls_after_rapid_swaps(self):
        bot = self._bot()
        bot._walls = {'right': 0.6, 'left': 0.3}
        stub = SimpleNamespace(enabled=True, capture=SimpleNamespace(
            mob_tracker=SimpleNamespace(set_blocked_side=lambda side: None)))
        clock = [50.0]
        with patch.object(bot_module, 'config', stub), \
                patch.object(bot_module, 'key_down'), patch.object(bot_module, 'key_up'), \
                patch.object(bot_module.time, 'monotonic', lambda: clock[0]):
            for tick in range(12):
                bot._set_auto_attack_movement('left' if tick % 2 else 'right')
                clock[0] += 0.05
        self.assertEqual(bot._walls, {})

    def test_target_right_above_does_not_make_it_step_back_and_forth(self):
        bot = self._bot()
        held, _ = self._drive(
            bot, 10, target=lambda tick: (6 if tick % 2 else -6, False),
            positions=lambda tick: (0.5, 0.7))
        self.assertEqual(set(held), {None})

    def test_switching_off_forgets_the_walls(self):
        bot = self._bot()
        bot._walls = {'right': 0.6}
        bot._blocked_side = 'right'
        calls = []
        stub = SimpleNamespace(capture=SimpleNamespace(
            mob_tracker=SimpleNamespace(set_blocked_side=calls.append)))
        with patch.object(bot_module, 'config', stub):
            bot._forget_walls()
        self.assertEqual(bot._walls, {})
        self.assertEqual(calls, [None])


if __name__ == '__main__':
    unittest.main()


class PigGroupAndLureTests(unittest.TestCase):
    """Q on piles of two pigs, and luring pigs down off the Pig Beach step."""

    def test_two_pigs_make_a_group_with_the_jumper_minimum(self):
        pair = [positioned_candidate(0, 100, 100), positioned_candidate(0, 140, 100)]
        self.assertEqual(MobTracker._nearest_enemy_group(pair, player_x=100), [])
        group = MobTracker._nearest_enemy_group(
            pair, player_x=100, min_enemies=MobTracker.JUMPER_GROUP_MIN_ENEMIES)
        self.assertEqual(len(group), 2)

    def test_a_pig_mid_hop_still_counts_for_q(self):
        # One on the ground, one ~90 px up mid-hop above it, nearest 80 px away
        pile = [positioned_candidate(0, 180, 100), positioned_candidate(0, 200, 10)]
        group = MobTracker._nearest_enemy_group(
            pile, player_x=100, min_enemies=2,
            vertical_radius=MobTracker.JUMPER_GROUP_VERTICAL_RADIUS)
        self.assertEqual(len(group), 2)
        self.assertFalse(MobTracker._group_in_attack_range(
            group, player_x=100, player_y=100, min_enemies=2))
        self.assertTrue(MobTracker._group_in_attack_range(
            group, player_x=100, player_y=100, min_enemies=2,
            attack_range=MobTracker.JUMPER_ATTACK_RANGE,
            vertical_tolerance=MobTracker.JUMP_HEIGHT))

    def test_two_pigs_closer_than_a_sample_match_twice_with_jumper_separation(self):
        rng = np.random.default_rng(3)
        template = rng.integers(0, 255, (30, 40, 3), dtype=np.uint8)
        search = np.zeros((40, 140, 3), dtype=np.uint8)
        search[5:35, 20:60] = template
        search[5:35, 46:86] = template      # 26 px apart: overlapping
        search[5:35, 20:46] = template[:, :26]
        single, _ = MobTracker._matching_locations(search, template, 0.6)
        double, _ = MobTracker._matching_locations(
            search, template, 0.6, MobTracker.JUMPER_MATCH_SEPARATION)
        self.assertEqual(len(single), 1)
        self.assertEqual(len(double), 2)

    def _bot(self):
        bot = Bot.__new__(Bot)
        bot._patrol_direction = 'left'
        bot._patrol_until = 0
        bot._auto_attack_movement = None
        bot._mob_attack_keys = ()
        bot._walls = {'right': 0.60}
        bot.command_book = SimpleNamespace(
            dict={'cruelstab': object()},
            module=SimpleNamespace(Key=SimpleNamespace(JUMP='alt', CRUEL_STAB='a')))
        return bot

    # ---- Two-level farming, against a small simulated Pig Beach ----

    class _Beach:
        """
        The ground floor up to x=0.60 (minimap y 0.70) and the raised step past
        it (y 0.66). Pigs on the character's level are in reach and die one per
        KILL_TICKS of standing among them. Jumping while pressed at the step's
        foot lands up top at x 0.63, unless CLIMBS is False; KNOCKBACKS lists
        jump numbers whose landing gets knocked back off the edge 3 ticks later.
        """
        KILL_TICKS = 8

        def __init__(self, ground=0, step=0, climbs=True, knockbacks=(),
                     landing_offset=300, start=0.45):
            self.pigs = {'ground': ground, 'step': step}
            self.level, self.x = 'ground', start
            self.climbs, self.knockbacks = climbs, set(knockbacks)
            self.landing_offset = landing_offset
            self.jumps, self.knock_at, self.fight_ticks = 0, None, 0
            self.last_pig_tick, self.tick = None, 0
            self.events = []            # (tick, 'up'/'down'/'knocked')
            self.walking = None

        @property
        def y(self):
            return 0.66 if self.level == 'step' else 0.70

        def advance(self, tick, pressed):
            self.tick = tick
            jumps = pressed.count('alt')
            if jumps > self.jumps:
                self.jumps = jumps
                if self.climbs and self.level == 'ground' and self.x >= 0.595:
                    self.level, self.x = 'step', 0.63
                    self.events.append((tick, 'up'))
                    if jumps in self.knockbacks:
                        self.knock_at = tick + 9    # lands, then a pig's touch
            elif self.knock_at is not None and tick >= self.knock_at:
                self.knock_at = None
                self.level, self.x = 'ground', 0.57
                self.events.append((tick, 'knocked'))
            elif self.walking == 'right':
                self.x = min(self.x + 0.02, 0.60 if self.level == 'ground' else 0.90)
            elif self.walking == 'left':
                self.x = max(self.x - 0.02, 0.05)
                if self.level == 'step' and self.x < 0.60:
                    self.level = 'ground'
                    self.events.append((tick, 'down'))
            here = self.pigs[self.level]
            if here:
                self.last_pig_tick = tick
                self.fight_ticks += 1
                if self.fight_ticks % self.KILL_TICKS == 0:
                    self.pigs[self.level] -= 1
            return (round(self.x, 4), self.y)

        # tracker views
        def target(self):
            return (40, True) if self.pigs[self.level] else None

        def ground_clear(self):
            if self.last_pig_tick is None:
                return float('inf')
            return (self.tick - self.last_pig_tick) * 0.1

        def ledge(self):
            n = self.pigs['step'] if self.level == 'ground' else 0
            return (n, self.landing_offset) if n else (0, None)

        def lower(self):
            n = self.pigs['ground'] if self.level == 'step' else 0
            return (n, -250) if n else (0, None)

    def _farm(self, beach, ticks, bot=None, enabled=True, on_tick=None):
        bot = bot or self._bot()
        tracker = SimpleNamespace(
            settings=lambda: {'Enabled': True},
            set_sweep_direction=lambda direction: None,
            set_blocked_side=lambda side: None,
            consume_attack_action=lambda: (None, False, 0),
            group_in_range=lambda: False,
            waiting=lambda: False,
            target=beach.target,
            ledge_enemies=beach.ledge,
            lower_enemies=beach.lower,
            ground_clear_seconds=beach.ground_clear,
            level_farming_enabled=lambda: enabled,
        )
        config_stub = SimpleNamespace(
            player_pos=None, player_pos_time=None, enabled=True,
            capture=SimpleNamespace(
                mob_tracker=tracker,
                loot_tracker=SimpleNamespace(consume_pickup=lambda: None),
            ),
            routine=SimpleNamespace(sequence=[], __len__=lambda: 0),
        )
        pressed, self.levels = [], []
        self._live_pressed = pressed
        clock = [100.0]
        with patch.object(bot_module, 'config', config_stub), \
                patch.object(bot_module, 'key_down'), patch.object(bot_module, 'key_up'), \
                patch.object(bot_module, 'press', lambda key, *a, **k: pressed.append(key)), \
                patch.object(bot_module, 'is_game_focused', return_value=True), \
                patch.object(bot_module, 'FALL_RECOVERY', False), \
                patch.object(bot_module.time, 'sleep'), \
                patch.object(bot_module.time, 'monotonic', lambda: clock[0]):
            bot._set_patrol_home = lambda: None
            for tick in range(ticks):
                beach.walking = bot._auto_attack_movement
                config_stub.player_pos = beach.advance(tick, pressed)
                if on_tick:
                    on_tick(tick, beach, bot)
                bot._run_standalone_mob_attack()
                self.levels.append(bot.__dict__.get('_level', 'ground'))
                clock[0] += 0.1
        return bot, pressed

    def test_pigs_only_on_the_step_get_farmed_up_there(self):
        # pig_beach_training_spawning_2: ground empty, pigs up on the step
        # (fewer than AGGRO_MIN_PIGS: farmed up there; more get gathered down)
        beach = self._Beach(step=2)
        bot, pressed = self._farm(beach, 120)
        self.assertEqual(pressed.count('alt'), 1)
        self.assertEqual(beach.events[0][1], 'up')
        self.assertEqual(beach.pigs['step'], 0)         # all farmed, none lured
        self.assertEqual(bot._level, 'step')
        self.assertEqual(bot._step['edge_x'], 0.60)

    def test_back_down_once_the_step_is_clear_and_pigs_are_on_the_ground(self):
        def respawn(tick, beach, bot):
            if beach.level == 'step' and beach.pigs['step'] == 0 \
                    and not getattr(beach, 'respawned', False):
                beach.respawned = True
                beach.pigs['ground'] = 3
        beach = self._Beach(step=2)
        bot, _ = self._farm(beach, 260, on_tick=respawn)
        self.assertEqual([e for _, e in beach.events][:2], ['up', 'down'])
        self.assertEqual(beach.pigs, {'ground': 0, 'step': 0})
        self.assertEqual(bot._level, 'ground')
        # the step's foot is a wall again for chasing on the ground
        self.assertEqual(bot._walls.get('right'), 0.60)

    def test_pig_on_the_landing_spot_is_waited_for_then_swung_at(self):
        # pig_beach_training_spawning_3: a pig right at the edge
        beach = self._Beach(step=3, landing_offset=60)
        _, pressed = self._farm(beach, 60)
        jump = pressed.index('alt')
        self.assertEqual(pressed[jump + 1], 'a')        # swing on the way up
        self.assertEqual(beach.events[0][1], 'up')

    def test_knocked_back_off_the_edge_is_a_miss_and_retried(self):
        # pig_beach_training_spawning_3: landed, then a pig's touch threw the
        # character back down; the old lure counted that as "up" and left
        beach = self._Beach(step=3, knockbacks={1})
        bot, pressed = self._farm(beach, 100)
        self.assertEqual([e for _, e in beach.events][:3], ['up', 'knocked', 'up'])
        self.assertEqual(pressed.count('alt'), 2)
        self.assertEqual(bot._level, 'step')
        self.assertEqual(beach.pigs['step'], 0)

    def test_knocked_back_every_time_still_ends_in_the_ban(self):
        # Each knock-back restarts the climb; the misses must still add up
        beach = self._Beach(step=60, knockbacks=set(range(1, 20)))
        beach.KILL_TICKS = 10 ** 6      # the pig at the edge never dies
        bot, pressed = self._farm(beach, 150)
        self.assertEqual(pressed.count('alt'), Bot.LEVEL_CLIMB_ATTEMPTS)
        self.assertGreater(bot._climb_ban_until, 100.0)

    def test_climb_that_never_lands_is_banned_and_the_ground_still_farmed(self):
        beach = self._Beach(step=3, climbs=False)
        bot, pressed = self._farm(beach, 100)
        self.assertEqual(pressed.count('alt'), Bot.LEVEL_CLIMB_ATTEMPTS)
        self.assertGreater(bot._climb_ban_until, 100.0)
        self.assertEqual(bot._walls.get('right'), 0.60)
        self.assertEqual(set(self.levels), {'ground'})

    def test_a_pig_coming_onto_the_ground_stops_the_climb(self):
        # pig_beach_training_spawning: pigs came down while the bot walked over
        def arrive(tick, beach, bot):
            if tick == 18:
                beach.pigs['ground'] = 1
        beach = self._Beach(step=3, start=0.30)
        bot, pressed = self._farm(beach, 20, on_tick=arrive)
        self.assertNotIn('alt', pressed)
        self.assertNotIn('_level_op', bot.__dict__)

    def test_knocked_off_while_farming_up_there_switches_to_the_ground(self):
        def knock(tick, beach, bot):
            if bot.__dict__.get('_level') == 'step' and not getattr(beach, 'hit', False):
                beach.hit = True
                beach.level, beach.x = 'ground', 0.55
        beach = self._Beach(step=2)
        bot, _ = self._farm(beach, 40, on_tick=knock)
        self.assertIn('ground', self.levels[self.levels.index('step'):])
        self.assertEqual(bot._walls.get('right'), 0.60)

    def test_many_pigs_up_there_get_hit_with_q_and_led_down(self):
        # spawn_issue.mp4: the ground clear, a crowd of pigs up on the step
        def aggro(tick, beach, bot):
            if beach.level == 'step' and self._pressed.count('q') > beach.__dict__.get('q', 0):
                beach.q = self._pressed.count('q')
                beach.angry = True
            if beach.level == 'ground' and getattr(beach, 'angry', False) and beach.pigs['step']:
                beach.pigs['ground'] += beach.pigs['step']      # hit pigs follow
                beach.pigs['step'] = 0
        beach = self._Beach(step=5, start=0.30)
        self._pressed = None
        original = self._farm

        def farm(*args, **kwargs):
            return original(*args, **kwargs)
        bot = self._bot()
        recorded = []
        with patch.object(bot_module, 'press', lambda key, *a, **k: recorded.append(key)):
            pass
        # run the farm with access to the pressed list
        bot, pressed = self._farm(beach, 200, bot=bot, on_tick=lambda t, b, bo: (
            setattr(self, '_pressed', self._pressed_now(bo)), aggro(t, b, bo)))
        kinds = [event for _, event in beach.events]
        self.assertEqual(kinds[:2], ['up', 'down'])
        self.assertEqual(pressed.count('q'), Bot.AGGRO_Q_PRESSES)
        self.assertEqual(beach.pigs, {'ground': 0, 'step': 0})   # all finished below
        self.assertEqual(bot._level, 'ground')

    def test_knocked_off_mid_gather_farms_the_ground(self):
        def knock(tick, beach, bot):
            op = bot.__dict__.get('_level_op') or {}
            if op.get('phase') == 'aggro_walk' and not getattr(beach, 'hit', False):
                beach.hit = True
                beach.level, beach.x = 'ground', 0.55
        beach = self._Beach(step=5, start=0.45)
        bot, pressed = self._farm(beach, 40, on_tick=knock)
        self.assertTrue(beach.hit)
        self.assertLess(pressed.count('q'), Bot.AGGRO_Q_PRESSES)    # stopped early
        self.assertEqual(bot._level, 'ground')
        self.assertIsNone(bot.__dict__.get('_level_op'))

    def _pressed_now(self, bot):
        return self.__dict__.get('_live_pressed', [])

    def test_normal_profiles_never_change_level(self):
        beach = self._Beach(step=4)
        bot, pressed = self._farm(beach, 60, enabled=False)
        self.assertNotIn('alt', pressed)
        self.assertEqual(beach.events, [])

    def test_stall_watchdog_forgets_walls_and_bans(self):
        bot = self._bot()
        bot._climb_ban_until = 1e9
        tracker = SimpleNamespace(ledge_enemies=lambda: (2, 300),
                                  lower_enemies=lambda: (0, None))
        with patch.object(bot_module, 'config', SimpleNamespace(capture=None)):
            for second in range(17):
                bot._check_stall(tracker, None, 100.0 + second)
        self.assertEqual(bot._walls, {})
        self.assertEqual(bot._climb_ban_until, float('-inf'))

    def test_stall_watchdog_quiet_while_fighting_or_nothing_in_view(self):
        bot = self._bot()
        tracker = SimpleNamespace(ledge_enemies=lambda: (0, None),
                                  lower_enemies=lambda: (0, None))
        for second in range(30):
            bot._check_stall(tracker, None, 100.0 + second)          # nothing seen
            bot._check_stall(tracker, (20, True), 100.5 + second)    # fighting
        self.assertEqual(bot._walls, {'right': 0.60})

    def test_no_level_change_while_a_pig_is_in_reach_or_with_none_elsewhere(self):
        beach = self._Beach(ground=50, step=4)
        _, pressed = self._farm(beach, 40)
        self.assertNotIn('alt', pressed)
        beach = self._Beach()
        _, pressed = self._farm(beach, 40)
        self.assertNotIn('alt', pressed)


class LowerLevelTrackingTests(unittest.TestCase):
    """Seeing pigs on the ground below while standing up on the step."""
    GROUND_Y = 400

    def _tracker(self):
        tracker = MobTracker.__new__(MobTracker)
        tracker._lower_spots = []
        return tracker

    def _enemy(self, x, depth, height=40):
        # centre y for an enemy whose feet are DEPTH px below the ground line
        return positioned_candidate(0, x, self.GROUND_Y + depth - height / 2, height=height)

    def test_pig_standing_on_the_ground_below_is_seen(self):
        tracker = self._tracker()
        found = []
        for step in range(6):
            found = tracker._lower_level_items(
                [self._enemy(300 + 6 * step, 65)], self.GROUND_Y, 10.0 + 0.1 * step)
        self.assertEqual(len(found), 1)

    def test_own_jump_does_not_make_ground_pigs_look_lower(self):
        tracker = self._tracker()
        # The character's jump: its ground line rises and falls, so the pig's
        # depth below it changes the whole time
        arc = [10, 35, 60, 78, 85, 80, 62, 40, 15, 0]
        for step, depth in enumerate(arc):
            found = tracker._lower_level_items(
                [self._enemy(300, 0)], self.GROUND_Y - depth, 10.0 + 0.07 * step)
            self.assertEqual(found, [], f'jump frame {step} counted as lower')

    def test_pig_on_my_level_is_never_lower(self):
        tracker = self._tracker()
        for step in range(10):
            self.assertEqual(tracker._lower_level_items(
                [self._enemy(300, 0)], self.GROUND_Y, 10.0 + 0.1 * step), [])
