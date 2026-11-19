"""Magician's ranged pull (Magic Missile, magic-class.mp4)."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

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


def _info(offset=-230, serial=1, health=None, closing=0.0, **extra):
    info = {'time': 0, 'serial': serial, 'offset': offset, 'dy': 0,
            'on_level': True, 'airborne': False, 'profile': 'normal',
            'health': health, 'real': False, 'closing': closing,
            'attackable': False, 'waiting': False}
    info.update(extra)
    return info


class FakeTracker:
    def __init__(self, enabled=True, info=None, offsets=None):
        self.enabled = enabled
        self.info = info
        self.offsets = offsets
        self.standing = 0

    def missile_settings(self):
        return {'enabled': self.enabled, 'key': 'e', 'range': 270}

    def target_details(self):
        return None if self.info is None else dict(self.info)

    def target(self):
        return None if self.info is None else (self.info['offset'], self.info['attackable'])

    def enemy_offsets(self):
        if self.offsets is not None:
            return list(self.offsets)
        return [] if self.info is None else [self.info['offset']]

    def note_standing(self):
        self.standing += 1


class Harness:
    """Runs Bot._service_missile with patched keys and clock."""

    def __init__(self, tracker, mp=80.0):
        self.bot = _bot()
        self.tracker = tracker
        self.clock = [100.0]
        self.presses, self.downs = [], []
        self.mp = mp

    def tick(self, advance=0.0, x=0.5):
        self.clock[0] += advance
        config = SimpleNamespace(capture=SimpleNamespace(mob_tracker=self.tracker), enabled=True)
        with patch.object(bot_module, 'config', config), \
                patch.object(bot_module, 'is_game_focused', return_value=True), \
                patch.object(bot_module, 'press', lambda key, *a, **k: self.presses.append(key)), \
                patch.object(bot_module, 'key_down', self.downs.append), \
                patch.object(bot_module, 'key_up', lambda key: None), \
                patch.object(bot_module.time, 'sleep', lambda s: None), \
                patch.object(bot_module.time, 'monotonic', lambda: self.clock[0]), \
                patch.object(Bot, '_percent', lambda _self, monitor: self.mp), \
                patch('builtins.print', lambda *a, **k: None):
            target = self.tracker.target()
            return self.bot._service_missile(self.tracker, target, x)


class OffByDefaultTests(unittest.TestCase):
    def test_beginner_never_presses_the_key(self):
        run = Harness(FakeTracker(enabled=False, info=_info()))
        for _ in range(20):
            self.assertFalse(run.tick(0.2))
        self.assertEqual(run.presses, [])
        self.assertEqual(run.downs, [])

    def test_tracker_without_the_option_is_off(self):
        class OldTracker(FakeTracker):
            missile_settings = None
        run = Harness(OldTracker(info=_info()))
        self.assertFalse(run.tick())
        self.assertEqual(run.presses, [])

    def test_settings_and_profiles_without_the_key_load_it_off(self):
        tracker = tracker_module.MobTracker.__new__(tracker_module.MobTracker)
        tracker._lock = tracker_module.threading.Lock()
        # Configurable.load_config fills a key missing from the file with ''
        tracker.config = dict(tracker_module.MobTracker.DEFAULT_CONFIG,
                              **{'Magic missile': '', 'Missile key': '', 'Missile range': ''})
        self.assertEqual(tracker.missile_settings(),
                         {'enabled': False, 'key': 'e', 'range': 270})
        # load_profile: loaded.get(key, default) for a pre-missile profile
        loaded = {'Enemy speeds': 'jumper', 'Group skill': True}
        config = {key: loaded.get(key, default)
                  for key, default in tracker_module.MobTracker.DEFAULT_CONFIG.items()}
        self.assertIs(config['Magic missile'], False)


class CastTests(unittest.TestCase):
    def test_casts_at_a_far_enemy_facing_it_then_stands_through_the_cast(self):
        tracker = FakeTracker(info=_info(offset=-230))
        run = Harness(tracker)
        self.assertTrue(run.tick())
        self.assertEqual(run.presses, ['e'])
        self.assertIn('left', run.downs)            # turned to face it first
        self.assertGreater(tracker.standing, 0)
        # Mid-cast: still standing, no second cast
        self.assertTrue(run.tick(0.3))
        self.assertEqual(run.presses, ['e'])

    def test_an_enemy_reaching_swing_range_mid_cast_gets_the_swing(self):
        tracker = FakeTracker(info=_info(offset=150))
        run = Harness(tracker)
        run.tick()
        tracker.info = _info(offset=65, attackable=True)
        self.assertTrue(run.tick(0.15))             # the missile is still going out
        self.assertFalse(run.tick(0.2))             # out: melee swings

    def test_no_turn_when_already_facing(self):
        run = Harness(FakeTracker(info=_info(offset=200)))
        run.bot._facing = 'right'
        run.tick()
        self.assertEqual(run.downs, [])
        self.assertEqual(run.presses, ['e'])

    def test_no_cast_out_of_band_or_off_level(self):
        for info in (_info(offset=-90), _info(offset=-300), _info(attackable=True),
                     # coming, too close for a cast before it arrives
                     _info(offset=-140, waiting=True, closing=30.0),
                     _info(on_level=False), _info(airborne=True)):
            run = Harness(FakeTracker(info=info))
            self.assertFalse(run.tick(), info)
            self.assertEqual(run.presses, [], info)

    def test_no_cast_with_another_enemy_next_to_the_character(self):
        run = Harness(FakeTracker(info=_info(offset=-230), offsets=[-230, 70]))
        self.assertFalse(run.tick())
        self.assertEqual(run.presses, [])

    def test_no_cast_when_out_of_mp(self):
        run = Harness(FakeTracker(info=_info()), mp=3.0)
        self.assertFalse(run.tick())
        self.assertEqual(run.presses, [])


class PullTests(unittest.TestCase):
    def test_hit_twice_while_it_comes_then_stands_for_the_melee(self):
        tracker = FakeTracker(info=_info(offset=-250))
        run = Harness(tracker)
        run.tick()
        tracker.info = _info(offset=-245, health=0.7, closing=30.0)
        run.tick(0.95)                              # judged: a hit
        self.assertEqual(run.bot._missile_stats[1], 1)
        self.assertTrue(run.tick(0.1))              # cooled down: a second one
        self.assertEqual(run.presses, ['e', 'e'])
        tracker.info = _info(offset=-215, health=0.5, closing=30.0)
        run.tick(0.95)                              # the second hit
        self.assertEqual(run.bot._missile_stats[1], 2)
        for _ in range(8):                          # coming: stand facing it
            self.assertTrue(run.tick(0.25))
        self.assertEqual(run.presses, ['e', 'e'])   # two missiles, then melee
        # The melee model takes over once it is close (waiting/attackable)
        tracker.info = _info(offset=-150, health=0.7, closing=30.0, waiting=True)
        self.assertFalse(run.tick(0.1))
        tracker.info = _info(offset=-60, health=0.7, attackable=True)
        self.assertFalse(run.tick(0.1))

    def test_hit_enemy_that_does_not_come_is_walked_to(self):
        tracker = FakeTracker(info=_info(offset=-230))
        run = Harness(tracker)
        run.tick()
        tracker.info = _info(offset=-230, health=0.7, closing=0.0)
        run.tick(0.95)
        self.assertTrue(run.tick(0.1))              # a second missile first
        tracker.info = _info(offset=-230, health=0.5, closing=0.0)
        run.tick(0.95)
        self.assertTrue(run.tick(0.5))              # a moment to start coming
        self.assertFalse(run.tick(1.0))             # not coming: normal approach
        self.assertEqual(run.presses, ['e', 'e'])

    def test_hold_never_outlasts_the_wait(self):
        tracker = FakeTracker(info=_info(offset=-240))
        run = Harness(tracker)
        run.tick()
        tracker.info = _info(offset=-240, health=0.7, closing=20.0)
        run.tick(0.95)
        run.bot._missile_stats[1] = Bot.MISSILE_HITS_PER_TARGET    # no casts left
        self.assertFalse(run.tick(Bot.LURE_WAIT_SECONDS + 0.1))

    def test_kill_by_missile_is_neither_hit_nor_miss(self):
        tracker = FakeTracker(info=_info(offset=-230))
        run = Harness(tracker)
        run.tick()
        tracker.info = None
        self.assertFalse(run.tick(0.95))
        self.assertEqual(run.bot.__dict__.get('_missile_bans', []), [])
        tracker.info = _info(offset=-210, serial=2)  # the next enemy: cast again
        self.assertTrue(run.tick(0.2))
        self.assertEqual(run.presses, ['e', 'e'])


class VolleyTests(unittest.TestCase):
    """bug-2.mp4: more missiles at a group, and at an enemy coming in."""

    def _hits(self, offsets):
        tracker = FakeTracker(info=_info(offset=-250), offsets=offsets)
        run = Harness(tracker)
        health = 0.9
        for _ in range(6):
            run.tick(0.1)
            health -= 0.2
            tracker.info = _info(offset=-250, health=health, closing=0.0)
            run.tick(0.95)
        return run.presses.count('e')

    def test_a_group_gets_more_missiles_than_one_enemy(self):
        self.assertEqual(self._hits([-250]), Bot.MISSILE_HITS_PER_TARGET)
        self.assertEqual(self._hits([-250, -200, 240]), Bot.MISSILE_GROUP_HITS)

    def test_casts_at_an_enemy_coming_in_while_it_is_still_far(self):
        run = Harness(FakeTracker(info=_info(offset=-200, waiting=True, closing=30.0)))
        self.assertTrue(run.tick())
        self.assertEqual(run.presses, ['e'])

    def test_stands_while_the_hit_is_judged(self):
        tracker = FakeTracker(info=_info(offset=-230))
        run = Harness(tracker)
        run.tick()
        tracker.info = _info(offset=-225)
        before = tracker.standing
        self.assertTrue(run.tick(0.7))      # cast done, hit not judged yet: stand
        self.assertGreater(tracker.standing, before)


class RealEnemyTests(unittest.TestCase):
    def test_hurt_enemy_bar_wobble_is_no_hit_and_no_ban(self):
        # magic-class.mp4: the snail showed a bar (~40 %) before the missile;
        # the reads wobble 0.39-0.43
        tracker = FakeTracker(info=_info(offset=-259, health=0.40, real=True))
        run = Harness(tracker)
        run.tick()
        tracker.info = _info(offset=-259, health=0.41, real=True)
        run.tick(0.95)
        self.assertEqual(run.bot.__dict__.get('_missile_bans', []), [])
        self.assertIsNone(run.bot.__dict__.get('_missile_hold'))

    def test_hurt_enemy_coming_after_the_cast_is_a_hit(self):
        tracker = FakeTracker(info=_info(offset=-259, health=0.43, real=True))
        run = Harness(tracker)
        run.tick()
        tracker.info = _info(offset=-250, health=0.42, real=True, closing=40.0)
        run.tick(0.95)
        self.assertIsNotNone(run.bot._missile_hold)
        self.assertTrue(run.tick(0.2))              # stands, lets it come

    def test_real_enemy_out_of_reach_is_walked_to_not_banned(self):
        tracker = FakeTracker(info=_info(offset=-260, real=True))
        run = Harness(tracker)
        for _ in range(12):
            run.tick(0.5)
        self.assertEqual(run.presses, ['e', 'e'])   # two tries, then melee
        self.assertEqual(run.bot.__dict__.get('_missile_bans', []), [])


class ScenerySafetyTests(unittest.TestCase):
    """A portal matching the samples must not get missiles forever."""

    def test_a_miss_bans_the_spot_for_new_locks_too(self):
        tracker = FakeTracker(info=_info(offset=-200))
        run = Harness(tracker)
        run.tick()
        run.tick(0.95)                              # no HP bar: a miss
        self.assertEqual(len(run.bot._missile_bans), 1)
        tracker.info = _info(offset=-200, serial=2)  # re-locked, same spot
        for _ in range(10):
            run.tick(0.5)
        self.assertEqual(run.presses, ['e'])

    def test_the_whole_portal_cluster_is_banned(self):
        tracker = FakeTracker(info=_info(offset=-200), offsets=[-200, -240, -170])
        run = Harness(tracker)
        run.tick()
        run.tick(0.95)
        for serial, offset in ((2, -240), (3, -170)):
            tracker.info = _info(offset=offset, serial=serial)
            run.tick(1.1)
        self.assertEqual(run.presses, ['e'])

    def test_misses_in_a_row_pause_the_missile(self):
        tracker = FakeTracker(info=_info(offset=-200))
        run = Harness(tracker)
        for serial, offset in enumerate((-200, 130, -150), start=1):
            tracker.info = _info(offset=offset, serial=serial)
            tracker.offsets = [offset]
            run.tick(1.1)
            run.tick(0.95)
        self.assertEqual(len(run.presses), 3)
        tracker.info = _info(offset=220, serial=9)
        tracker.offsets = [220]
        run.tick(1.1)
        self.assertEqual(len(run.presses), 3)      # paused
        run.tick(Bot.MISSILE_PAUSE_SECONDS)
        self.assertEqual(len(run.presses), 4)

    def test_no_left_right_cast_loop_between_two_matches(self):
        # Two matches either side (a portal under the character's feet) that
        # take turns as the target: at most one missile each, then nothing.
        tracker = FakeTracker(info=_info(offset=-180), offsets=[-180, 180])
        run = Harness(tracker)
        for i in range(40):
            offset = -180 if i % 2 == 0 else 180
            tracker.info = _info(offset=offset, serial=i + 1)
            run.tick(0.5)
        self.assertLessEqual(len(run.presses), 3)


class StandingHintTests(unittest.TestCase):
    def test_note_standing_stops_the_walk_assumption(self):
        tracker = tracker_module.MobTracker.__new__(tracker_module.MobTracker)
        tracker._lock = tracker_module.threading.Lock()
        tracker.note_standing()
        self.assertIn('_standing_at', tracker.__dict__)


if __name__ == '__main__':
    unittest.main()
