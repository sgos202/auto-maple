"""henny_hunting.mp4: a wall learnt by mistake at x ~0.29 kept the patrol on 0.145-0.29."""

import unittest
from types import SimpleNamespace
from unittest import mock

from src.common import config
from src.modules import bot as bot_module
from src.modules.bot import Bot


def make_bot():
    bot = Bot.__new__(Bot)
    bot.map_events = SimpleNamespace(limits=lambda: {})
    bot._blocked_side = None
    return bot


class WallRecheckTests(unittest.TestCase):
    def setUp(self):
        self.clock = [1000.0]
        patcher = mock.patch.object(bot_module.time, 'monotonic', lambda: self.clock[0])
        patcher.start()
        self.addCleanup(patcher.stop)
        tracker = SimpleNamespace(level_farming_enabled=lambda: False,
                                  set_blocked_side=lambda side: None)
        capture = mock.patch.object(config, 'capture', SimpleNamespace(mob_tracker=tracker),
                                    create=True)
        capture.start()
        self.addCleanup(capture.stop)

    def test_a_fresh_wall_turns_the_patrol(self):
        bot = make_bot()
        bot._learn_wall('right', 0.29)
        self.clock[0] += 10
        self.assertTrue(bot._wall_ahead('right', 0.285))

    def test_an_old_wall_is_walked_past_once(self):
        bot = make_bot()
        bot._learn_wall('right', 0.29)
        self.clock[0] += Bot.WALL_RECHECK_SECONDS + 1
        with mock.patch('builtins.print'):
            self.assertFalse(bot._wall_ahead('right', 0.285))
        self.assertNotIn('right', bot._walls)

    def test_a_real_wall_is_learnt_again(self):
        bot = make_bot()
        bot._learn_wall('left', 0.144)
        self.clock[0] += Bot.WALL_RECHECK_SECONDS + 1
        with mock.patch('builtins.print'):
            bot._wall_ahead('left', 0.15)
        bot._learn_wall('left', 0.144)      # the walk stopped there again
        self.assertTrue(bot._wall_ahead('left', 0.15))

    def test_hitting_a_known_wall_does_not_renew_it(self):
        bot = make_bot()
        bot._learn_wall('right', 0.29)
        learnt = bot._wall_learnt['right']
        self.clock[0] += 20
        with mock.patch.object(Bot, '_patrol_for_mob', lambda self: None), \
                mock.patch.object(Bot, '_set_blocked_side', lambda self, side: None), \
                mock.patch('builtins.print'):
            bot._turn_back_from_wall('right', 0.286)
        self.assertEqual(bot._wall_learnt['right'], learnt)

    def test_pig_beach_walls_stay(self):
        bot = make_bot()
        bot._learn_wall('right', 0.29)
        self.clock[0] += Bot.WALL_RECHECK_SECONDS + 1
        config.capture.mob_tracker.level_farming_enabled = lambda: True
        self.assertTrue(bot._wall_ahead('right', 0.285))


if __name__ == '__main__':
    unittest.main()
