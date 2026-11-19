import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.modules import bot as bot_module
from src.modules import mini_games
from src.modules.bot import Bot
from src.modules.mini_games import MiniGameTally, _result_label


class DropBanCrashTests(unittest.TestCase):
    """03:36:39 log: a ban made with no minimap position crashed the bot thread."""

    def test_a_ban_without_a_position_does_not_crash(self):
        bot = Bot.__new__(Bot)
        with patch('builtins.print'):
            bot._ban_drops('left', None, 10.0, 'a wall in the way')
        self.assertTrue(bot._drop_banned('left', 0.40, 11.0))      # covers that side
        self.assertFalse(bot._drop_banned('right', 0.40, 11.0))
        bot._ban_drops('right', 0.40, 10.0, 'a wall in the way')
        self.assertTrue(bot._drop_banned('right', None, 11.0))
        self.assertFalse(bot._drop_banned('left', 0.40, 30.0))     # expired


class GuardedTickTests(unittest.TestCase):
    def _bot(self, failing=True):
        bot = Bot.__new__(Bot)
        bot._auto_attack_movement = 'left'
        bot._mob_attack_keys = ('left', 'ctrl')

        def tick():
            if failing:
                raise TypeError("unsupported operand type(s) for -: 'float' and 'NoneType'")
        bot._run_standalone_mob_attack = tick
        return bot

    def test_an_error_is_logged_keys_released_and_farming_goes_on(self):
        bot = self._bot()
        stub = SimpleNamespace(enabled=True)
        released = []
        with patch.object(bot_module, 'config', stub), \
                patch.object(bot_module, 'key_up', released.append), \
                patch.object(bot_module.time, 'sleep'), \
                patch.object(bot_module.traceback, 'print_exc'), \
                patch('builtins.print'):
            bot._guarded_tick()
        self.assertTrue(stub.enabled)
        self.assertIsNone(bot._auto_attack_movement)
        self.assertEqual(bot._mob_attack_keys, ())
        self.assertEqual(set(released), {'left', 'ctrl'})

    def test_repeated_errors_pause_the_bot(self):
        bot = self._bot()
        stub = SimpleNamespace(enabled=True)
        with patch.object(bot_module, 'config', stub), \
                patch.object(bot_module, 'key_up'), \
                patch.object(bot_module.time, 'sleep'), \
                patch.object(bot_module.traceback, 'print_exc'), \
                patch('builtins.print'):
            for _ in range(Bot.TICK_ERROR_LIMIT):
                bot._guarded_tick()
        self.assertFalse(stub.enabled)


class TallyTests(unittest.TestCase):
    def test_running_count_per_game_and_outcome(self):
        tally = MiniGameTally()
        self.assertIn('none yet', tally.summary())
        tally.record('Violetta', 'done')
        tally.record('Shape (fading figure)', 'solved')
        line = tally.record('Violetta', 'done')
        self.assertIn('Mini-games this session: 3', line)
        self.assertIn('Violetta x2 (2 done)', line)
        self.assertIn('Shape (fading figure) x1 (1 solved)', line)

    def test_outcome_labels(self):
        self.assertEqual(_result_label(None), 'done')
        self.assertEqual(_result_label(False), 'failed')
        self.assertEqual(_result_label('error'), 'error')
        self.assertEqual(_result_label('timeout'), 'timeout')

    def test_a_finished_game_prints_the_tally(self):
        runner = mini_games.MiniGameRunner()
        fresh = MiniGameTally()
        printed = []
        with patch.object(mini_games, 'TALLY', fresh), \
                patch.object(mini_games, '_machine_lock', return_value=0), \
                patch.object(mini_games, 'config', SimpleNamespace(enabled=False)), \
                patch('builtins.print', lambda *a, **k: printed.append(' '.join(map(str, a)))):
            runner._run('pop-up-2', 'Violetta', lambda stop: 'done')
        self.assertTrue(any('Mini-games this session: 1' in line and 'Violetta x1' in line
                            for line in printed))


if __name__ == '__main__':
    unittest.main()
