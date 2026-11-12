"""Tests for starting mini-game solvers when their pop-ups appear."""

import threading
import unittest
from unittest import mock

from src.common import config
from src.modules import mini_games


class TestMiniGameRunner(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.release = threading.Event()

        def fake_game(stop):
            self.calls.append(config.enabled)   # routine state while the game runs
            self.release.wait(2)
            return 'done'

        self.games = mock.patch.dict(mini_games.GAMES, {
            'pop-up-1': ('Shape', fake_game),
            'pop-up-2': ('Violetta', fake_game),
        })
        self.games.start()
        self.print_state = mock.patch.object(mini_games.utils, 'print_state')
        self.print_state.start()
        self.sleep = mock.patch.object(mini_games.time, 'sleep')
        self.sleep.start()
        self.enabled = config.enabled

    def tearDown(self):
        self.release.set()
        self.games.stop()
        self.print_state.stop()
        self.sleep.stop()
        config.enabled = self.enabled

    def test_pauses_routine_during_game_and_resumes_after(self):
        config.enabled = True
        runner = mini_games.MiniGameRunner()
        self.assertTrue(runner.trigger('pop-up-1'))
        self.release.set()
        runner.thread.join(2)
        self.assertEqual(self.calls, [False])
        self.assertTrue(config.enabled)

    def test_leaves_routine_off_if_it_was_off(self):
        config.enabled = False
        runner = mini_games.MiniGameRunner()
        runner.trigger('pop-up-2')
        self.release.set()
        runner.thread.join(2)
        self.assertFalse(config.enabled)

    def test_one_game_at_a_time_and_cooldown(self):
        config.enabled = False
        runner = mini_games.MiniGameRunner()
        self.assertTrue(runner.trigger('pop-up-1'))
        self.assertFalse(runner.trigger('pop-up-2'))     # busy
        self.release.set()
        runner.thread.join(2)
        self.assertFalse(runner.trigger('pop-up-1'))     # same pop-up right after: cooldown
        self.assertTrue(runner.trigger('pop-up-2'))      # a different game is fine
        runner.thread.join(2)

    def test_disabled_or_unknown_pop_up_does_nothing(self):
        runner = mini_games.MiniGameRunner()
        with mock.patch.dict(mini_games.AUTO_SOLVE, {'pop-up-1': False}):
            self.assertFalse(runner.trigger('pop-up-1'))
        self.assertFalse(runner.trigger('pop-up-9'))
        self.assertEqual(self.calls, [])


class TestNotifierPopUps(unittest.TestCase):
    """The notifier's real pop-up loop, on the reference screenshots."""

    def test_pop_up_starts_its_solver_once(self):
        import cv2
        import numpy as np
        from src.modules import notifier

        watcher = notifier.Notifier()
        watcher._play_pop_up_sound = mock.Mock()
        watcher.mini_games = mock.Mock()
        blank = np.zeros((768, 1366, 3), np.uint8)
        for prefix in ('pop-up-1', 'pop-up-2'):
            screen = blank.copy()
            pop_up = cv2.imread(f'assets/{prefix}.png')
            screen[300:300 + pop_up.shape[0], 500:500 + pop_up.shape[1]] = pop_up
            watcher.mini_games.reset_mock()
            watcher._check_pop_ups(blank)
            watcher.mini_games.trigger.assert_not_called()
            watcher._check_pop_ups(screen)
            watcher._check_pop_ups(screen)      # still visible: no second start
            watcher.mini_games.trigger.assert_called_once_with(prefix)
            for _ in range(3):
                watcher._check_pop_ups(blank)   # gone: re-arm


if __name__ == '__main__':
    unittest.main()
