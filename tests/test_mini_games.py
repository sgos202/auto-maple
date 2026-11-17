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
            'pop-up-3': ('Letter quiz', fake_game),
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

    def test_question_pop_up_starts_letter_quiz(self):
        """pop-up-3 is followed by the "Click all X's" quiz, which has a solver."""
        self.assertTrue(mini_games.AUTO_SOLVE['pop-up-3'])
        config.enabled = True
        runner = mini_games.MiniGameRunner()
        self.assertTrue(runner.trigger('pop-up-3'))
        self.release.set()
        runner.thread.join(2)
        self.assertEqual(self.calls, [False])
        self.assertTrue(config.enabled)


class TestAlarmStopsWhenCleared(unittest.TestCase):
    """The siren and flashing button go away by themselves once a game is cleared."""

    def setUp(self):
        self.enabled = config.enabled
        config.enabled = False
        self.patches = [mock.patch.object(mini_games.phone_alert, 'wants', return_value=False),
                        mock.patch.object(mini_games.time, 'sleep'),
                        mock.patch.object(mini_games, 'ALARM')]
        self.alarm = [p.start() for p in self.patches][-1]

    def tearDown(self):
        for p in self.patches:
            p.stop()
        config.enabled = self.enabled

    def _finish(self, result):
        def game(stop):
            if isinstance(result, Exception):
                raise result
            return result
        with mock.patch.object(mini_games.traceback, 'print_exc'):
            mini_games.MiniGameRunner()._run('pop-up-2', 'Violetta', game)
        return self.alarm.dismiss.called

    def test_cleared_games_dismiss_the_alarm(self):
        for result in ('passed', 'done', 'answered [2, 5]'):
            self.alarm.reset_mock()
            self.assertTrue(self._finish(result), result)

    def test_games_not_cleared_keep_ringing(self):
        for result in ('no pass notice', 'timed out', 'no figure', 'stopped',
                       RuntimeError('boom')):
            self.alarm.reset_mock()
            self.assertFalse(self._finish(result), result)


class TestViolettaPassNotice(unittest.TestCase):
    """After a won Violetta game the "passed" notice's OK is clicked."""

    def _play(self, closed, stopped=False):
        from src.detection import pass_notice, violetta_solver
        stop = threading.Event()
        if stopped:
            stop.set()
        with mock.patch.object(violetta_solver, 'play_live') as play,                 mock.patch.object(pass_notice, 'click_ok', return_value=closed) as click:
            result = mini_games._violetta_game(stop)
        play.assert_called_once()
        return result, click

    def test_results(self):
        self.assertEqual(self._play('passed')[0], 'passed')
        self.assertEqual(self._play(None)[0], 'no pass notice')
        self.assertEqual(self._play('stuck')[0], 'passed but OK not clicked')

    def test_stopped_game_does_not_click(self):
        result, click = self._play('passed', stopped=True)
        self.assertEqual(result, 'stopped')
        click.assert_not_called()


class TestPassNotice(unittest.TestCase):
    def setUp(self):
        import cv2
        import numpy as np
        from src.detection import pass_notice
        self.pn = pass_notice
        self.screen = np.zeros((768, 1366, 3), np.uint8)
        self.reference = cv2.imread('assets/post_violetta.png')
        self.with_notice = self.screen.copy()
        h, w = self.reference.shape[:2]
        self.with_notice[100:100 + h, 400:400 + w] = self.reference

    def test_finds_ok_button_in_the_reference(self):
        notice = self.pn.find(self.with_notice)
        self.assertIsNotNone(notice)
        self.assertAlmostEqual(notice.ok[0], 400 + 342, delta=3)
        self.assertAlmostEqual(notice.ok[1], 100 + 125, delta=3)

    def test_other_screens_have_no_notice(self):
        import cv2
        self.assertIsNone(self.pn.find(self.screen))
        for name in ('pop-up-1', 'pop-up-2', 'pop-up-3', 'pop-up-3-quiz'):
            self.assertIsNone(self.pn.find(cv2.imread(f'assets/{name}.png')), name)

    def _click_ok(self, frames, **kwargs):
        frames = iter(frames)
        last = [self.screen]

        def grab():
            last[0] = next(frames, last[0])
            return last[0], (1151, 61)

        presses = []
        with mock.patch.object(self.pn, 'GONE_SECONDS', 0.0):
            result = self.pn.click_ok(grab=grab, press=lambda x, y: presses.append((x, y)),
                                      sleep=lambda s: None, **kwargs)
        return result, presses

    def test_clicks_ok_once_the_notice_shows_and_reports_passed(self):
        result, presses = self._click_ok([self.screen, self.with_notice, self.screen])
        self.assertEqual(result, 'passed')
        self.assertEqual(len(presses), 1)
        self.assertAlmostEqual(presses[0][0], 1151 + 400 + 342, delta=3)
        self.assertAlmostEqual(presses[0][1], 61 + 100 + 125, delta=3)

    def test_clicks_again_if_it_stays_then_gives_up(self):
        result, presses = self._click_ok([self.with_notice] * 50, wait=5.0)
        self.assertEqual(result, 'stuck')
        self.assertEqual(len(presses), self.pn.CLICK_TRIES)

    def test_no_notice_means_not_passed(self):
        result, presses = self._click_ok([self.screen], wait=0.3)
        self.assertIsNone(result)
        self.assertEqual(presses, [])


class TestNotifierPopUps(unittest.TestCase):
    """The notifier's real pop-up loop, on the reference screenshots."""

    def test_pop_up_starts_its_solver_once(self):
        import cv2
        import numpy as np
        from src.modules import notifier

        watcher = notifier.Notifier()
        watcher.mini_games = mock.Mock()
        blank = np.zeros((768, 1366, 3), np.uint8)
        with mock.patch.object(notifier, 'ALARM') as alarm,                 mock.patch.object(watcher, '_save_pop_up'):
            for prefix in ('pop-up-1', 'pop-up-2', 'pop-up-3'):
                screen = blank.copy()
                pop_up = cv2.imread(f'assets/{prefix}.png')
                screen[200:200 + pop_up.shape[0], 500:500 + pop_up.shape[1]] = pop_up
                watcher.mini_games.reset_mock()
                alarm.reset_mock()
                watcher._check_pop_ups(blank)
                watcher.mini_games.trigger.assert_not_called()
                watcher._check_pop_ups(screen)
                watcher._check_pop_ups(screen)      # still visible: no second start
                watcher.mini_games.trigger.assert_called_once_with(prefix)
                alarm.ring.assert_called_once()     # the same alarm for every game
                for _ in range(3):
                    watcher._check_pop_ups(blank)   # gone: re-arm
                alarm.dismiss.assert_not_called()   # only the user's click stops it


class TestMiniGameAlarm(unittest.TestCase):
    def test_rings_on_every_device_until_dismissed(self):
        import time
        from src.modules import mini_game_alarm
        alarm = mini_game_alarm.MiniGameAlarm()
        players = [mock.Mock(), mock.Mock()]
        with mock.patch.object(mini_game_alarm, 'output_devices', return_value=['A', 'B']),                 mock.patch.object(mini_game_alarm.subprocess, 'Popen', side_effect=players) as popen:
            alarm.ring('Violetta mini-game')
            alarm.ring('Violetta mini-game')        # already ringing: nothing more
            for _ in range(100):
                if alarm._players:
                    break
                time.sleep(0.01)
        self.assertEqual([c.args[0][-1] for c in popen.call_args_list], ['A', 'B'])
        self.assertTrue(alarm.active)
        alarm.dismiss()
        self.assertFalse(alarm.active)
        for player in players:
            player.stdin.close.assert_called_once()
            player.terminate.assert_called_once()


class TestAlarmPlacement(unittest.TestCase):
    def test_centre_when_the_game_is_elsewhere(self):
        from src.modules.gui import alarm_spot
        self.assertEqual(alarm_spot(2560, 1440, 520, 170, None), (1020, 635, False))
        self.assertEqual(alarm_spot(2560, 1440, 520, 170, (0, 0, 800, 600)), (1020, 635, False))

    def test_never_covers_the_game_when_there_is_room(self):
        from src.modules.gui import alarm_spot
        game = (1151, 61, 2517, 829)            # the user's layout, 2026-11-15
        x, y, overlaps = alarm_spot(2560, 1440, 520, 170, game)
        self.assertFalse(overlaps)
        self.assertTrue(x + 520 <= game[0] or x >= game[2] or y + 170 <= game[1] or y >= game[3])
        self.assertEqual(x, 1020)               # still centred across

    def test_no_room_says_so(self):
        from src.modules.gui import alarm_spot
        self.assertTrue(alarm_spot(1920, 1080, 520, 170, (0, 0, 1920, 1080))[2])


class TestStartStopDuringGame(unittest.TestCase):
    def setUp(self):
        self.enabled = config.enabled
        self.patches = [mock.patch.object(mini_games.utils, 'print_state'),
                        mock.patch.object(mini_games.utils, 'print_separator'),
                        mock.patch.object(mini_games.phone_alert, 'wants', return_value=False),
                        mock.patch.object(mini_games.time, 'sleep')]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        config.enabled = self.enabled

    def _run_with_key_presses(self, presses):
        config.enabled = True
        runner = mini_games.MiniGameRunner()
        runner.thread = threading.current_thread()      # looks busy, like a real game
        seen = []

        def game(stop):
            for _ in range(presses):
                seen.append(mini_games.handle_start_stop())
            seen.append(config.enabled)
            return 'done'

        runner._run('pop-up-1', 'Shape', game)
        self.assertEqual(seen, [True] * presses + [False])  # never starts mid-game
        return config.enabled

    def test_no_key_resumes(self):
        self.assertTrue(self._run_with_key_presses(0))

    def test_key_once_stays_stopped(self):
        self.assertFalse(self._run_with_key_presses(1))

    def test_key_twice_resumes(self):
        self.assertTrue(self._run_with_key_presses(2))

    def test_key_works_normally_without_a_game(self):
        mini_games.MiniGameRunner()
        self.assertFalse(mini_games.handle_start_stop())


if __name__ == '__main__':
    unittest.main()
