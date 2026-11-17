"""Tests for the automatic game restart, replayed on the screens of
assets/restart_process_maple.mp4 (one picture of each in assets/restart/)."""

import unittest
from unittest.mock import patch

import cv2

from src.modules import game_restart as gr


def picture(name):
    image = cv2.imread(f'assets/restart/{name}.jpg')
    assert image is not None, name
    return image


# Where a click has to land on each screen to move on (x0, y0, x1, y1), measured
# on the recording by eye - not taken from the code's own offsets.
EXIT_GAME = (768, 718, 851, 736)
WORLD_TAB = (268, 193, 371, 217)
CHARACTER = (320, 415, 370, 480)
SHUT_CHANNELS = {2, 3, 5, 6, 7, 10, 11}         # greyed out in the recording


def channel_at(point):
    """The channel whose cell POINT is in on the channel-list picture, or None."""
    for channel in range(1, 41):
        column, row = (channel - 1) % 8, (channel - 1) // 8
        x, y = 829 - 530 + 68 * column, 898 - 455 + 32 * row
        if abs(point[0] - x) <= 30 and abs(point[1] - y) <= 13:
            return channel
    return None


def inside(point, box):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


class FakeGame:
    """The game as the recording shows it: each screen moves on only when the
    right spot is clicked."""

    def __init__(self, state='map', launcher_needs_play=False):
        self.state = state
        self.frames = {name.split('-', 1)[1]: picture(name) for name in (
            '1-map', '2-menu', '3-quit', '4-login', '5-world', '6-channels', '7-character')}
        self.launcher = picture('launcher')
        self.launcher_needs_play = launcher_needs_play
        self.clicks = []
        self.channel = None
        self.launched = 0
        self.on_desktop = False

    # -- what GameRestart uses
    def window(self):
        return None if self.state in ('closed', 'launcher') else 1

    def focus(self):
        return True

    def game(self):
        self.on_desktop = False
        if self.window() is None:
            return None
        return self.frames['map' if self.state == 'map again' else self.state]

    def desktop(self):
        self.on_desktop = True
        return self.launcher

    def park(self):
        pass

    def click(self, point, double=False):
        self.clicks.append((self.state, point, double))
        if self.on_desktop:
            if self.state == 'launcher' and gr.find(self.launcher, 'restart_launcher_play'):
                x, y = gr.find(self.launcher, 'restart_launcher_play')
                if abs(point[0] - x) < 60 and abs(point[1] - y) < 20:
                    self.state = 'login'
            return
        if self.state == 'map' and inside(point, self._near('map', 'menu_button')):
            self.state = 'menu'
        elif self.state == 'menu' and inside(point, EXIT_GAME):
            self.state = 'quit'
        elif self.state == 'quit' and inside(point, self._near('quit', 'restart_quit_ok')):
            self.state = 'closed'
        elif self.state == 'login' and inside(point, self._near('login', 'restart_login_button')):
            self.state = 'world'
        elif self.state == 'world' and inside(point, WORLD_TAB):
            self.state = 'channels'
        elif self.state == 'channels' and double:
            channel = channel_at(point)
            if channel is not None and channel not in SHUT_CHANNELS:
                self.channel = channel
                self.state = 'character'
        elif self.state == 'character' and double and inside(point, CHARACTER):
            self.state = 'map again'

    def _near(self, state, name, reach=15):
        x, y = gr.find(self.frames[state], name)
        return x - reach, y - reach, x + reach, y + reach

    def launch(self):
        self.launched += 1
        self.state = 'launcher' if self.launcher_needs_play else 'login'


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def restart(game):
    clock = Clock()
    return gr.GameRestart(screen=game, launch=game.launch, clock=clock, sleep=clock.sleep,
                          record_dir=None, log=lambda text: None), clock


class ScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gr._load_templates()

    def test_each_screen_is_told_apart(self):
        for name, screen in (('1-map', 'map'), ('2-menu', 'map'), ('3-quit', 'quit'),
                             ('4-login', 'login'), ('5-world', 'world'),
                             ('6-channels', 'channels'), ('7-character', 'character'),
                             ('8-map', 'map')):
            self.assertEqual(gr.which_screen(picture(name)), screen, name)

    def test_game_menu_is_found_when_open(self):
        self.assertIsNotNone(gr.find(picture('2-menu'), 'menu_change_channel'))
        self.assertIsNone(gr.find(picture('1-map'), 'menu_change_channel'))

    def test_open_channels_skip_the_greyed_ones(self):
        frame = picture('6-channels')
        go = gr.find(frame, 'restart_world_go')
        self.assertEqual(gr.open_channels(frame, go),
                         [c for c in range(1, 41) if c not in SHUT_CHANNELS])

    def test_launcher_play_button(self):
        self.assertIsNotNone(gr.find(picture('launcher'), 'restart_launcher_play'))
        self.assertIsNone(gr.find(picture('4-login'), 'restart_launcher_play'))


class RestartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gr._load_templates()

    def test_whole_restart(self):
        game = FakeGame()
        run, clock = restart(game)
        self.assertEqual(run.run(), 'restarted')
        self.assertEqual(game.state, 'map again')
        self.assertEqual(game.launched, 1)
        self.assertNotIn(game.channel, SHUT_CHANNELS)
        self.assertLess(clock.now, 60)

    def test_launcher_that_waits_for_play(self):
        game = FakeGame(launcher_needs_play=True)
        run, _ = restart(game)
        self.assertEqual(run.run(), 'restarted')
        self.assertEqual(game.state, 'map again')

    def test_game_already_closed_is_just_started(self):
        game = FakeGame(state='closed')
        run, _ = restart(game)
        self.assertEqual(run.run(), 'restarted')
        self.assertEqual(game.launched, 1)

    def test_game_left_on_the_login_screen_is_not_relaunched(self):
        game = FakeGame(state='login')
        run, _ = restart(game)
        self.assertEqual(run.run(), 'restarted')
        self.assertEqual(game.launched, 0)

    def test_game_that_never_opens_fails_cleanly(self):
        game = FakeGame()
        game.launch = lambda: setattr(game, 'state', 'closed')
        game.launcher = picture('4-login')      # no PLAY button to press either
        run, clock = restart(game)
        run.launch = game.launch
        result = run.run()
        self.assertTrue(result.startswith('failed: no game window'), result)
        self.assertGreaterEqual(clock.now, gr.WINDOW_WAIT)

    def test_stop_calls_it_off(self):
        import threading
        game = FakeGame()
        run, _ = restart(game)
        run.stop = threading.Event()
        run.stop.set()
        self.assertEqual(run.run(), 'failed: stopped')
        self.assertEqual((game.state, game.clicks), ('map', []))

    def test_restart_stays_paused_if_minimap_is_not_found(self):
        from src.common import config, utils
        from src.modules import phone_alert

        was_enabled = config.enabled
        runner = gr.RestartRunner()
        try:
            config.enabled = True
            with patch.object(gr, 'GameRestart') as game_restart, \
                    patch.object(gr.RestartRunner, '_await_minimap', return_value=False), \
                    patch.object(gr.time, 'sleep'), \
                    patch.object(utils, 'print_separator'), \
                    patch.object(utils, 'print_state') as print_state, \
                    patch.object(phone_alert, 'wants', return_value=False):
                game_restart.return_value.run.return_value = 'restarted'
                runner._run('test')

            self.assertFalse(config.enabled)
            print_state.assert_not_called()
        finally:
            config.enabled = was_enabled


class LauncherRightsTests(unittest.TestCase):
    """Windows' permission box can't be clicked, so the launcher is re-opened as
    administrator when it isn't one (and only then)."""

    def prepare(self, admin, elevated):
        killed, said = [], []
        originals = gr.is_admin, gr._pids, gr._process_is_elevated, gr.subprocess.run, gr.time.sleep
        running = {'nexon_runtime.exe': [10], 'nexon_client.exe': [11, 12]}
        gr.is_admin = lambda: admin
        gr._pids = lambda image: [] if killed else running.get(image, [])
        gr._process_is_elevated = lambda pid: elevated
        gr.subprocess.run = lambda command, **kwargs: killed.append(command[-1])
        gr.time.sleep = lambda seconds: None
        try:
            result = gr.prepare_launcher(said.append)
        finally:
            gr.is_admin, gr._pids, gr._process_is_elevated, gr.subprocess.run, gr.time.sleep = originals
        return result, killed, said

    def test_plain_launcher_is_closed_so_an_administrator_one_starts(self):
        result, killed, said = self.prepare(admin=True, elevated=False)
        self.assertTrue(result)
        self.assertEqual(killed, list(gr.LAUNCHER_PROCESSES))
        self.assertNotIn('MapleStory.exe', killed)

    def test_administrator_launcher_is_left_alone(self):
        result, killed, said = self.prepare(admin=True, elevated=True)
        self.assertEqual((result, killed, said), (True, [], []))

    def test_without_administrator_rights_it_only_says_so(self):
        result, killed, said = self.prepare(admin=False, elevated=False)
        self.assertFalse(result)
        self.assertEqual(killed, [])
        self.assertIn('click Yes', said[0])


class TriggerTests(unittest.TestCase):
    def test_only_when_farming_low_on_memory_and_the_game_leaked(self):
        low, leaked = gr.RESTART_FREE_MB - 1, gr.RESTART_MIN_HANDLES
        never = float('-inf')
        self.assertTrue(gr.should_restart(low, leaked, True, 0.0, never))
        self.assertFalse(gr.should_restart(low, leaked, False, 0.0, never))         # paused
        self.assertFalse(gr.should_restart(low + 1, leaked, True, 0.0, never))      # enough memory
        self.assertFalse(gr.should_restart(low, leaked - 1, True, 0.0, never))      # not the game
        self.assertFalse(gr.should_restart(low, None, True, 0.0, never))            # game not open
        self.assertFalse(gr.should_restart(low, leaked, True, 100.0, 0.0))          # just restarted
        self.assertTrue(gr.should_restart(low, leaked, True,
                                          gr.RESTART_MIN_GAP_SECONDS, 0.0))

    def test_a_bare_memory_check_never_restarts(self):
        from src.modules import memory_guard as mg
        guard = mg.MemoryGuard()
        guard._started = 0.0
        originals = mg.system_commit, mg.game_handles
        mg.system_commit = lambda: (1000.0, 53000.0)
        mg.game_handles = lambda name='MapleStory': 200000
        try:
            guard.check(now=1000.0)
        finally:
            mg.system_commit, mg.game_handles = originals
        self.assertFalse(gr.RUNNER.busy)


if __name__ == '__main__':
    unittest.main()
