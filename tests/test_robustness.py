"""Long-run robustness (14:30 log) and the island gather's hit/Q handling."""

import os
import shutil
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.common import output
from src.modules import bot as bot_module
from src.modules.bot import Bot
from src.modules.watchdog import Watchdog


class WatchdogTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.dog = Watchdog()
        self.dog.HANG_DIR = self.dir
        self.dog.start = lambda: None       # checks are driven by the test

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_stalled_loop_is_dumped_and_restarted_once(self):
        restarts = []
        self.dog.register('capture', 10, restart=lambda: restarts.append(1))
        start = self.dog._beats['capture']
        with patch('src.common.vkeys.release_all_keys', return_value=[]):
            self.assertEqual(self.dog.check(start + 5), [])
            self.assertEqual(self.dog.check(start + 11), ['capture'])
        self.assertEqual(restarts, [1])
        dumps = os.listdir(self.dir)
        self.assertEqual(len(dumps), 1)
        with open(os.path.join(self.dir, dumps[0]), encoding='utf-8') as file:
            self.assertIn('capture stalled', file.read())

    def test_paused_loop_is_not_a_stall(self):
        self.dog.register('capture', 10, restart=lambda: None)
        self.dog.pause('capture')
        start = self.dog._beats['capture']
        self.assertEqual(self.dog.check(start + 100), [])


class ConsoleTests(unittest.TestCase):
    def test_a_stuck_console_does_not_block_print(self):
        # A console in QuickEdit selection blocks every write
        release = threading.Event()

        class StuckConsole:
            def write(self, text):
                release.wait(5)

            def flush(self):
                release.wait(5)

        capture = output.OutputCapture()
        stream = output.TeeStream(StuckConsole(), capture, 'stdout',
                                  output.ConsoleWriter())
        started = time.monotonic()
        for _ in range(50):
            stream.write('line\n')
            stream.flush()
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(len(capture.drain()), 50)   # the GUI log still gets it
        release.set()


class GatherTests(unittest.TestCase):
    """The walk to the island middle and the Qs there."""

    def _bot(self):
        bot = Bot.__new__(Bot)
        bot._patrol_direction = 'right'
        bot._auto_attack_movement = None
        bot._mob_attack_keys = ()
        bot._walls = {}
        bot.command_book = SimpleNamespace(
            dict={'cruelstab': object()},
            module=SimpleNamespace(Key=SimpleNamespace(JUMP='alt', CRUEL_STAB='a')))
        bot._level_of = lambda x, y: 'step'
        bot._wall_ahead = lambda direction, x: False
        bot._map_limit = lambda direction: None
        return bot

    def _run(self, bot, op, phase, ticks, hp, mp, x=lambda tick: 0.55):
        """Ticks PHASE handlers; hp/mp are callables of the tick number."""
        tick = [0]
        capture = SimpleNamespace(
            health=SimpleNamespace(current_percent=lambda: hp(tick[0])),
            mana=SimpleNamespace(current_percent=lambda: mp(tick[0])))
        pressed = []
        with patch.object(bot_module, 'config', SimpleNamespace(capture=capture, enabled=True)), \
                patch.object(bot_module, 'key_down'), patch.object(bot_module, 'key_up'), \
                patch.object(bot_module, 'press', lambda key, *a, **k: pressed.append(key)), \
                patch.object(bot_module.time, 'sleep'):
            for _ in range(ticks):
                now = 100.0 + 0.1 * tick[0]
                handler = getattr(bot, '_level_' + op['phase'])
                handler(op, None, None, x(tick[0]), 0.6, now, 99)
                tick[0] += 1
                if op['phase'] != phase:
                    break
        return pressed

    def _walk_op(self, bot):
        op = {'direction': 'right', 'pressed_x': 0.52, 'kind': 'climb'}
        with patch.object(bot_module, 'config',
                          SimpleNamespace(capture=None, enabled=True)):
            bot._start_aggro(op, 0.53, 100.0)
        op['middle'] = 0.59
        return op

    def test_hit_on_the_way_to_the_middle_uses_q_there_and_keeps_going(self):
        bot = self._bot()
        op = self._walk_op(bot)
        op['hp'] = 90.0
        # moving right, HP drops by 5 at tick 3
        pressed = self._run(bot, op, 'aggro_walk', 6,
                            hp=lambda t: 90.0 if t < 3 else 85.0, mp=lambda t: 80.0,
                            x=lambda t: 0.53 + 0.01 * t)
        self.assertEqual(pressed, ['q'])
        self.assertEqual(op['emergency_qs'], 1)
        self.assertEqual(op['phase'], 'aggro_walk')     # still heading for the middle

    def test_two_confirmed_qs_in_the_middle_then_retreat(self):
        bot = self._bot()
        op = self._walk_op(bot)
        op.update(phase='aggro_q', next_q=100.0)
        mp_level = [80.0]

        def mp(tick):
            return mp_level[0]
        original_press_q = bot._press_q

        def press_q(facing):
            original_press_q(facing)
            mp_level[0] -= 3.0      # every Q costs MP
        bot._press_q = press_q
        pressed = self._run(bot, op, 'aggro_q', 60, hp=lambda t: 90.0, mp=mp)
        self.assertEqual(pressed, ['q', 'q'])
        self.assertEqual(op['q_fired'], 2)
        self.assertEqual(op['phase'], 'aggro_retreat')

    def test_q_with_no_mp_drop_is_pressed_again(self):
        bot = self._bot()
        op = self._walk_op(bot)
        op.update(phase='aggro_q', next_q=100.0)
        pressed = self._run(bot, op, 'aggro_q', 80, hp=lambda t: 90.0, mp=lambda t: 80.0)
        # 2 wanted + 2 retries, then it gives up and leads down what was hit
        self.assertEqual(pressed, ['q'] * (Bot.AGGRO_Q_PRESSES + Bot.AGGRO_Q_RETRIES))
        self.assertEqual(op['phase'], 'aggro_retreat')

    def test_knocked_off_before_any_q_retries_the_gather_soon(self):
        bot = self._bot()
        op = self._walk_op(bot)
        bot._level_of = lambda x, y: 'ground'
        bot._switch_level = lambda level, reason=None: None
        with patch.object(bot_module, 'config',
                          SimpleNamespace(capture=None, enabled=True)), \
                patch.object(bot_module, 'key_down'), patch.object(bot_module, 'key_up'):
            self.assertFalse(bot._knocked_off_gather(op, 0.5, 0.7, 200.0))
            self.assertTrue(bot._knocked_off_gather(
                op, 0.5, 0.7, 200.0 + Bot.LEVEL_SETTLE_SECONDS))
        self.assertAlmostEqual(
            bot._aggro_until, 200.0 + Bot.LEVEL_SETTLE_SECONDS + Bot.AGGRO_RETRY_COOLDOWN)
        self.assertIsNone(bot._level_op)


if __name__ == '__main__':
    unittest.main()


class UnderfootLootTests(unittest.TestCase):
    def test_spams_a_seen_drop_then_pauses_if_it_never_clears(self):
        bot = Bot.__new__(Bot)
        bot._drop_underfoot = lambda: True      # e.g. the feet false-matching
        now, taps = 100.0, []
        while now < 104.0:
            if bot._underfoot_tap_due(now):
                taps.append(now)
                bot._last_drop_tap = now
            now += 0.02
        first_streak = [t for t in taps if t < 100.0 + 2.0]
        self.assertEqual(len(first_streak), Bot.UNDERFOOT_STREAK_MAX)
        self.assertLess(first_streak[-1] - first_streak[0], 1.6)     # player pace
        paused = [t for t in taps if first_streak[-1] < t < first_streak[-1] + 2.5]
        self.assertEqual(paused, [])


class MinimapEdgeMarkerTests(unittest.TestCase):
    """bug-2.mp4: the marker at the map's end sits half outside the map area."""

    def test_a_marker_on_the_border_is_found_and_clamped(self):
        import numpy as np
        from types import SimpleNamespace
        from unittest.mock import patch
        from src.modules import capture as capture_module
        template = capture_module.PLAYER_TEMPLATE
        if template is None:
            self.skipTest('no player template')
        frame = np.full((200, 300, 3), 60, np.uint8)
        h, w = template.shape
        # the minimap is (50, 50)-(250, 150); the marker centred on its right border
        x0, y0 = 250 - w // 2, 120
        frame[y0:y0 + h, x0:x0 + w] = template[:, :, None]
        stub = SimpleNamespace(capture=SimpleNamespace(minimap_ratio=2.0))
        with patch.object(capture_module.utils, 'config', stub):
            inside = capture_module.utils.multi_match(
                frame[50:150, 50:250], template, threshold=0.8)
            found = capture_module.locate_player(frame, (50, 50), (250, 150))
        self.assertEqual(inside, [])            # the old search: lost
        self.assertIsNotNone(found)
        self.assertGreater(found[0], 0.98)
        self.assertLessEqual(found[0], 1.0)
