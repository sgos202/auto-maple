import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.modules import memory_guard as guard_module
from src.modules.memory_guard import MemoryGuard, top_memory_users


class MemoryGuardTests(unittest.TestCase):
    def _guard(self):
        guard = MemoryGuard()
        guard._started = 0.0
        guard._last_report = 1e9        # no periodic report lines in tests
        return guard

    def _check(self, guard, config, free_mb, now, focused=True):
        with patch.object(guard_module, 'config', config), \
                patch.object(guard_module, 'system_commit', lambda: (free_mb, 33000.0)), \
                patch.object(guard_module, 'own_memory_mb', lambda: 450.0), \
                patch.object(guard_module, 'top_memory_users',
                             lambda count=5: [('chrome', 2800.0, 21)]), \
                patch('src.modules.health.is_game_focused', lambda: focused), \
                patch('src.common.utils.print_state'), \
                patch('builtins.print'):
            guard.check(now)

    def test_pauses_when_windows_runs_out_and_resumes_by_itself(self):
        # 16:52 log: paused at 754 MB free and stayed paused until a key press
        guard, config = self._guard(), SimpleNamespace(enabled=True)
        self._check(guard, config, 2000, 10)
        self.assertTrue(config.enabled)
        self._check(guard, config, guard.PAUSE_FREE_MB - 50, 25)
        self.assertFalse(config.enabled)
        self._check(guard, config, guard.RESUME_FREE_MB - 100, 30)     # not enough yet
        self.assertFalse(config.enabled)
        self._check(guard, config, guard.RESUME_FREE_MB + 100, 35)     # one good check
        self.assertFalse(config.enabled)
        self._check(guard, config, guard.RESUME_FREE_MB + 100, 40)     # holds: go
        self.assertTrue(config.enabled)

    def test_a_dip_between_good_checks_starts_the_count_again(self):
        guard, config = self._guard(), SimpleNamespace(enabled=True)
        self._check(guard, config, 100, 10)
        self._check(guard, config, guard.RESUME_FREE_MB + 100, 15)
        self._check(guard, config, guard.PAUSE_FREE_MB + 50, 20)
        self._check(guard, config, guard.RESUME_FREE_MB + 100, 25)
        self.assertFalse(config.enabled)

    def test_does_not_resume_what_the_user_stopped(self):
        guard, config = self._guard(), SimpleNamespace(enabled=True)
        self._check(guard, config, 100, 10)
        self.assertFalse(config.enabled)
        guard.user_toggled()            # Start/stop pressed (started, then stopped)
        for now in (15, 20, 25, 30):
            self._check(guard, config, 5000, now)
        self.assertFalse(config.enabled)

    def test_never_starts_a_bot_that_was_not_running(self):
        guard, config = self._guard(), SimpleNamespace(enabled=False)
        self._check(guard, config, 100, 10)
        for now in (15, 20, 25):
            self._check(guard, config, 5000, now)
        self.assertFalse(config.enabled)

    def test_waits_for_the_game_window_before_resuming(self):
        guard, config = self._guard(), SimpleNamespace(enabled=True)
        self._check(guard, config, 100, 10)
        for now in (15, 20, 25):
            self._check(guard, config, 5000, now, focused=False)
        self.assertFalse(config.enabled)
        self._check(guard, config, 5000, 30, focused=True)
        self.assertTrue(config.enabled)

    def test_lists_the_programs_holding_memory(self):
        users = top_memory_users(5)
        self.assertTrue(users)
        self.assertEqual(users, sorted(users, key=lambda user: user[1], reverse=True))
        for name, used_mb, processes in users:
            self.assertTrue(name)
            self.assertGreater(used_mb, 0)
            self.assertLess(used_mb, 512 * 1024)
            self.assertGreaterEqual(processes, 1)


if __name__ == '__main__':
    unittest.main()


class GameLeakWarningTests(unittest.TestCase):
    """2026-11-15: the MapleStory client leaked ~72k handles an hour."""

    def _check(self, guard, handles, now, printed):
        config = SimpleNamespace(enabled=True)
        with patch.object(guard_module, 'config', config), \
                patch.object(guard_module, 'system_commit', lambda: (9000.0, 33000.0)), \
                patch.object(guard_module, 'own_memory_mb', lambda: 170.0), \
                patch.object(guard_module, 'game_handles', lambda: handles), \
                patch('builtins.print', lambda *a, **k: printed.append(' '.join(map(str, a)))):
            guard.check(now)

    def test_a_leaking_client_is_reported_with_its_rate_once_per_window(self):
        guard = MemoryGuard()
        guard._started = 0.0
        guard._last_report = 0.0
        printed = []
        self._check(guard, 50000, 10.0, printed)
        self._check(guard, 62000, 610.0, printed)     # +12k in 10 min
        self._check(guard, 62500, 620.0, printed)     # within the window: quiet
        warnings = [p for p in printed if 'leaking memory' in p]
        self.assertEqual(len(warnings), 1, printed)
        self.assertIn('62,000 handles', warnings[0])
        self.assertIn('+72,000/hour', warnings[0])

    def test_the_real_process_table_reads(self):
        self.assertIsInstance(guard_module._process_table(), list)


class PausedStatusTests(unittest.TestCase):
    """2026-11-15: paused from 04:50 to 11:56 while 'Memory: ...' kept coming
    every 10 min - it read like a memory problem, and nobody was told."""

    def _check(self, guard, config, now, printed, sent):
        with patch.object(guard_module, 'config', config), \
                patch.object(guard_module, 'system_commit', lambda: (19000.0, 41930.0)), \
                patch.object(guard_module, 'own_memory_mb', lambda: 240.0), \
                patch.object(guard_module, 'game_handles', lambda: 30750), \
                patch('src.modules.phone_alert.wants', lambda kind: True), \
                patch('src.modules.phone_alert.send', lambda text, photo=False: sent.append(text)), \
                patch('builtins.print', lambda *a, **k: printed.append(' '.join(map(str, a)))):
            guard.check(now)

    def test_a_long_pause_is_said_plainly_alerted_once_and_reported_hourly(self):
        guard = MemoryGuard()
        guard._started = 0.0
        config = SimpleNamespace(enabled=True,
                                 listener=SimpleNamespace(config={'Start/stop': 'f3'}))
        printed, sent = [], []
        self._check(guard, config, 600.0, printed, sent)
        self.assertIn('Status: farming. Memory OK', printed[-1])
        config.enabled = False                      # switched off at t=700
        for now in range(700, 700 + 3 * 3600, 15):
            self._check(guard, config, float(now), printed, sent)
        statuses = [p for p in printed if 'Status:' in p]
        self.assertEqual(len(statuses), 1 + 3, statuses)    # hourly while paused
        self.assertIn('PAUSED for 58 min', statuses[1])   # first hourly line
        self.assertIn('Press F3', statuses[1])
        idle = [p for p in printed if 'has been paused for' in p]
        self.assertEqual(len(idle), 1, printed)
        self.assertEqual(len(sent), 1)
        config.enabled = True                       # started again: re-armed
        self._check(guard, config, 20000.0, printed, sent)
        config.enabled = False
        for now in range(20015, 20015 + 700, 15):
            self._check(guard, config, float(now), printed, sent)
        self.assertEqual(len(sent), 2)

    def test_never_started_is_not_an_idle_alert(self):
        guard = MemoryGuard()
        guard._started = 0.0
        config = SimpleNamespace(enabled=False)
        printed, sent = [], []
        for now in range(4000, 5800, 15):
            self._check(guard, config, float(now), printed, sent)
        self.assertFalse(sent)
        self.assertIn('not started yet', [p for p in printed if 'Status:' in p][0])
