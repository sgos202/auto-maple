"""The phone remote (phone_remote.py) and Auto Maple's side of it (remote_link.py)."""

import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch

from src.modules import phone_remote, remote_link


class InboxTests(unittest.TestCase):
    def test_a_posted_command_is_taken_once(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(remote_link, 'INBOX', folder):
            remote_link.post('go', folder=folder)
            remote_link.post('pause', folder=folder)
            self.assertEqual(remote_link._take_commands(), ['go', 'pause'])
            self.assertEqual(remote_link._take_commands(), [])

    def test_old_commands_are_dropped(self):
        # sent while Auto Maple was closed: must not fire when it starts later
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(remote_link, 'INBOX', folder):
            with open(os.path.join(folder, '1.json'), 'w') as file:
                json.dump({'command': 'go', 'time': time.time() - 600}, file)
            self.assertEqual(remote_link._take_commands(), [])
            self.assertEqual(os.listdir(folder), [])

    def test_state_can_be_read_outside_the_bot(self):
        info = remote_link.state()
        self.assertIn('pid', info)
        self.assertIn('enabled', info)


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.replies = []
        patches = [
            patch.object(phone_remote, 'reply', lambda text, image=None: self.replies.append(text)),
            patch.object(phone_remote, 'desktop_screenshot', lambda: b'jpg'),
            patch.object(phone_remote, 'log', lambda text: None),
            patch.object(phone_remote, 'start_auto_maple'),
            patch.object(phone_remote, 'stop_auto_maple', return_value=True),
            patch.object(phone_remote, 'wait_for_start', return_value={'pid': 2}),
            patch.object(phone_remote.remote_link, 'post'),
            patch.object(phone_remote, 'auto_maple_pids', return_value=([], [])),
            patch.object(phone_remote, '_game_running', return_value=True),
            patch.object(phone_remote.time, 'sleep'),
        ]
        self.mocks = [p.start() for p in patches]
        self.addCleanup(patch.stopall)

    def running(self, state):
        return patch.object(phone_remote, 'running_state', return_value=state)

    def test_start_when_not_running(self):
        with self.running(None):
            phone_remote.handle('/start')
        phone_remote.start_auto_maple.assert_called_once()
        self.assertTrue(any('is up' in r for r in self.replies))

    def test_start_when_already_running_does_nothing(self):
        with self.running({'pid': 1, 'started': time.time()}):
            phone_remote.handle('/start')
        phone_remote.start_auto_maple.assert_not_called()

    def test_restart_stops_then_starts(self):
        with self.running({'pid': 1, 'started': time.time()}):
            phone_remote.handle('/restart')
        phone_remote.stop_auto_maple.assert_called_once()
        phone_remote.start_auto_maple.assert_called_once()

    def test_go_pause_and_game_restart_are_passed_to_auto_maple(self):
        with self.running({'pid': 1, 'started': time.time()}):
            for text, command in (('/go', 'go'), ('/pause', 'pause'),
                                  ('/restartgame', 'restart_game'), ('/restart_game', 'restart_game')):
                phone_remote.handle(text)
                phone_remote.remote_link.post.assert_called_with(command)

    def test_go_needs_auto_maple_running(self):
        with self.running(None):
            phone_remote.handle('/go')
        phone_remote.remote_link.post.assert_not_called()
        self.assertIn('/start', self.replies[-1])

    def test_status_says_what_auto_maple_is_doing(self):
        with self.running({'pid': 1, 'started': time.time() - 3700, 'enabled': True,
                           'map': 'The Pig Beach', 'free_mb': 20000, 'limit_mb': 55000}):
            phone_remote.handle('/status')
        self.assertIn('farming', self.replies[-1])
        self.assertIn('The Pig Beach', self.replies[-1])
        self.assertIn('1 h 01 min', self.replies[-1])
        with self.running(None):
            phone_remote.handle('status')
        self.assertIn('NOT running', self.replies[-1])

    def test_anything_else_lists_the_commands(self):
        with self.running(None):
            phone_remote.handle('hello')
        self.assertIn('/restart', self.replies[-1])


if __name__ == '__main__':
    unittest.main()
