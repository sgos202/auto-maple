"""
The input layer (src/common/vkeys.py): scan codes, the extended-key flag, the
held-arrow repeat and the shared mouse path. user32 is replaced by a fake, so
no real key or click is ever sent (the bot may be running).
"""

import ctypes
import threading
import time
import unittest
from unittest.mock import patch

from src.common import config, vkeys


class FakeUser32:
    def __init__(self):
        self._real = ctypes.WinDLL('user32')
        self.sent = []          # ('key', vk, scan, flags) / ('mouse', flags)
        self.cursor = []
        self.lock = threading.Lock()

    def MapVirtualKeyExW(self, *args):
        return self._real.MapVirtualKeyExW(*args)

    def SendInput(self, n, ref, size):
        record = ref._obj
        with self.lock:
            if record.type == vkeys.INPUT_KEYBOARD:
                self.sent.append(('key', record.ki.wVk, record.ki.wScan, record.ki.dwFlags))
            else:
                self.sent.append(('mouse', record.mi.dwFlags))
        return 1

    def SetCursorPos(self, x, y):
        self.cursor.append((x, y))
        return 1

    def keys(self):
        with self.lock:
            return [s for s in self.sent if s[0] == 'key']


class InputTest(unittest.TestCase):
    def setUp(self):
        self.user32 = FakeUser32()
        patches = [
            patch.object(vkeys, 'user32', self.user32),
            patch.object(vkeys, 'SCANCODE_MODE', True),
            patch.object(vkeys, 'HOLD_REPEAT', True),
            patch.object(vkeys, 'HOLD_REPEAT_DELAY', (0.03, 0.03)),
            patch.object(vkeys, 'HOLD_REPEAT_GAP', (0.01, 0.01)),
            patch.object(vkeys, '_REPEATER', vkeys._HoldRepeater()),
            patch.object(config, 'enabled', True),
            patch('builtins.print'),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(vkeys.release_all_keys)

    # --- scan codes -------------------------------------------------------
    def test_arrow_is_sent_as_extended_scan_code(self):
        vkeys.key_down('left')
        vkeys.key_up('left')
        down, up = self.user32.keys()[:2]
        self.assertEqual(down, ('key', 0, 0x4B, vkeys.KEYEVENTF_SCANCODE | vkeys.KEYEVENTF_EXTENDEDKEY))
        self.assertEqual(up, ('key', 0, 0x4B, vkeys.KEYEVENTF_SCANCODE | vkeys.KEYEVENTF_EXTENDEDKEY
                              | vkeys.KEYEVENTF_KEYUP))

    def test_letter_is_scan_code_without_extended_flag(self):
        vkeys.press('z', 1, down_time=0, up_time=0)
        down, up = self.user32.keys()
        self.assertEqual(down, ('key', 0, 0x2C, vkeys.KEYEVENTF_SCANCODE))
        self.assertEqual(up[3], vkeys.KEYEVENTF_SCANCODE | vkeys.KEYEVENTF_KEYUP)

    def test_every_mapped_key_has_a_scan_code(self):
        for name in vkeys.KEY_MAP:
            record = vkeys._key_input(name)
            self.assertTrue(record.ki.dwFlags & vkeys.KEYEVENTF_SCANCODE, name)
            self.assertNotEqual(record.ki.wScan, 0, name)

    def test_old_mode_sends_virtual_key_like_before(self):
        with patch.object(vkeys, 'SCANCODE_MODE', False):
            vkeys.press('left', 1, down_time=0, up_time=0)
        down, up = self.user32.keys()
        self.assertEqual(down, ('key', 0x25, 0x4B, 0))
        self.assertEqual(up, ('key', 0x25, 0x4B, vkeys.KEYEVENTF_KEYUP))

    def test_remapped_direction_uses_the_game_key(self):
        with patch.dict(vkeys.HORIZONTAL_MOVEMENT_KEYS, {'left': 'a', 'right': 'd'}):
            vkeys.key_down('left')
            vkeys.key_up('left')
        self.assertEqual(self.user32.keys()[0][2], 0x1E)     # 'a'

    # --- held-arrow repeat ------------------------------------------------
    def test_held_arrow_repeats_until_released(self):
        vkeys.key_down('right')
        time.sleep(0.15)
        vkeys.key_up('right')
        sent = self.user32.keys()
        downs = [s for s in sent if not s[3] & vkeys.KEYEVENTF_KEYUP]
        self.assertGreater(len(downs), 3)
        time.sleep(0.05)
        self.assertEqual(len(self.user32.keys()), len(sent), 'repeat after key-up')
        self.assertTrue(self.user32.keys()[-1][3] & vkeys.KEYEVENTF_KEYUP)

    def test_no_repeat_before_the_delay(self):
        with patch.object(vkeys, 'HOLD_REPEAT_DELAY', (0.5, 0.5)):
            vkeys.key_down('right')
            time.sleep(0.1)
            vkeys.key_up('right')
        self.assertEqual(len(self.user32.keys()), 2)

    def test_other_key_stops_the_arrow_repeat_like_a_keyboard(self):
        vkeys.key_down('right')
        time.sleep(0.08)
        vkeys.key_down('x')         # e.g. the attack key
        count = len(self.user32.keys())
        time.sleep(0.08)
        self.assertEqual(len(self.user32.keys()), count)
        vkeys.key_up('x')
        vkeys.key_up('right')

    def test_non_arrow_keys_do_not_repeat(self):
        vkeys.key_down('x')
        time.sleep(0.1)
        vkeys.key_up('x')
        self.assertEqual(len(self.user32.keys()), 2)

    def test_no_repeat_while_paused(self):
        vkeys.key_down('right')
        config.enabled = False
        count = len(self.user32.keys())
        time.sleep(0.1)
        self.assertEqual(len(self.user32.keys()), count)
        vkeys.key_up('right')

    def test_resent_key_down_does_not_restart_the_delay(self):
        with patch.object(vkeys, 'HOLD_REPEAT_DELAY', (0.06, 0.06)):
            vkeys.key_down('right')
            for _ in range(6):
                time.sleep(0.02)
                vkeys.key_down('right')      # bot.py re-sends to be sure
            vkeys.key_up('right')
        # 7 bot key-downs + 1 key-up; any extra came from the repeater
        self.assertGreater(len(self.user32.keys()), 8)

    def test_repeat_can_be_switched_off(self):
        with patch.object(vkeys, 'HOLD_REPEAT', False):
            vkeys.key_down('right')
            time.sleep(0.1)
            vkeys.key_up('right')
        self.assertEqual(len(self.user32.keys()), 2)

    # --- mouse ------------------------------------------------------------
    def test_click_moves_then_presses_and_releases(self):
        vkeys.click((10, 20), button='right')
        self.assertEqual(self.user32.cursor, [(10, 20)])
        self.assertEqual(self.user32.sent, [('mouse', vkeys.MOUSEEVENTF_RIGHTDOWN),
                                            ('mouse', vkeys.MOUSEEVENTF_RIGHTUP)])

    def test_click_is_gated_by_pause_unless_allowed(self):
        config.enabled = False
        vkeys.click((1, 2))
        self.assertEqual(self.user32.sent, [])
        vkeys.click((1, 2), allow_paused=True)
        self.assertEqual(len(self.user32.sent), 2)

    def test_click_without_position_keeps_the_cursor(self):
        vkeys.click(None, allow_paused=True)
        self.assertEqual(self.user32.cursor, [])
        self.assertEqual(len(self.user32.sent), 2)

    def test_release_all_lets_go_of_a_held_button(self):
        vkeys.mouse_down('left')
        released = vkeys.release_all_keys()
        self.assertIn('mouse left', released)
        self.assertEqual(self.user32.sent[-1], ('mouse', vkeys.MOUSEEVENTF_LEFTUP))

    def test_call_sites_use_the_shared_path(self):
        from src.detection import violetta_solver
        with patch('time.sleep'):
            violetta_solver._click(5, 6)
        self.assertEqual(self.user32.cursor, [(5, 6)])
        self.assertEqual(len(self.user32.sent), 2)


if __name__ == '__main__':
    unittest.main()
