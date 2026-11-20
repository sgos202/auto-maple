import os
import tempfile
import unittest
from unittest.mock import patch

from src.common import config
from src.modules import buffs


class BuffManagerTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, 'buffs.json')
        self.manager = buffs.BuffManager(self.path)

    def tearDown(self):
        self.dir.cleanup()

    def test_add_and_reload(self):
        self.manager.add('Rage', 'q', 180, 2)
        again = buffs.BuffManager(self.path)
        [buff] = again.list()
        self.assertEqual((buff['name'], buff['key'], buff['interval'], buff['presses']),
                         ('Rage', 'q', 180, 2))

    def test_bad_input_is_refused(self):
        for args in (('', 'q', 60), ('Rage', 'notakey', 60), ('Rage', 'q', 1),
                     ('Rage', 'q', 'soon'), ('Rage', 'q', 60, 9)):
            with self.assertRaises(ValueError, msg=args):
                self.manager.add(*args)

    def test_due_at_start_then_after_its_interval(self):
        rage = self.manager.add('Rage', 'q', 180)
        self.assertEqual([b['id'] for b in self.manager.due(now=100)], [rage])
        self.manager.mark_cast(rage, now=100)
        self.assertEqual(self.manager.due(now=279), [])
        self.assertEqual(self.manager.seconds_left(rage, now=200), 80)
        self.assertEqual([b['id'] for b in self.manager.due(now=280)], [rage])

    def test_most_overdue_first_and_disabled_skipped(self):
        a = self.manager.add('A', 'q', 60)
        b = self.manager.add('B', 'w', 60)
        c = self.manager.add('C', 'e', 60)
        self.manager.mark_cast(a, now=0)
        self.manager.mark_cast(b, now=-50)
        self.manager.update(c, enabled=False)
        self.assertEqual([x['id'] for x in self.manager.due(now=100)], [b, a])
        self.manager.set_enabled(False)
        self.assertEqual(self.manager.due(now=100), [])

    def test_a_failed_cast_waits_before_retrying(self):
        a = self.manager.add('A', 'q', 60)
        self.manager.mark_failed(a, now=10)
        self.assertEqual(self.manager.due(now=11), [])
        self.assertEqual(len(self.manager.due(now=10 + buffs.RETRY_SECONDS)), 1)

    def test_service_casts_due_buffs_when_safe(self):
        self.manager.add('Rage', 'q', 180, 2)
        let_go = []
        with patch.object(buffs, '_safe_to_cast', return_value=True), \
                patch.object(buffs.vkeys, 'press') as press, \
                patch.object(buffs.time, 'sleep'):
            self.assertEqual(self.manager.service(lambda: let_go.append(1)), 1)
            press.assert_called_once_with('q', 2)
            self.assertEqual(let_go, [1])
            self.assertEqual(self.manager.service(), 0)        # not due again yet

    def test_nothing_is_cast_while_stopped_unfocused_or_in_a_mini_game(self):
        self.manager.add('Rage', 'q', 180)
        with patch.object(buffs.vkeys, 'press') as press:
            with patch.object(config, 'enabled', False):
                self.assertEqual(self.manager.service(), 0)
            with patch.object(config, 'enabled', True), \
                    patch('src.modules.health.is_game_focused', return_value=False):
                self.assertEqual(self.manager.service(), 0)
            press.assert_not_called()

    def test_format_duration(self):
        self.assertEqual(buffs.format_duration(45), '45s')
        self.assertEqual(buffs.format_duration(185), '3m 05s')
        self.assertEqual(buffs.format_duration(3720), '1h 02m')


if __name__ == '__main__':
    unittest.main()
