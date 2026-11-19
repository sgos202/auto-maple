"""AttackHand: the attack key held like a real key (strike-issue-2.mp4)."""

import threading
import time
import unittest

from src.modules.attack_hand import AttackHand


class AttackHandTests(unittest.TestCase):
    def _hand(self):
        events = []
        lock = threading.Lock()

        def log(kind):
            def send(key):
                with lock:
                    events.append((kind, key))
            return send
        hand = AttackHand(log('down'), log('up'), log('repeat'))
        hand.REPEAT_DELAY = (0.01, 0.01)
        hand.REPEAT_GAP = (0.005, 0.005)
        return hand, events

    def test_held_key_repeats_like_a_keyboard(self):
        hand, events = self._hand()
        hand.hold('ctrl')
        time.sleep(0.1)
        hand.release()
        self.assertEqual(events[0], ('down', 'ctrl'))
        self.assertEqual(events[-1], ('up', 'ctrl'))
        self.assertGreater(sum(1 for e in events if e == ('repeat', 'ctrl')), 3)

    def test_nothing_repeats_after_release(self):
        hand, events = self._hand()
        hand.hold('ctrl')
        time.sleep(0.05)
        hand.release()
        count = len(events)
        time.sleep(0.05)
        self.assertEqual(len(events), count)
        self.assertIsNone(hand.key)

    def test_holding_the_same_key_again_does_nothing(self):
        hand, events = self._hand()
        hand.hold('ctrl')
        hand.hold('ctrl')
        hand.release()
        self.assertEqual([e for e in events if e[0] != 'repeat'],
                         [('down', 'ctrl'), ('up', 'ctrl')])


class LeaseTests(unittest.TestCase):
    """attack-loop.mp4: a hold nobody renews lets go by itself."""

    def _hand(self, lease):
        events = []
        hand = AttackHand(lambda k: events.append(('down', k)),
                          lambda k: events.append(('up', k)),
                          lambda k: events.append(('repeat', k)), lease=lease)
        hand.REPEAT_DELAY = (0.01, 0.01)
        hand.REPEAT_GAP = (0.005, 0.005)
        return hand, events

    def test_an_unrenewed_hold_is_released(self):
        hand, events = self._hand(0.05)
        hand.hold('ctrl')
        time.sleep(0.2)
        self.assertIsNone(hand.key)
        self.assertEqual(events[-1], ('up', 'ctrl'))
        self.assertEqual(hand.expired, 1)

    def test_renewing_keeps_it_held(self):
        hand, events = self._hand(0.08)
        hand.hold('ctrl')
        for _ in range(6):
            time.sleep(0.04)
            hand.hold('ctrl')
        self.assertEqual(hand.key, 'ctrl')
        hand.release()
        self.assertEqual([e for e in events if e[0] != 'repeat'],
                         [('down', 'ctrl'), ('up', 'ctrl')])

    def test_after_expiry_a_new_hold_presses_again(self):
        hand, events = self._hand(0.03)
        hand.hold('ctrl')
        time.sleep(0.15)
        hand.hold('ctrl')
        self.assertEqual(hand.key, 'ctrl')
        hand.release()
        self.assertEqual([e for e in events if e[0] != 'repeat'],
                         [('down', 'ctrl'), ('up', 'ctrl'), ('down', 'ctrl'), ('up', 'ctrl')])


class HoldCapTests(unittest.TestCase):
    """The bot gives up a swing session with no HP bar after HOLD_MAX_SECONDS."""

    def _bot(self, clock):
        from unittest.mock import patch
        from src.modules import bot as bot_module
        bot = bot_module.Bot.__new__(bot_module.Bot)
        bot._human_hold = lambda tracker=None: True
        return bot, bot_module

    def _tracker(self, since_bar):
        calls = []

        class T:
            def status(self):
                return 'Locked enemy'

            def seconds_since_bar(self):
                return since_bar

            def give_up_here(self):
                calls.append('gave up')
        return T(), calls

    def _run(self, since_bar):
        from unittest.mock import patch
        clock = [100.0]
        bot, bot_module = self._bot(clock)
        tracker, calls = self._tracker(since_bar)
        results = []
        with patch.object(bot_module.time, 'monotonic', lambda: clock[0]),                 patch('builtins.print', lambda *a, **k: None):
            for _ in range(70):         # 14 s of passes, with turns in between
                results.append(bot._watch_hold('right', tracker))
                clock[0] += 0.2
        return results, calls

    def test_no_hp_bar_for_long_gives_up(self):
        results, calls = self._run(float('inf'))
        self.assertIn(False, results)
        self.assertEqual(calls, ['gave up'])

    def test_a_real_fight_with_hp_bars_is_left_alone(self):
        results, calls = self._run(0.5)
        self.assertNotIn(False, results)
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
