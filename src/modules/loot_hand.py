"""
The finger on the loot key.

A player holds an arrow (and the attack key) with some fingers and mashes
pickup with another: the taps never interrupt walking or swinging. The bot
used to press the loot key on its own thread with a blocking press (60-150 ms
per tap), so while it spammed pickup, 7-11 times a second, it was hardly
deciding anything else: turns came late and the character overshot drops and
walked into enemies (looting-strike-issue.mp4).

LootHand taps on a thread of its own. The bot only says "tap" and carries on.
Taps are a finger's: each press a slightly different length, gaps never even,
now and then a short hesitation, and never faster than a hand can go. Requests
do not pile up (a tap asked for a second ago is stale), and the key is always
let go, also when Auto Maple is paused or the game loses focus mid-press.
"""

import random
import threading
import time


class LootHand:
    DOWN_SECONDS = (0.035, 0.075)       # how long the key stays down
    UP_SECONDS = (0.045, 0.085)         # and up, before the next tap
    HESITATION_CHANCE = 0.06            # now and then a finger is a bit late
    HESITATION_SECONDS = (0.06, 0.16)
    MAX_PENDING = 2                     # older requests are stale: drop them

    def __init__(self, key_down, key_up, allowed, sleep=time.sleep):
        """
        key_down/key_up: send the key events. allowed(): True while pressing is
        fine (Auto Maple running, game focused); checked before every tap.
        """
        self._key_down = key_down
        self._key_up = key_up
        self._allowed = allowed
        self._sleep = sleep
        self._wake = threading.Condition()
        self._pending = 0
        self._key = None
        self._thread = None
        self.taps_sent = 0

    def tap(self, key, count=1):
        """Asks for COUNT taps of KEY; returns at once."""
        with self._wake:
            self._key = key
            self._pending = min(self.MAX_PENDING, self._pending + max(0, count))
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run, name='loot-hand', daemon=True)
                self._thread.start()
            self._wake.notify()

    def cancel(self):
        """Forgets taps not pressed yet (paused, or fighting for its life)."""
        with self._wake:
            self._pending = 0

    def _run(self):
        while True:
            with self._wake:
                while self._pending <= 0:
                    self._wake.wait()
                self._pending -= 1
                key = self._key
            try:
                if not self._allowed():
                    self.cancel()
                    continue
                self._press_once(key)
            except Exception:
                # A failed key event must never end the hand: the next tap
                # request starts it again anyway, but don't spin on an error
                self._sleep(0.2)

    def _press_once(self, key):
        self._key_down(key)
        try:
            self._sleep(random.uniform(*self.DOWN_SECONDS))
        finally:
            self._key_up(key)
        self.taps_sent += 1
        gap = random.uniform(*self.UP_SECONDS)
        if random.random() < self.HESITATION_CHANCE:
            gap += random.uniform(*self.HESITATION_SECONDS)
        self._sleep(gap)
