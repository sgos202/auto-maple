"""
Holds the attack key the way a person does (strike-issue-2.mp4).

The bot used to press the attack key once and hold the arrow with it. The
game walked the character toward the enemy between swings, into it, and the
swing often whiffed. With the attack key alone held through SendInput the game
swung only once (spawn_issue.mp4), because a single key-down is not what a real
keyboard sends: a held key repeats its key-down about 30 times a second after a
short delay. AttackHand does the same on its own thread, so the character
stands still and keeps swinging while the bot decides what comes next.
"""

import random
import threading
import time


class AttackHand:
    REPEAT_DELAY = (0.22, 0.30)         # s after the press before repeats start
    REPEAT_GAP = (0.028, 0.040)         # s between repeats (~30 per second)

    def __init__(self, key_down, key_up, key_repeat, sleep=time.sleep, lease=None,
                 clock=time.monotonic):
        # LEASE: s a hold lasts unless hold() is called again (attack-loop.mp4:
        # the bot stopped asking for the attack but the repeats went on, and
        # the character swung at the portal for good with nothing in the log).
        # None: held until release().
        self._lease = lease
        self._clock = clock
        self._expires = None
        self.expired = 0
        self._key_down = key_down
        self._key_up = key_up
        self._key_repeat = key_repeat
        self._sleep = sleep
        self._lock = threading.Lock()
        self._key = None
        self._serial = 0
        self.repeats_sent = 0

    @property
    def key(self):
        with self._lock:
            return self._key

    def hold(self, key):
        """Presses KEY and keeps it repeating until release() (or until the
        lease runs out; calling hold() again with the same key renews it).
        Returns at once."""
        with self._lock:
            if self._lease is not None:
                self._expires = self._clock() + self._lease
            if self._key == key:
                return
            old = self._key
            self._key = key
            self._serial += 1
            serial = self._serial
            if old is not None:
                self._key_up(old)
            self._key_down(key)
        threading.Thread(target=self._run, args=(key, serial),
                         name='attack-hand', daemon=True).start()

    def release(self):
        """Lets go of the held key (if any)."""
        with self._lock:
            key, self._key = self._key, None
            self._serial += 1
            if key is not None:
                self._key_up(key)

    def _run(self, key, serial):
        self._sleep(random.uniform(*self.REPEAT_DELAY))
        while True:
            with self._lock:
                if serial != self._serial:
                    return
                if self._expires is not None and self._clock() >= self._expires:
                    # nobody renewed it: let go (as release() does)
                    self._key = None
                    self._serial += 1
                    self.expired += 1
                    self._key_up(key)
                    return
                # (inside the lock: a release can't slip in between the check
                # and the send, which would leave the key down in the game)
                self._key_repeat(key)
                self.repeats_sent += 1
            self._sleep(random.uniform(*self.REPEAT_GAP))
