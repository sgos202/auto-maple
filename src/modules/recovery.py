"""
Gets the character back onto its patrol platform after it is knocked off.

The bot only knows the character's position on the minimap (x and y from 0 to 1,
y grows downwards). "Home" is the platform the patrol runs on. When the character
ends up below it for more than a moment, LevelRecovery walks along whatever
platform it landed on while holding Up, hopping now and then. Holding Up makes the
character grab any rope or ladder it passes, so no rope positions need to be
configured. Once it is climbing it keeps holding Up to the top, then checks the
level again and repeats until it is home (a fall can be more than one level).

Ropes that worked are remembered, so the next recovery walks straight to one.
The sweep stays inside the home platform's horizontal range: ropes up to a platform
hang beneath it, and portals (which Up would enter) sit at the far ends of a map.
"""

import time


class LevelRecovery:
    FALL_TOLERANCE = 0.06       # this far below home (minimap units) counts as fallen
    FALL_SECONDS = 1.0          # ...when it lasts this long (not a knock-back or a hop)
    CLIMB_RISE = 0.035          # risen this far above the landing level: on a rope
    CLIMB_SECONDS = 0.45        # ...for this long (a hop is shorter)
    CLIMB_STALL_SECONDS = 0.7   # climbing ends when the height stops changing
    CLIMB_TIMEOUT = 10.0
    HOP_INTERVAL = 0.9          # some ropes end above head height
    STUCK_SECONDS = 1.5         # not moving sideways: wall or edge, turn round
    RANGE_MARGIN = 0.03
    MIN_RANGE = 0.25
    TIMEOUT = 60.0
    RETRY_SECONDS = 15.0
    TICK = 0.05

    def __init__(self, key_down, key_up, press, position, jump_key='alt',
                 clock=time.monotonic, sleep=time.sleep, active=lambda: True,
                 log=print):
        self._key_down = key_down
        self._key_up = key_up
        self._press = press
        self._position = position       # () -> (x, y) on the minimap
        self.jump_key = jump_key
        self._clock = clock
        self._sleep = sleep
        self._active = active           # () -> False to abort (bot paused)
        self._log = log
        self.home_y = None
        self.range = None               # (x_low, x_high) of the home platform
        self.ropes = []                 # (x, level_y) of ropes that led upwards
        self._below_since = None
        self._last_failure = None
        self._held = set()

    # ------------------------------------------------------------------ home
    def set_home(self, y, x_low=None, x_high=None):
        self.home_y = y
        self.range = None if x_low is None else (min(x_low, x_high), max(x_low, x_high))
        self._below_since = None

    def clear_home(self):
        self.home_y = None
        self.range = None
        self._below_since = None

    def observe(self, position):
        """Call while patrolling: learns how far the home platform reaches."""
        if self.home_y is None or position is None:
            return
        x, y = position
        if abs(y - self.home_y) > self.FALL_TOLERANCE:
            return
        if self.range is None:
            self.range = (x, x)
        else:
            self.range = (min(self.range[0], x), max(self.range[1], x))

    def fallen(self, position=None):
        """True once the character has been below home for FALL_SECONDS."""
        if self.home_y is None:
            return False
        position = position or self._position()
        if position is None:
            return False
        now = self._clock()
        if position[1] - self.home_y <= self.FALL_TOLERANCE:
            self._below_since = None
            return False
        if self._below_since is None:
            self._below_since = now
        if self._last_failure is not None and now - self._last_failure < self.RETRY_SECONDS:
            return False
        return now - self._below_since >= self.FALL_SECONDS

    # ------------------------------------------------------------------ keys
    def _hold(self, key):
        if key not in self._held:
            self._key_down(key)
            self._held.add(key)

    def _release(self, key=None):
        for held in ([key] if key else list(self._held)):
            if held in self._held:
                self._key_up(held)
                self._held.discard(held)

    def _bounds(self, x):
        low, high = self.range if self.range else (x, x)
        if high - low < self.MIN_RANGE:
            middle = (low + high) / 2
            low, high = middle - self.MIN_RANGE / 2, middle + self.MIN_RANGE / 2
        return max(0.03, low - self.RANGE_MARGIN), min(0.97, high + self.RANGE_MARGIN)

    # ------------------------------------------------------------------ recover
    def recover(self):
        """Blocks until the character is back on the home level. Returns success."""
        started = self._clock()
        self._log('\n[~] Knocked off the patrol platform; climbing back.')
        try:
            while self._active() and self._clock() - started < self.TIMEOUT:
                x, y = self._position()
                offset = y - self.home_y
                if abs(offset) <= self.FALL_TOLERANCE:
                    self._log('[~] Back on the patrol platform.')
                    self._below_since = None
                    self._last_failure = None
                    return True
                if offset < 0:
                    self._drop_down()
                else:
                    self._climb_from(x, y, started)
        finally:
            self._release()
        if self._active():
            self._last_failure = self._clock()
            self._log('[!] Could not get back to the patrol platform; will retry shortly.')
        return False

    def _drop_down(self):
        """Climbed past home: drop through the platform."""
        self._hold('down')
        self._sleep(0.1)
        self._press(self.jump_key, 1)
        self._release('down')
        self._sleep(0.8)

    def _climb_from(self, x, level_y, started):
        """Sweeps the current platform holding Up until a rope is grabbed and climbed."""
        self._release()
        low, high = self._bounds(x)
        known = [rope_x for rope_x, rope_y in self.ropes
                 if abs(rope_y - level_y) <= self.FALL_TOLERANCE and low <= rope_x <= high]
        if known:
            goal = min(known, key=lambda rope_x: abs(rope_x - x))
            direction = 'left' if goal < x else 'right'
        else:   # head for the middle of the home platform first
            direction = 'left' if x > (low + high) / 2 else 'right'
        self._hold('up')
        self._hold(direction)
        last_hop = self._clock()
        last_x, last_moved = x, self._clock()
        rising_since = None
        while self._active() and self._clock() - started < self.TIMEOUT:
            self._sleep(self.TICK)
            now = self._clock()
            x, y = self._position()
            if y - self.home_y <= self.FALL_TOLERANCE and level_y - y < self.CLIMB_RISE:
                return      # walked or was carried back onto the home level
            if level_y - y >= self.CLIMB_RISE:
                rising_since = rising_since or now
                if now - rising_since >= self.CLIMB_SECONDS:
                    self._release(direction)
                    self._finish_climb(x, level_y)
                    return
            else:
                rising_since = None
                if y - level_y > self.FALL_TOLERANCE:
                    return  # fell further; start again from the new level
            if abs(x - last_x) > 0.004:
                last_x, last_moved = x, now
            turn = None
            if direction == 'left' and x <= low:
                turn = 'right'
            elif direction == 'right' and x >= high:
                turn = 'left'
            elif now - last_moved > self.STUCK_SECONDS and rising_since is None:
                turn = 'left' if direction == 'right' else 'right'
                last_moved = now
            if turn:
                self._release(direction)
                direction = turn
                self._hold(direction)
            if now - last_hop >= self.HOP_INTERVAL and rising_since is None:
                self._press(self.jump_key, 1)
                last_hop = now

    def _finish_climb(self, rope_x, level_y):
        """On a rope: keep holding Up until the height stops changing."""
        if not any(abs(rope_x - known_x) < 0.03 and abs(level_y - known_y) < self.FALL_TOLERANCE
                   for known_x, known_y in self.ropes):
            self.ropes.append((rope_x, level_y))
        began = self._clock()
        last_y, last_change = self._position()[1], began
        while self._active() and self._clock() - began < self.CLIMB_TIMEOUT:
            self._sleep(self.TICK)
            y = self._position()[1]
            now = self._clock()
            if abs(y - last_y) > 0.003:
                last_y, last_change = y, now
            elif now - last_change >= self.CLIMB_STALL_SECONDS:
                break
        self._release('up')
        self._sleep(0.2)
