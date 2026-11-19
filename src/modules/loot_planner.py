"""
Decides when to press the loot key, from where drops actually fall rather than
from seeing them.

Seeing loot under the character is unreliable (its own feet and shield match the
drop samples about as well as real drops do, and many drops have no sample), but
pressing pickup on bare ground does nothing. So drops are expected where the
bot fought and killed, and the key is pressed there:

  * right after a kill, a short burst while the drops land;
  * now and then between swings while fighting (drops pile up underfoot);
  * while walking over a remembered fight spot that has not been swept yet;
  * and, with nothing to fight, the bot walks to the nearest such spot.

Positions are minimap x (0..1) plus the level ('ground', 'step') the character
was on, so a spot on the step is never "swept" from the ground beneath it.

Like a player, the key is spammed over a pile (7-11 taps a second, never
evenly spaced), and how long depends on how much fell there: each kill adds
taps (each tap picks up one item). Guards: a cap per spot, a minimum gap, and
spots expire.
"""

import random


class DropSpot:
    __slots__ = ('x', 'level', 'created', 'last_fight', 'taps', 'swept',
                 'collect_since', 'kills')

    def __init__(self, x, level, now):
        self.x = x
        self.level = level
        self.created = now
        self.last_fight = now
        self.taps = 0
        self.swept = False
        self.collect_since = None
        self.kills = 0


class LootPlanner:
    SPOT_RADIUS = 0.012         # drops scatter this far around a fight (minimap)
    MERGE_RADIUS = 0.012        # fights this close together share one spot
    SPOT_TTL = 90.0             # drops vanish after a while anyway
    MAX_SPOTS = 16
    DROP_SETTLE = 0.3           # drops are still falling right after a fight
    BURST_DELAYS = (0.35, 0.5, 0.65, 0.82, 1.0)  # after a kill: taps while drops land
    BURST_JITTER = 0.04
    FIGHT_TAP_INTERVAL = 1.0    # while swinging: one tap this often
    SPAM_INTERVAL = (0.09, 0.15)    # over an unswept spot: a player's spamming pace
    WALK_TAP_INTERVAL = SPAM_INTERVAL[1]    # (slowest spam gap; kept for callers)
    MIN_TAP_GAP = 0.08          # never faster than this (bursts excepted)
    SWEEP_TAPS = 3              # taps that sweep a spot where nothing was killed
    TAPS_PER_KILL = 2           # each kill there drops ~1-3 items: tap more
    MAX_SPOT_TAPS = 12          # guard: a spot is never tapped more than this
    COLLECT_REACH = 0.2         # walk at most this far to collect a spot
    COLLECT_TIMEOUT = 5.0       # can't get there this long: give it up

    def __init__(self):
        self.spots = []
        self.burst = []
        self.last_tap = float('-inf')
        self.taps_sent = 0
        self.next_gap = self._spam_gap()

    @classmethod
    def _spam_gap(cls):
        return random.uniform(*cls.SPAM_INTERVAL)

    def taps_needed(self, spot):
        return min(self.MAX_SPOT_TAPS, self.SWEEP_TAPS + self.TAPS_PER_KILL * spot.kills)

    # ---- events ----

    def fought(self, now, x, level):
        """The bot is swinging at an enemy at minimap X: drops will fall here."""
        if x is None:
            return
        spot = self._spot_near(x, level, self.MERGE_RADIUS)
        if spot is None:
            spot = DropSpot(x, level, now)
            self.spots.append(spot)
            if len(self.spots) > self.MAX_SPOTS:
                self.spots.pop(0)
        spot.last_fight = now
        spot.swept = False
        spot.taps = 0
        spot.collect_since = None

    def killed(self, now, x, level):
        """An enemy died: tap a few times as its drops land."""
        self.fought(now, x, level)
        spot = self._spot_near(x, level, self.MERGE_RADIUS) if x is not None else None
        if spot is not None:
            spot.kills += 1
        self.burst = sorted(set(self.burst) | {
            now + d + random.uniform(-self.BURST_JITTER, self.BURST_JITTER)
            for d in self.BURST_DELAYS})

    def seen(self, now):
        """The loot detector saw something underfoot: tap at once."""
        self.burst = sorted(set(self.burst) | {now, now + 0.2})

    # ---- decisions ----

    def taps_due(self, now, x, level, attacking):
        """
        How many loot taps to send this tick (0 or 1), and the spot just swept
        by it (for the log) or None.
        """
        self._expire(now)
        if self.burst and self.burst[0] <= now:
            while self.burst and self.burst[0] <= now:
                self.burst.pop(0)
            return self._tap(now, self._spot_near(x, level, self.SPOT_RADIUS))
        if now - self.last_tap < self.MIN_TAP_GAP:
            return 0, None
        if attacking:
            if now - self.last_tap >= self.FIGHT_TAP_INTERVAL:
                return self._tap(now, None)
            return 0, None
        if self.burst:
            return 0, None      # a kill's burst is still coming: that covers it
        spot = self._unswept_near(now, x, level, self.SPOT_RADIUS)
        if spot is not None and now - self.last_tap >= self.next_gap:
            return self._tap(now, spot)
        return 0, None

    def collect_direction(self, now, x, level):
        """
        With nothing to fight: 'left'/'right' toward the nearest unswept spot in
        reach, 'stay' when standing on it, or None when there is none.
        """
        if x is None:
            return None
        self._expire(now)
        candidates = [
            spot for spot in self._ready(now, level)
            if abs(spot.x - x) <= self.COLLECT_REACH
        ]
        if not candidates:
            self.sweep_side = None
            return None
        spot = self._next_in_sweep(candidates, x)
        if spot.collect_since is None:
            spot.collect_since = now
        elif now - spot.collect_since >= self.COLLECT_TIMEOUT:
            spot.swept = True       # blocked or on another level after all
            return self.collect_direction(now, x, level)
        if abs(spot.x - x) <= self.SPOT_RADIUS * 0.5:
            return 'stay'
        return 'left' if spot.x < x else 'right'

    def _next_in_sweep(self, candidates, x):
        """
        The spot to walk to next. Going to whichever is nearest zig-zags
        through a row of spots (0.203 -> 0.189 -> 0.217 -> 0.231 in
        looting-strike-issue.mp4: five turns for four spots). Like a player:
        finish the one underfoot, walk to the nearer end of the row, then
        sweep straight through to the other end.
        """
        underfoot = [s for s in candidates if abs(s.x - x) <= self.SPOT_RADIUS * 0.5]
        if underfoot:
            return min(underfoot, key=lambda s: abs(s.x - x))
        left = [s for s in candidates if s.x < x]
        right = [s for s in candidates if s.x >= x]
        side = getattr(self, 'sweep_side', None)
        if side == 'left' and not left or side == 'right' and not right:
            side = None
        if side is None:
            if not left or not right:
                side = 'left' if left else 'right'
            else:   # the end of the row that is closer: less walking overall
                far_left = x - min(s.x for s in left)
                far_right = max(s.x for s in right) - x
                side = 'left' if far_left <= far_right else 'right'
        self.sweep_side = side
        return min(left if side == 'left' else right, key=lambda s: abs(s.x - x))

    def give_up(self, now, x, level, direction):
        """The way toward the spots in DIRECTION is blocked (a wall): drop them."""
        for spot in self._ready(now, level):
            if x is not None and (spot.x < x if direction == 'left' else spot.x > x):
                spot.swept = True

    def pending(self, now, level):
        self._expire(now)
        return len(self._ready(now, level))

    # ---- helpers ----

    def _tap(self, now, spot):
        self.last_tap = now
        self.taps_sent += 1
        self.next_gap = self._spam_gap()
        swept = None
        if spot is not None and not spot.swept:
            spot.taps += 1
            if spot.taps >= self.taps_needed(spot):
                spot.swept = True
                swept = spot
        return 1, swept

    def _ready(self, now, level):
        return [
            spot for spot in self.spots
            if not spot.swept and spot.level == level
            and now - spot.last_fight >= self.DROP_SETTLE
        ]

    def _unswept_near(self, now, x, level, radius):
        if x is None:
            return None
        near = [s for s in self._ready(now, level) if abs(s.x - x) <= radius]
        return min(near, key=lambda s: abs(s.x - x)) if near else None

    def _spot_near(self, x, level, radius):
        if x is None:
            return None
        near = [s for s in self.spots if s.level == level and abs(s.x - x) <= radius]
        return min(near, key=lambda s: abs(s.x - x)) if near else None

    def _expire(self, now):
        self.spots = [s for s in self.spots if now - s.last_fight <= self.SPOT_TTL]
