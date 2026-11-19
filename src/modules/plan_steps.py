"""
Small, reusable building blocks for scripted manoeuvres (escaping a boss,
hiding on a rope, changing channel, ...), run one bot tick at a time.

A Step is ticked every bot tick (~20 ms) and answers RUNNING, DONE or FAILED.
It never sleeps for long, so the bot stays responsive (it can be stopped at
any time) and steps can be tested with a fake clock.

A Plan runs Steps in order. Each entry says what to do when its step fails:
go on with the next one, jump back (or ahead) to a labelled entry, or abort
the plan. Jumps are counted per entry (RETRIES), then its EXHAUSTED label is
used instead, so a plan can never loop for ever.

Steps talk to the game only through Controls (keys, mouse, minimap position,
the latest frame, the clock), which tests replace with a simulator.
"""

import random
import time

import cv2
import numpy as np

from src.common import config
from src.common.vkeys import key_down, key_up, press

RUNNING, DONE, FAILED = 'running', 'done', 'failed'


class Controls:
    """The bot's hands and eyes, as the steps see them."""

    POSITION_STALE_SECONDS = 0.5

    def __init__(self, bot):
        self.bot = bot

    def now(self):
        return time.monotonic()

    def position(self):
        """Fresh minimap (x, y), or None when unknown or not seen lately."""
        position = getattr(config, 'player_pos', None)
        if not position or tuple(position) == (0, 0):
            return None
        seen = getattr(config, 'player_pos_time', None)
        if seen and self.now() - seen > self.POSITION_STALE_SECONDS:
            return None
        return tuple(position)

    def position_time(self):
        return getattr(config, 'player_pos_time', 0.0) or 0.0

    def walk(self, direction):
        """Hold 'left'/'right' (None: stand still). Bypasses the patrol's
        flip-flop guard: lining up under a rope swaps direction on purpose."""
        bot = self.bot
        current = bot._auto_attack_movement
        if direction == current:
            return
        if current is not None:
            self.key_up(current)
        bot._auto_attack_movement = direction
        if direction is not None:
            self.key_down(direction)

    def key_down(self, key):
        key_down(key)

    def key_up(self, key):
        key_up(key)

    def press(self, key, n=1, down_time=0.05, up_time=0.05):
        press(key, n, down_time=down_time, up_time=up_time)

    @property
    def jump_key(self):
        keys = getattr(getattr(self.bot.command_book, 'module', None), 'Key', None)
        return getattr(keys, 'JUMP', None) or 'alt'

    def frame(self):
        capture = getattr(config, 'capture', None)
        return getattr(capture, 'frame', None)

    def minimap(self):
        capture = getattr(config, 'capture', None)
        return (getattr(capture, 'minimap', None) or {}).get('minimap')

    def click(self, x, y):
        """Left-clicks the frame pixel (X, Y) of the game window."""
        from src.common import vkeys
        window = getattr(config.capture, 'window', {'left': 0, 'top': 0})
        vkeys.click((window['left'] + x, window['top'] + y),
                    settle=(0.05, 0.10), hold=(0.06, 0.12), allow_paused=True)

    def expect_black_screen(self, seconds):
        """A loading screen is coming: the notifier must not take it for a disconnect."""
        config.map_transition_until = self.now() + seconds

    def log(self, message):
        print(f'\n[~] {message}')

    def hp(self):
        """HP percent from the HUD bar, or None when it can't be read."""
        try:
            value = config.capture.health.current_percent()
        except Exception:
            return None
        return None if value is None else float(value)

    def enemy_count(self):
        """Enemies the mob tracker sees on the character's level right now."""
        tracker = getattr(getattr(config, 'capture', None), 'mob_tracker', None)
        if tracker is None:
            return 0
        offsets = tracker.enemy_offsets()
        if offsets:
            return len(offsets)
        return 1 if tracker.target() is not None else 0

    def seconds_since_attack(self):
        """Seconds since the bot last attacked (0 while an attack is held)."""
        since = getattr(self.bot, 'seconds_since_attack', None)
        return float('inf') if since is None else since(self.now())

    def release(self):
        """Lets go of everything the steps may hold."""
        self.walk(None)
        self.bot._release_mob_attack()
        self.key_up('up')
        self.key_up('down')


def black_fraction(frame):
    """How much of FRAME is (near) black: a loading screen is almost all black."""
    if frame is None:
        return 0.0
    gray = cv2.cvtColor(np.ascontiguousarray(frame[:, :, :3]), cv2.COLOR_BGR2GRAY)
    return float(np.count_nonzero(gray < 15)) / gray.size


class Step:
    """One manoeuvre. Subclasses override tick() and, if they hold keys, end()."""

    name = 'step'
    timeout = None              # seconds; FAILED when it takes longer

    def __init__(self):
        self.started = None
        self.reason = ''

    def begin(self, io):
        self.started = io.now()
        self.reason = ''

    def run(self, io):
        if self.started is None:
            self.begin(io)
        if self.timeout is not None and io.now() - self.started > self.timeout:
            return self.fail(f'took longer than {self.timeout:.0f} s')
        return self.tick(io)

    def tick(self, io):
        return DONE

    def end(self, io):
        io.release()

    def fail(self, reason):
        self.reason = reason
        return FAILED

    def reset(self):
        self.started = None


class Wait(Step):
    name = 'wait'

    def __init__(self, seconds):
        super().__init__()
        self.seconds = seconds

    def tick(self, io):
        return DONE if io.now() - self.started >= self.seconds else RUNNING


class Entry:
    """A step in a plan, and what to do if it fails."""

    def __init__(self, step, label=None, on_fail='abort', retries=3, exhausted='abort',
                 then=None):
        """
        :param on_fail:     'next', 'abort', or the label of the entry to go to.
        :param retries:     how often ON_FAIL may jump; after that EXHAUSTED applies.
        :param then:        label to go to when the step is DONE (None: the next entry).
        """
        self.step = step
        self.label = label
        self.then = then
        self.on_fail = on_fail
        self.retries = retries
        self.exhausted = exhausted
        self.jumps = 0


class Plan:
    def __init__(self, name, entries, new_channel=False):
        """:param new_channel: the plan ends in another channel (drops are gone)."""
        self.name = name
        self.entries = entries
        self.new_channel = new_channel
        self.index = 0
        self.reason = ''
        self.log = []           # (step name, outcome) as the plan went

    @property
    def current(self):
        return self.entries[self.index].step if self.index < len(self.entries) else None

    def tick(self, io):
        if self.index >= len(self.entries):
            return DONE
        entry = self.entries[self.index]
        status = entry.step.run(io)
        if status == RUNNING:
            return RUNNING
        entry.step.end(io)
        self.log.append((entry.step.name, status if status == DONE else f'failed: {entry.step.reason}'))
        if status == DONE:
            if getattr(entry.step, 'changes_channel', False):
                self.new_channel = True     # (drops left behind are gone)
            return self._go(self._find(entry.then) if entry.then else self.index + 1)
        target = entry.on_fail
        if target not in ('next', 'abort'):
            entry.jumps += 1
            if entry.jumps > entry.retries:
                target = entry.exhausted
        io.log(f'{self.name}: {entry.step.name} failed ({entry.step.reason}); '
               + {'next': 'carrying on', 'abort': 'giving up'}.get(target, f'going to "{target}"') + '.')
        if target == 'abort':
            self.reason = f'{entry.step.name}: {entry.step.reason}'
            return FAILED
        if target == 'next':
            return self._go(self.index + 1)
        return self._go(self._find(target))

    def _go(self, index):
        self.index = index
        if index < len(self.entries):
            self.entries[index].step.reset()
            return RUNNING
        return DONE

    def _find(self, label):
        for index, entry in enumerate(self.entries):
            if entry.label == label:
                return index
        raise KeyError(f'plan {self.name!r} has no step labelled {label!r}')

    def stop(self, io):
        step = self.current
        if step is not None and step.started is not None:
            step.end(io)


# ---- Movement steps (minimap units: x and y both 0..1 of the minimap width) ----

def steer(io, x, target_x, tolerance, state):
    """
    Walks toward TARGET_X: holds the key while far, then short taps (walking
    momentum overshoots small distances). True once within TOLERANCE.
    """
    offset = target_x - x
    if abs(offset) <= tolerance:
        io.walk(None)
        return True
    direction = 'right' if offset > 0 else 'left'
    if abs(offset) > SteerConfig.HOLD_BEYOND:
        io.walk(direction)
        return False
    io.walk(None)
    now = io.now()
    if now - state.get('last_tap', float('-inf')) >= SteerConfig.TAP_INTERVAL:
        state['last_tap'] = now
        io.press(direction, 1, down_time=SteerConfig.TAP_SECONDS, up_time=0.02)
    return False


class SteerConfig:
    HOLD_BEYOND = 0.02          # farther than this: hold the arrow key
    TAP_INTERVAL = 0.25         # nearer: one short tap this often
    TAP_SECONDS = 0.06


class MountRope(Step):
    """
    Gets onto a rope the way a player does: a run-up holding the direction key,
    a jump a little before the rope with up held, so the character passes the
    rope near the top of the jump and grabs it; then climbs to SAFE_Y.

    Standing under the rope and jumping straight up (the old way) gives one
    instant at the rope's tip, and the minimap is too coarse (1 px ~ 0.0047) to
    line up within the grab window: iron_hog_issue-2.mp4 missed 6 of 6.

    Each miss is measured: where the character was at the top of the jump
    compared with the rope moves the next take-off point (past it: jump
    earlier; short of it: later). The next run starts from whichever side it
    landed on (overshot, or knocked back by a pig).

    In the air it steers like a player (2026-11-18: misses landed up to 0.013
    either side of the rope, a pig's hit bending the arc): rope still ahead,
    keep holding toward it; passed it, hold back; level with it, let go.
    After a miss it pauses a moment, as a hand would, then goes again from
    the side it landed on: straight up when right under the rope, else a
    short walk back out past that side's jump point and a run at the rope.

    v4 (2026-11-18, iron_hog_escape.mp4): the take-off points are the user's
    recorded jump spots (JUMP_LEFT / JUMP_RIGHT, 0.019 out), not 0.011. The
    old Pig Beach spots dated from before the minimap outline finder and
    were a pixel off (rope 0.286 for 0.284) and asked for a height (0.200)
    the character can no longer read on the rope, so even catches counted
    as misses (run-20261118-203218: 8 of 10 tops right at the rope).
    """

    name = 'climb the rope'
    # Measured on iron_hog_escape.mp4 (2026-11-18, the user mounting the Pig
    # Beach rope by hand, 30 fps, minimap units; notes in
    # assets/maps/pig_beach/rope_mount.md):
    #  - every catch from a run came from a take-off 0.019-0.024 before the
    #    rope (recorded jump points 0.265 / 0.303 for the rope at 0.284), the
    #    catch 0.1-0.33 s after take-off while still rising, ~0.025 up
    #  - standing right under it and jumping straight up with up held caught it too
    #  - a plain jump is ~0.43 s in the air and tops out 0.033 up (0.222 there),
    #    the recorded spot on the rope (0.213) is above any jump's top
    RUN_SPEED = 0.045           # walking, per second
    APEX_SECONDS = 0.22         # take-off to the top of a jump
    LEAD = 0.019                # take off this far before the rope (no recorded jump point)
    LEAD_RANGE = (0.008, 0.030)
    LEAD_JITTER = 0.0015        # a hand is never exactly the same twice
    LEAD_LEARN = 0.6            # share of a miss's apex error taken into the next lead
    RUNUP_EXTRA = 0.010         # back off this far beyond the jump point for a run-up
    RUNUP_SLACK = 0.003         # nearer the rope than the jump point by this: back off first
    UNDER = 0.004               # this close: jump straight up and steer
    AIR_STEER_SECONDS = 0.45    # steer toward the rope this long after take-off
    AIR_DEADBAND = 0.0025       # this close to the rope in the air: let go of the arrows
    RETRY_PAUSE = (0.12, 0.35)  # seconds; a hand's pause before jumping again
    LAND_SECONDS = 0.38         # back at the ground's height this long after take-off: missed
    GRAB_SECONDS = 0.65         # still off the ground this long after take-off: on the rope
    GROUND_TOLERANCE = 0.008
    SAFE_TOLERANCE = 0.005      # (a jump's top, 0.222 there, stays out of this)
    CLIMB_SECONDS = 3.0         # on the rope but not at SAFE_Y by then: missed
    STUCK_SECONDS = 1.0         # walking but not moving: hop (a pig or a step in the way)
    LOST_SECONDS = 3.0          # no minimap position this long: fail
    ON_ROPE_X = 0.008
    MIN_RISE = 0.012            # a real jump lifts the minimap dot at least this much
    SAFE_HOLD_SECONDS = 0.35    # a jump's top lasts ~0.1 s; on the rope it stays

    def __init__(self, rope_x, mount_y, safe_y, attempts=3, timeout=45.0,
                 jump_left=None, jump_right=None):
        """
        :param jump_left:   recorded spot left of the rope to jump from, running right
        :param jump_right:  the same on the right, running left
        """
        super().__init__()
        self.rope_x = rope_x
        self.mount_y = mount_y
        self.safe_y = safe_y
        self.attempts = attempts
        self.timeout = timeout
        # How far before the rope to take off, per running direction; learnt from misses
        self.leads = {
            'right': rope_x - jump_left if jump_left is not None else self.LEAD,
            'left': jump_right - rope_x if jump_right is not None else self.LEAD,
        }
        for side, lead in self.leads.items():
            self.leads[side] = min(max(lead, self.LEAD_RANGE[0]), self.LEAD_RANGE[1])
        self.run_dir = None

    @property
    def lead(self):
        """The take-off distance for the current run (the larger one when standing)."""
        return self.leads.get(self.run_dir) or max(self.leads.values())

    def begin(self, io):
        super().begin(io)
        io.release()        # stop attacking: a swing in progress swallows the jump
        self.misses = 0
        self.lost_since = None
        self.high_since = None
        self.moved = (None, io.now())
        self.run_dir = None
        self.backoff_dir = None
        self._setup(io)

    def tick(self, io):
        now = io.now()
        position = io.position()
        if position is None:
            if self.lost_since is None:
                self.lost_since = now
            if now - self.lost_since > self.LOST_SECONDS:
                return self.fail('lost the minimap position')
            return RUNNING
        self.lost_since = None
        x, y = position
        high = y <= self.safe_y + self.SAFE_TOLERANCE and abs(x - self.rope_x) <= self.ON_ROPE_X
        if high:
            # The top of a plain jump passes this height too: only staying up
            # here (holding the rope) counts
            if self.high_since is None:
                self.high_since = now
            if now - self.high_since >= self.SAFE_HOLD_SECONDS:
                io.walk(None)
                io.key_up('up')     # high enough: hang there
                return DONE
        else:
            self.high_since = None
        handler = getattr(self, '_' + self.phase)
        return handler(io, x, y, now)

    # -- run-up --

    def _setup(self, io, x=None, y=None, now=None):
        """Chooses the side to run from: the one the character is on."""
        self.phase = 'setup'
        if x is None:
            return RUNNING
        side = -1 if x < self.rope_x else 1          # -1: left of the rope
        self.run_dir = 'right' if side < 0 else 'left'
        distance = abs(x - self.rope_x)
        if distance <= self.UNDER:
            self.run_dir = None                     # right under it: straight up
            self.phase = 'run'
        elif distance < self.lead - self.RUNUP_SLACK:
            # Inside the jump point: walk back out past it for a run-up, as
            # the user does in iron_hog_escape.mp4 (every try, misses too)
            self.backoff_dir = 'left' if side < 0 else 'right'
            self.phase = 'backoff'
        else:
            self.phase = 'run'
        self.moved = (x, now)
        return RUNNING

    def _pause(self, io, x, y, now):
        """A moment's pause after a miss, then go again from here."""
        io.walk(None)
        if now >= self.pause_until:
            self._setup(io, x, y, now)
        return RUNNING

    def _backoff(self, io, x, y, now):
        """Too close for a run-up: walk away from the rope first."""
        if abs(x - self.rope_x) >= self.lead + self.RUNUP_EXTRA:
            io.walk(None)
            self.phase = 'run'
            self.moved = (x, now)
            return RUNNING
        io.walk(self.backoff_dir)
        self._hop_if_stuck(io, x, now)
        return RUNNING

    def _run(self, io, x, y, now):
        """Holding the direction at the rope; take off at the lead point."""
        if self.run_dir is not None:
            sign = 1 if self.run_dir == 'right' else -1
            before = sign * (self.rope_x - x)       # distance still to go
            if before < -0.004:
                self._setup(io, x, y, now)          # ran past (position lag): turn round
                return RUNNING
            io.walk(self.run_dir)
            target = self.lead + random.uniform(-self.LEAD_JITTER, self.LEAD_JITTER)
            if before > target:
                self._hop_if_stuck(io, x, now)
                return RUNNING
        else:
            io.walk(None)
        # Take off: up first, then the jump, direction still held
        io.key_down('up')
        io.press(io.jump_key, 1, down_time=random.uniform(0.08, 0.13),
                 up_time=0.03)
        self.phase = 'air'
        self.jumped_at = now
        self.ground_y = y
        self.takeoff_x = x
        self.apex = (x, y)
        return RUNNING

    def _hop_if_stuck(self, io, x, now):
        last_x, last_moved = self.moved
        if last_x is None or abs(x - last_x) > 0.002:
            self.moved = (x, now)
        elif now - last_moved >= self.STUCK_SECONDS:
            io.press(io.jump_key, 1)
            self.moved = (x, now)

    # -- the jump --

    def _air(self, io, x, y, now):
        elapsed = now - self.jumped_at
        if y < self.apex[1]:
            self.apex = (x, y)
        if elapsed < self.AIR_STEER_SECONDS:
            # Steer at the rope: a pig's hit or a mistimed take-off bends the arc
            offset = self.rope_x - x
            io.walk(None if abs(offset) <= self.AIR_DEADBAND
                    else 'right' if offset > 0 else 'left')
        else:
            io.walk(None)       # (on the rope the arrows would do nothing anyway)
        on_ground = y >= self.ground_y - self.GROUND_TOLERANCE
        if elapsed >= self.LAND_SECONDS and on_ground:
            return self._missed(io, x, y, now)
        if elapsed >= self.GRAB_SECONDS and not on_ground \
                and abs(x - self.rope_x) <= self.ON_ROPE_X:
            self.phase = 'climb'        # still up there next to the rope: holding it
            self.climb_from = now
        elif elapsed >= self.GRAB_SECONDS + 1.0:
            return self._missed(io, x, y, now)     # landed somewhere higher
        return RUNNING

    def _climb(self, io, x, y, now):
        on_ground = y >= self.ground_y - self.GROUND_TOLERANCE
        if on_ground or now - self.climb_from >= self.CLIMB_SECONDS:
            return self._missed(io, x, y, now)
        return RUNNING

    def _missed(self, io, x, y, now):
        io.key_up('up')
        io.walk(None)
        self.misses += 1
        if self.run_dir is None:                    # a straight-up hop: no timing to learn
            sign = 1 if x >= self.takeoff_x else -1
        else:
            sign = 1 if self.run_dir == 'right' else -1
        apex_x = self.apex[0]
        error = sign * (apex_x - self.rope_x)       # >0: the top was past the rope
        # A pig's hit throws the character back against its run; a jump that
        # never rose (hit at take-off, key lost) says nothing about timing
        knocked = sign * (x - self.takeoff_x) < -0.004
        rose = self.ground_y - self.apex[1] >= self.MIN_RISE
        if not rose:
            knocked = True
        if not knocked and self.run_dir is not None:
            self.leads[self.run_dir] = min(max(self.lead + self.LEAD_LEARN * error,
                                               self.LEAD_RANGE[0]), self.LEAD_RANGE[1])
        if self.misses >= self.attempts:
            return self.fail(f'missed the rope {self.misses} times')
        where = ('did not get off the ground' if not rose else
                 'knocked back' if knocked else
                 'hopped straight up' if self.run_dir is None else
                 'jumped too late' if error > 0.002 else
                 'jumped too early' if error < -0.002 else 'right at it')
        io.log(f'Missed the rope (try {self.misses} of {self.attempts}; {where}, '
               f'top of the jump at {apex_x:.3f} (height {self.apex[1]:.3f}), rope '
               f'{self.rope_x:.3f}, took off at {self.takeoff_x:.3f}, landed at {x:.3f}); '
               f'going again (take-off {self.lead:.3f} before).')
        self.phase = 'pause'
        self.pause_until = now + random.uniform(*self.RETRY_PAUSE)
        return RUNNING

    def end(self, io):
        io.release()


class HoldOnRope(Step):
    """Hangs on the rope for SECONDS; FAILED when knocked off it."""

    name = 'wait on the rope'
    OFF_TOLERANCE = 0.02        # this far below the safe height: off the rope
    OFF_SECONDS = 0.3

    def __init__(self, rope_x, safe_y, seconds):
        super().__init__()
        self.rope_x = rope_x
        self.safe_y = safe_y
        self.seconds = seconds

    def begin(self, io):
        super().begin(io)
        self.off_since = None
        io.log(f'On the rope; waiting {self.seconds:.0f} s.')

    def _knocked_off(self, io, now):
        position = io.position()
        if position is not None:
            x, y = position
            off = y > self.safe_y + self.OFF_TOLERANCE or abs(x - self.rope_x) > 0.02
            if off:
                self.off_since = self.off_since or now
                return now - self.off_since >= self.OFF_SECONDS
            self.off_since = None
        return False

    def tick(self, io):
        now = io.now()
        if self._knocked_off(io, now):
            return self.fail('knocked off the rope')
        return DONE if now - self.started >= self.seconds else RUNNING


class HideUntilGone(HoldOnRope):
    """
    Stays on the rope until a sprite (the Iron Hog) has been out of sight for
    CLEAR_SECONDS - instead of changing channel, for "Stay on this channel".
    At least MIN_SECONDS on the rope; FAILED after MAX_SECONDS or when knocked off.
    """

    name = 'hide on the rope until it leaves'
    CHECK_SECONDS = 0.5         # between looks for the sprite (a look takes ~tens of ms)

    def __init__(self, rope_x, safe_y, watch_factory, what='Iron Hog',
                 min_seconds=10.0, clear_seconds=8.0, max_seconds=180.0):
        super().__init__(rope_x, safe_y, min_seconds)
        self.watch_factory = watch_factory
        self.what = what
        self.clear_seconds = clear_seconds
        self.max_seconds = max_seconds
        self.watch = None

    def begin(self, io):
        Step.begin(self, io)
        self.off_since = None
        if self.watch is None:
            self.watch = self.watch_factory()
        self.last_seen = self.started
        self.last_check = float('-inf')
        io.log(f'On the rope; staying on this channel, so waiting until the {self.what} '
               f'has been gone for {self.clear_seconds:.0f} s.')

    def tick(self, io):
        now = io.now()
        if self._knocked_off(io, now):
            return self.fail('knocked off the rope')
        if now - self.last_check >= self.CHECK_SECONDS:
            self.last_check = now
            if self.watch.find(io.frame()):
                self.last_seen = now
        if now - self.started >= self.seconds and now - self.last_seen >= self.clear_seconds:
            io.log(f'The {self.what} is gone.')
            return DONE
        if now - self.started >= self.max_seconds:
            return self.fail(f'the {self.what} was still here after {self.max_seconds:.0f} s')
        return RUNNING


class LeaveRope(Step):
    """Jumps off the rope towards DIRECTION and waits to land below it."""

    name = 'jump off the rope'
    LANDED_DROP = 0.03          # this far below the rope's safe height: off it
    TRY_SECONDS = 1.5
    TRIES = 3

    def __init__(self, safe_y, direction='right'):
        super().__init__()
        self.safe_y = safe_y
        self.direction = direction

    def begin(self, io):
        super().begin(io)
        self.tries = 0
        self.tried_at = None

    def tick(self, io):
        now = io.now()
        position = io.position()
        if position is not None and position[1] >= self.safe_y + self.LANDED_DROP:
            io.walk(None)
            return DONE
        if self.tried_at is None or now - self.tried_at >= self.TRY_SECONDS:
            if self.tries >= self.TRIES:
                return self.fail('still on the rope')
            self.tries += 1
            self.tried_at = now
            io.key_up('up')
            io.walk(self.direction)
            io.press(io.jump_key)
        return RUNNING


class WalkOff(Step):
    """
    After a channel change the character appears at the map's spawn point
    (Pig Beach: a cliff above the training ground). Walks DIRECTION until it
    drops off the edge and lands. Not at the spawn point: nothing to do.
    """

    name = 'walk off the cliff'
    SPAWN_TOLERANCE = 0.03
    DROP = 0.03                 # this much lower than the spawn: fell off
    LAND_SECONDS = 0.4          # no more falling for this long: landed
    STUCK_SECONDS = 1.5

    def __init__(self, direction, spawn, timeout=12.0):
        super().__init__()
        self.direction = direction
        self.spawn = spawn
        self.timeout = timeout

    def begin(self, io):
        super().begin(io)
        self.start_y = None
        self.fallen = False
        self.last = None        # (y, time it last changed)
        self.moved = (None, io.now())

    def tick(self, io):
        now = io.now()
        position = io.position()
        if position is None:
            return RUNNING
        x, y = position
        if self.start_y is None:
            if abs(y - self.spawn[1]) > self.SPAWN_TOLERANCE:
                return DONE     # not up on the spawn cliff: already where it farms
            self.start_y = y
        if not self.fallen:
            if y >= self.start_y + self.DROP:
                self.fallen = True
                self.last = (y, now)
            else:
                io.walk(self.direction)
                last_x, last_moved = self.moved
                if last_x is None or abs(x - last_x) > 0.002:
                    self.moved = (x, now)
                elif now - last_moved >= self.STUCK_SECONDS:
                    io.press(io.jump_key, 1)
                    self.moved = (x, now)
            return RUNNING
        io.walk(None)
        if abs(y - self.last[0]) > 0.002:
            self.last = (y, now)
        elif now - self.last[1] >= self.LAND_SECONDS:
            return DONE
        return RUNNING


class ReturnInside(Step):
    """
    Out past the map's limit (Pig Beach: fallen into the pit right of the pig
    island): holds the walking direction and taps up and jump back toward the
    farming area, then waits for the character to land past SAFE_X.
    """

    name = 'get back to the farming area'
    JUMP_INTERVAL = 0.5
    SETTLE_SECONDS = 0.3        # past SAFE_X this long (landed, not mid-hop): back
    Y_STABLE_EPSILON = 0.003
    LOST_SECONDS = 3.0

    def __init__(self, direction, safe_x, timeout=25.0):
        super().__init__()
        self.direction = direction
        self.safe_x = safe_x
        self.timeout = timeout

    def begin(self, io):
        super().begin(io)
        self.last_jump = float('-inf')
        self.inside_since = None
        self.inside_y = None
        self.lost_since = None
        io.log(f'Out of bounds; holding {self.direction}, tapping up and jumping '
               f'back to x={self.safe_x:.3f}.')

    def _inside(self, x):
        return x <= self.safe_x if self.direction == 'left' else x >= self.safe_x

    def tick(self, io):
        now = io.now()
        position = io.position()
        if position is None:
            io.walk(None)
            self.inside_since = None
            self.inside_y = None
            self.lost_since = self.lost_since or now
            if now - self.lost_since > self.LOST_SECONDS:
                return self.fail('lost the minimap position')
            return RUNNING
        self.lost_since = None
        x, y = position
        if self._inside(x):
            io.walk(None)
            if (self.inside_y is None
                    or abs(y - self.inside_y) > self.Y_STABLE_EPSILON):
                self.inside_y = y
                self.inside_since = now
            return DONE if now - self.inside_since >= self.SETTLE_SECONDS else RUNNING
        self.inside_since = None
        self.inside_y = None
        io.walk(self.direction)
        if now - self.last_jump >= self.JUMP_INTERVAL:
            self.last_jump = now
            io.press('up', 1)
            io.press(io.jump_key, 1)
        return RUNNING


# ---- One-platform farming (A Split Road): rest on a chair, back up, look around ----

def on_platform(position, platform, slack=0.004, height=0.006):
    """True when POSITION (x, y) stands on PLATFORM (left x, right x, y).
    (loop.mp4: A Split Road's safe spot read (0.209, 0.184), only 2 minimap
    px below the platform at 0.176: a 0.01 slack counted it as on top.)"""
    if position is None:
        return False
    x, y = position
    left, right, level = platform
    return left - slack <= x <= right + slack and abs(y - level) <= height


class GoToSpot(Step):
    """
    Walks to the recorded spot (X, Y), dropping off an edge on the way if the
    spot is lower (A Split Road: the safe spot is just past the platform's
    right end, a step down). DONE once within TOLERANCE and standing still at
    about Y; a wall in the way gets a hop, then FAILED.
    """

    name = 'walk to the spot'
    SETTLE_SECONDS = 0.3
    HEIGHT_TOLERANCE = 0.010
    STUCK_SECONDS = 1.2
    HOPS = 3
    LOST_SECONDS = 3.0

    def __init__(self, x, y, what='the spot', tolerance=0.005, timeout=25.0):
        super().__init__()
        self.x, self.y = x, y
        self.name = f'walk to {what}'
        self.tolerance = tolerance
        self.timeout = timeout

    def begin(self, io):
        super().begin(io)
        io.release()
        self.state = {}
        self.settled = None         # (y, since)
        self.moved = (None, io.now())
        self.hops = 0
        self.lost_since = None

    def tick(self, io):
        now = io.now()
        position = io.position()
        if position is None:
            io.walk(None)
            self.lost_since = self.lost_since or now
            if now - self.lost_since > self.LOST_SECONDS:
                return self.fail('lost the minimap position')
            return RUNNING
        self.lost_since = None
        x, y = position
        if steer(io, x, self.x, self.tolerance, self.state):
            if self.settled is None or abs(y - self.settled[0]) > 0.002:
                self.settled = (y, now)         # still dropping / landing
            elif now - self.settled[1] >= self.SETTLE_SECONDS:
                if abs(y - self.y) > self.HEIGHT_TOLERANCE:
                    return self.fail(f'reached x={x:.3f} but at height {y:.3f}, '
                                     f'not {self.y:.3f}')
                return DONE
            return RUNNING
        self.settled = None
        last_x, last_moved = self.moved
        if last_x is None or abs(x - last_x) > 0.002:
            self.moved = (x, now)
        elif now - last_moved >= self.STUCK_SECONDS:
            if self.hops >= self.HOPS:
                return self.fail(f'stuck at x={x:.3f}')
            self.hops += 1
            io.press(io.jump_key, 1)
            self.moved = (x, now)
        return RUNNING


class SitUntilHealed(Step):
    """
    Sits on the chair (its quick-slot KEY) and waits until HP is back to
    FULL percent. A chair heals in ticks several seconds apart, so HP not
    rising for RISE_SECONDS means the character is not sitting (the press
    was lost, or a later press stood it up): press again, a few times at
    most. Knocked off the spot: FAILED (the plan walks back). After
    MAX_SECONDS it gives up waiting and carries on with what it has.
    """

    name = 'sit on the chair until HP is full'
    FULL = 98.0                 # the bar reading wobbles ~1-2 %
    FULL_SECONDS = 1.0          # read full this long
    RISE_SECONDS = 15.0
    RISE_MIN = 1.0
    PRESSES = 4
    MAX_SECONDS = 600.0
    MOVED = 0.02                # this far from where it sat: knocked off

    def __init__(self, key='c', full=FULL):
        super().__init__()
        self.key = key
        self.full = full

    def begin(self, io):
        super().begin(io)
        io.release()
        self.presses = 0
        self.full_since = None
        self.spot = io.position()
        self._sit(io, io.now())

    def _sit(self, io, now):
        self.presses += 1
        io.press(self.key, 1, down_time=random.uniform(0.06, 0.12))
        self.pressed_at = now
        self.hp_at_press = io.hp()
        hp = '?' if self.hp_at_press is None else f'{self.hp_at_press:.0f}%'
        io.log(f'Sitting on the chair ({self.key.upper()}, press {self.presses}); HP {hp}.')

    def tick(self, io):
        now = io.now()
        position = io.position()
        if self.spot is None:
            self.spot = position
        elif position is not None and abs(position[0] - self.spot[0]) > self.MOVED:
            return self.fail(f'knocked off the spot (now at x={position[0]:.3f})')
        hp = io.hp()
        if hp is not None and hp >= self.full:
            self.full_since = self.full_since or now
            if now - self.full_since >= self.FULL_SECONDS:
                io.log(f'HP is back to {hp:.0f}%.')
                return DONE
        else:
            self.full_since = None
        if now - self.started >= self.MAX_SECONDS:
            shown = '?' if hp is None else f'{hp:.0f}'
            io.log(f'Still only {shown}% HP after {self.MAX_SECONDS / 60:.0f} min on '
                   'the chair; carrying on.')
            return DONE
        if now - self.pressed_at >= self.RISE_SECONDS:
            start = self.hp_at_press
            rising = hp is not None and start is not None and hp - start >= self.RISE_MIN
            if rising:
                self.pressed_at, self.hp_at_press = now, hp     # healing: keep sitting
            elif self.presses < self.PRESSES:
                self._sit(io, now)
            else:
                return self.fail(f'HP did not rise after {self.presses} presses of '
                                 f'{self.key.upper()} (is the chair on that key?)')
        return RUNNING

    def end(self, io):
        pass                    # holds no keys


class ClimbOnto(Step):
    """
    Back up onto PLATFORM (left x, right x, y) from the ground beside it,
    one level at a time. TIERS: [(take-off x, height y), ...] for each level
    of ground next to the platform. A Split Road: the safe ledge (0.209,
    0.192) and, one step lower, the bottom ground (stuck.mp4: (0.224, 0.209)).
    On a tier: walk to its take-off spot (only when right of it), then jump
    holding DIRECTION until a higher level is reached; repeat until standing
    on the platform. Already on it: DONE at once. Not on any known level
    (e.g. a channel change's spawn point): FAILED, naming the position.
    """

    name = 'jump back onto the platform'
    ONTO_SECONDS = 0.4          # on the platform this long: landed (not a jump's top)
    JUMP_GAP = 1.0              # between jumps (a jump is ~0.45 s in the air)
    TRIES = 6                   # jumps in all
    # The nearest tier within this counts (the safe ledge has read 0.184 and
    # 0.192 on different runs; the bottom 0.209)
    TIER_HEIGHT = 0.012
    WALK_SLACK = 0.008          # this far right of the take-off spot: walk there first
    LOST_SECONDS = 3.0

    def __init__(self, platform, tiers, direction='left', timeout=40.0):
        super().__init__()
        self.platform = platform
        # (one (x, y) take-off spot is a single tier, as before)
        self.tiers = [tiers] if isinstance(tiers[0], (int, float)) else list(tiers)
        self.direction = direction
        self.timeout = timeout

    def begin(self, io):
        super().begin(io)
        io.release()
        self.first = True
        self.onto_since = None
        self.tries = 0
        self.jumped_at = float('-inf')
        self.state = {}
        self.lost_since = None

    def _tier(self, y):
        near = [t for t in self.tiers if abs(y - t[1]) <= self.TIER_HEIGHT]
        return min(near, key=lambda t: abs(y - t[1])) if near else None

    def tick(self, io):
        now = io.now()
        position = io.position()
        if position is None:
            self.lost_since = self.lost_since or now
            if now - self.lost_since > self.LOST_SECONDS:
                return self.fail('lost the minimap position')
            return RUNNING
        self.lost_since = None
        if on_platform(position, self.platform):
            self.onto_since = self.onto_since or now
            if self.first or now - self.onto_since >= self.ONTO_SECONDS:
                io.walk(None)
                return DONE
            return RUNNING
        self.onto_since = None
        x, y = position
        in_air = now - self.jumped_at < self.JUMP_GAP
        tier = self._tier(y)
        if tier is None:
            if self.first:
                return self.fail(f'at ({x:.3f}, {y:.3f}): neither on the platform nor on '
                                 'the ground beside it, and no known way up from there')
            if not in_air:
                # Between levels after a jump (or knocked lower): give it a moment
                if now - self.jumped_at > 2 * self.JUMP_GAP + 1.0:
                    return self.fail(f'landed at ({x:.3f}, {y:.3f}), not on a known level')
            return RUNNING
        self.first = False
        if in_air:
            io.walk(self.direction)
            return RUNNING
        sign = -1 if self.direction == 'left' else 1
        if sign * (tier[0] - x) > self.WALK_SLACK:
            steer(io, x, tier[0], self.WALK_SLACK, self.state)     # to the take-off spot
            return RUNNING
        # Jump with the direction held (the first press may only stand it up off the chair)
        io.walk(self.direction)
        if self.tries >= self.TRIES:
            return self.fail(f'still at ({x:.3f}, {y:.3f}) after {self.tries} jumps')
        self.tries += 1
        self.jumped_at = now
        io.press(io.jump_key, 1, down_time=random.uniform(0.08, 0.13), up_time=0.03)
        return RUNNING


class ScoutPlatform(Step):
    """
    Stands on PLATFORM for a moment and counts the enemies the mob tracker
    sees there: DONE when at least MIN_ENEMIES were in sight at once, FAILED
    when fewer (the plan then tries another channel).
    """

    name = 'look for enemies on the platform'
    LOOK_DELAY = 1.0            # the tracker needs a few scans after a channel change
    LOOK_SECONDS = 3.0

    def __init__(self, platform, min_enemies=2):
        super().__init__()
        self.platform = platform
        self.min_enemies = min_enemies

    def begin(self, io):
        super().begin(io)
        io.release()
        self.most = 0

    def tick(self, io):
        now = io.now()
        position = io.position()
        if not on_platform(position, self.platform):
            where = 'unknown' if position is None else f'({position[0]:.3f}, {position[1]:.3f})'
            return self.fail(f'not on the platform (at {where})')
        if now - self.started < self.LOOK_DELAY:
            return RUNNING
        self.most = max(self.most, io.enemy_count())
        if self.most >= self.min_enemies:
            io.log(f'{self.most} enemies on the platform; farming here.')
            return DONE
        if now - self.started >= self.LOOK_DELAY + self.LOOK_SECONDS:
            return self.fail(f'only {self.most} enem{"y" if self.most == 1 else "ies"} '
                             f'on the platform (want {self.min_enemies})')
        return RUNNING
