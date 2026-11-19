"""
Map-specific events and the bot's scripted response to them.

Each map the bot knows is a MapProfile:
  * how to recognise it (its name in the minimap's title bar, as a crop),
  * the saved enemy-sample profile to load when it is recognised (if specified),
  * its fixed spots (ropes, the spawn point, ...), recorded with the
    "Record position" key (minimap units),
  * its events: what to watch for (e.g. a sprite: the Iron Hog) and the plan
    to run when it shows up (a list of plan_steps / channel steps).

MapEventWatcher runs on its own thread (the scans are too slow for the
capture loop): it works out which map this is now and then, scans for that
map's events, and posts one when it is confirmed. MapEventManager, called
from the bot's tick, takes the event, runs its plan one tick at a time, and
hands the controls back to farming when it is over.

Adding something new: write a plan factory from the steps (or a new Step),
add an Event with a watch to a profile, or add a MapProfile.
"""

import threading
import time

import cv2
import numpy as np

from src.common import config
from src.detection.sprite_watch import SpriteWatch
from src.modules.channel import ChangeChannel, FindQuietChannel
from src.modules.channel_settings import chosen_map, stay_on_channel
from src.modules.plan_steps import (
    ClimbOnto, Controls, Entry, GoToSpot, HideUntilGone, HoldOnRope, LeaveRope, MountRope,
    Plan, ReturnInside, ScoutPlatform, SitUntilHealed, Wait, WalkOff, on_platform,
    RUNNING, DONE, FAILED)

ENABLED = True


class Event:
    def __init__(self, name, watch_factory, plan_factory, cooldown=20.0, message=None,
                 changes_channel_only=False):
        """
        :param watch_factory:   () -> an object with update(frame, now) -> bool,
                                reset(), ready (e.g. a SpriteWatch).
        :param plan_factory:    (profile) -> Plan.
        :param cooldown:        after the plan, ignore this event this long.
        :param changes_channel_only: its whole answer is a channel change, so it
                                is skipped under "Stay on this channel".
        """
        self.changes_channel_only = changes_channel_only
        self.name = name
        self.watch_factory = watch_factory
        self.plan_factory = plan_factory
        self.cooldown = cooldown
        self.message = message or f'{name} spotted!'
        self._watch = None

    @property
    def watch(self):
        if self._watch is None:
            self._watch = self.watch_factory()
        return self._watch


class OutOfBounds:
    """
    Watches the minimap position: True once the character has been past the
    map's limit (beyond LIMIT in DIRECTION) for CONFIRM_SECONDS.
    """

    ready = True
    CONFIRM_SECONDS = 0.4
    STALE_SECONDS = 0.5

    def __init__(self, direction, limit, margin=0.005):
        self.direction = direction
        self.limit = limit
        self.margin = margin
        self.since = None

    def update(self, frame, now):
        position = getattr(config, 'player_pos', None)
        seen = getattr(config, 'player_pos_time', None) or 0.0
        if not position or tuple(position) == (0, 0) or time.monotonic() - seen > self.STALE_SECONDS:
            return False
        x = position[0]
        out = (x > self.limit + self.margin if self.direction == 'right'
               else x < self.limit - self.margin)
        if not out:
            self.since = None
            return False
        self.since = self.since or now
        return now - self.since >= self.CONFIRM_SECONDS

    def reset(self):
        self.since = None


class PlayersNearby:
    """
    Watches the minimap: True once at least MIN_PLAYERS other players (red
    dots) have been on the character's platform - about the same height, within
    NEAR_X to either side - for most (BUSY_SHARE) of the last CONFIRM_SECONDS.
    Someone just passing through doesn't count.
    """

    ready = True
    MIN_PLAYERS = 2
    NEAR_X = 0.15               # minimap units to either side (map width is 1)
    # Henesys Hunting Ground I (miss-20261115-042125): platforms ~0.08 apart; 4
    # players on one platform read 0.42-0.46 high (jumping, overlapping dots).
    SAME_Y = 0.035
    CONFIRM_SECONDS = 5.0
    BUSY_SHARE = 0.7
    STALE_SECONDS = 0.5

    def __init__(self):
        self.since = None
        self.count = 0
        self.history = []           # [(time, players near)]

    def update(self, frame, now):
        from src.modules.channel import players_near
        capture = getattr(config, 'capture', None)
        minimap = (getattr(capture, 'minimap', None) or {}).get('minimap')
        position = getattr(config, 'player_pos', None)
        seen = getattr(config, 'player_pos_time', None) or 0.0
        if (minimap is None or not position or tuple(position) == (0, 0)
                or time.monotonic() - seen > self.STALE_SECONDS):
            return False
        near = players_near(minimap, position, self.NEAR_X, self.SAME_Y,
                            getattr(capture, 'minimap_ratio', 1) or 1)
        self.count = len(near)
        self.history.append((now, self.count))
        self.history = [(t, n) for t, n in self.history if now - t <= self.CONFIRM_SECONDS]
        busy = [n for _, n in self.history if n >= self.MIN_PLAYERS]
        if not busy:
            self.since = None
            return False
        self.since = self.since or now
        if (now - self.since >= self.CONFIRM_SECONDS
                and len(busy) >= self.BUSY_SHARE * len(self.history)):
            CROWDED.message = (f'{max(busy)} other players on my platform for '
                               f'{self.CONFIRM_SECONDS:.0f} s.')
            return True
        return False

    def reset(self):
        self.since = None
        self.history = []


class MapProfile:
    TITLE_THRESHOLD = 0.8
    # The minimap's title bar, top left (pixels). Captures are the whole game
    # window (GetWindowRect: Windows title bar and border included), so the map
    # name sits ~40 px lower than in the client area. At (0, 0, 420, 90) "The
    # Pig Beach" (y 79-93) fell just outside after the 19 Nov restart: the map
    # was never recognised, pig-profile never loaded, and the bot saw no pigs.
    TITLE_REGION = (0, 0, 600, 200)

    def __init__(self, name, title, spots, events, limits=None, mob_profile=None,
                 after_channel=None):
        """
        :param title:   minimap title crop (None: only picked by hand in the GUI).
        :param limits:  {'left'/'right': minimap x} farming never walks past
                        (Bot._wall_ahead treats them as fixed walls).
        :param mob_profile: saved MobTracker profile to use on this map.
        :param after_channel: (profile) -> [Entry]: how to get from where a
                        channel change lands back to where the bot farms.
        """
        self.name = name
        self.title_path = title
        self.title = cv2.imread(title, cv2.IMREAD_GRAYSCALE) if title else None
        self.after_channel = after_channel
        self.spots = spots
        self.events = events
        self.limits = limits or {}
        self.mob_profile = mob_profile

    def visible(self, frame):
        """True when FRAME's minimap title shows this map's name."""
        if self.title is None or frame is None:
            return False
        left, top, right, bottom = self.TITLE_REGION
        area = cv2.cvtColor(np.ascontiguousarray(frame[top:bottom, left:right, :3]),
                            cv2.COLOR_BGR2GRAY)
        if area.shape[0] < self.title.shape[0] or area.shape[1] < self.title.shape[1]:
            return False
        score = cv2.minMaxLoc(cv2.matchTemplate(area, self.title, cv2.TM_CCOEFF_NORMED))[1]
        return score >= self.TITLE_THRESHOLD


# ---- Pig Beach: the Iron Hog ----

ROPE_TRIES = 3       # jumps at the rope before giving up on it and changing channel


def mount_rope(spots, attempts=ROPE_TRIES):
    """
    MountRope for a profile's recorded rope: 'rope_x', 'rope_mount_y' (the
    ground under it), 'rope_safe_y' (the spot on the rope), and optionally
    'rope_jump_left' / 'rope_jump_right' (where to take off, running at it
    from that side). Any map's rope works from those five recorded positions.
    """
    return MountRope(spots['rope_x'], spots['rope_mount_y'], spots['rope_safe_y'],
                     attempts=attempts, jump_left=spots.get('rope_jump_left'),
                     jump_right=spots.get('rope_jump_right'))


def iron_hog_escape(profile):
    """
    The Iron Hog hits hard: hide high on the rope out of its reach, wait for
    the fight to calm down, change channel (preferring one without other
    players), then walk off the spawn cliff to the training ground.
    3 tries at the rope; no luck (the hog's hits keep throwing the character
    off), change channel straight from the ground.
    """
    s = profile.spots
    if stay_on_channel():
        return iron_hog_hide(profile)
    return Plan('Iron Hog escape', new_channel=True, entries=[
        Entry(mount_rope(s), label='rope', on_fail='channel', retries=0, exhausted='channel'),
        Entry(HoldOnRope(s['rope_x'], s['rope_safe_y'], s['rope_wait']),
              on_fail='rope', retries=1, exhausted='channel'),
        Entry(FindQuietChannel(max_channels=s['channels_to_try']),
              label='channel', on_fail='abort'),
        Entry(WalkOff(s['drop_direction'], s['spawn']), on_fail='next'),
    ])


def iron_hog_watch():
    return SpriteWatch('Iron Hog', 'assets/maps/pig_beach/iron_hog_*.png',
                       region=(0.0, 0.10, 1.0, 0.88))


def iron_hog_hide(profile):
    """
    "Stay on this channel": hide on the rope like the escape, but instead of
    changing channel wait there until the Iron Hog has been gone a while, then
    jump off and farm on. 2026-11-18, user: "try 3 times to get on rope, if
    not possible change channel" - so when the rope can't be reached (or the
    hog never leaves) it changes channel after all, rather than pausing the
    bot next to the hog as it used to.
    """
    s = profile.spots
    return Plan('Iron Hog hide (staying on this channel)', entries=[
        Entry(mount_rope(s), label='rope', on_fail='channel', retries=0, exhausted='channel'),
        Entry(HideUntilGone(s['rope_x'], s['rope_safe_y'], iron_hog_watch,
                            min_seconds=s['rope_wait']),
              on_fail='rope', retries=1, exhausted='channel'),
        Entry(LeaveRope(s['rope_safe_y'], s.get('rope_leave_direction', 'right')),
              on_fail='abort', then='done'),
        # Only reached when the rope failed: escape to another channel instead
        Entry(FindQuietChannel(max_channels=s['channels_to_try']),
              label='channel', on_fail='abort'),
        Entry(WalkOff(s['drop_direction'], s['spawn']), on_fail='next'),
        Entry(Wait(0), label='done'),
    ])


def back_off_the_pit(profile):
    """
    Fell off the right end of the pig island into the pit beyond it, where
    walking alone never gets out: head left jumping (onto the island, across
    it) until back on the ground left of the island, where the bot farms.
    """
    s = profile.spots
    return Plan('Back from the pit', [
        Entry(ReturnInside('left', s['island_left'] - 0.01),
              label='return', on_fail='return', retries=2, exhausted='abort'),
    ])


PIG_BEACH = MapProfile(
    'The Pig Beach',
    title='assets/maps/pig_beach/title.png',
    spots={
        # Recorded 2026-11-18 in iron_hog_escape.mp4 (after the minimap outline
        # finder: the old 0.286/0.249/0.200 were read from the old minimap crop)
        'rope_x': 0.284,            # (0.284, 0.256): the rope, standing under it
        'rope_mount_y': 0.256,
        'rope_safe_y': 0.213,       # (0.284, 0.213): on the rope, out of the hog's reach
        'rope_jump_left': 0.265,    # (0.265, 0.256): take off here running right
        'rope_jump_right': 0.303,   # (0.303, 0.256): take off here running left
        'rope_wait': 10.0,
        'channels_to_try': 3,
        'spawn': (0.057, 0.147),    # after a channel change (measured on the recording)
        'drop_direction': 'right',  # the training ground is below, to the right
        'rope_leave_direction': 'right',    # staying on the channel: jump off the rope this way
        'island_left': 0.526,       # recorded: the ledge to jump up onto the pig island
        'island_right': 0.657,      # recorded: the island's right end; a pit beyond
        # User 2026-11-20: 1 lap of the map with no pig seen anywhere (the
        # island included): change channel (Bot._empty_turn; other maps: 3).
        'empty_map_laps': 1,
        # Once the pig island is cleared, this many laps of the flat ground
        # with no pigs: change channel (Bot._empty_turn)
        'island_clear_laps': 1,
    },
    # A little short of the island's end, so walking stops before the drop
    limits={'right': 0.657 - 0.012},
    mob_profile='pig-profile',
    events=[
        Event('Pit', lambda: OutOfBounds('right', 0.657), back_off_the_pit,
              cooldown=3.0, message='Fell off the pig island into the pit (recorded at x=0.704)!'),
        Event('Iron Hog', iron_hog_watch, iron_hog_escape),
    ],
)



# ---- A Split Road: one platform, two ends, a chair spot ----
#
# User request 2026-11-19: change channel until the platform has 2+ enemies,
# patrol between its two ends clearing them, change channel once it's clear;
# at 20 % HP walk to the safe spot, sit on the chair (C) until HP is full,
# then jump back up left onto the platform. Channel changes wait 4 s after
# the last attack (ChangeChannel.AFTER_ATTACK_SECONDS).

SPLIT_ROAD_MIN_ENEMIES = 2
SPLIT_ROAD_MAX_HOPS = 200       # channel changes looking for a busy platform (user: keep changing)
SPLIT_ROAD_REST_HP = 20         # % HP: go and sit on the chair
SPLIT_ROAD_CLEAR_SECONDS = 5.0  # no enemy on the platform this long: cleared
SPLIT_ROAD_CHAIR_DELAY = 4.0    # seconds standing on the safe spot before pressing C
SPLIT_ROAD_CHANNEL_ROUNDS = 4   # rounds of all 40 channels before giving up (pausing)


def _platform(spots):
    """(left x, right x, y) of the farming platform."""
    return (spots['left'][0], spots['right'][0], spots['left'][1])


def _tiers(spots):
    """The levels of ground right of the platform, highest first: [(take-off x, y)]."""
    return [spots['safe']] + list(spots.get('lower_tiers', ()))


def _to_platform(spots):
    """From the ground right of the platform: up level by level, jumping left."""
    return ClimbOnto(_platform(spots), _tiers(spots), direction='left')


class HopChannel(ChangeChannel):
    """ChangeChannel run again and again by one plan: start from where it went last."""

    changes_channel = True      # (Plan: drops left behind are gone)

    def begin(self, io):
        self.known_channel = self.channel or self.known_channel
        super().begin(io)


def platform_hunt(profile, change_first=True):
    """
    Change channel until the platform has SPLIT_ROAD_MIN_ENEMIES or more on
    it. CHANGE_FIRST False: look at this channel first (the bot was just
    started here). After SPLIT_ROAD_MAX_HOPS channels it farms wherever it is.
    """
    s = profile.spots
    if stay_on_channel():
        return Plan('Back onto the platform', [Entry(_to_platform(s), on_fail='next')])
    plan = Plan(f'Channel hunt ({SPLIT_ROAD_MIN_ENEMIES}+ enemies on the platform)',
                new_channel=change_first, entries=[
        Entry(HopChannel(), label='channel', on_fail='channel', retries=SPLIT_ROAD_CHANNEL_ROUNDS - 1,
              exhausted='abort'),
        Entry(_to_platform(s), label='platform', on_fail='abort'),
        Entry(ScoutPlatform(_platform(s), SPLIT_ROAD_MIN_ENEMIES),
              on_fail='channel', retries=SPLIT_ROAD_MAX_HOPS - 1, exhausted='next', then='done'),
        Entry(Wait(0), label='done'),
    ])
    if not change_first:
        plan.index = 1
    return plan


def rest_on_chair(profile):
    """
    Low HP: walk to the safe spot, sit on the chair until full, jump back up.
    User 2026-11-19: no enemy up there once healed: change channel (and hunt
    for a platform with SPLIT_ROAD_MIN_ENEMIES, as after a clear).
    """
    s = profile.spots
    platform = _platform(s)
    entries = [
        Entry(GoToSpot(*s['safe'], what='the safe spot'), label='walk', on_fail='abort'),
        # User 2026-11-19: the chair key does nothing straight after arriving
        Entry(Wait(s.get('chair_delay', SPLIT_ROAD_CHAIR_DELAY))),
        Entry(SitUntilHealed(s.get('chair_key', 'c')),
              on_fail='walk', retries=2, exhausted='next'),
        Entry(_to_platform(s), label='back', on_fail='back', retries=1, exhausted='abort'),
    ]
    if stay_on_channel():
        return Plan('Rest on the chair', entries)
    return Plan('Rest on the chair', entries + [
        Entry(ScoutPlatform(platform, 1), on_fail='channel', retries=0, exhausted='channel',
              then='done'),
        Entry(HopChannel(), label='channel', on_fail='channel', retries=SPLIT_ROAD_CHANNEL_ROUNDS - 1,
              exhausted='abort'),
        Entry(_to_platform(s), on_fail='abort'),
        Entry(ScoutPlatform(platform, SPLIT_ROAD_MIN_ENEMIES),
              on_fail='channel', retries=SPLIT_ROAD_MAX_HOPS - 1, exhausted='next', then='done'),
        Entry(Wait(0), label='done'),
    ])


def back_onto_platform(profile):
    """Down on the ground right of the platform while farming: jump back up."""
    return Plan('Back onto the platform', [
        Entry(_to_platform(profile.spots), label='climb', on_fail='climb', retries=2,
              exhausted='abort'),
    ])


class OffThePlatform:
    """
    True once the character has stood on the ground right of PLATFORM (any
    of the levels in TIERS: the safe ledge, or the bottom ground one step
    lower) for CONFIRM_SECONDS. loop.mp4: farming ended up on the safe ledge
    against the step, with a mushroom up top it could not reach and the map
    limit at its back; the patrol turned left/right for good. stuck.mp4: the
    same one level lower, at (0.224, 0.209), which the old check missed.
    """

    ready = True
    CONFIRM_SECONDS = 1.0
    STALE_SECONDS = 0.5
    PAST = 0.004                # this far past the platform's right end
    BELOW = 0.004               # at least this much lower than the platform
    GROUND = 0.012              # down to this far below the lowest tier
    REACH = 0.08                # this far right of the platform at most (then the hill)

    def __init__(self, platform, tiers):
        self.platform = platform
        if isinstance(tiers, (int, float)):
            tiers = [(platform[1], tiers)]
        self.lowest = max(t[1] for t in tiers)
        self.since = None

    def update(self, frame, now):
        position = getattr(config, 'player_pos', None)
        seen = getattr(config, 'player_pos_time', None) or 0.0
        if not position or tuple(position) == (0, 0) or time.monotonic() - seen > self.STALE_SECONDS:
            self.since = None
            return False
        x, y = position
        _, right, level = self.platform
        off = (right + self.PAST < x <= right + self.REACH
               and level + self.BELOW <= y <= self.lowest + self.GROUND)
        if not off:
            self.since = None
            return False
        self.since = now if self.since is None else self.since
        return now - self.since >= self.CONFIRM_SECONDS

    def reset(self):
        self.since = None


class HealthBelow:
    """True once HP has read at or under PERCENT for CONFIRM_SECONDS."""

    ready = True
    CONFIRM_SECONDS = 0.4

    def __init__(self, percent):
        self.percent = percent
        self.since = None

    def update(self, frame, now):
        from src.modules.health import is_game_focused
        try:
            hp = config.capture.health.current_percent()
        except Exception:
            hp = None
        if hp is None or hp > self.percent or not is_game_focused():
            self.since = None
            return False
        self.since = now if self.since is None else self.since
        if now - self.since >= self.CONFIRM_SECONDS:
            LOW_HP.message = f'HP is {hp:.0f}%.'
            return True
        return False

    def reset(self):
        self.since = None


class PlatformCleared:
    """
    True once the character has been on PLATFORM with no enemy in the mob
    tracker's sight for SECONDS (time off the platform doesn't count).
    """

    ready = True
    STALE_SECONDS = 0.5

    def __init__(self, platform, seconds=SPLIT_ROAD_CLEAR_SECONDS):
        self.platform = platform
        self.seconds = seconds
        self.since = None

    def update(self, frame, now):
        position = getattr(config, 'player_pos', None)
        seen = getattr(config, 'player_pos_time', None) or 0.0
        if (not position or time.monotonic() - seen > self.STALE_SECONDS
                or not on_platform(tuple(position), self.platform)):
            self.since = None
            return False
        tracker = getattr(getattr(config, 'capture', None), 'mob_tracker', None)
        if tracker is None or tracker.target() is not None or tracker.enemy_offsets():
            self.since = None
            return False
        self.since = now if self.since is None else self.since
        return now - self.since >= self.seconds

    def reset(self):
        self.since = None


class FirstLook:
    """True once: the first time the map is farmed this session."""

    ready = True

    def __init__(self):
        self.done = False

    def update(self, frame, now):
        if self.done:
            return False
        self.done = True
        return True

    def reset(self):
        pass


_SPLIT_ROAD_SPOTS = {
    # Recorded by the user 2026-11-19 ("Record position", minimap units)
    'left': (0.021, 0.176),         # the platform's left end
    'right': (0.197, 0.176),        # the platform's right end
    'safe': (0.209, 0.192),         # the chair spot: a step down right of the platform
    # stuck.mp4: the bottom ground one step below the safe ledge (stood at
    # (0.224, 0.209)); jump left from about here onto the ledge
    'lower_tiers': [(0.220, 0.209)],
    'chair_key': 'c',
}

LOW_HP = Event('Low HP', lambda: HealthBelow(SPLIT_ROAD_REST_HP), rest_on_chair,
               cooldown=5.0, message='HP is low.')

SPLIT_ROAD = MapProfile(
    'A Split Road',
    title=None,                     # no title crop yet: pick it in View > Status > Map
    spots=_SPLIT_ROAD_SPOTS,
    limits={'left': _SPLIT_ROAD_SPOTS['left'][0], 'right': _SPLIT_ROAD_SPOTS['right'][0]},
    after_channel=lambda profile: [Entry(_to_platform(profile.spots), on_fail='abort')],
    events=[
        LOW_HP,
        Event('Off the platform',
              lambda: OffThePlatform(_platform(_SPLIT_ROAD_SPOTS), _tiers(_SPLIT_ROAD_SPOTS)),
              back_onto_platform, cooldown=2.0,
              message='Down beside the platform.'),
        Event('First look', FirstLook, lambda p: platform_hunt(p, change_first=False),
              cooldown=0.0, message='Farming A Split Road.'),
        Event('Platform cleared', lambda: PlatformCleared(_platform(_SPLIT_ROAD_SPOTS)),
              platform_hunt, cooldown=10.0, message='The platform is clear.',
              changes_channel_only=True),
    ],
)

MAPS = [PIG_BEACH, SPLIT_ROAD]


# ---- Any map: nothing to farm ----

def empty_map_hop(profile):
    """
    The bot patrolled the whole map several times without seeing an enemy:
    change channel (preferring one without other players), then, on a known
    map, walk off its spawn point to the farming ground.
    """
    entries = [Entry(FindQuietChannel(max_channels=3), label='channel', on_fail='abort')]
    spots = getattr(profile, 'spots', {}) or {}
    if 'spawn' in spots and 'drop_direction' in spots:
        entries.append(Entry(WalkOff(spots['drop_direction'], spots['spawn']), on_fail='next'))
    after = getattr(profile, 'after_channel', None)
    if after is not None:
        entries.extend(after(profile))
    return Plan('Channel change (empty map)', new_channel=True, entries=entries)


EMPTY_MAP = Event('Empty map', lambda: None, empty_map_hop, cooldown=0.0,
                  message='No enemies in sight.', changes_channel_only=True)


# ---- Any map: other players crowding the character ----

def crowded_hop(profile):
    """2+ other players on the character's platform: move to a quieter channel."""
    plan = empty_map_hop(profile)
    plan.name = 'Channel change (players nearby)'
    return plan


# After a hop, give the new channel a minute before judging it again.
CROWDED = Event('Players nearby', PlayersNearby, crowded_hop, cooldown=60.0,
                message='Other players on my platform.', changes_channel_only=True)

# Watched on every map, known or not.
GLOBAL_EVENTS = [CROWDED]


class MapEventWatcher:
    INTERVAL = 0.5              # between event scans
    MAP_CHECK_SECONDS = 5.0     # between map (title) checks
    PIG_BEACH_PIT_MIN_Y = 0.255

    def __init__(self, maps=None):
        self.maps = MAPS if maps is None else maps
        self.map = None
        self._map_checked = float('-inf')
        self._lock = threading.Lock()
        self._pending = None
        self._quiet_until = {}      # event name -> time it may fire again
        self.paused = False         # while a plan runs: no scanning
        self.thread = None

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._main, name='map events',
                                           daemon=True)
            self.thread.start()

    def _main(self):
        from src.modules.watchdog import watchdog
        watchdog.register('map events', 30)
        while True:
            watchdog.heartbeat('map events')
            try:
                capture = getattr(config, 'capture', None)
                frame = getattr(capture, 'frame', None)
                if ENABLED and config.enabled and not self.paused and frame is not None:
                    self.scan(frame, time.monotonic())
            except Exception as error:
                print(f'\n[!] Map event check failed: {error}')
            time.sleep(self.INTERVAL)

    def scan(self, frame, now):
        if now - self._map_checked >= self.MAP_CHECK_SECONDS:
            self._map_checked = now
            picked = chosen_map()
            found = next((m for m in self.maps if m.name == picked), None) if picked else None
            if found is None:
                found = next((m for m in self.maps if m.visible(frame)), None)
            if found is not self.map:
                if found is not None:
                    names = ', '.join(e.name for e in found.events)
                    print(f'\n[~] Map: {found.name} (watching for: {names}).')
                    if not self._load_mob_profile(found):
                        return
                self.map = found
        if self.map is None:
            if self._pig_beach_pit_fallback(frame, now):
                return
            if self._pig_beach_iron_hog_fallback(frame, now):
                return
        events = (self.map.events if self.map is not None else []) + GLOBAL_EVENTS
        for event in events:
            if event.changes_channel_only and stay_on_channel():
                continue        # "Stay on this channel": nothing to do about it
            watch = event.watch
            if not watch.ready or now < self._quiet_until.get(event.name, float('-inf')):
                continue
            if watch.update(frame, now):
                watch.reset()
                with self._lock:
                    if self._pending is None:
                        self._pending = (self.map, event)
                return

    @staticmethod
    def _load_mob_profile(profile):
        name = getattr(profile, 'mob_profile', None)
        if not name:
            return True
        capture = getattr(config, 'capture', None)
        tracker = getattr(capture, 'mob_tracker', None)
        if tracker is None:
            error = 'the mob tracker is unavailable'
        else:
            try:
                tracker.load_profile(name, preserve_player_badge=True)
            except (OSError, ValueError) as exception:
                error = str(exception)
            else:
                print(f'\n[~] Loaded enemy profile {name!r} for {profile.name}.')
                return True
        config.enabled = False
        print(f'\n[!] Could not load enemy profile {name!r} for {profile.name}: {error}. '
              'Auto Maple is paused to avoid attacking scenery; fix the profile issue '
              'and re-enable the bot.')
        return False

    def _pig_beach_pit_fallback(self, frame, now):
        capture = getattr(config, 'capture', None)
        tracker = getattr(capture, 'mob_tracker', None)
        level_farming_enabled = getattr(tracker, 'level_farming_enabled', None)
        position = getattr(config, 'player_pos', None)
        if (not callable(level_farming_enabled) or not level_farming_enabled()
                or not position or position[1] < self.PIG_BEACH_PIT_MIN_Y):
            event = next(event for event in PIG_BEACH.events if event.name == 'Pit')
            event.watch.reset()
            return False

        event = next(event for event in PIG_BEACH.events if event.name == 'Pit')
        if not event.watch.update(frame, now):
            return False
        event.watch.reset()
        with self._lock:
            if self._pending is None:
                # The title-template match can fail on a scaled client; the pig
                # profile and lower-right minimap position still identify this pit.
                self._pending = (PIG_BEACH, event)
                return True
        return False

    def _pig_beach_iron_hog_fallback(self, frame, now):
        event = next(event for event in PIG_BEACH.events if event.name == 'Iron Hog')
        if (not event.watch.ready
                or now < self._quiet_until.get(event.name, float('-inf'))
                or not event.watch.update(frame, now)):
            return False
        event.watch.reset()
        with self._lock:
            if self._pending is None:
                # The title template can fail on a scaled client; the hog sprite is distinctive.
                self._pending = (PIG_BEACH, event)
                return True
        return False

    def take(self):
        with self._lock:
            pending, self._pending = self._pending, None
        return pending

    def post(self, event):
        """Queues EVENT for the current map (from outside the scans)."""
        with self._lock:
            if self._pending is None:
                self._pending = (self.map, event)
                return True
        return False

    def finished(self, event, now):
        self._quiet_until[event.name] = now + event.cooldown
        if event.watch is not None:
            event.watch.reset()
        self._map_checked = float('-inf')   # a channel change can mean anything


class MapEventManager:
    """The bot's side: runs the plan of an event the watcher saw."""

    def __init__(self, bot, watcher=None, controls=None):
        self.bot = bot
        self.watcher = watcher or MapEventWatcher()
        self.io = controls or Controls(bot)
        self.plan = None
        self.event = None

    def start(self):
        self.watcher.start()

    @property
    def active(self):
        return self.plan is not None

    def request_empty_map(self, laps, message=None):
        """The bot saw no enemy in LAPS laps of the map: change channel."""
        if not ENABLED or self.plan is not None:
            return
        message = message or f"No enemies after {laps} lap{'s' if laps != 1 else ''} of the map."
        if stay_on_channel():
            print(f'\n[~] {message} But "Stay on this channel" '
                  'is on: staying and patrolling on.')
            return
        EMPTY_MAP.message = message
        self.watcher.post(EMPTY_MAP)

    def limits(self):
        """The current map's fixed walls ({} on a map without a profile)."""
        profile = self.watcher.map if ENABLED else None
        return getattr(profile, 'limits', {}) or {}

    def service(self):
        """True while a plan has the controls (the bot does nothing else)."""
        if self.plan is None:
            if not ENABLED:
                return False
            pending = self.watcher.take()
            if pending is None:
                return False
            profile, event = pending
            self.event = event
            self.profile = profile
            self.plan = event.plan_factory(profile)
            self.watcher.paused = True
            self.bot._prepare_for_plan()
            self.io.log(f'{event.message} Starting: {self.plan.name}.')
        status = self.plan.tick(self.io)
        if status == RUNNING:
            return True
        self._finish(status)
        return True

    def _finish(self, status):
        plan, event = self.plan, self.event
        profile = self.__dict__.get('profile')
        self.plan = self.event = None
        self.io.release()
        self.watcher.paused = False
        self.watcher.finished(event, self.io.now())
        self.bot._resume_after_plan(new_channel=plan.new_channel)
        if getattr(profile, 'after_channel', None) is not None:
            # (the fall recovery's platform: where the plan left it, not
            # where the bot was switched on)
            forget = getattr(self.bot, '_forget_patrol_home', None)
            if forget is not None:
                forget()
        if status == DONE:
            self.io.log(f'{plan.name} done; back to farming.')
        else:
            config.enabled = False
            print(f'\n[!] {plan.name} failed ({plan.reason}). Auto Maple is paused '
                  f'so the character is not left fighting the {event.name}; check the game.')
            from src.modules import phone_alert
            if phone_alert.wants('bot_paused'):
                phone_alert.send(f'🛑 Auto Maple paused itself: {plan.name} failed '
                                 f'({plan.reason}). Check the game.', photo=True)

    def cancel(self):
        """The bot was switched off: drop the plan (keys are released by the bot)."""
        if self.plan is not None:
            print(f'\n[~] {self.plan.name} cancelled.')
            self.plan.stop(self.io)
            self.plan = self.event = None
            self.watcher.paused = False
            self.io.expect_black_screen(0)
