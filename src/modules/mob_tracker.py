"""Template-based tracking of nearby mobs on the player's current platform."""

import os
import pickle
import re
import shutil
import threading
import time

import cv2
import numpy as np

from src.common import config as _config
from src.common.interfaces import Configurable
from src.modules.health import is_game_focused


class MobTracker(Configurable):
    DEFAULT_CONFIG = {
        'Enabled': False,
        'Range': 140,
        'Threshold': 78,
        'Alignment Y': 0,
        'Enemy count': 1,
        'Enemy alignments': '',
        'Enemy speeds': '',
        'Focus configured': False,
        'Focused enemies': '',
        # The group skill (Q). Off for a character that does not have it: then
        # piles are fought one enemy at a time. (A settings file from before
        # this option loads it as '', which counts as on.)
        'Group skill': True,
        # A ranged pull (the magician's Magic Missile, magic-class.mp4): cast
        # at an enemy out of melee reach, let it come, finish it with the melee
        # swing. Off unless ticked: a settings file or profile without it (the
        # beginner's) loads it as off, so that character never presses the key.
        'Magic missile': False,
        'Missile key': 'e',
        'Missile range': 270,    # px; the clip's missile hit a snail 260 px away
        # Long maps with ramps (Perion Street Corner, stumps): walk the map end
        # to end clearing what is ahead, treat enemies up or down a ramp as on
        # the character's level, keep the attack held and step in between
        # swings. Off unless ticked (a settings file without it loads as off).
        'Sweep patrol': False,
        # Maps with drops that land up on scenery at the map's ends (the hay
        # bales left and right on the Henesys snail map, msg-boc.png): with
        # nothing to fight, the patrol jumps up where its walk stops and taps
        # loot before it turns. Off unless ticked.
        'Edge loot jumps': False,
        # Patrol with nothing in sight: hops now and then, short stops, legs
        # of different lengths, instead of the same walk every lap.
        'Idle variety': False
    }
    MISSILE_RANGE_LIMITS = (100, 400)
    TEMPLATE_DIRECTORY = '.settings'
    PLAYER_TEMPLATE_PATH = os.path.join(TEMPLATE_DIRECTORY, 'mob_player_badge.png')
    MOB_TEMPLATE_PATH = os.path.join(TEMPLATE_DIRECTORY, 'mob_target.png')
    PLAYER_SEARCH_HEIGHT_RATIO = 0.88
    PLAYER_MATCH_MIN_THRESHOLD = 0.60
    MOB_MATCH_MIN_THRESHOLD = 0.52
    MOB_MATCH_THRESHOLD_MARGIN = 0.20
    # Game sprites are always drawn at their real size; trying other sizes only
    # produced false matches (small copies of a sample fit clouds and bushes).
    MOB_TEMPLATE_SCALES = (1.0,)
    # A sample is a box around the enemy, so it includes scenery that changes as the
    # player moves. Matching uses only the enemy's own pixels (see _sample_mask), then
    # checks the colours really are the enemy's: soft cloud shading correlates with a
    # round mushroom's shape, but it is the wrong colour.
    MASK_MIN_FRACTION = 0.2
    MASK_MAX_FRACTION = 0.9
    COLOR_TOLERANCE = 50        # mean absolute BGR difference over the enemy's pixels
    JUMP_HEIGHT = 130           # enemies this far above the ground line are mid-jump
    GROUND_MARGIN = 12
    MASK_MATCH_STEP = 2         # shape matching runs at 1/2 size for speed
    SAME_ENEMY_OVERLAP = 0.6    # box overlap that makes two samples' matches one enemy
    MATCH_INTERVAL = 6
    FAST_MATCH_INTERVAL = 3
    ATTACK_COOLDOWN = 0.4
    ATTACK_RANGE = 60
    VERTICAL_TOLERANCE = 16
    SEARCH_VERTICAL_TOLERANCE = 64
    GROUP_ATTACK_VERTICAL_TOLERANCE = 64
    TARGET_CONFIRMATIONS = 2
    TARGET_LOST_GRACE_SECONDS = 1.5
    # A locked enemy that vanishes is most often dead, not hidden: keep swinging at
    # its last spot only this long, then just wait out the rest of the grace period.
    HIDDEN_ATTACK_SECONDS = 0.4
    LOW_HEALTH_RATIO = 0.15     # a locked enemy this low that vanishes was killed
    # A sample whose enemy could not be cut out is matched through an oval that still
    # holds some scenery; such a sample needs this much more score to count.
    LOOSE_MASK_MARGIN = 0.12
    # An enemy kept in attack range this long without its HP bar showing is not
    # taking hits, so it is not an enemy (scenery that resembles a sample). Its spot
    # is then ignored for a while.
    NO_DAMAGE_SECONDS = 2.5
    IGNORE_SPOT_SECONDS = 8.0
    IGNORE_SPOT_RADIUS = 30
    TARGET_LOCK_RADIUS = 32
    TARGET_LOCK_X_RADIUS = 48    # a locked enemy may jump; follow it by horizontal position
    HP_BAR_SEARCH_HEIGHT = 42
    HP_BAR_MAX_WIDTH = 120
    HP_BAR_MAX_CENTER_OFFSET = 48
    HP_BAR_MIN_WIDTH = 30
    HP_BAR_MATCH_X = 34         # a bar this close (sideways) to an enemy is that enemy's
    HP_BAR_SCORE = 0.5          # reported "match" for an enemy known only by its HP bar
    # With no enemy locked the tracker looks this far along the player's level to
    # pick the next one (the Range setting still applies while fighting).
    SCOUT_RANGE = 600
    # Choosing the next enemy: nearest first, but one that is already hurt (its HP
    # bar shows) is worth walking this much further for, and one behind the player
    # costs a turn.
    HURT_BONUS = 150
    TURN_PENALTY = 60
    GROUP_MIN_ENEMIES = 3
    GROUP_HORIZONTAL_RADIUS = 130   # widest pile (centre to centre) that counts as one group
    GROUP_SKILL_REACH = 190         # how far from the player the group skill (Q) still hits
    GROUP_VERTICAL_RADIUS = 64
    FAST_VERTICAL_SEARCH_EXTRA = 48
    FAST_VERTICAL_ATTACK_TOLERANCE = 48
    FAST_LOCK_RADIUS = 64
    SPEED_PROFILES = ('normal', 'fast', 'jumper')
    # ---- Slow walkers ('normal': snails, shrooms) - knock-back-dmg.mp4 ----
    # The character walked into (and through) snails before it swung: they
    # bite on touch (~35 px centre to centre), a scan came every 0.2-0.3 s
    # and a walk covers ~125 px/s, so "in range" at 60 px was seen too late.
    NORMAL_ATTACK_RANGE = 70        # start swinging from here
    NORMAL_RANGE_HYSTERESIS = 18    # knocked back this far past it: keep swinging...
    NORMAL_HYSTERESIS_SECONDS = 0.8 # ...this long after it was last in range
    NORMAL_WHIFF_SECONDS = 0.9      # swinging this long, no HP bar: step closer
    NORMAL_WHIFF_RANGE = 52
    # memory-issue.mp4 (magician): swings at a red snail 55-76 px away left its
    # HP bar at ~70% for a second; the snail crawled in and bit. The bar was
    # up from earlier hits, so the "no HP bar" rule above never fired.
    NORMAL_BAR_WHIFF_SECONDS = 1.6  # swinging this long, bar up but not dropping: step in
    NORMAL_HIT_DROP = 0.04          # HP ratio drop that counts as a hit (readings jitter)
    NORMAL_BAR_IDLE_RESET = 0.6     # s of not swinging that restarts that count
    NORMAL_CLOSING_MIN = 40         # px/s: walking at it (or it at us)
    NORMAL_KEY_LAG = 0.08           # s from the scan to the character stopping
    NORMAL_STOP_LEAD_MAX = 0.45     # s of walking allowed for, at most
    NORMAL_EARLY_MAX = 125          # never stop for it from further than this
    NORMAL_QUICK_LOCK = 130         # this close (or hurt): lock on the first sighting
    NORMAL_QUICK_LOCK_SCORE = 0.75  # ...if it matches this well (else: seen twice)
    # The nearest enemy inside THREAT_RANGE is fought first, whatever the
    # focus or the lock says: the bot walked through a blue snail to get at
    # a red one, and swung at a far one while a near one bit.
    THREAT_RANGE = 95
    THREAT_MARGIN = 15              # nearer than the current target by this: switch
    # memory-issue.mp4: with snails on both sides the nearest one kept changing
    # sides and the bot turned every swing, so its back was always to a snail.
    # A target on the other side takes over only when it is about to touch and
    # the one in front is out of reach; then no turning back for a while.
    NORMAL_BEHIND_TOUCH = 40        # px: the one behind is this close
    NORMAL_SIDE_COMMIT = 1.5        # s after turning before turning back
    # strike-issue.mp4: a hit landed, then the bot walked off (knocked-back
    # snail just out of range, or the snail on the other side picked) and
    # walked into a snail. After a swing, keep swinging that way this long
    # while an enemy is still this close on that side.
    NORMAL_FOLLOW_UP = 1.0          # s (~1.3 swings)
    NORMAL_FOLLOW_RANGE = 110       # px
    # knock-back-dmg-2.mp4: a snail that crawled into the character was swung
    # at (and missed) for 2.5 s, taken for scenery and ignored; the ignored
    # spot was kept in screen px, which the camera carries along with the
    # character, so "15 px to my left" stayed ignored for 8 s while it bit.
    KNOCKBACK_RISE = 8              # badge this far above its ground y: knocked back
    REAL_MOVE_PX = 20               # a lock that moved this far on the map is no scenery
    REAL_NEAR_PX = 90               # knocked back with the lock this close: it bit us
    KNOCK_MEMORY = 0.6              # s a knock-back still counts for that
    PX_PER_MINIMAP = 6000.0         # until the bot has measured it
    NORMAL_OVERLAP_PX = 40          # last seen this close: hidden behind the character
    # knock-back-dmg-3.mp4: the gap closes at walk + crawl speed while the
    # character walks, but only at crawl speed once it stops. Stopping on the
    # sum and then walking again on the crawl alone walked the character into
    # a shroom that was coming at it. One swing takes ~0.75 s (a beginner's
    # club), and lands ~0.27 s after the press.
    NORMAL_WALK_SPEED = 125         # px/s the character walks
    NORMAL_COMING_MIN = 15          # px/s of its own: it is coming at the character
    NORMAL_SWING_WINDUP = 0.27      # s from the press to the hit
    NORMAL_SWING_REACH = 62         # where a swing on a coming enemy should land
    NORMAL_WAIT_MAX = 160           # coming at us from this close: stand and let it come
    SCENERY_IGNORE_SECONDS = 60.0   # proven scenery (a portal), kept by its map spot
    SCENERY_CLUSTER_PX = 45         # matches this close to it are the same thing
    FAR_LOCK_PX = 200               # a lock this far away...
    # bug.mp4: standing in the portal, its 2-3 matches on both sides took
    # turns as "nearest", each switch a new lock with a fresh 2.5 s scenery
    # timer: the bot swung left and right at it for good. The area is timed
    # instead: swinging this long, no HP bar near and no knock-back: all of
    # it is scenery. A knock-back clears ignored spots by the character.
    BLANK_AREA_SECONDS = 2.5
    BLANK_AREA_PX = 135             # the scenery around the character, this far
    BLANK_GRACE = 1.0               # (old: not swinging this long restarted the count)
    # edge-portal-issue.mp4: at the right edge the bot walked back and forth
    # between the portal (~78% match) and the decorative mushrooms (~59%),
    # swinging at each in turn. Every walk between swings was over 1 s, or the
    # lock was "hidden" behind the character, so the count above restarted and
    # never got to 2.5 s. Now it is a running total: bar-less swinging adds
    # (hidden-lock swings too), not swinging only drains it slowly, and only an
    # HP bar, a knock-back or a proven-real lock clears it.
    BLANK_DRAIN = 0.25              # total lost per second of not swinging
    BLANK_MAX_STEP = 0.4            # s one scan can add (lag spikes don't count)
    # memory-leak.png (2026-11-15): stuck swinging at a portal / flowers for
    # good. The count above is cleared by weak proof too (a badge bump read as
    # a knock-back, a portal's animation "moving" 20 px), and once a lock is
    # marked real that sticks, so the count never fills again. Backstop: only
    # an HP bar clears this one. A hit enemy always shows its bar; FUTILE_SECONDS
    # of bar-less swinging means nothing is being hit. (A real snail biting the
    # character still gets its spot un-ignored by the knock-back.)
    FUTILE_SECONDS = 6.0
    # The same scenery caught again: ignored twice as long each time (up to
    # this), so hours by one portal don't cost a swing burst every minute.
    SCENERY_REPEAT_MAX = 600.0
    SCENERY_HISTORY_MAX = 40
    SCENERY_SPOT_RADIUS = 45        # (map spots: the map scale is only a guess)
    CAMERA_CLAMP_PX = 150           # the character this far off centre: camera stopped
    SCALE_MIN_MOVE = 0.012          # minimap units moved per scale sample
    FAR_LOCK_MARGIN = 100           # ...gives way to an enemy this much closer
    NORMAL_HIDDEN_SECONDS = 0.7     # ...keep swinging at it (not "killed") this long
    # 'jumper' (pigs, ribbon pigs): fast walkers that hop about 85 px up and land
    # within ~0.7 s. Measured from assets/pig_beach_training.mp4.
    JUMPER_LOCK_RADIUS = 80         # a hop carries a pig ~90 px sideways
    # A pig is ~70 px wide: at the normal 60 px the swing only starts at body
    # contact and the character takes touch damage. Start swinging earlier, also
    # when its heading puts it in range within the lead time.
    JUMPER_ATTACK_RANGE = 85
    JUMPER_RANGE_HYSTERESIS = 20    # keep swinging until it is this much past range
    JUMPER_REACH_HEIGHT = 60        # only near the top of a hop is a swing a waste
    JUMPER_LEAD_SECONDS = 0.2       # walk toward where it will be, not where it was
    JUMPER_MAX_LEAD = 60
    JUMPER_NO_DAMAGE_SECONDS = 3.5  # hops make some swings miss; be slower to give up
    # Swing timing (assets/attack-issue-2.mp4): a pig closing at ~150 px/s was
    # only swung at from ~70 px, i.e. at body contact, so it walked through the
    # character and hit it. The stab needs ~0.25 s to come out plus a scan of
    # latency: start it when the pig will be in reach by then, and stop walking
    # into a pig that is already coming (let it walk onto the blade).
    JUMPER_SWING_LEAD = 0.5         # s: allow a slower melee swing a little more wind-up
    JUMPER_CLOSING_MIN = 40         # px/s: slower than this is not "coming at us"
    JUMPER_EARLY_MAX = 170          # never swing from further than this
    JUMPER_BRACE_SECONDS = 0.9      # closing pig this close in time: stand, don't walk
    JUMPER_WHIFF_SECONDS = 1.2      # give a slow swing time to land before correcting
    JUMPER_WHIFF_RANGE = 60          # no hit feedback: step in to reliable melee range
    # A pig that walks through the character reappears on the other side, more
    # than JUMPER_LOCK_RADIUS from where it was locked: it is the same pig. And a
    # lost close pig is overlapping/crossed/dead - never walk to its last spot.
    JUMPER_CROSS_NEAR = 125         # locked this close to the player when it vanished
    JUMPER_CROSS_RADIUS = 140       # ...then anything this close to the player is it
    JUMPER_LOST_GRACE_SECONDS = 0.45
    JUMPER_QUICK_LOCK = 130         # this close (or hurt): lock on the first sighting
    # Pigs pile up and hop over each other: two already make Q worth it, one
    # mid-hop still belongs to the pile, and two side by side (closer than a pig
    # is wide) must still match as two.
    JUMPER_GROUP_MIN_ENEMIES = 2
    JUMPER_GROUP_VERTICAL_RADIUS = 110
    JUMPER_MATCH_SEPARATION = 0.55  # peaks this fraction of a sample apart are two pigs
    LEDGE_MEMORY_SECONDS = 1.5      # ledge_enemies() keeps a sighting this long
    # Another level in reach of the search band (e.g. the raised step of the Pig
    # Beach ground floor, ~65 px): an enemy that stays this high, at a steady
    # height, for long enough is standing up there. A hop goes higher than the
    # step, but its height changes the whole time it lasts.
    LEDGE_MIN_HEIGHT = 28
    LEDGE_CONFIRM_SECONDS = 0.5
    LEDGE_Y_JITTER = 10
    LEVEL_SPOT_X = 60               # spot matching between scans (enemy + camera motion)
    LEVEL_SPOT_Y = 14
    LEVEL_SPOT_EXPIRY = 0.8         # far scans (that see the step) are not every scan
    ELEVATED_DROP_SECONDS = 1.0     # a locked enemy up this long is not mid-hop
    # Once one enemy has been confirmed up there, the step's height is known, and
    # an enemy holding still at that height is recognised much sooner (a hop only
    # passes through that height).
    LEDGE_HEIGHT_MATCH = 14
    LEDGE_FAST_CONFIRM_SECONDS = 0.25
    # Two-level farming (jumper maps): standing up on the step, the search also
    # looks this far below the player's feet, to see enemies left on the ground
    # beneath. They are never targets from up there (the bot walks down to them);
    # one counts once it holds a steady depth, so the character's own jumps
    # (everything below drifts while it is in the air) never count.
    LOWER_LOOK = 100
    LOWER_CONFIRM_SECONDS = 0.4
    LOWER_MEMORY_SECONDS = 1.5      # lower_enemies() keeps a sighting this long
    HP_BAR_HEAD_GAP = 8             # an HP bar sits about this far above the head
    HP_BAR_FLOAT = 34               # ...or up to this much higher (until measured)
    ELEVATED_PENALTY = 200          # choosing: a raised enemy may be out of reach
    PROFILE_DIRECTORY = os.path.join(TEMPLATE_DIRECTORY, 'profiles')
    AUTOSAVE_PROFILE = '_autosave'

    def __init__(self):
        self._lock = threading.Lock()
        self._player_template = None
        self._mob_templates = []
        self._mask_cache = {}
        self._attack_direction = None
        self._target_offset_x = None
        self._target_vertical_offset = None
        self._target_attackable = False
        self._group_count = 0
        self._attack_is_group = False
        self._locked_target = None
        self._pending_target = None
        self._target_missed_scans = 0
        self._target_missed_since = None
        self._last_attack_time = 0
        self._group_in_range = False
        self._locked_health = None
        self._in_range_since = None
        self._ignored_spots = []
        self._player_position = None
        self._sweep_direction = 'right'
        self._blocked_side = None
        self._level_spots = []
        self._lower_spots = []
        self._lower_seen = None
        self._locked_elevated_since = None
        self._locked_velocity = 0.0
        self._locked_seen = None
        self._target_waiting = False
        self._frame_count = 0
        self._status = 'Calibrate the player badge and an enemy sample to begin.'
        self._load_error = ''
        super().__init__('mob_tracker')
        self._load_templates()

    def settings(self):
        with self._lock:
            return self.config.copy()

    def update_settings(self, enabled, distance, threshold):
        try:
            distance = int(distance)
            threshold = int(threshold)
        except (TypeError, ValueError) as error:
            raise ValueError('Mob range and match threshold must be whole numbers.') from error
        if not 20 <= distance <= 600:
            raise ValueError('Mob range must be between 20 and 600 pixels.')
        if not 50 <= threshold <= 99:
            raise ValueError('Match threshold must be between 50 and 99%.')

        with self._lock:
            self.config.update({
                'Enabled': bool(enabled),
                'Range': distance,
                'Threshold': threshold
            })
            self._attack_direction = None
            self._target_offset_x = None
            self._target_vertical_offset = None
            self._target_attackable = False
            self._group_count = 0
            self._attack_is_group = False
            self._locked_target = None
            self._pending_target = None
            self._target_missed_scans = 0
            self._target_missed_since = None
            self._last_attack_time = 0
            self._group_in_range = False
            self.save_config()

    def set_templates(self, frame, player_bounds, mob_bounds):
        height, width = frame.shape[:2]
        if (len(mob_bounds) == 2
                and isinstance(mob_bounds[0][0], (int, float))):
            mob_bounds = [mob_bounds]
        if not mob_bounds:
            raise ValueError('Select at least one enemy sample.')

        def crop(bounds, name, min_width, min_height):
            (x1, y1), (x2, y2) = bounds
            if (x1 < 0 or y1 < 0 or x2 > width or y2 > height
                    or x2 - x1 < min_width or y2 - y1 < min_height):
                raise ValueError(f'The selected {name} sample is too small or outside the screen.')
            return frame[y1:y2, x1:x2, :3].copy()

        player_template = cv2.cvtColor(
            crop(player_bounds, 'player badge', 20, 6),
            cv2.COLOR_BGR2GRAY
        )
        mob_templates = [
            (
                crop(bounds, f'enemy {index + 1}', 12, 12),
                round(
                    (bounds[0][1] + bounds[1][1]
                     - player_bounds[0][1] - player_bounds[1][1]) / 2
                )
            )
            for index, bounds in enumerate(mob_bounds)
        ]
        os.makedirs(self.TEMPLATE_DIRECTORY, exist_ok=True)
        if not cv2.imwrite(self.PLAYER_TEMPLATE_PATH, player_template):
            raise OSError(f'Could not save player badge sample to {self.PLAYER_TEMPLATE_PATH}.')
        for index, (mob_template, _) in enumerate(mob_templates):
            path = self._mob_template_path(index)
            if not cv2.imwrite(path, mob_template):
                raise OSError(f'Could not save enemy sample to {path}.')

        with self._lock:
            self._player_template = player_template
            self._mob_templates = mob_templates
            self.config['Focus configured'] = False
            self.config['Focused enemies'] = ''
            self.config['Alignment Y'] = mob_templates[0][1]
            self.config['Enemy count'] = len(mob_templates)
            self.config['Enemy alignments'] = ','.join(
                str(alignment_y) for _, alignment_y in mob_templates
            )
            self.config['Enemy speeds'] = ','.join(
                'normal' for _ in mob_templates
            )
            self._attack_direction = None
            self._target_offset_x = None
            self._target_vertical_offset = None
            self._target_attackable = False
            self._group_count = 0
            self._attack_is_group = False
            self._locked_target = None
            self._pending_target = None
            self._target_missed_scans = 0
            self._target_missed_since = None
            self._last_attack_time = 0
            self._group_in_range = False
            self._load_error = ''
            self._status = (
                f'{len(mob_templates)} enemy sample(s) saved. '
                'Enable auto-attack and press the Start/stop key to test.'
            )
            self.save_config()

    def has_player_sample(self):
        with self._lock:
            return self._player_template is not None

    def player_template(self):
        with self._lock:
            if self._player_template is None:
                return None
            return self._player_template.copy()

    def find_player_bounds(self, frame):
        template = self.player_template()
        if template is None or frame is None or frame.ndim != 3:
            return None

        with self._lock:
            threshold = max(
                self.PLAYER_MATCH_MIN_THRESHOLD,
                self.config['Threshold'] / 100 - 0.18
            )
        gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
        search_height = max(
            1, int(gray.shape[0] * self.PLAYER_SEARCH_HEIGHT_RATIO)
        )
        match = self._best_match(gray[:search_height], template)
        if match is None or match[1] < threshold:
            return None

        x, y = match[0]
        return ((x, y), (x + template.shape[1], y + template.shape[0]))

    def has_samples(self):
        with self._lock:
            return self._player_template is not None and bool(self._mob_templates)

    def enemy_samples(self):
        with self._lock:
            return [template.copy() for template, _ in self._mob_templates]

    def focused_enemy_samples(self):
        with self._lock:
            if not self.config.get('Focus configured'):
                return set(range(len(self._mob_templates)))
            try:
                indices = {
                    int(value)
                    for value in self.config.get('Focused enemies', '').split(',')
                    if value
                }
            except (AttributeError, ValueError):
                return set()
            return {
                index for index in indices
                if 0 <= index < len(self._mob_templates)
            }

    def enemy_speed_profiles(self):
        with self._lock:
            return self._enemy_speed_profiles_locked()

    def _enemy_speed_profiles_locked(self):
        values = self.config.get('Enemy speeds', '').split(',')
        return [
            values[index] if index < len(values) and values[index] in self.SPEED_PROFILES
            else 'normal'
            for index in range(len(self._mob_templates))
        ]

    def set_enemy_speed_profile(self, index, profile):
        if profile not in self.SPEED_PROFILES:
            raise ValueError(
                'Enemy speed profile must be one of: '
                + ', '.join(self.SPEED_PROFILES) + '.'
            )
        with self._lock:
            if not 0 <= index < len(self._mob_templates):
                raise ValueError(f'No enemy sample exists at index {index}.')
            profiles = self._enemy_speed_profiles_locked()
            profiles[index] = profile
            self.config['Enemy speeds'] = ','.join(profiles)
            self._locked_target = None
            self._pending_target = None
            self.save_config()

    def set_sweep_direction(self, direction):
        if direction not in {'left', 'right'}:
            raise ValueError("Sweep direction must be 'left' or 'right'.")
        with self._lock:
            self._sweep_direction = direction

    def set_blocked_side(self, direction):
        """
        The bot walked into a wall or a step on DIRECTION's side (None: no longer).
        Enemies past it, beyond attack range, cannot be reached on foot.
        """
        if direction not in {None, 'left', 'right'}:
            raise ValueError("Blocked side must be 'left', 'right' or None.")
        with self._lock:
            if direction == self.__dict__.get('_blocked_side'):
                return
            self._blocked_side = direction
            if (direction is not None and self._locked_target is not None
                    and self._player_position is not None):
                offset = self._locked_target[1] - self._player_position[0]
                if (offset > self.ATTACK_RANGE if direction == 'right'
                        else offset < -self.ATTACK_RANGE):
                    self._release_lock_locked()

    def ledge_enemies(self):
        """
        Enemies standing on a raised step (out of reach on foot), as
        (count, screen x offset of the nearest one), or (0, None). A sighting is
        kept for LEDGE_MEMORY_SECONDS, since scans far enough to see the step
        only run now and then.
        """
        with self._lock:
            seen = self.__dict__.get('_ledge_seen')
            if seen is None or time.monotonic() - seen[0] > self.LEDGE_MEMORY_SECONDS:
                return 0, None
            return seen[1], seen[2]

    def consume_kills(self):
        """Kills seen since the last call, as [(time, screen x offset)]."""
        with self._lock:
            kills = self.__dict__.get('_kill_events') or []
            self._kill_events = []
            return kills

    def lower_enemies(self):
        """
        Enemies on a level below the player's (seen from up on a raised step), as
        (count, screen x offset of the nearest one), or (0, None).
        """
        with self._lock:
            seen = self.__dict__.get('_lower_seen')
            if seen is None or time.monotonic() - seen[0] > self.LOWER_MEMORY_SECONDS:
                return 0, None
            return seen[1], seen[2]

    def level_farming_enabled(self):
        """Farming on both levels of a raised step: only for 'jumper' enemies."""
        with self._lock:
            return 'jumper' in self._enemy_speed_profiles_locked()

    def ground_clear_seconds(self):
        """
        How long since an enemy was last seen in reach on the player's own level
        (not on a raised step, not past a wall). Infinite if none seen yet.
        """
        with self._lock:
            seen = self.__dict__.get('_ground_seen')
            return float('inf') if seen is None else time.monotonic() - seen

    def ledge_height(self):
        """Height (px) of the raised step above the player's ground, once known."""
        with self._lock:
            return self.__dict__.get('_ledge_height')

    def _at_ledge_height(self, item, ground_y):
        """An enemy at the known step height: standing on the step (a hop only
        passes through it, so a hopping one is back in reach a moment later)."""
        return self._matches_ledge_height(self._elevation(item, ground_y))

    def _matches_ledge_height(self, elevation):
        ledge = self.__dict__.get('_ledge_height')
        return ledge is not None and abs(elevation - ledge) <= self.LEDGE_HEIGHT_MATCH

    def group_skill_enabled(self):
        """False when the character has no group skill (Q): never press it."""
        with self._lock:
            return self._group_skill_locked()

    def _group_skill_locked(self):
        return self.config.get('Group skill', True) is not False

    def set_group_skill(self, enabled):
        with self._lock:
            self.config['Group skill'] = bool(enabled)
            self._attack_is_group = False
            self._group_in_range = False
            self._group_count = 0
            self.save_config()

    def missile_settings(self):
        """{'enabled', 'key', 'range'} of the ranged pull (Magic Missile)."""
        with self._lock:
            config = self.config
            enabled = config.get('Magic missile') is True   # '' (old file) = off
            key = str(config.get('Missile key') or 'e').strip().lower() or 'e'
            try:
                reach = int(config.get('Missile range') or 270)
            except (TypeError, ValueError):
                reach = 270
        low, high = self.MISSILE_RANGE_LIMITS
        return {'enabled': enabled, 'key': key, 'range': max(low, min(high, reach))}

    def set_missile(self, enabled=None, key=None, reach=None):
        from src.common.vkeys import KEY_MAP
        updates = {}
        if enabled is not None:
            updates['Magic missile'] = bool(enabled)
        if key is not None:
            key = str(key).strip().lower()
            if key not in KEY_MAP:
                raise ValueError(f'{key!r} is not a supported key.')
            updates['Missile key'] = key
        if reach is not None:
            try:
                reach = int(reach)
            except (TypeError, ValueError) as error:
                raise ValueError('Missile range must be a whole number.') from error
            low, high = self.MISSILE_RANGE_LIMITS
            if not low <= reach <= high:
                raise ValueError(f'Missile range must be between {low} and {high} px.')
            updates['Missile range'] = reach
        with self._lock:
            self.config.update(updates)
            self.save_config()

    # ---- Sweep patrol (perion_street_corner.png, 2026-11-17) ----
    # The ground there is one long strip that rises and falls in ramps, so an
    # enemy a little way off stands higher or lower than the character. The
    # tracker read those as "up on a ledge" or never looked that low. With the
    # option on, an enemy inside the cone a ramp could make (SLOPE_RISE px up
    # or down per px sideways) counts as standing on the character's level.
    SLOPE_RISE = 0.7
    SLOPE_BASE = 14                 # px, on top of that
    SLOPE_MAX = 120                 # never further up or down than this
    SLOPE_LOOK = 90                 # search this far below the feet too
    # The sweep goes one way to the map's end, then back. Enemies behind it by
    # more than this wait for the way back (turning for them is what made the
    # character go left, right, left, right).
    SWEEP_BEHIND_PX = 100
    # Once swinging, the attack stays held while the enemy is this close on
    # that side: each hit knocks a stump back ~30-50 px, and letting go to
    # walk after it lost a swing every time. The bot steps in instead.
    SWEEP_HOLD_RANGE = 120

    def sweep_patrol_enabled(self):
        """True for the end-to-end sweep on a long, sloped map (never with
        'jumper' samples: Pig Beach has its own two-level farming)."""
        with self._lock:
            return self._sweep_locked()

    def _sweep_locked(self):
        return (self.config.get('Sweep patrol') is True
                and 'jumper' not in self._enemy_speed_profiles_locked())

    def set_sweep_patrol(self, enabled):
        with self._lock:
            self.config['Sweep patrol'] = bool(enabled)
            self._release_lock_locked()
            self.save_config()

    def edge_loot_enabled(self):
        with self._lock:
            return self.config.get('Edge loot jumps') is True

    def set_edge_loot(self, enabled):
        with self._lock:
            self.config['Edge loot jumps'] = bool(enabled)
            self.save_config()

    def idle_variety_enabled(self):
        with self._lock:
            return self.config.get('Idle variety') is True

    def set_idle_variety(self, enabled):
        with self._lock:
            self.config['Idle variety'] = bool(enabled)
            self.save_config()

    def _on_slope(self, item, player_x, ground_y):
        """ITEM moved onto the character's ground line when a ramp between the
        two explains its height (so everything after treats it as on this
        level); unchanged when it is higher or lower than a ramp could put it."""
        if self._is_stand_in(item):
            return item
        elevation = self._elevation(item, ground_y)
        allowed = min(self.SLOPE_MAX,
                      self.SLOPE_BASE + self.SLOPE_RISE * abs(item[1] - player_x))
        shift = int(round(elevation))
        if shift == 0 or abs(elevation) > allowed:
            return item
        candidate = item[0]
        return (candidate[:6] + (candidate[6] + shift,), item[1], item[2] + shift, item[3])

    # ---- Facing, and scenery by a portal (attack1.mp4 / attack2.mp4, 2026-11-17) ----
    # Real snails and shrooms matched their samples at 76-96 % in both clips;
    # the portal, the decor mushrooms beside it and a red bulb matched at
    # 63-72 %. A match under WEAK_SCORE with no HP bar is "weak": it is still
    # fought when nothing better is about, but a strong one up to
    # WEAK_PENALTY px further away is chosen first, a weak lock keeps the far
    # look running, and it gives way to a strong enemy at once (or, when it
    # is already being swung at, after WEAK_TRIAL_SECONDS with no HP bar).
    WEAK_SCORE = 0.75
    WEAK_PENALTY = 350
    WEAK_TRIAL_SECONDS = 1.2
    WEAK_AFTER_BAR = 0.7            # s a lock that had an HP bar stays weak and bar-less: dead
    WALK_NOTE_SECONDS = 0.15        # a walking note from the bot older than this is stale
    PORTAL_LIGHT = 0.5              # share of near-white pixels that makes a portal

    @staticmethod
    def _portal_light(frame, centre_x, ground_y, half=22, height=100):
        """Share (0-1) of bright, colourless pixels in the column standing on
        the ground at CENTRE_X."""
        ground_y, centre_x = int(round(ground_y)), int(round(centre_x))
        box = frame[max(0, ground_y - height):max(0, ground_y - 8),
                    max(0, centre_x - half):centre_x + half, :3]
        if box.size == 0:
            return 0.0
        hsv = cv2.cvtColor(np.ascontiguousarray(box), cv2.COLOR_BGR2HSV)
        return float(((hsv[:, :, 2] > 205) & (hsv[:, :, 1] < 95)).mean())

    def _facing_reader(self):
        reader = self.__dict__.get('_facing_reader_obj')
        if reader is None:
            from src.detection.facing import FacingReader
            reader = self._facing_reader_obj = FacingReader()
        return reader

    def note_walking(self, direction):
        """The bot is walking DIRECTION right now (arrow held, the character
        moving that way): the facing reader learns the head from it."""
        with self._lock:
            self._walk_note = (direction, time.monotonic())

    def facing(self):
        """'left'/'right' when the character's head was read that way twice
        in a row just now (src/detection/facing.py), else None."""
        with self._lock:
            return self._facing_reader().steady(time.monotonic())

    def facing_seen(self):
        """The last single reading if it is fresh, else None."""
        with self._lock:
            return self._facing_reader().seen(time.monotonic())

    def facing_ready(self):
        with self._lock:
            return self._facing_reader().ready

    @classmethod
    def _is_weak(cls, item):
        """A poor match with no HP bar: as likely scenery as an enemy."""
        return item[3] is None and item[0][0] < cls.WEAK_SCORE

    STANDING_HINT_SECONDS = 0.3

    def note_standing(self):
        """The bot is standing still on purpose (casting, letting it come)."""
        with self._lock:
            self._standing_at = time.monotonic()

    def target_details(self):
        """
        The locked target as the last scan saw it, for the ranged pull: a dict
        with 'serial' (changes with every new lock), 'offset'/'dy' (px from
        the player), 'on_level', 'airborne', 'health' (HP bar ratio or None),
        'real', 'closing' (px/s, positive while the gap shrinks), 'attackable',
        'waiting', 'profile' and 'time'. None without a target or when stale.
        """
        with self._lock:
            info = self.__dict__.get('_target_info')
            if (info is None or self._target_offset_x is None
                    or time.monotonic() - info['time'] > 0.6):
                return None
            return dict(info)

    def enemy_offsets(self):
        """
        Screen x offsets (px, from the player) of every enemy on the player's
        level in the last scan, target or not: what stands in the way of a walk.
        Empty when the last scan is older than a second.
        """
        with self._lock:
            seen = self.__dict__.get('_enemy_offsets')
            if seen is None or time.monotonic() - seen[0] > 1.0:
                return []
            return list(seen[1])

    def waiting(self):
        """True while a jumping target is mid-hop above the swing: stand and wait."""
        with self._lock:
            return bool(self.__dict__.get('_target_waiting'))

    def _release_lock_locked(self):
        """Forgets the current target. Call with _lock held."""
        self._locked_target = None
        self._pending_target = None
        self._target_missed_scans = 0
        self._target_missed_since = None
        self._target_offset_x = None
        self._target_vertical_offset = None
        self._target_attackable = False
        self._locked_health = None
        self._locked_elevated_since = None
        self._locked_seen = None
        self._locked_velocity = 0.0
        self._locked_rel = None
        self._locked_closing = 0.0
        self._locked_real = False
        self._locked_world = None
        self._normal_swing_since = None
        self._target_waiting = False
        self._blank_swing_at = None
        self._attack_direction = None
        self._attack_is_group = False

    # ---- Profiles: one saved set of samples + settings per map / enemy mix ----

    @classmethod
    def _profile_path(cls, name):
        name = name.strip() if isinstance(name, str) else ''
        if name != cls.AUTOSAVE_PROFILE and not re.fullmatch(
                r'[A-Za-z0-9][A-Za-z0-9 _-]{0,39}', name):
            raise ValueError(
                'Profile names use letters, digits, spaces, - and _ (max 40).'
            )
        return os.path.join(cls.PROFILE_DIRECTORY, name)

    @classmethod
    def profile_names(cls):
        if not os.path.isdir(cls.PROFILE_DIRECTORY):
            return []
        return sorted(
            name for name in os.listdir(cls.PROFILE_DIRECTORY)
            if os.path.isfile(os.path.join(cls.PROFILE_DIRECTORY, name, 'mob_tracker'))
        )

    def save_profile(self, name):
        """Copies the current player badge, enemy samples and settings to a profile."""
        path = self._profile_path(name)
        with self._lock:
            if self._player_template is None or not self._mob_templates:
                raise ValueError('Nothing to save: calibrate the player and enemies first.')
            count = len(self._mob_templates)
            settings = self.config.copy()
        os.makedirs(path, exist_ok=True)
        sources = [self.PLAYER_TEMPLATE_PATH] + [
            self._mob_template_path(index) for index in range(count)
        ]
        for source in sources:
            shutil.copy2(source, os.path.join(path, os.path.basename(source)))
        for stale in os.listdir(path):     # samples left over from a bigger save
            match = re.fullmatch(r'mob_target_(\d+)\.png', stale)
            if match and int(match.group(1)) >= count:
                os.remove(os.path.join(path, stale))
        with open(os.path.join(path, 'mob_tracker'), 'wb') as file:
            pickle.dump(settings, file)
        return path

    def load_profile(self, name, preserve_player_badge=False):
        """
        Makes a saved profile the active one. The active samples are first saved to
        the '_autosave' profile, so loading never loses an unsaved calibration.
        Map-specific loads can preserve the active character's player badge.

        :param preserve_player_badge: leave the current character badge in place.
        """
        path = self._profile_path(name)
        settings_path = os.path.join(path, 'mob_tracker')
        if not os.path.isfile(settings_path):
            raise ValueError(f'No saved profile named {name!r}.')
        with open(settings_path, 'rb') as file:
            loaded = pickle.load(file)
        try:
            count = max(1, int(loaded.get('Enemy count') or 1))
        except (TypeError, ValueError):
            count = 1
        player_badge = os.path.basename(self.PLAYER_TEMPLATE_PATH)
        preserve_badge = preserve_player_badge
        file_names = ([] if preserve_badge else [player_badge]) + [
            os.path.basename(self._mob_template_path(index)) for index in range(count)
        ]
        missing = [
            file_name for file_name in file_names
            if not os.path.isfile(os.path.join(path, file_name))
        ]
        if missing:
            raise ValueError(f'Profile {name!r} is missing: ' + ', '.join(missing))
        if name.strip() != self.AUTOSAVE_PROFILE and self.has_samples():
            self.save_profile(self.AUTOSAVE_PROFILE)
        os.makedirs(self.TEMPLATE_DIRECTORY, exist_ok=True)
        for file_name in file_names:
            shutil.copy2(
                os.path.join(path, file_name),
                os.path.join(self.TEMPLATE_DIRECTORY, file_name)
            )
        with self._lock:
            enabled = self.config.get('Enabled', False)
            group_skill = self._group_skill_locked()
            self.config = {
                key: loaded.get(key, default)
                for key, default in self.DEFAULT_CONFIG.items()
            }
            self.config['Enabled'] = enabled    # loading never switches attacking on
            if 'Group skill' not in loaded:     # per character, not per map
                self.config['Group skill'] = group_skill
            self._release_lock_locked()
            self._level_spots = []
            self._blocked_side = None
            self._ledge_height = None
            self._ledge_seen = None
            self._ground_seen = None
            self._lower_spots = []
            self._lower_seen = None
            self._mask_cache = {}
            self.save_config()
        self._load_templates()

    def player_x(self):
        with self._lock:
            return None if self._player_position is None else self._player_position[0]

    def group_in_range(self):
        with self._lock:
            return self._group_in_range

    def set_focused_enemy_samples(self, indices):
        with self._lock:
            valid_indices = {
                int(index) for index in indices
                if 0 <= int(index) < len(self._mob_templates)
            }
            self.config['Focus configured'] = True
            self.config['Focused enemies'] = ','.join(
                str(index) for index in sorted(valid_indices)
            )
            self._locked_target = None
            self._pending_target = None
            self._target_missed_scans = 0
            self._target_missed_since = None
            self.save_config()

    @staticmethod
    def _preferred_candidates(positioned_candidates, focused_samples):
        # (ghost-attack.mp4: the portal matched the one focused sample at
        # 65 %, so the snails matching 97 % on unfocused samples were not
        # even considered. A poor, bar-less match is no reason to prefer.)
        focused_candidates = [
            item for item in positioned_candidates
            if item[0][1] in focused_samples
            and (item[3] is not None or item[0][0] >= MobTracker.WEAK_SCORE)
        ]
        return focused_candidates or positioned_candidates

    def add_enemy_template(self, frame, player_bounds, mob_bounds):
        height, width = frame.shape[:2]
        (x1, y1), (x2, y2) = mob_bounds
        if (x1 < 0 or y1 < 0 or x2 > width or y2 > height
                or x2 - x1 < 12 or y2 - y1 < 12):
            raise ValueError('The selected enemy sample is too small or outside the screen.')
        (px1, py1), (px2, py2) = player_bounds
        if (px1 < 0 or py1 < 0 or px2 > width or py2 > height
                or px2 - px1 < 20 or py2 - py1 < 6):
            raise ValueError('The selected player badge sample is too small or outside the screen.')

        mob_template = frame[y1:y2, x1:x2, :3].copy()
        with self._lock:
            if self._player_template is None:
                raise ValueError(
                    'Calibrate the player badge with “Calibrate player + enemy” first.'
                )
            sample_count = len(self._mob_templates)
            alignment_y = round((y1 + y2 - py1 - py2) / 2)

        path = self._mob_template_path(sample_count)
        if not cv2.imwrite(path, mob_template):
            raise OSError(f'Could not save enemy sample to {path}.')

        with self._lock:
            self._mob_templates.append((mob_template, alignment_y))
            self.config['Enemy count'] = len(self._mob_templates)
            if not self.config.get('Focus configured'):
                self.config['Focused enemies'] = ','.join(
                    str(index) for index in range(len(self._mob_templates))
                )
            self.config['Enemy alignments'] = ','.join(
                str(offset) for _, offset in self._mob_templates
            )
            profiles = self._enemy_speed_profiles_locked()
            profiles.append('normal')
            self.config['Enemy speeds'] = ','.join(profiles)
            self._attack_direction = None
            self._target_offset_x = None
            self._target_vertical_offset = None
            self._target_attackable = False
            self._locked_target = None
            self._pending_target = None
            self._target_missed_scans = 0
            self._target_missed_since = None
            self._group_in_range = False
            self._status = (
                f'{len(self._mob_templates)} enemy samples saved. '
                'Enable auto-attack and press the Start/stop key to test.'
            )
            self.save_config()
            return len(self._mob_templates)

    @classmethod
    def _mob_template_path(cls, index):
        if index == 0:
            return cls.MOB_TEMPLATE_PATH
        return os.path.join(cls.TEMPLATE_DIRECTORY, f'mob_target_{index}.png')

    def _load_templates(self):
        player_template = cv2.imread(self.PLAYER_TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
        with self._lock:
            try:
                sample_count = int(self.config.get('Enemy count') or 1)
            except (TypeError, ValueError):
                sample_count = 1
            if sample_count < 1:
                sample_count = 1
            alignment_string = self.config.get('Enemy alignments', '')
            try:
                alignments = [
                    int(value) for value in alignment_string.split(',') if value
                ]
            except (AttributeError, ValueError):
                alignments = []
            if not alignments:
                try:
                    alignments = [int(self.config.get('Alignment Y') or 0)]
                except (TypeError, ValueError):
                    alignments = [0]

        mob_templates = []
        missing_samples = []
        for index in range(sample_count):
            path = self._mob_template_path(index)
            mob_template = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if mob_template is None:
                missing_samples.append(os.path.basename(path))
                continue
            alignment_y = alignments[index] if index < len(alignments) else alignments[-1]
            mob_templates.append((mob_template, alignment_y))

        with self._lock:
            if player_template is None or not mob_templates:
                self._load_error = 'Select the player badge and an enemy sample.'
                self._status = self._load_error
                return
            if missing_samples:
                self._load_error = (
                    'Enemy sample file(s) missing: '
                    + ', '.join(missing_samples)
                    + '. Recalibrate the enemy samples.'
                )
                self._status = self._load_error
                return
            self._player_template = player_template
            self._mob_templates = mob_templates
            self._load_error = ''
            self._status = (
                f'{len(mob_templates)} enemy sample(s) loaded. '
                'Waiting to detect the player and a nearby enemy.'
            )

    def status(self):
        with self._lock:
            return self._status

    def consume_attack(self):
        with self._lock:
            direction = self._attack_direction
            self._attack_direction = None
            self._attack_is_group = False
            return direction

    def consume_attack_action(self):
        with self._lock:
            action = (
                self._attack_direction,
                self._attack_is_group,
                self._group_count if self._attack_is_group else 0,
            )
            self._attack_direction = None
            self._attack_is_group = False
            return action

    def target(self):
        with self._lock:
            if self._target_offset_x is None:
                return None
            return self._target_offset_x, self._target_attackable

    def pursuit_target(self):
        with self._lock:
            if self._target_offset_x is None or self._target_vertical_offset is None:
                return None
            return (
                self._target_offset_x,
                self._target_attackable,
                self._target_vertical_offset,
            )

    def _hold_locked_target(self, player_x, player_y, enabled, bot_enabled):
        with self._lock:
            if self._locked_target is None:
                return False

            now = time.monotonic()
            if self._target_missed_since is None:
                self._target_missed_since = now
            self._target_missed_scans += 1
            missing_for = now - self._target_missed_since
            self._group_in_range = False
            killed = (
                self._locked_health is not None
                and self._locked_health <= self.LOW_HEALTH_RATIO
            )
            try:
                profiles = self._enemy_speed_profiles_locked()
            except AttributeError:
                profiles = ()
            index = self._locked_target[0]
            jumper = index < len(profiles) and profiles[index] == 'jumper'
            grace = (self.JUMPER_LOST_GRACE_SECONDS if jumper
                     else self.TARGET_LOST_GRACE_SECONDS)
            normal_profile = index >= len(profiles) or profiles[index] == 'normal'
            # (from the last sighting: a knock-back moves the character, and
            # with it the screen offset of where the enemy was)
            seen_rel = self.__dict__.get('_locked_rel')
            last_offset = (seen_rel[0] if seen_rel is not None
                           else self._locked_target[1] - player_x)
            # (knock-back-dmg-2.mp4: a shroom at 2% walked behind the character,
            # was taken for killed, and bit while the bot stood there)
            overlapped = (normal_profile
                          and abs(last_offset) <= self.NORMAL_OVERLAP_PX
                          and missing_for <= self.NORMAL_HIDDEN_SECONDS)
            if overlapped:
                killed = False
            if (self.__dict__.get('_player_knocked')
                    and abs(last_offset) <= self.REAL_NEAR_PX):
                self._locked_real = True    # it bit the character while hidden
            if killed or missing_for > grace:
                if killed:
                    # For looting: drops land where it died
                    self.__dict__.setdefault('_kill_events', []).append(
                        (now, self._locked_target[1] - player_x))
                self._locked_target = None
                self._pending_target = None
                self._target_missed_scans = 0
                self._target_missed_since = None
                self._target_offset_x = None
                self._target_vertical_offset = None
                self._target_attackable = False
                self._locked_health = None
                return False

            template_index, locked_x, locked_y = self._locked_target
            self._target_waiting = False
            self._target_offset_x = locked_x - player_x
            # A locked enemy was on the player's level when last seen (it may only
            # be hidden behind an effect or mid-jump), so only the distance matters.
            self._target_vertical_offset = 0
            normal = index >= len(profiles) or profiles[index] == 'normal'
            reach = (self.JUMPER_ATTACK_RANGE + self.JUMPER_RANGE_HYSTERESIS
                     if jumper else
                     self.NORMAL_ATTACK_RANGE + self.NORMAL_RANGE_HYSTERESIS
                     if normal else self.ATTACK_RANGE)
            self._target_attackable = (
                abs(self._target_offset_x) <= reach
                and missing_for <= self.HIDDEN_ATTACK_SECONDS
            )
            if overlapped:
                # Under the character: keep swinging, facing where it is going
                # (one walking through comes out on the other side)
                rel = self.__dict__.get('_locked_rel')
                closing = self.__dict__.get('_locked_closing', 0.0)
                offset = rel[0] if rel is not None else last_offset
                if closing >= self.NORMAL_CLOSING_MIN:
                    offset -= (1 if offset > 0 else -1) * closing * missing_for
                offset = max(-self.NORMAL_OVERLAP_PX, min(self.NORMAL_OVERLAP_PX, offset))
                self._target_offset_x = offset or (1 if last_offset >= 0 else -1)
                self._target_attackable = True
            near_overlap = normal_profile and abs(last_offset) <= self.NORMAL_OVERLAP_PX
            if (jumper or near_overlap) and not self._target_attackable:
                # Don't walk to where a pig was (attack-issue-2.mp4: it crossed
                # through and the bot walked the other way for 1.5 s)
                self._target_offset_x = 0
            if normal_profile:
                # (edge-portal-issue.mp4: swinging at a hidden scenery lock is
                # swinging at nothing; it counts toward the area's total)
                self._blank_step(now, self._target_attackable,
                                 bool(self.__dict__.get('_locked_real'))
                                 or self._locked_health is not None)
                if (self._futile_step(now, self._target_attackable,
                                      self._locked_health is not None)
                        >= self.FUTILE_SECONDS and self._target_attackable):
                    # (hidden scenery lock, swung at for good: drop it)
                    self._ignore_spot(locked_x, locked_y, now, self.SCENERY_IGNORE_SECONDS)
                    self._futile_total = 0.0
                    self._locked_real = False
                    self._release_lock_locked()
                    self._status = ('Swung at a hidden target for '
                                    f'{self.FUTILE_SECONDS:.0f} sec with no HP bar: '
                                    'scenery. Ignoring that spot.')
                    return False
            self._attack_direction = None
            if (self._target_attackable and enabled and bot_enabled
                    and is_game_focused()
                    and time.monotonic() - self._last_attack_time
                    >= self.ATTACK_COOLDOWN):
                self._attack_direction = (
                    'left' if self._target_offset_x < 0 else 'right'
                )
                self._last_attack_time = time.monotonic()
            self._status = (
                f'Locked enemy temporarily hidden '
                f'({missing_for:.1f}/{self.TARGET_LOST_GRACE_SECONDS:.1f} sec); '
                f'{"continuing attack" if self._target_attackable else "holding target"}.'
            )
            return True

    @classmethod
    def _sample_mask(cls, template):
        """
        The enemy's own pixels inside its sample box (uint8, 255 = enemy), or None
        when the split is not believable (then the whole box is matched as before).
        """
        if template.ndim != 3 or min(template.shape[:2]) < 12:
            return None
        height, width = template.shape[:2]
        image = np.ascontiguousarray(template[:, :, :3])
        for inset in (2, 1):
            mask = np.zeros((height, width), np.uint8)
            # GrabCut starts from random colour clusters; a fixed seed gives every
            # start of the bot the same cut-out for the same sample
            cv2.setRNGSeed(0)
            try:
                cv2.grabCut(
                    image, mask,
                    (inset, inset, width - 2 * inset, height - 2 * inset),
                    np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_RECT
                )
            except cv2.error:
                continue
            foreground = (
                (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
            ).astype(np.uint8)
            foreground = cv2.morphologyEx(
                foreground, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
            )
            if cls.MASK_MIN_FRACTION <= foreground.mean() <= cls.MASK_MAX_FRACTION:
                return foreground * 255
        # A tightly boxed enemy leaves too little scenery to tell apart. Use an oval
        # inside the box instead: it drops the corners, where the scenery shows.
        return cls._oval_mask(height, width)

    @staticmethod
    def _oval_mask(height, width):
        oval = np.zeros((height, width), np.uint8)
        cv2.ellipse(
            oval, (width // 2, height // 2),
            (max(1, width // 2 - 1), max(1, height // 2 - 1)), 0, 0, 360, 255, -1
        )
        return oval

    @classmethod
    def _is_loose_mask(cls, mask):
        """True for the oval fallback, which lets some scenery into the match."""
        return mask is not None and np.array_equal(mask, cls._oval_mask(*mask.shape))

    def _mask_for(self, template):
        cache = self.__dict__.setdefault('_mask_cache', {})
        key = (template.shape, hash(template.tobytes()))
        if key not in cache:
            if len(cache) > 64:
                cache.clear()
            cache[key] = self._sample_mask(template)
        return cache[key]

    def _variants_for(self, template, mask):
        """
        The sample as saved plus its mirror image: enemies turn round, and one
        sample should find them facing either way.
        """
        key = ('mirror', id(template))
        cache = self.__dict__.setdefault('_mask_cache', {})
        cached = cache.get(key)
        if cached is None or cached[0] is not template:
            mirrored = np.ascontiguousarray(template[:, ::-1])
            mirrored_mask = None if mask is None else np.ascontiguousarray(mask[:, ::-1])
            # A sample that already looks the same both ways needs no second pass
            difference = np.abs(
                template.astype(np.int16) - mirrored.astype(np.int16)
            ).mean()
            cached = (template, mirrored, mirrored_mask, difference >= 30)
            cache[key] = cached
        if not cached[3]:
            return ((template, mask),)
        return ((template, mask), (cached[1], cached[2]))

    @classmethod
    def _masked_matching_locations(cls, search, template, mask, threshold, separation=1.0):
        """Like _matching_locations, but on the enemy's pixels only plus a colour check."""
        if (search.size == 0 or template.shape[0] > search.shape[0]
                or template.shape[1] > search.shape[1]):
            return [], 0.0
        template = np.ascontiguousarray(template[:, :, :3])
        search = np.ascontiguousarray(search[:, :, :3])
        template_height, template_width = template.shape[:2]
        # The shape match runs at half size (masked matching is slow); the colour
        # check below runs at full size and also settles the exact position.
        step = cls.MASK_MATCH_STEP if min(template_height, template_width) >= 24 else 1
        if step > 1:
            small = (template_width // step, template_height // step)
            small_template = cv2.resize(template, small, interpolation=cv2.INTER_AREA)
            small_mask = cv2.resize(mask, small, interpolation=cv2.INTER_NEAREST)
            small_search = cv2.resize(
                search, (search.shape[1] // step, search.shape[0] // step),
                interpolation=cv2.INTER_AREA
            )
        else:
            small_template, small_mask, small_search = template, mask, search
        if (small_template.shape[0] > small_search.shape[0]
                or small_template.shape[1] > small_search.shape[1]):
            return [], 0.0
        result = cv2.matchTemplate(
            small_search, small_template, cv2.TM_CCOEFF_NORMED,
            mask=cv2.merge([small_mask] * 3)
        )
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        result[(result > 1.0) | (result < -1.0)] = 0.0
        kernel = (max(1, round(small_template.shape[0] * separation)),
                  max(1, round(small_template.shape[1] * separation)))
        maxima = cv2.dilate(result, np.ones(kernel, dtype=np.uint8))
        peak_y, peak_x = np.where((result >= threshold) & (result == maxima))
        order = np.argsort(result[peak_y, peak_x])[::-1]
        template_float = template.astype(np.float32)
        inside = mask > 0
        max_x = search.shape[1] - template_width
        max_y = search.shape[0] - template_height
        matches = []
        for index in order:
            score = float(result[peak_y[index], peak_x[index]])
            best = None
            for dy in range(0, step):
                for dx in range(0, step):
                    x = min(int(peak_x[index]) * step + dx, max_x)
                    y = min(int(peak_y[index]) * step + dy, max_y)
                    patch = search[y:y + template_height, x:x + template_width]
                    difference = np.abs(
                        patch.astype(np.float32) - template_float
                    )[inside].mean()
                    if best is None or difference < best[0]:
                        best = (difference, x, y)
            difference, x, y = best
            if difference > cls.COLOR_TOLERANCE:
                continue
            if any(
                abs(x - match[0][0]) < template_width * separation
                and abs(y - match[0][1]) < template_height * separation
                for match in matches
            ):
                continue
            matches.append(((x, y), score))
        if matches:
            best_score = matches[0][1]
        else:   # shape-only matches (wrong colours) don't count as near misses
            best_score = min(float(result.max()), threshold - 0.01) if result.size else 0.0
        return matches, best_score

    @classmethod
    def _merge_same_enemy(cls, positioned_candidates):
        """
        Different samples of one enemy match it at slightly different offsets.
        Keeps the best-scoring match of each set of heavily overlapping boxes, so
        one enemy is not counted twice (which would fake a group).
        """
        accepted = []
        for item in sorted(positioned_candidates, key=lambda c: c[0][0], reverse=True):
            candidate, center_x, center_y, _ = item
            height, width = candidate[2].shape[:2]
            duplicate = False
            for other in accepted:
                other_height, other_width = other[0][2].shape[:2]
                overlap_x = (min(center_x + width / 2, other[1] + other_width / 2)
                             - max(center_x - width / 2, other[1] - other_width / 2))
                overlap_y = (min(center_y + height / 2, other[2] + other_height / 2)
                             - max(center_y - height / 2, other[2] - other_height / 2))
                if overlap_x <= 0 or overlap_y <= 0:
                    continue
                intersection = overlap_x * overlap_y
                union = width * height + other_width * other_height - intersection
                if intersection / union >= cls.SAME_ENEMY_OVERLAP:
                    duplicate = True
                    break
            if not duplicate:
                accepted.append(item)
        return accepted

    @staticmethod
    def _ground_offset(mob_templates):
        """
        Where enemies' feet are relative to the player badge: the lowest bottom edge
        among the samples (a sample taken mid-jump sits higher and is ignored).
        """
        return max(
            alignment_y + template.shape[0] / 2
            for template, alignment_y in mob_templates
        )

    @staticmethod
    def _matching_locations(search, template, threshold, separation=1.0):
        if (search.size == 0 or template.shape[0] > search.shape[0]
                or template.shape[1] > search.shape[1]):
            return [], 0.0
        if template.ndim == 3:
            channel_results = [
                cv2.matchTemplate(
                    search[:, :, channel],
                    template[:, :, channel],
                    cv2.TM_CCOEFF_NORMED
                )
                for channel in range(template.shape[2])
            ]
            result = np.mean(channel_results, axis=0, dtype=np.float32)
        else:
            result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        best_score = float(np.max(result))
        maxima = cv2.dilate(
            result,
            np.ones((max(1, round(template.shape[0] * separation)),
                     max(1, round(template.shape[1] * separation))), dtype=np.uint8)
        )
        peak_y, peak_x = np.where((result >= threshold) & (result == maxima))
        order = np.argsort(result[peak_y, peak_x])[::-1]
        matches = []
        template_height, template_width = template.shape[:2]
        for index in order:
            x = int(peak_x[index])
            y = int(peak_y[index])
            if any(
                abs(x - match[0][0]) < template_width * separation
                and abs(y - match[0][1]) < template_height * separation
                for match in matches
            ):
                continue
            matches.append(((x, y), float(result[y, x])))
        return matches, best_score

    @staticmethod
    def _best_match(search, template):
        if (search.size == 0 or template.shape[0] > search.shape[0]
                or template.shape[1] > search.shape[1]):
            return None
        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, top_left = cv2.minMaxLoc(result)
        return top_left, score

    @classmethod
    def _nearest_enemy_group(cls, positioned_candidates, player_x,
                             min_enemies=None, vertical_radius=None):
        min_enemies = min_enemies or cls.GROUP_MIN_ENEMIES
        vertical_radius = vertical_radius or cls.GROUP_VERTICAL_RADIUS
        positioned_candidates = cls._deduplicate_positioned_candidates(
            positioned_candidates
        )
        enemies = [
            (
                item[1],
                item[2],
                item[0][2].shape[1],
                item[0][2].shape[0],
                item,
            )
            for item in positioned_candidates
        ]
        # Every window GROUP_HORIZONTAL_RADIUS wide, starting at each enemy in turn
        groups = []
        for start_x, start_y, _, _, _ in enemies:
            component = [
                enemy for enemy in enemies
                if 0 <= enemy[0] - start_x <= cls.GROUP_HORIZONTAL_RADIUS
                and abs(enemy[1] - start_y) <= vertical_radius
            ]
            if len(component) < min_enemies:
                continue
            members = [enemy[4] for enemy in component]
            center_x = sum(enemy[0] for enemy in component) / len(component)
            groups.append((abs(center_x - player_x), -len(component), members))

        if not groups:
            return []
        return min(groups, key=lambda group: (group[0], group[1]))[2]

    @staticmethod
    def _deduplicate_positioned_candidates(positioned_candidates):
        accepted = []
        for item in sorted(
            positioned_candidates,
            key=lambda candidate: candidate[0][0],
            reverse=True,
        ):
            candidate, center_x, center_y, _ = item
            width = candidate[2].shape[1]
            height = candidate[2].shape[0]
            duplicate = False
            for previous in accepted:
                previous_candidate, previous_x, previous_y, _ = previous
                previous_width = previous_candidate[2].shape[1]
                previous_height = previous_candidate[2].shape[0]
                close_centers = (
                    abs(center_x - previous_x) < min(width, previous_width) * 0.25
                    and abs(center_y - previous_y) < min(height, previous_height) * 0.25
                )
                if close_centers:
                    duplicate = True
                    break
            if not duplicate:
                accepted.append(item)
        return accepted

    def _takes_no_damage(self, attackable, health_ratio, now, seconds=None):
        """
        True once the locked enemy has stayed in attack range for NO_DAMAGE_SECONDS
        with no HP bar. An enemy taking hits shows its bar, so seeing one (or the
        enemy leaving range) starts the count again. Call with _lock held.
        """
        if not attackable or health_ratio is not None:
            self._in_range_since = None
            return False
        if self._in_range_since is None:
            self._in_range_since = now
            return False
        return now - self._in_range_since >= (
            self.NO_DAMAGE_SECONDS if seconds is None else seconds
        )

    @classmethod
    def _choose_target(cls, candidates, player_x, direction, ground_y=None,
                       weak_penalty=0):
        """
        The enemy to go for next: the nearest, preferring one that is already hurt
        (finishing it is quicker than starting a fresh one) and one that is ahead
        in the direction the player is already heading.
        """

        def cost(item):
            _, enemy_x, _, health_ratio = item
            value = abs(enemy_x - player_x)
            if health_ratio is not None and health_ratio < 0.97:
                value -= cls.HURT_BONUS * (1.5 - health_ratio)
            if ((direction == 'right' and enemy_x < player_x - cls.ATTACK_RANGE)
                    or (direction == 'left' and enemy_x > player_x + cls.ATTACK_RANGE)):
                value += cls.TURN_PENALTY
            if (ground_y is not None
                    and cls._elevation(item, ground_y) > cls.LEDGE_MIN_HEIGHT):
                value += cls.ELEVATED_PENALTY
            # (not one close enough to bite: if it is real, walking off to a
            # better match further away turns the character's back on it. A
            # swing or two settles it: WEAK_TRIAL_SECONDS.)
            if (weak_penalty and cls._is_weak(item)
                    and abs(enemy_x - player_x) > cls.THREAT_RANGE):
                value += weak_penalty       # likely scenery: real ones first
            return value, -item[0][0]

        return min(candidates, key=cost)

    @staticmethod
    def _sweep_candidate(candidates, player_x, direction):
        if direction == 'right':
            forward = [item for item in candidates if item[1] >= player_x]
            if forward:
                return min(forward, key=lambda item: (item[1], -item[0][0]))
        elif direction == 'left':
            forward = [item for item in candidates if item[1] <= player_x]
            if forward:
                return max(forward, key=lambda item: (item[1], item[0][0]))
        else:
            raise ValueError(f'Unsupported sweep direction: {direction}')
        return min(
            candidates,
            key=lambda item: (abs(item[1] - player_x), -item[0][0]),
        )

    def _px_per_minimap(self):
        bot = getattr(_config, 'bot', None)
        measured = getattr(bot, '__dict__', {}).get('_px_per_minimap')
        return measured or self.__dict__.get('_ppm_est') or self.PX_PER_MINIMAP

    def _calibrate_map_scale(self, player_x, width, now):
        """
        At a map's edge the camera stops following the character, so its
        screen movement is its real movement: against the minimap dot's, that
        is the map scale (px per minimap unit). Call with _lock held.
        """
        pos = getattr(_config, 'player_pos', None)
        when = getattr(_config, 'player_pos_time', 0.0) or 0.0
        if (not pos or tuple(pos) == (0, 0) or abs(now - when) > 0.5
                or abs(player_x - width / 2) < self.CAMERA_CLAMP_PX):
            self._ppm_anchor = None
            return
        anchor = self.__dict__.get('_ppm_anchor')
        if anchor is None or (anchor[0] < width / 2) != (player_x < width / 2):
            self._ppm_anchor = (player_x, pos[0])
            return
        moved = pos[0] - anchor[1]
        if abs(moved) < self.SCALE_MIN_MOVE:
            return
        sample = (player_x - anchor[0]) / moved
        self._ppm_anchor = (player_x, pos[0])
        if 1500 <= sample <= 20000:
            known = self.__dict__.get('_ppm_est')
            self._ppm_est = sample if known is None else 0.7 * known + 0.3 * sample

    # A Split Road (2-point.mp4, 2026-11-19): the snail up on the cliff past
    # the right end point (and the NPC on the platform above) counted as
    # enemies 300-500 px away. The bot walked at them, hit the map limit,
    # turned, saw them again (6 turns in 3 s, over and over), and the
    # platform never counted as cleared. Past a map limit = not farmed here.
    # The thing it kept walking at was the wooden signpost on the low ground
    # just past the right end (x ~0.207, read 0.203-0.212), matching a
    # mushroom sample; one minimap px is 0.004 there.
    LIMIT_MARGIN = 0.004            # minimap units
    # MapleStory minimaps are 1:16: 2-point.mp4 measured 15.97 screen px per
    # minimap px with the camera clamped (the snail map's 4100/unit fits too)
    MINIMAP_SCALE = 16.0

    def _limit_scale(self):
        """Screen px per minimap unit, from the minimap's width (else the estimate)."""
        capture = getattr(_config, 'capture', None)
        minimap = (getattr(capture, 'minimap', None) or {}).get('minimap')
        width = getattr(minimap, 'shape', (0, 0))[1] if minimap is not None else 0
        return self.MINIMAP_SCALE * width if width >= 50 else self._px_per_minimap()

    def _inside_map_limits(self, candidates, player_x):
        """CANDIDATES minus those standing past the map profile's limits (an
        HP bar shows it is a real, hurt enemy: always kept)."""
        bot = getattr(_config, 'bot', None)
        manager = getattr(bot, '__dict__', {}).get('map_events')
        limits = manager.limits() if manager is not None else {}
        if not limits:
            return candidates
        pos = getattr(_config, 'player_pos', None)
        when = getattr(_config, 'player_pos_time', 0.0) or 0.0
        if not pos or tuple(pos) == (0, 0) or abs(time.monotonic() - when) > 0.5:
            return candidates
        scale = self._limit_scale()
        left, right = limits.get('left'), limits.get('right')
        kept = []
        for item in candidates:
            world = pos[0] + (item[1] - player_x) / scale
            if item[3] is None and (
                    (right is not None and world > right + self.LIMIT_MARGIN)
                    or (left is not None and world < left - self.LIMIT_MARGIN)):
                continue
            kept.append(item)
        return kept

    def _world_x(self, screen_x, player_x):
        """Minimap x of a screen x, or None while the minimap fix is stale."""
        pos = getattr(_config, 'player_pos', None)
        when = getattr(_config, 'player_pos_time', 0.0) or 0.0
        if (not pos or tuple(pos) == (0, 0)
                or abs(time.monotonic() - when) > 0.5):
            return None
        return pos[0] + (screen_x - player_x) / self._px_per_minimap()

    def _map_anchor(self, screen_x, player_x):
        """(minimap x of the character, screen px from it): a map spot that
        stays right when the map scale is learnt later. None without a
        fresh minimap fix."""
        pos = getattr(_config, 'player_pos', None)
        when = getattr(_config, 'player_pos_time', 0.0) or 0.0
        if (not pos or tuple(pos) == (0, 0)
                or abs(time.monotonic() - when) > 0.5):
            return None
        return (pos[0], screen_x - player_x)

    def _map_gap(self, anchor, other):
        """Screen px between two map anchors, at the current map scale."""
        return abs((other[0] - anchor[0]) * self._px_per_minimap()
                   + other[1] - anchor[1])

    def _clamp_side(self, player_x):
        """'left'/'right' while the camera is stopped at that map edge (the
        character well off centre), else None. There screen px are map px."""
        width = self.__dict__.get('_frame_width')
        if not width or abs(player_x - width / 2) < self.CAMERA_CLAMP_PX:
            return None
        return 'left' if player_x < width / 2 else 'right'

    def _on_ignored_spot(self, item, spots, player_x):
        here = self._map_anchor(item[1], player_x)
        clamp = self._clamp_side(player_x)
        radius = self.IGNORE_SPOT_RADIUS
        for spot in spots:
            spot_x, spot_y = spot[0], spot[1]
            anchor = spot[3] if len(spot) > 3 else None
            if abs(item[2] - spot_y) > radius:
                continue
            if clamp is not None and len(spot) > 4 and spot[4] == clamp:
                # (henny_bug-2.mp4: by the minimap, one dot step is ~18 px on
                # a big map; with the camera stopped at the same edge, the
                # screen itself is exact)
                if abs(item[1] - spot_x) <= self.SCENERY_SPOT_RADIUS:
                    return True
                continue
            if anchor is not None:
                if here is not None and self._map_gap(anchor, here) <= self.SCENERY_SPOT_RADIUS:
                    return True
                continue        # a map spot: never matched by screen position
            if abs(item[1] - spot_x) <= radius:
                return True
        return False

    def _unignore_near(self, player_x, now):
        """Knocked back: ignored spots by the character were no scenery
        (call with _lock held)."""
        here = self._map_anchor(player_x, player_x)
        kept = []
        for spot in self.__dict__.get('_ignored_spots', []):
            anchor = spot[3] if len(spot) > 3 else None
            clamp = self._clamp_side(player_x)
            if clamp is not None and len(spot) > 4 and spot[4] == clamp:
                near = abs(spot[0] - player_x) <= self.REAL_NEAR_PX
            elif anchor is not None and here is not None:
                near = self._map_gap(anchor, here) <= self.REAL_NEAR_PX
            else:
                near = abs(spot[0] - player_x) <= self.REAL_NEAR_PX
            if not near:
                kept.append(spot)
        self._ignored_spots = kept
        self._blank_total = 0.0
        self._futile_total = 0.0
        # (bitten here: what was ignored around the character was a real enemy,
        # so it starts over at the short ignore)
        history = self.__dict__.get('_scenery_history')
        if history and here is not None:
            self._scenery_history = [
                entry for entry in history
                if entry[0][3] is None or self._map_gap(entry[0][3], here) > self.REAL_NEAR_PX]

    def _blank_step(self, now, swinging, proven):
        """
        The running total of bar-less swinging around the character (call with
        _lock held). PROVEN (an HP bar near, a knock-back, a real lock) clears
        it; swinging adds the time since the last scan; anything else drains it
        slowly, so walking between two scenery matches no longer restarts it.
        """
        last = self.__dict__.get('_blank_tick')
        step = 0.0 if last is None else max(0.0, min(now - last, self.BLANK_MAX_STEP))
        self._blank_tick = now
        total = self.__dict__.get('_blank_total', 0.0)
        if proven:
            total = 0.0
        elif swinging:
            total += step
        else:
            total = max(0.0, total - step * self.BLANK_DRAIN)
        self._blank_total = total
        return total

    def _futile_step(self, now, swinging, bar_seen):
        """
        Like _blank_step, but only an HP bar (BAR_SEEN) clears it: the backstop
        for scenery that weak proof keeps "real" (call with _lock held).
        """
        last = self.__dict__.get('_futile_tick')
        step = 0.0 if last is None else max(0.0, min(now - last, self.BLANK_MAX_STEP))
        self._futile_tick = now
        total = self.__dict__.get('_futile_total', 0.0)
        if bar_seen:
            total = 0.0
            self._last_bar_seen = now
        elif swinging:
            total += step
        else:
            total = max(0.0, total - step * self.BLANK_DRAIN)
        self._futile_total = total
        return total

    def _ignore_spot(self, mob_x, mob_y, now, seconds=None):
        """Remembers scenery that looks like a sample (call with _lock held)."""
        player = self._player_position
        anchor = None if player is None else self._map_anchor(mob_x, player[0])
        clamp = None if player is None else self._clamp_side(player[0])
        if seconds is None or (anchor is None and clamp is None):
            seconds = self.IGNORE_SPOT_SECONDS  # (by screen px: never for long)
        elif seconds >= self.SCENERY_IGNORE_SECONDS and player is not None:
            seconds = self._repeat_seconds(mob_x, mob_y, player[0], anchor, clamp,
                                           seconds, now)
        self._ignored_spots.append((mob_x, mob_y, now + seconds, anchor, clamp))

    def _repeat_seconds(self, mob_x, mob_y, player_x, anchor, clamp, seconds, now):
        """SECONDS, doubled for each earlier time this spot was found to be
        scenery (up to SCENERY_REPEAT_MAX). Call with _lock held."""
        history = self.__dict__.get('_scenery_history', [])
        item = (None, mob_x, mob_y)
        for index, entry in enumerate(history):
            spot, count, last = entry
            # (one burst drops a cluster of matches: count each spot once per burst)
            if self._on_ignored_spot(item, [spot], player_x):
                if now - last > 1.0:
                    count += 1
                history[index] = (spot, count, now)
                return min(seconds * (2 ** count), self.SCENERY_REPEAT_MAX)
        history.append(((mob_x, mob_y, None, anchor, clamp), 0, now))
        self._scenery_history = history[-self.SCENERY_HISTORY_MAX:]
        return seconds

    def _note_normal_hits(self, same_target, health, distance, swinging, now,
                          whiff_seconds=None):
        """
        Watches the locked enemy's HP bar while swinging. If it stays up without
        dropping, the attack is not connecting: step in. Call with self._lock held.
        """
        state = self.__dict__.get('_normal_hp') if same_target else None
        if health is None:
            self._normal_hp = state     # (no bar: the swing timer handles it)
            return False
        if state is None:
            state = {'low': health, 'since': None, 'hit_at': None, 'idle_at': None}
        elif health <= state['low'] - self.NORMAL_HIT_DROP:
            state['low'] = health       # a hit landed: it is in reach
            state['hit_at'] = now
            state['since'] = None
        elif health > state['low'] + 2 * self.NORMAL_HIT_DROP:
            state['low'] = health       # (it healed, or another one took its place)
        if swinging:
            state['idle_at'] = None
            if state['since'] is None:
                state['since'] = now
        elif state['idle_at'] is None:
            state['idle_at'] = now
        elif now - state['idle_at'] >= self.NORMAL_BAR_IDLE_RESET:
            state['since'] = None       # (a scan or two of not swinging doesn't count)
        self._normal_hp = state
        timeout = (self.NORMAL_BAR_WHIFF_SECONDS if whiff_seconds is None
                   else whiff_seconds)
        return (state['since'] is not None
                and now - state['since'] >= timeout
                and (state['hit_at'] is None
                     or now - state['hit_at'] >= timeout))

    def _jumper_attack_reach(self, same_target, was_attackable, health,
                             swing_since, distance, now):
        with self._lock:
            bar_whiff = self._note_normal_hits(
                same_target, health, distance, was_attackable, now,
                self.JUMPER_WHIFF_SECONDS)
        whiffing = bar_whiff or (
            health is None and swing_since is not None
            and now - swing_since >= self.JUMPER_WHIFF_SECONDS
        )
        if whiffing:
            return self.JUMPER_WHIFF_RANGE, True
        return (self.JUMPER_ATTACK_RANGE + (
            self.JUMPER_RANGE_HYSTERESIS if was_attackable else 0
        )), False

    def _normal_follow_up(self, attackable, waiting, pursuit_x, direction, delta_x,
                          candidates, player_x, ground_y, now):
        """
        strike-issue.mp4: one swing landed, the snail was knocked back a little
        past the range (or the other side's snail became the target), and the
        bot let go and walked - into a snail, every time. After a swing the bot
        keeps swinging the same way for NORMAL_FOLLOW_UP seconds while an enemy
        is still within NORMAL_FOLLOW_RANGE on that side. It turns only for one
        behind that is about to touch. Returns (attackable, waiting, pursuit_x,
        direction). Call with self._lock held.
        """
        side = -1 if delta_x < 0 else 1
        commit = self.__dict__.get('_normal_commit')
        active = commit is not None and now < commit[1]
        if attackable:
            if active and side != commit[0] and abs(delta_x) > self.NORMAL_BEHIND_TOUCH:
                keep = self._follow_offset(commit[0], candidates, player_x, ground_y)
                if keep is not None:    # the other side can wait: finish this one
                    return True, False, keep, 'left' if keep < 0 else 'right'
            self._normal_commit = (side, now + self.NORMAL_FOLLOW_UP)
            return attackable, waiting, pursuit_x, direction
        if not active:
            return attackable, waiting, pursuit_x, direction
        keep = self._follow_offset(commit[0], candidates, player_x, ground_y)
        if keep is None:
            return attackable, waiting, pursuit_x, direction
        return True, False, keep, 'left' if keep < 0 else 'right'

    def _follow_offset(self, side, candidates, player_x, ground_y):
        """Offset of the nearest enemy on SIDE within NORMAL_FOLLOW_RANGE (on
        the player's level), or None."""
        near = [item[1] - player_x for item in candidates
                if (item[1] - player_x) * side > 0
                and abs(item[1] - player_x) <= self.NORMAL_FOLLOW_RANGE
                and abs(self._elevation(item, ground_y)) <= self.LEDGE_MIN_HEIGHT]
        return min(near, key=abs) if near else None

    # pig-beach-attack-issue.mp4 (2026-11-17): Pig Beach was farmed with the
    # Henesys samples (snails, a mushroom) still loaded. No pig matched; the
    # snail sample matched the portal, flowers, fence and clouds at 60-75 %,
    # so the bot swung at those. With the pig profile the same clip locks only
    # the pigs, at 80-98 %. Nothing in the samples says which map they are for,
    # so say it when they clearly don't fit: targets keep being locked, yet for
    # SAMPLE_FIT_SECONDS no sample matched anything well and no HP bar showed.
    SAMPLE_FIT_SECONDS = 60.0
    SAMPLE_FIT_MIN_LOCKED_SCANS = 40
    SAMPLE_FIT_WARN_EVERY = 300.0

    def _note_sample_fit(self, best_score):
        """Called with every scan's best sample score (capture thread only)."""
        now = time.monotonic()
        fit = self.__dict__.get('_sample_fit')
        if fit is None or now < fit['since']:
            fit = self._sample_fit = {'since': now, 'best': 0.0, 'locked': 0,
                                      'warned': float('-inf')}
        fit['best'] = max(fit['best'], best_score)
        if self._locked_target is not None:
            fit['locked'] += 1
        if now - fit['since'] < self.SAMPLE_FIT_SECONDS:
            return None
        best, locked = fit['best'], fit['locked']
        fit.update(since=now, best=0.0, locked=0)
        last_bar = self.__dict__.get('_last_bar_seen')
        no_bar = last_bar is None or now - last_bar >= self.SAMPLE_FIT_SECONDS
        if (locked >= self.SAMPLE_FIT_MIN_LOCKED_SCANS and best < self.WEAK_SCORE and no_bar
                and now - fit['warned'] >= self.SAMPLE_FIT_WARN_EVERY):
            fit['warned'] = now
            message = (
                f'The enemy samples do not look like the enemies on this map: for '
                f'{self.SAMPLE_FIT_SECONDS:.0f} s the best match was only {best:.0%} and '
                f'nothing hit showed an HP bar, so the bot is most likely swinging at '
                f'scenery (portals, flowers). Load the profile for this map in the app '
                f'(Load profile), or add samples of the enemies here.')
            print(f'\n[!] {message}')
            return message
        return None

    def seconds_since_bar(self):
        """Seconds since an enemy HP bar was last seen near the character (or
        on the target) in a slow-walker fight; inf if never."""
        with self._lock:
            last = self.__dict__.get('_last_bar_seen')
        return float('inf') if last is None else max(0.0, time.monotonic() - last)

    def give_up_here(self):
        """The bot swung for long with nothing hit: forget the target and
        ignore its spot as scenery (attack-loop.mp4)."""
        with self._lock:
            target = self._locked_target
            if target is not None:
                self._ignore_spot(target[1], target[2], time.monotonic(),
                                  self.SCENERY_IGNORE_SECONDS)
            self._futile_total = 0.0
            self._blank_total = 0.0
            self._locked_real = False
            self._release_lock_locked()
            self._normal_commit = None      # no follow-up swings either
            self._status = ('Bot swung for long with no HP bar: gave up on this '
                            'target and ignored its spot.')

    def follow_up_direction(self):
        """'left'/'right' while a slow-walker fight's follow-up swings are due
        (the bot keeps swinging instead of walking off), else None."""
        with self._lock:
            commit = self.__dict__.get('_normal_commit')
            if commit is None or time.monotonic() >= commit[1]:
                return None
            return 'left' if commit[0] < 0 else 'right'

    def _may_switch_side(self, selected, threat, player_x, now):
        """
        Whether a nearer THREAT may take over from the SELECTED target. Same
        side: yes. Other side (memory-issue.mp4: red snails on both sides, the
        bot turned every 1-2 s, swinging at one while the other one bit its
        back): only when the one behind is about to touch and the one in front
        is out of swing reach, and not again within NORMAL_SIDE_COMMIT seconds.
        """
        ahead, behind = selected[1] - player_x, threat[1] - player_x
        if ahead * behind >= 0 or abs(ahead) > self.THREAT_RANGE:
            return True     # same side, or not really fighting the one in front
        if now - getattr(self, '_side_switch_at', float('-inf')) < self.NORMAL_SIDE_COMMIT:
            return False
        return (abs(behind) <= self.NORMAL_BEHIND_TOUCH
                and abs(ahead) > self.NORMAL_SWING_REACH)

    @classmethod
    def _nearest_threat(cls, positioned_candidates, player_x, ground_y,
                        speed_profiles, max_range=None):
        """
        The nearest slow walker standing on the player's level within
        THREAT_RANGE (or MAX_RANGE; any sample, focused or not), or None.
        """
        max_range = cls.THREAT_RANGE if max_range is None else max_range
        near = [
            item for item in positioned_candidates
            if abs(item[1] - player_x) <= max_range
            # a weak match is no threat until seen twice (or hurt)
            and (item[3] is not None or item[0][0] >= cls.NORMAL_QUICK_LOCK_SCORE)
            and abs(cls._elevation(item, ground_y)) <= cls.LEDGE_MIN_HEIGHT
            and (item[0][1] >= len(speed_profiles)
                 or speed_profiles[item[0][1]] == 'normal')
        ]
        if not near:
            return None
        return min(near, key=lambda item: (abs(item[1] - player_x), -item[0][0]))

    @classmethod
    def _group_in_attack_range(cls, group, player_x, player_y, min_enemies=None,
                               attack_range=None, vertical_tolerance=None):
        # The nearest enemy must be within normal attack range and at least
        # GROUP_MIN_ENEMIES of the pile within the group skill's reach.
        min_enemies = min_enemies or cls.GROUP_MIN_ENEMIES
        attack_range = attack_range or cls.ATTACK_RANGE
        vertical_tolerance = vertical_tolerance or cls.GROUP_ATTACK_VERTICAL_TOLERANCE
        distances = [
            abs(item[1] - player_x) for item in group
            if abs(item[2] - (player_y + item[0][3])) <= vertical_tolerance
            and abs(item[1] - player_x) <= cls.GROUP_SKILL_REACH
        ]
        return (
            len(distances) >= min_enemies
            and min(distances) <= attack_range
        )

    @classmethod
    def _find_health_bars(cls, region):
        """
        Enemy HP bars in REGION: a list of (centre x, top y, fraction of HP left).
        A bar is two white lines, one a few pixels under the other, with the green
        fill (then red or black for the HP lost) between them.
        """

        if region.size == 0 or region.shape[0] < 10 or region.shape[1] < cls.HP_BAR_MIN_WIDTH:
            return []
        region = region[:, :, :3]
        # The white lines are a pixel thick and can be broken up where the game or a
        # recording blurs them, so small breaks are bridged.
        white = cv2.morphologyEx(
            (region.min(axis=2) >= 200).astype(np.uint8), cv2.MORPH_CLOSE,
            np.ones((1, 7), np.uint8)
        ).astype(bool)
        dark = cv2.morphologyEx(
            (region.max(axis=2) < 90).astype(np.uint8), cv2.MORPH_CLOSE,
            np.ones((1, 7), np.uint8)
        ).astype(bool)
        bars = []
        for gap in (7, 6, 8):
            # ...with the bar's black inner edge right under the top line, which
            # rules out clouds and other white scenery cheaply
            both = white[:-gap] & white[gap:] & dark[1:1 - gap]
            for top in np.flatnonzero(both.sum(axis=1) >= cls.HP_BAR_MIN_WIDTH):
                row = np.concatenate(([False], both[top], [False]))
                edges = np.flatnonzero(row[1:] != row[:-1])
                for start, end in zip(edges[::2], edges[1::2]):
                    width = end - start
                    if not cls.HP_BAR_MIN_WIDTH <= width <= cls.HP_BAR_MAX_WIDTH:
                        continue
                    if any(abs(start - bar[0]) < bar[1] and abs(top - bar[2]) <= 4
                           for bar in bars):
                        continue    # the same bar again, one row or gap along
                    inner = region[top + 2:top + gap - 1, start + 2:end - 2].astype(np.int16)
                    blue, green, red = inner[..., 0], inner[..., 1], inner[..., 2]
                    filled = (green > 140) & (red < 120) & (blue < 120)
                    lost = (
                        ((red > 120) & (green < 90) & (blue < 90))
                        | (inner.max(axis=2) < 90)
                    )
                    if (filled | lost).mean() < 0.8:
                        continue
                    # The fill's rows differ in shade; the fullest row is the HP
                    ratio = float(filled.mean(axis=1).max())
                    bars.append((int(start), int(width), int(top), ratio))
        return [(start + width // 2, top, ratio) for start, width, top, ratio in bars]

    @classmethod
    def _health_bar_ratio(cls, frame, mob_bounds):
        (mob_x1, mob_y1), (mob_x2, _) = mob_bounds
        mob_center_x = (mob_x1 + mob_x2) // 2
        y1 = max(0, mob_y1 - cls.HP_BAR_SEARCH_HEIGHT)
        y2 = max(y1, mob_y1 - 4)
        horizontal_margin = (
            cls.HP_BAR_MAX_WIDTH // 2 + cls.HP_BAR_MAX_CENTER_OFFSET
        )
        x1 = max(0, mob_center_x - horizontal_margin)
        x2 = min(frame.shape[1], mob_center_x + horizontal_margin + 1)
        region = frame[y1:y2, x1:x2, :3]
        if region.size == 0:
            return None

        blue, green, red = cv2.split(region)
        fill_mask = (green > 200) & (red < 80) & (blue < 80)
        best = None
        for row_index, mask_row in enumerate(fill_mask):
            fill_x = np.flatnonzero(mask_row)
            if fill_x.size < 2:
                continue
            runs = []
            run_start = previous = int(fill_x[0])
            for value in fill_x[1:]:
                x = int(value)
                if x > previous + 1:
                    if previous - run_start + 1 >= 2:
                        runs.append((run_start, previous))
                    run_start = x
                previous = x
            if previous - run_start + 1 >= 2:
                runs.append((run_start, previous))

            dark_row = np.all(region[row_index] < 75, axis=1)
            for fill_start, fill_end in runs:
                bar_start = fill_start
                while bar_start > 0 and dark_row[bar_start - 1]:
                    bar_start -= 1
                bar_end = fill_end
                while bar_end + 1 < dark_row.size and dark_row[bar_end + 1]:
                    bar_end += 1
                bar_width = bar_end - bar_start + 1
                if not fill_end - fill_start + 1 < bar_width - 2:
                    continue
                if bar_width > cls.HP_BAR_MAX_WIDTH:
                    continue

                fill_center = x1 + (fill_start + fill_end) // 2
                if abs(fill_center - mob_center_x) > cls.HP_BAR_MAX_CENTER_OFFSET:
                    continue
                ratio = (fill_end - fill_start + 1) / bar_width
                vertical_distance = abs((y1 + row_index) - (mob_y1 - 14))
                rank = (abs(fill_center - mob_center_x), vertical_distance)
                if best is None or rank < best[0]:
                    best = (rank, ratio)
        return best[1] if best is not None else None

    @staticmethod
    def _elevation(item, ground_y):
        """How far above the player's ground line this enemy's feet are (px)."""
        return ground_y - (item[2] + item[0][2].shape[0] / 2)

    @classmethod
    def _is_stand_in(cls, item):
        """An enemy known only by its HP bar (no sample matched its sprite)."""
        candidate = item[0]
        return (candidate[0] == cls.HP_BAR_SCORE
                and candidate[5] == 0 and candidate[6] == 0)

    def _other_level_items(self, positioned_candidates, ground_y, now):
        """
        Follows raised enemies from scan to scan and returns the ones standing on a
        higher level (a raised step, a ledge): raised for LEDGE_CONFIRM_SECONDS at a
        steady height. A hopping enemy changes height all through its hop, so it
        never qualifies, even though a hop goes higher than a step.
        Call with _lock held.
        """
        spots = [
            spot for spot in self.__dict__.setdefault('_level_spots', [])
            if now - spot[3] <= self.LEVEL_SPOT_EXPIRY
        ]
        # An enemy seen down on the ground line was never standing on a step:
        # a pig hopping on the spot passes the same height on every hop, which
        # kept one old spot alive until it looked "raised for a while"
        # (looting-strike-issue.mp4, a grounded pig dropped as out of reach).
        grounded = [
            item[1] for item in positioned_candidates
            if abs(self._elevation(item, ground_y)) <= self.GROUND_MARGIN
            and not self._is_stand_in(item)
        ]
        if grounded:
            spots = [
                spot for spot in spots
                if not any(abs(spot[0] - x) <= self.LEVEL_SPOT_X / 2 for x in grounded)
            ]
        other_level = []
        used = set()
        for item in positioned_candidates:
            elevation = self._elevation(item, ground_y)
            if elevation <= self.LEDGE_MIN_HEIGHT:
                continue
            foot = ground_y - elevation
            best = None
            for index, spot in enumerate(spots):
                if index in used:
                    continue
                dx, dy = abs(item[1] - spot[0]), abs(foot - spot[1])
                if dx <= self.LEVEL_SPOT_X and dy <= self.LEVEL_SPOT_Y:
                    if best is None or dx + dy < best[0]:
                        best = (dx + dy, index)
            if best is None:
                # [x, foot y, first seen, last seen, lowest foot y, highest foot y]
                spots.append([item[1], foot, now, now, foot, foot])
                used.add(len(spots) - 1)
                continue
            used.add(best[1])
            spot = spots[best[1]]
            spot[0], spot[1], spot[3] = item[1], foot, now
            spot[4], spot[5] = min(spot[4], foot), max(spot[5], foot)
            confirm = (
                self.LEDGE_FAST_CONFIRM_SECONDS
                if self._matches_ledge_height(elevation)
                else self.LEDGE_CONFIRM_SECONDS
            )
            if (now - spot[2] >= confirm
                    and spot[5] - spot[4] <= self.LEDGE_Y_JITTER):
                other_level.append(item)
                if self._is_stand_in(item):
                    continue    # a bar's height is a guess: never the step's height
                ledge = self.__dict__.get('_ledge_height')
                self._ledge_height = (
                    elevation if ledge is None else 0.9 * ledge + 0.1 * elevation
                )
        self._level_spots = spots
        return other_level

    def _lower_level_items(self, positioned_candidates, ground_y, now):
        """
        Enemies standing on a level below the player's (the ground beneath the
        raised step while the character is up on it): deeper than
        LEDGE_MIN_HEIGHT below its feet, at a steady screen height AND a steady
        depth for LOWER_CONFIRM_SECONDS. While the character itself is in the air
        the depth of everything below keeps changing, so its jumps never count.
        Call with _lock held.
        """
        spots = [
            spot for spot in self.__dict__.setdefault('_lower_spots', [])
            if now - spot[3] <= self.LEVEL_SPOT_EXPIRY
        ]
        lower = []
        used = set()
        for item in positioned_candidates:
            depth = -self._elevation(item, ground_y)
            if depth <= self.LEDGE_MIN_HEIGHT:
                continue
            foot = ground_y + depth
            best = None
            for index, spot in enumerate(spots):
                if index in used:
                    continue
                dx, dy = abs(item[1] - spot[0]), abs(foot - spot[1])
                if dx <= self.LEVEL_SPOT_X and dy <= self.LEVEL_SPOT_Y:
                    if best is None or dx + dy < best[0]:
                        best = (dx + dy, index)
            if best is None:
                # [x, foot y, first seen, last seen, min foot, max foot,
                #  min depth, max depth]
                spots.append([item[1], foot, now, now, foot, foot, depth, depth])
                used.add(len(spots) - 1)
                continue
            used.add(best[1])
            spot = spots[best[1]]
            spot[0], spot[1], spot[3] = item[1], foot, now
            spot[4], spot[5] = min(spot[4], foot), max(spot[5], foot)
            spot[6], spot[7] = min(spot[6], depth), max(spot[7], depth)
            if (now - spot[2] >= self.LOWER_CONFIRM_SECONDS
                    and spot[5] - spot[4] <= self.LEDGE_Y_JITTER
                    and spot[7] - spot[6] <= self.LEDGE_Y_JITTER):
                lower.append(item)
        self._lower_spots = spots
        return lower

    def process_frame(self, frame, bot_enabled):
        with self._lock:
            self._frame_count += 1
            enabled = self.config['Enabled']
            if not enabled or not bot_enabled:
                self._attack_direction = None
                self._attack_is_group = False
                self._group_count = 0
                self._group_in_range = False
                self._target_offset_x = None
                self._target_vertical_offset = None
                self._target_attackable = False
                self._target_waiting = False
                self._status = 'Waiting for auto-attack to be enabled and started.'
                return
            speed_profiles = self._enemy_speed_profiles_locked()
            # (Slow enemies too once one is locked or being confirmed: at the
            # slow rate the character walked 30-40 px between two looks)
            match_interval = (
                self.FAST_MATCH_INTERVAL
                if {'fast', 'jumper'} & set(speed_profiles)
                or self._locked_target is not None
                or self._pending_target is not None
                else self.MATCH_INTERVAL
            )
            if self._frame_count % match_interval:
                return
            scan_now = time.monotonic()
            last_scan = self.__dict__.get('_last_scan')
            if last_scan is not None and 0 < scan_now - last_scan < 1.0:
                known = self.__dict__.get('_scan_dt')
                self._scan_dt = (
                    scan_now - last_scan if known is None
                    else 0.6 * known + 0.4 * (scan_now - last_scan))
            self._last_scan = scan_now
            self._attack_direction = None
            self._group_count = 0
            self._attack_is_group = False
            self._group_in_range = False
            player_template = self._player_template
            mob_templates = self._mob_templates.copy()
            threshold = max(
                self.MOB_MATCH_MIN_THRESHOLD,
                self.config['Threshold'] / 100
                - self.MOB_MATCH_THRESHOLD_MARGIN
            )
            player_threshold = max(
                self.PLAYER_MATCH_MIN_THRESHOLD,
                threshold - 0.18
            )
            distance = self.config['Range']
            sweep = self._sweep_locked()
            sweep_side = 1 if self._sweep_direction == 'right' else -1

        focused_samples = self.focused_enemy_samples()
        if player_template is None or not mob_templates:
            with self._lock:
                self._status = self._load_error
            return
        if frame is None or frame.ndim != 3:
            return

        gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
        search_height = max(
            1, int(gray.shape[0] * self.PLAYER_SEARCH_HEIGHT_RATIO)
        )
        player_match = self._best_match(gray[:search_height], player_template)
        if player_match is None or player_match[1] < player_threshold:
            with self._lock:
                self._status = 'Player badge not found; make sure it is visible.'
                self._attack_direction = None
                self._attack_is_group = False
                self._group_count = 0
                self._target_offset_x = None
                self._target_vertical_offset = None
                self._target_attackable = False
            return

        player_top, player_score = player_match
        player_x = player_top[0] + player_template.shape[1] // 2
        player_y = player_top[1] + player_template.shape[0] // 2
        with self._lock:
            self._player_position = (player_x, player_y)
            # The badge rises when the character is knocked back (it is not
            # jumping during a fight on a flat map): a hit taken
            seen_at = time.monotonic()
            base = self.__dict__.get('_player_ground')
            knocked = False
            if base is None or player_y >= base[0] or seen_at - base[1] > 1.5:
                base = (player_y, seen_at)      # standing lower, or a new level
            elif player_y >= base[0] - 2:
                base = (base[0], seen_at)       # back on the ground
            else:
                knocked = base[0] - player_y >= self.KNOCKBACK_RISE
            self._player_ground = base
            self._player_knocked = knocked
            if knocked:
                self._knocked_at = seen_at
                self._unignore_near(player_x, seen_at)
            self._frame_width = gray.shape[1]
            self._calibrate_map_scale(player_x, gray.shape[1], seen_at)
            try:
                reader = self._facing_reader()
                walk = self.__dict__.get('_walk_note')
                if walk is not None and seen_at - walk[1] <= self.WALK_NOTE_SECONDS:
                    reader.learn(frame, (player_x, player_y), walk[0], seen_at)
                reader.read(frame, (player_x, player_y), seen_at)
            except cv2.error:
                pass
            locked_target = self._locked_target
            locked_weak = bool(self.__dict__.get('_locked_weak'))
            # The far look is slow, so it runs on every third scan, or on every
            # scan while a far enemy is being confirmed.
            self._scout_tick = getattr(self, '_scout_tick', 0) + 1
            scout = self._scout_tick % 3 == 0 or self._pending_target is not None
        if locked_target is None or (locked_weak and scout):
            # (a weak lock - the portal, decor - must not shut the far look:
            # attack2.mp4, real enemies stood 300-700 px away unseen)
            if scout:
                distance = max(distance, self.SCOUT_RANGE)
        else:   # far enough to keep the locked enemy in view while walking to it
            distance = max(distance, min(
                self.SCOUT_RANGE,
                abs(locked_target[1] - player_x) + 2 * self.TARGET_LOCK_X_RADIUS
            ))
        candidates = []
        sample_best_scores = []
        # Every enemy on the player's platform stands on the same ground line, whatever
        # sample found it, and may be anywhere from there up to a jump above it.
        ground_offset = self._ground_offset(mob_templates)
        lower_look = (self.LOWER_LOOK if 'jumper' in speed_profiles
                      else self.SLOPE_LOOK if sweep else 0)
        for template_index, (mob_template, _) in enumerate(mob_templates):
            mob_search = gray
            if mob_template.ndim == 3:
                mob_search = frame[:, :, :3]
            mask = self._mask_for(mob_template)
            sample_candidates = []
            sample_best_score = 0.0
            sample_threshold = (
                min(0.95, threshold + self.LOOSE_MASK_MARGIN)
                if self._is_loose_mask(mask) else threshold
            )
            separation = (
                self.JUMPER_MATCH_SEPARATION
                if template_index < len(speed_profiles)
                and speed_profiles[template_index] == 'jumper' else 1.0
            )
            for scaled_template, scaled_mask in self._variants_for(mob_template, mask):
                template_height = scaled_template.shape[0]
                # Centre of this sample when its enemy stands on the ground line
                alignment_y = round(ground_offset - template_height / 2)
                ground_y = player_y + ground_offset
                y1 = max(
                    0,
                    round(ground_y - template_height - self.JUMP_HEIGHT)
                )
                y2 = min(
                    gray.shape[0],
                    round(ground_y + self.GROUND_MARGIN + lower_look)
                )
                x1 = max(
                    0,
                    player_x - distance - scaled_template.shape[1]
                )
                x2 = min(gray.shape[1], player_x + distance)
                if scaled_mask is not None:
                    mob_matches, best_score = self._masked_matching_locations(
                        mob_search[y1:y2, x1:x2],
                        scaled_template,
                        scaled_mask,
                        sample_threshold,
                        separation
                    )
                else:
                    mob_matches, best_score = self._matching_locations(
                        mob_search[y1:y2, x1:x2],
                        scaled_template,
                        sample_threshold,
                        separation
                    )
                sample_best_score = max(sample_best_score, best_score)
                sample_candidates.extend(
                    (
                        mob_score,
                        template_index,
                        scaled_template,
                        alignment_y,
                        mob_top,
                        x1,
                        y1
                    )
                    for mob_top, mob_score in mob_matches
                )
            sample_best_scores.append(sample_best_score)

            sample_candidates.sort(key=lambda candidate: candidate[0], reverse=True)
            for candidate in sample_candidates:
                _, _, scaled_template, _, mob_top, x1, y1 = candidate
                center_x = x1 + mob_top[0] + scaled_template.shape[1] // 2
                center_y = y1 + mob_top[1] + scaled_template.shape[0] // 2
                if any(
                    existing[1] == template_index
                    and abs(
                        existing[5] + existing[4][0]
                        + existing[2].shape[1] // 2 - center_x
                    ) < min(existing[2].shape[1], scaled_template.shape[1]) // 2
                    and abs(
                        existing[6] + existing[4][1]
                        + existing[2].shape[0] // 2 - center_y
                    ) < min(existing[2].shape[0], scaled_template.shape[0]) // 2
                    for existing in candidates
                ):
                    continue
                candidates.append(candidate)

        self._note_sample_fit(max(sample_best_scores, default=0.0))

        # A hurt enemy shows an HP bar. Its sprite is often in a hit pose or behind
        # an attack effect, where no sample matches, so the bar itself counts.
        ground_y = player_y + ground_offset
        tallest = max(template.shape[0] for template, _ in mob_templates)
        bar_x1 = max(0, player_x - distance)
        bar_y1 = max(0, round(ground_y - tallest - self.JUMP_HEIGHT - 30))
        health_bars = [
            (bar_x1 + bar_x, bar_y1 + bar_y, ratio)
            for bar_x, bar_y, ratio in self._find_health_bars(
                frame[bar_y1:max(bar_y1, round(ground_y - 20)),
                      bar_x1:player_x + distance]
            )
        ]
        # Pigs stack: two bars by one matched sprite are two pigs, so with
        # 'jumper' samples each matched enemy claims one bar only.
        stacking = 'jumper' in speed_profiles
        claimed = set()
        # (An all-black bar is still a live enemy on its last HP: the game takes
        # the bar away the moment an enemy dies, so any bar on screen, however
        # empty, means it is still there to hit.)
        for bar_x, bar_y, ratio in sorted(health_bars, key=lambda bar: bar[1]):
            owners = [
                index for index, item in enumerate(candidates)
                if abs(item[5] + item[4][0] + item[2].shape[1] // 2 - bar_x)
                <= self.HP_BAR_MATCH_X
            ]
            # How far a bar floats above its enemy's feet, learnt from matched
            # enemies standing on the ground: a stand-in is placed by it.
            for index in owners:
                item = candidates[index]
                foot = item[6] + item[4][1] + item[2].shape[0]
                if abs(ground_y - foot) <= self.GROUND_MARGIN and len(owners) == 1:
                    gap = self.__dict__.get('_bar_foot_gap')
                    sample = foot - bar_y
                    self._bar_foot_gap = (
                        sample if gap is None else 0.8 * gap + 0.2 * sample)
            if owners and not stacking:
                continue
            free = [index for index in owners if index not in claimed]
            if free:
                claimed.add(min(free, key=lambda index: abs(
                    candidates[index][5] + candidates[index][4][0]
                    + candidates[index][2].shape[1] // 2 - bar_x
                )))
                continue
            # Stand-in for the unseen enemy: the first sample's size, on the ground
            # under the bar, or right under the bar when that is clearly higher (a
            # hurt enemy up on the step must not count as one on the ground).
            stand_in = mob_templates[0][0]
            stand_in_alignment = round(ground_offset - stand_in.shape[0] / 2)
            stand_in_top = player_y + stand_in_alignment - stand_in.shape[0] // 2
            # (looting-strike-issue.mp4: the bar floats ~40 px over a pig's head,
            # so "right under the bar" put grounded pigs 33 px up, which was then
            # learnt as a raised step and every pig hopping through that height,
            # or known only by its bar, was dropped as out of reach.)
            bar_gap = self.__dict__.get('_bar_foot_gap')
            if bar_gap is not None:
                raised = ground_y - (bar_y + bar_gap)
                if raised > self.LEDGE_MIN_HEIGHT:
                    stand_in_top = round(bar_y + bar_gap - stand_in.shape[0])
            else:
                under_bar = bar_y + self.HP_BAR_HEAD_GAP
                if under_bar < stand_in_top - self.LEDGE_MIN_HEIGHT - self.HP_BAR_FLOAT:
                    stand_in_top = under_bar
            claimed.add(len(candidates))    # this bar is the stand-in's own
            candidates.append((
                self.HP_BAR_SCORE, 0, stand_in, stand_in_alignment,
                (bar_x - stand_in.shape[1] // 2, stand_in_top),
                0, 0
            ))

        if not candidates:
            with self._lock:
                self._enemy_offsets = (time.monotonic(), [])
                self._attack_direction = None
                self._attack_is_group = False
                self._group_count = 0
            if self._hold_locked_target(
                player_x, player_y, enabled, bot_enabled
            ):
                return
            with self._lock:
                self._target_offset_x = None
                self._target_vertical_offset = None
                self._target_attackable = False
                self._group_count = 0
                self._group_in_range = False
                score_summary = ', '.join(
                    f'{index + 1}:{score:.0%}'
                    for index, score in enumerate(sample_best_scores)
                )
                self._status = (
                    f'Player found ({player_score:.0%}); no nearby enemy match '
                    f'in {len(mob_templates)} sample(s); scores '
                    f'{score_summary}; threshold {threshold:.0%}.'
                )
            return

        positioned_candidates = []
        for candidate in candidates:
            mob_score, template_index, mob_template, alignment_y, mob_top, x1, y1 = candidate
            mob_x = x1 + mob_top[0] + mob_template.shape[1] // 2
            mob_y = y1 + mob_top[1] + mob_template.shape[0] // 2
            mob_bounds = (
                (x1 + mob_top[0], y1 + mob_top[1]),
                (
                    x1 + mob_top[0] + mob_template.shape[1],
                    y1 + mob_top[1] + mob_template.shape[0]
                )
            )
            own_bars = [
                bar for bar in health_bars
                if abs(bar[0] - mob_x) <= self.HP_BAR_MATCH_X and bar[1] < mob_y
            ]
            if own_bars:
                health_ratio = min(
                    own_bars, key=lambda bar: abs(bar[0] - mob_x)
                )[2]
            else:
                health_ratio = self._health_bar_ratio(frame, mob_bounds)
            # (An empty, all-black bar is not a dead enemy but one on its last
            # 1 HP that lives on and bites - knock-back-dmg-2.mp4. It used to be
            # dropped as "dying" and left alive. A dead enemy's bar is taken away
            # by the game, so an enemy is let go only once no bar shows at all.)
            positioned_candidates.append((
                candidate, mob_x, mob_y, health_ratio
            ))

        # A portal is a column of near-white light standing on the ground
        # (0.54-0.75 of the column in attack1/attack2/ghost-attack.mp4, under
        # 0.3 anywhere else). A poor, bar-less match inside one is the portal.
        positioned_candidates = [
            item for item in positioned_candidates
            if not (self._is_weak(item)
                    and self._portal_light(frame, item[1], ground_y) >= self.PORTAL_LIGHT)
        ]
        if sweep:
            # Up or down a ramp from the character: on its level all the same
            positioned_candidates = [
                self._on_slope(item, player_x, ground_y)
                for item in positioned_candidates
            ]
        now = time.monotonic()
        with self._lock:
            # Before the ignore filter: an enemy dropped for standing on the step
            # is still up there, and still counts for luring them down
            other_level = self._other_level_items(positioned_candidates, ground_y, now)
            if other_level:
                nearest = min(other_level, key=lambda item: abs(item[1] - player_x))
                self._ledge_seen = (
                    now, len(self._merge_same_enemy(
                        self._deduplicate_positioned_candidates(other_level))),
                    nearest[1] - player_x,
                )
            # Below the player's level (only searched on jumper maps): seen for
            # walking down to them, never a target from up here
            lower = self._lower_level_items(positioned_candidates, ground_y, now)
            if lower:
                nearest = min(lower, key=lambda item: abs(item[1] - player_x))
                self._lower_seen = (
                    now, len(self._merge_same_enemy(
                        self._deduplicate_positioned_candidates(lower))),
                    nearest[1] - player_x,
                )
            positioned_candidates = [
                item for item in positioned_candidates
                if self._elevation(item, ground_y) >= -self.LEDGE_MIN_HEIGHT
            ]
            self._ignored_spots = [
                spot for spot in self._ignored_spots if spot[2] > now
            ]
            ignored_spots = list(self._ignored_spots)
        if ignored_spots:
            # A hurt enemy (its HP bar shows) is real even on an ignored spot
            positioned_candidates = [
                item for item in positioned_candidates
                if item[3] is not None
                or not self._on_ignored_spot(item, ignored_spots, player_x)
            ]

        if 'jumper' not in speed_profiles:
            positioned_candidates = self._inside_map_limits(positioned_candidates, player_x)

        with self._lock:
            blocked_side = self.__dict__.get('_blocked_side')
            locked = self._locked_target
            if locked is not None and any(
                    abs(item[1] - locked[1]) <= self.TARGET_LOCK_X_RADIUS
                    and abs(item[2] - locked[2]) <= self.TARGET_LOCK_RADIUS
                    for item in other_level):
                # The locked enemy turned out to stand on a step out of reach:
                # let go now instead of walking at it for the whole grace period
                self._release_lock_locked()
        unreachable = {id(item) for item in other_level}
        # Past a wall means past the swing's reach from where the character
        # stands. Pigs pile up against the Pig Beach cliff, 60-105 px away:
        # within a jumper swing, but they were dropped at the normal 60 px
        # and the bot patrolled away from the pile.
        blocked_reach = (
            self.JUMPER_ATTACK_RANGE + self.JUMPER_RANGE_HYSTERESIS
            if 'jumper' in speed_profiles else self.ATTACK_RANGE
        )
        if blocked_side == 'right':
            unreachable |= {
                id(item) for item in positioned_candidates
                if item[1] > player_x + blocked_reach
            }
        elif blocked_side == 'left':
            unreachable |= {
                id(item) for item in positioned_candidates
                if item[1] < player_x - blocked_reach
            }
        if blocked_side is not None and 'jumper' not in speed_profiles:
            # Slow-walker maps: past the wall the walk stopped at, out of reach,
            # with no HP bar: scenery past the map's edge (knock-back-dmg-3.mp4:
            # the portal; the bot walked at it, hit the edge, turned back, saw
            # it again, and swung and used Q at it). Kept by its map spot.
            with self._lock:
                for item in positioned_candidates:
                    past = (item[1] - player_x if blocked_side == 'right'
                            else player_x - item[1])
                    if (id(item) in unreachable and item[3] is None
                            and past > blocked_reach):
                        self._ignore_spot(item[1], item[2], now,
                                          self.SCENERY_IGNORE_SECONDS)
        if unreachable:
            positioned_candidates = [
                item for item in positioned_candidates if id(item) not in unreachable
            ]
        with self._lock:
            # In reach on the player's own level: standing on the ground line (a
            # hopping one lands within a second), and not holding still at the
            # step's height
            if any(
                    self._elevation(item, ground_y) <= self.LEDGE_MIN_HEIGHT
                    for item in positioned_candidates):
                self._ground_seen = now
            on_step = [
                item for item in positioned_candidates
                if self._at_ledge_height(item, ground_y)
            ]
            locked = self._locked_target
            if on_step and locked is not None:
                # The locked enemy passing through the step's height is mid-hop
                # (it was just on the ground): keep it. One that really stands
                # up there holds that height, and is dropped a moment later by
                # the steady-height check below (LEDGE_FAST_CONFIRM_SECONDS).
                # Dropping it here lost the lock on every hop, and each re-lock
                # cost scans during which the pig walked into the character.
                on_step = [
                    item for item in on_step
                    if abs(item[1] - locked[1]) > self.JUMPER_LOCK_RADIUS
                ]
            if on_step:
                unreachable |= {id(item) for item in on_step}
                positioned_candidates = [
                    item for item in positioned_candidates
                    if id(item) not in unreachable
                ]

        if not positioned_candidates:
            with self._lock:
                self._enemy_offsets = (now, [])
                self._attack_direction = None
                self._attack_is_group = False
                self._group_count = 0
            if self._hold_locked_target(
                player_x, player_y, enabled, bot_enabled
            ):
                return
            with self._lock:
                self._target_offset_x = None
                self._target_vertical_offset = None
                self._target_attackable = False
                self._group_in_range = False
                self._target_waiting = False
                self._status = (
                    'Player found; only enemies out of reach (raised step or '
                    'past a wall) or scenery in view.'
                    if unreachable else 'Player found; only scenery in view.'
                )
            return

        positioned_candidates = self._merge_same_enemy(
            self._deduplicate_positioned_candidates(positioned_candidates)
        )
        # Slow walkers never jump: one standing well above the player's ground
        # is on a ledge out of reach (knock-back-dmg-2.mp4: the bot turned to
        # walk 430 px to a snail up on the ledge while a shroom bit it)
        reachable = [
            item for item in positioned_candidates
            if not (item[0][1] >= len(speed_profiles)
                    or speed_profiles[item[0][1]] == 'normal')
            or self._elevation(item, ground_y) <= self.LEDGE_MIN_HEIGHT + 10
        ]
        target_candidates = self._preferred_candidates(
            reachable, focused_samples
        ) if reachable else []
        left_behind = 0
        if sweep and target_candidates:
            # The sweep clears what is ahead; what is well behind it waits for
            # the way back (one close behind still bites, so it still counts)
            ahead = [
                item for item in target_candidates
                if (item[1] - player_x) * sweep_side >= -self.SWEEP_BEHIND_PX
            ]
            left_behind = len(target_candidates) - len(ahead)
            target_candidates = ahead
            if left_behind:
                with self._lock:
                    locked = self._locked_target
                    if locked is not None and (
                            (locked[1] - player_x) * sweep_side < -self.SWEEP_BEHIND_PX):
                        self._release_lock_locked()
        if not target_candidates:
            if self._hold_locked_target(player_x, player_y, enabled, bot_enabled):
                return
            with self._lock:
                self._enemy_offsets = (now, [])
                self._target_offset_x = None
                self._target_vertical_offset = None
                self._target_attackable = False
                self._group_in_range = False
                self._target_waiting = False
                self._attack_direction = None
                self._attack_is_group = False
                self._status = (
                    f'Player found; {left_behind} enemy(ies) behind the sweep, left '
                    'for the way back.' if left_behind
                    else 'Player found; only enemies up on a ledge in view.')
            return
        jumpers = 'jumper' in speed_profiles
        group_min = self.JUMPER_GROUP_MIN_ENEMIES if jumpers else self.GROUP_MIN_ENEMIES
        with self._lock:
            group_skill = self._group_skill_locked()
            self._enemy_offsets = (now, [
                item[1] - player_x for item in positioned_candidates
                if abs(self._elevation(item, ground_y)) <= self.LEDGE_MIN_HEIGHT
            ])
        group = self._nearest_enemy_group(
            positioned_candidates, player_x, min_enemies=group_min,
            vertical_radius=(
                self.JUMPER_GROUP_VERTICAL_RADIUS if jumpers else None
            ),
        ) if group_skill else []    # no group skill: one enemy at a time
        group_count = len(group)
        with self._lock:
            self._group_count = group_count
            locked_target = self._locked_target
        # Slow-walker maps: whatever stands nearest inside THREAT_RANGE is the
        # one that bites next, so it is the target (focus and lock come second)
        threat = None if jumpers else self._nearest_threat(
            positioned_candidates, player_x, ground_y, speed_profiles
        )
        selected = None
        if locked_target is not None:
            _, locked_x, locked_y = locked_target
            locked_candidates = [
                item for item in target_candidates
                if abs(item[1] - locked_x) <= {
                    'fast': self.FAST_LOCK_RADIUS,
                    'jumper': self.JUMPER_LOCK_RADIUS,
                }.get(
                    speed_profiles[locked_target[0]]
                    if locked_target[0] < len(speed_profiles) else 'normal',
                    self.TARGET_LOCK_X_RADIUS
                )
            ]
            locked_profile = (
                speed_profiles[locked_target[0]]
                if locked_target[0] < len(speed_profiles) else 'normal')
            if (not locked_candidates and locked_profile == 'jumper'
                    and abs(locked_x - player_x) <= self.JUMPER_CROSS_NEAR):
                crossed = [
                    item for item in target_candidates
                    if abs(item[1] - player_x) <= self.JUMPER_CROSS_RADIUS
                ]
                if crossed:
                    # It walked through the character: follow it to that side
                    # (nearest to the player, not to its old spot)
                    locked_x = player_x
                    locked_candidates = crossed
            if locked_candidates:
                selected = min(
                    locked_candidates,
                    key=lambda item: abs(item[1] - locked_x)
                    + 0.25 * abs(item[2] - locked_y)
                )
            elif threat is None:
                if self._hold_locked_target(
                    player_x, player_y, enabled, bot_enabled
                ):
                    return

        if threat is None and not jumpers and selected is not None                 and abs(selected[1] - player_x) > self.FAR_LOCK_PX:
            # A far lock (knock-back-dmg-3.mp4: the portal 500 px off) does not
            # win over an enemy much closer: walking away from a shroom that
            # is coming, toward something across the map, ends with a bite
            nearer = self._nearest_threat(
                target_candidates, player_x, ground_y, speed_profiles,
                max_range=abs(selected[1] - player_x) - self.FAR_LOCK_MARGIN)
            if nearer is not None:
                threat = nearer
        gave_way = False
        if (not jumpers and selected is not None and self._is_weak(selected)
                and not self.__dict__.get('_locked_bar_seen')):
            strong = [item for item in target_candidates
                      if item is not selected and not self._is_weak(item)]
            with self._lock:
                since = self.__dict__.get('_normal_swing_since')
            gap = abs(selected[1] - player_x)
            in_reach = gap <= self.NORMAL_ATTACK_RANGE + self.NORMAL_RANGE_HYSTERESIS
            # In reach it gets a swing or two first (a real one then shows its
            # HP bar and is weak no more). After that it gives way to any
            # strong enemy in view; one it is only walking to gives way to a
            # strong one up to WEAK_PENALTY px further off.
            tried = in_reach and since is not None and (
                now - since >= self.WEAK_TRIAL_SECONDS)
            if strong and (tried or not in_reach):
                best = min(strong, key=lambda item: abs(item[1] - player_x))
                if tried or abs(best[1] - player_x) <= gap + self.WEAK_PENALTY:
                    threat, gave_way = best, True
                    if tried:
                        with self._lock:    # swung at, nothing: scenery
                            self._ignore_spot(selected[1], selected[2], now,
                                              self.SCENERY_IGNORE_SECONDS)
        switched = False
        if threat is not None and (
                selected is None or gave_way
                or (selected is not threat and abs(selected[1] - player_x)
                    > abs(threat[1] - player_x) + self.THREAT_MARGIN
                    and self._may_switch_side(selected, threat, player_x, now))):
            if selected is not None and (selected[1] - player_x) * (threat[1] - player_x) < 0:
                self._side_switch_at = now
            if gave_way:
                with self._lock:
                    self._normal_commit = None      # no follow-up swings at the scenery
            with self._lock:
                if (selected is None and self._locked_target is not None
                        and self._locked_health is not None
                        and self._locked_health <= self.LOW_HEALTH_RATIO):
                    # The old target was nearly dead and is gone: a kill (loot)
                    self.__dict__.setdefault('_kill_events', []).append(
                        (now, self._locked_target[1] - player_x))
                selected = threat
                switched = True
                self._locked_target = (threat[0][1], threat[1], threat[2])
                self._pending_target = None
                self._in_range_since = None
                self._normal_reach_at = None

        fresh_lock = selected is None or switched
        if selected is None:
            with self._lock:
                sweep_direction = self._sweep_direction
            selected = self._choose_target(
                target_candidates, player_x, sweep_direction, ground_y,
                weak_penalty=0 if jumpers else self.WEAK_PENALTY
            )
            candidate, selected_x, selected_y, _ = selected
            template_index = candidate[1]
            with self._lock:
                pending = self._pending_target
                if (pending is not None
                        and abs(pending[1] - selected_x) <= self.TARGET_LOCK_RADIUS
                        and abs(pending[2] - selected_y) <= self.TARGET_LOCK_RADIUS):
                    confirmations = pending[3] + 1
                else:
                    confirmations = 1
                self._pending_target = (
                    template_index, selected_x, selected_y, confirmations
                )
                needed = self.TARGET_CONFIRMATIONS
                quick_lock = {
                    'jumper': self.JUMPER_QUICK_LOCK,
                    'normal': self.NORMAL_QUICK_LOCK,
                }.get(
                    speed_profiles[template_index]
                    if template_index < len(speed_profiles) else 'normal')
                slow = quick_lock == self.NORMAL_QUICK_LOCK and (
                    template_index >= len(speed_profiles)
                    or speed_profiles[template_index] == 'normal')
                if (quick_lock is not None
                        and (selected[3] is not None
                             or (abs(selected_x - player_x) <= quick_lock
                                 # (henny_bug-2.mp4: flowers by the portal
                                 # matched at 66% for one scan now and then)
                                 and (not slow
                                      or candidate[0] >= self.NORMAL_QUICK_LOCK_SCORE)))):
                    # Showing an HP bar (so it is real) or about to touch the
                    # character: a second look costs a scan it does not have
                    needed = 1
                if confirmations < needed:
                    self._target_offset_x = None
                    self._target_attackable = False
                    self._attack_direction = None
                    self._attack_is_group = False
                    self._status = (
                        f'Confirming enemy at ({selected_x}, {selected_y}) '
                        f'({confirmations}/{self.TARGET_CONFIRMATIONS}); '
                        f'match {candidate[0]:.0%}.'
                    )
                    return
                self._locked_target = (
                    template_index, selected_x, selected_y
                )
                self._pending_target = None
                self._in_range_since = None     # a newly chosen enemy
                self._normal_reach_at = None

        candidate, _, _, health_ratio = selected
        mob_score, template_index, mob_template, alignment_y, mob_top, x1, y1 = candidate
        mob_x = x1 + mob_top[0] + mob_template.shape[1] // 2
        mob_y = y1 + mob_top[1] + mob_template.shape[0] // 2
        delta_x = mob_x - player_x
        delta_y = mob_y - (player_y + alignment_y)
        direction = 'left' if delta_x < 0 else 'right'
        # delta_y is 0 for an enemy standing on the player's ground line and negative
        # while it is in the air; it comes back down, so a jump doesn't end the fight.
        airborne = delta_y < -self.VERTICAL_TOLERANCE
        on_level = -self.JUMP_HEIGHT - self.GROUND_MARGIN <= delta_y <= self.VERTICAL_TOLERANCE
        profile = (
            speed_profiles[template_index]
            if template_index < len(speed_profiles) else 'normal'
        )
        waiting = False
        pursuit_x = delta_x
        no_damage_seconds = None
        with self._lock:
            same_target = (
                not fresh_lock
                and self.__dict__.get('_locked_seen') is not None
            )
            was_attackable = self._target_attackable and same_target
            # A locked enemy that stays raised long after any hop would have
            # ended is standing on something higher
            elevated_since = self.__dict__.get('_locked_elevated_since')
            if delta_y < -self.LEDGE_MIN_HEIGHT:
                if elevated_since is None or not same_target:
                    elevated_since = now
            else:
                elevated_since = None
            self._locked_elevated_since = elevated_since
            # Holding still at the known step height: up there, drop it soon
            band = self.__dict__.get('_locked_band')
            if (elevated_since is not None and same_target
                    and self._matches_ledge_height(-delta_y)):
                if band is None or abs(-delta_y - band[1]) > self.LEDGE_Y_JITTER:
                    band = (now, -delta_y)
            else:
                band = None
            self._locked_band = band
            stuck_up = elevated_since is not None and (
                now - elevated_since >= self.ELEVATED_DROP_SECONDS
                or (band is not None
                    and now - band[0] >= self.LEDGE_FAST_CONFIRM_SECONDS)
            )
            velocity = self.__dict__.get('_locked_velocity', 0.0) if same_target else 0.0
            seen = self.__dict__.get('_locked_seen')
            if same_target and now > seen[1]:
                velocity = 0.5 * velocity + 0.5 * (mob_x - seen[0]) / (now - seen[1])
            self._locked_velocity = velocity
            self._locked_seen = (mob_x, now)
            # Closing speed from the player-relative offset (the camera and the
            # character both move): positive while it comes at the character
            rel = self.__dict__.get('_locked_rel') if same_target else None
            closing = self.__dict__.get('_locked_closing', 0.0) if same_target else 0.0
            if rel is not None and now > rel[1]:
                sample = (abs(rel[0]) - abs(delta_x)) / (now - rel[1])
                if (rel[0] < 0) != (delta_x < 0) and abs(delta_x) > 20:
                    sample = 0.0    # it passed through: not approaching any more
                closing = 0.5 * closing + 0.5 * sample
            self._locked_rel = (delta_x, now)
            self._locked_closing = closing
            # Proof that it is a real enemy, never scenery: it shows an HP bar,
            # it knocked the character back from close by, or it moved
            real = self.__dict__.get('_locked_real', False) if same_target else False
            if health_ratio is not None:
                real = True
            # (knocked back just now, maybe while it was hidden under the
            # character: the bite comes exactly then)
            # (...and only when nothing else stands nearer: attack1.mp4, a red
            # snail bit the character while the lock was on the portal behind
            # it; the portal counted as real and was swung at for 6 s more)
            if (now - self.__dict__.get('_knocked_at', float('-inf')) <= self.KNOCK_MEMORY
                    and abs(delta_x) <= self.REAL_NEAR_PX
                    and not any(
                        abs(item[1] - player_x) < abs(delta_x) - self.THREAT_MARGIN
                        for item in positioned_candidates if item is not selected)):
                real = True
            had_bar = bool(self.__dict__.get('_locked_bar_seen')) and same_target
            bar_seen = had_bar or health_ratio is not None
            self._locked_bar_seen = bar_seen
            # The enemy that showed an HP bar is gone, and what matches at its
            # spot now matches poorly: it died there and scenery is left (the
            # portal, a red bulb). attack1.mp4: the lock passed from the dead
            # snail to the portal behind it, kept the snail's "real" proof,
            # and only the 6 s backstop ended the swinging.
            faded = self.__dict__.get('_locked_faded_since') if same_target else None
            if (had_bar and health_ratio is None and profile == 'normal'
                    and mob_score < self.WEAK_SCORE):
                faded = now if faded is None else faded
            else:
                faded = None
            self._locked_faded_since = faded
            if faded is not None and now - faded >= self.WEAK_AFTER_BAR:
                self.__dict__.setdefault('_kill_events', []).append((now, delta_x))
                self._ignore_spot(mob_x, mob_y, now)
                self._locked_real = False
                self._locked_faded_since = None
                self._normal_commit = None      # no follow-up swings at it either
                self._release_lock_locked()
                self._status = (
                    f'The HP bar at ({mob_x}, {mob_y}) is gone and what is left there '
                    f'matches only {mob_score:.0%}: that enemy died. Moving on.')
                return
            self._locked_weak = (profile == 'normal' and not bar_seen
                                 and mob_score < self.WEAK_SCORE)
            # (Only with a measured map scale: on the snail map 1 minimap unit
            # was ~4100 px, not the default 6000, and the portal "moved")
            measured = getattr(getattr(_config, 'bot', None), '__dict__', {}).get(
                '_px_per_minimap')
            world = self._world_x(mob_x, player_x) if measured else None
            spread = self.__dict__.get('_locked_world') if same_target else None
            if world is not None:
                spread = (world, world) if spread is None else (
                    min(spread[0], world), max(spread[1], world))
                if (spread[1] - spread[0]) * self._px_per_minimap() >= self.REAL_MOVE_PX:
                    real = True
            self._locked_world = spread
            self._locked_real = real
            swing_since = (self.__dict__.get('_normal_swing_since')
                           if same_target else None)
            if stuck_up:
                self._ignore_spot(mob_x, mob_y, now)
                self._release_lock_locked()
                self._status = (
                    f'Dropped the target at ({mob_x}, {mob_y}): it stayed '
                    f'{-delta_y}px up, so it is on a raised step, not hopping.'
                )
                return
        if profile == 'jumper':
            # Rhythm for hopping enemies: keep swinging a little past range (no
            # walk/stab flicker as it bounces about), stand still while it is
            # mid-hop above the swing, and walk toward where it is heading.
            reach, whiffing = self._jumper_attack_reach(
                same_target, was_attackable, health_ratio, swing_since,
                abs(delta_x), now)
            lead = max(-self.JUMPER_MAX_LEAD, min(
                self.JUMPER_MAX_LEAD, velocity * self.JUMPER_LEAD_SECONDS
            ))
            pursuit_x = round(delta_x + lead)
            # Charging in: swing at where it will be when the stab comes out
            distance = abs(delta_x)
            coming = closing >= self.JUMPER_CLOSING_MIN
            early = (not whiffing and coming and distance <= self.JUMPER_EARLY_MAX
                     and distance - closing * self.JUMPER_SWING_LEAD <= reach)
            attackable = (distance <= reach or early) and on_level
            if (coming and not attackable and on_level
                    and distance - closing * self.JUMPER_BRACE_SECONDS <= reach):
                waiting = True      # it is coming: stand facing it, swing on arrival
                pursuit_x = delta_x
            if attackable and delta_y < -self.JUMPER_REACH_HEIGHT:
                attackable = False
                waiting = True
            with self._lock:
                self._normal_swing_since = (
                    (swing_since or now) if attackable and health_ratio is None
                    else None)
            no_damage_seconds = self.JUMPER_NO_DAMAGE_SECONDS
        elif profile == 'normal':
            # Slow walkers. Two speeds matter: how fast the gap shrinks (the
            # character's walk plus the enemy's crawl) decides when to stop
            # walking, since the walk goes on for about a scan after this look;
            # the enemy's own crawl decides when to swing, so the hit lands as
            # it reaches the weapon. A stopped character does not walk into an
            # enemy that is coming at it: it stands facing it and lets it come.
            # Knocked back just past the range: keep swinging; swings that
            # raise no HP bar: out of the weapon's reach, step in.
            distance = abs(delta_x)
            with self._lock:
                reach_at = self.__dict__.get('_normal_reach_at')
                scan_dt = self.__dict__.get('_scan_dt') or 0.2
                # (standing for a ranged cast or letting a pulled enemy come:
                # the bot says so, see note_standing)
                was_walking = (same_target and not was_attackable
                               and not self.__dict__.get('_target_waiting')
                               and now - self.__dict__.get('_standing_at', float('-inf'))
                               > self.STANDING_HINT_SECONDS)
                walk_part = (self.__dict__.get('_normal_walk_part', 0.0)
                             if same_target else 0.0)
                # (smoothed like the closing speed, so the two cancel)
                walk_part = 0.5 * walk_part + 0.5 * (
                    self.NORMAL_WALK_SPEED if was_walking else 0.0)
                self._normal_walk_part = walk_part
            own = closing - walk_part           # the enemy's own approach speed
            with self._lock:
                bar_whiff = self._note_normal_hits(
                    same_target, health_ratio, distance, was_attackable, now)
            reach = self.NORMAL_ATTACK_RANGE
            whiffing = bar_whiff or (
                health_ratio is None and swing_since is not None
                and now - swing_since >= self.NORMAL_WHIFF_SECONDS
            )
            enter = min(self.NORMAL_WHIFF_RANGE, reach) if whiffing else reach
            lead = min(self.NORMAL_STOP_LEAD_MAX, scan_dt + self.NORMAL_KEY_LAG)
            coming = own >= self.NORMAL_COMING_MIN
            arriving = (
                not whiffing and coming
                and distance - own * self.NORMAL_SWING_WINDUP
                <= min(self.NORMAL_SWING_REACH, reach)
            )
            stop_walk = (
                was_walking and not whiffing
                and distance <= self.NORMAL_EARLY_MAX
                and distance - max(closing, 0.0) * lead <= enter
            )
            if distance <= enter or arriving:
                reach_at = now
            held = (
                was_attackable and not whiffing and reach_at is not None
                and distance <= enter + self.NORMAL_RANGE_HYSTERESIS
                and now - reach_at <= self.NORMAL_HYSTERESIS_SECONDS
            )
            attackable = (distance <= enter or arriving or held) and on_level
            if (sweep and not attackable and was_attackable and on_level
                    and distance <= self.SWEEP_HOLD_RANGE):
                # Knocked back out of reach mid-fight: the attack stays held
                # and the bot steps in (Bot._service_step_in)
                attackable = True
            if not attackable and on_level and (
                    stop_walk or (coming and distance <= self.NORMAL_WAIT_MAX)):
                waiting = True      # stand facing it; swing as it arrives
                pursuit_x = delta_x
            with self._lock:
                self._normal_reach_at = reach_at
                self._normal_swing_since = (
                    (swing_since or now) if attackable and health_ratio is None
                    else None)
        else:
            attackable = abs(delta_x) <= self.ATTACK_RANGE and on_level
        group_direction = None
        group_in_range = False
        if group:
            group_center_x = sum(item[1] for item in group) / group_count
            group_direction = 'left' if group_center_x < player_x else 'right'
            if jumpers:
                # Hopping pigs are still in the pile, and the pig swing range applies
                group_in_range = self._group_in_attack_range(
                    group, player_x, player_y, min_enemies=group_min,
                    attack_range=self.JUMPER_ATTACK_RANGE + self.JUMPER_RANGE_HYSTERESIS,
                    vertical_tolerance=self.JUMP_HEIGHT,
                )
            else:
                group_in_range = self._group_in_attack_range(
                    group, player_x, player_y,
                    attack_range=self.NORMAL_ATTACK_RANGE,
                )

        with self._lock:
            if profile == 'normal':
                bar_near = any(
                    item[3] is not None and abs(item[1] - player_x) <= self.BLANK_AREA_PX
                    for item in positioned_candidates)
                knocked_lately = (now - self.__dict__.get('_knocked_at', float('-inf'))
                                  <= self.KNOCK_MEMORY)
                total = self._blank_step(now, attackable,
                                         bar_near or knocked_lately or real)
                futile = self._futile_step(now, attackable,
                                           bar_near or health_ratio is not None)
                if attackable and futile >= self.FUTILE_SECONDS:
                    total = self.BLANK_AREA_SECONDS     # backstop: drop it all
                if attackable and total >= self.BLANK_AREA_SECONDS:
                    seconds = self.SCENERY_IGNORE_SECONDS
                    dropped = 0
                    for item in positioned_candidates:
                        if (item[3] is None
                                and abs(item[1] - player_x) <= self.BLANK_AREA_PX):
                            self._ignore_spot(item[1], item[2], now, seconds)
                            dropped += 1
                    self._ignore_spot(mob_x, mob_y, now, seconds)
                    self._blank_total = 0.0
                    self._futile_total = 0.0
                    self._locked_real = False
                    self._release_lock_locked()
                    self._in_range_since = None
                    self._group_in_range = False
                    self._status = (
                        f'Swung for {self.BLANK_AREA_SECONDS:.1f} sec here with no HP '
                        f'bar and no hit taken: {dropped + 1} match(es) around the '
                        'character are scenery (a portal?). Ignoring them.'
                    )
                    return
            counting = attackable
            if profile == 'normal':
                # (within the swing range: the map's edge kept the character
                # ~55 px from the portal, never within the 52 px whiff range)
                counting = (attackable and not real
                            and abs(delta_x) <= self.NORMAL_ATTACK_RANGE)
            if self._takes_no_damage(counting, health_ratio, now, no_damage_seconds):
                seconds = (self.SCENERY_IGNORE_SECONDS if profile == 'normal'
                           else None)
                self._ignore_spot(mob_x, mob_y, now, seconds)
                if profile == 'normal':
                    # A portal gives 2-3 matches side by side: all of it
                    for item in positioned_candidates:
                        if (item[3] is None and item[1] != mob_x
                                and abs(item[1] - mob_x) <= self.SCENERY_CLUSTER_PX):
                            self._ignore_spot(item[1], item[2], now, seconds)
                self._release_lock_locked()
                self._in_range_since = None
                self._group_in_range = False
                self._status = (
                    f'Dropped the target at ({mob_x}, {mob_y}): attacked for '
                    f'{self.NO_DAMAGE_SECONDS:.1f} sec with no HP bar, so it is '
                    'scenery that looks like a sample. Ignoring that spot.'
                )
                return
            self._locked_target = (template_index, mob_x, mob_y)
            self._locked_health = health_ratio
            self._target_missed_scans = 0
            self._target_missed_since = None
            self._status = (
                f'Locked enemy {mob_score:.0%} among {len(mob_templates)} sample(s), '
                f'offset ({delta_x:+d}, {delta_y:+d}) px; '
                f'{f"HP ~{health_ratio:.0%}; " if health_ratio is not None else ""}'
                f'{"in attack range" if attackable else ("letting it come" if profile == "normal" else "waiting for it to land") if waiting else "approaching"}'
                f'{" (enemy is jumping)" if airborne and on_level else ""} '
                'on calibrated level.'
            )
            if group:
                self._status += (
                    f' Group of {group_count} detected; '
                    f'{"in group-attack range" if group_in_range else "approaching group"}.'
                )
            if profile == 'normal' and not jumpers and on_level:
                attackable, waiting, pursuit_x, direction = self._normal_follow_up(
                    attackable, waiting, pursuit_x, direction, delta_x,
                    positioned_candidates, player_x, ground_y, now)
            # (Slow-walker maps: walk at the nearest enemy, not at the middle of
            # the pile, which is a walk into the first snail of it)
            self._target_offset_x = (
                group_center_x - player_x if group and jumpers else pursuit_x)
            self._target_waiting = waiting and not group_in_range
            self._target_vertical_offset = 0 if on_level else delta_y
            self._target_attackable = attackable or group_in_range
            serial = self.__dict__.get('_lock_serial', 0)
            if not same_target:
                serial += 1
            self._lock_serial = serial
            self._target_info = {
                'time': now, 'serial': serial, 'offset': delta_x, 'dy': delta_y,
                'on_level': on_level, 'airborne': airborne, 'profile': profile,
                'health': health_ratio, 'real': real, 'closing': closing,
                'attackable': attackable or group_in_range,
                'waiting': self._target_waiting,
                'sweep': sweep,
            }
            self._group_in_range = group_in_range and group_count >= group_min
            if not enabled or not bot_enabled:
                self._attack_direction = None
                self._attack_is_group = False
            elif not is_game_focused():
                self._status += ' Waiting for MapleStory focus.'
                self._attack_direction = None
                self._attack_is_group = False
            elif time.monotonic() - self._last_attack_time >= self.ATTACK_COOLDOWN:
                if group_in_range:
                    self._attack_direction = group_direction
                    self._attack_is_group = True
                elif attackable:
                    self._attack_direction = direction
                    self._attack_is_group = False
                else:
                    self._attack_direction = None
                    self._attack_is_group = False
                if self._attack_direction is not None:
                    self._last_attack_time = time.monotonic()
