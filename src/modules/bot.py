"""An interpreter that reads and executes user-created routines."""

import threading
import random
import time
import git
import cv2
import inspect
import importlib
import traceback
from os.path import splitext, basename
from src.common import config, utils
from src.detection import detection
from src.routine import components
from src.routine.routine import Routine
from src.command_book.command_book import CommandBook
from src.routine.components import Point
from src.common import vkeys
from src.common.vkeys import press, click, key_down, key_up, key_repeat
from src.common.interfaces import Configurable
from src.modules.health import is_game_focused
from src.modules.recovery import LevelRecovery
from src.modules.loot_planner import LootPlanner
from src.modules.loot_hand import LootHand
from src.modules.attack_hand import AttackHand
from src.modules.map_events import MapEventManager


# (Tests swap `press` for a recorder; the loot key then stays on the calling
# thread so they see every tap. Live, it has a thread of its own: LootHand.)
_BLOCKING_PRESS = press

# The rune's buff icon
# Climb back to the patrol platform after being knocked off it (src/modules/recovery.py).
# Off for now: farming on the ground, where there is nothing to fall from.
FALL_RECOVERY = False

RUNE_BUFF_TEMPLATE = cv2.imread('assets/rune_buff_template.jpg', 0)


class Bot(Configurable):
    """A class that interprets and executes user-defined routines."""

    PATROL_TURN_INTERVAL = 2.5      # timed turns, only while the minimap position is unknown
    BLIND_PATROL_SECONDS = 15.0     # ...and only this long: then stand and wait for it
    PATROL_MAX_LEG = 12.0           # longest walk one way with no enemy in sight
    PATROL_STUCK_SECONDS = 1.0      # not moving for this long: a wall or the map's end
    PATROL_RESUME_GAP = 0.25        # a break in patrol ticks this long restarts the stuck timer
    PATROL_MOVE_EPSILON = 0.004     # minimap units
    # Walking at an enemy but not moving: a wall, the map's end, or a raised step of
    # the ground floor (Pig Beach) that only a jump gets onto. Turn back instead of
    # pushing into it; the spot is remembered so later passes turn there at once.
    PURSUIT_STUCK_SECONDS = 0.8
    WALL_NEAR = 0.008               # this close to a known wall counts as at it
    WALL_RELEASE = 0.04             # this far back from it, enemies past it count again
    # Safety against flip-flopping (seen at the Pig Beach step when the game lagged:
    # a frozen minimap position "blocked" both ways, so it turned every tick).
    POSITION_STALE_SECONDS = 0.5    # an older minimap fix counts as no position
    MIN_WALL_GAP = 0.05             # a left and a right wall closer than this conflict
    TURN_COOLDOWN = 0.5             # no wall turn sooner than this after the last one
    FLIP_LIMIT = 6                  # this many left/right swaps...
    FLIP_WINDOW = 3.0               # ...this quickly: forget walls and start over
    PURSUIT_DEADBAND = 12           # target this close sideways: don't step at it
    # Jumper maps: a pig overlapping the character flips left/right every scan,
    # and every turn restarts the swing. Keep facing until it is this far over.
    JUMPER_FACE_DEADBAND = 18
    NORMAL_FACE_DEADBAND = 10       # a snail right on top: don't flip every scan
    # Pig Beach: pigs spawn on the ground and on the raised step to its right, and
    # the ones up there stay up there until hit, so luring them down can't work
    # (unhit pigs never follow). The bot farms whichever level has pigs instead:
    # climbs onto the step when its own level is clear and pigs wait up there,
    # walks back down when the step is clear and pigs are on the ground. Only for
    # 'jumper' enemy profiles (MobTracker.level_farming_enabled).
    LEVEL_SWITCH_CLEAR_SECONDS = 1.5   # own level free of pigs this long first
    LEVEL_ABORT_SECONDS = 0.3          # a pig on my level this recently: stay
    LEVEL_MIN_DWELL = 4.0              # after a level change, stay this long at least
    LEVEL_SETTLE_SECONDS = 0.3         # minimap on another level this long: it is
    LEVEL_APPROACH_SECONDS = 3.0       # not getting closer to the step this long: give up
    LEVEL_APPROACH_MAX = 25.0          # (or this long in all: from the far end of the map)
    LEVEL_STUCK_SECONDS = 0.5          # not moving for this long: pressed on the step
    LEVEL_STEP_NEAR = 0.02             # within this of the known step x (minimap)
    LEVEL_LEDGE_CONFIRM_SECONDS = 1.5  # pressed, no pig seen on the step: give up
    LEVEL_LAND_CLEAR_PX = 120          # a step pig this close past the edge...
    LEVEL_LAND_BEHIND_PX = 40          # ...(or this far back over it) is in the way
    LEVEL_LAND_WAIT_SECONDS = 2.0      # wait this long for it, then jump swinging
    LEVEL_AIR_SECONDS = 0.45           # keep holding toward the step while in the air
    LEVEL_AIR_ATTACK_AT = 0.2          # the swing on the way up, this far into the jump
    LEVEL_LAND_SECONDS = 0.75          # then judge the landing from the minimap
    LEVEL_HOLD_SECONDS = 0.5           # up there this long (not knocked back): climbed
    LEVEL_VERIFY_SECONDS = 2.5         # no fresh minimap fix by this long: missed
    LEVEL_CLIMB_ADVANCE = 0.008        # this far past the pressed spot, or...
    LEVEL_CLIMB_RISE = 0.02            # ...this far above it (minimap y): on the step
    LEVEL_CLIMB_ATTEMPTS = 3
    LEVEL_CLIMB_BAN = 20.0             # after that many misses, farm the ground only
    LEVEL_MISS_MEMORY = 10.0           # misses this recent count toward that limit
    LEVEL_DESCEND_SECONDS = 4.0        # walking off the edge longer than this: stop
    LEVEL_EDGE_MARGIN = 0.012          # on the step, a wall this far in from its edge
    STALL_SECONDS = 15.0               # pigs in view, nothing hit this long: reset
    # Gather (spawn_issue.mp4): the ground is clear and AGGRO_MIN_PIGS or more
    # wait up on the step. Fighting them one by one up there is slow and they
    # knock the character off. Instead climb up, walk toward the step's middle,
    # hit them with Q a few times (hit pigs chase the character), then walk
    # back down to the flat ground and fight them there as they follow.
    AGGRO_MIN_PIGS = 3
    AGGRO_WALK_SECONDS = 2.5           # walking toward the step's middle, at most
    AGGRO_DEPTH = 0.05                 # (how far in, when the map gives no middle)
    AGGRO_Q_PRESSES = 2                # in the middle, confirmed by an MP drop each
    AGGRO_Q_INTERVAL = 0.6
    AGGRO_Q_RETRIES = 2                # Qs that showed no MP drop, pressed again at most
    AGGRO_Q_MP_DROP = 0.5              # MP % a fired Q costs at least
    AGGRO_Q_CONFIRM_SECONDS = 0.5      # MP read this long after the press
    # Hit or held up on the way to the middle (pigs on the island attack): Q
    # where it stands so the pigs that hit it are aggroed, then carry on
    AGGRO_HIT_HP_DROP = 1.0            # HP % lost since the last tick = a hit
    AGGRO_STALL_SECONDS = 0.6          # not moving this long while walking = held up
    AGGRO_EMERGENCY_QS = 2
    AGGRO_EMERGENCY_GAP = 0.8
    AGGRO_RETRY_COOLDOWN = 8.0         # knocked off before any Q fired: try again soon
    # Climbing (14:30 log): a pig standing on the landing spot knocked the
    # character back off three times in a row. Q it from the foot of the step
    # first (it comes down after the character), then jump.
    LEVEL_FOOT_Q_WAIT = 1.2
    AGGRO_RETREAT_PAST = 0.03          # this far past the edge on the flat ground: done
    AGGRO_RETREAT_SECONDS = 6.0
    AGGRO_COOLDOWN = 20.0              # then fight what follows before the next gather
    MAP_LIMIT_DESCEND_REACH = 0.2      # a map limit this close: never walk off that edge
    # Patrolled the whole map this many times (end to end and back) without
    # seeing a single enemy: change channel (map_events.EMPTY_MAP)
    EMPTY_MAP_LAPS = 3
    # Pig Beach: on the island this long with no pig in reach counts as cleared
    # (its laps rule: the 'island_clear_laps' spot in map_events.PIG_BEACH)
    ISLAND_CLEAR_SECONDS = 3.0
    # Visible drops (LootTracker.ground_drops): walk over and pick them up.
    DROP_PICK_RADIUS = 18           # px: this close, it is underfoot
    DROP_DETOUR = 160               # px: worth stepping aside for while a pig is coming
    DROP_TAP_INTERVAL = 0.25
    DROP_MAX_TAPS = 6               # still there after this many taps: not ours
    DROP_STUCK_SECONDS = 0.8        # walking at a drop, not moving: a wall
    DROP_CHASE_SECONDS = 5.0
    DROP_BAN_SECONDS = 8.0
    # looting-strike-issue.mp4: the cut log by the tree matches a bronze coin.
    # The bot walked back to it every 8 s, tapped six times and turned its
    # back on a pig each time. A "drop" that survives its taps is scenery:
    # remembered where it stands (minimap x) for the rest of the visit.
    SCENERY_DROP_SECONDS = 600.0
    SCENERY_DROP_RADIUS = 0.012     # minimap: drops this close to it are it
    SCENERY_MAX = 12
    PX_PER_MINIMAP = 6000.0         # screen px per minimap unit, until measured
    # A drop is only worth a detour while an enemy approaches if the enemy
    # stays a swing away from it; the bot walked onto a coin lying at a
    # pig's feet and was hit before it swung (same clip, 5.9 s).
    DROP_SAFE_GAP = 130             # px between the drop and the enemy, at least
    DROP_BEHIND_GAP = 260           # ...and this far when the drop is behind us
    # The target vanished with no kill confirmed (a hit pose, an effect over
    # it): stand and keep tapping for a moment instead of walking off. The
    # bot took a knocked-back pig for dead, turned to the drops and was hit
    # from behind (same clip, 8.2 s). Jumper maps only.
    VANISH_HOLD_SECONDS = 0.45
    # Q fired twice 0.5 s apart on one pile (same clip): the second cast only
    # costs MP and a second of swings. After a Q that cost MP, no new one this
    # soon; one that cost nothing (pressed mid-swing, the game dropped it) may
    # be pressed again after GROUP_Q_RETRY.
    GROUP_Q_COOLDOWN = 1.1
    GROUP_Q_RETRY = 0.35
    GROUP_Q_SETTLE = 0.09           # let the swing end before Q, or it is dropped
    # 2026-11-17 11:58 log (Perion Street Corner): CTRL went up and D down in
    # the same instant, and 0.8 s later the bot called it "Blocked walking
    # right" on open ground: the last swing was still playing, and the
    # character can't walk until it ends. That false wall started the left,
    # right, left, right. A walk that starts this soon after the attack key
    # was let go is not judged for being stuck yet.
    SWING_LOCK_SECONDS = 0.9
    # ---- Sweep patrol (MobTracker 'Sweep patrol'; perion_street_corner.png) ----
    # Walk the map from one end to the other clearing what is ahead, turn only
    # where the walk really stops, and come back the other way.
    SWEEP_REVERSE_SECONDS = 0.4     # no about-turn sooner than this after walking
    SWEEP_DROP_BEHIND = 150         # px: drops further back than this are left
    # In reach, the attack key stays down like a player's. After 2-3 swings
    # (or at once when a hit knocked the enemy past the weapon) the arrow goes
    # down with it for a moment, to make up the knock-back.
    STEP_IN_AFTER = 1.8             # s of held attack between steps (2-3 swings)
    STEP_IN_NOW_PX = 68             # further than this: out of reach, step now
    STEP_IN_TO_PX = 54              # a step ends about this far from the enemy
    STEP_IN_MIN = 0.06              # s the arrow is held, at least...
    STEP_IN_MAX = 0.35              # ...and at most
    STEP_IN_GAP = 0.5               # s between two steps, at least
    STEP_IN_COMING = 15             # px/s: it is walking in by itself, no step
    WALK_SPEED_PX = 125             # px/s
    # ---- Facing (src/detection/facing.py; attack1.mp4) ----
    WALK_NOTE_AFTER = 0.35          # s the arrow is held before the walk teaches the reader
    TURN_FIX_GAP = 0.6              # s between two corrections of the facing
    TURN_FIX_MIN = 0.12             # s the arrow is held for a correction, at least...
    TURN_FIX_MAX = 0.45             # ...and at most (until the head is seen turned)
    SURROUNDINGS_LOG_GAP = 1.0      # s between two "facing / enemies" log lines
    SURROUNDINGS_LOG_EVERY = 5.0    # ...and one this often while enemies are about
    # attack2.mp4: by the right-edge portal the walk flipped every 0.5 s
    # between the target on the left and a "drop" on the right (decor past
    # the map's edge). A drop on the other side from the last target waits
    # this long after the target was last seen...
    DROP_AFTER_TARGET = 1.5
    # ...and a drop that the walk can't reach twice running is scenery.
    DROP_WALL_MEMORY = 60.0
    # ---- Edge loot jumps (MobTracker 'Edge loot jumps'; msg-boc.png) ----
    # Henesys snail map: drops land up on the hay bales at the map's ends and
    # the patrol stopped at the foot of each bale and turned. With the option
    # on, where the walk stops it jumps up toward that end a time or two,
    # tapping loot, then turns as before.
    EDGE_LOOT_JUMPS = (1, 2)        # jumps per visit (random, inclusive)
    EDGE_LOOT_GAP = (0.6, 0.9)      # s between two jumps
    EDGE_LOOT_TAPS = 4              # loot taps after each jump
    EDGE_LOOT_COOLDOWN = 20.0       # s before the same end gets jumped at again
    EDGE_LOOT_MAX_SECONDS = 5.0
    # ---- Idle variety (MobTracker 'Idle variety') ----
    # Patrol with nothing in sight: a hop now and then (sometimes two), a
    # short stop now and then, and legs of different lengths.
    IDLE_HOP_EVERY = (3.0, 9.0)     # s between hops
    IDLE_DOUBLE_HOP = 0.25          # chance of a second hop right after
    IDLE_HOP_WALL_GAP = 0.03        # minimap: no hop this close to a wall ahead
    IDLE_PAUSE_EVERY = (10.0, 25.0) # s between stops
    IDLE_PAUSE_FOR = (0.4, 1.8)     # s a stop lasts
    IDLE_LEG = (8.0, 12.0)          # s a timed patrol leg lasts
    # Long quiet walk (no enemy for IDLE_QUIET_AFTER s): more jumping, with a
    # mix of single taps, quick doubles and the jump key held (several
    # bounces in a row), plus some plain walking stretches with no jumps.
    IDLE_QUIET_AFTER = 15.0         # s with no enemy before the walk gets jumpier
    IDLE_QUIET_HOP_EVERY = (1.2, 4.0)
    IDLE_QUIET_DOUBLE_HOP = 0.30
    IDLE_QUIET_HELD_JUMP = 0.20     # chance a hop is the jump key held down
    IDLE_QUIET_HOLD_FOR = (0.5, 1.3)  # s the jump key is held (2-3 bounces)
    IDLE_QUIET_HELD_WALL_GAP = 0.08 # held jumps travel further: need more room
    IDLE_QUIET_CALM = 0.15          # chance of a jump-free walk after a hop...
    IDLE_QUIET_CALM_FOR = (5.0, 10.0)  # ...this long
    DEFAULT_CONFIG = {
        'Interact': 'y',
        'Feed pet': '9'
    }

    def __init__(self):
        """Loads a user-defined routine on start up and initializes this Bot's main thread."""

        super().__init__('keybindings')
        config.bot = self

        self.rune_active = False
        self.rune_pos = (0, 0)
        self.rune_closest_pos = (0, 0)      # Location of the Point closest to rune
        self.submodules = []
        self.command_book = None            # CommandBook instance
        # self.module_name = None
        # self.buff = components.Buff()

        # self.command_book = {}
        # for c in (components.Wait, components.Walk, components.Fall,
        #           components.Move, components.Adjust, components.Buff):
        #     self.command_book[c.__name__.lower()] = c

        config.routine = Routine()

        self.ready = False
        self._auto_attack_movement = None
        self._patrol_direction = None
        self._patrol_until = 0
        self._mob_attack_keys = ()
        self._walls = {}                # 'left'/'right' -> minimap x where walking stops
        self._pursuit_watch = None      # (direction, x, last moved time)
        self._blocked_side = None
        # Climbs back to the patrol platform after being knocked off it
        self.recovery = LevelRecovery(
            key_down, key_up, press,
            position=lambda: config.player_pos,
            active=lambda: config.enabled,
        )
        self._home_key = None
        # Map events (e.g. the Iron Hog on Pig Beach): see src/modules/map_events.py
        self.map_events = MapEventManager(self)
        self.thread = threading.Thread(target=self._main, name='bot')
        self.thread.daemon = True

    def start(self):
        """
        Starts this Bot object's thread.
        :return:    None
        """

        self.update_submodules()
        print('\n[~] Started main bot loop')
        self.thread.start()
        self.map_events.start()
        from src.modules.memory_guard import memory_guard
        memory_guard.start()
        from src.modules.chat_watch import chat_watch
        chat_watch.start()

    def _main(self):
        """
        The main body of Bot that executes the user's routine.
        :return:    None
        """

        print('\n[~] Initializing detection algorithm:\n')
        try:
            model = detection.load_model()
        except FileNotFoundError as error:
            model = None
            print(f'[!] Rune solver disabled: {error}')
            print('[!] The routine will pause when a rune is detected.')
        else:
            print('\n[~] Initialized detection algorithm')

        self.ready = True
        config.listener.enabled = True
        last_fed = time.time()
        from src.modules.watchdog import watchdog
        watchdog.register('bot', self.THREAD_STALL_SECONDS)
        while True:
            watchdog.heartbeat('bot')
            if config.enabled and len(config.routine) == 0:
                self._guarded_tick()
            elif config.enabled and len(config.routine) > 0:
                # Buff and feed pets
                self.command_book.buff.main()
                pet_settings = config.gui.settings.pets
                auto_feed = pet_settings.auto_feed.get()
                num_pets = pet_settings.num_pets.get()
                now = time.time()
                if auto_feed and now - last_fed > 1200 / num_pets:
                    press(self.config['Feed pet'], 1)
                    last_fed = now

                self._set_patrol_home()
                if self._service_map_events():
                    time.sleep(0.02)
                    continue
                if self._recover_if_fallen():
                    continue
                self._service_loot_pickup()
                if self._service_mob_attack():
                    continue

                # Highlight the current Point
                config.gui.view.routine.select(config.routine.index)
                config.gui.view.details.display_info(config.routine.index)

                # Execute next Point in the routine
                element = config.routine[config.routine.index]
                if self.rune_active and isinstance(element, Point) \
                        and element.location == self.rune_closest_pos:
                    if model is None:
                        config.enabled = False
                        print(
                            '\n[!] Rune detected. Handle it manually, then '
                            'resume the routine with the Start/stop key.'
                        )
                        continue
                    self._solve_rune(model)
                element.execute()
                config.routine.step()
            else:
                self._cancel_map_events()
                self._set_auto_attack_movement(None)
                self._release_mob_attack()
                self._forget_patrol_home()
                self._forget_walls()
                time.sleep(0.01)

    THREAD_STALL_SECONDS = 30       # watchdog: a tick this long is a hang (stacks dumped)
    TICK_ERROR_LIMIT = 5            # this many errors within a minute: pause

    def _guarded_tick(self):
        """
        One farming tick. An unexpected error used to kill the bot thread (the
        bot then sat there doing nothing): log it, let go of the keys, carry
        on; only repeated errors pause Auto Maple.
        """
        try:
            self._run_standalone_mob_attack()
        except Exception:
            now = time.monotonic()
            errors = [t for t in self.__dict__.get('_tick_errors', []) if now - t < 60] + [now]
            self._tick_errors = errors
            print('\n[!] Error in the farming loop (carrying on):')
            traceback.print_exc()
            try:
                self._set_auto_attack_movement(None)
                self._release_mob_attack()
                self.__dict__.pop('_level_op', None)
                self._drop_chase = None
            except Exception:
                pass
            if len(errors) >= self.TICK_ERROR_LIMIT:
                config.enabled = False
                print(f'\n[!] {len(errors)} errors within a minute; Auto Maple is paused. '
                      'Please send the log.')
            time.sleep(0.2)

    def _set_patrol_home(self):
        """
        Tells the recovery which platform the patrol belongs on: the level of the
        routine's Points, or, without a routine, wherever the character stands
        when Auto Maple is switched on.
        """

        recovery = getattr(self, 'recovery', None)
        if recovery is None:
            return
        points = tuple(
            element.location for element in config.routine.sequence
            if isinstance(element, Point)
        ) if len(config.routine) > 0 else ()
        key = points or 'standalone'
        if key == self._home_key:
            return
        if points:
            heights = sorted(y for _, y in points)
            if heights[-1] - heights[0] > recovery.FALL_TOLERANCE:
                recovery.clear_home()       # the routine itself changes levels
            else:
                xs = [x for x, _ in points]
                recovery.set_home(heights[len(heights) // 2], min(xs), max(xs))
        else:
            position = config.player_pos
            if not position or tuple(position) == (0, 0):
                return                      # minimap position not known yet
            recovery.set_home(position[1])
        self._home_key = key

    def _forget_patrol_home(self):
        if getattr(self, '_home_key', None) is not None:
            self._home_key = None
            self.recovery.clear_home()

    def _recover_if_fallen(self):
        """Climbs back up if the character was knocked off the patrol platform."""

        recovery = getattr(self, 'recovery', None)
        if not FALL_RECOVERY or recovery is None or recovery.home_y is None:
            return False
        position = config.player_pos
        recovery.observe(position)
        if not recovery.fallen(position):
            return False
        if not is_game_focused():
            return False
        self._set_auto_attack_movement(None)
        self._release_mob_attack()
        keys = getattr(getattr(self.command_book, 'module', None), 'Key', None)
        recovery.jump_key = getattr(keys, 'JUMP', None) or 'alt'
        recovery.recover()
        self._patrol_direction = None
        return True

    def _run_standalone_mob_attack(self):
        tracker = config.capture.mob_tracker
        if not tracker.settings()['Enabled']:
            self._set_auto_attack_movement(None)
            self._release_mob_attack()
            time.sleep(0.05)
            return
        if not is_game_focused():
            self._set_auto_attack_movement(None)
            self._release_mob_attack()
            time.sleep(0.05)
            return

        self._expire_step_in()
        if self._service_map_events():
            time.sleep(0.02)
            return
        self._set_patrol_home()
        if self._recover_if_fallen():
            return
        self._service_loot_pickup()
        x = self._minimap_x()
        self._check_walls(x)
        target = tracker.target()
        if target is not None:
            self._last_target = ('left' if target[0] < 0 else 'right', time.monotonic())
        self._note_walk(tracker, x)
        self._log_surroundings(tracker)
        self._count_empty_laps(tracker, target)
        self._service_smart_loot(tracker, target, x)
        if self._service_levels(tracker, target, x):
            time.sleep(0.02)
            return
        if self._service_missile(tracker, target, x):
            time.sleep(0.02)
            return
        waiting = getattr(tracker, 'waiting', lambda: False)()
        if target is not None and waiting:
            # A pig mid-hop above the swing: stand still and swing when it lands
            # (it may come down on the other side). A snail or shroom coming at
            # the character: stand facing it, so the first swing needs no turn.
            self._pursuit_watch = None
            self._release_mob_attack()
            self._set_auto_attack_movement(None)
            if not self._jumper_farming(tracker):
                self._face(target[0])
        elif target is not None and not target[1] and self._collect_visible_drop(x, target):
            self._pursuit_watch = None      # a drop on the way: grab it first
        elif target is not None:
            offset_x, attackable = target
            if attackable:
                self._pursuit_watch = None
                self._set_auto_attack_movement(None)
                direction = 'left' if offset_x < 0 else 'right'
                stationary = self._wall_ahead(direction, x)
                self._service_mob_attack(stationary=stationary)
            elif self._follow_up_swing(tracker):
                self._pursuit_watch = None  # a swing just went out: one more, then move
            else:
                direction = 'left' if offset_x < 0 else 'right'
                self._release_mob_attack()
                if abs(offset_x) <= self.PURSUIT_DEADBAND:
                    # Right above/below the character (on a step, mid-hop): stepping
                    # at it would just swap left/right every scan
                    self._pursuit_watch = None
                    self._set_auto_attack_movement(None)
                elif self._pursuit_blocked(direction, x):
                    self._turn_back_from_wall(direction, x)
                else:
                    self._set_auto_attack_movement(direction)
        elif self._follow_up_swing(tracker):
            # Lost the lock right after a swing (knocked back out of view, or
            # under the character): keep swinging that way for a moment
            # instead of walking off into it (strike-issue.mp4)
            self._pursuit_watch = None
        else:
            self._pursuit_watch = None
            self._service_mob_attack()
            if time.monotonic() < self.__dict__.get('_vanish_hold_until', float('-inf')):
                # Lost sight of it mid-fight, no kill seen: it is likely still
                # there (and if it did die, its drops are landing right here)
                self._set_auto_attack_movement(None)
            elif not self._collect_visible_drop(x) and not self._collect_loot(x):
                self._patrol_for_mob()
        time.sleep(0.02)

    # ---- Map events: scripted responses (map_events.py) ----

    def _service_map_events(self):
        """True while a map event's plan (e.g. the Iron Hog escape) has the controls."""
        manager = self.__dict__.get('map_events')
        return manager is not None and manager.service()

    def _cancel_map_events(self):
        manager = self.__dict__.get('map_events')
        if manager is not None:
            manager.cancel()

    def _count_empty_laps(self, tracker, target):
        """Any enemy in sight (here, up on the step or down below) restarts the count."""
        if (target is not None or self._sighting(tracker, 'ledge_enemies')[0]
                or self._sighting(tracker, 'lower_enemies')[0]):
            self._empty_turns = 0
            self._ground_empty_turns = 0

    def _empty_turn(self):
        """The patrol turned at an end of the map: half a lap with nothing seen."""
        turns = self.__dict__.get('_empty_turns', 0) + 1
        self._empty_turns = turns
        manager = self.__dict__.get('map_events')
        # A map can ask for fewer laps ('empty_map_laps'; Pig Beach: 1, user 2026-11-20)
        empty_laps = self._map_spot('empty_map_laps') or self.EMPTY_MAP_LAPS
        if turns >= 2 * empty_laps:
            self._empty_turns = 0
            self._ground_empty_turns = 0
            if manager is not None:
                manager.request_empty_map(empty_laps)
            return
        # Pig Beach (user 2026-11-19): the pig island was cleared, then this many
        # laps of the flat ground with no pigs: change channel and farm on there
        laps = self._map_spot('island_clear_laps')
        if not laps or not self.__dict__.get('_island_cleared') \
                or self.__dict__.get('_level', 'ground') != 'ground':
            self._ground_empty_turns = 0
            return
        ground_turns = self.__dict__.get('_ground_empty_turns', 0) + 1
        self._ground_empty_turns = ground_turns
        if ground_turns % 2 == 0:
            print(f'\n[~] Pig island cleared; flat ground lap {ground_turns // 2}/{laps} '
                  'with no pigs.')
        if ground_turns >= 2 * laps:
            self._empty_turns = 0
            self._ground_empty_turns = 0
            if manager is not None:
                manager.request_empty_map(
                    laps, f'Pig island cleared and no pigs after {laps} laps of the '
                          'flat ground.')

    def _track_island(self, tracker, target, own_clear, now):
        """
        Pig Beach: whether the pig island (the raised step) has been cleared.
        Cleared: up there with no pig in reach for ISLAND_CLEAR_SECONDS. Not
        cleared again as soon as a pig is seen up there (they respawn).
        """
        if not self._map_spot('island_clear_laps'):
            return
        level = self.__dict__.get('_level', 'ground')
        if self.__dict__.get('_island_level') != level:
            self._island_level, self._island_level_since = level, now
        if level == 'step':
            if target is not None or own_clear < self.LEVEL_SWITCH_CLEAR_SECONDS:
                self._island_cleared = False
            elif not self.__dict__.get('_island_cleared') \
                    and now - self._island_level_since >= self.ISLAND_CLEAR_SECONDS:
                self._island_cleared = True
                self._ground_empty_turns = 0
                print('\n[~] Pig island cleared.')
        elif self._sighting(tracker, 'ledge_enemies')[0]:
            if self.__dict__.get('_island_cleared'):
                print('\n[~] Pigs back on the pig island.')
            self._island_cleared = False
            self._ground_empty_turns = 0

    def _prepare_for_plan(self):
        """A plan takes over: let go of everything farming was doing."""
        self._release_mob_attack()
        self._set_auto_attack_movement(None)
        self._end_level_op(None)
        self._pursuit_watch = None

    def _resume_after_plan(self, new_channel=True):
        """
        Back to farming after a plan. The character is on the ground (it walked
        off the spawn cliff / back off the pig island). After a channel change
        the drops and fight spots from before are gone too.
        """
        self._switch_level('ground')
        self._level_pending = None
        # Walls learnt out of bounds (e.g. in the Pig Beach pit) mean nothing here
        for walls in [self.__dict__.setdefault('_walls', {})] + list(
                self.__dict__.get('_level_walls', {}).values()):
            for side, wall in list(walls.items()):
                right, left = self._map_limit('right'), self._map_limit('left')
                if (right is not None and wall > right) or (left is not None and wall < left):
                    del walls[side]
        self._set_blocked_side(None)
        if new_channel:
            self._loot_planner = None
        self._loot_was_attacking = False
        self._patrol_direction = None
        self._patrol_seen = 0
        self._pursuit_watch = None
        self._stall_since = None
        self._empty_since = None
        self._empty_turns = 0
        self._ground_empty_turns = 0
        if new_channel:
            self._island_cleared = False    # a new channel's island is unchecked

    # ---- Two-level farming: the ground and a raised step (Pig Beach) ----

    def _service_levels(self, tracker, target, x):
        """
        Farms whichever level has pigs. True while a climb/descent has the
        controls this tick (the caller does nothing else), False to fight or
        patrol as usual on the current level. Only for 'jumper' enemy profiles.

        climb:   walk until pressed against the step with pigs above, wait for
                 the landing spot to clear (or swing in the air), jump, and count
                 it only once the character has stayed up there (a knock-back
                 off the edge is a miss); retried, then banned for a while.
        descend: walk off the step's edge until the minimap shows the ground.
        """
        enabled = getattr(tracker, 'level_farming_enabled', None)
        if enabled is None or not enabled():
            return False
        now = time.monotonic()
        y = self._minimap_y(x)
        own_clear = self._own_level_clear(tracker, target, now)
        op = self.__dict__.get('_level_op')
        if op is None:
            self._track_level(x, y, now)
            self._track_island(tracker, target, own_clear, now)
        self._check_stall(tracker, target, now)
        op = self.__dict__.get('_level_op')
        if op is None:
            op = self._choose_level_op(tracker, target, x, y, now, own_clear)
            if op is None:
                return False
        handler = getattr(self, '_level_' + op['phase'])
        return handler(op, tracker, target, x, y, now, own_clear)

    # -- what the bot knows about the levels --

    def _minimap_y(self, x):
        """The character's minimap y, when its x is freshly known."""
        position = getattr(config, 'player_pos', None)
        return position[1] if x is not None and position else None

    def _own_level_clear(self, tracker, target, now):
        """
        Seconds since a pig was last in reach on the character's own level. Uses
        the tracker's count (it sees pigs even with no target picked yet) when it
        has one, else how long there has been no target. A target means 0.
        """
        if target is not None:
            self._empty_since = None
            return 0.0
        if self.__dict__.get('_empty_since') is None:
            self._empty_since = now
        reported = getattr(tracker, 'ground_clear_seconds', None)
        if reported is not None:
            try:
                return float(reported())
            except (TypeError, ValueError):
                pass
        return now - self._empty_since

    @staticmethod
    def _sighting(tracker, name):
        """(count, screen x offset of the nearest) from a tracker helper, or (0, None)."""
        helper = getattr(tracker, name, None)
        if helper is None:
            return 0, None
        try:
            count, offset = helper()
        except (TypeError, ValueError):
            return 0, None
        return count or 0, offset

    def _level_of(self, x, y):
        """
        'step' or 'ground' from the minimap, once a climb has shown where the
        step is (its edge x and the ground's y), else None.
        """
        step = self.__dict__.get('_step')
        if step is None or x is None:
            return None
        sign = 1 if step['direction'] == 'right' else -1
        past = sign * (x - step['edge_x'])
        rise = step['ground_y'] - y if y is not None and step['ground_y'] is not None else 0.0
        if rise >= self.LEVEL_CLIMB_RISE or past >= self.LEVEL_CLIMB_ADVANCE:
            return 'step'
        return 'ground'

    def _track_level(self, x, y, now):
        """Notices the character leaving the step without a descent (knocked or
        walked off the edge) and switches to the ground's walls."""
        level = self._level_of(x, y)
        current = self.__dict__.get('_level', 'ground')
        if level is None or level == current:
            self._level_pending = None
            return
        pending = self.__dict__.get('_level_pending')
        if pending is None or pending[0] != level:
            self._level_pending = (level, now)
            return
        if now - pending[1] >= self.LEVEL_SETTLE_SECONDS:
            self._switch_level(
                level, 'Off the step; back on the ground' if level == 'ground'
                else 'Up on the step')

    def _switch_level(self, level, reason=None):
        """Each level keeps its own walls (the step's foot is a wall down here,
        nothing up there): swap them over."""
        current = self.__dict__.get('_level', 'ground')
        if level == current:
            return
        saved = self.__dict__.setdefault('_level_walls', {})
        op = self.__dict__.get('_level_op')
        if not (op and op['kind'] == 'climb'):
            # (a climb already put the ground's walls away when it started)
            saved[current] = dict(self.__dict__.get('_walls', {}))
        self._walls = dict(saved.get(level, {}))
        self._level = level
        self._level_pending = None
        self._pursuit_watch = None
        self._patrol_seen = 0
        self._set_blocked_side(None)
        if reason:
            print(f'\n[~] {reason}.')

    def _check_stall(self, tracker, target, now):
        """
        Farming must not stall: pigs in view (on any level) but nothing swung at
        for STALL_SECONDS means the bot is stuck on a wrong belief (a stale
        wall, a climb ban, a bad level choice). Forget them and choose afresh.
        """
        engaged = target is not None and target[1]
        op = self.__dict__.get('_level_op')
        if op is not None and now - op.get('moved', now) < self.LEVEL_APPROACH_SECONDS:
            engaged = True      # a climb/gather under way and getting somewhere
        visible = (
            target is not None
            or self._sighting(tracker, 'ledge_enemies')[0]
            or self._sighting(tracker, 'lower_enemies')[0]
        )
        if engaged or not visible:
            self._stall_since = None
            return
        since = self.__dict__.get('_stall_since')
        if since is None:
            self._stall_since = now
            return
        if now - since < self.STALL_SECONDS:
            return
        self._stall_since = now
        print(
            f'\n[!] Pigs in view but nothing hit for {self.STALL_SECONDS:.0f}s; '
            'forgetting walls and climb bans and choosing again.'
        )
        if self.__dict__.get('_level_op') is not None:
            self._end_level_op(None)
        self._walls = {}
        self._level_walls = {}
        self._climb_ban_until = float('-inf')
        self._last_switch = float('-inf')
        self._pursuit_watch = None
        self._set_blocked_side(None)

    def _choose_level_op(self, tracker, target, x, y, now, own_clear):
        """Starts a climb or a descent when this level is clear and another isn't."""
        if target is not None or own_clear < self.LEVEL_SWITCH_CLEAR_SECONDS:
            return None
        if now - self.__dict__.get('_last_switch', float('-inf')) < self.LEVEL_MIN_DWELL:
            return None
        up, up_offset = self._sighting(tracker, 'ledge_enemies')
        down, down_offset = self._sighting(tracker, 'lower_enemies')
        can_climb = (
            up >= 1 and up_offset is not None
            and now >= self.__dict__.get('_climb_ban_until', float('-inf'))
        )
        can_descend = down >= 1 and down_offset is not None
        if can_climb and (not can_descend or up >= down):
            op = self._start_climb(up, up_offset, x, now, own_clear)
            if up >= self.AGGRO_MIN_PIGS and self._group_skill() and \
                    now >= self.__dict__.get('_aggro_until', float('-inf')):
                op['mode'] = 'aggro'
                print(f'[~] {up} pigs up there: going up to hit them with Q and '
                      'lead them down to the flat ground.')
            return op
        if can_descend:
            return self._start_descend(down, down_offset, x, y, now)
        if (not can_climb and self.__dict__.get('_level') == 'step'
                and self.__dict__.get('_step') is not None
                and self.__dict__.get('_island_cleared')
                and self._map_spot('island_clear_laps')):
            # Pig Beach: the island is clear and no pigs show below: go down and
            # lap the flat ground (then change channel if it stays empty)
            return self._start_descend(0, 0, x, y, now)
        return None

    # -- climbing up --

    def _start_climb(self, count, offset, x, now, own_clear):
        direction = 'left' if offset < 0 else 'right'
        saved = self.__dict__.setdefault('_level_walls', {})
        walls = dict(self.__dict__.get('_walls', {}))
        saved['ground'] = walls
        step = self.__dict__.get('_step')
        step_x = walls.get(direction)
        if step_x is None and step is not None and step['direction'] == direction:
            step_x = step['edge_x']
        if step_x is None:
            step_x = self._map_spot('island_left' if direction == 'right' else 'island_right')
        op = self._level_op = {
            'kind': 'climb', 'phase': 'approach', 'direction': direction,
            'start': now, 'x': x, 'moved': now, 'step_x': step_x,
            # Misses carry over when a climb restarts soon after (a knock-back
            # makes the pig that did it count as "on this level" for a moment)
            'attempts': self._recent_climb_misses(now),
        }
        self._release_mob_attack()
        self._walls = {}            # walk into the step this time
        self._pursuit_watch = None
        self._set_blocked_side(None)
        clear_for = 'yet' if own_clear == float('inf') else f'for {own_clear:.1f}s'
        print(
            f'\n[~] No pigs on this level {clear_for}; {count} up on the step to '
            f'the {direction}. Climbing up to farm there.'
        )
        return op

    def _level_approach(self, op, tracker, target, x, y, now, own_clear):
        """Walks at the step until pressed against it, pigs above and the landing
        spot clear, then jumps."""
        if own_clear < self.LEVEL_ABORT_SECONDS:
            self._end_level_op('a pig came onto this level; fighting it first')
            return False
        direction = op['direction']
        sign = 1 if direction == 'right' else -1
        if x is None:
            op['moved'] = now   # position lost/lagging: can't tell stuck from walking
        elif op['x'] is None or abs(x - op['x']) > self.PATROL_MOVE_EPSILON:
            op['x'], op['moved'] = x, now
        still = now - op['moved']
        step_x = op['step_x']
        pressed = (
            x is not None and still >= self.LEVEL_STUCK_SECONDS
            and (step_x is None or abs(x - step_x) <= self.LEVEL_STEP_NEAR)
        )
        if not pressed:
            # Walking from the far side of the map takes a while: only stuck
            # (not moving, away from the step) or far too long gives up
            if still >= self.LEVEL_APPROACH_SECONDS or \
                    now - op['start'] >= self.LEVEL_APPROACH_MAX:
                self._end_level_op('could not reach the step', ban=True)
                return False
            self._release_mob_attack()
            self._set_auto_attack_movement(direction)
            return True
        self._release_mob_attack()
        count, offset = self._sighting(tracker, 'ledge_enemies')
        if count < 1 or offset is None:
            if still >= self.LEVEL_STUCK_SECONDS + self.LEVEL_LEDGE_CONFIRM_SECONDS:
                self._end_level_op('at the step, but no pigs are up there any more')
                return False
            self._set_auto_attack_movement(None)    # wait here for a step scan
            return True
        ahead = sign * offset
        op['air_attack'] = False
        if -self.LEVEL_LAND_BEHIND_PX <= ahead <= self.LEVEL_LAND_CLEAR_PX:
            # A pig right at the edge would knock the character straight back off
            if op.get('crowded_since') is None:
                op['crowded_since'] = now
                print(f'\n[~] A pig is on the landing spot ({ahead:.0f}px in); '
                      'waiting for it to move.')
            if now - op['crowded_since'] < self.LEVEL_LAND_WAIT_SECONDS:
                self._set_auto_attack_movement(None)
                return True
            if not op.get('foot_q_at') and self._group_skill():
                # It stayed: hit it with Q from down here; a hit pig follows
                # the character down, and normal targeting takes it from there
                print('[~] It stayed; using Q on it from the foot of the step.')
                self._press_q(direction)
                op['foot_q_at'] = now
                return True
            if op.get('foot_q_at') and now - op['foot_q_at'] < self.LEVEL_FOOT_Q_WAIT:
                self._set_auto_attack_movement(None)
                return True
            op['air_attack'] = True     # still there: swing on the way up
        return self._level_jump(op, x, y)

    def _level_jump(self, op, x, y):
        module = getattr(self.command_book, 'module', None)
        keys = getattr(module, 'Key', None)
        jump_key = getattr(keys, 'JUMP', None) or 'alt'
        op['attempts'] += 1
        op.update(
            phase='climb', pressed_x=x, pressed_y=y, jump_time=time.monotonic(),
            air_attacked=False, up_since=None, crowded_since=None,
        )
        self._jump_toward(op['direction'], jump_key)     # hold toward the step
        print(
            f'\n[~] Pressed against the step at x={x:.3f}; jumping up '
            f'(try {op["attempts"]}/{self.LEVEL_CLIMB_ATTEMPTS}'
            f'{", swinging in the air" if op["air_attack"] else ""}).'
        )
        return True

    def _climb_progress(self, op, x, y):
        """(advance past the pressed spot, rise above it) in minimap units."""
        sign = 1 if op['direction'] == 'right' else -1
        advance = sign * (x - op['pressed_x'])
        rise = op['pressed_y'] - y if y is not None and op['pressed_y'] is not None else 0.0
        return advance, rise

    def _level_climb(self, op, tracker, target, x, y, now, own_clear):
        """Through the jump, then confirms on the minimap that it stays up there."""
        elapsed = now - op['jump_time']
        if op['air_attack'] and not op['air_attacked'] \
                and elapsed >= self.LEVEL_AIR_ATTACK_AT:
            attack_key = getattr(
                getattr(getattr(self.command_book, 'module', None), 'Key', None),
                'CRUEL_STAB', None)
            if attack_key:
                press(attack_key, 1, down_time=0.08, up_time=0.05)
            op['air_attacked'] = True
        if elapsed < self.LEVEL_AIR_SECONDS:
            self._release_mob_attack()
            self._set_auto_attack_movement(op['direction'])
            return True
        self._set_auto_attack_movement(None)
        # Up there next to a pig: hit it rather than stand and take the touch
        if target is not None and target[1]:
            self._service_mob_attack()
        else:
            self._release_mob_attack()
        if elapsed < self.LEVEL_LAND_SECONDS:
            return True
        seen = getattr(config, 'player_pos_time', None)
        fresh = x is not None and (
            not seen or seen >= op['jump_time'] + self.LEVEL_LAND_SECONDS - 0.05)
        if fresh:
            advance, rise = self._climb_progress(op, x, y)
            if advance >= self.LEVEL_CLIMB_ADVANCE or rise >= self.LEVEL_CLIMB_RISE:
                if op['up_since'] is None:
                    op['up_since'] = now
                if now - op['up_since'] >= self.LEVEL_HOLD_SECONDS:
                    result = self._climb_succeeded(op, advance, rise, now)
                    if op.get('mode') == 'aggro':
                        return self._start_aggro(op, x, now)
                    return result
                return True
            reason = ('knocked back off the edge' if op['up_since'] is not None
                      else 'did not land on the step')
            return self._climb_missed(op, x, now, reason)
        if elapsed >= self.LEVEL_VERIFY_SECONDS:
            return self._climb_missed(op, x, now, 'no minimap fix after the jump')
        return True

    def _climb_succeeded(self, op, advance, rise, now):
        self._step = {
            'direction': op['direction'], 'edge_x': op['pressed_x'],
            'ground_y': op['pressed_y'],
        }
        # The step's foot is a wall for chasing on the ground
        self.__dict__.setdefault('_level_walls', {}).setdefault(
            'ground', {}).setdefault(op['direction'], op['pressed_x'])
        self._switch_level('step')
        # Up here the edge is a wall: patrolling an empty step turns back at it
        # instead of dropping off (only a descent walks down, on purpose)
        sign = 1 if op['direction'] == 'right' else -1
        edge_side = 'left' if sign > 0 else 'right'
        self._walls.setdefault(edge_side, op['pressed_x'] + sign * self.LEVEL_EDGE_MARGIN)
        self._last_switch = now
        self._level_op = None
        self._climb_misses = None
        self._pursuit_watch = None
        print(f'\n[~] Up on the step (advance {advance:.3f}, rise {rise:.3f}); '
              'farming up here.')
        return False    # fight up there as usual from this tick on

    # -- gathering the step's pigs down to the flat ground --

    def _map_spot(self, name):
        manager = self.__dict__.get('map_events')
        profile = getattr(getattr(manager, 'watcher', None), 'map', None)
        return (getattr(profile, 'spots', None) or {}).get(name)

    def _start_aggro(self, op, x, now):
        sign = 1 if op['direction'] == 'right' else -1
        left, right = self._map_spot('island_left'), self._map_spot('island_right')
        if left is not None and right is not None:
            middle = (left + right) / 2
        else:
            middle = op['pressed_x'] + sign * self.AGGRO_DEPTH
        op.update(phase='aggro_walk', middle=middle, phase_start=now,
                  q_left=self.AGGRO_Q_PRESSES, next_q=now, q_facing=op['direction'],
                  q_fired=0, q_retries=0, q_pending=None, emergency_qs=0,
                  last_emergency=float('-inf'), hp=self._percent('health'),
                  walk_x=x, walk_moved=now)
        self._level_op = op
        print(f'[~] Up on the step: walking to its middle (x={middle:.3f}) to gather the pigs.')
        return True

    def _percent(self, monitor):
        """HP or MP percent from the capture's monitors, or None."""
        try:
            value = getattr(config.capture, monitor).current_percent()
        except Exception:
            return None
        return None if value is None else float(value)

    def _group_skill(self):
        """The character has a group skill (Q) - the GUI's 'Use Q' box."""
        check = getattr(getattr(config.capture, 'mob_tracker', None),
                        'group_skill_enabled', None)
        try:
            return check is None or bool(check())
        except Exception:
            return True

    def _press_q(self, facing):
        if not self._group_skill():
            return False                # no Q on this character: never press it
        self._release_mob_attack()
        self._set_auto_attack_movement(None)
        if facing:
            key_down(facing)
            time.sleep(0.06)
            key_up(facing)
        press('q', 1, down_time=0.08, up_time=0.1)
        self._attack_released_at = time.monotonic()
        return True

    def _gather_hit(self, op, x, now):
        """(hit, stalled): HP fell since the last tick / not moving while walking
        (a pig in the way, a knock-back)."""
        hp = self._percent('health')
        hit = (hp is not None and op.get('hp') is not None
               and op['hp'] - hp >= self.AGGRO_HIT_HP_DROP)
        if hp is not None:
            op['hp'] = hp
        if x is not None and (op.get('walk_x') is None
                              or abs(x - op['walk_x']) > self.PATROL_MOVE_EPSILON):
            op['walk_x'], op['walk_moved'] = x, now
        stalled = now - op.get('walk_moved', now) >= self.AGGRO_STALL_SECONDS
        return hit, stalled

    def _level_aggro_walk(self, op, tracker, target, x, y, now, own_clear):
        """Toward the step's middle. Hit or held up on the way: Q right there
        (the pigs doing it get aggroed), then keep going."""
        if self._knocked_off_gather(op, x, y, now):
            return False
        sign = 1 if op['direction'] == 'right' else -1
        there = x is not None and sign * (x - op['middle']) >= -0.005
        blocked = x is not None and self._wall_ahead(op['direction'], x) and \
            self._map_limit(op['direction']) is not None
        if there or blocked or now - op['phase_start'] >= self.AGGRO_WALK_SECONDS:
            self._set_auto_attack_movement(None)
            op.update(phase='aggro_q', phase_start=now, next_q=now)
            print(f'[~] In the middle of the step; {op["q_left"]} Qs to gather the pigs.')
            return True
        hit, stalled = self._gather_hit(op, x, now)
        if (hit or stalled) and op['emergency_qs'] < self.AGGRO_EMERGENCY_QS \
                and now - op['last_emergency'] >= self.AGGRO_EMERGENCY_GAP:
            facing = op['direction']
            if target is not None:
                facing = 'left' if target[0] < 0 else 'right'
            print(f'[~] {"Hit" if hit else "Held up"} on the way to the middle; '
                  f'Q facing {facing} right here.')
            self._press_q(facing)
            op['emergency_qs'] += 1
            op['last_emergency'] = now
            op['q_fired'] += 1          # those pigs are aggroed too
            op['walk_moved'] = now
            op['phase_start'] += self.AGGRO_EMERGENCY_GAP   # time lost to the Q
            return True
        self._release_mob_attack()
        self._set_auto_attack_movement(op['direction'])
        return True

    def _level_aggro_q(self, op, tracker, target, x, y, now, own_clear):
        """Q in the middle, turning each time (pigs on both sides). Each Q is
        confirmed by an MP drop; one that did not fire (hit mid-cast, knocked
        about) is pressed again."""
        if self._knocked_off_gather(op, x, y, now):
            return False
        pending = op.get('q_pending')
        if pending is not None:
            if now - pending[0] < self.AGGRO_Q_CONFIRM_SECONDS:
                return True
            mp = self._percent('mana')
            fired = mp is None or pending[1] is None or \
                pending[1] - mp >= self.AGGRO_Q_MP_DROP
            op['q_pending'] = None
            if fired:
                op['q_left'] -= 1
                op['q_fired'] += 1
                op['q_facing'] = 'left' if pending[2] == 'right' else 'right'
            elif op['q_retries'] < self.AGGRO_Q_RETRIES:
                op['q_retries'] += 1
                print('[~] That Q did not fire (no MP used); pressing it again.')
            else:
                op['q_left'] -= 1   # out of MP or Q not bound: don't stay up here
                print('[!] Q is not firing (no MP used); leading down what was hit.')
            op['next_q'] = now + max(
                0.0, self.AGGRO_Q_INTERVAL - self.AGGRO_Q_CONFIRM_SECONDS)
        if now < op['next_q']:
            return True
        if op['q_left'] <= 0:
            back = 'left' if op['direction'] == 'right' else 'right'
            op.update(phase='aggro_retreat', phase_start=now, back=back)
            print(f'[~] Pigs hit ({op["q_fired"]} Q); leading them down off the '
                  f'{back} edge.')
            return True
        facing = op['q_facing']
        mp = self._percent('mana')
        self._press_q(facing)
        op['q_pending'] = (now, mp, facing)
        return True

    def _knocked_off_gather(self, op, x, y, now):
        """A pig knocked the character off the step mid-gather: farm down here."""
        if self._level_of(x, y) != 'ground':
            op.pop('off_since', None)
            return False
        op.setdefault('off_since', now)
        if now - op['off_since'] < self.LEVEL_SETTLE_SECONDS:
            return False
        if op.get('q_fired'):
            self._finish_gather(now, 'Knocked off the step mid-gather; the pigs hit '
                                     'follow, fighting them down here')
        else:
            # Nothing aggroed yet: fight whatever is down here, then try again soon
            self._finish_gather(now, 'Knocked off the step before any Q; fighting '
                                     'down here, then trying the gather again',
                                cooldown=self.AGGRO_RETRY_COOLDOWN)
        return True

    def _finish_gather(self, now, reason, cooldown=None):
        self._set_auto_attack_movement(None)
        self._release_mob_attack()
        self._switch_level('ground', reason)
        self._last_switch = now
        self._climb_misses = None
        self._aggro_until = now + (
            self.AGGRO_COOLDOWN if cooldown is None else cooldown)
        self._level_op = None
        self._pursuit_watch = None
        self._patrol_seen = 0

    def _level_aggro_retreat(self, op, tracker, target, x, y, now, own_clear):
        """Back down off the edge onto the flat ground, a little way past it."""
        sign = 1 if op['back'] == 'right' else -1
        past = x is not None and sign * (x - op['pressed_x']) >= self.AGGRO_RETREAT_PAST
        if past or now - op['phase_start'] >= self.AGGRO_RETREAT_SECONDS:
            self._finish_gather(now, 'Back on the flat ground; fighting the pigs as they follow')
            return False
        self._release_mob_attack()
        self._set_auto_attack_movement(op['back'])
        return True

    def _recent_climb_misses(self, now):
        misses = self.__dict__.get('_climb_misses')
        if misses is None or now - misses[1] > self.LEVEL_MISS_MEMORY:
            return 0
        return misses[0]

    def _climb_missed(self, op, x, now, reason):
        self._climb_misses = (op['attempts'], now)
        if op['attempts'] < self.LEVEL_CLIMB_ATTEMPTS:
            print(f'\n[~] Climb missed ({reason}); pressing in and trying again.')
            op.update(phase='approach', start=now, x=x, moved=now)
            self._set_auto_attack_movement(op['direction'])
            return True
        self._end_level_op(
            f'could not get up the step after {op["attempts"]} jumps ({reason}); '
            f'farming this level for {self.LEVEL_CLIMB_BAN:.0f}s', ban=True)
        return False

    # -- walking back down --

    def _start_descend(self, count, offset, x, y, now):
        step = self.__dict__.get('_step')
        if step is not None and self.__dict__.get('_level') == 'step':
            direction = 'left' if step['direction'] == 'right' else 'right'
        else:
            direction = 'left' if offset < 0 else 'right'
        limit = self._map_limit(direction)
        if limit is not None and x is not None and abs(limit - x) <= self.MAP_LIMIT_DESCEND_REACH:
            # That edge drops past the map's limit (Pig Beach: into the pit
            # right of the pig island): go down the other side
            direction = 'left' if direction == 'right' else 'right'
        op = self._level_op = {
            'kind': 'descend', 'phase': 'descend', 'direction': direction,
            'start': now, 'start_y': y, 'x': x, 'moved': now, 'hopped': False,
        }
        self._release_mob_attack()
        self._pursuit_watch = None
        if count:
            print(f'\n[~] This level is clear; {count} pigs below. Walking down '
                  f'off the {direction} edge.')
        else:
            print(f'\n[~] Pig island cleared; walking down off the {direction} edge '
                  'to check the flat ground.')
        return op

    def _level_descend(self, op, tracker, target, x, y, now, own_clear):
        level = self._level_of(x, y)
        dropped = (
            y is not None and op['start_y'] is not None
            and y - op['start_y'] >= self.LEVEL_CLIMB_RISE
        )
        if level == 'ground' or dropped:
            self._switch_level('ground')
            self._last_switch = now
            self._end_level_op(None)
            print('\n[~] Down on the ground; farming here.')
            return False
        if now - op['start'] > self.LEVEL_ABORT_SECONDS \
                and own_clear < self.LEVEL_ABORT_SECONDS:
            self._end_level_op(None)    # a pig on this level after all: fight it
            return False
        if now - op['start'] >= self.LEVEL_DESCEND_SECONDS:
            self._end_level_op('could not get down off the step')
            self._last_switch = now     # don't retry at once
            return False
        if x is not None and (op['x'] is None
                              or abs(x - op['x']) > self.PATROL_MOVE_EPSILON):
            op['x'], op['moved'] = x, now
        elif x is not None and not op['hopped'] \
                and now - op['moved'] >= 2 * self.LEVEL_STUCK_SECONDS:
            # Held up (a pig in the way): hop over it
            keys = getattr(getattr(self.command_book, 'module', None), 'Key', None)
            self._jump_toward(op['direction'], getattr(keys, 'JUMP', None) or 'alt')
            op['hopped'] = True
        self._set_auto_attack_movement(op['direction'])
        return True

    def _end_level_op(self, reason, ban=False):
        op = self.__dict__.pop('_level_op', None)
        if op is None:
            return
        now = time.monotonic()
        self._set_auto_attack_movement(None)
        if op['kind'] == 'climb' and self.__dict__.get('_level', 'ground') == 'ground':
            # Still down here: the ground's walls (the step's foot) apply again
            self._walls = dict(self.__dict__.get('_level_walls', {}).get('ground', {}))
        if ban:
            self._climb_ban_until = now + self.LEVEL_CLIMB_BAN
            self._climb_misses = None
        self._pursuit_watch = None
        self._patrol_seen = 0
        if reason:
            print(f'\n[~] Level change stopped: {reason}.')

    # ---- Walls: the map's ends and raised steps the character cannot walk up ----

    def _minimap_x(self):
        """The character's minimap x, or None when unknown or not freshly seen."""
        position = getattr(config, 'player_pos', None)
        if not position or tuple(position) == (0, 0):
            return None
        seen = getattr(config, 'player_pos_time', None)
        if seen and time.monotonic() - seen > self.POSITION_STALE_SECONDS:
            return None     # minimap marker lost or game lagging: don't judge walls
        return position[0]

    def _learn_wall(self, direction, x):
        """
        Remembers a wall. One on the other side too close to it is contradictory
        (both came from a position that froze, not from real walls), so the
        older one is dropped.
        """
        walls = self.__dict__.setdefault('_walls', {})
        other = 'left' if direction == 'right' else 'right'
        if other in walls and abs(walls[other] - x) < self.MIN_WALL_GAP:
            del walls[other]
        walls[direction] = x
        self.__dict__.setdefault('_wall_learnt', {})[direction] = time.monotonic()

    # henny_hunting.mp4 (2026-11-19, Henesys Hunting Ground I): a wall learnt
    # by mistake at x ~0.29 on the open floor (short stops to tap loot add up
    # to "1 s not moving"). Walls on the right are only forgotten once passed,
    # and the patrol always turned there; enemies past it re-saved it on each
    # pursuit and were treated as out of reach. The bot paced 0.145-0.29 of a
    # floor that runs to 0.86. A learnt wall this old is dropped when reached:
    # a real one stops the walk again within PATROL_STUCK_SECONDS.
    WALL_RECHECK_SECONDS = 30.0

    def _drop_stale_wall(self, direction):
        """Forgets DIRECTION's learnt wall once it is WALL_RECHECK_SECONDS old
        (not on 'jumper' maps: Pig Beach's step walls stay as they were)."""
        walls = self.__dict__.get('_walls') or {}
        if direction not in walls:
            return
        learnt = self.__dict__.setdefault('_wall_learnt', {}).get(direction)
        now = time.monotonic()
        if learnt is None:
            self._wall_learnt[direction] = now      # (from before this rule)
            return
        if now - learnt < self.WALL_RECHECK_SECONDS:
            return
        tracker = getattr(getattr(config, 'capture', None), 'mob_tracker', None)
        if self._jumper_farming(tracker):
            return
        x = walls.pop(direction)
        self._wall_learnt.pop(direction, None)
        print(f'\n[~] Re-checking the wall on the {direction} at x={x:.3f} '
              f'(learnt {now - learnt:.0f} s ago): walking on.')
        if self.__dict__.get('_blocked_side') == direction:
            self._set_blocked_side(None)

    def _wall_ahead(self, direction, x):
        if x is None:
            return False
        self._drop_stale_wall(direction)
        for wall in (self.__dict__.setdefault('_walls', {}).get(direction),
                     self._map_limit(direction)):
            if wall is None:
                continue
            if direction == 'right' and x >= wall - self.WALL_NEAR:
                return True
            if direction == 'left' and x <= wall + self.WALL_NEAR:
                return True
        return False

    def _map_limit(self, direction):
        """
        A fixed wall from the map's profile (map_events.py), e.g. Pig Beach's
        pig island ends at 0.657 with a pit beyond it that there is no walking
        out of. Never learnt or forgotten: farming just never goes past it.
        """
        manager = self.__dict__.get('map_events')
        limits = manager.limits() if manager is not None else {}
        return limits.get(direction)

    def _check_walls(self, x):
        """
        Forgets a wall the character is now past (it was a mob or a stun, not a
        wall), and lets enemies on a blocked side count again once the character
        has walked well back from that wall.
        """
        if x is None:
            return
        walls = self.__dict__.setdefault('_walls', {})
        margin = 2 * self.PATROL_MOVE_EPSILON + self.WALL_NEAR
        if 'right' in walls and x > walls['right'] + margin:
            del walls['right']
        if 'left' in walls and x < walls['left'] - margin:
            del walls['left']
        side = self.__dict__.get('_blocked_side')
        if side is not None and (
                side not in walls or abs(x - walls[side]) >= self.WALL_RELEASE):
            self._set_blocked_side(None)

    def _pursuit_blocked(self, direction, x):
        """True when walking toward the target has stopped moving the character."""
        if x is None:
            self._pursuit_watch = None
            return False
        now = time.monotonic()
        watch = self.__dict__.get('_pursuit_watch')
        if (watch is None or watch[0] != direction
                or self._swing_locked(now)
                or abs(x - watch[1]) > self.PATROL_MOVE_EPSILON):
            self._pursuit_watch = (direction, x, now)
            return self._wall_ahead(direction, x)
        return (now - watch[2] >= self.PURSUIT_STUCK_SECONDS
                or self._wall_ahead(direction, x))

    def _note_walk(self, tracker, x):
        """Tells the tracker's facing reader when the character is really
        walking one way: arrow held a moment, no swing playing, and the
        minimap position moving that way (a knock-back moves it the other)."""
        direction = self.__dict__.get('_auto_attack_movement')
        now = time.monotonic()
        walk = self.__dict__.get('_walk_watch')
        if direction is None or x is None:
            self._walk_watch = None
            return
        if walk is None or walk[0] != direction:
            self._walk_watch = (direction, now, x)
            return
        moved = (x - walk[2]) * (1 if direction == 'right' else -1)
        if (now - walk[1] >= self.WALK_NOTE_AFTER and moved >= self.PATROL_MOVE_EPSILON
                and not self._swing_locked(now) and not self._mob_attack_keys):
            note = getattr(tracker, 'note_walking', None)
            if note is not None:
                note(direction)
            self._walk_watch = (direction, walk[1], x)     # must keep moving

    def _log_surroundings(self, tracker):
        """One log line when it changes: which way the character faces (as
        seen from its head, else as the keys left it) and how many enemies
        stand in front of it and behind it."""
        offsets_of = getattr(tracker, 'enemy_offsets', None)
        if offsets_of is None:
            return
        now = time.monotonic()
        last = self.__dict__.get('_surroundings_log')
        if last is not None and now - last[1] < self.SURROUNDINGS_LOG_GAP:
            return
        try:
            offsets = list(offsets_of())
            seen = getattr(tracker, 'facing', lambda: None)()
        except Exception:
            return
        believed = self.__dict__.get('_facing')
        facing = seen or believed
        if facing is None:
            return
        side = 1 if facing == 'right' else -1
        front = sorted(abs(o) for o in offsets if o * side > 0)
        behind = sorted(abs(o) for o in offsets if o * side <= 0)
        key = (facing, seen is not None, len(front), len(behind))
        if last is not None and last[0] == key and (
                not offsets or now - last[1] < self.SURROUNDINGS_LOG_EVERY):
            return
        self._surroundings_log = (key, now)
        if seen is None:
            how = 'by the keys; head not read'
        elif believed is not None and believed != seen:
            how = f'seen from the head; the keys said {believed}'
        else:
            how = 'seen from the head'

        def part(distances):
            if not distances:
                return '0'
            return f'{len(distances)} (nearest {distances[0]:.0f} px)'

        print(f'\n[~] Facing {facing} ({how}). Enemies in front: {part(front)}, '
              f'behind: {part(behind)}.')

    def _fix_facing(self, direction, tracker):
        """attack1.mp4: the turn tap landed mid-swing, the game ignored it and
        the character swung the wrong way for 6 s. When its head is seen
        facing the other way than the swing is meant to go: let go of the
        attack and hold the arrow until it is seen turned. True if it did."""
        seen_of = getattr(tracker, 'facing', None)
        if seen_of is None:
            return False
        now = time.monotonic()
        if now - self.__dict__.get('_turn_fix_at', float('-inf')) < self.TURN_FIX_GAP:
            return False
        try:
            seen = seen_of()
        except Exception:
            return False
        if seen is None or seen == direction:
            return False
        self._turn_fix_at = now
        print(f'\n[~] The character faces {seen} but the swing is meant to go {direction} '
              '(the turn was lost mid-swing); turning it again.')
        self._release_mob_attack()
        latest = getattr(tracker, 'facing_seen', seen_of)
        key_down(direction)
        try:
            started = time.monotonic()
            while time.monotonic() - started < self.TURN_FIX_MAX:
                time.sleep(0.03)
                if (time.monotonic() - started >= self.TURN_FIX_MIN
                        and latest() == direction):
                    break
        finally:
            key_up(direction)
        self._facing = direction
        return True

    def seconds_since_attack(self, now=None):
        """How long since the last attack (0 while one is held). The game
        refuses a channel change for a few seconds after attacking."""
        now = time.monotonic() if now is None else now
        hand = self.__dict__.get('_attack_hand_obj')
        if getattr(self, '_mob_attack_keys', ()) or getattr(hand, 'key', None) is not None:
            return 0.0
        return now - self.__dict__.get('_attack_released_at', float('-inf'))

    def _swing_locked(self, now=None):
        """True just after the attack key was let go: the swing is still
        playing and the character can't walk yet, so it is not stuck."""
        now = time.monotonic() if now is None else now
        return (now - self.__dict__.get('_attack_released_at', float('-inf'))
                < self.SWING_LOCK_SECONDS)

    def _sweeping(self, tracker=None):
        """True with the tracker's 'Sweep patrol' option on (a long, sloped map)."""
        if tracker is None:
            tracker = getattr(getattr(config, 'capture', None), 'mob_tracker', None)
        check = getattr(tracker, 'sweep_patrol_enabled', None)
        try:
            return bool(check and check())
        except Exception:
            return False

    def _turn_back_from_wall(self, direction, x):
        """
        The target is past a wall or up a step: remember the wall, tell the tracker
        that side is out of reach, and patrol the other way.
        """
        known = self._wall_ahead(direction, x)
        if not known:
            # (a wall already known is not renewed here: henny_hunting.mp4, an
            # enemy past a mistaken wall kept it alive for good)
            self._learn_wall(direction, x)
        self._set_blocked_side(direction)
        self._pursuit_watch = None
        if not known:
            print(
                f'\n[~] Blocked walking {direction} (a wall or a raised step); '
                'turning back to patrol.'
            )
        self._patrol_direction = 'left' if direction == 'right' else 'right'
        self._patrol_seen = 0           # start a fresh patrol leg
        self._patrol_for_mob()

    def _flip_flopping(self, direction):
        """
        Watchdog: FLIP_LIMIT left/right swaps within FLIP_WINDOW means something
        is contradicting itself (e.g. walls learnt from a frozen position). Forget
        the walls, take the scan direction fresh, and say so once.
        """
        now = time.monotonic()
        last = self.__dict__.get('_last_walk_direction')
        self._last_walk_direction = direction
        if last is None or last == direction:
            return False
        flips = [t for t in self.__dict__.get('_flip_times', []) if now - t <= self.FLIP_WINDOW]
        flips.append(now)
        self._flip_times = flips
        if len(flips) < self.FLIP_LIMIT:
            return False
        self._flip_times = []
        self._last_walk_direction = None
        print(
            f'\n[!] Turned left/right {len(flips)} times in {self.FLIP_WINDOW:.0f} sec; '
            'forgetting learnt walls and pausing a moment.'
        )
        self._forget_walls()
        self._patrol_seen = 0
        self._last_turn = now
        return True

    def _set_blocked_side(self, side):
        if side == self.__dict__.get('_blocked_side'):
            return
        self._blocked_side = side
        tracker = getattr(getattr(config, 'capture', None), 'mob_tracker', None)
        if hasattr(tracker, 'set_blocked_side'):
            tracker.set_blocked_side(side)

    def _forget_walls(self):
        """Walls belong to one map: forget them whenever the bot is switched off."""
        self.__dict__.pop('_level_op', None)
        self._stall_since = None
        self._empty_since = None
        if self.__dict__.get('_walls'):
            self._walls = {}
        self._wall_learnt = {}
        self._pursuit_watch = None
        self._empty_turns = 0
        self._set_blocked_side(None)

    def _patrol_for_mob(self):
        """
        Nothing in sight: walk one way until the character stops moving (a wall or
        the end of the map), then walk back. The tracker looks far along the level
        while this runs, so the walk ends as soon as an enemy comes into view.
        """

        now = time.monotonic()
        x = self._minimap_x()
        known = x is not None
        if known:
            if self.__dict__.get('_blind_since') is not None and \
                    self.__dict__.get('_blind_stopped'):
                print('\n[~] Minimap position is back; patrolling again.')
            self._blind_since = None
            self._blind_stopped = False
        else:
            edge = self._last_edge_side()
            if self.__dict__.get('_blind_since') is None:
                self._blind_since = now
                # bug-2.mp4: the position went at the map's right end, and the
                # 12 s leg (with no stuck check while lost) kept walking into
                # the wall. Lost: short timed legs, and never on toward an end.
                if self._patrol_direction is not None:
                    self._patrol_until = min(
                        getattr(self, '_patrol_until', now),
                        now + self.PATROL_TURN_INTERVAL)
            if now - self._blind_since >= self.BLIND_PATROL_SECONDS:
                # Lost for a while (capture failing, minimap hidden, low memory):
                # walking left/right on a timer forever goes nowhere useful
                if not self.__dict__.get('_blind_stopped'):
                    self._blind_stopped = True
                    print(
                        f'\n[!] No minimap position for {self.BLIND_PATROL_SECONDS:.0f}s; '
                        'standing still until it comes back (check the minimap is '
                        'visible and the console for capture errors).'
                    )
                away = None
                if edge is not None:
                    # Last seen at a map end (marker on the minimap's border,
                    # the name tag off screen): a short step away now and then
                    # brings both back; standing there never would
                    if now - self.__dict__.get('_blind_nudge_at', float('-inf')) \
                            >= self.BLIND_NUDGE_INTERVAL:
                        self._blind_nudge_at = now
                    if now - self._blind_nudge_at < self.BLIND_NUDGE_SECONDS:
                        away = 'left' if edge == 'right' else 'right'
                self._set_auto_attack_movement(away)
                self._patrol_seen = now
                return
        if self._service_edge_loot(now, x):
            self._patrol_seen = now
            return
        gap = now - getattr(self, '_patrol_seen', 0)
        if gap > self.PATROL_RESUME_GAP:
            # Something else had the controls (a loot tap, a hold, a swing):
            # standing still then is no wall (henny_hunting.mp4: short stops
            # added up to "stuck" and a wall was learnt on the open floor)
            self._patrol_x, self._patrol_moved = x, now
        if gap > 0.5 or self.__dict__.get('_quiet_since') is None:
            self._quiet_since = now     # (enemies were around until now)
        if gap > 0.5:
            # Patrol is starting again (after a fight): don't count old stillness
            self._patrol_x, self._patrol_moved = x, now
            if self._patrol_direction is not None:
                self._patrol_until = now + (
                    self._leg_seconds() if known else self.PATROL_TURN_INTERVAL
                )
        self._patrol_seen = now
        if known and self._idle_pause(now, x):
            return
        if self._swing_locked(now):
            self._patrol_x, self._patrol_moved = x, now     # can't walk yet

        if self._patrol_direction is None:
            direction = 'right'
        else:
            direction = self._patrol_direction
            if x is not None and (
                    self._patrol_x is None
                    or abs(x - self._patrol_x) > self.PATROL_MOVE_EPSILON):
                self._patrol_x, self._patrol_moved = x, now
            stuck = known and now - self._patrol_moved >= self.PATROL_STUCK_SECONDS
            if stuck:   # a wall, the map's end or a raised step: remember where
                self._learn_wall(direction, x)
            # A remembered wall turns the patrol early, but never straight after a
            # turn: two walls at one spot must not make it turn every tick
            at_wall = (
                known and self._wall_ahead(direction, x)
                and now - self.__dict__.get('_last_turn', 0) >= self.TURN_COOLDOWN
            )
            # (lost after being seen at a map end: walk away from that end
            # only; its wall can't be seen without a position)
            lost_end = None if known else self._last_edge_side()
            other = 'left' if direction == 'right' else 'right'
            # (a sweep turns only where the walk stops: a long map takes
            # longer than PATROL_MAX_LEG to cross)
            sweeping = known and self._sweeping()
            turn = stuck or at_wall or (now >= self._patrol_until and not sweeping)
            if turn and sweeping:
                print(f'\n[~] Sweep: the walk {direction} ended at x={x:.3f}; '
                      f'heading {other} now.')
            if lost_end == direction:
                turn = True
            elif turn and lost_end == other:
                turn = False            # (a timed turn would go back to it)
                self._patrol_until = now + self.PATROL_TURN_INTERVAL
            if turn and known and (stuck or at_wall) \
                    and self._start_edge_loot(direction, x, now):
                self._set_auto_attack_movement(direction)
                return
            if turn:
                direction = other

        if direction != self._patrol_direction:
            if self._patrol_direction is not None and known:
                self._empty_turn()     # (timed turns while lost don't count)
            self._last_turn = now
            self._patrol_until = now + (
                self._leg_seconds() if known else self.PATROL_TURN_INTERVAL
            )
            self._patrol_x, self._patrol_moved = x, now
        self._patrol_direction = direction
        config.capture.mob_tracker.set_sweep_direction(direction)
        self._set_auto_attack_movement(direction)
        if known:
            self._idle_hop(direction, x, now)

    # ---- Edge loot jumps and idle variety (both opt-in per profile) ----

    def _tracker_option(self, name):
        tracker = getattr(getattr(config, 'capture', None), 'mob_tracker', None)
        check = getattr(tracker, name, None)
        try:
            return bool(check and check())
        except Exception:
            return False

    def _jump_key(self):
        module = getattr(getattr(self, 'command_book', None), 'module', None)
        return getattr(getattr(module, 'Key', None), 'JUMP', None) or 'alt'

    def _leg_seconds(self):
        """How long a patrol leg may last before a timed turn. Random with
        'Idle variety', but never shorter with 'Edge loot jumps' (the walk
        must reach the map's ends)."""
        if (self._tracker_option('idle_variety_enabled')
                and not self._tracker_option('edge_loot_enabled')):
            return random.uniform(*self.IDLE_LEG)
        return self.PATROL_MAX_LEG

    def _start_edge_loot(self, side, x, now):
        """The patrol's walk toward SIDE stopped: jump up there for loot first."""
        if not self._tracker_option('edge_loot_enabled'):
            return False
        last = self.__dict__.setdefault('_edge_loot_at', {})
        if now - last.get(side, float('-inf')) < self.EDGE_LOOT_COOLDOWN:
            return False
        last[side] = now
        jumps = random.randint(*self.EDGE_LOOT_JUMPS)
        self._edge_loot = {'side': side, 'jumps': jumps, 'next': now,
                           'until': now + self.EDGE_LOOT_MAX_SECONDS}
        print(f'\n[~] Patrol stopped at the {side} end (x={x:.3f}); '
              f'jumping up there for loot ({jumps}x) before turning.')
        return True

    def _service_edge_loot(self, now, x):
        """True while an edge-loot visit has the controls."""
        op = self.__dict__.get('_edge_loot')
        if op is None:
            return False
        if now - getattr(self, '_patrol_seen', now) > 0.5 or now >= op['until']:
            self._edge_loot = None      # a fight came first, or it took too long
            return False
        side = op['side']
        if now < op['next']:
            self._set_auto_attack_movement(side)
            return True
        loot_key = getattr(getattr(config.capture, 'loot_tracker', None), 'LOOT_KEY', 'x')
        if op['jumps'] > 0:
            op['jumps'] -= 1
            self._jump_toward(side, self._jump_key())
            self._tap_loot(loot_key, self.EDGE_LOOT_TAPS)
            op['next'] = time.monotonic() + random.uniform(*self.EDGE_LOOT_GAP)
            return True
        # done: a last few taps where it landed, then turn for home
        self._tap_loot(loot_key, self.EDGE_LOOT_TAPS)
        self._edge_loot = None
        other = 'left' if side == 'right' else 'right'
        self._patrol_direction = other
        self._last_turn = now
        self._patrol_until = now + self._leg_seconds()
        self._patrol_x, self._patrol_moved = x, now
        self._empty_turn()
        config.capture.mob_tracker.set_sweep_direction(other)
        self._set_auto_attack_movement(other)
        return True

    def _idle_pause(self, now, x):
        """'Idle variety': stands still a moment now and then. True while it does."""
        if not self._tracker_option('idle_variety_enabled'):
            return False
        if now < self.__dict__.get('_idle_pause_until', 0):
            self._set_auto_attack_movement(None)
            self._patrol_x, self._patrol_moved = x, now     # standing isn't stuck
            return True
        due = self.__dict__.get('_idle_pause_due')
        if due is None:
            self._idle_pause_due = now + random.uniform(*self.IDLE_PAUSE_EVERY)
            return False
        if now >= due:
            self._idle_pause_until = now + random.uniform(*self.IDLE_PAUSE_FOR)
            self._idle_pause_due = self._idle_pause_until + random.uniform(*self.IDLE_PAUSE_EVERY)
            self._set_auto_attack_movement(None)
            self._patrol_x, self._patrol_moved = x, now
            return True
        return False

    def _idle_hop(self, direction, x, now):
        """'Idle variety': a hop (sometimes two) while walking, away from walls.
        After IDLE_QUIET_AFTER s with no enemy: hops come more often and vary
        (single, double, or the jump key held for a few bounces), with the odd
        jump-free stretch in between."""
        if not self._tracker_option('idle_variety_enabled') or self._sweeping():
            return
        quiet = now - self.__dict__.get('_quiet_since', now) >= self.IDLE_QUIET_AFTER
        every = self.IDLE_QUIET_HOP_EVERY if quiet else self.IDLE_HOP_EVERY
        due = self.__dict__.get('_idle_hop_due')
        if due is None:
            self._idle_hop_due = now + random.uniform(*every)
            return
        if now < due:
            return
        ahead = [w for w in (self.__dict__.get('_walls', {}).get(direction),
                             self._map_limit(direction)) if w is not None]
        room = min((abs(w - x) for w in ahead), default=float('inf'))
        if room < self.IDLE_HOP_WALL_GAP:
            self._idle_hop_due = now + 1.0      # too close to an end: later
            return
        held = (quiet and room >= self.IDLE_QUIET_HELD_WALL_GAP
                and random.random() < self.IDLE_QUIET_HELD_JUMP)
        if held:
            self._held_jump_toward(direction, self._jump_key(),
                                   random.uniform(*self.IDLE_QUIET_HOLD_FOR))
        else:
            self._jump_toward(direction, self._jump_key())
        later = time.monotonic()
        double = self.IDLE_QUIET_DOUBLE_HOP if quiet else self.IDLE_DOUBLE_HOP
        if not held and random.random() < double:
            self._idle_hop_due = later + random.uniform(0.45, 0.7)
        elif quiet and random.random() < self.IDLE_QUIET_CALM:
            self._idle_hop_due = later + random.uniform(*self.IDLE_QUIET_CALM_FOR)
        else:
            self._idle_hop_due = later + random.uniform(*every)

    def _held_jump_toward(self, direction, jump_key, seconds):
        """Like _jump_toward, but the jump key stays down SECONDS, so the
        character bounces a few times in a row."""
        self._release_mob_attack()
        self._set_auto_attack_movement(direction)
        if direction is not None:
            key_down(direction)
        time.sleep(0.06)
        key_down(jump_key)
        try:
            time.sleep(seconds)
        finally:
            key_up(jump_key)

    MAP_END_MARGIN = 0.06           # last minimap x this close to 0 or 1: at a map end
    BLIND_NUDGE_INTERVAL = 6.0      # lost at a map end: step away this often...
    BLIND_NUDGE_SECONDS = 0.8       # ...for this long

    def _last_edge_side(self):
        """'left'/'right' when the last minimap fix (however old) was at that
        end of the map, else None."""
        position = getattr(config, 'player_pos', None)
        if not position or tuple(position) == (0, 0):
            return None
        if position[0] >= 1 - self.MAP_END_MARGIN:
            return 'right'
        if position[0] <= self.MAP_END_MARGIN:
            return 'left'
        return None

    FACE_TAP_SECONDS = 0.04         # an arrow tap this short turns without a step
    FACE_TAP_GAP = 0.3

    def _face(self, offset):
        """Turns toward a target OFFSET px away (a short arrow tap), if needed."""
        if abs(offset) < self.NORMAL_FACE_DEADBAND:
            return
        direction = 'left' if offset < 0 else 'right'
        if self.__dict__.get('_facing') == direction:
            return
        now = time.monotonic()
        if now - self.__dict__.get('_face_tap_at', float('-inf')) < self.FACE_TAP_GAP:
            return
        self._face_tap_at = now
        key_down(direction)
        time.sleep(self.FACE_TAP_SECONDS)
        key_up(direction)
        self._facing = direction

    def _set_auto_attack_movement(self, direction):
        if direction is not None:
            last = self.__dict__.get('_walk_last')
            now = time.monotonic()
            if (last is not None and last[0] != direction
                    and now - last[1] < self.SWEEP_REVERSE_SECONDS and self._sweeping()):
                direction = None    # was just walking the other way: stand a moment
            else:
                self._walk_last = (direction, now)
        if direction is not None:
            self._facing = direction
        if direction == self._auto_attack_movement:
            return
        if direction is not None and self._flip_flopping(direction):
            direction = None    # stand still this tick; walls were just reset
            if direction == self._auto_attack_movement:
                return
        if self._auto_attack_movement is not None and                 self._auto_attack_movement not in getattr(self, "_mob_attack_keys", ()):
            key_up(self._auto_attack_movement)
        self._auto_attack_movement = direction
        if direction is not None and config.enabled:
            key_down(direction)

    def _service_loot_pickup(self):
        """The loot detector's request (routine mode, and a hint in mob mode)."""
        key = config.capture.loot_tracker.consume_pickup()
        if key is None:
            return False
        if not is_game_focused():
            print('\n[!] Loot detected, but MapleStory is not focused; pickup cancelled.')
            return False
        planner = self.__dict__.get('_loot_planner')
        if planner is not None:
            planner.seen(time.monotonic())  # taps follow in _service_smart_loot
            return True
        print(f'\n[~] Loot is under the player; sending quick {key.upper()} pickup taps.')
        press(key, 3, down_time=0.04, up_time=0.05)
        return True

    # ---- Looting where the drops fall (see loot_planner.py) ----

    def _tap_loot(self, key, count=1):
        """Taps the loot key without holding up walking or swinging."""
        if press is not _BLOCKING_PRESS:
            press(key, count, down_time=0.05, up_time=0.03)
            return
        hand = self.__dict__.get('_loot_hand')
        if hand is None:
            hand = self._loot_hand = LootHand(
                lambda k: key_down(k), lambda k: key_up(k),
                lambda: bool(config.enabled) and is_game_focused())
        hand.tap(key, count)

    def _loot_enabled(self):
        loot = getattr(getattr(config, 'capture', None), 'loot_tracker', None)
        settings = getattr(loot, 'settings', None)
        try:
            return bool(settings and settings().get('Enabled'))
        except Exception:
            return False

    def _service_smart_loot(self, tracker, target, x):
        """Presses pickup where drops are expected: after kills, between swings,
        over fight spots not swept yet."""
        if not self._loot_enabled():
            return
        planner = self.__dict__.get('_loot_planner')
        if planner is None:
            planner = self._loot_planner = LootPlanner()
        loot_key = getattr(config.capture.loot_tracker, 'LOOT_KEY', 'x')
        vkeys.QUIET_KEYS.add(loot_key)
        now = time.monotonic()
        level = self.__dict__.get('_level', 'ground')
        attacking = target is not None and bool(target[1])
        kills = getattr(tracker, 'consume_kills', None)
        kills = kills() if kills is not None else []
        was_attacking = self.__dict__.get('_loot_was_attacking', False)
        if not kills and was_attacking and target is None:
            kills = [(now, 0)]      # no kill report: the target just vanished
            if self._jumper_farming(tracker):
                self._vanish_hold_until = now + self.VANISH_HOLD_SECONDS
        self._loot_was_attacking = attacking
        if attacking:
            planner.fought(now, x, level)
        for _ in kills:
            planner.killed(now, x, level)
        taps, swept = planner.taps_due(now, x, level, attacking)
        if not taps and self._underfoot_tap_due(now):
            taps = 1
        if taps:
            self._last_drop_tap = now
            # (its own finger: never holds up the arrows or the attack key)
            self._tap_loot(loot_key, taps)
        if kills:
            print(f'\n[~] Kill: picking up its drops ({loot_key.upper()} taps).')
        elif swept is not None:
            print(f'\n[~] Swept the drop spot at x={swept.x:.3f} '
                  f'({planner.SWEEP_TAPS} {loot_key.upper()} taps).')

    def _visible_drops(self):
        loot = getattr(getattr(config, 'capture', None), 'loot_tracker', None)
        finder = getattr(loot, 'ground_drops', None)
        if finder is None or not self._loot_enabled():
            return []
        try:
            return finder()
        except Exception:
            return []

    UNDERFOOT_STREAK_MAX = 10       # taps in a row at "loot underfoot" before a pause
    UNDERFOOT_PAUSE = 3.0           # (the character's feet can look like a drop)

    def _underfoot_tap_due(self, now):
        """Spams pickup while the detector sees a drop underfoot, like a player,
        but a streak that never clears it is likely a false match: pause."""
        if now < self.__dict__.get('_underfoot_pause_until', float('-inf')):
            return False
        if not self._drop_underfoot():
            self._underfoot_streak = 0
            return False
        gap = self.__dict__.get('_underfoot_gap') or self.DROP_TAP_INTERVAL
        if now - self.__dict__.get('_last_drop_tap', float('-inf')) < gap:
            return False
        self._underfoot_gap = random.uniform(0.09, 0.15)
        streak = self.__dict__.get('_underfoot_streak', 0) + 1
        self._underfoot_streak = streak
        if streak > self.UNDERFOOT_STREAK_MAX:
            self._underfoot_streak = 0
            self._underfoot_pause_until = now + self.UNDERFOOT_PAUSE
            return False
        return True

    def _drop_underfoot(self):
        return any(abs(d[0]) <= self.DROP_PICK_RADIUS for d in self._visible_drops())

    def _drop_banned(self, direction, x, now):
        bans = [b for b in self.__dict__.get('_drop_bans', []) if b[2] > now]
        self._drop_bans = bans
        # (A ban made, or checked, without a minimap position covers that whole
        # direction while it lasts: comparing with None crashed the bot thread)
        return any(
            b[0] == direction and (
                x is None or b[1] is None or abs(x - b[1]) <= self.WALL_RELEASE)
            for b in bans)

    def _ban_drops(self, direction, x, now, reason):
        self.__dict__.setdefault('_drop_bans', []).append(
            (direction, x, now + self.DROP_BAN_SECONDS))
        self._drop_chase = None
        print(f'\n[~] Leaving the drops {"here" if direction == "here" else "to the " + direction}'
              f' for now ({reason}).')

    DROP_LIMIT_MARGIN = 0.004       # minimap units

    def _drop_past_limit(self, x, offset):
        """
        True when a drop OFFSET screen px away lies past the map profile's
        limits. stuck.mp4 (A Split Road): a drop down on the low ground right
        of the platform was walked to, off the edge, and the character was
        stranded one level below the safe spot.
        """
        if x is None:
            return False
        right, left = self._map_limit('right'), self._map_limit('left')
        if right is None and left is None:
            return False
        tracker = getattr(config.capture, 'mob_tracker', None)
        scale_of = getattr(tracker, '_limit_scale', None)
        try:
            scale = scale_of() if scale_of else None
        except Exception:
            scale = None
        scale = scale or self.__dict__.get('_px_per_minimap') or self.PX_PER_MINIMAP
        world = x + offset / scale
        return ((right is not None and world > right + self.DROP_LIMIT_MARGIN)
                or (left is not None and world < left - self.DROP_LIMIT_MARGIN))

    def _drop_past_wall(self, direction, x, offset, now):
        """The walk to a drop stopped at a wall. The second time at the same
        place it is no drop of ours (decor past the map's edge that looks like
        one): remembered as scenery, so the walk is not tried every 8 s."""
        if x is None:
            return
        hits = [h for h in self.__dict__.get('_drop_wall_hits', [])
                if now - h[2] <= self.DROP_WALL_MEMORY]
        again = any(h[0] == direction and abs(h[1] - x) <= self.WALL_RELEASE for h in hits)
        hits.append((direction, x, now))
        self._drop_wall_hits = hits[-8:]
        if again:
            scale = self.__dict__.get('_px_per_minimap') or self.PX_PER_MINIMAP
            self._remember_scenery_drop(x + offset / scale, now)
            print(f'\n[~] The drop to the {direction} can\'t be reached (twice now): '
                  'scenery past the edge. Ignoring it.')

    def _drop_safe(self, offset, enemy_offset):
        """With an enemy on its way (ENEMY_OFFSET px, not in reach yet): is the
        drop at OFFSET worth stepping to? Only one close by, and only if the
        enemy is still more than a swing away from where the bot will stand."""
        if abs(offset) > self.DROP_DETOUR:
            return False
        if abs(offset) <= self.DROP_PICK_RADIUS:
            return True                     # underfoot: no step needed
        same_side = (offset < 0) == (enemy_offset < 0)
        if same_side:
            return abs(enemy_offset) - abs(offset) >= self.DROP_SAFE_GAP
        return abs(enemy_offset) >= self.DROP_BEHIND_GAP

    def _remember_scenery_drop(self, x, now):
        if x is None:
            return
        spots = [s for s in self.__dict__.get('_scenery_drops', []) if s[2] > now]
        spots.append((x, self.__dict__.get('_level', 'ground'),
                      now + self.SCENERY_DROP_SECONDS))
        self._scenery_drops = spots[-self.SCENERY_MAX:]

    def _scenery_drop(self, x, offset, now):
        """Is the drop OFFSET px from the character one that never picked up?"""
        spots = self.__dict__.get('_scenery_drops')
        if not spots or x is None:
            return False
        level = self.__dict__.get('_level', 'ground')
        scale = self.__dict__.get('_px_per_minimap') or self.PX_PER_MINIMAP
        drop_x = x + offset / scale
        return any(
            expires > now and spot_level == level
            and abs(drop_x - spot_x) <= self.SCENERY_DROP_RADIUS
            for spot_x, spot_level, expires in spots)

    def _collect_visible_drop(self, x, target=None):
        """
        Walks to the nearest drop in view and picks it up. With a pig on its
        way (TARGET, not yet in reach) only a drop close by, and not farther
        than the pig, is worth it. True while it does.
        """
        drops = self._visible_drops()
        if not drops:
            self._drop_chase = None
            return False
        now = time.monotonic()
        # Every enemy on this level, not just the target: a walk to a drop
        # must not go into (or through) another one (knock-back-dmg.mp4)
        offsets_of = getattr(getattr(config.capture, 'mob_tracker', None),
                             'enemy_offsets', None)
        try:
            enemies = list(offsets_of()) if offsets_of else []
        except Exception:
            enemies = []
        if target is not None:
            enemies.append(target[0])
        sweeping = self._sweeping()
        sweep_side = -1 if self.__dict__.get('_patrol_direction') == 'left' else 1
        last_target = self.__dict__.get('_last_target')
        candidates = []
        for offset, _, _ in drops:
            direction = 'here' if abs(offset) <= self.DROP_PICK_RADIUS else (
                'left' if offset < 0 else 'right')
            if self._drop_banned(direction, x, now):
                continue
            if direction != 'here' and self._map_limit(direction) is not None \
                    and self._wall_ahead(direction, x):
                continue                    # past the map's limit (the pit)
            if direction != 'here' and self._drop_past_limit(x, offset):
                continue                    # lies past the limit: walking there leaves the platform
            if any(not self._drop_safe(offset, enemy) for enemy in enemies):
                continue
            if self._scenery_drop(x, offset, now):
                continue                    # looks like a drop, never picks up
            if sweeping and offset * sweep_side < -self.SWEEP_DROP_BEHIND:
                continue                    # well behind the sweep: on the way back
            if (target is None and direction != 'here' and last_target is not None
                    and direction != last_target[0]
                    and now - last_target[1] < self.DROP_AFTER_TARGET):
                continue                    # the target was the other way a moment ago
            candidates.append((abs(offset), offset, direction))
        if not candidates:
            self._drop_chase = None
            return False
        _, offset, direction = min(candidates)
        chase = self.__dict__.get('_drop_chase')
        if chase is None or chase['direction'] != direction:
            previous = chase
            chase = self._drop_chase = {
                'direction': direction, 'start': now, 'x': x, 'moved': now, 'taps': 0,
                'from': (x, offset)}
            if (direction == 'here' and previous is not None
                    and previous.get('from') and x is not None
                    and previous['from'][0] is not None):
                # Walked to it: how many screen px one minimap unit is
                start_x, start_offset = previous['from']
                walked = x - start_x
                if (abs(walked) >= 0.01 and abs(start_offset) >= 60
                        and (walked > 0) == (start_offset > 0)):
                    measured = abs(start_offset / walked)
                    if 1500 <= measured <= 20000:
                        known = self.__dict__.get('_px_per_minimap')
                        self._px_per_minimap = (
                            measured if known is None else 0.7 * known + 0.3 * measured)
        loot_key = getattr(config.capture.loot_tracker, 'LOOT_KEY', 'x')
        if direction == 'here':
            self._set_auto_attack_movement(None)
            if now - self.__dict__.get('_last_drop_tap', float('-inf')) >= self.DROP_TAP_INTERVAL:
                self._last_drop_tap = now
                chase['taps'] += 1
                self._tap_loot(loot_key, 1)
                if chase['taps'] > self.DROP_MAX_TAPS:
                    self._remember_scenery_drop(x, now)
                    self._ban_drops('here', x, now, 'it will not pick up')
            return True
        if x is not None and (chase['x'] is None or self._swing_locked(now)
                              or abs(x - chase['x']) > self.PATROL_MOVE_EPSILON):
            chase['x'], chase['moved'] = x, now
        if now - chase['moved'] >= self.DROP_STUCK_SECONDS:
            self._ban_drops(direction, x, now, 'a wall in the way')
            self._drop_past_wall(direction, x, offset, now)
            return False
        if now - chase['start'] >= self.DROP_CHASE_SECONDS:
            self._ban_drops(direction, x, now, 'could not get there')
            return False
        self._set_auto_attack_movement(direction)
        return True

    def _collect_loot(self, x):
        """
        With nothing to fight: walks to the nearest drop spot not swept yet.
        True while it does (instead of patrolling).
        """
        planner = self.__dict__.get('_loot_planner')
        if planner is None or not self._loot_enabled():
            return False
        now = time.monotonic()
        level = self.__dict__.get('_level', 'ground')
        direction = planner.collect_direction(now, x, level)
        if direction is None:
            return False
        if direction == 'stay':
            self._set_auto_attack_movement(None)
            return True
        if self._wall_ahead(direction, x):
            planner.give_up(now, x, level, direction)
            return False
        heading = self.__dict__.get('_patrol_direction')
        if heading in ('left', 'right') and direction != heading and self._sweeping():
            planner.give_up(now, x, level, direction)   # behind the sweep
            return False
        self._set_auto_attack_movement(direction)
        return True

    def _service_mob_attack(self, movement_direction=None, stationary=False):
        tracker = config.capture.mob_tracker
        direction, group_attack, group_count = (
            tracker.consume_attack_action()
        )
        if not is_game_focused():
            self._release_mob_attack()
            return False

        target = tracker.target()
        if (group_attack and direction is not None and self._group_skill()
                and self._group_q_ready()):
            was_swinging = bool(self._mob_attack_keys)
            mp_before = self._percent('mana')
            self._release_mob_attack()
            if movement_direction:
                key_up(movement_direction)
            print(
                f'\n[~] Group of {group_count} nearby enemies detected on the '
                f'{direction}; turning and using Q.'
            )
            if was_swinging:
                time.sleep(self.GROUP_Q_SETTLE)     # mid-swing the game drops Q
            try:
                key_down(direction)
                time.sleep(0.06)
            finally:
                key_up(direction)
            press('q', 1, down_time=0.08, up_time=0.1)
            self._group_q = (time.monotonic(), mp_before)
            return True

        jumper = self._jumper_farming(tracker)
        # (A group in range but no Q this tick - Q cooling down, out of MP, or
        # no Q at all: keep swinging at the nearest instead of standing there.
        # knock-back-dmg.mp4: on the snail map the bot let go of the attack
        # between Qs and stood in the pile taking bites.)
        if target is None or not target[1]:
            self._release_mob_attack()
            return False

        if (self.command_book is None
                or 'cruelstab' not in self.command_book.dict):
            self._release_mob_attack()
            print(
                '\n[!] Nearby enemy detected, but the loaded command book '
                'does not provide CruelStab.'
            )
            return False
        attack_key = getattr(
            self.command_book.module.Key, 'CRUEL_STAB', None
        )
        if not attack_key:
            self._release_mob_attack()
            print(
                '\n[!] The loaded command book does not define a CruelStab key.'
            )
            return False

        attack_direction = 'left' if target[0] < 0 else 'right'
        held = (self.__dict__.get('_facing') if self._human_hold(tracker)
                else self._mob_attack_keys[0] if self._mob_attack_keys else None)
        deadband = self.JUMPER_FACE_DEADBAND if jumper else self.NORMAL_FACE_DEADBAND
        if held in ('left', 'right') and held != attack_direction                 and abs(target[0]) < deadband:
            attack_direction = held     # overlapping: don't turn back and forth
        if movement_direction:
            key_up(movement_direction)
        if self._auto_attack_movement is not None:
            self._set_auto_attack_movement(None)
        self._hold_attack(
            attack_direction, attack_key, tracker, stationary=stationary
        )
        return True

    def _follow_up_swing(self, tracker):
        """While the tracker's follow-up window is open (a swing just went out
        at a slow walker), keep the attack held that way and stand. True if so."""
        follow = getattr(tracker, 'follow_up_direction', lambda: None)()
        if follow is None or self._jumper_farming(tracker):
            return False
        attack_key = getattr(getattr(getattr(self.command_book, 'module', None), 'Key', None),
                             'CRUEL_STAB', None) if self.command_book is not None else None
        if not attack_key:
            return False
        self._set_auto_attack_movement(None)
        self._hold_attack(follow, attack_key, tracker)
        return True

    def _group_q_ready(self):
        """False right after a Q that fired (it cost MP): no double cast."""
        last = self.__dict__.get('_group_q')
        if last is None:
            return True
        elapsed = time.monotonic() - last[0]
        if elapsed >= self.GROUP_Q_COOLDOWN:
            return True
        if elapsed < self.GROUP_Q_RETRY:
            return False
        mp_now = self._percent('mana')
        if last[1] is None or mp_now is None:
            return False            # can't tell: assume it fired
        return mp_now > last[1] - self.AGGRO_Q_MP_DROP      # no MP spent: it didn't

    # ---- Ranged pull: the magician's Magic Missile (magic-class.mp4) ----
    # In the clip the missile left ~0.2 s after the key, flew from ~135 px in
    # front of the caster and hit a snail ~260 px away ~0.45 s after the key;
    # the next cast came ~1.25 s later. Idea: hit an enemy from afar so it
    # comes, stand facing it and let the melee model swing as it arrives,
    # instead of walking into it. Only with the GUI's "Magic missile" box
    # ticked (off for the beginner, whose profiles don't have the option).
    MISSILE_MIN_RANGE = 110         # closer: the melee approach handles it
    MISSILE_CAST_SECONDS = 0.6      # stand still this long after the key
    MISSILE_RELEASE_SECONDS = 0.3   # (the missile is out: in swing range, swing)
    MISSILE_JUDGE_SECONDS = 0.9     # an HP bar on it by then, or it missed
    MISSILE_COOLDOWN = 1.0
    # bug-2.mp4: one missile, then the bot waited (or walked) for melee and
    # the red snail bit the character. More missiles while it is still far:
    MISSILE_HITS_PER_TARGET = 2     # then let it come and finish it with melee
    MISSILE_GROUP_HITS = 3          # ...per enemy while a group is in range
    MISSILE_GROUP_MIN = 2           # enemies within missile range: a group
    MISSILE_KITE_MIN = 130          # px it must still be away when the cast ends
    MISSILE_MISSES_PER_TARGET = 2   # then walk to it the normal way
    MISSILE_BAN_SECONDS = 60.0      # a spot that took a missile and showed no
    MISSILE_BAN_PX = 45             # bar (a portal, scenery): no more casts at it
    MISSILE_BAR_DROP = 0.03         # HP bar ratio drop that is a hit, not read noise
    MISSILE_CLUSTER_PX = 60         # matches this close to a missed one: banned too
    MISSILE_MISS_STREAK = 3         # this many misses in a row anywhere...
    MISSILE_PAUSE_SECONDS = 30.0    # ...no casting for this long
    MISSILE_CROWD_PX = 110          # another enemy this close: it would bite mid-cast
    MISSILE_MIN_MP = 8.0            # MP %; below this, melee only
    MISSILE_TURN_TAP = 0.06
    LURE_WAIT_SECONDS = 3.0         # after a hit, stand while it comes this long at most
    LURE_COMING_GRACE = 1.2         # not coming at us by then: walk to it instead
    LURE_COMING_MIN = 15            # px/s it closes the gap by (we stand): coming

    def _missile_settings(self, tracker):
        get = getattr(tracker, 'missile_settings', None)
        try:
            settings = get() if get is not None else None
        except Exception:
            settings = None
        return settings if settings and settings.get('enabled') else None

    def _missile_world(self, x, offset):
        """Minimap x of a spot OFFSET screen px from the character, or None."""
        if x is None:
            return None
        scale = self.__dict__.get('_px_per_minimap') or self.PX_PER_MINIMAP
        return x + offset / scale

    def _missile_banned(self, world, now):
        if world is None:
            return False
        scale = self.__dict__.get('_px_per_minimap') or self.PX_PER_MINIMAP
        radius = self.MISSILE_BAN_PX / scale
        bans = [b for b in self.__dict__.get('_missile_bans', []) if b[1] > now]
        self._missile_bans = bans
        return any(abs(spot - world) <= radius for spot, _ in bans)

    def _missile_stand(self, tracker, offset=None):
        self._release_mob_attack()
        self._set_auto_attack_movement(None)
        note = getattr(tracker, 'note_standing', None)
        if note is not None:
            note()
        if offset is not None:
            self._face(offset)

    def _judge_missile(self, cast, info, x, now, others=()):
        """After a cast: did it hit (an HP bar came up, or the bar shrank)?"""
        self._missile_cast = None
        if info is None or info['serial'] != cast['serial']:
            # The lock moved on. Something still standing where the missile
            # went was not killed by it: a miss (two portal matches taking
            # turns as the target got a missile each, forever). Nothing there:
            # it died.
            near = [o for o in others
                    if abs(o - cast['offset']) <= self.MISSILE_CLUSTER_PX]
            if not near:
                return None
            self._missile_missed(near[0], x, now, others)
            return False
        stats = self.__dict__.get('_missile_stats')
        if stats is None or stats[0] != cast['serial']:
            stats = [cast['serial'], 0, 0]
        health = info.get('health')
        if cast['health'] is None:
            hit = health is not None            # the first bar: it took the hit
        else:
            # Hurt already: bar reads wobble by a few %, so a clear drop, or
            # it coming at the character, counts as the hit
            hit = health is not None and (
                health < cast['health'] - self.MISSILE_BAR_DROP
                or info['closing'] >= self.LURE_COMING_MIN)
        if hit:
            stats[1] += 1
            self._missile_streak = 0
            self._missile_hold = (cast['serial'], now)
        elif info.get('real') or cast['health'] is not None or health is not None:
            # A proven enemy (HP bar, moved, bit us): out of reach, or a
            # wobbly bar read; never scenery. No more missiles at this one
            # after MISSILE_MISSES_PER_TARGET; walk to it the normal way.
            stats[2] += 1
        else:
            stats[2] += 1
            self._missile_missed(info['offset'], x, now, others)
        self._missile_stats = stats
        return hit

    def _missile_missed(self, offset, x, now, others):
        streak = self.__dict__.get('_missile_streak', 0) + 1
        self._missile_streak = streak
        # (a portal is 2-3 matches side by side: the whole cluster)
        spots = [offset] + [
            o for o in others
            if 2 < abs(o - offset) <= self.MISSILE_CLUSTER_PX]
        bans = self.__dict__.get('_missile_bans', [])
        for spot in spots:
            world = self._missile_world(x, spot)
            if world is not None:
                bans = bans + [(world, now + self.MISSILE_BAN_SECONDS)]
        self._missile_bans = bans
        note = ''
        if streak >= self.MISSILE_MISS_STREAK:
            self._missile_paused_until = now + self.MISSILE_PAUSE_SECONDS
            self._missile_streak = 0
            note = (f' {self.MISSILE_MISS_STREAK} misses in a row: melee only '
                    f'for {self.MISSILE_PAUSE_SECONDS:.0f} sec.')
        print(f'\n[~] Magic missile showed no HP bar on the target '
              f'({offset:+d} px): not casting at that spot again for '
              f'{self.MISSILE_BAN_SECONDS:.0f} sec.{note}')

    @staticmethod
    def _enemy_offsets(tracker):
        try:
            return list(tracker.enemy_offsets())
        except Exception:
            return []

    def _service_missile(self, tracker, target, x):
        """
        The ranged pull. True while it has the controls this tick (casting,
        or standing while a hit enemy comes); False hands the tick back to the
        melee flow, which also swings once the enemy arrives.
        """
        settings = self._missile_settings(tracker)
        if settings is None:
            self._missile_cast = None
            self._missile_hold = None
            return False
        now = time.monotonic()
        info = tracker.target_details() if target is not None else None
        cast = self.__dict__.get('_missile_cast')
        if cast is not None:
            elapsed = now - cast['at']
            in_reach = info is not None and info['attackable']
            if elapsed < self.MISSILE_CAST_SECONDS and not (
                    in_reach and elapsed >= self.MISSILE_RELEASE_SECONDS):
                self._missile_stand(tracker)    # mid-cast: don't walk, don't swing
                return True
            if now - cast['at'] >= self.MISSILE_JUDGE_SECONDS:
                self._judge_missile(cast, info, x, now, self._enemy_offsets(tracker))
            elif not in_reach:
                # Waiting to see the hit: stand (walking at it here was a
                # walk into a snail coming the other way)
                self._missile_stand(tracker, info['offset'] if info else None)
                return True
        if info is None or info['attackable']:
            self._missile_hold = None   # nothing to pull, or in swing range
            return False
        offset = info['offset']
        distance = abs(offset)
        offsets = self._enemy_offsets(tracker)
        recast = self._missile_recast_ok(info, distance, settings, offsets, x, now)
        if info['waiting'] and not recast:
            return False    # coming, too close for a cast: melee swings as it arrives
        hold = self.__dict__.get('_missile_hold')
        if hold is not None:
            elapsed = now - hold[1]
            coming = info['closing'] >= self.LURE_COMING_MIN
            if (hold[0] == info['serial'] and elapsed < self.LURE_WAIT_SECONDS
                    and (coming or elapsed < self.LURE_COMING_GRACE)):
                if not recast:
                    self._missile_stand(tracker, offset)    # hit: let it come
                    return True
            else:
                self._missile_hold = None
        if not recast:
            return False
        if not is_game_focused():
            return False
        side = 'left' if offset < 0 else 'right'
        self._missile_stand(tracker)
        if self.__dict__.get('_facing') != side:
            key_down(side)
            time.sleep(self.MISSILE_TURN_TAP)
            key_up(side)
            self._facing = side
            time.sleep(0.03)
        press(settings['key'], 1, down_time=0.08, up_time=0.05)
        self._attack_released_at = time.monotonic()
        self._missile_last = now
        self._missile_cast = {'at': now, 'serial': info['serial'], 'offset': offset,
                              'health': info.get('health')}
        stats = self.__dict__.get('_missile_stats')
        hits = stats[1] if stats is not None and stats[0] == info['serial'] else 0
        group = ' (group in range)' if self._missile_group(offsets, settings) else ''
        print(f'\n[~] Magic missile ({settings["key"].upper()}) at the enemy '
              f'{distance} px {side}{group}; hit {hits} so far. Melee when it arrives.')
        return True

    def _missile_group(self, offsets, settings):
        """True with MISSILE_GROUP_MIN or more enemies within missile range."""
        return sum(1 for o in offsets if abs(o) <= settings['range']) >= self.MISSILE_GROUP_MIN

    def _missile_recast_ok(self, info, distance, settings, offsets, x, now):
        """May a missile go at the locked target now? (Same ground, in range,
        still far enough when the cast ends, hits left for it, nothing at the
        character's side, MP, not banned or paused, cooled down.)"""
        if not info['on_level'] or info['airborne']:
            return False                # it flies straight: same ground only
        if not self.MISSILE_MIN_RANGE <= distance <= settings['range']:
            return False
        # Coming at us: only while it will still be well away when the cast
        # ends; a missile now beats standing there waiting for its bite
        arrival = distance - max(info.get('closing') or 0.0, 0.0) * self.MISSILE_CAST_SECONDS
        if arrival < self.MISSILE_KITE_MIN:
            return False
        if now < self.__dict__.get('_missile_paused_until', float('-inf')):
            return False
        if now - self.__dict__.get('_missile_last', float('-inf')) < self.MISSILE_COOLDOWN:
            return False
        limit = (self.MISSILE_GROUP_HITS if self._missile_group(offsets, settings)
                 else self.MISSILE_HITS_PER_TARGET)
        stats = self.__dict__.get('_missile_stats')
        if stats is not None and stats[0] == info['serial'] and (
                stats[1] >= limit or stats[2] >= self.MISSILE_MISSES_PER_TARGET):
            return False
        if self._missile_banned(self._missile_world(x, info['offset']), now):
            return False
        if any(abs(o) < self.MISSILE_CROWD_PX for o in offsets):
            return False                # one is next to us: it would bite mid-cast
        mp = self._percent('mana')
        if mp is not None and mp < self.MISSILE_MIN_MP:
            return False
        return True

    def _jumper_farming(self, tracker):
        check = getattr(tracker, 'level_farming_enabled', None)
        try:
            return bool(check and check())
        except Exception:
            return False

    def _set_mob_attack_keys(self, keys):
        keys = tuple(keys)
        if keys and keys[0] in ('left', 'right'):
            self._facing = keys[0]
        if keys == self._mob_attack_keys:
            return
        self._release_mob_attack()
        for key in keys:
            key_down(key)
        self._mob_attack_keys = keys

    # strike-issue-2.mp4 (Swordsman, snails): with the arrow held next to the
    # attack key the character walked toward the snail between swings - into
    # it - and most swings whiffed. Slow-walker maps now face with an arrow tap
    # and hold the attack key alone, repeated like a real held key (AttackHand)
    # so the game keeps swinging in place. Pig maps keep arrow + attack
    # (spawn_issue.mp4: there the attack key alone, sent once, swung once).
    HUMAN_ATTACK_HOLD = True

    def _human_hold(self, tracker=None):
        if not self.HUMAN_ATTACK_HOLD:
            return False
        tracker = tracker or getattr(getattr(config, 'capture', None), 'mob_tracker', None)
        return not (tracker is not None and self._jumper_farming(tracker))

    def _attack_hand(self):
        hand = self.__dict__.get('_attack_hand_obj')
        if hand is None:
            # (looked up at each press, so tests that swap the key functions
            # never reach the real keyboard)
            hand = self._attack_hand_obj = AttackHand(
                lambda key: key_down(key), lambda key: key_up(key),
                lambda key: key_repeat(key), lease=self.HOLD_LEASE)
        return hand

    # attack-loop.mp4 (2026-11-15): by the right-edge portal the character
    # swung for good. CTRL went down once, AttackHand kept repeating it, the
    # tracker was not even calling anything attackable, and the console said
    # nothing (repeats aren't logged). Now: the hold lasts HOLD_LEASE unless
    # the bot asks for it again on its next pass (any path that forgets to
    # release, or a stalled bot thread, lets go by itself); a hold that runs
    # HOLD_MAX_SECONDS with no HP bar seen anywhere near is given up; and a
    # long hold says so in the log every HOLD_LOG_SECONDS.
    HOLD_LEASE = 1.2
    HOLD_MAX_SECONDS = 12.0
    HOLD_LOG_SECONDS = 5.0
    HOLD_GIVE_UP_PAUSE = 1.5        # s no attack is held after giving up
    HOLD_SESSION_GAP = 1.0          # s without swinging that ends a session

    def _watch_hold(self, direction, tracker):
        """Called on every pass that keeps the attack held. False: give up."""
        now = time.monotonic()
        started = self.__dict__.get('_hold_started')
        # (one swinging session, turns included: a portal matched on both
        # sides had the bot turning back and forth, each turn a release)
        if started is None or now - self.__dict__.get('_hold_pass', now) > self.HOLD_SESSION_GAP:
            started = self._hold_started = now
            self._hold_logged = now
        self._hold_pass = now
        held = now - started
        if now - self.__dict__.get('_hold_logged', now) >= self.HOLD_LOG_SECONDS:
            self._hold_logged = now
            status = ''
            try:
                status = tracker.status()[:110] if tracker is not None else ''
            except Exception:
                pass
            print(f'\n[~] Attack held {direction} for {held:.0f} s. Tracker: {status}')
        since_bar = float('inf')
        try:
            since_bar = tracker.seconds_since_bar() if tracker is not None else since_bar
        except Exception:
            pass
        # (slow-walker maps only: the bar clock is kept by their tracker path)
        if (held >= self.HOLD_MAX_SECONDS and since_bar >= self.HOLD_MAX_SECONDS
                and self._human_hold(tracker)):
            print(f'\n[!] Swung {direction} for {held:.0f} s and no enemy HP bar showed: '
                  'nothing is being hit (a portal or decor?). Letting go and moving on.')
            try:
                tracker.give_up_here()
            except Exception:
                pass
            self._hold_paused_until = now + self.HOLD_GIVE_UP_PAUSE
            self._hold_started = None
            return False
        return True

    def _hold_attack(self, direction, attack_key, tracker=None, stationary=False):
        """Swing toward DIRECTION until released: arrow + attack held (pig
        maps), or face with an arrow tap and hold the attack key alone. At a
        wall, use the stationary form even for pigs instead of walking into it."""
        if time.monotonic() < self.__dict__.get('_hold_paused_until', float('-inf')):
            self._release_mob_attack()      # just gave up swinging here
            return
        if not self._watch_hold(direction, tracker):
            self._release_mob_attack()
            return
        if not stationary and not self._human_hold(tracker):
            # (Arrow + attack held together: with the attack key alone held, the
            # game swung once and stopped - spawn_issue.mp4)
            self._set_mob_attack_keys((direction, attack_key))
            return
        if self.__dict__.get('_facing') != direction:
            if self._mob_attack_keys:
                self._release_mob_attack()
                time.sleep(self.GROUP_Q_SETTLE)     # mid-swing the game ignores the turn
            key_down(direction)
            time.sleep(self.FACE_TAP_SECONDS)       # a tap turns without a step
            key_up(direction)
            self._facing = direction
        elif self._fix_facing(direction, tracker):
            pass                    # seen facing the other way: turned again
        if self._mob_attack_keys != (attack_key,):
            self._release_mob_attack()
            self._mob_attack_keys = (attack_key,)
        self._attack_hand().hold(attack_key)        # renews the lease each pass
        self._service_step_in(direction, tracker)

    def _service_step_in(self, direction, tracker):
        """Sweep maps, attack held: now and then the arrow goes down with it
        for a moment, to close the gap each hit's knock-back opens."""
        now = time.monotonic()
        step = self.__dict__.get('_step_in')
        if step is not None:
            if now >= step['until'] or step['direction'] != direction:
                self._end_step_in()
            return
        if not self._sweeping(tracker):
            return
        details = getattr(tracker, 'target_details', None)
        info = details() if details is not None else None
        if info is None or now - info['time'] > 0.35 or not info.get('on_level'):
            return                  # hidden or stale: no telling how far it is
        offset = info['offset']
        if (offset < 0) != (direction == 'left'):
            return
        distance = abs(offset)
        if distance <= self.STEP_IN_TO_PX + 4:
            return
        if info.get('closing', 0.0) >= self.STEP_IN_COMING:
            return                  # it is coming by itself
        last_step = self.__dict__.get('_step_in_at', float('-inf'))
        since = max(last_step, self.__dict__.get('_hold_started') or now)
        if now - last_step < self.STEP_IN_GAP:
            return
        if distance < self.STEP_IN_NOW_PX and now - since < self.STEP_IN_AFTER:
            return
        seconds = max(self.STEP_IN_MIN, min(
            self.STEP_IN_MAX, (distance - self.STEP_IN_TO_PX) / self.WALK_SPEED_PX))
        key_down(direction)
        self._step_in = {'direction': direction, 'until': now + seconds}

    def _expire_step_in(self):
        step = self.__dict__.get('_step_in')
        if step is not None and time.monotonic() >= step['until']:
            self._end_step_in()

    def _end_step_in(self):
        step = self.__dict__.get('_step_in')
        if step is None:
            return
        self._step_in = None
        self._step_in_at = time.monotonic()
        if step['direction'] != getattr(self, '_auto_attack_movement', None):
            key_up(step['direction'])

    def _release_mob_attack(self):
        self._end_step_in()
        hand = self.__dict__.get('_attack_hand_obj')
        released = hand.key if hand is not None else None
        if released is not None or getattr(self, '_mob_attack_keys', ()):
            self._attack_released_at = time.monotonic()
        if released is not None:
            hand.release()
        # The attack's arrow may also be the walk arrow: keep that one held
        moving = getattr(self, '_auto_attack_movement', None)
        for key in reversed(getattr(self, "_mob_attack_keys", ())):
            if key != moving and key != released:
                key_up(key)
        self._mob_attack_keys = ()

    def _jump_toward(self, direction, jump_key):
        """Jump while pushing toward `direction`. A held attack key (ctrl)
        keeps the character mid-swing and the game drops the jump, so it is
        released first and the arrow pressed again before the jump."""
        self._release_mob_attack()
        self._set_auto_attack_movement(direction)
        if direction is not None:
            key_down(direction)     # make sure the arrow really is down
        time.sleep(0.06)            # let the last swing end
        press(jump_key, 1, down_time=0.12, up_time=0.05)

    @utils.run_if_enabled
    def _solve_rune(self, model):
        """
        Moves to the position of the rune and solves the arrow-key puzzle.
        :param model:   The TensorFlow model to classify with.
        :param sct:     The mss instance object with which to take screenshots.
        :return:        None
        """

        move = self.command_book['move']
        move(*self.rune_pos).execute()
        adjust = self.command_book['adjust']
        adjust(*self.rune_pos).execute()
        time.sleep(0.2)
        press(self.config['Interact'], 1, down_time=0.2)        # Inherited from Configurable

        print('\nSolving rune:')
        inferences = []
        for _ in range(15):
            frame = config.capture.frame
            solution = detection.merge_detection(model, frame)
            if solution:
                print(', '.join(solution))
                if solution in inferences:
                    print('Solution found, entering result')
                    for arrow in solution:
                        press(arrow, 1, down_time=0.1)
                    time.sleep(1)
                    for _ in range(3):
                        time.sleep(0.3)
                        frame = config.capture.frame
                        rune_buff = utils.multi_match(frame[:frame.shape[0] // 8, :],
                                                      RUNE_BUFF_TEMPLATE,
                                                      threshold=0.9)
                        if rune_buff:
                            rune_buff_pos = min(rune_buff, key=lambda p: p[0])
                            target = (
                                round(rune_buff_pos[0] + config.capture.window['left']),
                                round(rune_buff_pos[1] + config.capture.window['top'])
                            )
                            click(target, button='right')
                    self.rune_active = False
                    break
                elif len(solution) == 4:
                    inferences.append(solution)

    def load_commands(self, file):
        try:
            self.command_book = CommandBook(file)
            config.gui.settings.update_class_bindings()
        except ValueError:
            pass    # TODO: UI warning popup, say check cmd for errors
        #
        # utils.print_separator()
        # print(f"[~] Loading command book '{basename(file)}':")
        #
        # ext = splitext(file)[1]
        # if ext != '.py':
        #     print(f" !  '{ext}' is not a supported file extension.")
        #     return False
        #
        # new_step = components.step
        # new_cb = {}
        # for c in (components.Wait, components.Walk, components.Fall):
        #     new_cb[c.__name__.lower()] = c
        #
        # # Import the desired command book file
        # module_name = splitext(basename(file))[0]
        # target = '.'.join(['resources', 'command_books', module_name])
        # try:
        #     module = importlib.import_module(target)
        #     module = importlib.reload(module)
        # except ImportError:     # Display errors in the target Command Book
        #     print(' !  Errors during compilation:\n')
        #     for line in traceback.format_exc().split('\n'):
        #         line = line.rstrip()
        #         if line:
        #             print(' ' * 4 + line)
        #     print(f"\n !  Command book '{module_name}' was not loaded")
        #     return
        #
        # # Check if the 'step' function has been implemented
        # step_found = False
        # for name, func in inspect.getmembers(module, inspect.isfunction):
        #     if name.lower() == 'step':
        #         step_found = True
        #         new_step = func
        #
        # # Populate the new command book
        # for name, command in inspect.getmembers(module, inspect.isclass):
        #     new_cb[name.lower()] = command
        #
        # # Check if required commands have been implemented and overridden
        # required_found = True
        # for command in [components.Buff]:
        #     name = command.__name__.lower()
        #     if name not in new_cb:
        #         required_found = False
        #         new_cb[name] = command
        #         print(f" !  Error: Must implement required command '{name}'.")
        #
        # # Look for overridden movement commands
        # movement_found = True
        # for command in (components.Move, components.Adjust):
        #     name = command.__name__.lower()
        #     if name not in new_cb:
        #         movement_found = False
        #         new_cb[name] = command
        #
        # if not step_found and not movement_found:
        #     print(f" !  Error: Must either implement both 'Move' and 'Adjust' commands, "
        #           f"or the function 'step'")
        # if required_found and (step_found or movement_found):
        #     self.module_name = module_name
        #     self.command_book = new_cb
        #     self.buff = new_cb['buff']()
        #     components.step = new_step
        #     config.gui.menu.file.enable_routine_state()
        #     config.gui.view.status.set_cb(basename(file))
        #     config.routine.clear()
        #     print(f" ~  Successfully loaded command book '{module_name}'")
        # else:
        #     print(f" !  Command book '{module_name}' was not loaded")

    def update_submodules(self, force=False):
        """
        Pulls updates from the submodule repositories. If FORCE is True,
        rebuilds submodules by overwriting all local changes.
        """

        utils.print_separator()
        print('[~] Retrieving latest submodules:')
        self.submodules = []
        repo = git.Repo.init()
        with open('.gitmodules', 'r') as file:
            lines = file.readlines()
            i = 0
            while i < len(lines):
                if lines[i].startswith('[') and i < len(lines) - 2:
                    path = lines[i + 1].split('=')[1].strip()
                    url = lines[i + 2].split('=')[1].strip()
                    self.submodules.append(path)
                    try:
                        repo.git.clone(url, path)       # First time loading submodule
                        print(f" -  Initialized submodule '{path}'")
                    except git.exc.GitCommandError:
                        sub_repo = git.Repo(path)
                        if not force:
                            sub_repo.git.stash()        # Save modified content
                        sub_repo.git.fetch('origin', 'main')
                        sub_repo.git.reset('--hard', 'FETCH_HEAD')
                        if not force:
                            try:                # Restore modified content
                                sub_repo.git.checkout('stash', '--', '.')
                                print(f" -  Updated submodule '{path}', restored local changes")
                            except git.exc.GitCommandError:
                                print(f" -  Updated submodule '{path}'")
                        else:
                            print(f" -  Rebuilt submodule '{path}'")
                        sub_repo.git.stash('clear')
                    i += 3
                else:
                    i += 1
