"""A module for tracking useful in-game information."""

import gc
import time
import cv2
import threading
import ctypes
import mss
import mss.windows
import numpy as np
from src.common import config, utils
from ctypes import wintypes
from src.modules.health import HealthPotionMonitor, ManaPotionMonitor
from src.modules.loot_tracker import LootTracker
from src.modules.mob_tracker import MobTracker
from src.modules.minimap_finder import find_minimap_area
from src.modules.watchdog import watchdog
user32 = ctypes.windll.user32
user32.SetProcessDPIAware()


# The distance between the top of the minimap and the top of the screen
MINIMAP_TOP_BORDER = 5

# The thickness of the other three borders of the minimap
MINIMAP_BOTTOM_BORDER = 9

# Minimum normalized template score accepted for each minimap corner
MINIMAP_MATCH_THRESHOLD = 0.7
MINIMAP_TEMPLATE_SCALES = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5)

# Offset in pixels to adjust for windowed mode
WINDOWED_OFFSET_TOP = 36
WINDOWED_OFFSET_LEFT = 10

# The top-left and bottom-right corners of the minimap
MM_TL_TEMPLATE = cv2.imread('assets/minimap_tl_template.png', 0)
MM_BR_TEMPLATE = cv2.imread('assets/minimap_br_template.png', 0)

MMT_HEIGHT = max(MM_TL_TEMPLATE.shape[0], MM_BR_TEMPLATE.shape[0])
MMT_WIDTH = max(MM_TL_TEMPLATE.shape[1], MM_BR_TEMPLATE.shape[1])

# The player's symbol on the minimap
PLAYER_TEMPLATE = cv2.imread('assets/player_template.png', 0)
PT_HEIGHT, PT_WIDTH = PLAYER_TEMPLATE.shape

# bug-2.mp4: at the map's very end the marker sits on the minimap's border,
# half outside the cropped map area. It matched at 0.88 in a crop a few px
# wider and not at all in the map area, so the bot lost its position there.
MINIMAP_EDGE_PAD = 8
PLAYER_MATCH_THRESHOLD = 0.8

# The minimap is found on its own from its white outline (minimap_finder).
# It keeps looking; after this long without finding it the console says so.
# The draw-it-by-hand window only opens when asked for (Start/stop, reload).
WARN_NOT_FOUND_AFTER = 8.0
# Asking to recalibrate again this soon after an automatic find means the
# automatic area is wrong: the user then draws it by hand.
MANUAL_OVERRIDE_WINDOW = 15.0
# While running: how often the game window's place and a lost marker are checked,
# and how long the marker must be gone before the minimap is looked for again
# (a new map has a minimap of another size).
RECHECK_SECONDS = 2.0
MARKER_LOST_SECONDS = 3.0
# The old corner pictures are only a fallback for when the white outline can't
# be seen. A mouse pointer resting on the minimap hides the outline, and the
# corners then matched the inventory window beside it: 262 px wide for a 211 px
# map (run-20261118-170231, fall_recovery.mp4), so every position read ~16%
# short and the bot walked off the pig island. So the outline gets this long
# first, and while running on corners it keeps being looked for and replaces them.
CORNER_FALLBACK_AFTER = 3.0
OUTLINE_SLACK = 3           # px an outline may differ by before it counts as moved


def locate_player(frame, mm_tl, mm_br, template=None, pad=MINIMAP_EDGE_PAD):
    """
    The player's marker as (x, y) relative to the minimap area (x in 0..1,
    y scaled like utils.convert_to_relative), or None. The map area is
    searched first; failing that, a crop PAD px wider on each side, with the
    result clamped onto the map's edge.
    """
    template = PLAYER_TEMPLATE if template is None else template
    minimap = frame[mm_tl[1]:mm_br[1], mm_tl[0]:mm_br[0]]
    found = utils.multi_match(minimap, template, threshold=PLAYER_MATCH_THRESHOLD)
    if found:
        return utils.convert_to_relative(found[0], minimap)
    x1, y1 = max(0, mm_tl[0] - pad), max(0, mm_tl[1] - pad)
    x2, y2 = min(frame.shape[1], mm_br[0] + pad), min(frame.shape[0], mm_br[1] + pad)
    wide = frame[y1:y2, x1:x2]
    if wide.shape[0] < template.shape[0] or wide.shape[1] < template.shape[1]:
        return None
    gray = cv2.cvtColor(wide, cv2.COLOR_BGR2GRAY) if wide.ndim == 3 else wide
    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    _, best, _, top_left = cv2.minMaxLoc(result)
    if best < PLAYER_MATCH_THRESHOLD:
        return None
    width, height = minimap.shape[1], minimap.shape[0]
    cx = top_left[0] + template.shape[1] / 2 + x1 - mm_tl[0]
    cy = top_left[1] + template.shape[0] / 2 + y1 - mm_tl[1]
    point = (min(max(cx, 0), width - 1), min(max(cy, 0), height - 1))
    return utils.convert_to_relative(point, minimap)


class Capture:
    """
    A class that tracks player position and various in-game events. It constantly updates
    the config module with information regarding these events. It also annotates and
    displays the minimap in a pop-up window.
    """

    def __init__(self):
        """Initializes this Capture object's main thread."""

        config.capture = self

        self.frame = None
        self.health = HealthPotionMonitor()
        self.mana = ManaPotionMonitor()
        self.mob_tracker = MobTracker()
        self.loot_tracker = LootTracker()
        self.minimap = {}
        self.minimap_ratio = 1
        self.minimap_sample = None
        self.sct = None
        self.window = {
            'left': 0,
            'top': 0,
            'width': 1366,
            'height': 768
        }

        self.ready = False
        self.calibrated = False
        self.manual_calibration_requested = threading.Event()
        self.found_automatically_at = float('-inf')
        self.selection_requested = threading.Event()
        self.selection_completed = threading.Event()
        self.selection_frame = None
        self.selection_result = None
        self._generation = 0
        self.thread = self._new_thread()

    STALL_SECONDS = 10      # no frame for this long: the watchdog restarts capture

    def _new_thread(self):
        self._generation += 1
        thread = threading.Thread(target=self._main, args=(self._generation,),
                                  name='capture', daemon=True)
        return thread

    def start(self):
        """Starts this Capture's thread."""

        print('\n[~] Started video capture')
        watchdog.register('capture', self.STALL_SECONDS, restart=self.restart)
        self.thread.start()

    def restart(self):
        """A fresh capture thread (the watchdog's hook). The stuck one sees its
        generation is old and exits if it ever wakes up."""
        self.calibrated = False
        self.thread = self._new_thread()
        self.thread.start()

    def _stale(self, generation):
        return generation != self._generation

    def request_minimap_selection(self, frame):
        self.selection_frame = frame.copy()
        self.selection_result = None
        self.selection_completed.clear()
        self.selection_requested.set()
        watchdog.pause('capture')      # waiting for the user is not a stall
        try:
            completed = self.selection_completed.wait(timeout=120)
        finally:
            watchdog.resume('capture')
        if not completed:
            print('\n[!] Manual minimap selection timed out.')
            return None
        return self.selection_result

    def complete_minimap_selection(self, selection):
        self.selection_result = selection
        self.selection_completed.set()

    def find_outline(self, frame):
        """The minimap's map area found from its white outline, with the
        player's marker seen inside it; None when it is not on screen."""
        try:
            area = find_minimap_area(
                frame, accept=lambda tl, br: locate_player(frame, tl, br) is not None,
                strict=True)
        except Exception as error:
            self._report_error('minimap finder', error)
            return None
        if area is None:
            return None
        (x1, y1), (x2, y2) = area
        if x2 - x1 < PT_WIDTH or y2 - y1 < PT_HEIGHT:
            return None
        return area

    @staticmethod
    def _game_restarting():
        try:
            from src.modules import game_restart
            return game_restart.RUNNER.busy
        except Exception:
            return False

    @staticmethod
    def game_window_rect():
        handle = user32.FindWindowW(None, 'MapleStory')
        rect = wintypes.RECT()
        user32.GetWindowRect(handle, ctypes.pointer(rect))
        return handle, (rect.left, rect.top, rect.right, rect.bottom)

    ERROR_REPORT_SECONDS = 30       # one console line per failing part this often

    def _process_modules(self, frame):
        """The per-frame detectors, each guarded: one failing (e.g. out of
        memory) skips only its own work for this frame."""
        steps = (
            ('health', lambda: self.health.process_frame(frame)),
            ('mana', lambda: self.mana.process_frame(frame, self.health.last_region())),
            ('mob tracker', lambda: self.mob_tracker.process_frame(frame, config.enabled)),
            ('loot tracker', lambda: self.loot_tracker.process_frame(frame, config.enabled)),
        )
        for name, step in steps:
            try:
                step()
            except Exception as error:
                self._report_error(name, error)
                if isinstance(error, MemoryError) or 'Insufficient memory' in str(error):
                    gc.collect()
                    time.sleep(0.2)     # let memory free up instead of failing every frame

    def _report_error(self, name, error):
        reported = self.__dict__.setdefault('_error_reports', {})
        now = time.monotonic()
        last, count = reported.get(name, (0.0, 0))
        count += 1
        if now - last >= self.ERROR_REPORT_SECONDS:
            print('\n' f'[!] Capture: {name} failed ({count}x in the last '
                  f'{self.ERROR_REPORT_SECONDS}s): {str(error).strip()[:200]}')
            reported[name] = (now, 0)
        else:
            reported[name] = (last, count)

    def _main(self, generation=None):
        """Runs the capture loop for good: an error restarts it after a pause
        instead of ending the thread (the bot would lose the minimap position
        and patrol blind - left/right on a timer - for as long as it runs)."""
        backoff = 1.0
        generation = self._generation if generation is None else generation
        while not self._stale(generation):
            started = time.monotonic()
            try:
                self._capture_loop(generation)
            except Exception as error:
                self.calibrated = False
                self._report_error('capture loop', error)
                gc.collect()
                if time.monotonic() - started > 60:
                    backoff = 1.0       # it had been running fine
                waited = 0.0
                while waited < backoff:     # an error pause is not a stall
                    watchdog.heartbeat('capture')
                    time.sleep(0.5)
                    waited += 0.5
                backoff = min(backoff * 2, 30.0)

    def _capture_loop(self, generation):
        """Constantly monitors the player's position and in-game events."""

        mss.windows.CAPTUREBLT = 0
        last_calibration_warning = None
        failing_since = None
        outline_missing_since = None
        while not self._stale(generation):
            watchdog.heartbeat('capture')
            # Calibrate screen capture
            handle, rect = self.game_window_rect()
            window_rect = rect
            covered = self._window_covered(handle, rect)
            rect = tuple(max(0, x) for x in rect)

            self.window['left'] = rect[0]
            self.window['top'] = rect[1]
            self.window['width'] = max(rect[2] - rect[0], MMT_WIDTH)
            self.window['height'] = max(rect[3] - rect[1], MMT_HEIGHT)

            # Calibrate by finding the top-left and bottom-right corners of the minimap
            with mss.mss() as self.sct:
                self.frame = self.screenshot()
            if self.frame is None:
                continue
            if not self.ready:
                self.ready = True
            self._process_modules(self.frame)

            manual_requested = self.manual_calibration_requested.is_set()
            # A second request right after an automatic find: draw it by hand.
            by_hand = manual_requested and (
                time.monotonic() - self.found_automatically_at < MANUAL_OVERRIDE_WINDOW)
            outline = None if by_hand else self.find_outline(self.frame)
            if outline is None:
                outline_missing_since = outline_missing_since or time.monotonic()
            else:
                outline_missing_since = None
            manual_calibration = False
            if outline is not None:
                automatic_match_failed = False
                manual_requested = False
            else:
                tl, _, tl_score, _ = utils.multi_scale_match(
                    self.frame, MM_TL_TEMPLATE, MINIMAP_TEMPLATE_SCALES
                )
                _, br, br_score, _ = utils.multi_scale_match(
                    self.frame, MM_BR_TEMPLATE, MINIMAP_TEMPLATE_SCALES
                )
                automatic_match_failed = (
                    min(tl_score, br_score) < MINIMAP_MATCH_THRESHOLD
                    or br[0] <= tl[0]
                    or br[1] <= tl[1]
                )
                if not automatic_match_failed:
                    # The corner pictures are from another game version and can
                    # match scenery: only believed with the player's marker inside.
                    corner_tl = (tl[0] + MINIMAP_BOTTOM_BORDER, tl[1] + MINIMAP_TOP_BORDER)
                    corner_br = (
                        max(corner_tl[0] + PT_WIDTH, br[0] - MINIMAP_BOTTOM_BORDER),
                        max(corner_tl[1] + PT_HEIGHT, br[1] - MINIMAP_BOTTOM_BORDER)
                    )
                    automatic_match_failed = locate_player(
                        self.frame, corner_tl, corner_br) is None
            if not automatic_match_failed or covered or self._game_restarting():
                failing_since = None        # nothing to find yet is not a failure
            elif failing_since is None:
                failing_since = time.monotonic()
            if automatic_match_failed and covered:
                # The screenshot shows whatever is on top of the game, so neither
                # the automatic match nor a manual selection can work yet.
                if last_calibration_warning != 'covered':
                    print(
                        '\n[!] The MapleStory window is missing, minimised or covered '
                        'by another window, so the minimap cannot be seen. Bring the '
                        'game to the front; calibration continues automatically.'
                    )
                    last_calibration_warning = 'covered'
                time.sleep(0.5)
                continue
            if (automatic_match_failed and not manual_requested
                    and failing_since is not None
                    and time.monotonic() - failing_since >= WARN_NOT_FOUND_AFTER
                    and last_calibration_warning != 'not found'):
                print(
                    '\n[!] The minimap has not been found yet. Keep it open on the '
                    'map view (not collapsed) with nothing over it, the mouse '
                    'pointer included; looking continues on its own. To draw it by '
                    'hand instead, press Start/stop.'
                )
                last_calibration_warning = 'not found'
            if manual_requested:
                self.manual_calibration_requested.clear()
                print(
                    '[~] Select the minimap map area in the app window '
                    '(include the player marker, exclude the title bar), '
                    'then press Enter. Press Cancel to retry automatically.'
                )
                selection = self.request_minimap_selection(self.frame)
                if selection is None:
                    time.sleep(0.5)
                    continue
                mm_tl, mm_br = selection
                manual_calibration = True
            elif automatic_match_failed:
                time.sleep(0.5)
                continue
            elif outline is not None:
                mm_tl, mm_br = outline
            elif time.monotonic() - outline_missing_since < CORNER_FALLBACK_AFTER:
                time.sleep(0.5)     # give the outline a moment (a passing mouse pointer)
                continue
            else:
                mm_tl = (
                    tl[0] + MINIMAP_BOTTOM_BORDER,
                    tl[1] + MINIMAP_TOP_BORDER
                )
                mm_br = (
                    max(mm_tl[0] + PT_WIDTH, br[0] - MINIMAP_BOTTOM_BORDER),
                    max(mm_tl[1] + PT_HEIGHT, br[1] - MINIMAP_BOTTOM_BORDER)
                )

            if mm_br[0] <= mm_tl[0] or mm_br[1] <= mm_tl[1]:
                warning_key = 'bounds'
                warning = '[!] Detected minimap bounds are invalid; recalibrating.'
                if warning_key != last_calibration_warning:
                    print(f'\n{warning}')
                    last_calibration_warning = warning_key
                time.sleep(0.5)
                continue

            by_corners = not manual_calibration and outline is None
            if manual_calibration:
                print('\n[~] Minimap calibrated using the selected area.')
            elif by_corners:
                print(f'\n[~] Minimap found from its corner pictures (outline not visible): '
                      f'{mm_br[0] - mm_tl[0]}x{mm_br[1] - mm_tl[1]} px at {mm_tl}. '
                      'Still looking for its outline, which is more exact.')
            elif outline is not None:
                self.found_automatically_at = time.monotonic()
                print(f'\n[~] Minimap found automatically: {mm_br[0] - mm_tl[0]}x'
                      f'{mm_br[1] - mm_tl[1]} px at ({mm_tl[0]}, {mm_tl[1]}) in the game '
                      'window. (Wrong area? Recalibrate twice within '
                      f'{MANUAL_OVERRIDE_WINDOW:.0f} s to draw it by hand.)')
            self.manual_calibration_requested.clear()
            self.minimap_ratio = (mm_br[0] - mm_tl[0]) / (mm_br[1] - mm_tl[1])
            self.minimap_sample = self.frame[mm_tl[1]:mm_br[1], mm_tl[0]:mm_br[0]]
            last_calibration_warning = None
            self.calibrated = True
            last_recheck = last_seen = time.monotonic()

            with mss.mss() as self.sct:
                while not self._stale(generation):
                    watchdog.heartbeat('capture')
                    if not self.calibrated:
                        break
                    now = time.monotonic()
                    if now - last_recheck >= RECHECK_SECONDS:
                        last_recheck = now
                        if self._needs_recalibration(
                                window_rect, (mm_tl, mm_br), manual_calibration,
                                now - last_seen >= MARKER_LOST_SECONDS, by_corners):
                            self.calibrated = False
                            break

                    # Take screenshot
                    self.frame = self.screenshot()
                    if self.frame is None:
                        continue
                    self._process_modules(self.frame)

                    # Crop the frame to only show the minimap
                    minimap = self.frame[mm_tl[1]:mm_br[1], mm_tl[0]:mm_br[0]]

                    # Determine the player's position
                    player = locate_player(self.frame, mm_tl, mm_br)
                    if player is not None:
                        config.player_pos = player
                        config.player_pos_time = last_seen = time.monotonic()

                    # Package display information to be polled by GUI
                    self.minimap = {
                        'minimap': minimap,
                        'rune_active': config.bot.rune_active,
                        'rune_pos': config.bot.rune_pos,
                        'path': config.path,
                        'player_pos': config.player_pos
                    }

                    time.sleep(0.001)

    def _needs_recalibration(self, window_rect, area, drawn_by_hand, marker_lost,
                             by_corners=False):
        """True when the minimap has to be found again: the game window moved,
        was resized or is a new one (the game was restarted), or the player's
        marker is gone and the minimap now sits elsewhere or has another size
        (a new map), or the area came from the corner pictures and the outline
        now shows a different one. An area drawn by hand is only dropped for a
        window change."""
        try:
            handle, rect = self.game_window_rect()
        except Exception:
            return False
        if rect != window_rect:
            if handle:
                print('\n[~] The game window moved or is a new one; finding the minimap again.')
            return True
        if drawn_by_hand or self.frame is None:
            return False
        # Checked even with the marker in sight: a minimap that shrank (its -/+
        # buttons, a new map) can still hold the marker inside the old, larger
        # area, and every position read from it is then off.
        outline = self.find_outline(self.frame)
        differs = outline is not None and max(
            abs(a - b) for p, q in zip(outline, area) for a, b in zip(p, q)) > OUTLINE_SLACK
        if differs:
            if by_corners:
                why = 'its outline is visible now and differs from the corner-picture guess'
            elif marker_lost:
                why = 'a new map?'
            else:
                why = f'now {outline[1][0] - outline[0][0]}x{outline[1][1] - outline[0][1]} px'
            print(f'\n[~] The minimap changed ({why}); finding it again.')
            return True
        return False

    @staticmethod
    def _window_covered(handle, rect):
        """
        True when the game window is missing or another window sits on top of its
        top-left area (where the minimap is). Screen capture would then show that
        other window instead of the game.
        """

        if not handle or user32.IsIconic(handle):
            return True
        left, top, right, bottom = rect
        width, height = right - left, bottom - top
        if width <= 0 or height <= 0:
            return True
        user32.WindowFromPoint.argtypes = [wintypes.POINT]
        user32.WindowFromPoint.restype = wintypes.HWND
        user32.GetAncestor.argtypes = [wintypes.HWND, ctypes.c_uint]
        user32.GetAncestor.restype = wintypes.HWND
        for fraction_x, fraction_y in ((0.03, 0.08), (0.12, 0.08), (0.03, 0.22), (0.12, 0.22)):
            point = wintypes.POINT(
                int(left + width * fraction_x), int(top + height * fraction_y)
            )
            if point.x < 0 or point.y < 0:
                return True         # that part of the window is off the screen
            owner = user32.GetAncestor(user32.WindowFromPoint(point), 2)   # GA_ROOT
            if owner != handle:
                return True
        return False

    def screenshot(self, delay=1):
        try:
            return np.array(self.sct.grab(self.window))
        except mss.exception.ScreenShotError:
            print(f'\n[!] Error while taking screenshot, retrying in {delay} second'
                  + ('s' if delay != 1 else ''))
            time.sleep(delay)
