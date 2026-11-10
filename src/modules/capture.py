"""A module for tracking useful in-game information."""

import time
import cv2
import threading
import ctypes
import mss
import mss.windows
import numpy as np
from src.common import config, utils
from ctypes import wintypes
from src.modules.health import HealthPotionMonitor
from src.modules.loot_tracker import LootTracker
from src.modules.mob_tracker import MobTracker
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
        self.selection_requested = threading.Event()
        self.selection_completed = threading.Event()
        self.selection_frame = None
        self.selection_result = None
        self.thread = threading.Thread(target=self._main)
        self.thread.daemon = True

    def start(self):
        """Starts this Capture's thread."""

        print('\n[~] Started video capture')
        self.thread.start()

    def request_minimap_selection(self, frame):
        self.selection_frame = frame.copy()
        self.selection_result = None
        self.selection_completed.clear()
        self.selection_requested.set()
        if not self.selection_completed.wait(timeout=120):
            print('\n[!] Manual minimap selection timed out.')
            return None
        return self.selection_result

    def complete_minimap_selection(self, selection):
        self.selection_result = selection
        self.selection_completed.set()

    def _main(self):
        """Constantly monitors the player's position and in-game events."""

        mss.windows.CAPTUREBLT = 0
        last_calibration_warning = None
        manual_calibration_prompted = False
        while True:
            # Calibrate screen capture
            handle = user32.FindWindowW(None, 'MapleStory')
            rect = wintypes.RECT()
            user32.GetWindowRect(handle, ctypes.pointer(rect))
            rect = (rect.left, rect.top, rect.right, rect.bottom)
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
            self.health.process_frame(self.frame)
            self.mob_tracker.process_frame(self.frame, config.enabled)
            self.loot_tracker.process_frame(self.frame, config.enabled)

            tl, _, tl_score, _ = utils.multi_scale_match(
                self.frame, MM_TL_TEMPLATE, MINIMAP_TEMPLATE_SCALES
            )
            _, br, br_score, _ = utils.multi_scale_match(
                self.frame, MM_BR_TEMPLATE, MINIMAP_TEMPLATE_SCALES
            )
            manual_calibration = False
            automatic_match_failed = (
                min(tl_score, br_score) < MINIMAP_MATCH_THRESHOLD
                or br[0] <= tl[0]
                or br[1] <= tl[1]
            )
            manual_requested = self.manual_calibration_requested.is_set()
            if automatic_match_failed:
                warning_key = 'corners'
                warning = (
                    '[!] Could not reliably locate the minimap corners '
                    f'(scores: {tl_score:.2f}, {br_score:.2f}). '
                    'Keep the game minimap visible and at its default size.'
                )
                if warning_key != last_calibration_warning:
                    print(f'\n{warning}')
                    last_calibration_warning = warning_key
            if manual_requested or (automatic_match_failed and not manual_calibration_prompted):
                manual_calibration_prompted = True
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

            if manual_calibration:
                print('\n[~] Minimap calibrated using the selected area.')
            self.manual_calibration_requested.clear()
            self.minimap_ratio = (mm_br[0] - mm_tl[0]) / (mm_br[1] - mm_tl[1])
            self.minimap_sample = self.frame[mm_tl[1]:mm_br[1], mm_tl[0]:mm_br[0]]
            last_calibration_warning = None
            self.calibrated = True

            with mss.mss() as self.sct:
                while True:
                    if not self.calibrated:
                        break

                    # Take screenshot
                    self.frame = self.screenshot()
                    if self.frame is None:
                        continue
                    self.health.process_frame(self.frame)
                    self.mob_tracker.process_frame(self.frame, config.enabled)
                    self.loot_tracker.process_frame(self.frame, config.enabled)

                    # Crop the frame to only show the minimap
                    minimap = self.frame[mm_tl[1]:mm_br[1], mm_tl[0]:mm_br[0]]

                    # Determine the player's position
                    player = utils.multi_match(minimap, PLAYER_TEMPLATE, threshold=0.8)
                    if player:
                        config.player_pos = utils.convert_to_relative(player[0], minimap)

                    # Package display information to be polled by GUI
                    self.minimap = {
                        'minimap': minimap,
                        'rune_active': config.bot.rune_active,
                        'rune_pos': config.bot.rune_pos,
                        'path': config.path,
                        'player_pos': config.player_pos
                    }

                    time.sleep(0.001)

    def screenshot(self, delay=1):
        try:
            return np.array(self.sct.grab(self.window))
        except mss.exception.ScreenShotError:
            print(f'\n[!] Error while taking screenshot, retrying in {delay} second'
                  + ('s' if delay != 1 else ''))
            time.sleep(delay)
