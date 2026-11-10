"""Screen-based HP bar monitoring and optional potion use."""

import ctypes
import threading
import time

import cv2
import numpy as np

from src.common import vkeys
from src.common.interfaces import Configurable
from src.common.vkeys import press

user32 = ctypes.windll.user32

HP_SEARCH_TOP_RATIO = 0.70
HP_SEARCH_BOTTOM_RATIO = 0.98
HP_SEARCH_LEFT_RATIO = 0.03
HP_SEARCH_RIGHT_RATIO = 0.65
HP_BAR_MIN_WIDTH = 50
HP_BAR_MAX_WIDTH = 180
HP_BAR_MIN_HEIGHT = 6
HP_BAR_MAX_HEIGHT = 22
HP_TRIGGER_PERCENT = 50
POTION_COOLDOWN = 1.5


def is_game_focused():
    """Return whether the MapleStory window is the current foreground window."""

    game_window = user32.FindWindowW(None, 'MapleStory')
    return bool(game_window and user32.GetForegroundWindow() == game_window)


def estimate_hp_percent(frame, region):
    """Estimate HP from the red fill within a selected screen rectangle."""

    if frame is None or region is None or frame.ndim != 3:
        return None

    x1, y1, x2, y2 = region
    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
        return None
    if x2 - x1 < 10 or y2 - y1 < 3:
        return None

    bar = frame[y1:y2, x1:x2]
    if bar.shape[2] == 4:
        hsv = cv2.cvtColor(bar[:, :, :3], cv2.COLOR_BGR2HSV)
    elif bar.shape[2] == 3:
        hsv = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
    else:
        return None

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    red = (((hue <= 10) | (hue >= 170))
           & (saturation >= 100)
           & (value >= 70))
    red_columns = np.count_nonzero(red, axis=0) >= max(1, round(bar.shape[0] * 0.25))
    filled = np.flatnonzero(red_columns)
    if len(filled) == 0:
        return None
    if filled[0] > max(2, round(bar.shape[1] * 0.1)):
        return None

    return min(100.0, (filled[-1] + 1) * 100.0 / bar.shape[1])


def find_hp_bar_region(frame):
    """Find the red HP fill and its neutral-colored track near the HUD bottom."""

    if frame is None or frame.ndim != 3 or frame.shape[2] < 3:
        return None

    height, width = frame.shape[:2]
    search_x1 = round(width * HP_SEARCH_LEFT_RATIO)
    search_x2 = round(width * HP_SEARCH_RIGHT_RATIO)
    search_y1 = round(height * HP_SEARCH_TOP_RATIO)
    search_y2 = round(height * HP_SEARCH_BOTTOM_RATIO)
    if search_x2 - search_x1 < HP_BAR_MIN_WIDTH:
        return None

    search = frame[search_y1:search_y2, search_x1:search_x2, :3]
    hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
    red_mask = (
        ((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170))
        & (hsv[:, :, 1] >= 100)
        & (hsv[:, :, 2] >= 70)
    ).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(red_mask)
    candidates = []
    for index in range(1, count):
        x, y, component_width, component_height, area = stats[index]
        if (component_width < 3 or component_width > HP_BAR_MAX_WIDTH
                or not HP_BAR_MIN_HEIGHT <= component_height <= HP_BAR_MAX_HEIGHT
                or area < 12):
            continue

        x += search_x1
        y += search_y1
        row = frame[y + component_height // 2, :, :3].astype(np.int16)
        neutral = (
            (row.max(axis=1) - row.min(axis=1) <= 12)
            & (row.mean(axis=1) >= 50)
            & (row.mean(axis=1) < 250)
        )
        track_end = x + component_width - 1
        while track_end + 1 < width and neutral[track_end + 1]:
            track_end += 1
        track_width = track_end - x + 1
        if HP_BAR_MIN_WIDTH <= track_width <= HP_BAR_MAX_WIDTH:
            candidates.append((
                y + component_height,
                abs(track_width - 108),
                x,
                y,
                x + track_width,
                y + component_height
            ))

    if not candidates:
        return None

    _, _, x1, y1, x2, y2 = max(
        candidates,
        key=lambda candidate: (
            candidate[0],
            -candidate[1],
            -candidate[2]
        )
    )
    return x1, y1, x2, y2


class HealthPotionMonitor(Configurable):
    """Stores HP potion options and monitors the fixed bottom HUD HP bar."""

    DEFAULT_CONFIG = {
        'Enabled': False,
        'Threshold': 50,
        'Key': 'r',
        'Region': ''
    }

    def __init__(self):
        self._lock = threading.Lock()
        self._current_percent = None
        self._last_potion_time = 0
        self._last_action = ''
        super().__init__('hp_potion')

    def settings(self):
        with self._lock:
            return self.config.copy()

    def update_settings(self, enabled, threshold, key):
        try:
            threshold = int(threshold)
        except (TypeError, ValueError) as error:
            raise ValueError('HP threshold must be a whole number from 1 to 99.') from error
        key = str(key).strip().lower()
        if not 1 <= threshold <= 99:
            raise ValueError('HP threshold must be between 1 and 99.')
        if key not in vkeys.KEY_MAP:
            raise ValueError(f"'{key}' is not a supported keyboard key.")

        with self._lock:
            self.config.update({
                'Enabled': bool(enabled),
                'Threshold': HP_TRIGGER_PERCENT,
                'Key': key
            })
            self.save_config()

    def set_region(self, bounds, frame):
        (x1, y1), (x2, y2) = bounds
        height, width = frame.shape[:2]
        if (x1 < 0 or y1 < 0 or x2 > width or y2 > height
                or x2 - x1 < 10 or y2 - y1 < 3):
            raise ValueError('The selected HP bar area is too small or outside the screen.')
        if estimate_hp_percent(frame, (x1, y1, x2, y2)) in (None, 0.0):
            raise ValueError(
                'No red HP fill was found. Select the full HP bar, including its current red fill.'
            )

        region = ','.join(str(value) for value in (x1, y1, x2, y2, width, height))
        with self._lock:
            self.config['Region'] = region
            self._current_percent = None
            self.save_config()

    def current_percent(self):
        with self._lock:
            return self._current_percent

    def last_action(self):
        with self._lock:
            return self._last_action

    def process_frame(self, frame):
        with self._lock:
            enabled = self.config['Enabled']
            key = self.config['Key']

        if frame is None or frame.ndim != 3:
            percent = None
        else:
            region = find_hp_bar_region(frame)
            percent = estimate_hp_percent(frame, region)

        use_potion = False
        focus_warning = False
        now = time.monotonic()
        with self._lock:
            self._current_percent = percent
            if (percent is not None and percent <= HP_TRIGGER_PERCENT
                    and enabled
                    and now - self._last_potion_time >= POTION_COOLDOWN):
                if is_game_focused():
                    self._last_potion_time = now
                    use_potion = True
                    self._last_action = (
                        f'Used {key.upper()} at {percent:.0f}%; '
                        'will repeat if HP stays low.'
                    )
                else:
                    focus_warning = not self._last_action.startswith(
                        'Waiting for MapleStory focus'
                    )
                    self._last_action = 'Waiting for MapleStory focus to use potion.'

        if use_potion:
            print(
                f'\n[~] HP is {percent:.0f}%; sending potion key {key.upper()}.'
            )
            press(key, 1, allow_paused=True)
        elif focus_warning:
            print('\n[!] HP is below the potion threshold, but MapleStory is not focused; waiting.')
