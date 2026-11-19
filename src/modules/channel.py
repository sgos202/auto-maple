"""
Changing channel through the game's own UI, as in assets/iron_hog_issue.mp4:

    MENU button (bottom bar) -> GAME MENU -> CHANGE CHANNEL -> channel grid
    (the current channel is highlighted blue) -> click the next channel ->
    CHANGE CHANNEL button -> a black loading screen -> the map again, with the
    character at the map's spawn point.

Every screen is recognised from small crops of the game's UI (assets/ui/*.png)
and the clicks are placed relative to them, so the window may sit anywhere.

FindQuietChannel also counts the other players on the minimap (red dots)
after landing and moves on when the channel is busy, up to a limit.
"""

import random

import cv2
import numpy as np

from src.modules.plan_steps import Step, RUNNING, DONE, FAILED, black_fraction

UI_MATCH_THRESHOLD = 0.8
UI_SCALES = (1.0, 0.95, 1.05, 0.9, 1.1)


def _load(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f'\n[!] Missing UI template {path}; channel changes will not work.')
    return image


MENU_BUTTON = _load('assets/ui/menu_button.png')
MENU_CHANGE_CHANNEL = _load('assets/ui/menu_change_channel.png')
DIALOG_BUTTON = _load('assets/ui/channel_dialog_button.png')

# The channel grid, from the centre of the dialog's blue CHANGE CHANNEL button
# (measured on the recording: button at (835, 433), CH.1 at (434.5, 312.5))
GRID_COLUMNS, GRID_ROWS = 8, 5
GRID_FIRST = (-400.5, -120.5)
GRID_PITCH = (70, 22)
CELL_HALF = (26, 5)                 # the patch read to see a cell's colour
HIGHLIGHT_MIN_BLUE = 50             # blue minus red of the highlighted cell, at least
CHANNELS = GRID_COLUMNS * GRID_ROWS


def find_ui(frame, template, region=None, threshold=UI_MATCH_THRESHOLD):
    """Centre (x, y) of TEMPLATE in FRAME (optionally inside REGION, fractions
    left, top, right, bottom), or None."""
    if frame is None or template is None:
        return None
    gray = cv2.cvtColor(np.ascontiguousarray(frame[:, :, :3]), cv2.COLOR_BGR2GRAY)
    left = top = 0
    if region is not None:
        height, width = gray.shape
        left, top = int(region[0] * width), int(region[1] * height)
        gray = gray[top:int(region[3] * height), left:int(region[2] * width)]
    best = None
    for scale in UI_SCALES:
        scaled = template if scale == 1.0 else cv2.resize(
            template, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        if scaled.shape[0] > gray.shape[0] or scaled.shape[1] > gray.shape[1]:
            continue
        _, score, _, loc = cv2.minMaxLoc(cv2.matchTemplate(gray, scaled, cv2.TM_CCOEFF_NORMED))
        if best is None or score > best[0]:
            best = (score, loc, scaled.shape, scale)
        if score >= 0.95:
            break
    if best is None or best[0] < threshold:
        return None
    score, loc, shape, scale = best
    return (left + loc[0] + shape[1] / 2, top + loc[1] + shape[0] / 2, scale)


def cell_centre(button, channel):
    """Frame pixel centre of CHANNEL's (1..40) cell, from the dialog button's centre."""
    index = channel - 1
    column, row = index % GRID_COLUMNS, index // GRID_COLUMNS
    scale = button[2] if len(button) > 2 else 1.0
    return (button[0] + (GRID_FIRST[0] + GRID_PITCH[0] * column) * scale,
            button[1] + (GRID_FIRST[1] + GRID_PITCH[1] * row) * scale)


def current_channel(frame, button):
    """The highlighted (blue) channel in the open dialog, or None."""
    best = None
    for channel in range(1, CHANNELS + 1):
        x, y = cell_centre(button, channel)
        x, y = int(round(x)), int(round(y))
        patch = frame[max(0, y - CELL_HALF[1]):y + CELL_HALF[1] + 1,
                      max(0, x - CELL_HALF[0]):x + CELL_HALF[0] + 1, :3]
        if patch.size == 0:
            continue
        blue, _, red = patch.reshape(-1, 3).astype(np.float32).mean(axis=0)
        blueness = blue - red
        if best is None or blueness > best[0]:
            best = (blueness, channel)
    if best is None or best[0] < HIGHLIGHT_MIN_BLUE:
        return None
    return best[1]


def count_other_players(minimap):
    """
    Other players are red dots on the minimap. Counted as small red blobs:
    the notifier's exact-colour template misses them in a compressed or
    scaled picture (their hue wraps round to 177-179 there).
    """
    return len(other_player_dots(minimap))


def other_player_dots(minimap):
    """Pixel centres (x, y) of the other players' red dots on MINIMAP."""
    if minimap is None or minimap.size == 0:
        return []
    hsv = cv2.cvtColor(np.ascontiguousarray(minimap[:, :, :3]), cv2.COLOR_BGR2HSV)
    red = (cv2.inRange(hsv, (0, 200, 170), (10, 255, 255))
           | cv2.inRange(hsv, (170, 200, 170), (180, 255, 255)))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(red)
    return [
        (float(centroids[index][0]), float(centroids[index][1]))
        for index in range(1, count)
        if 6 <= stats[index, cv2.CC_STAT_AREA] <= 64
        and stats[index, cv2.CC_STAT_WIDTH] <= 10 and stats[index, cv2.CC_STAT_HEIGHT] <= 10
    ]


def players_near(minimap, me, near_x, same_y, ratio):
    """Other players on MINIMAP within NEAR_X of ME (minimap units, as
    config.player_pos) and within SAME_Y of its height (the same platform).
    RATIO: the minimap's width / height (capture.minimap_ratio)."""
    if minimap is None or minimap.size == 0 or not me:
        return []
    height, width = minimap.shape[:2]
    found = []
    for px, py in other_player_dots(minimap):
        x, y = px / width, py / ratio / height
        if abs(x - me[0]) <= near_x and abs(y - me[1]) <= same_y:
            found.append((x, y))
    return found


class ChangeChannel(Step):
    """Moves to the next channel (the one after the highlighted one)."""

    name = 'change channel'
    CLICK_GAP = 0.35            # between UI clicks (the game needs a moment)
    MENU_WAIT = 1.5             # menu/dialog not up this long after a click: retry
    BLACK_WAIT = 6.0            # no loading screen this long after confirming: retry
    LOAD_TIMEOUT = 30.0
    BLACK = 0.85                # this much of the screen black: loading
    LANDED_SECONDS = 0.8        # a fresh minimap fix this long after the screen came back
    # User 2026-11-19 ("don't give up on 2, keep on changing channels till it
    # works"): channels 1, 2 and 3 in a row did not load and the bot paused.
    # Each failed try moves on to the next channel; every channel gets a go.
    ATTEMPTS = CHANNELS
    RETRY_PAUSE = (1.5, 3.0)    # seconds between tries (the game may still be busy)
    # The game won't change channel for a few seconds after an attack
    # (user, 2026-11-19: "wait like 3 or 4 sec after attacking")
    AFTER_ATTACK_SECONDS = 4.0

    def __init__(self, timeout=None):
        super().__init__()
        self.timeout = timeout
        self.channel = None         # where it went
        self.known_channel = None   # where it came from, if the dialog showed it

    def begin(self, io):
        super().begin(io)
        self.phase = 'open'
        self.attempt = 1
        self.at = io.now()
        self.tries = 0
        self.target = None
        self._told_wait = False
        io.release()

    def tick(self, io):
        now = io.now()
        if self.phase == 'open' and self.attempt == 1 and self.tries == 0:
            since = getattr(io, 'seconds_since_attack', lambda: float('inf'))()
            if since < self.AFTER_ATTACK_SECONDS:
                if not self._told_wait:
                    self._told_wait = True
                    io.log(f'Waiting {self.AFTER_ATTACK_SECONDS - since:.1f} s after the last '
                           'attack before changing channel.')
                return RUNNING
        if now - self.at < self.CLICK_GAP and self.phase not in ('loading', 'wait_black'):
            return RUNNING
        frame = io.frame()
        handler = getattr(self, f'_{self.phase}')
        return handler(io, frame, now)

    def _next(self, phase, now):
        self.phase = phase
        self.at = now
        self.tries = 0
        return RUNNING

    def _open(self, io, frame, now):
        dialog = find_ui(frame, DIALOG_BUTTON)
        if dialog is not None:
            return self._pick(io, frame, now)
        item = find_ui(frame, MENU_CHANGE_CHANNEL, region=(0.4, 0.3, 1.0, 1.0))
        if item is not None:
            io.click(item[0], item[1])
            return self._next('wait_dialog', now)
        button = find_ui(frame, MENU_BUTTON, region=(0.4, 0.75, 1.0, 1.0))
        if button is not None and self.tries < 2:
            io.click(button[0], button[1])
        else:
            io.press('esc', 1)          # no MENU button seen: Esc opens the game menu
        self.tries += 1
        self.phase, self.at = 'wait_menu', now
        return RUNNING

    def _wait_menu(self, io, frame, now):
        item = find_ui(frame, MENU_CHANGE_CHANNEL, region=(0.4, 0.3, 1.0, 1.0))
        if item is not None:
            io.click(item[0], item[1])
            return self._next('wait_dialog', now)
        if now - self.at > self.MENU_WAIT:
            if self.tries >= 4:
                return self._retry(io, now, 'the game menu did not open')
            self.phase = 'open'
        return RUNNING

    def _wait_dialog(self, io, frame, now):
        if find_ui(frame, DIALOG_BUTTON) is not None:
            return self._pick(io, frame, now)
        if now - self.at > self.MENU_WAIT:
            return self._retry(io, now, 'the channel list did not open')
        return RUNNING

    def _pick(self, io, frame, now):
        button = find_ui(frame, DIALOG_BUTTON)
        current = current_channel(frame, button)
        if current is not None:
            self.known_channel = self.known_channel or current
        if self.target is not None:
            base = self.target      # a retry: that channel would not load, go past it
        else:
            base = current or self.known_channel or random.randint(1, CHANNELS)
        self.target = base % CHANNELS + 1
        cell = cell_centre(button, self.target)
        io.log(f'Changing channel: {current or "?"} -> {self.target}.')
        io.click(cell[0], cell[1])
        self.button = button
        return self._next('confirm', now)

    def _confirm(self, io, frame, now):
        io.expect_black_screen(self.LOAD_TIMEOUT + 10)
        io.click(self.button[0], self.button[1])
        return self._next('wait_black', now)

    def _wait_black(self, io, frame, now):
        if black_fraction(frame) >= self.BLACK:
            self.black_seen = now
            return self._next('loading', now)
        if now - self.at > self.BLACK_WAIT:
            # A full channel, a "cannot change channel now" notice, a missed click
            return self._retry(io, now, f'channel {self.target} did not load')
        return RUNNING

    def _loading(self, io, frame, now):
        if black_fraction(frame) >= self.BLACK:
            self.black_seen = now
        elif (now - self.black_seen >= self.LANDED_SECONDS
              and io.position() is not None and io.position_time() > self.black_seen):
            self.channel = self.target
            io.expect_black_screen(0)
            return DONE
        if now - self.at > self.LOAD_TIMEOUT:
            return self.fail('the loading screen never ended')
        return RUNNING

    def _retry(self, io, now, reason):
        io.press('esc', 2, down_time=0.05, up_time=0.15)      # close whatever is open
        io.expect_black_screen(0)
        if self.attempt >= self.ATTEMPTS:
            return self.fail(reason)
        io.log(f'Channel change: {reason}; trying the next one '
               f'(try {self.attempt + 1} of {self.ATTEMPTS}).')
        self.attempt += 1
        self._next('open', now)
        self.at = now + random.uniform(*self.RETRY_PAUSE)     # (tick waits CLICK_GAP after .at)
        return RUNNING


class FindQuietChannel(Step):
    """
    Changes channel, then counts the other players on the minimap; a busy
    channel is left for the next one, at most MAX_CHANNELS changes in all
    (the last one is kept, busy or not).
    """

    name = 'find a quiet channel'
    changes_channel = True      # a plan that ran this ends in another channel
    LOOK_DELAY = 1.0            # let the minimap fill in after landing
    LOOK_SECONDS = 2.0
    BUSY_SAMPLES = 3            # players seen in this many looks: busy

    def __init__(self, max_channels=3, change_factory=ChangeChannel):
        super().__init__()
        self.max_channels = max_channels
        self.change_factory = change_factory

    def begin(self, io):
        super().begin(io)
        self.changes = 0
        self.change = None
        self.looks = []
        self.known_channel = None
        self._start_change(io)

    def _start_change(self, io):
        self.change = self.change_factory()
        self.change.known_channel = self.known_channel
        self.changes += 1
        self.phase = 'change'

    def tick(self, io):
        now = io.now()
        if self.phase == 'change':
            status = self.change.run(io)
            if status == RUNNING:
                return RUNNING
            self.change.end(io)
            if status == FAILED:
                return self.fail(self.change.reason)
            self.known_channel = self.change.channel
            self.phase, self.landed, self.looks = 'look', now, []
            return RUNNING
        if now - self.landed < self.LOOK_DELAY:
            return RUNNING
        self.looks.append(count_other_players(io.minimap()))
        if now - self.landed < self.LOOK_DELAY + self.LOOK_SECONDS:
            return RUNNING
        busy = sum(1 for n in self.looks if n > 0) >= self.BUSY_SAMPLES
        players = max(self.looks) if self.looks else 0
        channel = self.known_channel
        if not busy:
            io.log(f'Channel {channel} is free of other players; staying.')
            return DONE
        if self.changes >= self.max_channels:
            io.log(f'Channel {channel} has {players} other player(s), but that was '
                   f'{self.changes} channels; staying here.')
            return DONE
        io.log(f'Channel {channel} has {players} other player(s); trying the next one.')
        self._start_change(io)
        return RUNNING

    def end(self, io):
        io.release()
