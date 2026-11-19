"""
Chat watch: a Telegram message (with a screenshot) when someone talks in the
chat box while the bot runs, so you can answer them yourself.

The chat log sits right above the blue "All" button (assets/msg-boc.png,
assets/henny.png). Its lines come in three looks:
 - megaphones: a yellow or pink band behind the text      -> ignored
 - notices: yellow text on the dark log                     -> ignored
 - normal chat (white text) on the dark log: someone on this map -> alert
 - whispers (green text) -> alert only with "chat_whispers": true

2026-11-19 (stuck.mp4, "stop sending every chat line"): the log area reaches
above the chat box into the game picture and over the All/Social tabs; as the
camera moved, scenery there read as "new white lines" (alerts of 9 lines at
a time). Only rows on the log's own near-black background count now
(LOG_DARK: the real line in msg-boc.png sits on 27-40, the scenery 65-100).

Each chat line is remembered by the shape of its text, not its place, so a
megaphone that pushes an old line up the log does not alert again. Lines
already there when the watch starts are only remembered.

Your own messages are white lines too, so typing in chat while the bot runs
sends an alert as well.

Settings in .settings/telegram.json (same file as the other alerts):
  "chat": false           alert on chat lines (default off)
  "chat_gap": 20          seconds between two chat alerts at least
  "chat_whispers": false  also alert on whispers (green)
"""

import hashlib
import threading
import time

import cv2
import numpy as np

ALL_BAR_BGR = (119, 85, 51)     # fill of the "All" button, same in both screenshots
ALL_BAR_TOL = 14
INPUT_WHITE = 245               # the text field right of it is white
LOG_WIDTH_PER_BAR = 6.5         # henny.png: bar 63 px wide, log ~420 px
LOG_LINES = 6                   # look this many line heights above the bar
CHECK_INTERVAL = 1.0
DEFAULT_GAP = 20.0
MEMORY_SECONDS = 900.0
MAX_REMEMBERED = 200
BAND_SATURATION = 90            # a megaphone band: the row's middle colour this saturated...
BAND_VALUE = 120                # ...and this bright
LOG_DARK = 60                   # the chat log's own background is darker than this


def find_all_bar(frame):
    """(x, y, w, h) of the "All" button, or None."""
    height = frame.shape[0]
    top = int(height * 0.55)
    area = frame[top:, :, :3].astype(np.int16)
    diff = np.abs(area - np.array(ALL_BAR_BGR, np.int16)).max(axis=2)
    mask = (diff <= ALL_BAR_TOL).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for i in range(1, count):
        x, y, w, h, filled = stats[i]
        if h < 6 or w < 3 * h or w > 12 * h or filled < 0.5 * w * h:
            continue
        # the white input field starts within a few px to its right
        right = frame[top + y + h // 2, x + w: min(frame.shape[1], x + w + 3 * h), :3]
        if right.size == 0 or not (right.min(axis=1) >= INPUT_WHITE).any():
            continue
        if best is None or w * h > best[2] * best[3]:
            best = (int(x), int(top + y), int(w), int(h))
    return best


def log_region(frame, bar):
    """(x1, y1, x2, y2) of the chat log above the bar."""
    x, y, w, h = bar
    line = h + 1
    x2 = min(frame.shape[1], x + int(w * LOG_WIDTH_PER_BAR))
    y2 = max(0, y - max(2, h // 3))
    y1 = max(0, y2 - LOG_LINES * line)
    return x, y1, x2, y2


def chat_lines(frame, region):
    """[(kind, signature, row)] of the white (say) and green (whisper) lines in
    the log, top to bottom."""
    x1, y1, x2, y2 = region
    log = frame[y1:y2, x1:x2, :3]
    if log.size == 0:
        return []
    hsv = cv2.cvtColor(np.ascontiguousarray(log), cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1].astype(np.int16), hsv[:, :, 2].astype(np.int16)
    b, g, r = (log[:, :, i].astype(np.int16) for i in range(3))
    white = (val >= 200) & (sat <= 40)
    green = (g >= 170) & (g - r >= 60) & (g - b >= 60)
    # a megaphone row: its middle colour is a bright, saturated band
    band = (np.median(sat, axis=1) >= BAND_SATURATION) & (np.median(val, axis=1) >= BAND_VALUE)
    dark = np.median(val, axis=1) < LOG_DARK
    keep = (dark & ~band)[:, None]
    white &= keep
    green &= keep
    text = white | green
    rows = text.sum(axis=1) >= 2
    lines, start = [], None
    for row, on in enumerate(list(rows) + [False]):
        if on and start is None:
            start = row
        elif not on and start is not None:
            if row - start >= 4:
                lines.append((start, row))
            start = None
    found = []
    for start, end in lines:
        part = text[start:end]
        cols = np.where(part.any(axis=0))[0]
        if cols.size < 6:
            continue
        crop = part[:, cols[0]:cols[-1] + 1]
        kind = 'whisper' if green[start:end].sum() > white[start:end].sum() else 'say'
        found.append((kind, _signature(crop), y1 + start))
    return found


def _signature(mask):
    """A position-free fingerprint of a text line (shrunk, so a pixel of
    anti-aliasing doesn't make an old line look new)."""
    height = 8
    width = max(8, int(mask.shape[1] * height / max(1, mask.shape[0]) / 2))
    small = cv2.resize(mask.astype(np.uint8) * 255, (width, height),
                       interpolation=cv2.INTER_AREA) >= 128
    return hashlib.sha1(np.packbits(small).tobytes() + bytes([width % 256])).hexdigest()[:16]


class ChatWatch:
    def __init__(self, frame_source=None, send=None, wants=None, running=None, clock=None):
        self._frame = frame_source or self._capture_frame
        self._send = send
        self._wants = wants
        self._running = running or self._bot_running
        self._clock = clock or time.monotonic
        self._seen = {}             # signature -> last time seen
        self._primed = False
        self._last_alert = float('-inf')
        self._pending = []          # kinds of new lines not alerted yet (gap)
        self._bar = None
        self._bar_checked = float('-inf')
        self._thread = None
        self.whispers = False       # (tests: whispers alert too)

    @staticmethod
    def _capture_frame():
        from src.common import config
        return getattr(getattr(config, 'capture', None), 'frame', None)

    @staticmethod
    def _bot_running():
        from src.common import config
        return bool(getattr(config, 'enabled', False))

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._main, name='chat watch', daemon=True)
            self._thread.start()

    def _main(self):
        while True:
            try:
                self.check()
            except Exception as error:
                print(f'\n[!] Chat watch error (carrying on): {error}')
            time.sleep(CHECK_INTERVAL)

    def _settings(self):
        from src.modules import phone_alert
        s = phone_alert._load()
        return (phone_alert, bool(s.get('chat', False)), float(s.get('chat_gap', DEFAULT_GAP)),
                bool(s.get('chat_whispers', False)))

    def check(self):
        """One look at the chat box. Returns the new lines' kinds."""
        if self._send is None:
            alert, on, gap, whispers_on = self._settings()
            send, wanted = alert.send, on and alert.enabled()
        else:
            send, wanted, gap = self._send, (self._wants() if self._wants else True), DEFAULT_GAP
            whispers_on = self.whispers
        if not wanted:
            return []
        if not self._running():
            self._primed = False    # what was said while paused isn't news
            return []
        frame = self._frame()
        if frame is None:
            return []
        now = self._clock()
        if self._bar is None or now - self._bar_checked > 30:
            self._bar = find_all_bar(frame) or self._bar
            self._bar_checked = now
        if self._bar is None:
            return []
        lines = chat_lines(frame, log_region(frame, self._bar))
        if not lines and not find_all_bar(frame):
            self._bar = None        # window moved or chat hidden
            return []
        new = [kind for kind, sig, _ in lines if sig not in self._seen]
        for _, sig, _ in lines:
            self._seen[sig] = now
        self._forget(now)
        if not self._primed:
            self._primed = True
            return []
        if not whispers_on:
            new = [kind for kind in new if kind != 'whisper']
        self._pending.extend(new)
        if self._pending and now - self._last_alert >= gap:
            kinds = self._pending
            self._pending = []
            self._last_alert = now
            whispers = kinds.count('whisper')
            says = len(kinds) - whispers
            parts = []
            if whispers:
                parts.append(f'{whispers} whisper{"s" if whispers > 1 else ""}')
            if says:
                parts.append(f'{says} chat line{"s" if says > 1 else ""} from someone on this map')
            text = f'New in chat: {" and ".join(parts)}. Check the game if it is for you.'
            print(f'\n[~] {text}')      # (no emoji here: the Windows console can't print it)
            send('💬 ' + text, photo=True)
        return new

    def _forget(self, now):
        old = [s for s, t in self._seen.items() if now - t > MEMORY_SECONDS]
        for s in old:
            del self._seen[s]
        if len(self._seen) > MAX_REMEMBERED:
            for s, _ in sorted(self._seen.items(), key=lambda kv: kv[1])[:len(self._seen) - MAX_REMEMBERED]:
                del self._seen[s]


chat_watch = ChatWatch()
