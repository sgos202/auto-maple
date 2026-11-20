"""
Solve the Violetta challenge mini-game ("keep your eyes on the Violetta who's
wearing make-up") from screen frames.

How the game moves the four Violettas (verified against recorded footage; the
geometry comes from https://github.com/hikaricalyx/violetta-sim, MIT):

* All coordinates below are in *window space*: the 441x318 game window
  at 1x. The real window may be drawn at any scale; ``Calibration`` maps it.
* The four stand in slots whose pivots are at x = 44 + 104 * slot, y = 121.
* During a shuffle each one walks at its OWN depth row: the front row and the
  three deepest rows, one each. A row is drawn at a fixed scale and height, so
  the row a Violetta walks in identifies it for the whole shuffle -- even when
  two of them cross, they are at different sizes.
* Every walk is: step from its slot into its row (x unchanged), move sideways
  along the row, then step out of the row into its final slot (x unchanged).

So the solver never has to follow a sprite through a crossing. It scans each
row's horizontal band for the walking sprite at that row's scale. The x where
a row's walker first appears is its starting slot, and the x where it last
appears is its final slot. The target's row is the one that started in the
target's slot, and its final slot is the answer.
"""

import itertools
import os
import time as _time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np


ASSET_DIR = os.path.join('assets', 'violetta')
_LOADED_AT = _time.time()   # code edited after this is not running (see play_live)

WINDOW_SIZE = (441, 318)
SLOT_COUNT = 4
SLOT_PIVOT_X = 44.0
SLOT_STEP = 104.0
STAND_PIVOT_Y = 121.0
DEPTH_STEP = 9.0            # each row further back is drawn this much higher
DEPTH_SCALE = 0.8           # ...and this much smaller
STAND_ROW = 1
WALK_ROWS = (0, 2, 3, 4)    # front row + three deepest, one Violetta each
BACKDROP_ORIGIN = (13.0, 65.0)
BUTTON_CENTRES = tuple((65.0 + SLOT_STEP * i, 280.0) for i in range(SLOT_COUNT))
BUTTON_SIZE = (102, 40)
# A lit answer button, any style: >25% of it this saturated (disabled grey ~25;
# old cyan and the 2026-11-19 orange are >150), or this much brighter than its own
# disabled grey (a white-lit style).
BUTTON_LIT_SATURATION = 110
BUTTON_LIT_BRIGHTER = 35
BUTTON_CLICK_RAISE = 8     # window px above BUTTON_CENTRES to click (see click_point)
TIMER_CENTRE = (222.0, 113.0)   # the red countdown digit shown while an answer is accepted

ROW_MATCH_THRESHOLD = 0.5   # walking sprite found in a row band
MIN_ROWS_WALKING = 3        # rows that must match at once to count as shuffling
FALSE_START_FRAMES = 40     # a "shuffle" over within ~2 s with nothing readable was not one
MAX_SAVED_CANVASES = 240    # frames kept for violetta_rows (280 KB each: ~67 MB, 8-12 s of play)


def slot_x(slot):
    return SLOT_PIVOT_X + SLOT_STEP * slot


def _depth_row_geometry(row):
    return DEPTH_SCALE ** (row - STAND_ROW), STAND_PIVOT_Y + (STAND_ROW - row) * DEPTH_STEP


# The game has more than one set of walking rows. Either way each Violetta keeps
# ONE size and height for its whole walk, so the row still identifies it:
#  * classic -- the simulator's: the front row plus the three deepest depth rows.
#  * light   -- measured from a live game (logs/violetta/violetta-20261113-201642.mp4,
#              free search over size and height): four rows close together just
#              behind the slots, a slower walk (~4 s), and brighter stage lights
#              that wash the walkers out (they still show in local contrast).
# Row ids are unique across layouts. Each row: (id, scale, pivot y).
LAYOUTS = {
    'classic': tuple((row, *_depth_row_geometry(row)) for row in WALK_ROWS),
    'light': ((10, 0.88, 120.0), (11, 0.82, 113.7), (12, 0.775, 109.5), (13, 0.63, 120.4)),
}
_LAYOUT_ROWS = {row: (scale, y) for rows in LAYOUTS.values() for row, scale, y in rows}


def row_geometry(row):
    """Returns (scale, pivot y) of a Violetta walking in ROW (of any layout)."""
    if row in _LAYOUT_ROWS:
        return _LAYOUT_ROWS[row]
    return _depth_row_geometry(row)


def front_row(rows):
    """The layout's biggest (nearest) row: drawn over the others, so it never hides."""
    return max(rows, key=lambda row: row_geometry(row)[0])


def nearest_slots(xs):
    """
    Assigns each x in XS (one per row) to a distinct slot, minimising the total
    distance. Returns (slots, worst distance).
    """
    best = None
    for perm in itertools.permutations(range(SLOT_COUNT), len(xs)):
        cost = [abs(x - slot_x(s)) for x, s in zip(xs, perm)]
        if best is None or sum(cost) < best[0]:
            best = (sum(cost), perm, max(cost))
    return list(best[1]), best[2]


def end_slots_with_lost(xs, lost):
    """
    nearest_slots for the tracks still followed; the LOST ones (their x is just
    where they were last seen) take the slots left over, so a stale place can't
    push a followed track off her slot. Returns (slots, worst distance of the
    followed tracks).
    """
    if not lost or not any(lost) or all(lost):
        return nearest_slots(xs)
    followed = [k for k in range(len(xs)) if not lost[k]]
    slots, error = nearest_slots([xs[k] for k in followed])
    result = [None] * len(xs)
    for k, slot in zip(followed, slots):
        result[k] = slot
    # Left to right in the order they were last seen: lost walkers who went
    # the same way can't have passed each other (002959 round 4)
    missing = sorted((k for k in range(len(xs)) if lost[k]), key=lambda k: xs[k])
    left = sorted(s for s in range(SLOT_COUNT) if s not in slots)
    for k, slot in zip(missing, left):
        result[k] = slot
    return result, error


def local_contrast(gray):
    """
    Keeps fine outlines and removes smooth shading. The stage lights are drawn
    over the Violettas at 80% and change colour twice a second, which crushes
    and shifts their contrast; normalising it locally makes them matchable.
    """
    g = gray.astype(np.float32)
    detail = g - cv2.GaussianBlur(g, (0, 0), 2.0)
    spread = np.sqrt(cv2.GaussianBlur(detail * detail, (0, 0), 4.0)) + 4.0
    return detail / spread


def _gray(frame):
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)


class Sprites:
    """The game's own artwork, with each image's pivot (origin)."""

    def __init__(self, asset_dir=ASSET_DIR):
        self.pivots = {}
        with open(os.path.join(asset_dir, 'sprite-pivots.txt')) as stream:
            for line in stream:
                parts = line.split()
                if len(parts) == 3 and not line.startswith('#'):
                    self.pivots[parts[0]] = (float(parts[1]), float(parts[2]))
        self.dir = asset_dir
        backdrop = cv2.imread(os.path.join(asset_dir, 'lieDetector.back.png'), cv2.IMREAD_UNCHANGED)
        if backdrop is None:
            raise FileNotFoundError(f'Missing Violetta sprites in {asset_dir!r}.')
        self.backdrop = _gray(backdrop)
        self.walk = self._group('lieDetector.npcCommon.move')
        self.stand = self._group('lieDetector.npc.stand')
        self.special = self._group('lieDetector.npcSpecial.stand')

    def _group(self, name):
        frames = []
        for i in range(8):
            image = cv2.imread(os.path.join(self.dir, f'{name}.{i}.png'), cv2.IMREAD_UNCHANGED)
            if image is None:
                break
            alpha = image[:, :, 3].astype(np.float32) / 255.0
            # Composite over mid-grey so the transparent surround is neutral.
            gray = _gray(image[:, :, :3]).astype(np.float32) * alpha + 128 * (1 - alpha)
            frames.append((gray.astype(np.uint8), image[:, :, 3], self.pivots[f'{name}.{i}']))
        if not frames:
            raise FileNotFoundError(f'Missing sprite group {name!r} in {self.dir!r}.')
        return frames


def _alpha_bounds(group):
    bounds = []
    for _, alpha, (px, py) in group:
        ys, xs = np.where(alpha > 128)
        if xs.size:
            bounds.append((xs.min() - px, ys.min() - py,
                           xs.max() + 1 - px, ys.max() + 1 - py))
    if not bounds:
        raise ValueError('Sprite group has no opaque pixels.')
    return (min(b[0] for b in bounds), min(b[1] for b in bounds),
            max(b[2] for b in bounds), max(b[3] for b in bounds))


class Calibration:
    """Maps window space to frame pixels: frame = origin + window * scale."""

    def __init__(self, scale, origin, score):
        self.scale = scale
        self.origin = origin
        self.score = score

    def to_frame(self, point):
        return (self.origin[0] + point[0] * self.scale, self.origin[1] + point[1] * self.scale)

    def window_canvas(self, frame):
        """Resamples FRAME into a WINDOW_SIZE greyscale image in window space."""
        matrix = np.float32([[self.scale, 0, self.origin[0]], [0, self.scale, self.origin[1]]])
        gray = _gray(frame)
        if self.scale > 1.5:    # pre-shrink so each window pixel averages its area
            factor = 1.0 / self.scale
            gray = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
            matrix = matrix * factor
        return cv2.warpAffine(gray, matrix, WINDOW_SIZE,
                              flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    @classmethod
    def find(cls, frame, sprites, scales=None, min_score=0.6):
        """
        Locates the stage backdrop (curtains, steps and carpet) in FRAME.
        Call it while the stage is unlit, i.e. during the opening reveal.
        Returns a Calibration, or None if the stage was not found.
        """
        gray = _gray(frame)
        band_top, band_bottom = 30, 56      # above the Violettas' heads
        band = sprites.backdrop[band_top:band_bottom]
        if scales is None:
            scales = np.geomspace(0.75, 5.0, 70)

        def score_at(scale):
            factor = 1.0 / scale
            small = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
            if small.shape[0] < band.shape[0] or small.shape[1] < band.shape[1]:
                return -1.0, None
            result = cv2.matchTemplate(small, band, cv2.TM_CCOEFF_NORMED)
            _, value, _, loc = cv2.minMaxLoc(result)
            return value, loc

        coarse = max(((score_at(s), s) for s in scales), key=lambda item: item[0][0])
        (best_score, best_loc), best_scale = coarse
        for scale in np.linspace(best_scale * 0.97, best_scale * 1.03, 13):
            value, loc = score_at(scale)
            if value > best_score:
                best_score, best_loc, best_scale = value, loc, scale
        if best_loc is None or best_score < min_score:
            return None
        # The band's top-left, in frame pixels, is window (13, 65 + band_top).
        origin = (best_loc[0] * best_scale - BACKDROP_ORIGIN[0] * best_scale,
                  best_loc[1] * best_scale - (BACKDROP_ORIGIN[1] + band_top) * best_scale)
        return cls(float(best_scale), origin, float(best_score))


class _ScaledTemplates:
    """
    A sprite group pre-scaled and pre-filtered for one size and height, built
    once. ``profile`` scores it at every pivot x across the window.
    """

    def __init__(self, frames, scale, pivot_y, y_slack=5):
        self.scale, self.pivot_y = scale, pivot_y
        self.items = []
        for gray, alpha, (px, py) in frames:
            template = local_contrast(cv2.resize(gray, None, fx=scale, fy=scale,
                                                 interpolation=cv2.INTER_AREA))
            mask = (cv2.resize(alpha, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_NEAREST) > 128).astype(np.uint8)
            top = int(round(pivot_y - py * scale)) - y_slack
            bottom = top + template.shape[0] + 2 * y_slack
            self.items.append((template, mask, top, bottom, int(round(px * scale))))

    @staticmethod
    def _match(canvas, item):
        template, mask, top, bottom, offset = item
        if top < 0 or bottom > canvas.shape[0]:
            return None
        # The mask matters: without it, clutter around the sprite (other
        # Violettas, the lamps' faint figures) pulls the small rows off target.
        result = cv2.matchTemplate(canvas[top:bottom], template, cv2.TM_CCOEFF_NORMED, mask=mask)
        return offset, np.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0).max(axis=0)

    def profile(self, canvas, pool=None):
        width = canvas.shape[1]
        profile = np.full(width, -1.0, dtype=np.float32)
        if pool is None:
            results = [self._match(canvas, item) for item in self.items]
        else:   # OpenCV releases the GIL, so the frames match in parallel
            results = pool.map(lambda item: self._match(canvas, item), self.items)
        for found in results:
            if found is None:
                continue
            offset, result = found
            xs = np.arange(len(result)) + offset
            inside = xs < width
            profile[xs[inside]] = np.maximum(profile[xs[inside]], result[inside])
        return profile


class _RowTemplates(_ScaledTemplates):
    """Walking templates for one depth row."""

    def __init__(self, sprites, row):
        self.row = row
        super().__init__(sprites.walk, *row_geometry(row))  # any layout's row id


def find_special_slot(canvas, special_templates, normal_templates, pool=None):
    """
    During the reveal, returns (slot, margin) of the Violetta whose eyes are open
    (the make-up one), comparing the game's special sprite with the sleeping one.
    """
    special = special_templates.profile(canvas, pool)
    normal = normal_templates.profile(canvas, pool)
    advantage = []
    for slot in range(SLOT_COUNT):
        x = int(slot_x(slot))
        window = slice(max(0, x - 12), x + 13)
        advantage.append(float(special[window].max() - normal[window].max()))
    order = np.argsort(advantage)[::-1]
    return int(order[0]), advantage[order[0]] - advantage[order[1]]


def standing_slot_scores(canvas, special_templates, normal_templates, pool=None):
    """Returns the strongest standing-sprite match in each fixed start slot."""
    special = special_templates.profile(canvas, pool)
    normal = normal_templates.profile(canvas, pool)
    scores = []
    for slot in range(SLOT_COUNT):
        x = int(slot_x(slot))
        window = slice(max(0, x - 12), min(canvas.shape[1], x + 13))
        scores.append(float(max(special[window].max(), normal[window].max())))
    return scores


def find_reveal_arrow(frame, calibration):
    """Finds the bright orange reveal arrow above the make-up Violetta."""
    hsv = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2HSV)
    scores = []
    for slot in range(SLOT_COUNT):
        x = slot_x(slot) + 18
        x0, y0 = calibration.to_frame((x - 30, 55))
        x1, y1 = calibration.to_frame((x + 31, 125))
        x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
        x1, y1 = min(hsv.shape[1], int(round(x1))), min(hsv.shape[0], int(round(y1)))
        roi = hsv[y0:y1, x0:x1]
        if roi.size == 0:
            scores.append(0.0)
            continue
        orange = cv2.inRange(roi, (0, 120, 100), (24, 255, 255))
        scores.append(cv2.countNonZero(orange) / orange.size)

    order = np.argsort(scores)[::-1]
    best, second = scores[order[0]], scores[order[1]]
    if best < 0.035 or best - second < 0.015:
        return None
    return int(order[0])


class ShuffleRecorder:
    """Collects row detections during one shuffle and resolves it."""

    EDGE_FRAMES = 6     # frames at each end of the walk used to read slots

    def __init__(self, rows=WALK_ROWS):
        self.rows = tuple(rows)
        self.frames = []    # (frame_index, {row: (x, score)}, rows walking)

    def add(self, frame_index, detections):
        walking = sum(score >= ROW_MATCH_THRESHOLD for _, score in detections.values())
        self.frames.append((frame_index, dict(detections), walking))

    def _edge_x(self, row, frames, count=3):
        """Median x of ROW's first COUNT confident detections in FRAMES."""
        xs = [d[row][0] for _, d, _ in frames if d[row][1] >= ROW_MATCH_THRESHOLD]
        return float(np.median(xs[:count])) if xs else None

    def resolve(self):
        """
        Returns ({row: (start slot, end slot)}, report), or (None, reason).

        Slots are read only while the walk is in full swing (at least
        MIN_ROWS_WALKING rows matched). The steps into and out of the rows change
        size, and a Violetta caught mid-step can briefly match another row.
        """
        full = [f for f in self.frames if f[2] >= MIN_ROWS_WALKING]
        if len(full) < 2 * self.EDGE_FRAMES:
            return None, f'walk seen in only {len(full)} frames'
        rows = list(self.rows)
        tail = full[::-1][:self.EDGE_FRAMES * 3]
        ends = [self._edge_x(r, tail) for r in rows]
        # Walkers start moving sideways the moment they reach their row, so the
        # earliest frames (including a few from before the shuffle was confirmed)
        # give the most exact start -- but a Violetta caught mid-step can match
        # the wrong row there. Read it both ways; keep whichever fits the slots.
        early = [f for f in self.frames if f[2] >= 2][:self.EDGE_FRAMES * 3]
        candidates = []
        for head, count in ((early, 2), (full[:self.EDGE_FRAMES * 3], 3)):
            xs = [self._edge_x(r, head, count) for r in rows]
            if None not in xs:
                candidates.append((nearest_slots(xs), xs))
        if not candidates or None in ends:
            return None, f'a row was not seen at the start/end: {candidates} -> {ends}'
        (start_slots, start_err), starts = min(candidates, key=lambda c: c[0][1])
        end_slots, end_err = nearest_slots(ends)
        stayed = sum(a == b for a, b in zip(start_slots, end_slots))
        report = {
            'start_x': [round(x) for x in starts],
            'end_x': [round(x) for x in ends],
            'worst_slot_error': round(max(start_err, end_err), 1),
            'stayed': stayed,
            # The game keeps exactly one Violetta in place and moves the rest.
            'consistent': stayed == 1 and max(start_err, end_err) < SLOT_STEP / 3,
        }
        mapping = {r: (s, e) for r, s, e in zip(rows, start_slots, end_slots)}
        return mapping, report


# Trusting the make-up Violetta's track alone, when the other tracks aren't all
# clean. Measured on every recorded round with a known answer (2026-11-15):
# tracks passing these were right 11 times of 12; the row search they replace
# was right 3 times of 5.
TRUST_SEEN = 0.6            # matched in at least this share of the tracked frames
TRUST_MAX_STEP = 25.0       # px: no bigger move between two matches (a swap onto another)
TRUST_MAX_SPEED = 61.5      # px/frame: FAST_WALK_SPEED (60, same-row walks only) plus 1.5 px tolerance
TRUST_END_ERROR = 15.0      # px: ended this close to a slot
# Live tracking that skipped frames can't be trusted: over a gap a walker
# moves 60-120 px and a box can land on a passing Violetta, and the per-frame
# speed check can't tell (violetta-20261118-021248: a 125 px jump over ~4
# frames read as 32 px/frame; three rounds were clicked wrong with confidence).
TRUST_MIN_COVERAGE = 0.85   # share of the round's frames the live tracker processed
TRACKER_CATCH_UP_SECONDS = 2.0  # at answer time, wait this long at most for queued frames
MIN_TRACK_FRAMES = 6        # reject a stationary answer-screen sample as a shuffle track


def trusted_track(tracked, report, slot):
    """Whether the track that started in SLOT can be trusted on its own."""
    if tracked is None or slot is None or not isinstance(report, dict):
        return False
    try:
        end_error = report['nearest_end_error'] if 'nearest_end_error' in report \
            else report['end_error']
        terminal_confirmed = report.get('terminal_confirmed', [])
        if report.get('coverage', 1.0) < TRUST_MIN_COVERAGE:
            return False        # frames were skipped: a box may have jumped Violettas
        ambiguous = report.get('ambiguous', [])
        if slot < len(ambiguous) and ambiguous[slot]:
            return False
        lost = report.get('lost', [])
        if slot < len(lost) and lost[slot]:
            return False        # still waiting to be seen again
        max_speed = report.get('max_speed')
        step_ok = ((max_speed[slot] <= TRUST_MAX_SPEED if max_speed is not None
                    else report['max_step'][slot] <= TRUST_MAX_STEP)
                   or (slot < len(terminal_confirmed) and terminal_confirmed[slot]))
        return (report['frames'] >= MIN_TRACK_FRAMES
                and report['seen'][slot] >= TRUST_SEEN
                and step_ok
                and end_error[slot] <= TRUST_END_ERROR)
    except (KeyError, IndexError, TypeError):
        return False


UNSURE_MIN_SEEN = 0.5       # an untrusted track used as a fallback must be this well seen...
# ...and a LOST one this well seen and last seen this close to a slot: she went
# out of sight standing on it, not mid-walk (20261120-100529 round 1: her track
# went lost 2.6 px from slot 4 as the buttons lit; the start-slot guess clicked 1)
UNSURE_LOST_MIN_SEEN = 0.75
UNSURE_LOST_END_ERROR = SLOT_STEP / 6
# Live: when the buttons light while her track is lost (hidden in a spotlight),
# keep tracking this long for her to be found again before answering. The
# countdown runs 5 s; this leaves >3 s for the click and its retries.
LOST_TRACK_GRACE_SECONDS = 1.5


def _usable_unsure_track(tracked, report, slot):
    """Whether an untrusted track for SLOT is still a better answer than her
    start slot: seen most of the time, ending on a slot, and not lost (or lost
    while standing on that slot). A track that lost her early (seen 0.2, 100 px
    from any slot) is no better than a guess."""
    if not isinstance(tracked, dict) or slot not in tracked or not isinstance(report, dict):
        return False
    try:
        seen = report['seen'][slot]
        lost = report.get('lost', [False] * SLOT_COUNT)[slot]
        end_error = (report['nearest_end_error'] if 'nearest_end_error' in report
                     else report['end_error'])[slot]
    except (KeyError, IndexError, TypeError):
        return False
    if not lost:
        return seen >= UNSURE_MIN_SEEN and end_error <= SLOT_STEP / 3
    nearest = report.get('nearest_end_slot')
    try:
        on_own_slot = nearest is not None and nearest[slot] == tracked[slot]
    except (IndexError, TypeError):
        on_own_slot = False
    return (seen >= UNSURE_LOST_MIN_SEEN and end_error <= UNSURE_LOST_END_ERROR
            and on_own_slot)


def trusted_mapping_by_exclusion(tracked, report):
    """Complete a track-to-slot map when all but one track are individually trusted."""
    if not isinstance(tracked, dict) or set(tracked) != set(range(SLOT_COUNT)):
        return None
    ends = list(tracked.values())
    if any(not isinstance(end, (int, np.integer)) or not 0 <= end < SLOT_COUNT
           for end in ends) or set(ends) != set(range(SLOT_COUNT)):
        return None
    trusted = [slot for slot in range(SLOT_COUNT)
               if trusted_track(tracked, report, slot)]
    if len(trusted) == SLOT_COUNT:
        return dict(tracked), None
    missing = set(range(SLOT_COUNT)) - set(trusted)
    if len(missing) != 1:
        return None
    missing_slot = missing.pop()
    remaining_ends = set(range(SLOT_COUNT)) - {tracked[slot] for slot in trusted}
    if len(remaining_ends) != 1:
        return None
    inferred = dict(tracked)
    inferred[missing_slot] = remaining_ends.pop()
    return inferred, missing_slot


class ViolettaSolver:
    """
    Feed it frames in order with ``update(frame)``. It reports the target's slot
    after every shuffle and says when an answer should be clicked.
    """

    def __init__(self, sprites=None, rounds=4):
        self.sprites = sprites or Sprites()
        self._walk_pose_bounds = [_alpha_bounds((frame,)) for frame in self.sprites.walk]
        self._walk_union_bounds = _alpha_bounds(self.sprites.walk)
        self._standing_bounds = _alpha_bounds(self.sprites.stand + self.sprites.special)
        self._all_pose_bounds = _alpha_bounds(
            self.sprites.walk + self.sprites.stand + self.sprites.special)
        self.layout_rows = {
            name: [_RowTemplates(self.sprites, row) for row, _, _ in rows]
            for name, rows in LAYOUTS.items()
        }
        self.rows = self.layout_rows['classic']
        self.layout = None          # the layout of the shuffle being recorded
        self._special = _ScaledTemplates(self.sprites.special, 1.0, STAND_PIVOT_Y)
        self._normal = _ScaledTemplates(self.sprites.stand, 1.0, STAND_PIVOT_Y)
        self._standing_scan = None
        self._overlay_layout = None
        self._overlay_rows = {}
        self._pool = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1))
        self.calibration = None
        self.target_slot = None
        self._slot_identity = list(range(SLOT_COUNT))  # current slot -> opening-reveal color ID
        self._target_identity = None
        self.rounds_left = rounds
        self.state = 'calibrate'
        self.frame_index = 0
        self._reveal_votes = []
        self._walking_frames = 0
        self._prelude = []      # recent frames, so a shuffle's first steps are kept
        self._idle_frames = 0
        self._recorder = None
        self.last_detections = {}
        self.last_report = None
        self.confident = None   # whether the last shuffle passed the consistency check
        self.events = []        # (frame_index, message)
        # Canvases since the last round ended, for reading a shuffle whose rows
        # aren't one of LAYOUTS (src/detection/violetta_rows.py)
        self._canvases = []
        self._canvas_frames = []    # the frame index of each saved canvas
        self._row_finder = None
        # Use the original depth-row tracking path; motion tracking is opt-in.
        self.tracker_mode = None
        self.overlay_motion_only = False
        self._tracker = None
        self._target_flow = None
        self._target_flow_frame = None
        self._target_flow_history = []
        self._flow_seed_key = None
        # Per round read by violetta_rows: {'walks': {row: ([(frame, x)], (scale, y))},
        # 'target_row': row}, for drawing what it saw (replay --output)
        self.found_walks = []

    def _log(self, message):
        self.events.append((self.frame_index, message))

    def detect_rows(self, canvas, layout='classic'):
        detections = {}
        for templates in self.layout_rows[layout]:
            profile = templates.profile(canvas, self._pool)
            x = int(np.argmax(profile))
            detections[templates.row] = (x, float(profile[x]))
        return detections

    @staticmethod
    def _walking(detections):
        return sum(score >= ROW_MATCH_THRESHOLD for _, score in detections.values())

    def _update_overlay_rows(self, layout, detections):
        if self._overlay_layout is None:
            self._overlay_layout = layout
        if layout != self._overlay_layout:
            return

        found = [(row, x, score) for row, (x, score) in detections.items()
                 if score >= ROW_MATCH_THRESHOLD]
        assigned_slots = {track['slot'] for track in self._overlay_rows.values()}
        new_rows = [(row, x, score) for row, x, score in found
                    if row not in self._overlay_rows]
        free_slots = [slot for slot in range(SLOT_COUNT) if slot not in assigned_slots]
        new_rows = sorted(new_rows, key=lambda item: item[2], reverse=True)[:len(free_slots)]
        if new_rows:
            slots = min(
                itertools.permutations(free_slots, len(new_rows)),
                key=lambda candidate: sum(
                    abs(x - slot_x(slot))
                    for (_, x, _), slot in zip(new_rows, candidate)))
            for (row, x, score), slot in zip(new_rows, slots):
                self._overlay_rows[row] = {
                    'slot': slot, 'x': x, 'score': score, 'missed': 0,
                }

        found_rows = {row: (x, score) for row, x, score in found}
        for row, track in self._overlay_rows.items():
            if row in found_rows:
                track['x'], track['score'] = found_rows[row]
                track['missed'] = 0
            else:
                track['missed'] += 1

    def _start_overlay_tracking(self, layout, detections):
        self._overlay_layout = layout
        self._overlay_rows = {}
        recorder = self._recorders[layout]
        early = [frame for frame in recorder.frames if frame[2] >= 2][
            :ShuffleRecorder.EDGE_FRAMES * 3]
        rows, starts = [], []
        for row in recorder.rows:
            start = recorder._edge_x(row, early, count=2)
            if start is not None:
                rows.append(row)
                starts.append(start)
        if starts:
            slots, _ = nearest_slots(starts)
            for row, start, slot in zip(rows, starts, slots):
                x, score = detections[row]
                self._overlay_rows[row] = {
                    'slot': slot, 'x': x if score >= ROW_MATCH_THRESHOLD else start,
                    'score': score, 'missed': 0,
                }
        self._update_overlay_rows(layout, detections)

    def _best_layout(self, canvas):
        """
        Scores every layout's rows; returns (layout, its detections, rows walking)
        for the layout with the most rows matching (ties: higher total score).
        """
        best = None
        for name in self.layout_rows:
            detections = self.detect_rows(canvas, name)
            walking = self._walking(detections)
            key = (walking, sum(score for _, score in detections.values()))
            if best is None or key > best[0]:
                best = (key, name, detections, walking)
        return best[1], best[2], best[3]

    def update(self, frame):
        """Processes one frame. Returns the answer slot (0-3) once per round, else None."""
        index = self.frame_index
        self.frame_index += 1
        answer = None

        if self.state == 'calibrate':
            # The full search tries 70 sizes and takes ~2 s on a big browser
            # window: long enough to miss a whole game. Try the size found last
            # time first (milliseconds) and fall back to the full search now and then.
            hint = getattr(self, 'scale_hint', None)
            if hint:
                self.calibration = Calibration.find(
                    frame, self.sprites, scales=np.geomspace(hint * 0.94, hint * 1.06, 7))
                if self.calibration is None and index % 10 == 9:
                    self.calibration = Calibration.find(frame, self.sprites)
            else:
                self.calibration = Calibration.find(frame, self.sprites)
            if self.calibration is not None:
                self._log(f'window found at scale {self.calibration.scale:.2f} '
                          f'(match {self.calibration.score:.2f})')
                self.state = 'reveal'
            return None

        canvas = local_contrast(self.calibration.window_canvas(frame))

        if self.state == 'reveal':
            if self._standing_scan is None:
                self._standing_scan = standing_slot_scores(
                    canvas, self._special, self._normal, self._pool)
            arrow_slot = find_reveal_arrow(frame, self.calibration)
            if arrow_slot is not None:
                self._reveal_votes.append(arrow_slot)
            else:
                slot, margin = find_special_slot(canvas, self._special, self._normal, self._pool)
                if margin > 0.05:
                    self._reveal_votes.append(slot)
            votes = self._reveal_votes[-7:]
            if len(votes) >= 5 and votes.count(max(set(votes), key=votes.count)) >= 5:
                self.target_slot = max(set(votes), key=votes.count)
                if self._target_identity is None:
                    self._target_identity = self._slot_identity[self.target_slot]
                self._log(f'make-up Violetta starts in slot {self.target_slot + 1}')
                self.state = 'wait_shuffle'
                self._start_tracker(index)
            return None

        if not self.overlay_motion_only:
            self._advance_target_flow(index, frame)
        self._feed_tracker(index, canvas)
        if not self.overlay_motion_only:
            self._seed_target_flow(index, frame)
        self._canvases.append(canvas.astype(np.float16))
        self._canvas_frames.append(index)
        del self._canvases[:-MAX_SAVED_CANVASES]
        del self._canvas_frames[:-MAX_SAVED_CANVASES]

        # Every layout is scored on every frame. Stepping into the rows, a classic
        # walker shrinks through the light rows' sizes, so which layout this
        # shuffle uses is only decided at its end, from whichever reads cleanly.
        by_layout = {name: self.detect_rows(canvas, name) for name in self.layout_rows}
        walking = {name: self._walking(d) for name, d in by_layout.items()}
        shown = max(by_layout, key=lambda name: (walking[name],
                                                 sum(s for _, s in by_layout[name].values())))
        self.last_detections = by_layout[shown]
        if self.state == 'shuffle':
            overlay_layout = self._overlay_layout or shown
            self._update_overlay_rows(overlay_layout, by_layout[overlay_layout])

        if self.state == 'wait_shuffle':
            self._prelude = (self._prelude + [(index, by_layout)])[-6:]
            self._walking_frames = (
                self._walking_frames + 1 if max(walking.values()) >= MIN_ROWS_WALKING else 0
            )
            if self._walking_frames >= 3:
                self._recorders = {
                    name: ShuffleRecorder([row for row, _, _ in LAYOUTS[name]])
                    for name in self.layout_rows
                }
                # The recent frames (this one included) hold the walk's first steps
                for prelude_index, prelude in self._prelude:
                    for name, recorder in self._recorders.items():
                        recorder.add(prelude_index, prelude[name])
                self.state = 'shuffle'
                self._idle_frames = 0
                self._shuffle_start = index
                self._target_flow_history = []
                self._start_overlay_tracking(shown, by_layout[shown])
                self._log(f'shuffle started (looks like {shown} rows)')
                return None
        if self.state == 'shuffle':
            recorders = getattr(self, '_recorders', None) or (
                {'classic': self._recorder} if self._recorder is not None else {})
            idle_counts = self.__dict__.setdefault('_idle_by_layout', {})
            all_idle, walked_idle = True, False
            for name, recorder in recorders.items():
                recorder.add(index, by_layout[name])
                # A layout's walk is over once its front-row walker has stepped
                # back too. The smaller rows alone can vanish mid-walk when all
                # four pile up behind it, but the front row is drawn on top.
                front = front_row(recorder.rows)
                front_walking = by_layout[name][front][1] >= ROW_MATCH_THRESHOLD
                idle = walking[name] < 2 and not front_walking
                idle_counts[name] = idle_counts.get(name, 0) + 1 if idle else 0
                all_idle = all_idle and idle
                # A layout that saw a real walk ends the shuffle when IT stops,
                # even if another layout still half-matches the standing Violettas
                walked = sum(f[2] >= MIN_ROWS_WALKING for f in recorder.frames)
                if walked >= 2 * ShuffleRecorder.EDGE_FRAMES and idle_counts[name] >= 8:
                    walked_idle = True
            self._idle_frames = self._idle_frames + 1 if all_idle else 0
            if walked_idle or self._idle_frames >= 8:   # walkers are back in their slots
                self._idle_by_layout = {}
                start = getattr(self, '_shuffle_start', None)
                if (start is not None and index - start < FALSE_START_FRAMES
                        and self._resolve_best()[0] is None):
                    # Over almost at once with nothing readable: Violettas
                    # stepping into or out of their rows briefly looked like a
                    # walk. Keep waiting; this round's answer is still to come.
                    self._log(f'false start ({index - start} frames, nothing readable); '
                              'still waiting for the shuffle')
                    self.state = 'wait_shuffle'
                    self._recorders, self._shuffle_start = {}, None
                    self._walking_frames = self._idle_frames = 0
                    self._overlay_layout = None
                    self._overlay_rows = {}
                    return None
                answer = self._finish_shuffle()
        return answer

    def _resolve_best(self):
        """
        Resolves the shuffle in every layout it was recorded in and keeps the best
        reading: a consistent one (exactly one stayed, slots line up), then the
        one with the longest walk seen, then the smallest slot error.
        """
        recorders = getattr(self, '_recorders', None) or (
            {'classic': self._recorder} if self._recorder is not None else {})
        if not recorders:
            return None, 'no shuffle was seen', None
        readings, reasons = [], []
        for name, recorder in recorders.items():
            mapping, report = recorder.resolve()
            if mapping is None:
                reasons.append(f'{name}: {report}')
                continue
            walked = sum(f[2] >= MIN_ROWS_WALKING for f in recorder.frames)
            readings.append(((report['consistent'], walked, -report['worst_slot_error']),
                             name, mapping, report))
        if not readings:
            return None, '; '.join(reasons), None
        _, name, mapping, report = max(readings, key=lambda r: r[0])
        report['layout'] = name
        return mapping, report, name

    # ---- frame-by-frame tracking (violetta_tracker) ----

    def _start_tracker(self, index):
        """A round begins: the four Violettas stand in their slots."""
        if self.tracker_mode is None:
            return
        self._flow_seed_key = None
        self._target_flow = None
        self._target_flow_frame = None
        self._target_flow_history = []
        try:
            if self._tracker is None:
                from src.detection import violetta_tracker as vt
                if self.tracker_mode == 'process':
                    try:    # its own process: doesn't slow this loop down
                        self._tracker = vt.ProcessTracker.shared()
                    except Exception as error:
                        self._log(f'tracker process failed ({error!r}); using a thread')
                        self.tracker_mode = 'background'
                if self._tracker is None:
                    self._tracker = (vt.BackgroundTracker(self.sprites)
                                     if self.tracker_mode == 'background'
                                     else vt.ViolettaTracker(self.sprites))
            self._tracker.start(index)
            self._tracker_started = index
        except Exception as error:
            self._log(f'tracker unavailable: {error!r}')
            self.tracker_mode = None

    def _flow_box(self, track):
        scale, x, y = track['scale'], track['x'], track['y']
        pose = int(track.get('pose', 0))
        bounds = (self._walk_union_bounds if track.get('age', 0)
                  else self._walk_pose_bounds[pose])
        left, top = self.calibration.to_frame((x + bounds[0] * scale,
                                               y + bounds[1] * scale))
        right, bottom = self.calibration.to_frame((x + bounds[2] * scale,
                                                   y + bounds[3] * scale))
        return left, top, right, bottom

    def _flow_position(self):
        flow = self._target_flow
        scale = flow.width / (85 * self.calibration.scale)
        center_x = (float(flow.center[0]) - self.calibration.origin[0]) \
            / self.calibration.scale
        center_y = (float(flow.center[1]) - self.calibration.origin[1]) \
            / self.calibration.scale
        return center_x - 23.5 * scale, center_y - 48 * scale, scale

    def _advance_target_flow(self, index, frame):
        if (self.state not in ('wait_shuffle', 'shuffle')
                or self._target_flow is None
                or self._target_flow_frame == index):
            return
        self._target_flow.update(frame)
        self._target_flow_frame = index
        if self._target_flow.visible:
            x, _, _ = self._flow_position()
            self._target_flow_history.append(
                (index, x, float(self._target_flow.confidence)))
            del self._target_flow_history[:-1000]

    def _seed_target_flow(self, index, frame):
        if (self.target_slot is None or self._tracker is None
                or self.state not in ('wait_shuffle', 'shuffle')):
            return
        if self._target_flow is not None and self._target_flow.visible:
            return
        snapshot_fn = getattr(self._tracker, 'snapshot', None)
        snapshot = snapshot_fn() if callable(snapshot_fn) else None
        if not snapshot:
            return
        track = next((item for item in snapshot['tracks']
                      if item['slot'] == self.target_slot), None)
        if (track is None or track['age'] != 0 or track['score'] is None
                or track['score'] < snapshot['threshold']):
            return
        seed_key = (self._tracker_started, track['matched'])
        if seed_key == self._flow_seed_key:
            return
        self._flow_seed_key = seed_key
        box = self._flow_box(track)
        if self._target_flow is not None:
            from src.detection.violetta_tracker import SLOT_OFFSET
            nearest_error = min(
                abs(track['x'] - SLOT_OFFSET - slot_x(slot))
                for slot in range(SLOT_COUNT))
            if (track['score'] < max(snapshot['threshold'], FLOW_REACQUIRE_SCORE)
                    or nearest_error > SLOT_STEP / 3):
                return
            self._target_flow_history = []
        from src.detection.violetta_challenge import ViolettaTracker as FeatureTracker
        try:
            self._target_flow = FeatureTracker.from_box(frame, box)
        except ValueError as error:
            self._log(f'target optical flow could not start: {error}')
            return
        self._target_flow_frame = index
        x, _, _ = self._flow_position()
        self._target_flow_history.append((index, x, float(track['score'])))
        del self._target_flow_history[:-1000]

    def _target_flow_reading(self):
        flow = self._target_flow
        history = self._target_flow_history
        if (self.target_slot is None or flow is None or not flow.visible
                or len(history) < FLOW_MIN_SAMPLES):
            return None
        recent_confidence = float(np.median([sample[2] for sample in history[-5:]]))
        max_step = max(abs(b[1] - a[1]) for a, b in zip(history, history[1:]))
        x, _, _ = self._flow_position()
        from src.detection.violetta_tracker import SLOT_OFFSET
        slot = min(range(SLOT_COUNT),
                   key=lambda candidate: abs(x - SLOT_OFFSET - slot_x(candidate)))
        error = abs(x - SLOT_OFFSET - slot_x(slot))
        if (recent_confidence < FLOW_MIN_CONFIDENCE
                or max_step > TRUST_MAX_STEP or error > TRUST_END_ERROR):
            return None
        return {
            'slot': slot,
            'error': error,
            'confidence': recent_confidence,
            'frames': len(history),
            'max_step': max_step,
        }

    def _feed_tracker(self, index, canvas):
        if self._tracker is None or self.state not in ('wait_shuffle', 'shuffle'):
            return
        if self.tracker_mode in ('background', 'process'):
            self._tracker.offer(index, canvas)
            self._last_fed = index
        elif (index - self._tracker_started) % TRACK_STEP == 0:
            self._tracker.update(index, canvas)

    def _tracked(self):
        """The tracker's ({start slot: end slot}, report), or (None, reason)."""
        if self._tracker is None:
            return None, 'not tracking'
        try:
            mapping, report = self._tracker.result()
        except Exception as error:
            return None, f'tracker failed: {error!r}'
        if not report.get('frames'):
            return None, 'tracker saw no frames'
        if report['frames'] < MIN_TRACK_FRAMES:
            report = dict(report, consistent=False)
            self._log(f'tracker saw only {report["frames"]} frames; not trusting its result')
        return mapping, report

    def _overlay_tracking_result(self):
        """A fresh motion mapping for the overlay and trusted live-answer fallback."""
        snapshot_fn = getattr(self._tracker, 'snapshot', None)
        snapshot = snapshot_fn() if callable(snapshot_fn) else None
        if (not snapshot or snapshot['last_frame'] < self._tracker_started
                or snapshot['frames'] < MIN_TRACK_FRAMES):
            return None, None
        frame_lag = max(0, self.__dict__.get('_last_fed', self.frame_index - 1)
                        - snapshot['last_frame'])
        if frame_lag > TRACK_STEP * 4 or len(snapshot['tracks']) != SLOT_COUNT:
            return None, None
        tracks = sorted(snapshot['tracks'], key=lambda track: track['slot'])
        seen = [track['seen'] for track in tracks]
        from src.detection.violetta_tracker import SLOT_OFFSET
        lost = [bool(track.get('lost', False)) for track in tracks]
        end_slots, error = end_slots_with_lost(
            [track['x'] - SLOT_OFFSET for track in tracks], lost)
        mapping = {track['slot']: end for track, end in zip(tracks, end_slots)}
        if set(mapping) != set(range(SLOT_COUNT)):
            return None, None
        nearest_end_slot = [
            min(range(SLOT_COUNT),
                key=lambda slot: abs(track['x'] - SLOT_OFFSET - slot_x(slot)))
            for track in tracks
        ]
        nearest_end_error = [
            abs(track['x'] - SLOT_OFFSET - slot_x(slot))
            for track, slot in zip(tracks, nearest_end_slot)
        ]
        span = snapshot['last_frame'] - self._tracker_started
        coverage = snapshot['frames'] / span if span > 0 else 1.0
        if self.tracker_mode not in ('background', 'process'):
            coverage = 1.0      # (replays track every TRACK_STEP-th frame on purpose)
        report = {
            'coverage': coverage,
            'consistent': bool(error < SLOT_STEP / 3 and min(seen) >= 0.5),
            'worst_slot_error': error,
            'seen': seen,
            'max_step': [track['max_step'] for track in tracks],
            'ambiguous': [track.get('identity_ambiguous', False) for track in tracks],
            'nearest_end_error': nearest_end_error,
            'nearest_end_slot': nearest_end_slot,
            'terminal_confirmed': [
                track.get('terminal_confirmed', False) for track in tracks
            ],
            'frames': snapshot['frames'],
            'lost': lost,
            'mapping': mapping,
        }
        if all('max_speed' in track for track in tracks):
            report['max_speed'] = [track['max_speed'] for track in tracks]
        return mapping, report

    def _advance_slot_identities(self, mapping):
        """Move color IDs with their Violettas after a complete, reliable shuffle."""
        new_identities = [None] * SLOT_COUNT
        for value in mapping.values():
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                return False
            start, end = value
            if not isinstance(start, (int, np.integer)) \
                    or not isinstance(end, (int, np.integer)) \
                    or not 0 <= start < SLOT_COUNT or not 0 <= end < SLOT_COUNT \
                    or new_identities[end] is not None:
                return False
            new_identities[end] = self._slot_identity[start]
        if sorted(identity for identity in new_identities if identity is not None) \
                != list(range(SLOT_COUNT)):
            return False
        self._slot_identity = new_identities
        return True

    def _catch_up_tracker(self):
        """Live: waits (up to TRACKER_CATCH_UP_SECONDS) until the tracker has
        processed every frame sent to it, so its reading covers the whole shuffle."""
        import time
        last = self.__dict__.get('_last_fed')
        if self.tracker_mode not in ('background', 'process') or last is None                 or self._tracker is None:
            return
        snapshot_fn = getattr(self._tracker, 'snapshot', None)
        if not callable(snapshot_fn):
            return
        started = time.monotonic()
        snapshot = snapshot_fn()
        behind = last - snapshot['last_frame'] if snapshot else None
        while snapshot is None or snapshot['last_frame'] < last:
            if time.monotonic() - started >= TRACKER_CATCH_UP_SECONDS:
                left = last - snapshot['last_frame'] if snapshot else '?'
                self._log(f'tracker still {left} frames behind after '
                          f'{TRACKER_CATCH_UP_SECONDS:.1f} s; reading what it has')
                return
            time.sleep(0.01)
            snapshot = snapshot_fn()
        if behind:
            self._log(f'tracker caught up {behind} queued frames in '
                      f'{time.monotonic() - started:.2f} s')

    def _finish_shuffle(self, fast=False):
        """Resolves the recorded shuffle, ends the round and returns the answer slot."""
        self._catch_up_tracker()
        target_start = self.target_slot
        mapping, report, self.layout = self._resolve_best()
        if self.overlay_motion_only:
            overlay_tracked, overlay_report = self._overlay_tracking_result()
            tracked, tracked_report, flow_reading = None, None, None
        else:
            overlay_tracked, overlay_report = None, None
            tracked, tracked_report = self._tracked()
            flow_reading = self._target_flow_reading()
        overlay_mapping = trusted_mapping_by_exclusion(overlay_tracked, overlay_report)
        overlay_answer = None
        if overlay_mapping is not None:
            overlay_answer = overlay_mapping[0][target_start]
        elif overlay_tracked is not None and trusted_track(
                overlay_tracked, overlay_report, target_start):
            overlay_answer = overlay_tracked[target_start]
        if (flow_reading is not None and tracked is not None
                and isinstance(tracked_report, dict)):
            motion_slots = tracked_report.get('nearest_end_slot')
            motion_errors = tracked_report.get('nearest_end_error')
            target = self.target_slot
            if (motion_slots is not None and motion_errors is not None
                    and target is not None and target < len(motion_slots)
                    and target < len(motion_errors)
                    and trusted_track(tracked, tracked_report, target)
                    and motion_errors[target] <= SLOT_STEP / 3
                    and motion_slots[target] != flow_reading['slot']):
                self._log(
                    f'target optical flow ended in slot {flow_reading["slot"] + 1}, '
                    f'but motion tracking places her in slot {motion_slots[target] + 1}; '
                    'not trusting the conflicting flow result')
                flow_reading = None
        fixed_ok = mapping is not None and report['consistent']
        tracked_ok = (not fixed_ok and tracked is not None and tracked_report['consistent']
                      and (fixed_ok or trusted_track(
                          tracked, tracked_report, self.target_slot)))
        if tracked_ok:
            # Clean tracks win: they don't depend on the walkers' sizes
            if fixed_ok:
                fixed_end = mapping[next(r for r, (s, _) in mapping.items()
                                         if s == self.target_slot)][1]
                if fixed_end != tracked[self.target_slot]:
                    self._log(f'{report["layout"]} rows say slot {fixed_end + 1}, tracking '
                              f'says {tracked[self.target_slot] + 1}; going with tracking')
            else:
                self._log(f'fixed layouts: {report if mapping is None else "read badly"}; '
                          'using the tracked paths')
            mapping = {f't{s}': (s, e) for s, e in tracked.items()}
            report = tracked_report
        elif not fixed_ok and flow_reading is not None:
            t = self.target_slot
            mapping = {f't{s}': (s, e) for s, e in (tracked or {}).items()}
            mapping[f't{t}'] = (t, flow_reading['slot'])
            report = dict(
                tracked_report if isinstance(tracked_report, dict) else {},
                consistent=True, layout='target optical flow')
            self._log(
                f'target optical flow tracked slot {flow_reading["slot"] + 1} '
                f'({flow_reading["frames"]} frames, '
                f'confidence {flow_reading["confidence"]:.2f}, '
                f'{flow_reading["error"]:.1f} px from slot)')
        elif not fixed_ok and trusted_track(tracked, tracked_report, self.target_slot):
            # Her own track is clean even though another one got lost (a walker
            # hidden in a spotlight): trust it rather than the slow row search
            # (violetta-20261115-140537 rounds 2 and 3: her track read 2->2 and
            # 2->3, the row search answered 1 and 2, ~2 s into the timer)
            t = self.target_slot
            nearest_end_slot = tracked_report.get('nearest_end_slot')
            end_slot = nearest_end_slot[t] if nearest_end_slot is not None \
                else tracked[t]
            nearest_end_error = tracked_report.get('nearest_end_error',
                                                   tracked_report['end_error'])
            self._log(f'her track is clean (seen {tracked_report["seen"][t]}, biggest move '
                      f'{tracked_report["max_step"][t]} px, {nearest_end_error[t]} px '
                      f'from slot {end_slot + 1}); using it (other tracks: {tracked_report["seen"]})')
            mapping = {f't{s}': (s, e) for s, e in tracked.items()}
            mapping[f't{t}'] = (t, end_slot)
            report = dict(tracked_report, consistent=True, layout='her track')
        elif not fixed_ok and overlay_answer is not None:
            if overlay_mapping is not None:
                complete_mapping, inferred_slot = overlay_mapping
                mapping = {f't{s}': (s, end) for s, end in complete_mapping.items()}
                layout = 'tracked by exclusion' if inferred_slot is not None else 'tracked'
                report = dict(overlay_report, layout=layout)
                if inferred_slot is not None:
                    identity = self._slot_identity[inferred_slot] + 1
                    end_slot = complete_mapping[inferred_slot] + 1
                    self._log(f'recovered ID{identity} by excluding the other three tracks; '
                              f'its slot is {end_slot}')
            else:
                mapping = {f't{target_start}': (target_start, overlay_answer)}
                report = dict(overlay_report, layout='trusted target track',
                              target_only=True)
            self.layout = report['layout']
            report['consistent'] = True
            self._log(f'target track is clean; using slot {overlay_answer + 1} '
                      'for the live answer')
        elif not fixed_ok and fast:
            # No trusted reading and no time for the row search. Her untrusted
            # track's end still beats her start slot: the spotlight version's
            # decoy deliberately ends there (20261119-195802 round 1: the
            # track said 3, correct; the start-slot guess clicked 4)
            t = self.target_slot
            unsure, unsure_report = ((overlay_tracked, overlay_report)
                                     if overlay_tracked is not None
                                     else (tracked, tracked_report))
            if _usable_unsure_track(unsure, unsure_report, t):
                mapping = {f't{t}': (t, unsure[t])}
                report = dict(unsure_report if isinstance(unsure_report, dict) else {},
                              consistent=False, layout='unsure target track',
                              target_only=True)
                self._log(f'tracker was not trustworthy for target slot {t + 1}; '
                          f'answering its best guess, slot {unsure[t] + 1}, rather '
                          f'than the start slot ({unsure_report})')
            elif unsure is not None:
                self._log(f'tracker was not trustworthy for target slot {t + 1}; '
                          f'skipping it rather than delaying the live answer '
                          f'({unsure_report})')
        elif not fixed_ok:
            # Nothing clean yet: find this shuffle's own rows in the saved
            # frames (slow, ~2 s); unsure tracks only when nothing else reads
            # (violetta-20261113-215235 round 1: fixed rows and the row search
            # both said 4 -> 2, the tracks had lost two walkers near the end)
            found, found_report = self._read_unknown_rows()
            use_found = found is not None and (
                mapping is None or found_report['worst_slot_error'] < report['worst_slot_error'])
            if use_found:
                old = report if mapping is None else f'{report["layout"]} rows read {report}'
                self._log(f'fixed layouts: {old}; using rows found in this shuffle instead')
                mapping, report = found, dict(found_report, layout='found')
                frames = self._canvas_frames
                self.found_walks.append({'walks': {
                    row: ([(frames[i], x) for i, x in path], geometry)
                    for row, (path, geometry) in self._row_finder.last_walks.items()},
                    'target_row': None})
            elif tracked is not None and mapping is None:
                self._log(f'no clean reading; using the tracked paths (unsure: {tracked_report})')
                mapping = {f't{s}': (s, e) for s, e in tracked.items()}
                report = tracked_report
        if (not fast and mapping is not None and isinstance(report, dict)
                and not report.get('consistent', False) and tracked is not None):
            nearest_end_slot = tracked_report.get('nearest_end_slot')
            nearest_end_error = tracked_report.get('nearest_end_error')
            t = self.target_slot
            target_row = next((row for row, (start, _) in mapping.items()
                               if start == t), None)
            if (target_row is not None and nearest_end_slot is not None
                    and nearest_end_error is not None
                    and t < len(nearest_end_slot) and t < len(nearest_end_error)
                    and nearest_end_error[t] < SLOT_STEP / 3):
                mapping[target_row] = (t, nearest_end_slot[t])
                self._log(f"using the target track's nearest slot {nearest_end_slot[t] + 1} "
                          'as a low-confidence fallback')
        self._canvases = []
        self._canvas_frames = []
        self.last_report = report
        self._recorder = None
        self._recorders = {}
        self.layout = None
        self.state = 'wait_shuffle'
        self._walking_frames = 0
        self._idle_frames = 0
        self._overlay_layout = None
        self._overlay_rows = {}
        if mapping is None:
            # Letting the timer run out fails the game outright, so
            # answer anyway: the last known slot is still a 1-in-4 guess.
            self.confident = False
            self._log(f'could not read the shuffle ({report}); '
                      f'guessing slot {self.target_slot + 1}')
        else:
            if isinstance(report, dict):
                target_row = next(r for r, (s, _) in mapping.items()
                                  if s == self.target_slot)
                target_end = mapping[target_row][1]
                if report.get('consistent') and not report.get('target_only'):
                    self._advance_slot_identities(mapping)
                elif (overlay_mapping is not None
                      and overlay_mapping[0].get(self.target_slot) == target_end):
                    color_mapping = {
                        f't{start}': (start, end)
                        for start, end in overlay_mapping[0].items()
                    }
                    if self._advance_slot_identities(color_mapping):
                        self._log('carrying color IDs with the trusted motion mapping')
                else:
                    self._log('keeping color IDs; the shuffle mapping was not trusted')
            target_row = next(r for r, (s, _) in mapping.items() if s == self.target_slot)
            if report.get('layout') == 'found' and self.found_walks:
                self.found_walks[-1]['target_row'] = target_row
            self.target_slot = mapping[target_row][1]
            self.confident = report['consistent']
            self._log(f'shuffle read {report}; target walked row {target_row}, '
                      f'now in slot {self.target_slot + 1}')
        self.rounds_left -= 1
        if self.rounds_left <= 0:
            self.state = 'done'
        else:
            self._start_tracker(self.frame_index)
        return self.target_slot

    def _read_unknown_rows(self):
        """Reads the saved frames with violetta_rows. Returns (mapping, report) or (None, reason)."""
        from src.detection.violetta_rows import RowFinder
        try:
            if self._row_finder is None:
                self._row_finder = RowFinder(self.sprites)
            mapping, report = self._row_finder.read(
                [c.astype(np.float32) for c in self._canvases])
        except Exception as error:      # a guess still beats letting the timer run out
            return None, f'row search failed: {error!r}'
        if mapping is None:
            self._log(f'row search: {report}')
        return mapping, report

    def target_track_lost(self):
        """Live: True while her track is lost (out of sight, e.g. under a
        spotlight) but was well followed until then, so waiting a moment for
        the tracker to find her again beats answering from a guess."""
        if (self.target_slot is None or self.state not in ('wait_shuffle', 'shuffle')
                or self._tracker is None):
            return False
        try:
            tracked, report = self._overlay_tracking_result()
        except Exception:
            return False
        if not isinstance(report, dict):
            return False
        t = self.target_slot
        try:
            return bool(report['lost'][t]) and report['seen'][t] >= UNSURE_MIN_SEEN
        except (KeyError, IndexError, TypeError):
            return False

    def force_answer(self, fast=False):
        """
        Call when the game is already asking for an answer but ``update`` has not
        produced one (the shuffle's start or end was missed). Uses whatever was
        recorded, else the last known slot. FAST skips the slow row search if no
        trustworthy reading is available. Returns the slot, or None if there is
        nothing to answer yet.
        """
        if self.target_slot is None or self.state not in ('wait_shuffle', 'shuffle'):
            return None
        self._log(f'the answer timer is running but the shuffle was not read to its end '
                  f'(state {self.state}, rows {self.row_summary()})')
        return self._finish_shuffle(fast=fast)

    def row_summary(self):
        """Match score of each depth row in the last frame, e.g. '0:0.71 2:0.12 ...'."""
        return ' '.join(f'{row}:{score:.2f}'
                        for row, (_, score) in self.last_detections.items()) or 'none yet'

    def overlay_shapes(self):
        """
        What to draw over the game, in frame pixels:
        ([(x0, y0, x1, y1, kind, thickness)], [(x, y, text, kind)]), with each
        idN kind tied to the same Violetta across reliable shuffles.
        """
        boxes, labels = [], []
        if self.calibration is None:
            return boxes, labels
        to_frame = self.calibration.to_frame
        moving = {}
        motion = {}
        if self.state in ('wait_shuffle', 'shuffle'):
            moving = {self._slot_identity[track['slot']]: (row, track)
                      for row, track in self._overlay_rows.items()}
            snapshot_fn = getattr(self._tracker, 'snapshot', None)
            snapshot = snapshot_fn() if callable(snapshot_fn) else None
            if snapshot and snapshot['last_frame'] >= self._tracker_started:
                frame_lag = max(0, self.frame_index - 1 - snapshot['last_frame'])
                coast = (1 - 0.8 ** frame_lag) / 0.2 if frame_lag else 0
                for track in snapshot['tracks']:
                    if not track['matched']:
                        continue
                    slot = track['slot']
                    identity = self._slot_identity[slot]
                    scale = track['predicted_scale'] * np.exp(
                        track['vlog_scale'] * coast)
                    motion[identity] = {
                        'x': track['predicted_x'] + track['vx'] * coast,
                        'y': track['predicted_y'] + track['vy'] * coast,
                        'scale': scale,
                        'pose': track['pose'],
                        'score': track['score'],
                        'age': track['age'] + frame_lag,
                        'lost': track.get('lost', False),
                    }
        # The star follows the slot she starts this round in. Colour IDs only
        # move after a trusted mapping, so the colour picked at the reveal can
        # sit on another Violetta (190834 round 4: the star was on slot 2 while
        # the solver followed her from slot 4 and answered 3)
        target_identity = (self._slot_identity[self.target_slot]
                           if self.target_slot is not None else self._target_identity)

        real = None
        for identity in range(SLOT_COUNT):
            current_slot = self._slot_identity.index(identity)
            kind = f'id{identity + 1}'
            target = '*' if identity == target_identity else ''
            if identity in motion:
                track = motion[identity]
                x0, y0, x1, y1 = self._flow_box(track)
                detail = (f' LOST, waiting {track["age"]}f' if track['lost']
                          else f' PREDICTED {track["age"]}f' if track['age']
                          else f' motion {track["score"]:.2f}')
            elif identity in moving:
                row, track = moving[identity]
                scale, pivot_y = row_geometry(row)
                x = track['x']
                x0, y0 = to_frame((x - 19 * scale, pivot_y))
                x1, y1 = to_frame((x + 66 * scale, pivot_y + 96 * scale))
                detail = f' row {row} {track["score"]:.2f}'
                if track['missed']:
                    detail += f' hold {track["missed"]}'
            else:
                left, top, right, bottom = self._standing_bounds
                x = slot_x(current_slot)
                x0, y0 = to_frame((x + left, STAND_PIVOT_Y + top))
                x1, y1 = to_frame((x + right, STAND_PIVOT_Y + bottom))
                score = (self._standing_scan[identity]
                         if self._standing_scan is not None else None)
                detail = f' scan {score:.2f}' if score is not None else ''
            boxes.append((x0, y0, x1, y1, kind,
                          max(1, int(self.calibration.scale))
                          + int(identity == target_identity)))
            labels.append((int(x0), max(0, int(y0) - 18),
                           f'ID{identity + 1}{target}{detail}', kind))
            if identity == target_identity:
                # User 2026-11-19: make the one the bot believes is real obvious:
                # a magenta frame around her box and a REAL tag above it
                pad = 4 * self.calibration.scale
                # Only a trusted reading is called REAL; after a guessed round
                # her start slot is itself a guess, so she is tagged GUESS
                sure = self.confident is not False
                kind_, text = ('target', REAL_LABEL) if sure else ('guess', GUESS_LABEL)
                real = [(x0 - pad, y0 - pad, x1 + pad, y1 + pad, kind_,
                         max(2, int(2 * self.calibration.scale))),
                        (int(x0), max(0, int(y0) - 36), text, kind_)]
        if real is not None:
            boxes.append(real[0])       # last, so it is drawn on top of the others
            labels.append(real[1])
        return boxes, labels

    def button_point(self, slot):
        """Frame-pixel reference point of answer button SLOT (the old style's centre)."""
        return self.calibration.to_frame(BUTTON_CENTRES[slot])

    def click_point(self, slot):
        """
        Where to click answer button SLOT. The 2026-11-19 buttons span 18 px above
        to 1 px below button_point (the old ones +-20 px), so aim 8 px higher: the
        middle of the new buttons and still well inside the old ones.
        """
        x, y = self.button_point(slot)
        return x, y - BUTTON_CLICK_RAISE * self.calibration.scale

    def answer_open(self, frame):
        """True while the game accepts an answer: the buttons are lit or the red timer runs."""
        return self.buttons_enabled(frame) or self.timer_visible(frame)

    def timer_visible(self, frame):
        """True while the red countdown digit (5 to 1) is shown above the stage."""
        x0, y0 = self.calibration.to_frame((TIMER_CENTRE[0] - 16, TIMER_CENTRE[1] - 22))
        x1, y1 = self.calibration.to_frame((TIMER_CENTRE[0] + 16, TIMER_CENTRE[1] + 22))
        region = frame[max(0, int(y0)):max(0, int(y1)), max(0, int(x0)):max(0, int(x1)), :3]
        if region.size == 0:
            return False
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        red = cv2.inRange(hsv, (0, 150, 120), (10, 255, 255)) | \
            cv2.inRange(hsv, (170, 150, 120), (180, 255, 255))
        # The digit covers ~32% of this box ("1", the thinnest, ~17%); 0-4% otherwise.
        return cv2.countNonZero(red) > 0.10 * red.size

    def buttons_enabled(self, frame):
        """
        True once the answer buttons light up, i.e. a click is accepted.

        Style-independent: in every version so far the buttons are grey while
        disabled and change when a click is accepted (cyan until 2026-11-19, orange
        since). So a button counts as lit when it is clearly coloured, or clearly
        brighter (e.g. white) than the grey it showed while disabled, learned from
        the frames since calibration. All four must be lit, so the cursor hovering
        one button never counts.
        """
        hsv = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2HSV)
        baseline = getattr(self, '_button_grey', None)
        if baseline is None or baseline[0] is not self.calibration:
            baseline = self._button_grey = (self.calibration, [None] * SLOT_COUNT)
        coloured_slots, brightness = [], []
        for slot in range(SLOT_COUNT):
            cx, cy = self.button_point(slot)
            half_w = BUTTON_SIZE[0] * self.calibration.scale * 0.35
            # The 2026-11-19 buttons are ~22 px tall and end ~1 px below CY (the old
            # ones spanned CY +-20); below them is the blue window frame. Sample a
            # strip inside both.
            top = cy - 9 * self.calibration.scale
            bottom = cy - 1 * self.calibration.scale
            region = hsv[int(top):int(bottom), int(cx - half_w):int(cx + half_w)]
            if region.size == 0:
                return False
            coloured = cv2.inRange(region, (0, BUTTON_LIT_SATURATION, 120), (180, 255, 255))
            coloured_slots.append(cv2.countNonZero(coloured) > 0.25 * coloured.size)
            brightness.append(float(np.median(region[:, :, 2])))
        greys = baseline[1]
        lit = 0
        for slot in range(SLOT_COUNT):
            grey = greys[slot]
            brighter = grey is not None and brightness[slot] >= min(250.0, grey + BUTTON_LIT_BRIGHTER)
            lit += coloured_slots[slot] or brighter
        if lit < SLOT_COUNT and not any(coloured_slots):
            # Disabled: remember the grey (the darkest seen, so one bright frame
            # right after calibration cannot raise it)
            for slot in range(SLOT_COUNT):
                if greys[slot] is None or brightness[slot] < greys[slot]:
                    greys[slot] = brightness[slot]
        return lit == SLOT_COUNT

    def draw(self, frame):
        """Draws all four Violetta boxes, using row tracking during shuffles."""
        output = frame[:, :, :3].copy()
        if self.calibration is None:
            return output
        boxes, labels = self.overlay_shapes()

        def colour(kind):
            return ({'target': TARGET_BGR, 'guess': GUESS_BGR}.get(kind)
                    or SLOT_BGR_COLORS[int(kind[2:]) - 1])

        for x0, y0, x1, y1, kind, thickness in boxes:
            cv2.rectangle(output, (int(x0), int(y0)), (int(x1), int(y1)),
                          colour(kind), max(1, thickness))
        for x, y, text, kind in labels:
            big = kind in ('target', 'guess')
            origin = (int(x), int(y) + 14)
            if big:     # dark outline so the tag reads on any spotlight colour
                cv2.putText(output, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(output, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                        0.55 if big else 0.42, colour(kind), 2 if big else 1, cv2.LINE_AA)
        target_identity = self._target_identity
        if target_identity is None and self.target_slot is not None:
            target_identity = self._slot_identity[self.target_slot]
        if target_identity is not None:
            target_slot = self._slot_identity.index(target_identity)
            cx, cy = self.button_point(target_slot)
            cv2.circle(output, (int(cx), int(cy)), max(4, int(18 * self.calibration.scale)),
                       (0, 255, 0), max(1, int(2 * self.calibration.scale)))
            x, y = self.calibration.to_frame((4, 22))
            if self.target_slot is not None and target_slot != self.target_slot:
                status = (f'{self.state} ID{target_identity + 1} held slot '
                          f'{target_slot + 1}; guess {self.target_slot + 1}')
            else:
                status = f'{self.state} target slot {target_slot + 1}'
            cv2.putText(output, status,
                        (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1, cv2.LINE_AA)
        return output


#################################
#      Replay and live play     #
#################################
CLICK_RIPPLE_FRAMES = 16


def _draw_click_ripple(frame, center, slot, age):
    """Draw a short simulated click ripple on a replay frame."""
    if not 0 <= age < CLICK_RIPPLE_FRAMES:
        return
    for delay in (0, 5, 10):
        wave = age - delay
        if 0 <= wave < 6:
            radius = 8 + wave * 5
            intensity = max(40, int(255 * (1 - wave / 6)))
            cv2.circle(frame, center, radius, (intensity, 0, intensity), 2, cv2.LINE_AA)
    cv2.circle(frame, center, 4, (255, 255, 255), -1, cv2.LINE_AA)
    text = f'SIM CLICK #{slot}'
    position = (max(4, center[0] - 56), max(16, center[1] - 48))
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 0, 255), 1, cv2.LINE_AA)


def replay(video_path, output_path=None):
    """Runs the solver over a recording. Returns the list of answer slots (1-4)."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f'Could not open video: {video_path}')
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    writer = None
    solver = ViolettaSolver()
    solver.tracker_mode = 'synchronous'
    solver.overlay_motion_only = True
    answers = []
    pending = None
    open_frames = closed_frames = 0
    answered = False
    frame_index = 0
    click_ripple = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            answer = solver.update(frame)
            # Same forcing rule as live play: answer phase open, nothing read
            opening = solver.calibration is not None and solver.answer_open(frame)
            open_frames = open_frames + 1 if opening else 0
            closed_frames = 0 if opening else closed_frames + 1
            if closed_frames >= TIMER_FRAMES_TO_REARM:
                answered = False
            if answer is None and pending is None and not answered \
                    and open_frames >= TIMER_FRAMES_TO_FORCE:
                answer = solver.force_answer()
            if answer is not None:
                answers.append(answer + 1)
                pending = answer
            if pending is not None and solver.answer_open(frame):   # same rule as live play
                answered = True
                if solver.confident:
                    x, y = solver.click_point(pending)
                    solver._log(f'would click button {pending + 1} at frame pixel ({x:.0f}, {y:.0f}) '
                                f'[buttons lit: {solver.buttons_enabled(frame)}, '
                                f'red timer: {solver.timer_visible(frame)}]')
                    click_ripple = (frame_index, int(x), int(y), pending + 1)
                else:
                    solver._log(f'low-confidence answer {pending + 1}; live mode would not click')
                pending = None
            if output_path:
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'),
                                             fps, (width, height))
                annotated = solver.draw(frame)
                if click_ripple:
                    start, x, y, slot = click_ripple
                    _draw_click_ripple(annotated, (x, y), slot, frame_index - start)
                writer.write(annotated)
            frame_index += 1
    finally:
        capture.release()
        if writer:
            writer.release()
    for frame_index, message in solver.events:
        print(f'[VIOLETTA] {frame_index / fps:6.2f}s  {message}')
    print(f'[VIOLETTA] Answers: {answers}')
    return answers


def _phone(text):
    """A Telegram alert with a screenshot (src/modules/phone_alert.py), if set up."""
    try:
        from src.modules import phone_alert
        if phone_alert.wants('mini_game_start'):
            phone_alert.send(text, photo=True)
    except Exception:
        pass


def _client_region(title):
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32

    def client(window):
        rect = wintypes.RECT()
        user32.GetClientRect(window, ctypes.byref(rect))
        origin = wintypes.POINT(0, 0)
        user32.ClientToScreen(window, ctypes.byref(origin))
        return {'left': origin.x, 'top': origin.y,
                'width': rect.right - rect.left, 'height': rect.bottom - rect.top}

    def usable(window):     # shown, not minimised, and big enough to hold the game
        if not window or not user32.IsWindowVisible(window) or user32.IsIconic(window):
            return False
        region = client(window)
        return region['width'] >= 200 and region['height'] >= 150

    handle = user32.FindWindowW(None, title)
    if not usable(handle):
        # No usable exact match: take the biggest visible, not-minimised window
        # whose title contains TITLE (a browser adds its own name to the page
        # title, and the same page may be open in two browsers).
        found = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        def visit(window, _):
            text = ctypes.create_unicode_buffer(300)
            user32.GetWindowTextW(window, text, 300)
            if title.lower() in text.value.lower() and usable(window):
                found.append(window)
            return True

        user32.EnumWindows(visit, 0)
        handle = max(found, key=lambda w: client(w)['width'] * client(w)['height']) \
            if found else None
    if not handle:
        raise RuntimeError(f'Could not find an open (not minimised) window titled {title!r}.')
    return handle, client(handle)


def _click(x, y):
    # Mini-games run while Auto Maple is paused: the click must still go through.
    from src.common import vkeys
    vkeys.click((x, y), settle=(0.05, 0.10), hold=(0.06, 0.12), allow_paused=True)


PARK_OFFSET = 40            # window px beyond the game window's edge to park the cursor
CLICK_CONFIRM_SECONDS = 0.7 # all buttons still lit this long after a click: it was missed
CLICK_RETRIES = 2           # re-clicks of the same slot for a missed click
POST_GAME_SECONDS = 4.0     # keep watching (and recording) after the last click


def park_point(calibration, region):
    """
    Client-pixel spot for the idle cursor: beside the game window (right, else
    left, else below), so it covers no Violetta, button or title strip.
    20261119-225627 round 1: the cursor sat over slot 4 for the whole shuffle,
    that track was lost and the answer only came from excluding the other three
    (a replay of the same footage got it wrong). None if no spot fits.
    """
    y = BUTTON_CENTRES[0][1]
    for point in ((WINDOW_SIZE[0] + PARK_OFFSET, y), (-PARK_OFFSET, y),
                  (WINDOW_SIZE[0] / 2, WINDOW_SIZE[1] + PARK_OFFSET)):
        x, fy = calibration.to_frame(point)
        if 0 <= x < region['width'] and 0 <= fy < region['height']:
            return x, fy
    return None


def _park_cursor(calibration, region, handle):
    """Moves the cursor off the game window (only while the game is focused)."""
    import ctypes
    if ctypes.windll.user32.GetForegroundWindow() != handle:
        return None
    point = park_point(calibration, region)
    if point is None:
        return None
    from src.common import vkeys
    screen_point = (region['left'] + point[0], region['top'] + point[1])
    return screen_point if vkeys.move_cursor(*screen_point) else None


def _send_answer_click(solver, slot, region, handle):
    import ctypes
    import random
    import time
    if ctypes.windll.user32.GetForegroundWindow() != handle:
        return None
    x, y = solver.click_point(slot)
    point = (region['left'] + x, region['top'] + y)
    time.sleep(0.08 + 0.12 * random.random())
    _click(*point)
    return point


class _LiveRecording:
    """Saves the game window from each live frame, to replay with --video."""

    def __init__(self, directory, fps=20.0):
        import time
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, time.strftime('violetta-%Y%m%d-%H%M%S.mp4'))
        self.fps = fps
        self.box = None
        self.writer = None
        self.frames = 0
        self.stills = 0
        self.last_still = None

    STILL_SECONDS = 2.0     # while not calibrated: one full-window JPEG this often
    STILL_LIMIT = 60

    def add_uncalibrated(self, frame):
        """
        Before calibration there is no stage box to crop to, so nothing goes into the
        video. Keeps a full-window still now and then instead, so a stage the solver
        cannot find (e.g. a redesigned one) can still be examined afterwards.
        """
        import time
        now = time.monotonic()
        if self.stills >= self.STILL_LIMIT or (
                self.last_still is not None and now - self.last_still < self.STILL_SECONDS):
            return
        self.last_still = now
        folder = self.path[:-4] + '-uncalibrated'
        try:
            os.makedirs(folder, exist_ok=True)
            cv2.imwrite(os.path.join(folder, f'{self.stills:02d}.jpg'), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            self.stills += 1
        except Exception:
            self.stills = self.STILL_LIMIT     # disk trouble: stop trying

    def video_time(self):
        """Where the current moment is in the saved video (s), for matching logs to it."""
        return self.frames / self.fps

    def add(self, frame, calibration, pad=24):
        if self.box is None:
            x0, y0 = calibration.to_frame((0, 0))
            x1, y1 = calibration.to_frame(WINDOW_SIZE)
            height, width = frame.shape[:2]
            self.box = (max(0, int(x0) - pad), max(0, int(y0) - pad),
                        min(width, int(x1) + pad), min(height, int(y1) + pad))
        x0, y0, x1, y1 = self.box
        crop = frame[y0:y1, x0:x1]
        if crop.shape[0] != y1 - y0 or crop.shape[1] != x1 - x0:
            return      # the game window was resized
        if self.writer is None:
            self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*'mp4v'),
                                          self.fps, (x1 - x0, y1 - y0))
        self.writer.write(np.ascontiguousarray(crop))
        self.frames += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
            print(f'[VIOLETTA] Saved what the solver saw to {self.path}')
        if self.stills:
            print(f'[VIOLETTA] Saved {self.stills} uncalibrated stills to {self.path[:-4]}-uncalibrated')


def _open_overlay():
    try:
        from src.detection.screen_overlay import ScreenOverlay
        return ScreenOverlay()
    except Exception as error:      # the game still gets solved without the boxes
        print(f'[VIOLETTA] Overlay unavailable: {error}')
        return None


def _draw_overlay(view, solver, region, note):
    from src.detection import screen_overlay as so
    colours = {
        'id1': so.RED, 'id2': so.GREEN, 'id3': so.CYAN, 'id4': so.YELLOW,
        'target': so.MAGENTA, 'guess': so.WHITE,
    }
    boxes, labels = solver.overlay_shapes()
    boxes = [(x0, y0, x1, y1, colours[kind], t)
             for x0, y0, x1, y1, kind, t in boxes]
    labels = [(x, y, text, colours[kind])
              for x, y, text, kind in labels]
    if solver.calibration is None:
        labels.append((10, 10, 'VIOLETTA: looking for the game window...', so.RED))
    elif note:
        x, y = solver.calibration.to_frame((4, 22))
        labels.append((x, y, note, so.WHITE))
    view.show(region, boxes, labels)


TRACK_STEP = 3                  # sync tracking: every this many frames
TIMER_FRAMES_TO_FORCE = 4       # ignore brief blue-button flashes before answering
TIMER_FRAMES_TO_REARM = 10      # frames it must be gone before the next round
SLOT_BGR_COLORS = ((0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255))
TARGET_BGR = (255, 0, 255)      # magenta: the Violetta the bot thinks is real
REAL_LABEL = "REAL (bot's pick)"
GUESS_BGR = (255, 255, 255)     # white: her slot is only a guess this round
GUESS_LABEL = 'GUESS (bot not sure)'
FLOW_MIN_SAMPLES = 6            # consecutive optical-flow observations before trusting its slot
FLOW_MIN_CONFIDENCE = 0.3
FLOW_REACQUIRE_SCORE = 0.58
# The window's title strip matches its calibration-time look at ~0.97 while open
# and ~0.0 once the game has closed it (failed, or all rounds done).
WINDOW_CHECK_SECONDS = 1.0
WINDOW_GONE_SCORE = 0.6
WINDOW_GONE_CHECKS = 3


SCALE_HINTS_PATH = os.path.join('.settings', 'violetta_scales.json')


def _load_scale_hint(title):
    import json
    try:
        with open(SCALE_HINTS_PATH, encoding='utf-8') as file:
            return float(json.load(file).get(title))
    except (OSError, ValueError, TypeError):
        return None


def _save_scale_hint(title, scale):
    import json
    try:
        with open(SCALE_HINTS_PATH, encoding='utf-8') as file:
            hints = json.load(file)
    except (OSError, ValueError):
        hints = {}
    hints[title] = round(float(scale), 3)
    try:
        os.makedirs(os.path.dirname(SCALE_HINTS_PATH), exist_ok=True)
        with open(SCALE_HINTS_PATH, 'w', encoding='utf-8') as file:
            json.dump(hints, file, indent=1)
    except OSError:
        pass


def _title_strip(frame, calibration):
    x0, y0 = calibration.to_frame((0, 0))
    x1, y1 = calibration.to_frame((WINDOW_SIZE[0], 24))
    strip = frame[max(0, int(y0)):max(0, int(y1)), max(0, int(x0)):max(0, int(x1)), :3]
    return cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY) if strip.size else None


def _window_still_open(frame, calibration, reference):
    strip = _title_strip(frame, calibration)
    if strip is None or reference is None or strip.shape != reference.shape:
        return True     # can't tell: keep going
    return float(cv2.matchTemplate(strip, reference, cv2.TM_CCOEFF_NORMED)[0][0]) \
        >= WINDOW_GONE_SCORE


# The player can drag the mini-game window (20261120-123844: it moved down a
# few px after round 3, the title strip matched 0.01 at its old place, and
# the bot stopped with a round left as if the window had closed).
WINDOW_MOVE_SEARCH = 80         # px around the old place to look for the title strip
WINDOW_MOVED_SCORE = 0.8


def _find_moved_window(frame, calibration, reference, search=WINDOW_MOVE_SEARCH):
    """Where the title strip went: (dx, dy) in frame px from its old place, or
    None when it is nowhere near (the window really closed)."""
    if reference is None or frame is None:
        return None
    x0, y0 = (int(v) for v in calibration.to_frame((0, 0)))
    h, w = reference.shape[:2]
    left, top = max(0, x0 - search), max(0, y0 - search)
    right = min(frame.shape[1], x0 + w + search)
    bottom = min(frame.shape[0], y0 + h + search)
    area = frame[top:bottom, left:right, :3]
    if area.shape[0] < h or area.shape[1] < w:
        return None
    scores = cv2.matchTemplate(cv2.cvtColor(area, cv2.COLOR_BGR2GRAY), reference,
                               cv2.TM_CCOEFF_NORMED)
    _, best, _, (bx, by) = cv2.minMaxLoc(scores)
    if best < WINDOW_MOVED_SCORE:
        return None
    dx, dy = left + bx - x0, top + by - y0
    return (dx, dy) if (dx, dy) != (0, 0) else None


def play_live(title='MapleStory', select=False, timeout=120.0, overlay=True, record_dir=None):
    """
    Watches the game window and, with SELECT, clicks each answer once the
    buttons light up. Start it before the opening reveal ends.

    OVERLAY draws all four slot boxes on top of the game, using the row tracker
    to move them during shuffles. RECORD_DIR saves frames for replay with --video.
    """
    import ctypes
    import random
    import time
    import mss

    ctypes.windll.user32.SetProcessDPIAware()
    from src.detection.violetta_tracker import source_stamp
    if source_stamp() > _LOADED_AT:
        print('[VIOLETTA] WARNING: the Violetta code changed on disk after Auto Maple '
              'started. The tracker process restarts by itself, but restart Auto Maple '
              'to use the rest of the changes.')
    waited = time.monotonic()
    while True:     # the window may be minimised or not open yet: wait for it
        try:
            handle, region = _client_region(title)
            break
        except RuntimeError as error:
            if time.monotonic() - waited > timeout:
                raise
            if int(time.monotonic() - waited) % 5 == 0:
                print(f'[VIOLETTA] {error} Waiting for it...')
            time.sleep(1.0)
    solver = ViolettaSolver()
    solver.tracker_mode = 'process'
    solver.overlay_motion_only = True
    solver.scale_hint = _load_scale_hint(title)
    pending = None
    logged = 0
    timer_frames = timer_gone = 0
    opened_at = None            # when this round's answer buttons lit
    answered = False            # this round's answer was clicked (or announced)
    note = ''
    frames = 0
    started = status_time = time.monotonic()
    view = _open_overlay() if overlay else None
    recording = _LiveRecording(record_dir) if record_dir else None
    print(f'[VIOLETTA] Watching {title!r}; clicking is {"ON" if select else "OFF (preview)"}; '
          f'overlay boxes {"ON" if view else "OFF" if not overlay else "FAILED to open"}'
          f'{f"; recording to {recording.path}" if recording else ""}.')
    title_reference = None
    window_check = time.monotonic()
    window_misses = 0
    parked = False
    last_click = None           # {'slot', 'time', 'retries'} until the buttons go dark
    done_at = None              # when the last round's answer was clicked

    def tag():
        # Position in the saved recording, to jump straight to this moment in it
        return f'[video {recording.video_time():5.1f}s] ' if recording else ''

    try:
        with mss.mss() as screen:
            while time.monotonic() - started < timeout:
                try:
                    handle, region = _client_region(title)
                except RuntimeError:
                    pass    # title changed for a moment (tab switch etc.): keep the last window
                try:
                    frame = np.asarray(screen.grab(region))[:, :, :3]
                except Exception:   # minimised or moved off-screen: try again shortly
                    time.sleep(0.2)
                    continue
                answer = solver.update(frame)
                frames += 1
                calibrated = solver.calibration is not None
                if recording and calibrated:
                    recording.add(frame, solver.calibration)
                elif recording:
                    recording.add_uncalibrated(frame)

                # The game has closed the window (failed, or finished): stop instead
                # of staring at the map until the timeout
                now = time.monotonic()
                if calibrated and title_reference is None:
                    title_reference = _title_strip(frame, solver.calibration)
                    _save_scale_hint(title, solver.calibration.scale)
                if calibrated and select and not parked:
                    # A cursor over the stage hides a Violetta from the tracker.
                    # Only here, before any round: moving it straight after an
                    # answer click made the game drop the click (20261120-002414:
                    # round 1 took 3 clicks, round 2 none landed, timer ran out,
                    # jailed). After a click it rests on the buttons, under the stage.
                    parked = True
                    spot = _park_cursor(solver.calibration, region, handle)
                    print(f'[VIOLETTA] {tag()}' + (
                        f'Moved the cursor off the stage to screen {spot}.' if spot else
                        'Could not move the cursor off the stage (window not focused?).'))

                # A click the game did not take leaves all four buttons lit:
                # click the same slot again before the timer runs out.
                if last_click is not None and calibrated:
                    since = now - last_click['time']
                    if not solver.buttons_enabled(frame):
                        print(f'[VIOLETTA] {tag()}Answer {last_click["slot"] + 1} accepted '
                              f'(buttons went dark {since:.1f} s after the click).')
                        last_click = None
                    elif since >= CLICK_CONFIRM_SECONDS:
                        if last_click['retries'] >= CLICK_RETRIES:
                            print(f'[VIOLETTA] {tag()}Buttons still lit after '
                                  f'{CLICK_RETRIES + 1} clicks on answer '
                                  f'{last_click["slot"] + 1}; giving up on it.')
                            last_click = None
                        else:
                            point = _send_answer_click(solver, last_click['slot'], region, handle)
                            last_click['retries'] += 1
                            last_click['time'] = time.monotonic()
                            print(f'[VIOLETTA] {tag()}Buttons still lit {since:.1f} s after '
                                  f'the click: clicking answer {last_click["slot"] + 1} again '
                                  f'(retry {last_click["retries"]}) at screen {point}.')

                if done_at is not None and last_click is None \
                        and now - done_at >= POST_GAME_SECONDS:
                    print(f'[VIOLETTA] {tag()}Last answer clicked {now - done_at:.1f} s ago; '
                          'done watching.')
                    return
                if calibrated and now - window_check >= WINDOW_CHECK_SECONDS:
                    window_check = now
                    if _window_still_open(frame, solver.calibration, title_reference):
                        window_misses = 0
                    else:
                        moved = _find_moved_window(frame, solver.calibration, title_reference)
                        if moved is not None:
                            # Dragged, not closed: follow it (tracking, clicks
                            # and the overlay are all placed from the origin)
                            ox, oy = solver.calibration.origin
                            solver.calibration.origin = (ox + moved[0], oy + moved[1])
                            window_misses = 0
                            print(f'[VIOLETTA] {tag()}The game window moved by '
                                  f'({moved[0]:+d}, {moved[1]:+d}) px; following it.')
                        else:
                            window_misses += 1
                    if window_misses >= WINDOW_GONE_CHECKS and solver.state in ('calibrate', 'reveal'):
                        # Found only as a game was ending (no round played):
                        # start over and wait for the next game
                        print(f'[VIOLETTA] {tag()}The window closed before any round was '
                              'played; watching for the next game.')
                        solver = ViolettaSolver(solver.sprites)
                        solver.tracker_mode = 'process'
                        solver.overlay_motion_only = True
                        solver.scale_hint = _load_scale_hint(title)
                        title_reference, window_misses, logged = None, 0, 0
                        pending, answered, timer_frames, timer_gone = None, False, 0, 0
                        continue
                    if window_misses >= WINDOW_GONE_CHECKS:
                        print(f'[VIOLETTA] {tag()}The game window has closed '
                              f'(state {solver.state!r}, {solver.rounds_left} round(s) left); '
                              'stopping.')
                        return

                # If the answer phase is already open (buttons lit, or the red
                # countdown running) and the shuffle was never read to its end,
                # answer with the best guess: letting the timer run out always fails.
                opening = calibrated and solver.answer_open(frame)
                timer_frames = timer_frames + 1 if opening else 0
                timer_gone = 0 if opening else timer_gone + 1
                if timer_gone >= TIMER_FRAMES_TO_REARM:
                    answered = False
                if not opening:
                    opened_at = None
                elif opened_at is None:
                    opened_at = now
                forced_answer = False
                waiting_for_her = False
                if answer is None and pending is None and not answered \
                        and timer_frames >= TIMER_FRAMES_TO_FORCE \
                        and now - opened_at < LOST_TRACK_GRACE_SECONDS \
                        and solver.target_track_lost():
                    # Her track went out of sight as the shuffle ended (a
                    # spotlight): she stands still now, so the tracker usually
                    # finds her again within a few frames. Keep feeding it.
                    waiting_for_her = True
                    if note != 'waiting for her track':
                        note = 'waiting for her track'
                        print(f'[VIOLETTA] {tag()}Buttons are lit but her track is lost; '
                              f'tracking up to {LOST_TRACK_GRACE_SECONDS:.1f} s more '
                              'before answering.')
                if answer is None and pending is None and not answered \
                        and not waiting_for_her \
                        and timer_frames >= TIMER_FRAMES_TO_FORCE:
                    if note == 'waiting for her track':
                        found_again = not solver.target_track_lost()
                        print(f'[VIOLETTA] {tag()}Her track was '
                              f'{"found again" if found_again else "not found again"} '
                              f'after {now - opened_at:.1f} s.')
                    answer = solver.force_answer(fast=True)
                    forced_answer = answer is not None
                    found = isinstance(solver.last_report, dict) and \
                        solver.last_report.get('layout') == 'found'
                    if answer is not None and solver.confident:
                        layout = solver.last_report.get('layout', 'shuffle') \
                            if isinstance(solver.last_report, dict) else 'shuffle'
                        print(f'[VIOLETTA] {tag()}Answer buttons are lit; trusted {layout} '
                              f'reading: slot {answer + 1}.')
                    elif answer is not None and found:
                        print(f'[VIOLETTA] {tag()}Answer buttons are lit; low-confidence row '
                              f'reading ({solver.last_report.get("ms")} ms): slot {answer + 1}.')
                    elif answer is not None:
                        print(f'[VIOLETTA] {tag()}Answer buttons are lit but no trusted '
                              f'shuffle read was available; guessing slot {answer + 1}.')

                for _, message in solver.events[logged:]:
                    print(f'[VIOLETTA] {tag()}{message}')
                logged = len(solver.events)
                if now - status_time >= 2.0:
                    rate = frames / (now - status_time)
                    if solver.state in ('wait_shuffle', 'shuffle'):
                        print(f'[VIOLETTA] {tag()}{solver.state}: rows {solver.row_summary()} '
                              f'(a shuffle needs {MIN_ROWS_WALKING} rows >= {ROW_MATCH_THRESHOLD}), '
                              f'{rate:.0f} fps')
                    elif solver.state in ('calibrate', 'reveal'):
                        print(f'[VIOLETTA] {tag()}{solver.state}: still looking, {rate:.1f} fps')
                    status_time, frames = now, 0

                if answer is not None:
                    pending = answer
                answer_frame = frame
                answer_capture_failed = False
                if pending is not None:
                    try:
                        answer_frame = np.asarray(screen.grab(region))[:, :, :3]
                    except Exception as error:
                        answer_frame = None
                        answer_capture_failed = True
                        note = f'answer {pending + 1}: not clicked, screen capture failed'
                        print(f'[VIOLETTA] {tag()}Could not recheck the answer screen: {error!r}')
                if pending is not None and answer_frame is not None \
                        and solver.answer_open(answer_frame):
                    low = '' if solver.confident else ' (low confidence)'
                    if select:
                        if not solver.confident:
                            print(f'[VIOLETTA] {tag()}Low-confidence answer {pending + 1}; '
                                  'sending the best available slot to avoid timing out.')
                        point = _send_answer_click(solver, pending, region, handle)
                        if point is None:
                            note = f'answer {pending + 1}: NOT clicked, game window not focused'
                            print(f'[VIOLETTA] {tag()}Game window is not focused; not clicking.')
                        else:
                            x, y = point
                            note = f'click sent for answer {pending + 1}{low}'
                            print(f'[VIOLETTA] {tag()}Sent click for answer {pending + 1}{low} at screen '
                                  f'({x:.0f}, {y:.0f}).')
                            last_click = {'slot': pending, 'time': time.monotonic(),
                                          'retries': 0}
                    else:
                        note = f'answer is {pending + 1}{low}'
                        print(f'[VIOLETTA] {tag()}Answer is {pending + 1}{low}.')
                    pending = None
                    answered = True
                    if solver.state == 'done' and done_at is None:
                        # Keep watching: confirm the last click landed and record
                        # the result (the recording used to stop 1 s after it,
                        # before the last round's tick or cross was shown)
                        done_at = time.monotonic()
                elif pending is not None and forced_answer:
                    if not answer_capture_failed:
                        note = f'answer {pending + 1}: NOT clicked, answer window closed while reading'
                        print(f'[VIOLETTA] {tag()}Answer window closed while reading; '
                              'not sending a late click.')
                    pending = None
                    answered = True
                elif pending is not None and answer_frame is not None:
                    note = f'answer {pending + 1}: waiting for the buttons to light up'
                if view:
                    _draw_overlay(view, solver, region, note)
        print(f'[VIOLETTA] {tag()}Timed out in state {solver.state!r} (rows {solver.row_summary()}).')
    finally:
        if view:
            view.close()
        if recording:
            recording.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Follow the make-up Violetta and pick her slot.')
    parser.add_argument('--video', help='Replay a recording instead of watching the game.')
    parser.add_argument('--output', help='With --video: write an annotated copy here.')
    parser.add_argument('--title', default='MapleStory', help='Game window title.')
    parser.add_argument('--select', action='store_true', help='Live: click the answers.')
    parser.add_argument('--timeout', type=float, default=120.0, help='Live: seconds to run.')
    parser.add_argument('--overlay', dest='overlay', action='store_true', default=True,
                        help='Live: show all four tracking boxes (default).')
    parser.add_argument('--no-overlay', dest='overlay', action='store_false',
                        help='Live: disable the tracking boxes.')
    parser.add_argument('--record', metavar='DIR', help='Live: save the frames seen to DIR.')
    parser.add_argument('--track-step', type=int, default=TRACK_STEP,
                        help='With --video: track every Nth frame (default %(default)s). '
                             'Use 1 for recordings made at ~10 fps, where the live '
                             'tracker kept up with every frame.')
    args = parser.parse_args()
    if args.video:
        globals()['TRACK_STEP'] = max(1, args.track_step)
        replay(args.video, args.output)
    else:
        play_live(args.title, args.select, args.timeout, args.overlay, args.record)


if __name__ == '__main__':
    main()
