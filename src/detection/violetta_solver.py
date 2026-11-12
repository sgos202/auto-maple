"""
Solve the Violetta lie-detector mini-game ("keep your eyes on the Violetta who's
wearing make-up") from screen frames.

How the game moves the four Violettas (verified against recorded footage; the
geometry comes from https://github.com/hikaricalyx/violetta-sim, MIT):

* All coordinates below are in *window space*: the 441x318 lie-detector window
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
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np


ASSET_DIR = os.path.join('assets', 'violetta')

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
TIMER_CENTRE = (222.0, 113.0)   # the red countdown digit shown while an answer is accepted

ROW_MATCH_THRESHOLD = 0.5   # walking sprite found in a row band
MIN_ROWS_WALKING = 3        # rows that must match at once to count as shuffling


def slot_x(slot):
    return SLOT_PIVOT_X + SLOT_STEP * slot


def row_geometry(row):
    """Returns (scale, pivot y) of a Violetta walking in ROW."""
    return DEPTH_SCALE ** (row - STAND_ROW), STAND_PIVOT_Y + (STAND_ROW - row) * DEPTH_STEP


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
        super().__init__(sprites.walk, *row_geometry(row))


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


class ShuffleRecorder:
    """Collects row detections during one shuffle and resolves it."""

    EDGE_FRAMES = 6     # frames at each end of the walk used to read slots

    def __init__(self):
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
        rows = list(WALK_ROWS)
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


class ViolettaSolver:
    """
    Feed it frames in order with ``update(frame)``. It reports the target's slot
    after every shuffle and says when an answer should be clicked.
    """

    def __init__(self, sprites=None, rounds=4):
        self.sprites = sprites or Sprites()
        self.rows = [_RowTemplates(self.sprites, row) for row in WALK_ROWS]
        self._special = _ScaledTemplates(self.sprites.special, 1.0, STAND_PIVOT_Y)
        self._normal = _ScaledTemplates(self.sprites.stand, 1.0, STAND_PIVOT_Y)
        self._pool = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1))
        self.calibration = None
        self.target_slot = None
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

    def _log(self, message):
        self.events.append((self.frame_index, message))

    def detect_rows(self, canvas):
        detections = {}
        for templates in self.rows:
            profile = templates.profile(canvas, self._pool)
            x = int(np.argmax(profile))
            detections[templates.row] = (x, float(profile[x]))
        return detections

    def update(self, frame):
        """Processes one frame. Returns the answer slot (0-3) once per round, else None."""
        index = self.frame_index
        self.frame_index += 1
        answer = None

        if self.state == 'calibrate':
            self.calibration = Calibration.find(frame, self.sprites)
            if self.calibration is not None:
                self._log(f'window found at scale {self.calibration.scale:.2f} '
                          f'(match {self.calibration.score:.2f})')
                self.state = 'reveal'
            return None

        canvas = local_contrast(self.calibration.window_canvas(frame))

        if self.state == 'reveal':
            slot, margin = find_special_slot(canvas, self._special, self._normal, self._pool)
            if margin > 0.05:
                self._reveal_votes.append(slot)
            votes = self._reveal_votes[-7:]
            if len(votes) >= 5 and votes.count(max(set(votes), key=votes.count)) >= 5:
                self.target_slot = max(set(votes), key=votes.count)
                self._log(f'make-up Violetta starts in slot {self.target_slot + 1}')
                self.state = 'wait_shuffle'
            return None

        detections = self.detect_rows(canvas)
        self.last_detections = detections
        walking = sum(score >= ROW_MATCH_THRESHOLD for _, score in detections.values())

        if self.state == 'wait_shuffle':
            self._prelude = (self._prelude + [(index, detections)])[-6:]
            self._walking_frames = self._walking_frames + 1 if walking >= MIN_ROWS_WALKING else 0
            if self._walking_frames >= 3:
                self._recorder = ShuffleRecorder()
                for prelude_index, prelude in self._prelude[:-1]:
                    self._recorder.add(prelude_index, prelude)
                self.state = 'shuffle'
                self._idle_frames = 0
                self._log('shuffle started')
        if self.state == 'shuffle':
            self._recorder.add(index, detections)
            # The walk is over once the front-row walker has stepped back too. The
            # smaller rows alone can vanish mid-walk when all four pile up behind
            # it, but the front row is drawn on top of everything.
            front_walking = detections[WALK_ROWS[0]][1] >= ROW_MATCH_THRESHOLD
            idle = walking < 2 and not front_walking
            self._idle_frames = self._idle_frames + 1 if idle else 0
            if self._idle_frames >= 8:      # walkers have stepped back into their slots
                mapping, report = self._recorder.resolve()
                self.last_report = report
                self.state = 'wait_shuffle'
                self._walking_frames = 0
                if mapping is None:
                    # Letting the timer run out fails the game outright, so
                    # answer anyway: the last known slot is still a 1-in-4 guess.
                    self.confident = False
                    self._log(f'could not read the shuffle ({report}); '
                              f'guessing slot {self.target_slot + 1}')
                else:
                    target_row = next(r for r, (s, _) in mapping.items() if s == self.target_slot)
                    self.target_slot = mapping[target_row][1]
                    self.confident = report['consistent']
                    self._log(f'shuffle read {report}; target walked row {target_row}, '
                              f'now in slot {self.target_slot + 1}')
                answer = self.target_slot
                self.rounds_left -= 1
                if self.rounds_left <= 0:
                    self.state = 'done'
        return answer

    def button_point(self, slot):
        """Frame-pixel centre of answer button SLOT."""
        return self.calibration.to_frame(BUTTON_CENTRES[slot])

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
        """True once the answer buttons light up (cyan), i.e. a click is accepted."""
        hsv = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2HSV)
        lit = 0
        for slot in range(SLOT_COUNT):
            cx, cy = self.button_point(slot)
            half_w = BUTTON_SIZE[0] * self.calibration.scale * 0.35
            half_h = BUTTON_SIZE[1] * self.calibration.scale * 0.25
            region = hsv[int(cy - half_h):int(cy + half_h), int(cx - half_w):int(cx + half_w)]
            if region.size == 0:
                return False
            cyan = cv2.inRange(region, (80, 120, 120), (100, 255, 255))
            lit += cv2.countNonZero(cyan) > 0.25 * cyan.size
        return lit == SLOT_COUNT

    def draw(self, frame):
        """Annotates FRAME with the calibration, row detections and target slot."""
        output = frame[:, :, :3].copy()
        if self.calibration is None:
            return output
        s = self.calibration.scale
        colours = {0: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0), 4: (0, 255, 255)}
        for row, (x, score) in self.last_detections.items():
            if score < ROW_MATCH_THRESHOLD or self.state != 'shuffle':
                continue
            scale, pivot_y = row_geometry(row)
            top_left = self.calibration.to_frame((x - 19 * scale, pivot_y))
            bottom_right = self.calibration.to_frame((x + 66 * scale, pivot_y + 96 * scale))
            cv2.rectangle(output, tuple(map(int, top_left)), tuple(map(int, bottom_right)),
                          colours[row], max(1, int(s / 2)))
        if self.target_slot is not None:
            cx, cy = self.button_point(self.target_slot)
            cv2.circle(output, (int(cx), int(cy)), int(14 * s), (0, 255, 0), max(2, int(s)))
        label = f'{self.state}  target slot: ' + \
            (str(self.target_slot + 1) if self.target_slot is not None else '?')
        cv2.putText(output, label, (10, int(12 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.25 * s, (0, 0, 255), max(1, int(s / 2)))
        return output


#################################
#      Replay and live play     #
#################################
def replay(video_path, output_path=None):
    """Runs the solver over a recording. Returns the list of answer slots (1-4)."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f'Could not open video: {video_path}')
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    writer = None
    solver = ViolettaSolver()
    answers = []
    pending = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            answer = solver.update(frame)
            if answer is not None:
                answers.append(answer + 1)
                pending = answer
            if pending is not None and solver.answer_open(frame):   # same rule as live play
                x, y = solver.button_point(pending)
                solver._log(f'would click button {pending + 1} at frame pixel ({x:.0f}, {y:.0f}) '
                            f'[buttons lit: {solver.buttons_enabled(frame)}, '
                            f'red timer: {solver.timer_visible(frame)}]')
                pending = None
            if output_path:
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'),
                                             fps, (width, height))
                writer.write(solver.draw(frame))
    finally:
        capture.release()
        if writer:
            writer.release()
    for frame_index, message in solver.events:
        print(f'[VIOLETTA] {frame_index / fps:6.2f}s  {message}')
    print(f'[VIOLETTA] Answers: {answers}')
    return answers


def _client_region(title):
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    handle = user32.FindWindowW(None, title)
    if not handle:
        raise RuntimeError(f'Could not find a window titled {title!r}.')
    rect = wintypes.RECT()
    user32.GetClientRect(handle, ctypes.byref(rect))
    origin = wintypes.POINT(0, 0)
    user32.ClientToScreen(handle, ctypes.byref(origin))
    return handle, {'left': origin.x, 'top': origin.y,
                    'width': rect.right - rect.left, 'height': rect.bottom - rect.top}


def _click(x, y):
    import ctypes
    import random
    import time
    user32 = ctypes.windll.user32
    user32.SetCursorPos(int(x), int(y))
    time.sleep(0.05 + 0.05 * random.random())
    user32.mouse_event(0x0002, 0, 0, 0, 0)      # left down
    time.sleep(0.06 + 0.06 * random.random())
    user32.mouse_event(0x0004, 0, 0, 0, 0)      # left up


def play_live(title='MapleStory', select=False, timeout=120.0):
    """
    Watches the game window and, with SELECT, clicks each answer once the
    buttons light up. Start it before the opening reveal ends.
    """
    import ctypes
    import random
    import time
    import mss

    ctypes.windll.user32.SetProcessDPIAware()
    handle, region = _client_region(title)
    solver = ViolettaSolver()
    pending = None
    logged = 0
    started = time.monotonic()
    print(f'[VIOLETTA] Watching {title!r}; clicking is {"ON" if select else "OFF (preview)"}.')
    with mss.mss() as screen:
        while time.monotonic() - started < timeout:
            handle, region = _client_region(title)
            frame = np.asarray(screen.grab(region))[:, :, :3]
            answer = solver.update(frame)
            for _, message in solver.events[logged:]:
                print(f'[VIOLETTA] {message}')
            logged = len(solver.events)
            if answer is not None:
                pending = answer
            if pending is not None and solver.answer_open(frame):
                x, y = solver.button_point(pending)
                note = '' if solver.confident else ' (low confidence)'
                if select:
                    if ctypes.windll.user32.GetForegroundWindow() != handle:
                        print('[VIOLETTA] Game window is not focused; not clicking.')
                    else:
                        time.sleep(0.2 + 0.4 * random.random())
                        _click(region['left'] + x, region['top'] + y)
                        print(f'[VIOLETTA] Clicked answer {pending + 1}{note}.')
                else:
                    print(f'[VIOLETTA] Answer is {pending + 1}{note}.')
                pending = None
                if solver.state == 'done':
                    return
    print('[VIOLETTA] Timed out.')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Follow the make-up Violetta and pick her slot.')
    parser.add_argument('--video', help='Replay a recording instead of watching the game.')
    parser.add_argument('--output', help='With --video: write an annotated copy here.')
    parser.add_argument('--title', default='MapleStory', help='Game window title.')
    parser.add_argument('--select', action='store_true', help='Live: click the answers.')
    parser.add_argument('--timeout', type=float, default=120.0, help='Live: seconds to run.')
    args = parser.parse_args()
    if args.video:
        replay(args.video, args.output)
    else:
        play_live(args.title, args.select, args.timeout)


if __name__ == '__main__':
    main()
