"""
Follow-the-fading-figure mini-game: keep the mouse on a shape that fades into a
transparent "lens" over a scrolling background full of look-alike outlines.

How it works
------------
* Start: during the countdown the figure is solid white. It is found and followed
  as a white blob, which also gives its size (every later setting is relative to it).
* After it fades, the figure is only visible because it warps the background behind
  it. The background (and every decoy outline on it) scrolls as ONE rigid layer, so
  anything that does not move with that layer is the figure. Three cues measure this:
    - optical flow minus the background's flow (where there is texture to trust it),
    - the current frame vs. older frames aligned on the background (median of diffs),
    - the current frame vs. a median background built from older aligned frames.
  Each cue is matched with a disk the size of the figure (rotation doesn't matter),
  z-scored, and the positive parts are summed.
* Clutter memory: false alarms come from background features (flat holes, edges that
  align imperfectly), so they ride along with the background. A memory that scrolls
  with the background collects them and is subtracted. The figure keeps moving across
  the background, so it does not build up there (and is learned only slowly near the
  current estimate, so a figure drifting with the background isn't erased).
* Static screen parts (window borders, instruction bars) never change while the
  background scrolls; they are masked out.
* A Kalman filter (constant velocity with friction) with a gate sized to the figure
  picks the evidence peak near the prediction, and evidence is carried along with the
  predicted motion between frames.
* The mouse is driven like a person would: a short reaction delay, smooth
  acceleration towards the estimate and a little hand jitter. Near the centre is good
  enough for the game, so smoothness wins over chasing every wobble.

Usage (from the repo root)::

    python -m src.detection.fading_figure --video clip.mp4 --output annotated.mp4
    python -m src.detection.fading_figure              # live preview, mouse untouched
    python -m src.detection.fading_figure --move       # live, moves the mouse

Start live mode before the countdown ends: it needs to see the solid white figure.
"""

import cv2
import numpy as np


# --- tuning (relative to the figure's equivalent radius r, in working pixels) --------
WORK_RADIUS = 14.0        # frames are scaled so the figure's radius is about this
CLUTTER_WEIGHT = 1.0
CLUTTER_DECAY = 0.85
CLUTTER_NEAR_RATE = 0.4   # clutter learns this much slower within ...
CLUTTER_NEAR = 1.5        # ... this many radii of the estimate
VELOCITY_FRICTION = 0.9   # per frame
GATE_MAX = 0.8            # radii
GATE_GROWTH = 0.15        # radii added per frame without a measurement (re-acquire)
GATE_LOST = 1.5           # radii: widest search while lost
SPEED_MAX = 0.25          # radii per frame
MEAS_NOISE = 0.7          # radii (std) for a lens measurement of unit strength
MIN_STRENGTH = 3.5
SEARCH_FRAMES = 10        # frames the white figure must hold still before tracking starts
WHITE_MISSES = 8          # frames the white figure may be missing before it is "faded"
WHITE_SIZE_FRAMES = 10    # white frames used to measure the figure's size
NOMINAL_FPS = 30.0


def green_mask(frame):
    """The game's green crosshair cursor (and anything like it)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (35, 90, 80), (95, 255, 255))


def find_white_figure(frame, near=None, area0=None, cursor=None, last_area=None):
    """
    Returns (centre, area) of the white figure, or (None, 0).

    Without NEAR, looks for a solid, compact white blob, preferring big ones near
    the middle of the frame (countdown digits sit above the figure). With NEAR,
    follows the figure as it fades: the closest blob of a similar size to the
    original (AREA0) and to last frame's (LAST_AREA) that is clearly brighter than
    its surroundings, trying looser thresholds as it gets fainter. Pale patches of
    background pass the loose thresholds too, but aren't brighter than their
    surroundings and don't keep the figure's size, so they are refused.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width = frame.shape[:2]
    levels = [(190, 60)] if near is None else [(190, 60), (160, 80), (140, 95)]
    for value, saturation in levels:
        white = cv2.inRange(hsv, (0, 0, value), (180, saturation, 255))
        mask = white.copy()
        if cursor is not None:
            mask[cursor > 0] = 255  # the cursor sits on the figure; don't cut a hole
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, labels, stats, centres = cv2.connectedComponentsWithStats(mask)
        best, best_score = None, None
        for k in range(1, count):
            x, y, w, h, area = stats[k]
            if w > width * 0.4 or h > height * 0.4:
                continue
            if near is None:
                if area < 60 or area / float(w * h) < 0.3:
                    continue
                if cursor is not None and np.count_nonzero(white[labels == k]) < 0.5 * area:
                    continue  # mostly cursor, not a white figure
                offset = np.hypot(*(centres[k] - (width / 2, height / 2))) / np.hypot(width, height)
                score = -area / (1 + 4 * offset)
            else:
                score = np.linalg.norm(centres[k] - near)
                if score > 25 or area < 0.35 * area0 or area > 1.6 * area0:
                    continue
                if last_area and not 0.6 < area / last_area < 1.5:
                    continue
                if area / float(w * h) < 0.25:  # thin rings are decoy outlines, not the figure
                    continue
                if not _stands_out(hsv[..., 2], labels == k, (x, y, w, h)):
                    continue
            if best_score is None or score < best_score:
                best, best_score = k, score
        if best is not None:
            return centres[best].astype(np.float64), int(stats[best, 4])
    return None, 0


def _stands_out(value, blob, box, margin=6, contrast=20):
    """Is the blob clearly brighter than a ring around it?"""
    x, y, w, h = box
    height, width = value.shape
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(width, x + w + margin), min(height, y + h + margin)
    inside = blob[y0:y1, x0:x1]
    ring = cv2.dilate(inside.astype(np.uint8), np.ones((2 * margin + 1,) * 2, np.uint8)) > 0
    ring &= ~inside
    if not ring.any():
        return True
    patch = value[y0:y1, x0:x1].astype(np.float32)
    return patch[inside].mean() - patch[ring].mean() > contrast


class FadingFigureTracker:
    """
    Feed frames in order with update(); it returns the figure's centre in frame
    pixels, or None until the white figure has been found.
    """

    def __init__(self):
        self.state = 'search'          # search -> white -> lens
        self.position = None
        self.radius = None             # figure radius in frame pixels
        self._candidate = None         # (centre, area, frames seen) while searching
        self._white = []               # buffered (frame, time, centre) before the size is known
        self._white_area = None
        self._misses = 0
        self._last_white = None
        self._last_white_area = None
        self._prev_small = None
        self._prev_time = None
        self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    # ------------------------------------------------------------------ public
    def update(self, frame, time):
        """FRAME: BGR image. TIME: seconds (video: frame index / fps)."""
        cursor = green_mask(frame)
        if self.state == 'search':
            self._search(frame, time, cursor)
            return self.position
        white = None
        if self.state == 'white':
            centre, area = find_white_figure(frame, self._last_white, self._white_area, cursor,
                                             self._last_white_area)
            if centre is None:
                self._misses += 1
                if self._misses > WHITE_MISSES:
                    self.state = 'lens'
            else:
                self._misses = 0
                self._last_white = white = centre
                self._last_white_area = area
            if self._white is not None:  # still measuring the figure's size
                if centre is not None:
                    self._white_area = max(self._white_area, area)
                self._white.append((frame, time, white))
                if len(self._white) >= WHITE_SIZE_FRAMES or self.state == 'lens':
                    self._setup()
                return self.position
        self._step(frame, time, white, cursor)
        return self.position

    # ------------------------------------------------------------------ start
    def _search(self, frame, time, cursor):
        centre, area = find_white_figure(frame, cursor=cursor)
        if centre is None:
            self._candidate = None
            return
        if self._candidate is not None:
            c0, a0, seen = self._candidate
            if np.linalg.norm(centre - c0) < 15 and 0.6 < area / a0 < 1.6:
                self._candidate = (centre, area, seen + 1)
            else:
                self._candidate = (centre, area, 1)
        else:
            self._candidate = (centre, area, 1)
        if self._candidate[2] >= SEARCH_FRAMES:  # held still long enough: that's the figure
            self.state = 'white'
            self._white_area = area
            self._last_white = centre
            self._last_white_area = area
            self._white = [(frame, time, centre)]
            self.position = centre.copy()

    def _setup(self):
        """The figure's size is known: fix the working scale and replay the buffer."""
        self.radius = float(np.sqrt(self._white_area / np.pi))
        self.scale = min(1.0, WORK_RADIUS / self.radius)
        self.r = self.radius * self.scale
        r = self.r
        R = int(2.2 * r) + 2
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        dist = np.hypot(xx, yy)
        inner = np.clip(r - dist + 0.5, 0, 1)
        ring = ((dist > 1.25 * r) & (dist < 2.0 * r)).astype(np.float32)
        self.kernel = (inner / inner.sum() - ring / ring.sum()).astype(np.float32)

        first = next(c for _, _, c in self._white if c is not None) * self.scale
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        self.kf.statePost = np.array([[first[0]], [first[1]], [0], [0]], np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.history = []              # (small gray, cumulative background offset)
        self.evidence = None
        self.clutter = None
        self.activity = None
        self.coasting = 0
        buffered, self._white = self._white, None
        for frame, time, white in buffered:
            self._step(frame, time, white, green_mask(frame))

    # ------------------------------------------------------------------ tracking
    def _prep(self, frame, cursor):
        if cursor.any():
            frame = cv2.inpaint(frame, cv2.dilate(cursor, np.ones((9, 9), np.uint8)), 3, cv2.INPAINT_TELEA)
        small = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def _zscore(self, cue):
        cue = cv2.filter2D(cue, -1, self.kernel, borderType=cv2.BORDER_REFLECT)
        b = int(self.r)
        inner = cue[b:-b, b:-b] if min(cue.shape) > 2 * b + 4 else cue
        return (cue - np.median(inner)) / (inner.std() + 1e-6)

    def _evidence(self, gray, flow, background_flow, offset):
        H, W = gray.shape
        g32 = gray.astype(np.float32)
        # optical flow that disagrees with the background, where texture makes it reliable
        deviation = np.linalg.norm(flow - background_flow, axis=2)
        eig = cv2.cornerMinEigenVal(gray, 5)
        deviation *= np.clip(cv2.GaussianBlur(eig, (0, 0), 3) / (np.median(eig) + 1e-6), 0, 1)
        total = np.clip(self._zscore(np.minimum(deviation, np.percentile(deviation, 99.5) + 1e-6)), 0, None)

        # photometric residuals against older frames aligned on the background,
        # normalised by local gradient (sub-pixel misalignment on edges is not the figure)
        def aligned(k):
            old, old_offset = self.history[-1 - k]
            d = offset - old_offset
            return cv2.warpAffine(old.astype(np.float32), np.float32([[1, 0, d[0]], [0, 1, d[1]]]), (W, H),
                                  flags=cv2.INTER_LINEAR, borderValue=np.nan)
        gradient = np.hypot(cv2.Sobel(g32, cv2.CV_32F, 1, 0), cv2.Sobel(g32, cv2.CV_32F, 0, 1))
        normaliser = cv2.GaussianBlur(gradient, (0, 0), 1.5) * 0.25 + 8.0
        cache = {}
        for kind, lags in (('short', (2, 4, 6)), ('long', (3, 6, 9, 12))):
            lags = [k for k in lags if len(self.history) > k]
            if len(lags) < 2:
                continue
            for k in lags:
                if k not in cache:
                    cache[k] = aligned(k)
            with np.errstate(all='ignore'):
                if kind == 'short':
                    residual = np.nanmedian(np.stack([np.abs(g32 - cache[k]) for k in lags]), 0)
                else:
                    residual = np.abs(g32 - np.nanmedian(np.stack([cache[k] for k in lags]), 0))
            fill = np.nanmedian(residual)
            residual = np.nan_to_num(residual, nan=float(fill) if np.isfinite(fill) else 0.0) / normaliser
            total += np.clip(self._zscore(np.minimum(residual, np.percentile(residual, 99.5))), 0, None)
        return total

    def _step(self, frame, time, white, cursor):
        gray = self._prep(frame, cursor)
        if self._prev_small is not None and white is None and \
                cv2.absdiff(gray, self._prev_small).mean() < 0.8:
            return  # repeated game frame: nothing new to see
        dt = 1.0 if self._prev_time is None else max((time - self._prev_time) * NOMINAL_FPS, 0.5)
        self._prev_time = time
        self._prev_small = gray
        H, W = gray.shape
        if not self.history:
            self.history.append((gray, np.zeros(2)))
            self.position = self.kf.statePost[:2, 0] / self.scale
            return

        previous, previous_offset = self.history[-1]
        flow = self._dis.calc(previous, gray, None)
        background_flow = np.median(flow.reshape(-1, 2), 0)
        offset = previous_offset + background_flow
        self.history.append((gray, offset))
        evidence = self._evidence(gray, flow, background_flow, offset)
        self.history = self.history[-14:]

        # static screen parts don't change while the background scrolls
        change = cv2.absdiff(gray, previous).astype(np.float32)
        self.activity = change if self.activity is None else 0.9 * self.activity + 0.1 * change
        if len(self.history) > 8 and np.hypot(*background_flow) > 0.5:
            static = cv2.dilate((self.activity < 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            evidence[static] = 0

        # clutter memory, scrolled with the background
        if self.clutter is None:
            self.clutter = np.zeros_like(evidence)
        else:
            self.clutter = cv2.warpAffine(self.clutter, np.float32([[1, 0, background_flow[0]],
                                                                    [0, 1, background_flow[1]]]),
                                          (W, H), borderMode=cv2.BORDER_REPLICATE)
        raw = evidence
        evidence = np.clip(raw - CLUTTER_WEIGHT * self.clutter, 0, None)
        ex, ey = self.kf.statePost[0, 0], self.kf.statePost[1, 0]
        Y, X = np.ogrid[:H, :W]
        far = (X - ex) ** 2 + (Y - ey) ** 2 > (CLUTTER_NEAR * self.r) ** 2
        learned = CLUTTER_DECAY * self.clutter + (1 - CLUTTER_DECAY) * raw
        self.clutter = np.where(far, learned, (1 - CLUTTER_NEAR_RATE) * self.clutter + CLUTTER_NEAR_RATE * learned)

        # predict, and carry the evidence along with the figure's predicted motion
        self._set_dt(dt)
        prediction = self.kf.predict()
        px, py, vx, vy = prediction[:, 0]
        if self.evidence is None:
            self.evidence = evidence
        else:
            moved = cv2.warpAffine(self.evidence, np.float32([[1, 0, vx * dt], [0, 1, vy * dt]]), (W, H))
            self.evidence = 0.5 * moved + 0.5 * evidence

        if white is not None:
            measurement, strength = white * self.scale, None
        else:
            measurement, strength = self._measure(px, py)
        if measurement is not None and (strength is None or strength > MIN_STRENGTH):
            if strength is None:
                noise = (0.1 * self.r) ** 2
            else:
                noise = (MEAS_NOISE * self.r) ** 2 * float(np.clip(4.0 / strength, 0.5, 3))
            self.kf.measurementNoiseCov = (np.eye(2) * noise).astype(np.float32)
            self.kf.correct(measurement.astype(np.float32).reshape(2, 1))
            self.coasting = 0
        else:  # nothing convincing: coast on the prediction
            self.coasting += dt
            self.kf.statePost = prediction.copy()
            self.kf.errorCovPost = self.kf.errorCovPre.copy()
        state = self.kf.statePost.copy()
        limit = SPEED_MAX * self.r
        state[2:, 0] = np.clip(state[2:, 0], -limit, limit)
        state[0, 0] = np.clip(state[0, 0], 0, W - 1)
        state[1, 0] = np.clip(state[1, 0], 0, H - 1)
        self.kf.statePost = state
        self.position = state[:2, 0].astype(np.float64) / self.scale

    def _set_dt(self, dt):
        q = 0.02 * self.r
        a = VELOCITY_FRICTION ** dt
        self.kf.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, a, 0], [0, 0, 0, a]], np.float32)
        self.kf.processNoiseCov = (q * np.array([[dt**3 / 3, 0, dt**2 / 2, 0], [0, dt**3 / 3, 0, dt**2 / 2],
                                                [dt**2 / 2, 0, dt, 0], [0, dt**2 / 2, 0, dt]])).astype(np.float32)

    def _measure(self, px, py):
        """Best evidence peak inside the gate around the prediction, refined by centroid."""
        evidence = self.evidence
        H, W = evidence.shape
        P = self.kf.errorCovPre
        sigma = float(np.sqrt(0.5 * (P[0, 0] + P[1, 1])))
        widest = min(GATE_MAX + GATE_GROWTH * self.coasting, GATE_LOST)
        gate = min(2.0 * sigma + 0.3 * self.r, widest * self.r)
        if self.coasting:
            gate = widest * self.r
        Y, X = np.ogrid[:H, :W]
        d2 = (X - px) ** 2 + (Y - py) ** 2
        score = evidence - 0.5 * d2 / gate ** 2
        score[d2 > (1.4 * gate) ** 2] = -1e9
        _, best, _, (lx, ly) = cv2.minMaxLoc(score)
        if best < -1e8:
            return None, 0.0
        strength = float(evidence[ly, lx])
        rr = max(2, int(self.r * 0.6))
        y0, x0 = max(0, ly - rr), max(0, lx - rr)
        patch = evidence[y0:ly + rr + 1, x0:lx + rr + 1]
        weight = np.clip(patch - 0.6 * patch.max(), 0, None)
        if weight.sum() > 0:
            yy, xx = np.mgrid[y0:y0 + patch.shape[0], x0:x0 + patch.shape[1]]
            return np.array([(xx * weight).sum(), (yy * weight).sum()]) / weight.sum(), strength
        return np.array([lx, ly], float), strength


class HumanMouse:
    """
    Turns a jittery target into hand-like motion: reacts after a short delay, eases
    towards the target with limited speed, and wobbles a little.
    """

    def __init__(self, reaction=0.10, ease=0.06, lead=0.12, max_speed=900.0, jitter=1.5, seed=None):
        self.reaction = reaction        # seconds
        self.ease = ease                # seconds to close ~63% of the gap
        self.lead = lead                # seconds of anticipation along the target's motion
        self.max_speed = max_speed      # pixels per second
        self.jitter = jitter            # pixels
        self.position = None
        self._targets = []
        self._time = None
        self._wobble = np.zeros(2)
        self._rng = np.random.default_rng(seed)

    def step(self, target, time):
        if target is None:
            return self.position
        self._targets.append((time, np.asarray(target, float)))
        while len(self._targets) > 1 and self._targets[1][0] <= time - self.reaction:
            self._targets.pop(0)
        goal = self._targets[0][1]
        # anticipate: people lead a smoothly moving target instead of trailing it
        if len(self._targets) > 1:
            span = max(self._targets[-1][0] - self._targets[0][0], 1e-3)
            velocity = (self._targets[-1][1] - self._targets[0][1]) / span
            goal = goal + velocity * (self.reaction + self.lead) * 0.5
        if self.position is None:
            self.position, self._time = goal.copy(), time
            return self.position
        dt = max(time - self._time, 1e-3)
        self._time = time
        move = (goal - self.position) * (1 - np.exp(-dt / self.ease))
        length = np.linalg.norm(move)
        if length > self.max_speed * dt:
            move *= self.max_speed * dt / length
        self._wobble = 0.85 * self._wobble + self._rng.normal(0, self.jitter * 0.5, 2)
        self.position = self.position + move
        return self.position + self._wobble


# ---------------------------------------------------------------------- replay / live
def replay(path, output=None, roi=None):
    """Runs the tracker over a recording; returns {frame: (x, y)} in video pixels."""
    capture = cv2.VideoCapture(path)
    fps = capture.get(cv2.CAP_PROP_FPS) or NOMINAL_FPS
    tracker = FadingFigureTracker()
    mouse = HumanMouse(seed=0)
    writer = None
    estimates, pointer = {}, {}
    ox, oy = (roi[0], roi[1]) if roi else (0, 0)
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        view = frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]] if roi else frame
        position = tracker.update(view, index / fps)
        if position is not None:
            estimates[index] = (position[0] + ox, position[1] + oy)
            pointer[index] = tuple(mouse.step(estimates[index], index / fps))
        if output:
            if writer is None:
                writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                         (frame.shape[1], frame.shape[0]))
            shown = frame.copy()
            if index in estimates:
                x, y = estimates[index]
                cv2.circle(shown, (int(x), int(y)), int(tracker.radius or 10), (255, 0, 255), 1)
                cv2.drawMarker(shown, (int(x), int(y)), (255, 0, 255), cv2.MARKER_CROSS, 16, 2)
                mx, my = pointer[index]
                cv2.circle(shown, (int(mx), int(my)), 5, (0, 165, 255), -1)
            cv2.putText(shown, f'{index} {tracker.state}', (6, 20), 0, 0.6, (0, 0, 255), 2)
            writer.write(shown)
        index += 1
    if writer is not None:
        writer.release()
    print(f'[FADING] {index} frames; figure radius {tracker.radius or 0:.0f}px; '
          f'tracked {len(estimates)} frames.')
    return estimates, pointer


def _cursor_position():
    import ctypes
    from ctypes import wintypes
    point = wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
    return np.array((point.x, point.y), float)


def play_live(title='MapleStory', move=False, timeout=60.0, roi=None, start_delay=0.0,
              search_timeout=45.0, game_seconds=25.0, stop=None):
    """
    Watches the game window and, with MOVE, keeps the mouse on the figure. Moving
    the mouse yourself (more than ~60 px from where it was put) hands control back.

    START_DELAY: seconds to wait first (e.g. while the warning pop-up is up).
    SEARCH_TIMEOUT: give up if no white figure shows up in this long.
    GAME_SECONDS: stop this long after the figure fades (the game is over by then).
    STOP: optional threading.Event to end early.
    Returns why it ended: 'done', 'no figure', 'taken over', 'stopped' or 'timed out'.
    """
    import ctypes
    import time
    import mss
    from src.detection.violetta_solver import _client_region

    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    print(f'[FADING] Watching {title!r}; mouse control is {"ON" if move else "OFF (preview)"}.')
    if start_delay and (stop.wait(start_delay) if stop else time.sleep(start_delay)):
        return 'stopped'
    tracker = FadingFigureTracker()
    mouse = HumanMouse()
    placed = None
    state = None
    started = time.monotonic()
    faded_at = None
    with mss.mss() as screen:
        while time.monotonic() - started < timeout:
            if stop is not None and stop.is_set():
                return 'stopped'
            handle, region = _client_region(title)
            if roi:
                region = {'left': region['left'] + roi[0], 'top': region['top'] + roi[1],
                          'width': roi[2], 'height': roi[3]}
            frame = np.ascontiguousarray(np.asarray(screen.grab(region))[:, :, :3])
            now = time.monotonic()
            position = tracker.update(frame, now)
            if tracker.state != state:
                state = tracker.state
                print(f'[FADING] {state}')
                if state == 'lens':
                    faded_at = now
            if tracker.state == 'search' and now - started > search_timeout:
                print('[FADING] No white figure appeared; giving up.')
                return 'no figure'
            if faded_at is not None and now - faded_at > game_seconds:
                print('[FADING] Game time is up; done.')
                return 'done'
            if position is None:
                continue
            target = mouse.step((region['left'] + position[0], region['top'] + position[1]), now)
            if not move:
                continue
            if user32.GetForegroundWindow() != handle:
                continue
            if placed is not None and np.linalg.norm(_cursor_position() - placed) > 60:
                print('[FADING] Mouse moved by hand; stopping.')
                return 'taken over'
            user32.SetCursorPos(int(target[0]), int(target[1]))
            placed = np.array((int(target[0]), int(target[1])), float)
    print('[FADING] Timed out.')
    return 'timed out'


def _roi(text):
    return tuple(int(v) for v in text.split(',')) if text else None


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Keep the mouse on the fading figure.')
    parser.add_argument('--video', help='Replay a recording instead of watching the game.')
    parser.add_argument('--output', help='With --video: write an annotated copy here.')
    parser.add_argument('--roi', help='x,y,w,h of the mini-game inside the frame/window (optional).')
    parser.add_argument('--title', default='MapleStory', help='Game window title.')
    parser.add_argument('--move', action='store_true', help='Live: move the mouse.')
    parser.add_argument('--timeout', type=float, default=60.0, help='Live: seconds to run.')
    args = parser.parse_args()
    if args.video:
        replay(args.video, args.output, _roi(args.roi))
    else:
        play_live(args.title, args.move, args.timeout, _roi(args.roi))


if __name__ == '__main__':
    main()
