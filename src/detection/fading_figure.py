"""
Follow-the-fading-figure mini-game: keep the mouse on a shape that fades into a
transparent "lens" over a scrolling background full of look-alike outlines.

How it works
------------
* Start: during the countdown the figure is solid white. It is found and followed
  as a white blob. Its outline is traced from the white pixels - no shape is
  assumed, it can be any polygon or several fused together - and its centre is the
  centre of that blob.
* After it fades, the figure is a lens: the background shows through it almost
  unchanged, except along its edge, where the picture is bent. Compared with the
  frame from a moment before (shifted by how far the background scrolled), that
  edge is what doesn't match, so the faded figure shows up as its own outline.
* A filter made from the traced outline, spun about its centre (the figure turns),
  marks where an outline of that size is centred. Busy patches of background and
  the crosshair don't look like that ring; the crosshair is also blanked out, or
  the tracker would end up following its own cursor.
* The figure can only move so far in a frame. Every place it could be is scored
  by the outline evidence along the best path leading there, and the estimate is
  the end of the best path. Nothing is ever locked on to: if the estimate slips,
  the true path keeps collecting evidence and takes over again.
* The mouse is driven like a person would: a short reaction delay, smooth
  acceleration towards the estimate and a little hand jitter. Near the centre is good
  enough for the game, so smoothness wins over chasing every wobble.

Usage (from the repo root)::

    python -m src.detection.fading_figure --video clip.mp4 --output annotated.mp4
    python -m src.detection.fading_figure              # live preview, mouse untouched
    python -m src.detection.fading_figure --move       # live, moves the mouse

Start live mode before the countdown ends: it needs to see the solid white figure.
"""

import os

import cv2
import numpy as np


# --- tuning ----------------------------------------------------------------------------
WORK_RADIUS = 14.0        # frames are scaled so a disc of the figure's area has this radius
SPEED_MAX = 0.18          # radii a frame (at 30 a second) the figure is given to travel, plus a pixel.
                          # Was 0.12 (~350 px/s at 44 fps for a 56 px figure); in the jailed game
                          # shape-20261119-175343 the figure moved 306 px/s median, 857 at p90, and
                          # the estimate trailed ~6 frames then jumped to catch up.
COMPARE_FRAMES = 2.0      # frames (at 30 a second) back to compare with: long enough for the
                          # figure to have moved, short enough for the background to line up
HISTORY = 6               # past frames kept to choose that one from
OUTLINE_SURROUND = 1.25   # the outline filter subtracts a disc this many times its size
CURSOR_MARGIN = 21        # frame pixels around the crosshair that are not looked at
OUTLINE_STEP = 20.0       # degrees between the exact-outline filters (0: the outline spun into a ring)
SURROUND_WEIGHT = 1.0     # share of the surround's average taken off the outline fit
VELOCITY_PRIOR = 0.8      # share of the estimate's recent motion paths are carried along each frame
                          # (175343: crosshair on the figure 52 -> 56%; no game of 5 worse; 1.0
                          # also 56% there but cost another game a point)
VELOCITY_SMOOTH = 0.6     # of the previous velocity kept each frame
SWITCH_MARGIN = 4.0       # how far ahead a path elsewhere must be before the estimate moves to it
MIN_STRENGTH = 1.0        # outline evidence below this at the estimate counts as "not seen"
SEARCH_FRAMES = 10        # frames the white figure must hold still before tracking starts
WHITE_MISSES = 8            # frames the white figure may be missing before it is "faded"
PANEL_WAIT = 30.0           # seconds to look for the mini-game window before giving up
WHITE_SIZE_FRAMES = 10    # white frames used to measure the figure's size and outline
WHITE_REACH = 0.7         # radii: how far from its last place the white figure is looked for ...
WHITE_REACH_GROWTH = 0.35 # ... plus this much for every frame it has been missing
NOMINAL_FPS = 30.0
# Cold start: the white figure was never seen (the bot arrived after it faded,
# shape-20261119-012428). After this long in the game with no white figure, look
# for the faded figure's outline at the usual size instead (radii 49-57 px in the
# 18 Nov games), with no idea where it starts: every place starts level and the
# path that keeps showing an outline of that size wins.
COLD_START_SECONDS = 1.5
COLD_RADIUS = 52.0        # frame pixels
COLD_RADII = (0.75, 1.2)  # outline points are spread over this share of it (tuned on 4 games)
COLD_SWITCH_MARGIN = 1.5  # SWITCH_MARGIN while cold-started (quicker to move to a better path)
COLD_SETTLE_SECONDS = 0.4 # evidence gathered before the first estimate is given out
SEARCH_MIN_AREA = 3000    # px; the white figure is 7,000-11,000 (radius 47-59), pale background patches less
COLD_BROWN = 0.5          # share of the frame that must be the game's sand-brown background


def green_mask(frame):
    """The game's green crosshair cursor (and anything like it)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (35, 90, 80), (95, 255, 255))


def find_white_figure(frame, near=None, area0=None, cursor=None, last_area=None, misses=0,
                      also_near=None):
    """
    Returns (centre, area) of the white figure, or (None, 0).

    Without NEAR, looks for a solid, compact white blob, preferring big ones near
    the middle of the frame (countdown digits sit above the figure). With NEAR,
    follows the figure as it fades: the closest blob of a similar size to the
    original (AREA0) and to last frame's (LAST_AREA) that is clearly brighter than
    its surroundings, trying looser thresholds as it gets fainter. Pale patches of
    background pass the loose thresholds too, but aren't brighter than their
    surroundings and don't keep the figure's size, so they are refused.

    MISSES: frames since it was last seen. The figure keeps moving while it is
    missed (it travels about a third of its radius a frame), so the search reaches
    further and allows more change in size for every frame missed; without that,
    one frame arriving late during the fade loses the figure for good.
    ALSO_NEAR: a second place to look around (the tracker's current estimate).
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
                if also_near is not None:
                    score = min(score, np.linalg.norm(centres[k] - also_near))
                reach = (WHITE_REACH + WHITE_REACH_GROWTH * misses) * np.sqrt(area0 / np.pi)
                if score > reach or area < 0.35 * area0 or area > 1.6 * area0:
                    continue
                slack = 1.15 ** misses
                if last_area and not 0.6 / slack < area / last_area < 1.5 * slack:
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


def _game_background(frame):
    """Is FRAME the mini-game's sand-brown background (not a map, the jail room, ...)?"""
    small = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    brown = cv2.inRange(hsv, (4, 50, 50), (32, 255, 255))
    return cv2.countNonZero(brown) >= COLD_BROWN * brown.size


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


def _nanmedian(stack):
    """
    np.nanmedian(stack, 0) for a short stack of images, many times faster (the
    tracker's frame rate depends on it). NaN where every layer is NaN.
    """
    ordered = np.sort(stack, axis=0)                # NaNs sort to the end
    count = np.count_nonzero(np.isfinite(stack), axis=0)
    low = np.clip((count - 1) // 2, 0, None)[None]
    high = np.clip(count // 2, 0, len(stack) - 1)[None]
    median = 0.5 * (np.take_along_axis(ordered, low, 0) + np.take_along_axis(ordered, high, 0))[0]
    median[count == 0] = np.nan
    return median


def figure_outline(frame, centre, cursor=None):
    """
    The outline of the solid white figure at CENTRE, as an array of (x, y) points
    relative to the figure's centroid; None if there is no white blob there.
    Nothing is assumed about the shape: it is whatever the white pixels trace.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 190), (180, 60, 255))
    if cursor is not None:
        mask[cursor > 0] = 255      # the cursor sits on the figure; don't cut a notch
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, centres = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        return None
    k = 1 + int(np.argmin(np.linalg.norm(centres[1:] - centre, axis=1)))
    if np.linalg.norm(centres[k] - centre) > 20:
        return None
    contours, _ = cv2.findContours((labels == k).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    points = max(contours, key=len)[:, 0, :].astype(np.float64)
    return points - centres[k]


def outline_kernel(radii):
    """
    A filter that answers "is the figure's outline centred here?". RADII: how far
    each outline point is from the figure's centre (working pixels). The figure
    turns as it moves, so the filter is the outline spun about its centre: bright
    where the outline passes often. A broad disc is subtracted, so a patch that is
    busy all over (a wobbling piece of background) scores nothing; only a ring of
    the right size does.
    """
    reach = float(np.percentile(radii, 99))
    size = int(OUTLINE_SURROUND * reach) + 3
    yy, xx = np.mgrid[-size:size + 1, -size:size + 1]
    distance = np.hypot(xx, yy)
    bins = np.arange(0, size + 2)
    histogram, _ = np.histogram(radii, bins)
    histogram = cv2.GaussianBlur(histogram.astype(np.float32).reshape(1, -1), (0, 0), 1.5).ravel()
    density = histogram / np.maximum(bins[:-1] + 0.5, 1.0)      # per pixel, not per ring
    ring = np.interp(distance, bins[:-1] + 0.5, density, right=0).astype(np.float32)
    surround = (distance <= OUTLINE_SURROUND * reach).astype(np.float32)
    return (ring / ring.sum() - surround / surround.sum()).astype(np.float32)


def outline_kernels(outline, scale, step):
    """
    The exact traced outline as filters, one per turn of STEP degrees (working
    pixels). Unlike outline_kernel's ring, which only knows how far the outline
    is from the centre, these know the shape: background outlines of about the
    right size but the wrong shape score much lower.
    """
    points = np.asarray(outline, float) * scale
    points = points[np.linspace(0, len(points), 96, endpoint=False).astype(int)]
    size = int(np.ceil(np.hypot(points[:, 0], points[:, 1]).max())) + 3
    kernels = []
    for angle in np.radians(np.arange(0.0, 360.0, step)):
        kernel = np.zeros((2 * size + 1,) * 2, np.float32)
        xs = np.round(size + points[:, 0] * np.cos(angle) - points[:, 1] * np.sin(angle)).astype(int)
        ys = np.round(size + points[:, 0] * np.sin(angle) + points[:, 1] * np.cos(angle)).astype(int)
        np.add.at(kernel, (ys, xs), 1.0)
        kernel = cv2.GaussianBlur(kernel, (0, 0), 1.0)
        kernels.append(kernel / kernel.sum())
    return kernels


def outline_kernel_parts(radii):
    """outline_kernel's two halves, (ring, surround), each summing to 1."""
    full = outline_kernel(radii)
    size = full.shape[0] // 2
    yy, xx = np.mgrid[-size:size + 1, -size:size + 1]
    reach = float(np.percentile(radii, 99))
    surround = (np.hypot(xx, yy) <= OUTLINE_SURROUND * reach).astype(np.float32)
    surround /= surround.sum()
    ring = full + surround          # outline_kernel = ring - surround
    return ring.astype(np.float32), surround.astype(np.float32)


class FadingFigureTracker:
    """
    Feed frames in order with update(); it returns the figure's centre in frame
    pixels, or None until the white figure has been found.
    """

    def __init__(self):
        self.state = 'search'          # search -> white -> lens
        self.position = None
        self.radius = None             # radius of a disc with the figure's area, frame pixels
        self.outline = None            # the white figure's outline around its centre, frame pixels
        self.coasting = 0              # frames since the outline was last seen at the estimate
        self.jumps = 0                 # times the estimate moved to a better path elsewhere
        self._candidate = None         # (centre, area, frames seen) while searching
        self._white = []               # buffered (frame, time, centre) before the size is known
        self._white_area = None
        self._misses = 0
        self._last_white = None
        self._last_white_area = None
        self._prev_small = None
        self._prev_time = None
        self.cold = False              # started without the white figure (see COLD_START_SECONDS)
        self._search_since = None
        self._cold_since = None
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
                                             self._last_white_area, self._misses,
                                             self.position if self.radius else None)
            if centre is None:
                self._misses += 1
                if self._misses > WHITE_MISSES:
                    self.state = 'lens'
            else:
                self._misses = 0
                self._last_white = white = centre
                self._last_white_area = area
            if self._white is not None:  # still measuring the figure
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
        if self._search_since is None:
            self._search_since = time
        centre, area = find_white_figure(frame, cursor=cursor)
        if centre is not None and area < SEARCH_MIN_AREA:
            centre = None       # a pale patch of background, not the solid figure
        if centre is None:
            self._candidate = None
            if time - self._search_since >= COLD_START_SECONDS and _game_background(frame):
                self._cold_start(frame, time, cursor)
            return
        self._search_since = time   # (a white blob that is still settling: not late)
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

    def _cold_start(self, frame, time, cursor):
        """No white figure to learn from: track the faded figure as an outline of
        the usual size, starting from nowhere in particular."""
        self.cold = True
        self._cold_since = time
        self.state = 'lens'
        self.radius = COLD_RADIUS
        angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        self.outline = None             # (unknown: the overlay draws a box)
        self.scale = min(1.0, WORK_RADIUS / self.radius)
        self.r = self.radius * self.scale
        radii = np.linspace(*COLD_RADII, 200) * self.radius * self.scale
        self.kernel = outline_kernel(radii)
        self.kernel_parts = outline_kernel_parts(radii)
        self.outline_kernels = None    # no outline to fit: the ring of the usual size
        self.history = []
        self.score = None
        self.evidence = None
        self._spot = None
        self.coasting = 0
        self._white = None
        self._step(frame, time, None, cursor)

    def _setup(self):
        """
        The white figure has been watched for a while: take its outline (whatever
        shape it is), fix the working scale and replay the buffered frames.
        """
        self.radius = float(np.sqrt(self._white_area / np.pi))
        outlines = [figure_outline(frame, centre, green_mask(frame))
                    for frame, _, centre in self._white if centre is not None]
        outlines = [points for points in outlines if points is not None and len(points) >= 8]
        if outlines:
            self.outline = max(outlines, key=len)
            radii = np.concatenate([np.linalg.norm(points, axis=1) for points in outlines])
        else:           # no outline could be traced: fall back to a circle of the same area
            angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
            self.outline = self.radius * np.stack([np.cos(angles), np.sin(angles)], 1)
            radii = np.full(64, self.radius)
        self.scale = min(1.0, WORK_RADIUS / self.radius)
        self.r = self.radius * self.scale
        self.kernel = outline_kernel(radii * self.scale)
        self.kernel_parts = outline_kernel_parts(radii * self.scale)
        self.outline_kernels = (outline_kernels(self.outline, self.scale, OUTLINE_STEP)
                                if OUTLINE_STEP else None)
        self.history = []              # (time, gray, gray with the cursor blanked, background offset)
        self.score = None              # best path score ending at each pixel
        self.evidence = None
        self._spot = None              # the estimate, in working pixels
        self.coasting = 0
        buffered, self._white = self._white, None
        for frame, time, white in buffered:
            self._step(frame, time, white, green_mask(frame))

    # ------------------------------------------------------------------ tracking
    def _evidence(self, reference, gray, offset, time):
        """
        How strongly each pixel looks like the centre of the figure's outline.

        Once faded, the figure is a lens: inside it the background shows through
        almost unchanged, but along its edge the picture is bent. Against a frame
        from a moment ago, shifted by how far the background has scrolled since,
        that edge is what doesn't match - an outline of the figure. (Older frames
        are no use: the background wobbles as it scrolls and stops lining up.)
        """
        H, W = gray.shape
        ages = [abs((time - entry[0]) * NOMINAL_FPS - COMPARE_FRAMES) for entry in self.history[:-1]]
        _, _, old, old_offset = self.history[int(np.argmin(ages))]
        shift = offset - old_offset
        moved = cv2.warpAffine(old, np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]), (W, H),
                               flags=cv2.INTER_LINEAR, borderValue=np.nan)
        difference = np.abs(reference - moved)      # NaN where either frame had the cursor
        known = np.isfinite(difference)
        typical = float(np.median(difference[known])) if known.any() else 0.0
        g32 = gray.astype(np.float32)
        gradient = np.hypot(cv2.Sobel(g32, cv2.CV_32F, 1, 0), cv2.Sobel(g32, cv2.CV_32F, 0, 1))
        # a sharp background edge that is a fraction of a pixel out is not the figure
        difference = np.where(known, difference, typical) / (cv2.GaussianBlur(gradient, (0, 0), 1.5) * 0.25 + 8.0)
        difference = np.minimum(difference, np.percentile(difference, 99.5))
        difference[~known] = np.median(difference)
        self.edges = difference         # (read by OutlineTurn for the overlay only)
        if getattr(self, 'outline_kernels', None):
            # The exact outline, at every turn: the best-fitting turn at each spot,
            # less the surround's average (a patch busy all over scores nothing).
            # Against full-resolution ground truth, the frame's strongest spot is
            # on the figure 57-70% of the time this way, 34-63% with the ring
            # (shape-20261119-134149 / 144318 / 175343).
            match = self._best_outline_fit(difference)
            if SURROUND_WEIGHT:
                match -= SURROUND_WEIGHT * cv2.filter2D(difference, -1, self.kernel_parts[1],
                                                        borderType=cv2.BORDER_REFLECT)
        else:
            match = cv2.filter2D(difference, -1, self.kernel, borderType=cv2.BORDER_REFLECT)
        b = int(self.r)
        inner = match[b:-b, b:-b] if min(match.shape) > 2 * b + 4 else match
        match = np.clip((match - np.median(inner)) / (inner.std() + 1e-6), 0, None)
        return cv2.GaussianBlur(match, (0, 0), 1.0)

    def _best_outline_fit(self, image):
        """
        The best of the outline_kernels at every pixel (what filter2D with each
        and a per-pixel max gives), done with one forward DFT shared by all the
        turns: ~5x faster, so the frame rate holds.
        """
        kernels = self.outline_kernels
        half = kernels[0].shape[0] // 2
        height, width = image.shape
        padded = cv2.copyMakeBorder(image, half, half, half, half, cv2.BORDER_REFLECT)
        size = (cv2.getOptimalDFTSize(padded.shape[0]), cv2.getOptimalDFTSize(padded.shape[1]))
        cache = getattr(self, '_outline_spectra', None)
        if cache is None or cache[0] != size:
            spectra = []
            for kernel in kernels:
                placed = np.zeros(size, np.float32)
                placed[:kernel.shape[0], :kernel.shape[1]] = kernel[::-1, ::-1]   # correlation
                spectra.append(cv2.dft(placed, flags=cv2.DFT_COMPLEX_OUTPUT))
            cache = self._outline_spectra = (size, spectra)
        canvas = np.zeros(size, np.float32)
        canvas[:padded.shape[0], :padded.shape[1]] = padded
        spectrum = cv2.dft(canvas, flags=cv2.DFT_COMPLEX_OUTPUT)
        best = None
        for kernel_spectrum in cache[1]:
            fit = cv2.idft(cv2.mulSpectrums(spectrum, kernel_spectrum, 0),
                           flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
            fit = fit[2 * half:2 * half + height, 2 * half:2 * half + width]
            best = fit if best is None else np.maximum(best, fit)
        return best

    def _step(self, frame, time, white, cursor):
        small = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if self._prev_small is not None and white is None and \
                cv2.absdiff(gray, self._prev_small).mean() < 0.8:
            return  # repeated game frame: nothing new to see
        dt = 1.0 if self._prev_time is None else max((time - self._prev_time) * NOMINAL_FPS, 0.5)
        self._prev_time = time
        self._prev_small = gray
        H, W = gray.shape
        # The crosshair rides on the figure and moves with the mouse, not with the
        # background: left in, the tracker ends up following its own cursor.
        reference = gray.astype(np.float32)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(cursor)
        solid = np.isin(labels, [k for k in range(1, count) if stats[k, 4] >= 40]).astype(np.uint8)
        if solid.any():
            solid = cv2.dilate(solid, np.ones((CURSOR_MARGIN,) * 2, np.uint8))
            reference[cv2.resize(solid, (W, H), interpolation=cv2.INTER_AREA) > 0] = np.nan
        Y, X = np.ogrid[:H, :W]
        if not self.history:
            self.history.append((time, gray, reference, np.zeros(2)))
        else:
            _, previous, _, previous_offset = self.history[-1]
            flow = self._dis.calc(previous, gray, None)
            offset = previous_offset + np.median(flow.reshape(-1, 2), 0)
            self.history.append((time, gray, reference, offset))
            self.history = self.history[-HISTORY:]
            self.evidence = self._evidence(reference, gray, offset, time)

        if self.cold and white is None and self.score is None:
            if self.evidence is None:
                return              # one more frame needed to compare with
            # Nowhere is favoured: every place starts level; the evidence decides
            self.score = self.evidence.astype(np.float32).copy()
            _, _, _, best = cv2.minMaxLoc(self.score)
            self._spot = np.array(best, np.float64)
        elif white is not None or self.score is None or self.evidence is None:
            # The white figure is in plain sight: every path starts from it.
            spot = (white * self.scale) if white is not None else self._spot
            if spot is None:
                spot = self.position * self.scale
            self._spot = np.asarray(spot, np.float64)
            self.score = (-0.5 * ((X - spot[0]) ** 2 + (Y - spot[1]) ** 2)).astype(np.float32)
            self.coasting = 0
        else:
            # The figure can have gone anywhere within reach of where it was. The
            # score of a pixel is the outline evidence along the best path that
            # ends there, so every place it might be is followed at once and a
            # look-alike that scores well for a few frames cannot capture it.
            reach = int(round(SPEED_MAX * self.r * dt)) + 1
            near = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1,) * 2)
            if VELOCITY_PRIOR and getattr(self, '_velocity', None) is not None:
                # It keeps going the way it was going: every path is carried
                # along the estimate's recent motion before spreading
                shift = VELOCITY_PRIOR * self._velocity * dt
                self.score = cv2.warpAffine(self.score, np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]),
                                            (W, H), flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_REPLICATE)
            self.score = cv2.dilate(self.score, near) + self.evidence
            self.score -= self.score.max()
            _, _, _, best = cv2.minMaxLoc(self.score)
            close = (X - self._spot[0]) ** 2 + (Y - self._spot[1]) ** 2 <= (reach + 1.5) ** 2
            _, here, _, stay = cv2.minMaxLoc(np.where(close, self.score, -np.inf).astype(np.float32))
            margin = COLD_SWITCH_MARGIN if self.cold else SWITCH_MARGIN
            if np.hypot(best[0] - stay[0], best[1] - stay[1]) > self.r and here > -margin:
                best = stay         # a path elsewhere is ahead, but not by enough to believe yet
            elif np.hypot(best[0] - self._spot[0], best[1] - self._spot[1]) > reach + 1.5:
                self.jumps += 1
            # finer than a pixel: centre of the score right around the peak
            x0, y0 = max(0, best[0] - 1), max(0, best[1] - 1)
            patch = np.exp(self.score[y0:best[1] + 2, x0:best[0] + 2] - self.score[best[1], best[0]])
            yy, xx = np.mgrid[y0:y0 + patch.shape[0], x0:x0 + patch.shape[1]]
            spot = np.array(((xx * patch).sum(), (yy * patch).sum())) / patch.sum()
            step = (spot - self._spot) / dt
            if np.hypot(*step) > reach / dt + 1.5:
                step = np.zeros(2)          # a switch to another path is not motion
            previous = getattr(self, '_velocity', None)
            self._velocity = step if previous is None else \
                VELOCITY_SMOOTH * previous + (1 - VELOCITY_SMOOTH) * step
            self._spot = spot
            seen = float(self.evidence[best[1], best[0]]) >= MIN_STRENGTH
            self.coasting = 0 if seen else self.coasting + dt
        if self.cold and time - self._cold_since < COLD_SETTLE_SECONDS:
            return                  # still gathering evidence: no estimate to act on yet
        self.position = np.clip(self._spot / self.scale, 0, (frame.shape[1] - 1, frame.shape[0] - 1))


class OutlineTurn:
    """
    For the overlay only (the tracking itself never uses it): which way the
    figure is turned, so its outline can be drawn on it and turn with it.
    The outline traced from the white figure is tried at a range of angles
    around the last one, and the angle whose outline lies along the most
    edge wins: the white figure's own edge while it is white, the bent
    background along its edge (the tracker's lens edges) once it has faded.
    """

    POINTS = 72             # outline points tried
    STEP = 3.0              # degrees between angles tried
    REACH = 24.0            # degrees either side of the last angle
    FULL_EVERY = 30         # frames: now and then try every angle
    SMOOTH = 0.5            # of the new angle taken each frame

    def __init__(self, outline):
        outline = np.asarray(outline, float)
        pick = np.linspace(0, len(outline), self.POINTS, endpoint=False).astype(int)
        self.outline = outline                  # full outline, frame pixels around the centre
        self.points = outline[pick]
        self.angle = 0.0                        # degrees, against the outline as traced
        self._frames = 0

    def _score(self, edges, centre, angles):
        rad = np.radians(angles)[:, None]
        x, y = self.points[:, 0][None], self.points[:, 1][None]
        xs = centre[0] + x * np.cos(rad) - y * np.sin(rad)
        ys = centre[1] + x * np.sin(rad) + y * np.cos(rad)
        h, w = edges.shape
        xi = np.clip(np.round(xs).astype(int), 0, w - 1)
        yi = np.clip(np.round(ys).astype(int), 0, h - 1)
        return edges[yi, xi].mean(axis=1)

    def update(self, frame, tracker):
        """Re-estimates the angle from this frame; returns it (degrees)."""
        if tracker.position is None or tracker.outline is None:
            return self.angle
        self._frames += 1
        full = self._frames % self.FULL_EVERY == 1
        if tracker.state == 'white' or getattr(tracker, 'edges', None) is None:
            # the white figure is in plain sight: its own edge, full size
            reach = int(np.abs(self.points).max()) + 6
            cx, cy = int(tracker.position[0]), int(tracker.position[1])
            x0, y0 = max(0, cx - reach), max(0, cy - reach)
            crop = frame[y0:cy + reach, x0:cx + reach]
            if crop.size == 0:
                return self.angle
            white = cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV), (0, 0, 190), (180, 60, 255))
            edges = cv2.GaussianBlur(cv2.morphologyEx(white, cv2.MORPH_GRADIENT,
                                                      np.ones((3, 3), np.uint8)).astype(np.float32),
                                     (0, 0), 1.5)
            centre = tracker.position - (x0, y0)
            scale = 1.0
        else:
            edges = tracker.edges
            scale = tracker.scale
            centre = tracker.position * scale
        if full:
            angles = np.arange(0.0, 360.0, self.STEP)
        else:
            angles = self.angle + np.arange(-self.REACH, self.REACH + 0.1, self.STEP)
        saved = self.points
        self.points = saved * scale
        try:
            scores = self._score(edges, centre, angles)
        finally:
            self.points = saved
        best = float(angles[int(np.argmax(scores))])
        if full and abs(((best - self.angle + 180) % 360) - 180) > self.REACH:
            # a full look found it elsewhere: only believe it if it beats the
            # best near the old angle clearly
            near = np.abs(((angles - self.angle + 180) % 360) - 180) <= self.REACH
            if near.any() and scores.max() < 1.15 * scores[near].max():
                best = float(angles[near][int(np.argmax(scores[near]))])
        turn = ((best - self.angle + 180) % 360) - 180
        self.angle = (self.angle + self.SMOOTH * turn) % 360
        return self.angle

    def polygon(self, centre):
        """The outline turned by the current angle, around CENTRE (frame pixels)."""
        rad = np.radians(self.angle)
        c, s = np.cos(rad), np.sin(rad)
        x, y = self.outline[:, 0], self.outline[:, 1]
        step = max(1, len(self.outline) // 120)
        return np.stack([centre[0] + x * c - y * s, centre[1] + x * s + y * c], 1)[::step]


class HumanMouse:
    """
    Turns a jittery target into hand-like motion: reacts after a short delay, eases
    towards the target with limited speed, and wobbles a little.
    """

    # Quicker than it was (0.10 / 0.06 / 0.12 / 900): the game judges the
    # crosshair, and in shape-20261119-175343 (jailed) it trailed the estimate
    # by 0.1-0.2 s at 300-850 px/s. Scored on the crosshair against
    # full-resolution truth over 5 games, these lose no game and add 2-6 points.
    def __init__(self, reaction=0.05, ease=0.03, lead=0.15, max_speed=1600.0, jitter=1.5, seed=None):
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
def _recorded_frame_times(path):
    """{frame index: seconds} from a debug recording's .txt log, or {} without one."""
    log = os.path.splitext(path)[0] + '.txt'
    times = {}
    try:
        with open(log) as handle:
            for line in handle:
                parts = line.split()
                if 'frame' in parts:
                    try:
                        times[int(parts[parts.index('frame') + 1])] = float(parts[0])
                    except (ValueError, IndexError):
                        pass
    except OSError:
        return {}
    return times


def replay(path, output=None, roi=None):
    """Runs the tracker over a recording; returns {frame: (x, y)} in video pixels."""
    capture = cv2.VideoCapture(path)
    fps = capture.get(cv2.CAP_PROP_FPS) or NOMINAL_FPS
    times = _recorded_frame_times(path)
    if times:
        print(f'[FADING] Using the live frame times from {os.path.splitext(path)[0]}.txt.')
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
        # A debug recording is written at a nominal 20 a second whatever the
        # live rate was (44 in 175343): its .txt has each frame's real time
        now = times.get(index, index / fps)
        position = tracker.update(view, now)
        if position is not None:
            estimates[index] = (position[0] + ox, position[1] + oy)
            pointer[index] = tuple(mouse.step(estimates[index], now))
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


# The mini-game window: a flat blue title bar ("The Lie Detector Mini-game") with
# the play area directly below it, the same width. The game draws its windows at a
# fixed pixel size, whatever the resolution.
TITLE_BAR_HSV = ((102, 135, 165), (108, 180, 205))
TITLE_BAR_WIDTH = (600, 900)
TITLE_BAR_HEIGHT = (60, 120)
PLAY_AREA_ASPECT = 0.67         # height / width (750 x 500 under a 750 x 88 title bar)


def find_game_panel(frame):
    """
    The mini-game's play area inside the game screen, as (x, y, width, height),
    or None while it isn't showing. Everything around it (the map, other windows)
    would mislead the tracker, which expects only the game's scrolling background.

    The play area is found from the window's title bar, not from its own colour:
    its sand-brown texture looks just like the dirt and rock of many maps.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, *TITLE_BAR_HSV)
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))   # title text
    count, _, stats, _ = cv2.connectedComponentsWithStats(blue)
    for k in sorted(range(1, count), key=lambda k: -stats[k, 4]):
        x, y, w, h, area = (int(v) for v in stats[k])
        if not (TITLE_BAR_WIDTH[0] <= w <= TITLE_BAR_WIDTH[1]
                and TITLE_BAR_HEIGHT[0] <= h <= TITLE_BAR_HEIGHT[1]):
            continue
        if area < 0.8 * w * h:      # sky and water are this blue too, but not a solid bar
            continue
        top, tall = y + h, int(round(w * PLAY_AREA_ASPECT))
        if top + tall > frame.shape[0]:
            continue
        # The same title bar tops the warning pop-up; the game is on once the
        # area below it shows the sand-brown background.
        below = hsv[top:top + tall, x:x + w]
        brown = cv2.inRange(below, (4, 50, 50), (32, 255, 255))
        if cv2.countNonZero(brown) < 0.5 * brown.size:
            continue
        inset = 4
        return (x + inset, top + inset, w - 2 * inset, tall - 2 * inset)
    return None


class _SessionRecorder:
    """Saves what the solver saw and did, so a game that went wrong can be replayed."""

    DIRECTORY = os.path.join('.settings', 'minigame_debug')
    KEEP = 6

    def __init__(self):
        import queue
        import threading
        import time
        self.videos = {}        # file suffix -> writer
        self.frames = 0         # frames written to the main video
        self.log = None
        # Encoding video costs several milliseconds a frame, and the tracker
        # needs every frame it can get, so the files are written on another thread.
        self._queue = queue.Queue(maxsize=90)
        self._thread = threading.Thread(target=self._write, daemon=True)
        self._thread.start()
        try:
            os.makedirs(self.DIRECTORY, exist_ok=True)
            old = sorted(name for name in os.listdir(self.DIRECTORY) if name.endswith('.txt'))
            for name in old[:-self.KEEP]:
                for path in (name, name[:-4] + '.mp4', name[:-4] + '-window.mp4'):
                    try:
                        os.remove(os.path.join(self.DIRECTORY, path))
                    except OSError:
                        pass
            self.base = os.path.join(self.DIRECTORY, time.strftime('shape-%Y%m%d-%H%M%S'))
            self.log = open(self.base + '.txt', 'w', encoding='utf-8')
        except OSError:
            self.log = None
        self._last = 0.0

    def note(self, text):
        if self.log is not None:
            self.log.write(text + '\n')
            self.log.flush()

    def frame(self, frame, now, suffix='', rate=120.0):
        """
        Queues FRAME for the video named by SUFFIX, at most RATE times a second.
        Returns its index in that video (to tie log lines to frames), or None.
        """
        if self.log is None or now - self._last < 1 / rate:
            return None
        self._last = now
        try:
            self._queue.put_nowait((suffix, frame))
        except Exception:       # the writer has fallen behind: skip rather than stall
            return None
        if suffix:
            return None
        self.frames += 1
        return self.frames - 1

    def _write(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            suffix, frame = item
            height, width = frame.shape[:2]
            video = self.videos.get(suffix)
            if video is None:
                video = self.videos[suffix] = (cv2.VideoWriter(
                    self.base + suffix + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20,
                    (width, height)), (width, height))
            if video[1] == (width, height):
                video[0].write(frame)

    def close(self):
        self._queue.put(None)
        self._thread.join(timeout=10)
        for video, _ in self.videos.values():
            video.release()
        if self.log is not None:
            self.log.close()


class _Grabber:
    """
    Grabs the screen on its own thread, so a grab (about as long as tracking one
    frame) overlaps with the tracking instead of adding to it.
    """

    def __init__(self):
        import threading
        self.area = None            # set by the reader; None pauses grabbing
        self._latest = None         # (area, frame, time)
        self._fresh = threading.Event()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        import time
        import mss
        with mss.mss() as screen:
            while not self._stop:
                area = self.area
                if area is None:
                    time.sleep(0.005)
                    continue
                frame = np.ascontiguousarray(np.asarray(screen.grab(area))[:, :, :3])
                self._latest = (area, frame, time.monotonic())
                self._fresh.set()

    def read(self, area, timeout=2.0):
        """The next unseen frame of AREA and when it was grabbed, or (None, None)."""
        import time
        self.area = area
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self._fresh.wait(0.05):
                continue
            self._fresh.clear()
            got, frame, when = self._latest
            if got == area:
                return frame, when
        return None, None

    def close(self):
        self._stop = True
        self._thread.join(timeout=2)


def _cursor_position():
    import ctypes
    from ctypes import wintypes
    point = wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
    return np.array((point.x, point.y), float)


RATE_SECONDS = 6.0  # the frame rate is measured over this long after the figure fades
LOST_FRAMES = 6     # nominal frames without a convincing measurement before it counts as lost


class CrosshairOffset:
    """
    Where the game draws its green crosshair, compared with the real mouse
    point. In some games it sits right on it, in others about 30 px right and
    down (recordings 2026-11-17 15:38, 19:37, 2026-11-18 13:04, 13:33; on
    14:45, 16:16 and 18:50 it was on it), and the crosshair is what the game
    counts. Within one game it stays put, so it is measured early while the
    mouse stands still (no lag in the picture), then locked for the rest of
    the game (once the game is over the screen is full of other things), and
    the mouse is put that far the other way so the crosshair lands on the
    figure's centre.
    """

    SAMPLES = 15            # recent measurements kept
    MIN_SAMPLES = 8         # needed before the correction is set (then locked)
    MAX_SPREAD = 8.0        # px: measurements must agree this well
    MAX_OFFSET = 45.0       # px: anything further is not the crosshair
    STILL = 2.0             # px the mouse may have moved since the last look
    LOOK = 80               # px around the mouse point searched for the crosshair
    MIN_PIXELS = 40         # green pixels of a crosshair, at least

    def __init__(self):
        from collections import deque
        self.samples = deque(maxlen=self.SAMPLES)
        self.offset = np.zeros(2)
        self.locked = False
        self._last_cursor = None

    def observe(self, frame, mouse):
        """FRAME: the latest grab; MOUSE: the real mouse point in FRAME pixels
        (read after the grab). Returns True when the correction changed."""
        if self.locked:
            return False
        mouse = np.asarray(mouse, float)
        still = self._last_cursor is not None and \
            np.abs(mouse - self._last_cursor).max() <= self.STILL
        self._last_cursor = mouse
        if not still:
            return False
        x, y = int(round(mouse[0])), int(round(mouse[1]))
        x0, y0 = max(0, x - self.LOOK), max(0, y - self.LOOK)
        crop = frame[y0:y + self.LOOK, x0:x + self.LOOK]
        if crop.size == 0:
            return False
        mask = (green_mask(crop) > 0).astype(np.uint8)
        count, _, stats, centres = cv2.connectedComponentsWithStats(mask)
        if count < 2:
            return False
        best = 1 + int(np.argmax(stats[1:, 4]))
        if stats[best, 4] < self.MIN_PIXELS:
            return False
        sample = centres[best] + (x0, y0) - mouse
        if np.abs(sample).max() > self.MAX_OFFSET:
            return False
        self.samples.append(sample)
        if len(self.samples) < self.MIN_SAMPLES:
            return False
        recent = np.array(self.samples)
        middle = np.median(recent, axis=0)
        spread = np.median(np.abs(recent - middle), axis=0)
        if spread.max() > self.MAX_SPREAD:
            return False
        self.offset = np.round(middle)
        self.locked = True
        return True


def _draw_overlay(view, tracker, region, origin, panel, position, taken_over, turn=None):
    from src.detection import screen_overlay as so
    boxes, labels, outlines = [], [], []
    if panel is not None:
        x, y, w, h = panel
        boxes.append((x, y, x + w, y + h, so.CYAN, 1))
    lost = tracker.state == 'lens' and tracker.coasting >= LOST_FRAMES
    status = 'LOST - no sign of the figure here' if lost else tracker.state
    if taken_over:
        status += ' | mouse is yours'
    if position is not None:
        half = tracker.radius or 30
        cx, cy = origin[0] + position[0], origin[1] + position[1]
        # (magenta, not green: green is the game's own crosshair colour)
        colour = so.RED if lost else so.MAGENTA
        if turn is not None:
            # the figure's own outline, turned the way the figure is turned
            outlines.append((turn.polygon((cx, cy)), colour, 3))
        else:
            boxes.append((cx - half, cy - half, cx + half, cy + half, colour, 3))
        labels.append((cx - half, cy - half - 20, status, colour))
    else:
        labels.append((origin[0] + 8, origin[1] + 8,
                       'SHAPE: ' + ('looking for the white figure' if panel is not None
                                    else 'looking for the mini-game window'), so.YELLOW))
    view.show(region, boxes, labels, outlines)


def play_live(title='MapleStory', move=False, timeout=60.0, roi=None, start_delay=0.0,
              search_timeout=45.0, game_seconds=25.0, stop=None, overlay=False):
    """
    Watches the game window and, with MOVE, keeps the mouse on the figure. Moving
    the mouse yourself (more than ~60 px from where it was put) hands control back;
    it then keeps watching (and recording) until the game is over.

    OVERLAY draws a box on the game where it thinks the figure is: magenta while it
    is tracking, red once it has lost it.

    START_DELAY: seconds to wait first (e.g. while the warning pop-up is up).
    SEARCH_TIMEOUT: give up if no white figure shows up in this long.
    GAME_SECONDS: stop this long after the figure fades (the game is over by then).
    STOP: optional threading.Event to end early.
    Returns why it ended: 'done', 'no window', 'no figure', 'taken over', 'stopped'
    or 'timed out'.
    """
    import ctypes
    import time
    from src.detection.violetta_solver import _client_region

    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    print(f'[FADING] Watching {title!r}; mouse control is {"ON" if move else "OFF (preview)"}.')
    if start_delay and (stop.wait(start_delay) if stop else time.sleep(start_delay)):
        return 'stopped'
    tracker = FadingFigureTracker()
    mouse = HumanMouse()
    crosshair = CrosshairOffset()
    turn = None             # the outline drawn on the overlay (OutlineTurn)
    recorder = _SessionRecorder()
    placed = None
    state = None
    started = time.monotonic()
    faded_at = None
    panel = None            # the mini-game window inside the game screen
    seen_panel = None       # where it was seen in the previous frame
    whole_screen = False    # gave up finding it; track on the whole screen instead
    not_focused_noted = False
    taken_over = False
    was_lost = False
    lens_frames = 0
    gone_since = None       # the game's background last seen (game over)
    view = None
    if overlay:
        try:
            from src.detection.screen_overlay import ScreenOverlay
            view = ScreenOverlay()
        except Exception as error:      # the game still gets played without the box
            print(f'[FADING] Overlay unavailable: {error}')

    def finish(result, message=None):
        if message:
            print(f'[FADING] {message}')
        if taken_over:
            result = 'taken over'
        if faded_at is not None and lens_frames:
            rate = lens_frames / min(max(time.monotonic() - faded_at, 1e-3), RATE_SECONDS)
            # Measured on a recorded game: it follows the figure at 15 frames a
            # second and loses it at 7.5.
            print(f'[FADING] Tracked at {rate:.0f} frames a second'
                  f'{" - too slow to follow the figure reliably" if rate < 12 else ""}.')
            recorder.note(f'rate {rate:.1f} fps')
        recorder.note(f'{time.monotonic() - started:.2f} end {result}')
        recorder.close()
        grabber.close()
        if view:
            view.close()
        return result

    grabber = _Grabber()
    try:
        while time.monotonic() - started < timeout:
            if stop is not None and stop.is_set():
                return finish('stopped')
            handle, region = _client_region(title)
            if roi:
                region = {'left': region['left'] + roi[0], 'top': region['top'] + roi[1],
                          'width': roi[2], 'height': roi[3]}
            # Once the mini-game window is known, grab only that: a smaller grab
            # is a faster one, and the tracker loses the figure at low frame rates.
            area = region if panel is None else {
                'left': region['left'] + panel[0], 'top': region['top'] + panel[1],
                'width': panel[2], 'height': panel[3]}
            frame, now = grabber.read(area)
            if frame is None:
                continue
            index = None
            if panel is None and not roi and not whole_screen:
                recorder.frame(frame, now, '-window', rate=5.0)     # how the window was looked for
            else:
                index = recorder.frame(frame, now)
            if not roi and panel is None and not whole_screen:
                found = find_game_panel(frame)
                if found is not None and found == seen_panel:
                    panel = found       # same place twice running: the window has settled
                seen_panel = found
                if panel is None:
                    if now - started > PANEL_WAIT:
                        recorder.note(f'{now - started:.2f} panel not found')
                        if move:
                            # Never steer the mouse by what is on the map: it is
                            # full of things that look like the figure.
                            return finish('no window', 'Mini-game window not recognised; '
                                                       'the mouse is left to you.')
                        whole_screen = True
                        print('[FADING] Mini-game window not recognised; watching the whole screen.')
                    else:
                        time.sleep(0.05)
                    if view:
                        _draw_overlay(view, tracker, region, (0, 0), None, None, taken_over)
                    continue
                print(f'[FADING] Mini-game window found at {panel}.')
                recorder.note(f'{now - started:.2f} panel {panel}')
                continue        # the next grab is of the panel alone
            origin = (panel[0], panel[1]) if panel is not None else (0, 0)
            position = tracker.update(frame, now)
            if faded_at is not None and now - faded_at <= RATE_SECONDS:
                lens_frames += 1    # (later the game is over and the screen stands still)
            if tracker.state != state:
                state = tracker.state
                print(f'[FADING] {state}')
                recorder.note(f'{now - started:.2f} state {state}')
                if state == 'lens':
                    faded_at = now
                    if tracker.cold:
                        print('[FADING] No white figure seen (arrived after it faded); '
                              'looking for the faded figure by its outline instead.')
                        recorder.note(f'{now - started:.2f} cold start')
            if tracker.state == 'search' and now - started > search_timeout:
                return finish('no figure', 'No white figure appeared; giving up.')
            if faded_at is not None and now - faded_at > game_seconds:
                return finish('done', 'Game time is up; done.')
            if tracker.state == 'lens' and not _game_background(frame):
                gone_since = gone_since or now
                if now - gone_since > 1.0:
                    # 2026-11-19: after a lost game the jail room's lit window
                    # was taken for the figure for 44 s. Off the game: stop.
                    return finish('done', 'The mini-game window has closed; done.')
            else:
                gone_since = None
            if view:
                try:
                    if turn is None and tracker.outline is not None:
                        turn = OutlineTurn(tracker.outline)
                    if turn:
                        turn.update(frame, tracker)
                except Exception as error:     # a box instead; the game is still played
                    print(f'[FADING] Outline drawing off: {error}')
                    turn = False
                _draw_overlay(view, tracker, region, origin, panel, position, taken_over,
                              turn or None)
            if position is None:
                continue
            lost = tracker.state == 'lens' and tracker.coasting >= LOST_FRAMES
            if lost != was_lost:
                was_lost = lost
                print('[FADING] Lost the figure.' if lost else '[FADING] Following the figure again.')
            target = mouse.step((region['left'] + origin[0] + position[0],
                                 region['top'] + origin[1] + position[1]), now)
            cursor = _cursor_position()
            if crosshair.observe(frame, (cursor[0] - region['left'] - origin[0],
                                         cursor[1] - region['top'] - origin[1])):
                dx, dy = crosshair.offset
                print(f'[FADING] The game draws the crosshair {dx:+.0f},{dy:+.0f} px from the '
                      'mouse; aiming so the crosshair sits on the figure.')
                recorder.note(f'{now - started:.2f} crosshair offset {dx:.0f},{dy:.0f}')
            recorder.note(
                f'{now - started:.2f} figure {origin[0] + position[0]:.0f},'
                f'{origin[1] + position[1]:.0f} mouse {target[0]:.0f},{target[1]:.0f} '
                f'cursor {cursor[0]:.0f},{cursor[1]:.0f} coasting {tracker.coasting:.0f}'
                f'{" by-hand" if taken_over else ""}'
                f'{"" if index is None else f" frame {index}"}'
            )
            if not move or taken_over:
                continue
            if user32.GetForegroundWindow() != handle:
                if not not_focused_noted:
                    not_focused_noted = True
                    print('[FADING] The game window is not in front; not moving the mouse.')
                    recorder.note(f'{now - started:.2f} game not focused')
                continue
            if placed is not None and np.linalg.norm(cursor - placed) > 60:
                # Keep watching: the rest of the recording then shows where the
                # figure really was, and the routine stays paused until the game ends.
                taken_over = True
                print('[FADING] Mouse moved by hand; it is yours now. Still watching.')
                recorder.note(f'{now - started:.2f} taken over')
                continue
            aim = (int(target[0] - crosshair.offset[0]), int(target[1] - crosshair.offset[1]))
            user32.SetCursorPos(*aim)
            placed = np.array(aim, float)
    except Exception:
        finish('error')     # close the recording and the overlay
        raise
    return finish('timed out', 'Timed out.')


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
    parser.add_argument('--overlay', action='store_true',
                        help='Live: draw a box on the game where the figure is thought to be.')
    args = parser.parse_args()
    if args.video:
        replay(args.video, args.output, _roi(args.roi))
    else:
        play_live(args.title, args.move, args.timeout, _roi(args.roi), overlay=args.overlay)


if __name__ == '__main__':
    main()
