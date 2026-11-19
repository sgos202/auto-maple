"""
Reads a Violetta shuffle whose walking rows are not known in advance.

Some games don't use the fixed layouts in violetta_solver.LAYOUTS: every round
picks new sizes and heights for the walkers (seen in
logs/violetta/violetta-20261114-213630.mp4), not every round keeps exactly one
Violetta in place, and the biggest walkers get cut off at the stage edges.
Each walker still keeps ONE size and height for its whole walk, so the rows
can be found from the frames of that shuffle:

1. The walk is the last run of motion before the answer buttons light up.
2. A few frames from it are scanned at many sizes (coarse), and every hit is
   refined to an exact size, height and x.
3. Hits that share a size and height across frames form a row.
4. Each row is tracked through every frame of the walk. A row seen only while
   Violettas step in or out of their rows (sizes in between) is dropped, as is
   a row that sits on the same Violetta as a better one.
5. Each row's continuous path gives its start and end slots. Violettas whose
   row was never seen walking (hidden, or stepped into a row in place) take
   the slots left over, preferring to have stayed put.

All coordinates are window space, like violetta_solver.
"""

import itertools
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from src.detection.violetta_solver import (SLOT_COUNT, SLOT_STEP, STAND_PIVOT_Y, _ScaledTemplates,
                                           local_contrast, nearest_slots, slot_x)


PAD = 64                    # canvas padding, so walkers cut off at the stage edge still match
Y_MIN, Y_MAX = 88, 152      # pivot heights a walking row can have
# Coarse sizes, evenly spaced: a size error costs matching score in pixels, so
# big sprites need finer steps than small ones (~8 px of sprite size apart)
SCALES = np.arange(0.40, 1.85, 0.09)
SEEN = 0.5                  # match score that counts as the walker being there
SAMPLES = 8                 # frames of the walk scanned to find the rows
MAX_ROWS = 6                # rows tracked (spares; duplicates are dropped after tracking)

_inner = _outer = None


def _pools():
    global _inner, _outer
    if _inner is None:
        _inner, _outer = ThreadPoolExecutor(16), ThreadPoolExecutor(12)
    return _inner, _outer


def pad(canvas):
    return cv2.copyMakeBorder(np.asarray(canvas, np.float32), 0, 0, PAD, PAD,
                              cv2.BORDER_CONSTANT, value=0)


# OpenCV's masked TM_CCOEFF_NORMED is 2-4x slower than three plain
# correlations giving the same scores (checked equal to 1e-4 on stage
# canvases, 2026-11-18): the Violetta tracker spends nearly all its time here
# and must keep up with every live frame.
_PREPARED = {}


def _prepared(template, mask):
    key = (id(template), id(mask))
    entry = _PREPARED.get(key)
    if entry is None or entry[0] is not template or entry[1] is not mask:
        weights = (np.asarray(mask) > 0).astype(np.float32)
        count = float(weights.sum())
        values = np.asarray(template, np.float32)
        mean = float((values * weights).sum() / max(count, 1.0))
        centred = ((values - mean) * weights).astype(np.float32)
        entry = (template, mask, weights, centred, float((centred * centred).sum()), count)
        _PREPARED[key] = entry
    return entry[2:]


def _best_match(image, image_squared, template, mask):
    """(score, y, x) of _match's best place, or None. OpenCV calls only (they
    release the GIL, so the tracker's search threads really run in parallel;
    numpy's small-array steps capped them at ~4x). IMAGE_SQUARED = image**2."""
    if image.shape[0] < template.shape[0] or image.shape[1] < template.shape[1]:
        return None
    weights, centred, centred_energy, count = _prepared(template, mask)
    if count < 1 or centred_energy <= 0:
        return None
    numerator = cv2.matchTemplate(image, centred, cv2.TM_CCORR)
    total = cv2.matchTemplate(image, weights, cv2.TM_CCORR)
    energy = cv2.matchTemplate(image_squared, weights, cv2.TM_CCORR)
    variance = cv2.subtract(energy, cv2.multiply(total, total, scale=1.0 / count))
    _, flat = cv2.threshold(variance, 1e-6 * count, 1.0, cv2.THRESH_BINARY_INV)
    variance = cv2.max(variance, 1e-12)
    result = cv2.divide(numerator, cv2.sqrt(cv2.multiply(variance, centred_energy)))
    # flat patches (padding) score -1, as _match: result*(1-flat) - flat
    result = cv2.subtract(cv2.multiply(result, cv2.subtract(1.0, flat)), flat)
    _, value, _, (x, y) = cv2.minMaxLoc(result)
    return float(value), y, x


def _match(image, template, mask):
    """Masked TM_CCOEFF_NORMED; -1 where it is undefined (a flat patch)."""
    if image.shape[0] < template.shape[0] or image.shape[1] < template.shape[1]:
        return None
    weights, centred, centred_energy, count = _prepared(template, mask)
    if count < 1 or centred_energy <= 0:
        return None
    image = np.asarray(image, np.float32)
    numerator = cv2.matchTemplate(image, centred, cv2.TM_CCORR)
    total = cv2.matchTemplate(image, weights, cv2.TM_CCORR)
    energy = cv2.matchTemplate(image * image, weights, cv2.TM_CCORR)
    variance = energy - total * total / count
    flat = variance <= 1e-6 * count
    result = numerator / np.sqrt(np.maximum(variance, 1e-12) * centred_energy)
    result[flat] = -1.0
    return np.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0)


class _SizeBank:
    """The 8 walking poses at a size, built once. Height only moves where they are matched."""

    def __init__(self, sprites):
        self.sprites, self.cache = sprites, {}

    def get(self, scale):
        key = round(float(scale), 3)
        if key not in self.cache:
            items = []
            for gray, alpha, (px, py) in self.sprites.walk:
                template = local_contrast(cv2.resize(gray, None, fx=key, fy=key,
                                                     interpolation=cv2.INTER_AREA))
                mask = (cv2.resize(alpha, None, fx=key, fy=key,
                                   interpolation=cv2.INTER_NEAREST) > 128).astype(np.uint8)
                items.append((template, mask, px * key, py * key))
            self.cache[key] = items
        return self.cache[key]


def _coarse(canvas, bank, pool, threshold=0.22, keep=14):
    """Candidate walkers [(score, scale, pivot x, pivot y)] at the coarse sizes."""
    def one(job):
        scale, (template, mask, ox, oy) = job
        top = max(0, int(round(Y_MIN - oy)))
        bottom = min(canvas.shape[0], int(round(Y_MAX - oy)) + template.shape[0])
        result = _match(canvas[top:bottom], template, mask)
        if result is None:
            return scale, None, None
        profile = np.full(canvas.shape[1] + 200, -1.0, np.float32)
        heights = np.zeros_like(profile)
        xs = np.arange(result.shape[1]) + int(round(ox))
        profile[xs] = result.max(axis=0)
        heights[xs] = result.argmax(axis=0) + top + oy
        return scale, profile, heights

    jobs = [(scale, item) for scale in SCALES for item in bank.get(scale)]
    per_scale = {}
    for scale, profile, heights in pool.map(one, jobs):
        if profile is None:
            continue
        if scale not in per_scale:
            per_scale[scale] = [profile, heights]
        else:       # best of the 8 poses
            better = profile > per_scale[scale][0]
            per_scale[scale][0][better] = profile[better]
            per_scale[scale][1][better] = heights[better]
    candidates = []
    for scale, (profile, heights) in per_scale.items():
        taken = []
        for x in np.argsort(profile)[::-1][:60]:
            if profile[x] < threshold or len(taken) >= 3:
                break
            if all(abs(x - t) >= 12 for t in taken):
                taken.append(x)
                candidates.append((float(profile[x]), scale, int(x), float(heights[x])))
    candidates.sort(reverse=True)
    kept = []       # neighbouring sizes find the same Violetta: one candidate per place and size
    for c in candidates:
        if all(abs(c[2] - k[2]) > 10 or abs(c[1] - k[1]) > 0.1 for k in kept):
            kept.append(c)
    return kept[:keep]


def _refine(canvas, bank, candidate, y_window=9, x_window=8):
    """Best (score, scale, pivot x, pivot y, pose) near a coarse candidate."""
    best = None

    def try_scale(scale, x0, y0):
        nonlocal best
        for pose, (template, mask, ox, oy) in enumerate(bank.get(scale)):
            top, left = int(round(y0 - y_window - oy)), int(round(x0 - x_window - ox))
            if top < 0 or left < 0:
                continue
            image = canvas[top:top + template.shape[0] + 2 * y_window,
                           left:left + template.shape[1] + 2 * x_window]
            result = _match(image, template, mask)
            if result is None:
                continue
            value = float(result.max())
            if best is None or value > best[0]:
                yy, xx = np.unravel_index(np.argmax(result), result.shape)
                best = (value, round(scale, 3), left + xx + ox, top + yy + oy, pose)

    _, scale0, x0, y0 = candidate
    for k in (-2, -1, 0, 1, 2):
        try_scale(scale0 * 1.05 ** k, x0, y0)
    if best is not None:
        _, scale1, x1, y1 = best[:4]
        for k in (-2, -1, 1, 2):
            try_scale(scale1 * 1.015 ** k, x1, y1)
    return best


def _motion(canvases, lag=3):
    energy = [0.0] * len(canvases)
    for i in range(lag, len(canvases)):
        a = np.clip(canvases[i][75:250, PAD + 15:PAD + 425], -3, 3)
        b = np.clip(canvases[i - lag][75:250, PAD + 15:PAD + 425], -3, 3)
        energy[i] = float(np.mean(cv2.GaussianBlur(np.abs(a - b), (0, 0), 3) > 0.6))
    return energy


def walk_segment(canvases, threshold=0.06, gap=4):
    """(first, last) frame of the last run of motion: the walk just before the answer."""
    energy = _motion(canvases)
    end = len(energy) - 1
    while end > 0 and energy[end] < threshold:
        end -= 1
    start, quiet = end, 0
    while start > 0:
        if energy[start - 1] >= threshold:
            quiet = 0
        else:
            quiet += 1
            if quiet > gap:
                break
        start -= 1
    return start + quiet, end


def _discover(canvases, bank, frames):
    """Rows [(scale, pivot y, frames seen in)] from the walkers found in FRAMES."""
    inner, outer = _pools()

    def frame_hits(i):
        hits = []
        for candidate in _coarse(canvases[i], bank, inner):
            hit = _refine(canvases[i], bank, candidate)
            if hit and hit[0] >= SEEN and all(abs(hit[2] - h[2]) > 8 or
                                              abs(np.log(hit[1] / h[1])) > 0.06 for h in hits):
                hits.append(hit)
        return [(*hit[:4], int(i)) for hit in hits]

    hits = [h for hs in outer.map(frame_hits, frames) for h in hs]
    # Violettas standing in their slots are not walking
    hits = [h for h in hits if not (abs(np.log(h[1])) < 0.07 and abs(h[3] - STAND_PIVOT_Y) < 5 and
                                    min(abs(h[2] - PAD - slot_x(s)) for s in range(SLOT_COUNT)) < 10)]
    hits.sort(reverse=True)
    clusters = []
    for h in hits:
        for cluster in clusters:
            if abs(np.log(h[1] / cluster[0][1])) < 0.045 and abs(h[3] - cluster[0][3]) < 4:
                if all(m[4] != h[4] for m in cluster):
                    cluster.append(h)
                break
        else:
            clusters.append([h])
    clusters.sort(key=lambda cluster: -sum(m[0] for m in cluster))
    rows = []
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        scale = float(np.median([m[1] for m in cluster]))
        y = float(np.median([m[3] for m in cluster]))
        if any(abs(np.log(scale / r[0])) < 0.045 and abs(y - r[1]) < 4 for r in rows):
            continue
        rows.append((scale, y, len(cluster)))
        if len(rows) == MAX_ROWS:
            break
    strong = [r for r in rows if r[2] >= 3]
    return strong if len(strong) >= 2 else rows


def _track(canvases, sprites, rows, frames, standing_margin=0.03):
    """
    Per row, {frame: (x, score)} of its best match in each frame. A match that
    a Violetta standing in a slot explains better (a row close to standing size
    fits the ones that have already stepped out) gets score -1.
    """
    _, outer = _pools()
    templates = [_ScaledTemplates(sprites.walk, scale, y) for scale, y, _ in rows]
    standing = _ScaledTemplates(sprites.stand + sprites.walk, 1.0, STAND_PIVOT_Y)

    near_standing = [abs(np.log(scale)) < 0.25 for scale, _, _ in rows]

    def standing_score(canvas, x, reach=8):
        best = -1.0
        for template, mask, top, bottom, offset in standing.items:
            left = x - reach - offset
            if left < 0 or bottom > canvas.shape[0]:
                continue
            result = _match(canvas[top:bottom, left:left + template.shape[1] + 2 * reach],
                            template, mask)
            if result is not None:
                best = max(best, float(result.max()))
        return best

    def one(i):
        found = []
        for template, near in zip(templates, near_standing):
            profile = template.profile(canvases[i])
            x = int(np.argmax(profile))
            score = float(profile[x])
            if near and score >= SEEN and \
                    standing_score(canvases[i], x) > score + standing_margin:
                score = -1.0
            found.append((x - PAD, score))
        return i, found

    tracks = [{} for _ in rows]
    for i, found in outer.map(one, frames):
        for k, (x, score) in enumerate(found):
            tracks[k][i] = (x, score)
    return tracks


def _segments(points, max_gap=4, speed=40):
    """Links (frame, x) detections a few frames apart into pieces of walk."""
    pieces = []
    for i, x in points:
        best = None
        for piece in pieces:
            j, last_x = piece[-1]
            if 0 < i - j <= max_gap and abs(x - last_x) <= speed * (i - j) \
                    and (best is None or len(piece) > len(best)):
                best = piece
        if best is None:
            pieces.append([(i, x)])
        else:
            best.append((i, x))
    return pieces


def longest_path(points, min_piece=3, max_gap=50, gap_speed=12, gap_margin=20):
    """
    The longest continuous walk in (frame, x) detections. Pieces of walk are
    joined across gaps where the walker was hidden behind another, if the jump
    fits a walking pace (the new-style walkers move ~3-10 px a frame). Stray
    matches are left out: a row's template briefly fitting someone else, or a
    Violetta whose size passes through this row's while she steps in or out.
    """
    def blip(piece):
        # Short and standing still: another Violetta passing through this row's
        # size as she steps in or out of her own row, in her slot
        xs = [x for _, x in piece]
        return len(piece) < 8 and max(xs) - min(xs) < 6

    everything = _segments(points)
    pieces = [p for p in everything if len(p) >= min_piece and not blip(p)]
    if not pieces:
        return max(everything, key=len) if points else []
    pieces.sort(key=lambda p: p[0][0])
    best = []       # best[n]: longest chain of pieces ending with piece n
    for n, piece in enumerate(pieces):
        chain = []
        for m in range(n):
            (j, last_x), (i, x) = pieces[m][-1], piece[0]
            if 0 < i - j <= max_gap and abs(x - last_x) <= gap_margin + gap_speed * (i - j)                     and len(best[m]) > len(chain):
                chain = best[m]
        best.append(chain + piece)
    return max(best, key=len)


def _seen(track):
    return [(i, x) for i, (x, score) in sorted(track.items()) if score >= SEEN]


def _walk_rows(tracks):
    """
    Indices of the rows that are real walkers. Each frame's Violetta belongs to
    the row matching her best, so a row that only fits other rows' Violettas
    (a duplicate, or a size close to another walker's) is left with little.
    What remains must be a walk seen for a while: a row seen only while
    Violettas step into or out of their rows is a size in between.
    """
    tracks = _drop_shared_matches(tracks, range(len(tracks)))
    paths = {k: longest_path(_seen(t)) for k, t in enumerate(tracks)}
    walking = [k for k, path in paths.items()
               if len(path) >= max(5, 0.1 * len(tracks[k]))]
    walking.sort(key=lambda k: -len(paths[k]))
    return walking[:SLOT_COUNT]


def _drop_shared_matches(tracks, rows, near=18):
    """Where two rows match the same Violetta in a frame (she crosses another
    row's walker), only the better match is that row's walker."""
    tracks = [dict(t) for t in tracks]
    for a, b in itertools.combinations(rows, 2):
        for i in set(tracks[a]) & set(tracks[b]):
            (xa, sa), (xb, sb) = tracks[a][i], tracks[b][i]
            if sa >= SEEN and sb >= SEEN and abs(xa - xb) < near:
                loser = b if sa >= sb else a
                tracks[loser][i] = (tracks[loser][i][0], -1.0)
    return tracks


def _resolve(tracks, rows, count=2):
    tracks = _drop_shared_matches(tracks, rows)
    starts, ends, paths = [], [], {}
    for k in rows:
        path = longest_path(_seen(tracks[k]))
        paths[f'r{k}'] = path
        starts.append(float(np.median([x for _, x in path[:count]])))
        ends.append(float(np.median([x for _, x in path[-count:]])))
    start_slots, start_error = nearest_slots(starts)
    end_slots, end_error = nearest_slots(ends)
    mapping = {f'r{k}': (a, b) for k, a, b in zip(rows, start_slots, end_slots)}
    # The Violettas not seen walking take the slots left over, staying put if they can
    free_starts = [s for s in range(SLOT_COUNT) if s not in start_slots]
    free_ends = [s for s in range(SLOT_COUNT) if s not in end_slots]
    fill = min(itertools.permutations(free_ends),
               key=lambda p: sum(a != b for a, b in zip(free_starts, p)))
    for n, (a, b) in enumerate(zip(free_starts, fill)):
        mapping[f'unseen{n}'] = (a, b)
    worst = round(max(start_error, end_error), 1)
    report = {'start_x': [round(x) for x in starts], 'end_x': [round(x) for x in ends],
              'worst_slot_error': worst, 'rows walking': len(rows),
              'consistent': worst < SLOT_STEP / 3}
    return mapping, report, paths


class RowFinder:
    """Finds and follows the walking rows of one shuffle from its buffered canvases."""

    def __init__(self, sprites):
        self.sprites = sprites
        self.last_walks = {}
        self.bank = _SizeBank(sprites)
        for scale in SCALES:
            self.bank.get(scale)

    def read(self, canvases):
        """
        CANVASES: local-contrast window canvases (unpadded) from the end of the
        last round up to now. Returns ({row: (start slot, end slot)}, report),
        or (None, reason).
        """
        started = time.perf_counter()
        if len(canvases) < 8:
            return None, f'only {len(canvases)} frames saved'
        canvases = [pad(c) for c in canvases]
        first, last = walk_segment(canvases)
        if last - first < 6:
            return None, f'no walk found (motion only in frames {first}-{last})'
        low, high = first + (last - first) * 0.03, first + (last - first) * 0.97
        samples = np.unique(np.linspace(low, high, SAMPLES).round().astype(int))
        rows = _discover(canvases, self.bank, samples)
        if not rows:
            return None, 'no walking rows found'
        frames = list(range(max(0, first - 8), min(len(canvases), last + 8)))
        tracks = _track(canvases, self.sprites, rows, frames)
        walking = _walk_rows(tracks)
        if not walking:
            return None, 'rows found but none seen walking: ' + \
                ' '.join(f'({s:.2f}, y{y:.0f})' for s, y, _ in rows)
        mapping, report, paths = _resolve(tracks, walking)
        # Each walker's path [(canvas index, x)] and row (scale, pivot y), for drawing
        self.last_walks = {key: (path, rows[int(key[1:])][:2]) for key, path in paths.items()}
        report['rows'] = ' '.join(f'({rows[k][0]:.2f}, y{rows[k][1]:.0f})' for k in walking)
        report['ms'] = round(1000 * (time.perf_counter() - started))
        return mapping, report
