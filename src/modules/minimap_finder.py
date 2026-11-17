"""
Finds the minimap's map area on its own, from the white outline the game draws
around the map picture (below the 'MINI MAP' title bar and the map's name):

    +--------------------------+
    | MINI MAP        [-][+]   |
    | (icon) Hidden Street     |
    |##########################|   <- 2 px white line
    |##                      ##|
    |##     map picture      ##|   <- 3 px white sides
    |##                      ##|
    |##########################|
    +--------------------------+

The outline is the same on every map, whatever the map's size, and wherever
the game window or the minimap sits, so nothing has to be drawn by hand.
"""

import numpy as np


WHITE_MIN = 225         # every colour channel at least this: the outline's white
MIN_WIDTH = 60          # px; the shortest outline (a very small map)
MIN_HEIGHT = 20
MAX_TOP_ROWS = 6        # the top line is 2 px thick
MAX_BOTTOM_ROWS = 12    # the bottom one can look thicker: pale map rows touch it
MAX_SIDE = 6            # the sides are 3 px thick
EDGE_SLACK = 2          # px the lines' ends may differ by
SIDE_FILLED = 0.9       # share of a side column that must be white
MAX_INSIDE_WHITE = 0.5  # a map picture is not mostly white


def _white_runs(white):
    """(row, start, end) of every horizontal white run at least MIN_WIDTH long;
    END is exclusive. Sorted by row."""
    height, width = white.shape
    padded = np.zeros((height, width + 2), np.int8)
    padded[:, 1:-1] = white
    edges = np.diff(padded, axis=1)
    rows, starts = np.nonzero(edges == 1)
    _, ends = np.nonzero(edges == -1)
    keep = ends - starts >= MIN_WIDTH
    return rows[keep], starts[keep], ends[keep]


def _bands(white):
    """Groups runs on consecutive rows with the same ends into lines:
    [first row, last row, left, right) with the ends they all share."""
    bands, active = [], []
    last_row = None
    for row, start, end in zip(*_white_runs(white)):
        row, start, end = int(row), int(start), int(end)
        if row != last_row:
            active = [band for band in active if band[1] >= row - 1]
            last_row = row
        for band in active:
            if (band[1] == row - 1 and abs(band[4] - start) <= EDGE_SLACK
                    and abs(band[5] - end) <= EDGE_SLACK):
                band[1] = row
                band[2] = max(band[2], start)
                band[3] = min(band[3], end)
                break
        else:
            band = [row, row, start, end, start, end]
            bands.append(band)
            active.append(band)
    return [tuple(band[:4]) for band in bands]


def _side(columns):
    """How many leading columns are white nearly all the way down."""
    count = 0
    for share in columns:
        if share < SIDE_FILLED:
            break
        count += 1
    return count


def find_minimap_areas(frame):
    """
    Every white-outlined map area in FRAME (a BGR or BGRA screenshot), as
    ((left, top), (right, bottom)) with right/bottom exclusive, nearest the
    top-left corner first. Empty when none is visible (minimap collapsed or
    hidden, a window or the mouse pointer over its outline, the map loading).
    """
    if frame is None or frame.ndim != 3 or frame.shape[0] < MIN_HEIGHT or frame.shape[1] < MIN_WIDTH:
        return []
    white = frame[:, :, :3].min(axis=2) >= WHITE_MIN
    bands = _bands(white)
    tops = [band for band in bands if band[1] - band[0] < MAX_TOP_ROWS]
    areas = []
    for top in tops:
        best = None
        for bottom in bands:
            if bottom[0] - top[1] - 1 < MIN_HEIGHT:
                continue
            if bottom[1] - bottom[0] >= MAX_BOTTOM_ROWS:
                continue
            if abs(bottom[2] - top[2]) > EDGE_SLACK or abs(bottom[3] - top[3]) > EDGE_SLACK:
                continue
            if best is None or bottom[0] < best[0]:
                best = bottom
        if best is None:
            continue
        left, right = max(top[2], best[2]), min(top[3], best[3])
        y1, y2 = top[1] + 1, best[0]
        inside = white[y1:y2, left:right]
        shares = inside.mean(axis=0)
        left_side, right_side = _side(shares), _side(shares[::-1])
        if not (1 <= left_side <= MAX_SIDE and 1 <= right_side <= MAX_SIDE):
            continue
        x1, x2 = left + left_side, right - right_side
        if x2 - x1 < MIN_WIDTH - 2 * MAX_SIDE:
            continue
        if white[y1:y2, x1:x2].mean() > MAX_INSIDE_WHITE:
            continue
        areas.append(((x1, y1), (x2, y2)))
    areas.sort(key=lambda area: area[0][0] + area[0][1])
    return areas


def find_minimap_area(frame, accept=None, strict=False):
    """The map area to use, or None. ACCEPT(top_left, bottom_right), when given,
    picks among several (the one showing the player's marker). When none is
    picked: None if STRICT, else the one nearest the top-left corner."""
    areas = find_minimap_areas(frame)
    if not areas:
        return None
    if accept is not None:
        for area in areas:
            try:
                if accept(*area):
                    return area
            except Exception:
                pass
        if strict:
            return None
    return areas[0]
