"""
Solves the "Click all X's (N total)" quiz that follows pop-up-3
("Question in 11 seconds / Answer within 10 seconds").

The quiz is a see-through blue panel with a white title, a row of letter tiles
and a CONFIRM button. Tiles come in different art styles (green square tiles
with the letter painted on, snow-capped red/green letters, ...), so nothing
here depends on one style:

1. Title: the white text is matched against "Click all" rendered in Arial at
   several scales, then each candidate "<c>'s" (A-Z, 0-9) and "(<n> total)" is
   scored at that spot to read the target character and the count.
2. Tiles: bright, non-blue blobs in the band under the title (the translucent
   panel turns everything behind it dark blue, so only the tiles survive).
3. Letters: each tile's letter shape (the coloured letter itself, or for a solid
   tile the colour cluster that looks most like a letter) is compared with
   bold fonts' glyphs. Tiles that look alike are grouped, so one clearly read
   X lifts its look-alikes; then the N best matches for the target are chosen.

A second kind of question names a picture instead of a letter: "Click a
popsicle" over a few item pictures (pop-up-quiz-3.mp4). Clicking the item turns
it green and that is the answer - no CONFIRM. Items are cut out by their dark
pixel-art outline and compared with the pictures in assets/quiz_items/<name>.png
(the file name is the item name in the title). A new item is added with
    python -m src.detection.letter_quiz --add-item NAME SCREENSHOT --at X Y
(X, Y: any point on the item in the screenshot).

A third kind shows no name at all: "Click the one that is different" over a row
of pictures (pop-up-3-quiz-4.png: two red 3s and a cat face), then CONFIRM. The
pictures are cut out the same way and compared with each other; the one that
looks least like the rest is the answer (solve_odd). Nothing has to be known
about the pictures beforehand. When the pixels don't settle it (the "same"
pictures aren't pixel-for-pixel copies), the CLIP image model in item_clip.py
judges which picture is unlike the others.

A fourth kind is a yes/no question (pop-up-5-6-quiz.png): "Is there a spear
here?" over a few pictures, or "Is this a wand?" over one, on a light grey band
with an O (yes) and an X (no) under it, then CONFIRM. The item's name is read
off the title letter by letter (helped by YESNO_WORDS), the pictures are cut out
of the band, and each is judged: a picture known from assets/quiz_items settles
it by its name, anything else is put to the CLIP image model (solve_yesno).

Anything else - a kind of question not seen before - goes to the AI fallback
(quiz_ai.py): once the panel's "Remain : N Sec" footer shows and nothing here
has read the question, the screenshot is sent to a Claude vision model, whose
answer is clicked if the solvers here still have none a second later.

Run on a screenshot to check it:  python -m src.detection.letter_quiz IMAGE
"""

import os

import cv2
import numpy as np

CHARACTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
TITLE_FONT = 'arial.ttf'
TITLE_FONT_SIZE = 20                # the game draws the title at ~this size at scale 1
GLYPH_FONTS = ('ariblk.ttf', 'arialbd.ttf', 'verdanab.ttf', 'impact.ttf', 'comicbd.ttf')
SCALES = tuple(np.round(np.arange(0.5, 1.61, 0.05), 2))
TITLE_THRESHOLD = 0.75              # "Click all" match (0.92-0.95 on the samples)
TEXT_THRESHOLD = 0.7                # "<c>'s" match
TILE_PITCH = 110                    # px between tile centres at scale 1
BAND = (18, 160)                    # tiles lie this far below the title top (scale 1)
BUTTON_OFFSET = (18, 173)           # CONFIRM centre: right of the last tile's centre, below title top
GLYPH_SIZE = 32
SAME_LOOK = 0.8                     # tiles this alike (0-1) count as the same letter
ITEM_DIR = 'assets/quiz_items'      # <item name>.png, cut out (alpha) at title scale 1
ITEM_THRESHOLD = 0.6                # item likeness (0-1) needed to click it
# Items the quiz has asked for that may have no picture yet. For one of these the
# answer is found by elimination (the item that looks like none of the known
# pictures), and its picture is saved so later questions are matched directly.
ITEM_NAMES = ('bat',)               # 20261115 04:2x run: "Click a bat"
PANEL_HALF = 150                    # quiz panel half-width at scale 1 (items lie inside)
ITEM_MIN_HEIGHT = 45                # item pictures are at least this tall (scale 1)
OUTLINE_VALUE = 70                  # HSV value below this is an item's dark outline
ODD_TITLE = 'Click the one that is different'
SAME_PICTURE = 0.75                 # pictures this alike (0-1) are copies of each other
ODD_MARGIN = 0.15                   # the odd one is at least this much less alike
# Odd-one-out by slots (pop-up-7: two straw hats and a grey "9" tile - a tile
# has no dark outline, so the picture finders miss it). The panel's slots sit
# TILE_PITCH apart around the title's centre, the row's middle ROW_CENTRE below
# the title top; each slot is cut out as it is and compared with the others.
ROW_CENTRE = 83
SLOT_HALF = 44                      # a slot cut is (2 * SLOT_HALF) px square at scale 1
SLOT_REACH = 6                      # px a slot may sit off its place
SLOT_SCALES = (0.9, 1.0, 1.1)       # copies drawn a little bigger or smaller
SLOT_COUNTS = (3, 4)                # pictures in a row
# Yes/no questions: "Is there a <item> here?" / "Is this a <item>?"
YESNO_LEADS = (('there', 'Is there a ', ' here?'), ('there', 'Is there an ', ' here?'),
               ('this', 'Is this a ', '?'), ('this', 'Is this an ', '?'))
# Words the title is likely to ask for. Any other word is still read, letter by
# letter; a word from this list wins when the letters are too blurry to tell
# ("spear" read as "apear").
YESNO_WORDS = ('spear', 'wand', 'staff', 'sword', 'axe', 'bow', 'crossbow', 'claw', 'dagger',
               'mace', 'hammer', 'polearm', 'gun', 'knuckle', 'arrow', 'shield', 'hat',
               'helmet', 'cape', 'glove', 'shoe', 'boot', 'ring', 'earring', 'pendant',
               'belt', 'potion', 'scroll', 'candy', 'wing', 'wings', 'button', 'bat',
               'popsicle', 'witch hat', 'mushroom', 'snail', 'slime', 'pig', 'star', 'heart',
               'apple', 'cake', 'fish', 'cat', 'dog', 'bird', 'key', 'book', 'coin', 'fan',
               'umbrella', 'flower', 'leaf', 'tree', 'ghost', 'pumpkin', 'number', 'letter',
               'monster', 'weapon', 'animal', 'fruit', 'food')
YESNO_WORD_BONUS = 0.03             # a listed word beats a made-up one this much worse
YESNO_MAX_LETTERS = 14
YESNO_YES = 0.5                     # image model: this sure a picture is the item -> yes
# Where things are, from the grey band's (left, bottom) in band widths
# (pop-up-5-6-quiz.png; the O and X there are drawn by hand, so their real look
# is not known and they are found by place, or by their red colour if they are red).
YESNO_O = (0.245, 0.105)
YESNO_X = (0.764, 0.105)
YESNO_CONFIRM = (0.893, 0.30)
YESNO_MARK_HALF = 0.06              # half the O / X size


def _font(name, size):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return None


# ---------------------------------------------------------------- title ----

def text_mask(image):
    """Bright, colourless strokes brighter than their surroundings (the white
    title), as a 0-255 image. Works on blurry, downscaled screenshots too."""
    hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2].astype(np.float32)
    background = cv2.morphologyEx(value, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    strokes = np.clip((value - background) * 3, 0, 255)
    strokes[hsv[:, :, 1] > 90] = 0
    return cv2.GaussianBlur(strokes.astype(np.uint8), (3, 3), 0)


class _TitleRenderer:
    def __init__(self):
        self.font = _font(TITLE_FONT, TITLE_FONT_SIZE)
        if self.font is None:
            raise RuntimeError(f'Font {TITLE_FONT} not found.')
        _, self.top, _, self.bottom = self.font.getbbox("Click all (0)'gy")
        self._cache = {}

    def offset(self, text):
        return self.font.getlength(text)

    def render(self, text, scale):
        key = (text, scale)
        if key not in self._cache:
            from PIL import Image, ImageDraw
            width = int(self.font.getlength(text)) + 4
            image = Image.new('L', (width, self.bottom - self.top + 4), 0)
            ImageDraw.Draw(image).text((2, 2 - self.top), text, fill=255, font=self.font)
            array = np.array(image)
            if scale != 1:
                array = cv2.resize(array, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
            self._cache[key] = cv2.GaussianBlur(array, (3, 3), 0)
        return self._cache[key]


class Title:
    def __init__(self, x, y, scale, score, char, char_score, count, count_score, width,
                 item=None, odd=False, ask=None, about=None):
        self.x, self.y, self.scale, self.score = x, y, scale, score
        self.ask = ask              # yes/no question: 'there' (is there a ...) or 'this'
        self.about = about          # ... and the item it asks about
        self.item = item            # item quiz: the item's name ('?' if not known)
        self.odd = odd              # "Click the one that is different"
        self.char, self.char_score = char, char_score
        self.count, self.count_score = count, count_score
        self.width = width

    @property
    def centre_x(self):
        return self.x + self.width / 2

    @property
    def target(self):
        if self.ask:
            return f'is {self.ask} a {self.about}?'
        return 'the different one' if self.odd else self.item or self.char

    def __repr__(self):
        return (f"Title({self.target!r} x{self.count}, at ({self.x}, {self.y}) scale {self.scale}, "
                f"scores {self.score:.2f}/{self.char_score:.2f}/{self.count_score:.2f})")


def read_title(frame, renderer, scales=SCALES, items=()):
    """The quiz title in FRAME as a Title, or None if it isn't on screen.
    ITEMS: the item names known for "Click a <item>" questions."""
    mask = text_mask(frame)
    best = (0.0, None, None)
    question = (0.0, None, None)
    for scale in scales:
        # "Click a" starts both kinds of title ("Click all X's", "Click a popsicle");
        # "Is th" starts the yes/no ones ("Is there a ... here?", "Is this a ...?").
        for opening in ('Click a', 'Is th'):
            template = renderer.render(opening, scale)
            if template.shape[0] > mask.shape[0] or template.shape[1] > mask.shape[1]:
                continue
            _, score, _, location = cv2.minMaxLoc(
                cv2.matchTemplate(mask, template, cv2.TM_CCOEFF_NORMED))
            if opening == 'Click a' and score > best[0]:
                best = (score, scale, location)
            elif opening == 'Is th' and score > question[0]:
                question = (score, scale, location)
    read_title.last_score = float(max(best[0], question[0]))    # (for the miss report)
    read_title.last_item = None
    if question[0] >= TITLE_THRESHOLD and question[0] > best[0]:
        title = _read_question(mask, renderer, *question)
        if title is not None:
            return title
    score, scale, location = best
    if score < TITLE_THRESHOLD:
        return None
    x, y = location

    def match(text, offset=0.0, slack=4):
        template = renderer.render(text, scale)
        x0, y0 = int(x + offset * scale) - slack, y - slack
        if x0 < 0 or y0 < 0:
            return -1.0
        window = mask[y0:y0 + template.shape[0] + 2 * slack,
                      x0:x0 + template.shape[1] + 2 * slack]
        if window.shape[0] < template.shape[0] or window.shape[1] < template.shape[1]:
            return -1.0
        return float(cv2.minMaxLoc(cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED))[1])

    def read_count(offset):
        counts = sorted(((match(f'({n} total)', offset), n) for n in range(1, 10)),
                        reverse=True)
        count_score, count = counts[0]
        return (count if count_score >= TEXT_THRESHOLD else None), count_score

    # Odd-one-out question: "Click the one that is different". ("Click a" also
    # matches the start of "Click the" well enough to get here: 0.84.) Its tail
    # scores 0.81 on pop-up-3-quiz-4.png and under 0.25 on the other kinds.
    odd_score = match('Click the one')
    if odd_score >= TEXT_THRESHOLD:
        tail_score = match('that is different', renderer.offset('Click the one '), slack=8)
        if tail_score >= TEXT_THRESHOLD:
            return Title(x, y, scale, score, None, odd_score, 1, tail_score,
                         renderer.offset(ODD_TITLE) * scale, odd=True)

    # Letter question: "Click all <c>'s (<n> total)".
    all_score = match('Click all')
    chars = sorted(((match(f"{c}'s", renderer.offset('Click all ')), c) for c in CHARACTERS),
                   reverse=True)
    char_score, char = chars[0]
    letter_score = match(f"Click all {char}'s") if all_score >= TITLE_THRESHOLD else -1.0

    # Item question: "Click a/an <item>" (or "Click all <item>s (<n> total)").
    item_score, item, item_text = -1.0, None, ''
    for name in items:
        for text in (f'Click a {name}', f'Click an {name}', f'Click all {name}s'):
            score_ = match(text)
            if score_ > item_score:
                item_score, item, item_text = score_, name, text

    if item is not None and item_score >= TEXT_THRESHOLD and item_score > letter_score:
        count, count_score = 1, item_score
        if item_text.startswith('Click all'):
            count, count_score = read_count(renderer.offset(item_text + ' '))
        width = renderer.offset(item_text) * scale
        read_title.last_item = item
        return Title(x, y, scale, score, None, item_score, count, count_score, width, item=item)
    if all_score >= TITLE_THRESHOLD and char_score >= TEXT_THRESHOLD:
        count, count_score = read_count(renderer.offset(f"Click all {char}'s "))
        # (count None: can't read it, fall back to choosing by looks alone)
        width = renderer.offset(f"Click all {char}'s ({count or 0} total)") * scale
        return Title(x, y, scale, score, char, char_score, count, count_score, width)
    if all_score < TITLE_THRESHOLD and match('Click a ') >= TITLE_THRESHOLD:
        # "Click a <something>" with no picture for that something.
        read_title.last_item = '?'
    return None


read_title.last_item = None     # '?' when the last frame asked for an unknown item
read_title.last_score = 0.0


def _read_question(mask, renderer, score, scale, location):
    """The yes/no title starting at LOCATION ("Is there a spear here?", "Is this
    a wand?") as a Title, or None. The item's name is read letter by letter."""
    x, y = location

    def match(text, offset=0.0, slack=3):
        template = renderer.render(text, scale)
        x0, y0 = int(x + offset * scale) - slack, y - slack
        if x0 < 0 or y0 < 0:
            return -1.0
        window = mask[y0:y0 + template.shape[0] + 2 * slack,
                      x0:x0 + template.shape[1] + 2 * slack]
        if window.shape[0] < template.shape[0] or window.shape[1] < template.shape[1]:
            return -1.0
        return float(cv2.minMaxLoc(cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED))[1])

    def title(ask, lead, word, tail, word_score):
        return Title(x, y, scale, score, None, word_score, 1, word_score,
                     renderer.offset(lead + word + tail) * scale, ask=ask, about=word)

    # The same question as on the last frame: no need to spell it out again.
    last = _read_question.last
    if last is not None:
        again = match(''.join(last[1:]))
        if again >= 0.8:
            return title(last[0], last[1], last[2], last[3], again)

    best = None                 # (score with bonus, score, ask, lead, word, tail)
    for ask, lead, tail in YESNO_LEADS:
        if match(lead.rstrip()) < TEXT_THRESHOLD:
            continue
        base = renderer.offset(lead)
        words = set(YESNO_WORDS)
        # Spell the word: keep the few best beginnings, add a letter at a time
        # (each new letter is scored in place together with the two before it).
        beams = ['']
        for _ in range(YESNO_MAX_LETTERS):
            grown = []
            for word in beams:
                for c in 'abcdefghijklmnopqrstuvwxyz ':
                    if c == ' ' and (not word or word.endswith(' ')):
                        continue
                    k = max(0, len(word) - 2)
                    grown.append((match((word + c)[k:], base + renderer.offset(word[:k])),
                                  word + c))
            grown.sort(reverse=True)
            beams = [word for _, word in grown[:6]]
            words.update(word for word in beams if not word.endswith(' '))
        for word in words:
            word_score = match(word + tail, base)
            bonus = YESNO_WORD_BONUS if word in YESNO_WORDS else 0.0
            if best is None or word_score + bonus > best[0]:
                best = (word_score + bonus, word_score, ask, lead, word, tail)
    if best is None or best[1] < TEXT_THRESHOLD:
        return None
    _, word_score, ask, lead, word, tail = best
    _read_question.last = (ask, lead, word, tail)
    return title(ask, lead, word, tail, word_score)


_read_question.last = None      # (ask, lead, word, tail) read on an earlier frame


# ---------------------------------------------------------------- tiles ----

def colour_mask(image):
    """Tile pixels: saturated and not blue. (Blue is the panel's tint, and the
    bright sky beside the panel would join up with the tiles.)"""
    hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    return ((hue < 85) | (hue > 140)) & (sat > 120) & (val > 90)


def find_tiles(frame, title):
    """Boxes (x0, y0, x1, y1) of the letter tiles under TITLE, left to right."""
    s = title.scale
    height, width = frame.shape[:2]
    y0 = max(0, int(title.y + BAND[0] * s))
    y1 = min(height, int(title.y + BAND[1] * s))
    band = colour_mask(frame[y0:y1]).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(band)
    pitch = TILE_PITCH * s
    blobs = []
    for i in range(1, count):
        bx, by, bw, bh, area = stats[i]
        if area < (8 * s) ** 2:
            continue
        if by == 0 or by + bh >= band.shape[0]:
            continue        # runs out of the band: scenery, not a tile
        if bw > 1.2 * pitch or bh > 1.2 * pitch:
            continue
        blobs.append([bx, by + y0, bx + bw, by + bh + y0, area])

    # Pieces of one letter (snow, a split stroke) share a slot: merge by x.
    blobs.sort(key=lambda b: (b[0] + b[2]) / 2)
    tiles = []
    for blob in blobs:
        centre = (blob[0] + blob[2]) / 2
        if tiles and abs(centre - (tiles[-1][0] + tiles[-1][2]) / 2) < 0.45 * pitch:
            last = tiles[-1]
            tiles[-1] = [min(last[0], blob[0]), min(last[1], blob[1]),
                         max(last[2], blob[2]), max(last[3], blob[3]), last[4] + blob[4]]
        else:
            tiles.append(list(blob))
    tiles = [t for t in tiles
             if t[4] >= (22 * s) ** 2 and t[3] - t[1] >= 30 * s]

    # Keep the run of tiles around the title's centre (spaced about one pitch
    # apart); anything beyond a gap is scenery next to the panel.
    if not tiles:
        return []
    centres = [(t[0] + t[2]) / 2 for t in tiles]
    middle = min(range(len(tiles)), key=lambda i: abs(centres[i] - title.centre_x))
    if abs(centres[middle] - title.centre_x) > 0.75 * pitch:
        return []
    lo = hi = middle
    while lo > 0 and centres[lo] - centres[lo - 1] < 1.5 * pitch:
        lo -= 1
    while hi < len(tiles) - 1 and centres[hi + 1] - centres[hi] < 1.5 * pitch:
        hi += 1
    return [tuple(int(v) for v in t[:4]) for t in tiles[lo:hi + 1]]


# -------------------------------------------------------------- letters ----

def _normalise(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    crop = cv2.resize(crop, (GLYPH_SIZE, GLYPH_SIZE), interpolation=cv2.INTER_AREA)
    crop = cv2.GaussianBlur(crop, (5, 5), 0)
    crop -= crop.mean()
    norm = np.linalg.norm(crop)
    return crop / norm if norm > 1e-6 else None


def _clean(mask):
    """Drops specks: keeps components at least a fifth the size of the biggest."""
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    if count <= 1:
        return mask
    biggest = stats[1:, 4].max()
    return np.isin(labels, [i for i in range(1, count) if stats[i, 4] >= 0.2 * biggest])


class Glyphs:
    def __init__(self, fonts=GLYPH_FONTS, characters=CHARACTERS):
        from PIL import Image, ImageDraw
        self.templates = {c: [] for c in characters}
        for name in fonts:
            font = _font(name, 64)
            if font is None:
                continue
            for c in characters:
                image = Image.new('L', (120, 120), 0)
                ImageDraw.Draw(image).text((20, 10), c, fill=255, font=font)
                glyph = _normalise(np.array(image) > 127)
                if glyph is not None:
                    self.templates[c].append(glyph)
        if not any(self.templates.values()):
            raise RuntimeError('No glyph fonts found.')

    def scores(self, shape):
        """{character: 0-1 likeness} for a normalised shape."""
        if shape is None:
            return {c: 0.0 for c in self.templates}
        return {c: max((float((shape * t).sum()) for t in ts), default=0.0)
                for c, ts in self.templates.items()}

    def best(self, shape):
        return max(self.scores(shape).values())


def letter_shape(tile, glyphs):
    """The letter in a tile image, normalised for comparing (or None)."""
    coloured = colour_mask(tile)
    if coloured.mean() < 0.75:
        return _normalise(_clean(coloured))
    # A solid tile with the letter painted on it: split its colours and take
    # the cluster that looks most like a character.
    lab = cv2.cvtColor(tile[:, :, :3], cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    _, labels, _ = cv2.kmeans(lab, 3, None, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                                             20, 1.0), 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(tile.shape[:2])
    shapes = []
    for k in range(3):
        part = _clean(labels == k)
        if 0.08 < part.mean() < 0.6:
            shapes.append(_normalise(part))
    shapes = [s for s in shapes if s is not None]
    if not shapes:
        return _normalise(_clean(coloured))
    return max(shapes, key=glyphs.best)


# ---------------------------------------------------------------- items ----

def outline_fill(image):
    """Pixels walled in by a dark outline (pixel-art items), as a bool mask.
    The dimmed scenery behind the panel has few closed dark outlines."""
    value = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)[:, :, 2]
    dark = cv2.dilate((value < OUTLINE_VALUE).astype(np.uint8), np.ones((2, 2), np.uint8))
    # Open areas reaching the border are outside; the rest (and the outline) is inside.
    _, labels = cv2.connectedComponents((1 - dark).astype(np.uint8), connectivity=4)
    edge = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    inside = ~np.isin(labels, edge[edge > 0])
    # Drop bare outline strokes of things that aren't closed.
    return cv2.morphologyEx(inside.astype(np.uint8), cv2.MORPH_OPEN,
                            np.ones((5, 5), np.uint8)).astype(bool)


class Items:
    """Item pictures for "Click a <item>" questions: {name: (bgr, mask)} at scale 1."""

    def __init__(self, directory=ITEM_DIR):
        self.pictures = {}
        if directory and os.path.isdir(directory):
            for file in sorted(os.listdir(directory)):
                name, ext = os.path.splitext(file)
                if ext.lower() != '.png':
                    continue
                image = cv2.imread(os.path.join(directory, file), cv2.IMREAD_UNCHANGED)
                if image is None:
                    continue
                if image.ndim == 3 and image.shape[2] == 4:
                    mask = image[:, :, 3] > 127
                else:
                    mask = outline_fill(image)
                self.pictures[name.replace('_', ' ')] = (image[:, :, :3], mask)

    @property
    def names(self):
        """Every item name a title can ask for: pictured ones and ITEM_NAMES."""
        return tuple(self.pictures) + tuple(n for n in ITEM_NAMES if n not in self.pictures)

    def add(self, name, picture):
        """Uses PICTURE (BGRA, scale 1) for NAME from now on."""
        self.pictures[name] = (picture[:, :, :3], picture[:, :, 3] > 127)

    def likeness(self, name, image, box, mask, scale=1.0):
        """0-1: how much the cut-out (IMAGE where MASK, inside BOX) looks like item
        NAME: the better of its outline shape (a recoloured item still counts) and
        the picture itself matched in place (still works while the mouse hovers
        the item, which draws a box over it and hides part of the outline)."""
        return max(self._shape(name, image, mask), self._picture(name, image, box, scale))

    def _picture(self, name, image, box, scale):
        picture, picture_mask = self.pictures[name]
        x0, y0, x1, y1 = box
        pad = int(0.25 * max(x1 - x0, y1 - y0))
        region = image[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad, :3]
        best = 0.0
        for factor in (0.95, 1.0, 1.05):
            s = scale * factor
            template = cv2.resize(picture, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            template_mask = cv2.resize(picture_mask.astype(np.uint8) * 255, None, fx=s, fy=s,
                                       interpolation=cv2.INTER_NEAREST)
            if template.shape[0] > region.shape[0] or template.shape[1] > region.shape[1]:
                continue
            result = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED,
                                       mask=cv2.merge([template_mask] * 3))
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            best = max(best, float(result.max()))
        return min(1.0, best)

    def _shape(self, name, image, mask):
        picture, picture_mask = self.pictures[name]
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            return 0.0
        crop = image[ys.min():ys.max() + 1, xs.min():xs.max() + 1, :3]
        crop_mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = picture_mask.shape
        ratio = (crop_mask.shape[1] / crop_mask.shape[0]) / (w / h)
        if not 0.7 < ratio < 1.4:
            return 0.0
        crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
        crop_mask = cv2.resize(crop_mask.astype(np.uint8), (w, h),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
        both = crop_mask & picture_mask
        shape = both.sum() / max(1, (crop_mask | picture_mask).sum())
        if both.sum() < 20:
            return float(shape)
        a = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)[both].astype(np.float32)
        b = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)[both].astype(np.float32)
        a, b = a - a.mean(), b - b.mean()
        shading = float((a * b).sum() / max(1e-6, np.linalg.norm(a) * np.linalg.norm(b)))
        return float(0.7 * shape + 0.3 * max(0.0, shading))


def find_items(frame, title):
    """Cut-outs [(box, mask)] of the item pictures under TITLE, left to right."""
    s = title.scale
    height, width = frame.shape[:2]
    y0 = max(0, int(title.y + BAND[0] * s))
    y1 = min(height, int(title.y + BAND[1] * s))
    x0 = max(0, int(title.centre_x - 2.5 * TILE_PITCH * s))
    x1 = min(width, int(title.centre_x + 2.5 * TILE_PITCH * s))
    filled = outline_fill(frame[y0:y1, x0:x1])
    count, labels, stats, _ = cv2.connectedComponentsWithStats(filled.astype(np.uint8))
    found = []
    for i in range(1, count):
        bx, by, bw, bh, area = stats[i]
        if area < (25 * s) ** 2 or max(bw, bh) < 40 * s or max(bw, bh) > 1.3 * TILE_PITCH * s:
            continue
        if by == 0 or by + bh >= filled.shape[0]:
            continue        # runs out of the band: scenery
        mask = np.zeros(frame.shape[:2], bool)
        mask[y0:y1, x0:x1] = labels == i
        found.append(((int(x0 + bx), int(y0 + by), int(x0 + bx + bw), int(y0 + by + bh)), mask))
    found.sort(key=lambda item: item[0][0])
    return found


def solve_items(frame, title, items):
    """An Answer for an item question (no CONFIRM button), or None."""
    if items is None:
        return None
    found = find_items(frame, title)
    if not found:
        return None
    if title.item not in items.pictures:
        return _eliminate(frame, title, items, found)
    likeness = [items.likeness(title.item, frame, box, mask, title.scale) for box, mask in found]
    order = sorted(range(len(found)), key=lambda i: likeness[i], reverse=True)
    chosen = sorted(i for i in order[:title.count or 1] if likeness[i] >= ITEM_THRESHOLD)
    scores = [{title.item: v} for v in likeness]
    looks = [f'{title.item if v >= ITEM_THRESHOLD else "other"} {v:.2f}' for v in likeness]
    return Answer(title, [box for box, _ in found], chosen, scores, looks, None)


def _eliminate(frame, title, items, found):
    """Answer for an item with no picture: of the pictures inside the panel, the
    one that looks least like any known item. Answer.learn is set to the (box,
    mask) to save as the item's picture when only one picture was unknown."""
    s = title.scale
    inside = [(box, mask) for box, mask in found
              if abs((box[0] + box[2]) / 2 - title.centre_x) <= PANEL_HALF * s
              and box[3] - box[1] >= ITEM_MIN_HEIGHT * s]
    if not inside:
        return None
    looks, best = [], []
    for box, mask in inside:
        known = [(items.likeness(n, frame, box, mask, s), n) for n in items.pictures]
        score, name = max(known, default=(0.0, 'other'))
        best.append(score)
        looks.append(f'{name} {score:.2f}' if score >= ITEM_THRESHOLD else f'new {score:.2f}')
    unknown = [i for i, v in enumerate(best) if v < ITEM_THRESHOLD]
    if not unknown:
        return Answer(title, [b for b, _ in inside], [], [{} for _ in inside], looks, None)
    pick = min(unknown, key=lambda i: best[i])
    if len(unknown) > 1:
        # Several new pictures: ask the image/text model which looks like the item.
        from src.detection import item_clip
        cuts = []
        for i in unknown:
            (x0, y0, x1, y1), mask = inside[i]
            cuts.append((frame[y0:y1, x0:x1], mask[y0:y1, x0:x1]))
        ai = item_clip.rank(title.item, cuts)
        if ai is not None:
            for i, p in zip(unknown, ai):
                looks[i] = f'new (AI {title.item} {p:.2f})'
            pick = unknown[int(np.argmax(ai))]
    looks[pick] = f'{title.item}? {looks[pick]}'
    answer = Answer(title, [b for b, _ in inside], [pick], [{} for _ in inside], looks, None)
    # Sure only when every other picture is a known item.
    answer.learn = inside[pick] if len(unknown) == 1 and len(inside) > 1 else None
    answer.guessed = True
    return answer


def alike(a, b, reach=3):
    """0-1: how much two cut-outs (bgr, mask) are the same picture: how well
    their outlines overlap times how close the colours are where they do. The
    cut-outs are laid centre on centre, and moved up to REACH px for the best fit."""
    height = max(a[0].shape[0], b[0].shape[0]) + 2 * reach
    width = max(a[0].shape[1], b[0].shape[1]) + 2 * reach

    def centred(picture, mask):
        h, w = mask.shape
        y, x = (height - h) // 2, (width - w) // 2
        image = np.zeros((height, width, 3), np.float32)
        solid = np.zeros((height, width), bool)
        image[y:y + h, x:x + w] = picture[:, :, :3]
        solid[y:y + h, x:x + w] = mask
        return image, solid

    image_a, mask_a = centred(*a)
    image_b, mask_b = centred(*b)
    best = 0.0
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            moved = np.roll(mask_b, (dy, dx), (0, 1))
            both = mask_a & moved
            overlap = both.sum() / max(1, (mask_a | moved).sum())
            if overlap <= best or not both.any():
                continue        # (colour can only lower it)
            colour = np.abs(image_a - np.roll(image_b, (dy, dx), (0, 1)))[both].mean() / 255.0
            best = max(best, float(overlap * (1.0 - colour)))
    return best


def odd_one(likeness):
    """(index, sure) of the picture least like the others, from the pictures'
    likeness matrix. SURE: the others are copies of each other and it is not."""
    n = len(likeness)
    means = [np.mean([likeness[i][j] for j in range(n) if j != i]) for i in range(n)]
    pick = int(np.argmin(means))
    rest = [i for i in range(n) if i != pick]
    same = min((likeness[i][j] for i in rest for j in rest if i < j), default=0.0)
    nearest = max(likeness[pick][j] for j in rest)
    return pick, bool(same >= SAME_PICTURE and nearest <= same - ODD_MARGIN)


def find_row(frame, title):
    """find_items, kept to the row of pictures under TITLE: full-height ones on
    one line, spaced about a tile apart around the title's centre."""
    s = title.scale
    found = [(box, mask) for box, mask in find_items(frame, title)
             if box[3] - box[1] >= ITEM_MIN_HEIGHT * s]
    if not found:
        return []
    line = float(np.median([(box[1] + box[3]) / 2 for box, _ in found]))
    found = [(box, mask) for box, mask in found
             if abs((box[1] + box[3]) / 2 - line) <= 20 * s]
    centres = [(box[0] + box[2]) / 2 for box, _ in found]
    pitch = TILE_PITCH * s
    middle = min(range(len(found)), key=lambda i: abs(centres[i] - title.centre_x))
    if abs(centres[middle] - title.centre_x) > 0.75 * pitch:
        return []
    lo = hi = middle
    while lo > 0 and centres[lo] - centres[lo - 1] < 1.5 * pitch:
        lo -= 1
    while hi < len(found) - 1 and centres[hi + 1] - centres[hi] < 1.5 * pitch:
        hi += 1
    return found[lo:hi + 1]


def slot_cuts(frame, title, count):
    """[(box, picture)] of COUNT slots in the row under TITLE, left to right, or
    [] when a slot runs off the frame."""
    s = title.scale
    half = int(round(SLOT_HALF * s))
    cy = int(round(title.y + ROW_CENTRE * s))
    cuts = []
    for k in range(count):
        cx = int(round(title.centre_x + (k - (count - 1) / 2) * TILE_PITCH * s))
        x0, y0, x1, y1 = cx - half, cy - half, cx + half, cy + half
        if x0 < 0 or y0 < 0 or x1 > frame.shape[1] or y1 > frame.shape[0]:
            return []
        cuts.append(((x0, y0, x1, y1), frame[y0:y1, x0:x1, :3].astype(np.float32)))
    return cuts


def slot_alike(a, b, reach=SLOT_REACH):
    """0-1: how much slot B's middle shows up in slot A, moved up to REACH px,
    mirrored or a little resized (a copy drawn differently is still a copy)."""
    best = -1.0
    for scale in SLOT_SCALES:
        core = b[reach:-reach, reach:-reach]
        if scale != 1.0:
            core = cv2.resize(core, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        if core.shape[0] > a.shape[0] or core.shape[1] > a.shape[1]:
            continue
        for picture in (core, core[:, ::-1]):
            result = cv2.matchTemplate(a, np.ascontiguousarray(picture), cv2.TM_CCOEFF_NORMED)
            best = max(best, float(result.max()))
    return max(0.0, best)


def solve_odd_slots(frame, title):
    """(boxes, likeness, pick, sure) for the odd-one quiz from the panel's slots,
    or None. Tried with 3 and 4 slots; a sure answer wins."""
    best = None
    for count in SLOT_COUNTS:
        cuts = slot_cuts(frame, title, count)
        if len(cuts) < 3:
            continue
        likeness = np.eye(count)
        for i in range(count):
            for j in range(i):
                likeness[i, j] = likeness[j, i] = min(slot_alike(cuts[i][1], cuts[j][1]),
                                                      slot_alike(cuts[j][1], cuts[i][1]))
        pick, sure = odd_one(likeness)
        if sure:
            return [box for box, _ in cuts], likeness, pick, True
        if best is None:
            best = ([box for box, _ in cuts], likeness, pick, False)
    return best


def solve_odd(frame, title):
    """An Answer for "Click the one that is different" (then CONFIRM), or None.
    Reasoned out, no stored pictures: the odd one is the picture the others are
    not copies of. First the pictures found by their outline; when there aren't
    three or they don't settle it, the panel's slots as they are; only when that
    is not sure either, the image model (CLIP)."""
    found = find_row(frame, title)
    cuts = [(frame[y0:y1, x0:x1], mask[y0:y1, x0:x1]) for (x0, y0, x1, y1), mask in found]
    boxes = [box for box, _ in found]
    sure = False
    if len(cuts) >= 3:
        n = len(cuts)
        likeness = np.eye(n)
        for i in range(n):
            for j in range(i):
                likeness[i, j] = likeness[j, i] = alike(cuts[i], cuts[j])
        pick, sure = odd_one(likeness)
    how = ''
    if not sure:
        slots = solve_odd_slots(frame, title)
        if slots is not None and (slots[3] or len(cuts) < 3):
            boxes, likeness, pick, sure = slots
            how = 'slots '
            # For the image model, should the slots not settle it: whole slots.
            cuts = [(frame[y0:y1, x0:x1], np.ones((y1 - y0, x1 - x0), bool))
                    for x0, y0, x1, y1 in boxes]
    if not boxes or len(boxes) < 3:
        return None             # two pictures: no telling which is the odd one
    n = len(boxes)
    if not sure and cuts:
        # Not plain copies: ask the image model which picture is unlike the rest.
        from src.detection import item_clip
        ai = item_clip.alike(cuts)
        if ai is not None:
            likeness, how = ai, 'AI '
            pick = odd_one(ai)[0]
    looks = []
    for i in range(n):
        nearest = max(likeness[i][j] for j in range(n) if j != i)
        looks.append(f'{"different" if i == pick else "same"} ({how}{nearest:.2f} alike)')
    last = boxes[-1]
    button = ((last[0] + last[2]) / 2 + BUTTON_OFFSET[0] * title.scale,
              title.y + BUTTON_OFFSET[1] * title.scale)
    answer = Answer(title, boxes, [pick], [{} for _ in boxes], looks, button)
    answer.guessed = not sure
    return answer


# --------------------------------------------------------------- yes/no ----

def find_band(frame, title):
    """The light grey band the yes/no pictures lie on, under TITLE, as
    (x0, y0, x1, y1), or None."""
    s = title.scale
    height, width = frame.shape[:2]
    ax0, ax1 = max(0, int(title.centre_x - 260 * s)), min(width, int(title.centre_x + 260 * s))
    ay0, ay1 = max(0, int(title.y + 12 * s)), min(height, int(title.y + 230 * s))
    if ax1 - ax0 < 10 or ay1 - ay0 < 10:
        return None
    hue, sat, val = cv2.split(cv2.cvtColor(frame[ay0:ay1, ax0:ax1, :3], cv2.COLOR_BGR2HSV))
    grey = ((hue >= 80) & (hue <= 120) & (sat < 75) & (val >= 140) & (val <= 220))
    grey = cv2.morphologyEx(grey.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(grey)
    if count <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, 4]))
    x, y, w, h, area = stats[i]
    if w < 150 * s or h < 50 * s or area < 0.4 * w * h:
        return None
    if not x <= title.centre_x - ax0 <= x + w:
        return None
    return int(ax0 + x), int(ay0 + y), int(ax0 + x + w), int(ay0 + y + h)


def find_pictures(frame, band, scale):
    """Cut-outs [(box, mask)] of the pictures on the grey BAND, left to right:
    whatever isn't the band's pale blue-grey (colour, white, dark outlines)."""
    x0, y0, x1, y1 = band
    hue, sat, val = cv2.split(cv2.cvtColor(frame[y0:y1, x0:x1, :3], cv2.COLOR_BGR2HSV))
    thing = (((hue < 80) | (hue > 125)) & (sat > 60)) | (val > 222) | (val < 110) | (sat > 110)
    thing = cv2.morphologyEx(thing.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(thing)
    parts = sorted([int(v) for v in stats[i]] + [i] for i in range(1, count)
                   if stats[i][4] >= (4 * scale) ** 2)
    groups = []                 # pieces of one picture (a pair of wings) lie side by side
    for x, y, w, h, _, i in parts:
        if groups and x <= groups[-1][2] + 3 * scale:
            g = groups[-1]
            g[:4] = min(g[0], x), min(g[1], y), max(g[2], x + w), max(g[3], y + h)
            g[4].append(i)
        else:
            groups.append([x, y, x + w, y + h, [i]])
    found = []
    for gx0, gy0, gx1, gy1, ids in groups:
        if max(gx1 - gx0, gy1 - gy0) < 30 * scale:
            continue
        mask = np.zeros(frame.shape[:2], np.uint8)
        mask[y0:y1, x0:x1] = np.isin(labels, ids)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, 1, -1)
        found.append(((x0 + gx0, y0 + gy0, x0 + gx1, y0 + gy1), mask.astype(bool)))
    return found


def same_item(name, asked):
    """Whether a picture called NAME answers a question about ASKED: the same
    word, its plural, or part of it ("witch hat" is a "hat")."""
    def words(text):
        return [w[:-1] if len(w) > 3 and w.endswith('s') else w for w in text.lower().split()]
    a, b = words(name), words(asked)
    return bool(a and b) and (a == b or all(w in a for w in b) or all(w in b for w in a))


def find_marks(frame, band):
    """Boxes of the O (yes) and the X (no) under BAND: where they should be,
    moved onto a red mark when one is drawn about there."""
    x0, y0, x1, y1 = band
    width = x1 - x0
    half = YESNO_MARK_HALF * width
    zone_y0, zone_y1 = int(y1 + 0.02 * width), int(y1 + 0.2 * width)
    zone = frame[zone_y0:zone_y1, x0:x1, :3]
    blobs = []
    if zone.size:
        hue, sat, val = cv2.split(cv2.cvtColor(zone, cv2.COLOR_BGR2HSV))
        red = (((hue < 12) | (hue > 168)) & (sat > 140) & (val > 110)).astype(np.uint8)
        red = cv2.dilate(red, np.ones((3, 3), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(red)
        for i in range(1, count):
            bx, by, bw, bh, _ = stats[i]
            if 0.05 * width <= max(bw, bh) <= 0.3 * width:
                blobs.append((x0 + bx + bw / 2, zone_y0 + by + bh / 2))
    marks = []
    for fx, fy in (YESNO_O, YESNO_X):
        cx, cy = x0 + fx * width, y1 + fy * width
        near = [b for b in blobs if abs(b[0] - cx) <= 0.12 * width]
        if near:
            cx, cy = min(near, key=lambda b: abs(b[0] - cx))
        marks.append((int(cx - half), int(cy - half), int(cx + half), int(cy + half)))
    return marks


def _yesno_button(frame, band):
    """CONFIRM's centre under BAND: the pale (not yet lit) button when it is
    seen at the bottom right, else where it should be."""
    x0, y0, x1, y1 = band
    width = x1 - x0
    estimate = (x0 + YESNO_CONFIRM[0] * width, y1 + YESNO_CONFIRM[1] * width)
    zx0, zy0 = int(x0 + 0.6 * width), int(y1 + 0.18 * width)
    zone = frame[zy0:int(y1 + 0.42 * width), zx0:int(x1 + 0.08 * width), :3]
    if zone.size == 0:
        return estimate
    hue, sat, val = cv2.split(cv2.cvtColor(zone, cv2.COLOR_BGR2HSV))
    pale = ((sat < 60) & (val > 165)).astype(np.uint8)
    pale = cv2.morphologyEx(pale, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(pale)
    best = None
    for i in range(1, count):
        bx, by, bw, bh, area = stats[i]
        if 0.12 * width <= bw <= 0.32 * width and 2 <= bw / max(1, bh) <= 6 \
                and area >= 0.6 * bw * bh and (best is None or area > stats[best, 4]):
            best = i
    if best is None:
        return estimate
    bx, by, bw, bh, _ = stats[best]
    return zx0 + bx + bw / 2, zy0 + by + bh / 2


def solve_yesno(frame, title, items):
    """An Answer for "Is there a <item> here?" / "Is this a <item>?": the O (yes)
    or the X (no) to click, then CONFIRM. None when the band isn't found."""
    band = find_band(frame, title)
    if band is None:
        return None
    s = title.scale
    found = find_pictures(frame, band, s)
    if not found:
        return None
    asked = title.about
    looks, verdicts, unsure = [], [], False
    for box, mask in found:
        known = [(items.likeness(n, frame, box, mask, s), n)
                 for n in (items.pictures if items else ())]
        likeness, name = max(known, default=(0.0, None))
        if likeness >= ITEM_THRESHOLD:
            verdicts.append(same_item(name, asked))
            looks.append(f'{name} {likeness:.2f}')
            continue
        from src.detection import item_clip
        x0, y0, x1, y1 = box
        ai = item_clip.rank(asked, [(frame[y0:y1, x0:x1], mask[y0:y1, x0:x1])])
        if ai is None:
            verdicts.append(False)
            looks.append('unknown picture')
            unsure = True
            continue
        verdicts.append(ai[0] >= YESNO_YES)
        looks.append(f'AI {asked} {ai[0]:.2f}')
        unsure = unsure or 0.25 <= ai[0] < 0.75
    yes = any(verdicts)
    o_box, x_box = find_marks(frame, band)
    boxes = [box for box, _ in found] + [o_box, x_box]
    looks += ['O = yes', 'X = no']
    x0, y0, x1, y1 = band
    button = _yesno_button(frame, band)
    answer = Answer(title, boxes, [len(found) if yes else len(found) + 1],
                    [{} for _ in boxes], looks, button)
    answer.guessed = unsure
    return answer


def cut_item(frame, box, mask, scale):
    """The item at BOX/MASK in FRAME as a BGRA picture at title scale 1."""
    x0, y0, x1, y1 = box
    alpha = mask[y0:y1, x0:x1].astype(np.uint8) * 255
    picture = np.dstack([frame[y0:y1, x0:x1, :3], alpha])
    if scale != 1:
        picture = cv2.resize(picture, None, fx=1 / scale, fy=1 / scale,
                             interpolation=cv2.INTER_AREA if scale > 1 else cv2.INTER_CUBIC)
    return picture


def add_item(name, image_path, point, directory=ITEM_DIR):
    """Cuts the item at POINT out of a screenshot of an item question and saves
    it as DIRECTORY/<name>.png, scaled to title scale 1. Returns the path."""
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(image_path)
    renderer, mask = _TitleRenderer(), text_mask(frame)
    scale = max(SCALES, key=lambda s: cv2.minMaxLoc(cv2.matchTemplate(
        mask, renderer.render('Click a', s), cv2.TM_CCOEFF_NORMED))[1])
    x, y = point
    pad = int(TILE_PITCH * scale)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    crop = frame[y0:y + pad, x0:x + pad]
    count, labels = cv2.connectedComponents(outline_fill(crop).astype(np.uint8))
    label = labels[y - y0, x - x0]
    if label == 0:
        raise ValueError(f'No outlined item at ({x}, {y}).')
    ys, xs = np.nonzero(labels == label)
    crop = crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    alpha = (labels == label)[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8) * 255
    picture = np.dstack([crop, alpha])
    if scale != 1:
        picture = cv2.resize(picture, None, fx=1 / scale, fy=1 / scale,
                             interpolation=cv2.INTER_AREA if scale > 1 else cv2.INTER_CUBIC)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name.strip().lower().replace(' ', '_') + '.png')
    cv2.imwrite(path, picture)
    return path


class Answer:
    def __init__(self, title, tiles, chosen, scores, looks, button):
        self.title = title
        self.tiles = tiles          # boxes, left to right
        self.chosen = chosen        # indexes into tiles to click
        self.scores = scores        # per tile: {character: likeness}
        self.looks = looks          # per tile: best-guess character
        self.button = button        # CONFIRM centre (x, y); None: item question, no CONFIRM
        self.guessed = False        # item with no picture, chosen by elimination
        self.learn = None           # (box, mask) of that item when the elimination was sure

    def points(self):
        return [((x0 + x1) / 2, (y0 + y1) / 2)
                for i, (x0, y0, x1, y1) in enumerate(self.tiles) if i in self.chosen]

    def summary(self):
        marks = ' '.join(f'{c}{"*" if i in self.chosen else ""}' for i, c in enumerate(self.looks))
        if self.title.item or self.title.odd or self.title.ask:
            marks = ', '.join(f'{c}{"*" if i in self.chosen else ""}'
                              for i, c in enumerate(self.looks))
        return (f"target {self.title.target!r} x{self.title.count}; tiles read as [{marks}] "
                f"(* = clicking)")


def choose(shapes, scores, target, count):
    """Indexes of the tiles to click."""
    n = len(shapes)
    own = [s.get(target, 0.0) for s in scores]
    # Look-alike groups: a tile's evidence is its group's average.
    group = list(range(n))
    for i in range(n):
        for j in range(i):
            if shapes[i] is not None and shapes[j] is not None \
                    and float((shapes[i] * shapes[j]).sum()) >= SAME_LOOK:
                old, new = group[i], group[j]
                group = [new if g == old else g for g in group]
    pooled = [np.mean([own[j] for j in range(n) if group[j] == group[i]]) for i in range(n)]
    order = sorted(range(n), key=lambda i: (pooled[i], own[i]), reverse=True)
    if count is None:
        # No count: every tile whose best guess is (nearly) the target.
        return sorted(i for i in range(n) if own[i] >= max(scores[i].values()) - 0.03)
    return sorted(order[:min(count, n)])


def solve(frame, renderer, glyphs, scales=SCALES, items=None):
    """An Answer for the quiz in FRAME, or None if no quiz is visible."""
    title = read_title(frame, renderer, scales, items.names if items else ())
    if title is None:
        return None
    if title.ask:
        return solve_yesno(frame, title, items)
    if title.odd:
        return solve_odd(frame, title)
    if title.item:
        return solve_items(frame, title, items)
    tiles = find_tiles(frame, title)
    if not tiles:
        return None
    shapes = [letter_shape(frame[y0:y1, x0:x1], glyphs) for x0, y0, x1, y1 in tiles]
    scores = [glyphs.scores(s) for s in shapes]
    looks = [max(s, key=s.get) for s in scores]
    chosen = choose(shapes, scores, title.char, title.count)
    last = tiles[-1]
    button = ((last[0] + last[2]) / 2 + BUTTON_OFFSET[0] * title.scale,
              title.y + BUTTON_OFFSET[1] * title.scale)
    return Answer(title, tiles, chosen, scores, looks, button)


def find_button(frame, answer):
    """The lit CONFIRM button's centre near where it should be, else the estimate."""
    return lit_button(frame, answer.button, answer.title.scale)


def lit_button(frame, point, s, reach=(80, 40, 20)):
    """The lit (blue) CONFIRM button's centre near POINT, else POINT. REACH: px
    to the left, right and up/down of POINT to look in, at scale 1."""
    bx, by = point
    x0, x1 = int(max(0, bx - reach[0] * s)), int(max(0, bx + reach[1] * s))
    y0, y1 = int(max(0, by - reach[2] * s)), int(max(0, by + reach[2] * s))
    region = frame[y0:y1, x0:x1, :3]
    if region.size == 0:
        return point
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    lit = ((hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 115) &
           (hsv[:, :, 1] > 120) & (hsv[:, :, 2] > 180)).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(lit)
    best = None
    for i in range(1, count):
        w, h = stats[i, 2], stats[i, 3]
        if w >= 30 * s and 8 * s <= h <= 30 * s and (best is None or stats[i, 4] > stats[best, 4]):
            best = i
    if best is None:
        return point
    return x0 + float(centroids[best][0]), y0 + float(centroids[best][1])


# --------------------------------------------------------------- footer ----

# Every quiz panel seen so far (all four kinds, live and in screenshots) has
# "Remain : N Sec" at its bottom left while a question is up. It tells that a
# question is on screen even when its wording is new, which is when the AI
# fallback (quiz_ai.py) is asked. Both "Remain :" and "Sec" must match: small
# quest-window text alone reaches 0.82 on "Remain :", real footers 0.84-0.97.
FOOTER_THRESHOLD = 0.8
FOOTER_LIVE_SCALES = (0.95, 1.0, 1.05)  # the game window's own frames


def find_footer(frame, renderer, scales=SCALES):
    """(x, y, scale, score) of the "Remain : N Sec" footer, or None."""
    mask = text_mask(frame)
    best = None
    for scale in scales:
        template = renderer.render('Remain :', scale)
        if template.shape[0] > mask.shape[0] or template.shape[1] > mask.shape[1]:
            continue
        _, score, _, (x, y) = cv2.minMaxLoc(
            cv2.matchTemplate(mask, template, cv2.TM_CCOEFF_NORMED))
        if score < FOOTER_THRESHOLD or (best and score <= best[3]):
            continue
        sec = renderer.render('Sec', scale)
        x0 = max(0, int(x + renderer.offset('Remain : 0 ') * scale) - 4)
        x1 = int(x + renderer.offset('Remain : 10 ') * scale) + sec.shape[1] + 4
        window = mask[max(0, y - 4):y + sec.shape[0] + 4, x0:x1]
        if window.shape[0] < sec.shape[0] or window.shape[1] < sec.shape[1]:
            continue
        sec_score = cv2.minMaxLoc(cv2.matchTemplate(window, sec, cv2.TM_CCOEFF_NORMED))[1]
        score = min(score, sec_score)
        if score >= FOOTER_THRESHOLD and (best is None or score > best[3]):
            best = (x, y, scale, float(score))
    return best


def read_remain(frame, renderer, footer):
    """The N of "Remain : N Sec" at FOOTER (0-10), or None if it can't be read."""
    x, y, s, _ = footer
    slack = 4
    width = int(renderer.offset('Remain : 10 Sec') * s) + 20
    height = int((renderer.bottom - renderer.top + 4) * s) + 2 * slack
    x0, y0 = max(0, x - slack), max(0, y - slack)
    mask = text_mask(frame[y0:y0 + height, x0:x0 + width])
    best = (-1.0, None)
    for n in range(11):
        template = renderer.render(f'Remain : {n} Sec', s)
        if template.shape[0] > mask.shape[0] or template.shape[1] > mask.shape[1]:
            continue
        score = cv2.minMaxLoc(cv2.matchTemplate(mask, template, cv2.TM_CCOEFF_NORMED))[1]
        if score > best[0]:
            best = (score, n)
    return best[1] if best[0] >= TEXT_THRESHOLD else None


def panel_box(frame, footer):
    """A box (x0, y0, x1, y1) around the quiz panel above FOOTER, with room to
    spare for bigger panels than seen so far."""
    x, y, s, _ = footer
    h, w = frame.shape[:2]
    return (max(0, int(x - 90 * s)), max(0, int(y - 320 * s)),
            min(w, int(x + 540 * s)), min(h, int(y + 80 * s)))


def annotate(frame, answer):
    out = np.ascontiguousarray(frame[:, :, :3]).copy()
    for i, (x0, y0, x1, y1) in enumerate(answer.tiles):
        colour = (0, 255, 0) if i in answer.chosen else (0, 0, 255)
        cv2.rectangle(out, (x0, y0), (x1, y1), colour, 2)
        cv2.putText(out, answer.looks[i], (x0 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, colour, 2)
    if answer.button:
        bx, by = answer.button
        cv2.circle(out, (int(bx), int(by)), 6, (255, 255, 0), 2)
    t = answer.title
    cv2.putText(out, f'{t.target} x{t.count}',(t.x, max(12, t.y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return out


# ----------------------------------------------------------------- live ----

# The game first shows "Question in N seconds / Answer within 10 seconds" with
# N up to ~30 (miss-20261115-042125 started at 29 and was given up at 5).
QUIZ_WAIT_SECONDS = 40.0
STABLE_READS = 2             # same answer this many reads in a row before clicking
GONE_SECONDS = 4.0           # quiz absent this long after answering: done
# 17:10 run: "The quiz never appeared" with nobody watching and nothing saved,
# so there was no telling a false alarm from a quiz the solver could not read.
# While no quiz is read, what the game shows is kept (small JPEGs in memory)
# and written out only if the wait ends without an answer.
MISS_FRAME_SECONDS = 0.5     # one frame this often
MISS_FRAMES = 60             # at most this many (the newest are kept)
# AI fallback (quiz_ai.py): asked as soon as a question's footer shows, so its
# reply (a few seconds) is ready if the solvers here can't read the question.
AI_GRACE = 1.0               # the solvers here get this long before an AI answer is clicked
AI_ASKS = 2                  # per question (a second try if the first failed)
FOOTER_GONE = 1.5            # footer away this long: the next one is a new question
AI_PROBE_START = 4.0         # no footer: ask with the whole frame after this long ...
AI_PROBE_SECONDS = 6.0       # ... then this often (or when its countdown says) ...
AI_PROBES = 4                # ... at most this many times


def play_live(title='MapleStory', select=True, timeout=60.0, stop=None, record_dir=None,
              overlay=False, phone=False):
    """
    Waits for the quiz in the game window, clicks the matching tiles and then
    CONFIRM (an item question - "Click a popsicle" - only needs the item
    clicked, and only once: clicking it again would unpick it). Handles another question appearing afterwards. RECORD_DIR saves
    each decision as a screenshot (and an annotated copy) for checking.
    PHONE sends each answer (screenshot with it drawn on, plus the AI's
    confidence and reasoning) to Telegram (src/modules/phone_alert.py).
    """
    import ctypes
    import random
    import time
    import mss
    from src.detection.violetta_solver import _client_region, _click

    ctypes.windll.user32.SetProcessDPIAware()
    renderer, glyphs, items = _TitleRenderer(), Glyphs(), Items()
    # Load the image model during the "Question in N seconds" wait, so a yes/no
    # question's first read doesn't lose ~2 s of its 10.
    import threading
    from src.detection import item_clip
    threading.Thread(target=item_clip.available, daemon=True).start()
    scales = SCALES
    picked = None               # item question already clicked (title, item boxes)
    told_unknown = told_unsure = False
    started = time.monotonic()
    last_seen = None            # when the quiz was last on screen
    answered_at = None
    previous = None
    stable = 0
    answers = 0
    view = None
    missed = []                 # [(seconds since start, title score, jpeg)]
    last_missed = float('-inf')
    best_title = 0.0
    # AI fallback state, per question (a question = one stretch of footer on screen)
    ai = _quiz_ai()
    question_at = None          # when this question's footer (or a solver read) first showed
    footer_at = float('-inf')   # last time the footer or a solver read was seen
    footer = None
    footer_scales, footer_tries = FOOTER_LIVE_SCALES, 0
    ai_asks = 0                # asked this many times about this question
    ai_done = False             # this question is answered (by either side)
    ai_reply = None             # a usable reply waiting to be clicked
    ai_shown = None             # the reply drawn on the overlay (kept after clicking)
    ai_asked_at = ai_error = None
    ai_clicked = drew_ai = False
    probes, next_probe = 0, AI_PROBE_START

    def gave_up(result):
        if record_dir and not answers:
            _save_miss(record_dir, missed, best_title, result)
        return result
    if overlay:
        try:
            from src.detection.screen_overlay import ScreenOverlay
            view = ScreenOverlay()
        except Exception as error:
            print(f'[QUIZ] Overlay unavailable: {error}')
    print(f'[QUIZ] Watching {title!r} for the "Click all ..." / "Click a ..." / '
          f'"Click the one that is different" / "Is there a ... here?" / "Is this a ...?" quiz '
          f'(items known: {", ".join(items.names) or "none"}); '
          f'clicking is {"ON" if select else "OFF (preview)"}.')
    try:
        with mss.mss() as screen:
            while time.monotonic() - started < timeout:
                if stop is not None and stop.is_set():
                    return 'stopped'
                now = time.monotonic()
                if answered_at and last_seen and now - last_seen > GONE_SECONDS:
                    return f'answered {answers}'
                if not answers and now - started > QUIZ_WAIT_SECONDS and last_seen is None \
                        and not (ai and ai.busy):
                    if told_unknown:
                        return gave_up('unknown item')
                    print('[QUIZ] The quiz never appeared '
                          f'(best "Click a" match {best_title:.2f}, '
                          f'needs {TITLE_THRESHOLD:.2f}).')
                    return gave_up('no quiz')
                try:
                    handle, region = _client_region(title)
                    frame = np.asarray(screen.grab(region))[:, :, :3]
                except Exception:
                    time.sleep(0.3)
                    continue
                answer = solve(frame, renderer, glyphs, scales, items)
                if answer is None and read_title.last_item == '?' and not told_unknown:
                    told_unknown = True
                    print('[QUIZ] "Click a ..." asks for an item there is no picture of '
                          f'in {ITEM_DIR}. Add it with: python -m src.detection.letter_quiz '
                          '--add-item NAME SCREENSHOT --at X Y (screenshots are saved below).')
                if answer is not None and answer.title.item and not answer.chosen:
                    if not told_unsure:
                        told_unsure = True
                        print(f'[QUIZ] {answer.summary()} - none looks enough like it; '
                              'not clicking.')
                    last_seen = now
                    answer = None
                if answer is None and ai:
                    # Nothing here can read it: is a question up anyway? Then the AI
                    # is asked (in the background) and its answer clicked.
                    # The game's own frames show it at ~scale 1 (66 ms); all scales
                    # (420 ms) only every few frames, for odd window sizes.
                    footer_tries += 1
                    footer = find_footer(frame, renderer, footer_scales)
                    if footer is None and footer_tries % 4 == 0:
                        footer = find_footer(frame, renderer, SCALES)
                    if footer:
                        footer_scales = tuple(s for s in SCALES if abs(s - footer[2]) <= 0.051)
                        if now - footer_at > FOOTER_GONE:
                            question_at, ai_asks, ai_done, ai_reply = now, 0, False, None
                            ai_shown = ai_asked_at = ai_error = None
                            ai_clicked = False
                            ai.forget()
                        footer_at = last_seen = now
                    reply = ai.take()
                    if isinstance(reply, Exception):
                        ai_error = str(reply)
                        print(f'[QUIZ-AI] No answer: {reply}')
                    elif reply is not None:
                        if reply.usable and not ai_done:
                            if question_at is None or now - footer_at > FOOTER_GONE:
                                # seen by a probe: it has waited long enough already
                                question_at, footer_at = now - AI_GRACE, now
                            last_seen = now
                            ai_reply = ai_shown = reply
                            print(f'[QUIZ-AI] {reply.summary()}')
                            if record_dir:
                                _save_ai(record_dir, frame, reply)
                        elif reply.countdown > 0:
                            next_probe = now - started + reply.countdown + 0.5
                            print(f'[QUIZ-AI] Question in {reply.countdown} s.')
                        elif not ai_done:
                            ai_error = 'nothing to click'
                            print(f'[QUIZ-AI] Nothing to click: {reply.summary()}')
                    in_question = now - footer_at <= FOOTER_GONE
                    if in_question and not ai_done and not ai.busy and ai_asks < AI_ASKS                             and ai_reply is None and footer:
                        ai_asks += 1
                        ai_asked_at, ai_error = now, None
                        ai.ask(frame, panel_box(frame, footer))
                        print('[QUIZ-AI] The question is up but not read here; '
                              'asking the AI.')
                    elif not in_question and not answers and ai.probe and not ai.busy                             and probes < AI_PROBES and now - started >= next_probe:
                        probes += 1
                        next_probe = now - started + AI_PROBE_SECONDS
                        ai.ask(frame)
                    # Seconds left on the quiz timer: the lower of what it shows
                    # and 10 s from when the footer first showed (a misread digit
                    # must not make the click late).
                    remain = None
                    if in_question and question_at is not None:
                        remain = 10.0 - (now - question_at)
                        shown = read_remain(frame, renderer, footer) if footer else None
                        if shown is not None:
                            remain = min(remain, shown + 0.5)
                    click_now = False
                    if ai_reply is not None and not ai_done and question_at is not None                             and now - question_at >= AI_GRACE                             and not (answered_at and now - answered_at < 2.5):
                        click_now = not ai.click_at or remain is None or remain <= ai.click_at
                    if view and (in_question or ai.busy or ai_shown):
                        _show_ai(view, region, footer, ai, ai_shown, ai_asked_at, ai_error,
                                 now, remain, ai_clicked, select)
                        drew_ai = True
                    elif view and drew_ai:
                        view.show(region)
                        drew_ai = False
                    if click_now:
                        reply, ai_reply, ai_done = ai_reply, None, True
                        if select and _click_ai(screen, region, handle, reply, footer, _click):
                            ai_clicked = True
                            answers += 1
                            answered_at = time.monotonic()
                        if phone:
                            _phone_ai(frame, ai, reply, ai_clicked,
                                      'preview mode' if not select else 'game window not focused')
                        time.sleep(0.1)
                        continue
                if answer is None:
                    previous, stable = None, 0
                    score = read_title.last_score
                    best_title = max(best_title, score)
                    if record_dir and not answers and now - last_missed >= MISS_FRAME_SECONDS:
                        last_missed = now
                        ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if ok:
                            missed.append((now - started, score, jpeg))
                            del missed[:-MISS_FRAMES]
                    time.sleep(0.15)
                    continue
                last_seen = now
                if now - footer_at > FOOTER_GONE:       # (a question read here counts as one)
                    question_at, ai_asks, ai_done, ai_reply = now, 0, False, None
                footer_at = now
                scales = tuple(s for s in SCALES if abs(s - answer.title.scale) <= 0.051)
                key = (answer.title.target, answer.title.count, len(answer.tiles),
                       tuple(answer.chosen))
                stable = stable + 1 if key == previous else 1
                previous = key
                if picked and answer.title.item:
                    time.sleep(0.15)   # already picked; the quiz stays up until its timer ends
                    continue
                if answered_at and now - answered_at < 2.5:
                    time.sleep(0.15)   # our answer is being processed
                    continue
                if stable < STABLE_READS:
                    time.sleep(0.1)
                    continue
                print(f'[QUIZ] {answer.summary()}')
                if ai:                  # answered here: the AI's answer isn't needed
                    ai_done, ai_reply, ai_shown = True, None, None
                    ai.forget()
                if record_dir:
                    _save(record_dir, frame, answer)
                if view:
                    _show(view, region, answer)
                if phone:
                    focused = ctypes.windll.user32.GetForegroundWindow() == handle
                    _phone_solver(frame, answer, select and focused,
                                  'preview mode' if not select else 'game window not focused')
                if select:
                    if ctypes.windll.user32.GetForegroundWindow() != handle:
                        print('[QUIZ] Game window is not focused; not clicking.')
                    else:
                        for x, y in answer.points():
                            time.sleep(0.15 + 0.2 * random.random())
                            _click(region['left'] + x, region['top'] + y)
                        if answer.button is None:
                            picked = answer.title.item
                            print(f'[QUIZ] Clicked the {answer.title.item} (no CONFIRM needed'
                                  f'{"; picked by elimination" if answer.guessed else ""}).')
                            if answer.guessed:
                                _learn_item(frame, answer, items, record_dir)
                            answers += 1
                            answered_at = time.monotonic()
                            previous, stable = None, 0
                            continue
                        time.sleep(0.25 + 0.2 * random.random())
                        frame = np.asarray(screen.grab(region))[:, :, :3]
                        bx, by = find_button(frame, answer)
                        _click(region['left'] + bx, region['top'] + by)
                        print(f'[QUIZ] Clicked {len(answer.chosen)} tile(s) and CONFIRM'
                              f'{" (not sure - best guess)" if answer.guessed else ""}.')
                answers += 1
                answered_at = time.monotonic()
                previous, stable = None, 0
        return gave_up(f'timed out ({answers} answered)')
    finally:
        if view:
            view.close()


def _save(directory, frame, answer):
    import time
    try:
        os.makedirs(directory, exist_ok=True)
        stamp = time.strftime('quiz-%Y%m%d-%H%M%S')
        cv2.imwrite(os.path.join(directory, f'{stamp}.png'), frame)
        cv2.imwrite(os.path.join(directory, f'{stamp}-answer.png'), annotate(frame, answer))
    except Exception as error:
        print(f'[QUIZ] Could not save the screenshot: {error}')


def _phone(text, picture):
    """Sends TEXT with PICTURE to Telegram if set up and 'quiz_answer' is on."""
    try:
        from src.modules import phone_alert
        if phone_alert.wants('quiz_answer'):
            phone_alert.send(text, frame=picture)
    except Exception as error:
        print(f'[QUIZ] Phone message not sent: {error}')


def _phone_ai(frame, ai, reply, clicked, why_not):
    from src.detection import quiz_ai
    picks = ', '.join(what for _, _, what in reply.points) or 'nothing'
    lines = [f'🧠 Quiz answered by the AI ({ai.backend}: {ai.model})',
             f'Q: {reply.question}',
             f'Answer: {picks}{" + CONFIRM" if reply.confirm else ""}',
             f'Confidence: {reply.confidence}%   Reply time: {reply.seconds:.1f} s',
             f'Why: {reply.reasoning}',
             'Clicked ✅' if clicked else f'NOT clicked ⚠️ ({why_not})']
    _phone('\n'.join(lines), quiz_ai.annotate(frame, reply))


def _phone_solver(frame, answer, clicking, why_not):
    lines = ['🧩 Quiz answered by the built-in solver',
             answer.summary(),
             'Not sure: best guess' if answer.guessed else 'Confident (matched a known question type)',
             'Clicking ✅' if clicking else f'NOT clicking ⚠️ ({why_not})']
    _phone('\n'.join(lines), annotate(frame, answer))


def _quiz_ai():
    """The AI fallback (quiz_ai.QuizAI) if it is set up, else None."""
    try:
        from src.detection import quiz_ai
        ai = quiz_ai.QuizAI()
    except Exception as error:
        print(f'[QUIZ-AI] Unavailable: {error}')
        return None
    if not ai.on:
        print(f'[QUIZ-AI] AI fallback is off: {ai.why_off}.')
        return None
    print(f'[QUIZ-AI] AI fallback is on ({ai.backend}: {ai.model}).')
    return ai


def _click_ai(screen, region, handle, reply, footer, click):
    """Clicks the AI's answer, then CONFIRM (snapped to the lit button). True if clicked."""
    import ctypes
    import random
    import time
    if ctypes.windll.user32.GetForegroundWindow() != handle:
        print('[QUIZ-AI] Game window is not focused; not clicking.')
        return False
    for x, y, _ in reply.points:
        time.sleep(0.15 + 0.2 * random.random())
        click(region['left'] + x, region['top'] + y)
    if reply.confirm:
        time.sleep(0.25 + 0.2 * random.random())
        frame = np.asarray(screen.grab(region))[:, :, :3]
        bx, by = lit_button(frame, reply.confirm, footer[2] if footer else 1.0, (60, 60, 25))
        click(region['left'] + bx, region['top'] + by)
    print(f'[QUIZ-AI] Clicked {len(reply.points)} point(s)'
          f'{" and CONFIRM" if reply.confirm else ""}.')
    return True


def _save_ai(directory, frame, reply):
    """The screenshot, the AI's answer drawn on it and its reply, for checking
    (and for teaching the solvers here the new kind of question later)."""
    import json
    import time
    from src.detection import quiz_ai
    try:
        os.makedirs(directory, exist_ok=True)
        stamp = time.strftime('ai-%Y%m%d-%H%M%S')
        cv2.imwrite(os.path.join(directory, f'{stamp}.png'), frame)
        cv2.imwrite(os.path.join(directory, f'{stamp}-answer.png'), quiz_ai.annotate(frame, reply))
        with open(os.path.join(directory, f'{stamp}.json'), 'w', encoding='utf-8') as file:
            json.dump({**reply.data, 'seconds': round(reply.seconds, 2),
                       'frame_points': reply.points, 'frame_confirm': reply.confirm},
                      file, indent=2)
    except Exception as error:
        print(f'[QUIZ-AI] Could not save the screenshot: {error}')


def _learn_item(frame, answer, items, record_dir):
    """Keeps the picture of an item that was chosen by elimination: straight into
    ITEM_DIR when the elimination was sure, else into RECORD_DIR for a look."""
    import time
    name, s = answer.title.item, answer.title.scale
    try:
        if answer.learn is not None:
            box, mask = answer.learn
            picture = cut_item(frame, box, mask, s)
            os.makedirs(ITEM_DIR, exist_ok=True)
            path = os.path.join(ITEM_DIR, name.replace(' ', '_') + '.png')
            cv2.imwrite(path, picture)
            items.add(name, picture)
            print(f'[QUIZ] Saved the {name} picture to {path}; next time it is matched directly.')
        elif record_dir:
            folder = os.path.join(record_dir, time.strftime(f'learn-{name}-%Y%m%d-%H%M%S'))
            os.makedirs(folder, exist_ok=True)
            cv2.imwrite(os.path.join(folder, 'quiz.png'), frame)
            for i, box in enumerate(answer.tiles):
                x0, y0, x1, y1 = box
                tag = 'clicked' if i in answer.chosen else 'other'
                cv2.imwrite(os.path.join(folder, f'{i}-{tag}.png'), frame[y0:y1, x0:x1])
            print(f'[QUIZ] Not sure which picture is the {name}; saved them to {folder}. '
                  f'Add the right one with --add-item "{name}" {folder}\\quiz.png --at X Y')
    except Exception as error:
        print(f'[QUIZ] Could not save the {name} picture: {error}')


def _save_miss(directory, missed, best_title, result):
    """Writes what the game showed while no quiz was read, for a look later."""
    import time
    if not missed:
        return
    try:
        folder = os.path.join(directory, time.strftime('miss-%Y%m%d-%H%M%S'))
        os.makedirs(folder, exist_ok=True)
        for seconds, score, jpeg in missed:
            with open(os.path.join(
                    folder, f'{seconds:05.1f}s-title{score:.2f}.jpg'), 'wb') as file:
                file.write(jpeg.tobytes())
        print(f'[QUIZ] Saved {len(missed)} screenshots of the wait ({result}; best title '
              f'match {best_title:.2f}) to {folder} - send that folder to get this fixed.')
    except Exception as error:
        print(f'[QUIZ] Could not save the screenshots: {error}')


def _show(view, region, answer):
    from src.detection import screen_overlay as so
    boxes = [(x0, y0, x1, y1, so.GREEN if i in answer.chosen else so.RED, 2)
             for i, (x0, y0, x1, y1) in enumerate(answer.tiles)]
    labels = [(x0, y1 + 2, answer.looks[i], so.WHITE)
              for i, (x0, y0, x1, y1) in enumerate(answer.tiles)]
    if answer.tiles:
        x0 = min(t[0] for t in answer.tiles)
        y0 = min(t[1] for t in answer.tiles)
        labels.append((x0, max(0, y0 - 60), 'SAVED question - solved here, no AI '
                       f'({answer.title.target})', so.GREEN))
    view.show(region, boxes, labels)


def _show_ai(view, region, footer, ai, reply, asked_at, error, now, remain, clicked, select):
    """Draws the AI fallback's state over the game: that the question is new,
    how long the AI has taken, its answer and confidence, the timer, and pink
    circles where it will click (CONFIRM: a smaller pink circle)."""
    from src.detection import screen_overlay as so
    if footer:
        x, y, s, _ = footer
        left, top = max(4, int(x - 10 * s)), int(y + 40 * s)
    else:
        left, top = 10, 60
    lines = [('NEW question - not saved here, the AI answers it', so.PINK)]
    if reply is not None:
        colour = so.GREEN if reply.confidence >= 80 else so.YELLOW if reply.confidence >= 50             else so.RED
        lines.append((f'AI answered in {reply.seconds:.1f} s - {reply.confidence}% sure', colour))
        lines.append((f'Q: {reply.question}'[:70], so.WHITE))
        clicks = ', '.join(what for _, _, what in reply.points)
        lines.append((f'Clicks: {clicks}{" + CONFIRM" if reply.confirm else ""}'[:70], so.WHITE))
    elif ai.busy and asked_at is not None:
        lines.append((f'Asking the AI ... {now - asked_at:.1f} s', so.YELLOW))
    elif error:
        lines.append((f'AI failed: {error}'[:70], so.RED))
    if clicked:
        lines.append(('Clicked.', so.GREEN))
    elif reply is not None and not select:
        lines.append(('Preview only - clicking is off', so.CYAN))
    elif reply is not None and remain is not None:
        when = f'clicks at {ai.click_at} s' if ai.click_at else 'clicking now'
        lines.append((f'Timer ~{max(0.0, remain):.0f} s - {when}', so.CYAN))
    labels = [(left, top + 18 * i, text, colour) for i, (text, colour) in enumerate(lines)]
    circles = []
    if reply is not None:
        for i, (px, py, what) in enumerate(reply.points):
            circles.append((px, py, 18, so.PINK, 3))
            labels.append((px + 20, py - 8, str(i + 1), so.PINK))
        if reply.confirm:
            circles.append((reply.confirm[0], reply.confirm[1], 10, so.PINK, 3))
    view.show(region, labels=labels, circles=circles)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Solve the "Click all X\'s" quiz.')
    parser.add_argument('image', nargs='?', help='Screenshot to solve (omit to watch the game).')
    parser.add_argument('--output', help='With IMAGE: save an annotated copy here.')
    parser.add_argument('--title', default='MapleStory')
    parser.add_argument('--select', action='store_true', help='Live: click the answer.')
    parser.add_argument('--timeout', type=float, default=60.0)
    parser.add_argument('--add-item', metavar='NAME',
                        help='With IMAGE and --at: save the item at that point as NAME '
                             '(as written in "Click a NAME").')
    parser.add_argument('--at', nargs=2, type=int, metavar=('X', 'Y'),
                        help='A point on the item in IMAGE.')
    args = parser.parse_args()
    if args.add_item:
        if not args.image or not args.at:
            parser.error('--add-item needs IMAGE and --at X Y')
        print(f'Saved {add_item(args.add_item, args.image, tuple(args.at))}')
        return
    if args.image:
        frame = cv2.imread(args.image)
        answer = solve(frame, _TitleRenderer(), Glyphs(), items=Items())
        if answer is None:
            print('No quiz found.')
            return
        print(answer.title)
        print(answer.summary())
        if args.output:
            cv2.imwrite(args.output, annotate(frame, answer))
    else:
        print(play_live(args.title, args.select, args.timeout, record_dir='logs/letter_quiz'))


if __name__ == '__main__':
    main()
