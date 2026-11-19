"""
Which way the character is facing, read from its head (hair, eyes, ear).

A MapleStory character facing right is the mirror image of itself facing
left: the face and eyes sit toward the side it looks at, the ear and the back
of the hair toward the other. Nothing here is drawn for one character. The
reader keeps a few pictures of THIS character's head, all turned the same way
round, and compares the head on screen with them and with their mirror
images: whichever fits better tells which of the two ways it is turned.

Which of the two is "left" comes from the bot's walking: while it holds the
left arrow the character faces left, while it holds right it faces right.
Each such moment is one vote, and the majority decides, so a vote taken at a
bad moment (the key had not reached the game yet, a knock-back) is outvoted
instead of teaching the reader the wrong way round.

Why (attack1.mp4, 2026-11-17): the bot let go of the attack, tapped the left
arrow 90 ms later to turn and held the attack again. The tap landed in the
middle of the last swing, the game ignored it, and the character swung to the
right at nothing for six seconds while the bot believed it faced left.

The head is found from the name tag under the character (the tracker's
"player badge"): HEAD_* are where a head sits above the middle of the tag.
"""

import cv2
import numpy as np

HEAD_HALF_WIDTH = 26        # px either side of the name tag's middle
HEAD_TOP = 80               # px above the name tag's middle: top of the hair
HEAD_BOTTOM = 30            # ...and the chin
SLACK_X = 12                # a swing leans the body: look this far sideways
SLACK_Y = 6
MAX_SAMPLES = 6
SAMPLE_GAP = 0.4            # s between two learnt pictures / votes
# (attack1.mp4: standing in a portal, whose light is drawn over the character,
# and mid-swing the right way round fitted 0.36-0.42 and the wrong way
# 0.22-0.30; walking in the open it is 0.6-1.0 against 0.2-0.4)
MIN_SCORE = 0.33            # a head fits at least this well, or nothing is said
MIN_MARGIN = 0.08           # one way must fit this much better than the other
STEADY_READINGS = 2         # this many readings in a row agree: it is certain
FRESH_SECONDS = 0.5         # a reading older than this is not "seen"
STEADY_VARIATION = 28       # pixels that change less than this between
#                             pictures are the character, the rest scenery
MIN_VOTES = 2               # walking moments needed before a side is named...
MIN_LEAD = 2                # ...and how far the majority must lead
MAX_VOTES = 40              # (old votes fade, so a change of hat is relearnt)
ANCHOR_TRIES = 10           # the first picture fits nothing this often: start over


class FacingReader:
    def __init__(self):
        self.reset()

    def reset(self):
        self.samples = []           # head pictures, all turned the same way (BGR)
        self._mask = None           # the character's own pixels (uint8), when known
        self._learnt_at = float('-inf')
        # Votes on what the stored pictures show: +1 = they face left
        self._votes = []
        self._misfits = 0
        self.last = None            # (direction or None, left score, right score, time)
        self._streak = (None, 0, float('-inf'))     # (direction, readings in a row, time)

    # ---------------------------------------------------------- helpers ----

    @staticmethod
    def _head(frame, centre, slack_x=0, slack_y=0):
        x, y = int(centre[0]), int(centre[1])
        x1, x2 = x - HEAD_HALF_WIDTH - slack_x, x + HEAD_HALF_WIDTH + slack_x
        y1, y2 = y - HEAD_TOP - slack_y, y - HEAD_BOTTOM + slack_y
        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            return None
        return np.ascontiguousarray(frame[y1:y2, x1:x2, :3])

    @staticmethod
    def _best(region, picture, mask):
        """(fit 0-1, x, y) of PICTURE at its best place in REGION."""
        if mask is None:
            result = cv2.matchTemplate(region, picture, cv2.TM_CCOEFF_NORMED)
        else:
            result = cv2.matchTemplate(region, picture, cv2.TM_CCOEFF_NORMED,
                                       mask=cv2.merge([mask] * 3))
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        result[(result > 1.0) | (result < -1.0)] = 0.0
        _, score, _, place = cv2.minMaxLoc(result)
        return float(score), place

    def _fits(self, region):
        """(fit as stored, fit mirrored, place of the better one) of the head
        somewhere in REGION."""
        same = mirrored = 0.0
        place = (0, 0)
        mask = self._mask
        mirrored_mask = None if mask is None else np.ascontiguousarray(mask[:, ::-1])
        for picture in self.samples:
            score, at = self._best(region, picture, mask)
            if score > max(same, mirrored):
                place = at
            same = max(same, score)
            score, at = self._best(
                region, np.ascontiguousarray(picture[:, ::-1]), mirrored_mask)
            if score > max(same, mirrored):
                place = at
            mirrored = max(mirrored, score)
        return same, mirrored, place

    def _stored_side(self):
        """'left'/'right': the way the stored pictures face, by the votes."""
        lead = sum(self._votes)
        if len(self._votes) < MIN_VOTES or abs(lead) < MIN_LEAD:
            return None
        return 'left' if lead > 0 else 'right'

    def _steady_mask(self):
        """The pixels that look the same in every picture: the character. The
        scenery behind the head is different each time."""
        if len(self.samples) < 3:
            return None
        stack = np.stack([s.astype(np.float32) for s in self.samples])
        variation = stack.std(axis=0).max(axis=2)
        mask = (variation < STEADY_VARIATION).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        if not 0.15 <= mask.mean() <= 0.95:
            return None
        return mask * 255

    @property
    def ready(self):
        return bool(self.samples) and self._stored_side() is not None

    # ------------------------------------------------------------ learn ----

    def learn(self, frame, centre, direction, now):
        """The bot is walking DIRECTION ('left'/'right') right now: one vote on
        which way round the head is, and maybe one more picture of it."""
        if direction not in ('left', 'right') or now - self._learnt_at < SAMPLE_GAP:
            return False
        region = self._head(frame, centre, SLACK_X, SLACK_Y)
        head = self._head(frame, centre)
        if region is None or head is None:
            return False
        if not self.samples:
            self.samples = [head.copy()]            # the first picture: the reference
            self._votes = [1 if direction == 'left' else -1]
            self._learnt_at = now
            return True
        same, mirrored, place = self._fits(region)
        if max(same, mirrored) < MIN_SCORE or abs(same - mirrored) < MIN_MARGIN:
            # Not recognisably this head either way (hidden by an effect, a
            # hit pose): no vote. A reference that never fits is a bad one.
            self._misfits += 1
            if len(self.samples) == 1 and self._misfits >= ANCHOR_TRIES:
                self.reset()
            return False
        self._misfits = 0
        as_stored = same > mirrored
        # (the stored pictures face left if this one does and looks like them,
        # or if it faces right and looks like their mirror image)
        self._votes.append(1 if (direction == 'left') == as_stored else -1)
        del self._votes[:-MAX_VOTES]
        self._learnt_at = now
        if max(same, mirrored) < 0.97:              # not the very same picture again
            height = HEAD_TOP - HEAD_BOTTOM
            picture = region[place[1]:place[1] + height,
                             place[0]:place[0] + 2 * HEAD_HALF_WIDTH]
            if picture.shape[:2] == self.samples[0].shape[:2]:
                if not as_stored:
                    picture = picture[:, ::-1]
                # (the reference stays; the others are replaced oldest first)
                self.samples.append(np.ascontiguousarray(picture).copy())
                if len(self.samples) > MAX_SAMPLES:
                    del self.samples[1]
                self._mask = self._steady_mask()
        return True

    # ------------------------------------------------------------- read ----

    def read(self, frame, centre, now):
        """'left', 'right' or None (not learnt yet, or can't tell). CENTRE is
        the middle of the name tag."""
        stored = self._stored_side() if self.samples else None
        region = self._head(frame, centre, SLACK_X, SLACK_Y) if stored else None
        if region is None:
            self.last = (None, 0.0, 0.0, now)
            return None
        same, mirrored, _ = self._fits(region)
        left, right = (same, mirrored) if stored == 'left' else (mirrored, same)
        direction = None
        if max(left, right) >= MIN_SCORE and abs(left - right) >= MIN_MARGIN:
            direction = 'left' if left > right else 'right'
        self.last = (direction, left, right, now)
        if direction is not None:
            # (readings that can't tell don't break a run; the other way does)
            run = self._streak[1] + 1 if self._streak[0] == direction else 1
            self._streak = (direction, run, now)
        return direction

    def steady(self, now):
        """The way it faces when the last STEADY_READINGS readings that could
        tell all agree and the newest is fresh, else None."""
        direction, run, at = self._streak
        if direction is None or run < STEADY_READINGS or now - at > FRESH_SECONDS:
            return None
        return direction

    def seen(self, now):
        """The last reading if it is fresh, else None."""
        if self.last is None or now - self.last[3] > FRESH_SECONDS:
            return None
        return self.last[0]
