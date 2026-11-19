"""
Follows all four Violettas through a shuffle, frame by frame.

The row-based readers (violetta_solver.LAYOUTS, violetta_rows) group walkers
by their size; in logs/violetta/violetta-20261115-045012.mp4 three walkers
were 0.77, 0.80 and 0.86 of standing size, too close to tell apart that way,
and the round failed. Here each Violetta is a track instead:

* Every processed frame, walkers are found at any size and height
  (violetta_rows' coarse scan + refine): (score, scale, x, y, pose).
* Each track takes the detection that best continues it - near where her pace
  would put her, at a similar size and height. A track with no detection
  (hidden behind another, or the lights-out when all four vanish) keeps its
  last place and waits.
* Tracks start in the four slots (the Violettas always stand there before a
  shuffle; in the lights-out game they reappeared in the same slots), so the
  make-up Violetta's track is known from the reveal.
* Her track's last place is the answer.

Processing a frame takes ~0.2 s, so the solver feeds this on a background
thread with whichever frame is newest (every ~3rd frame at 13 fps); walkers
move < 10 px a frame, slots are 104 px apart.

Coordinates are window space (unpadded), like violetta_solver.
"""

import itertools
import threading
import time

import cv2
import numpy as np

from src.detection import violetta_rows as vr
from src.detection.violetta_solver import (SLOT_COUNT, SLOT_STEP, STAND_PIVOT_Y,
                                           end_slots_with_lost, nearest_slots, slot_x)

SEEN = 0.5                  # detection score that counts
TRACK_CANDIDATE_SEEN = 0.25 # coarse local peaks are refined before being accepted
TRACK_SEEN = 0.35           # refined, motion-gated matches survive changing stage lights
SLOT_OFFSET = -4            # detections' x at a slot, relative to slot_x (measured: 47/142/242/342)
GATE_X = 14                 # px a track may move beyond its predicted place, plus...
GATE_X_PER_FRAME = 14       # ...this much per frame since it was last seen
# A Violetta changes size only while stepping into or out of her row on the
# spot; walking, her size stays put. So a big size change is allowed only for
# a match that has barely moved (the 13 Nov game: a track whose Violetta hid at
# 0.51 jumped onto a 0.8 walker passing by).
# Stepping in or out, the size changes gradually (~0.05 of log size a frame:
# 0.78 -> 0.59 over 6 frames in the lights-out game); a walker passing a
# Violetta who stands still is a sudden jump (seed-3 test: 0.5 -> 0.66 in one
# step, which swapped them).
GATE_SCALE_WALKING = 0.10   # |log(size ratio)| allowed while moving
GATE_SCALE_PER_FRAME = 0.15 # ...in place, per frame since the last match (002959 at ~9 fps:
                            # sinking into the back row 0.97 -> 0.73 at ~0.09 a frame)
# ...but never less than this for a single frame. 20261119-195802 at ~10 fps:
# stepping into the back row went 0.69 -> 0.59 -> 0.51, 0.157 a frame; at 0.15
# she was dropped mid-step, went lost, and her walk left along the back row was
# never followed. Only the one-frame allowance is raised: 0.20 a frame over a
# 3-frame gap (0.60) let a walker of another size take over a track in
# 20261117-102544 round 2 replayed at every 3rd frame.
GATE_SCALE_ONE_FRAME = 0.20
GATE_SCALE_IN_PLACE = 0.60  # ...up to this (002959 round 4: stepping out of the far-back
                            # row, 0.57 -> 1.0, at ~0.15 a frame with misses between)
GATE_SCALE_REACQUIRE = 0.19 # after a missed frame, allow a larger scale change
GATE_SCALE_REACQUIRE_PER_FRAME = 0.08
GATE_SCALE_REACQUIRE_MOVING = 0.13  # small drift only; reject a foreground size jump
IN_PLACE_PX = 8             # moved at most this (plus 0.5 px a frame): "in place"
GATE_Y = 14                 # px of height change allowed per match
COST_SCALE = 150.0          # px of x error one unit of |log size ratio| is worth
# During a long moving re-acquisition, scale continuity outranks a nearer clone.
COST_SCALE_REACQUIRE = 700.0
COST_Y = 1.5                # px of x error per px of height error
ASSOCIATION_AMBIGUITY_MARGIN = 12.0  # px: don't swap IDs on near-equal assignments
RESIDUAL_REACQUIRE_MIN_MISSES = 3
MAX_SPEED = 12.0            # px/frame: a move this fast needs no extra evidence
# Walkers can be much faster: the pace changes between games and rounds, and a
# live capture at ~9 fps doubles the px per frame (violetta-20261117-002959
# round 4: the far-back walker moved ~25 px/frame and was gated out on her
# first step while plainly visible). A walker keeps her size and height exactly
# while she walks, and each walker in a round has her own row, so a faster move
# is accepted when both match tightly.
FAST_WALK_SPEED = 60.0      # px/frame: the fastest walk accepted (also the velocity clamp);
                            # the simulator's walkers dash at up to ~53 px/frame at 8 fps
# Walkers can also turn round mid-walk (violetta-sim 031612 round 4: the make-up
# Violetta walked left, back right, then dashed left to slot 1; the far-back one
# went left to slot 1's x and back to slot 4). On her own row a turn is allowed,
# and the search covers both ways around where she was. Not at standing size:
# a standing Violetta must never leap onto her neighbour.
STANDING_LOG_SCALE = 0.05
FAST_GATE_SCALE = 0.06      # |log size ratio| allowed for a move faster than MAX_SPEED
FAST_GATE_Y = 10            # px of height change allowed for a move faster than MAX_SPEED
MAX_LOG_SCALE_SPEED = 0.10
MOTION_SPEED_TOLERANCE = 1.5
MOTION_UNCERTAINTY_PER_FRAME = 3.0
TERMINAL_SLOT_TOLERANCE = 18
TERMINAL_MAX_JUMP = SLOT_STEP + 30
TERMINAL_X_WINDOW = 14
TERMINAL_MIN_SCORE = 0.58
TERMINAL_CONFIRM_MATCHES = 3
IDENTITY_AMBIGUITY_DISTANCE = 18.0
IDENTITY_AMBIGUITY_FRAME_GAP = 3
# Passing close only confuses two IDs of about the same size: size is part of
# every match (002959: a 1.54 front walker passing a 1.37 Violetta stepping in
# place flagged both, and every round's tracks were distrusted for it).
IDENTITY_AMBIGUITY_LOG_SCALE = 0.08
# Stepping into or out of a row drifts a few px one way while the size changes,
# then the walk often goes the other way (violetta-20261116-190834 round 2: the
# back-row walker drifted 345 -> 361 shrinking, then walked left at ~9 px/frame;
# the step-in velocity kept her prediction in place and a 0.71 match 28 px away
# was gated out). So a step-in forgets the velocity, and the tight directional
# gate is only used once two walking matches agree on the direction.
STEP_IN_LOG_SCALE = 0.03    # in-place size change per match that counts as stepping
WALK_MIN_SPEED = 2.5        # px/frame: a match this fast is a walking step
WALK_CONFIRM_MATCHES = 2
# A track's height is seeded at STAND_PIVOT_Y, but standing pivots measure
# 110-145 (190834 round 4: seeded 121, stood at ~140, the first step forward
# at y143 was gated out by 2 px and she was never matched).
GATE_Y_UNSEEDED = 25        # extra px of height allowed before a track's 2nd match
GATE_Y_PER_LOG_SCALE = 60.0 # px of pivot drop per unit of log size, stepping in place
# A track missed this long is LOST: it stops competing for detections near its
# last place (190834 round 4: a track lost for 37 frames still sat on slot 1
# and made the yellow walker arriving there ambiguous, so she lost her match
# and the stale track took her). It waits for a detection nobody else claims.
LOST_AFTER_FRAMES = 9
LOST_SEEN = 0.5             # score a detection needs to re-identify a lost track
LOST_STAND_SCALES = (0.94, 1.0, 1.07, 1.15, 1.24)   # searched at every slot when lost
STANDING_SCALES = (0.94, 1.0, 1.07)   # a lost track may come back at any size if standing
LOST_GATE_Y = 60
# Two lost Violettas on the same row, walking, can't pass each other: when they
# come back they are in the order they went in (002959 round 4: two walkers
# vanished into the far-back row behind the front walker and reappeared at
# slots 1 and 2 a frame apart; by distance alone either way cost the same).
# So a same-row group is only re-identified once all of it is visible again,
# left to right in its old order.
SAME_ROW_LOG_SCALE = 0.15
SAME_ROW_Y = 15
# A missing track on the same row as a lost one joins its group at once: in
# 002959 round 4 the make-up Violetta went lost one frame before her row-mate,
# took the first back-row walker to reappear alone, and the two came out
# swapped (she walked left ahead of her, so she must end further left).
GROUP_MISSED_FRAMES = 3
LOST_SEARCH_LOG_SCALE = 0.30    # lost walkers keep shrinking out of sight
# A hidden walker keeps walking at her pace (190834 round 2: hidden behind the
# front walker for 6 frames, her coasted prediction stalled at x 259 with the
# old 0.8/frame decay while she walked on to 209 and was gated out).
COAST_DECAY = 0.8           # per frame, for tracks with no confirmed walk
COAST_DECAY_WALKING = 0.95  # ...and for confirmed walkers
# Matching used to maximise the number of matches first, so two bad
# re-acquisitions (size jumps costing ~250) beat one clean match (190834
# round 2 frame 201: blue took the back-row walker and she took blue). Now
# each track left unmatched costs this much instead.
MISS_COST = 120.0
LOST_MISS_COST = 1e6        # lost tracks: as many matches as possible


class Track:
    def __init__(self, slot, frame):
        self.slot = slot                    # where she stood when the round began
        self.x = slot_x(slot) + SLOT_OFFSET
        self.scale, self.y = 1.0, STAND_PIVOT_Y
        self.vx = 0.0
        self.vy = 0.0
        self.vlog_scale = 0.0
        self.direction = 0
        self._coast_dx = 0.0
        self._coast_dy = 0.0
        self._coast_log_scale = 0.0
        self.seen_at = frame
        self.last_update_frame = frame
        self.missed_frames = 0
        self.path = [(frame, self.x, self.scale, self.y, 1.0)]
        self.matched = 0
        self.pose = 0
        self.terminal_slot = None
        self.terminal_matches = 0
        self.terminal_confirmed = False
        self.walk_matches = 0               # consecutive walking matches one way
        self.lost = False

    def predict(self, frame):
        return self.predict_state(frame)[0]

    def predict_state(self, frame):
        elapsed = max(0, frame - self.last_update_frame)
        return (
            self.x + self._coast_dx + self.vx * elapsed,
            self.y + self._coast_dy + self.vy * elapsed,
            float(np.exp(np.log(self.scale) + self._coast_log_scale
                         + self.vlog_scale * elapsed)),
        )

    def take(self, frame, hit, terminal_slot=None):
        score, scale, x, y = hit[:4]
        gap = max(1, frame - self.seen_at)
        movement = x - self.x
        vx = movement / gap
        vy = (y - self.y) / gap
        vlog_scale = np.log(scale / self.scale) / gap
        stepping = (abs(movement) <= IN_PLACE_PX + 0.5 * gap
                    and abs(np.log(scale / self.scale)) >= STEP_IN_LOG_SCALE * min(gap, 3))
        walking = abs(vx) >= WALK_MIN_SPEED
        restart = stepping or self.lost     # a lost track's jump says nothing of her pace
        if restart:
            # Stepping in or out on the spot: the next walk may go either way
            self.vx = 0.0
            self.direction = 0
            self.walk_matches = 0
        elif walking and (not self.walk_matches or np.sign(vx) != np.sign(self.vx)):
            # Setting off or turning: her pace is this step's, with no lag
            # (half of a 25 px/frame step missed the directional gate)
            self.vx = float(np.clip(vx, -FAST_WALK_SPEED, FAST_WALK_SPEED))
            self.walk_matches = 1
        else:
            self.vx = float(np.clip(0.5 * self.vx + 0.5 * vx,
                                    -FAST_WALK_SPEED, FAST_WALK_SPEED))
            if walking:
                self.walk_matches += 1
        self.vy = float(np.clip(0.5 * self.vy + 0.5 * vy, -MAX_SPEED, MAX_SPEED))
        self.vlog_scale = float(np.clip(
            0.5 * self.vlog_scale + 0.5 * vlog_scale,
            -MAX_LOG_SCALE_SPEED, MAX_LOG_SCALE_SPEED))
        if abs(movement) >= GATE_X and not restart:
            self.direction = 1 if movement > 0 else -1
        self.lost = False
        if terminal_slot is not None:
            if terminal_slot != self.terminal_slot:
                self.terminal_matches = 0
                self.terminal_confirmed = False
            self.terminal_slot = terminal_slot
            self.terminal_matches += 1
        elif self.terminal_slot is not None:
            anchor = slot_x(self.terminal_slot) + SLOT_OFFSET
            if (abs(x - anchor) <= TERMINAL_SLOT_TOLERANCE
                    and score >= SEEN):
                self.terminal_matches += 1
            else:
                self.terminal_slot = None
                self.terminal_matches = 0
                self.terminal_confirmed = False
        if self.terminal_matches >= TERMINAL_CONFIRM_MATCHES:
            self.terminal_confirmed = True
        self.x, self.scale, self.y, self.seen_at = x, scale, y, frame
        if len(hit) > 4:
            self.pose = int(hit[4])
        self._coast_dx = self._coast_dy = self._coast_log_scale = 0.0
        self.last_update_frame = frame
        self.missed_frames = 0
        self.path.append((frame, x, scale, y, score))
        self.matched += 1

    def wait(self, frame):
        elapsed = max(1, frame - self.last_update_frame)
        self.missed_frames += elapsed
        decay = (COAST_DECAY_WALKING if self.walk_matches >= WALK_CONFIRM_MATCHES
                 else COAST_DECAY)
        integrated_decay = (1 - decay ** elapsed) / (1 - decay)
        self._coast_dx += self.vx * integrated_decay
        self._coast_dy += self.vy * integrated_decay
        self._coast_log_scale += self.vlog_scale * integrated_decay
        self.vx *= decay ** elapsed
        self.vy *= decay ** elapsed
        self.vlog_scale *= decay ** elapsed
        self.last_update_frame = frame


def detect(canvas, bank, pool):
    """Walkers in one padded canvas: [(score, scale, x, y, pose)], x unpadded."""
    candidates = vr._coarse(canvas, bank, pool)
    hits = []
    for hit in pool.map(lambda c: vr._refine(canvas, bank, c), candidates):
        if hit is None or hit[0] < SEEN:
            continue
        score, scale, x, y, pose = hit
        x -= vr.PAD
        if all(abs(x - h[2]) > 10 or abs(np.log(scale / h[1])) > 0.08 for h in hits):
            hits.append((float(score), float(scale), float(x), float(y), pose))
    return hits


# Matched in the frame just before, her next pose is nearly always this one,
# the next, or their spin mirrors (+4, +5): 98.5% of 867 frame-to-frame matches
# in violetta-20261118-184548. Searching those 4 of the 8 poses halves the cost
# of a tracker update, so live play can track every frame (at ~120 ms an update
# it saw only ~60% of frames there, and a walker passing behind a standing one
# over a skipped frame took the wrong ID: round 3 clicked 2, truth 3). After any
# miss, or when lost or reacquiring, all 8 are searched.
LIKELY_POSE_STEPS = (0, 1, 4, 5)
# ...and walking she keeps her size: matched in the frame before, the wide
# (walking) window is searched at her size +-1 grid step (~4.6%, refine adds
# ~3%); bigger size changes are looked for in place only, as stepping.
STEADY_WALK_STEPS = 1


def _likely_poses(track, gap, reacquire):
    if gap != 1 or reacquire or track.lost or track.matched < 1:
        return None
    return tuple(sorted({(track.pose + step) % 8 for step in LIKELY_POSE_STEPS}))


LOCAL_MAX_X = 240           # px: cap on the local search half-width for a long-lost track
# Sizes are snapped to a fixed 1.5% grid so the scaled poses are built once and
# reused: a track's own size is a new number every frame, and scaling 8 poses
# per call made the local search as slow as the full one.
GRID_STEP = 1.015
GRID_MIN, GRID_MAX = 0.35, 2.0
LOCAL_GRID = 3              # local search size step, in grid steps (~4.6%)


def snap(scale):
    k = round(np.log(scale) / np.log(GRID_STEP))
    return float(np.clip(GRID_STEP ** k, GRID_MIN, GRID_MAX))


def grid_scales():
    lo = int(np.ceil(np.log(GRID_MIN) / np.log(GRID_STEP)))
    hi = int(np.floor(np.log(GRID_MAX) / np.log(GRID_STEP)))
    return [GRID_STEP ** k for k in range(lo, hi + 1)]


def _size_gate(track, frame, reacquire=False):
    gap = max(1, frame - track.seen_at)
    gate = max(GATE_SCALE_WALKING, GATE_SCALE_ONE_FRAME,
               min(GATE_SCALE_IN_PLACE, GATE_SCALE_PER_FRAME * gap))
    if reacquire and gap > 1:
        reacquire_gate = GATE_SCALE_REACQUIRE + GATE_SCALE_REACQUIRE_PER_FRAME * (gap - 2)
        gate = max(gate, min(GATE_SCALE_IN_PLACE, reacquire_gate))
    return gate


def _terminal_slot(x):
    slot = min(range(SLOT_COUNT),
               key=lambda s: abs(x - (slot_x(s) + SLOT_OFFSET)))
    if abs(x - (slot_x(slot) + SLOT_OFFSET)) > TERMINAL_SLOT_TOLERANCE:
        return None
    return slot


def _y_gate(track, gap):
    """px of pivot height error allowed, around _expected_y."""
    gate = GATE_Y + 2 * gap
    if track.matched < 2:
        gate += GATE_Y_UNSEEDED
    return gate


def _expected_y(track, scale):
    """Where her pivot should be at SCALE: walking keeps her row, stepping in
    place moves it down as she grows and up as she shrinks (002959 round 4: a
    0.59 track took a 0.43 'detection' whose pivot had dropped 27 px)."""
    return track.y + GATE_Y_PER_LOG_SCALE * np.log(scale / track.scale)


def _motion_gate(track, frame):
    gap = max(1, frame - track.seen_at)
    if not track.direction:
        return FAST_WALK_SPEED * gap + GATE_X   # _speed_ok decides beyond MAX_SPEED
    return min(SLOT_STEP / 2, GATE_X + MOTION_UNCERTAINTY_PER_FRAME * gap)


def _terminal_handoff(track, frame, hit):
    score, scale, x, _ = hit[:4]
    gap = max(1, frame - track.seen_at)
    if track.matched < 3 or score < TERMINAL_MIN_SCORE:
        return None
    if abs(x - track.x) > TERMINAL_MAX_JUMP:
        return None
    movement = x - track.x
    if abs(movement) > MAX_SPEED * gap + GATE_X:
        return None
    if track.direction and movement * track.direction < 0 and abs(movement) > GATE_X:
        return None
    if abs(np.log(scale / track.scale)) > GATE_SCALE_WALKING:
        return None
    if abs(x - track.predict(frame)) <= _motion_gate(track, frame):
        return None
    return _terminal_slot(x)


def _search_one(canvas, bank, scale, x0, y0, x_window, y_window, poses=None,
                canvas_squared=None):
    """Best (score, scale, x, y, pose) of the poses at SCALE near pivot (x0, y0); padded x."""
    if canvas_squared is None:
        canvas_squared = cv2.multiply(canvas, canvas)
    best = None
    scale = snap(scale)
    items = bank.get(scale)
    for pose in (range(len(items)) if poses is None else poses):
        template, mask, ox, oy = items[pose]
        top = int(round(y0 - y_window - oy))
        left = int(round(x0 - x_window - ox))
        bottom = top + template.shape[0] + 2 * y_window
        right = left + template.shape[1] + 2 * x_window
        top, left = max(0, top), max(0, left)
        bottom, right = min(canvas.shape[0], bottom), min(canvas.shape[1], right)
        match = vr._best_match(canvas[top:bottom, left:right],
                               canvas_squared[top:bottom, left:right], template, mask)
        if match is None:
            continue
        value, yy, xx = match
        if best is None or value > best[0]:
            best = (value, scale, left + xx + ox, top + yy + oy, pose)
    return best


def detect_near(canvas, bank, pool, tracks, frame, reacquire=False):
    """
    Walkers near where each track may be (the windows its gates allow):
    [(score, scale, x, y, pose)], x unpadded. detect() scans the whole stage at every
    size; 2026-11-15 live game: 280 ms a frame, so the background tracker saw ~3
    frames a second and slowed the solver loop too.

    * Walking, a Violetta keeps her size: the wide window (where her pace may
      take her) is searched at sizes within GATE_SCALE_WALKING only.
    * Bigger size changes happen stepping in or out on the spot: those sizes
      are searched only in a narrow window around where she was.
    * During reacquisition, expanded scales stay near a track's last confirmed
      position; only the normal walking-size range follows its motion prediction.
      This avoids taking a larger foreground Violetta as the occluded track.
    * Each hit's size is then refined with the pose that matched it only.
    """
    log_step = LOCAL_GRID * np.log(GRID_STEP)
    jobs = []
    for track in tracks:
        gap = max(1, frame - track.seen_at)
        track_reacquire = reacquire or gap > 1 or track.missed_frames >= 2
        walker = abs(np.log(track.scale)) > STANDING_LOG_SCALE
        per_frame = FAST_WALK_SPEED if walker else GATE_X_PER_FRAME
        x_window = int(min(LOCAL_MAX_X, GATE_X + per_frame * gap))
        y_window = int(_y_gate(track, gap))
        walking = int(np.ceil(GATE_SCALE_WALKING / log_step))
        steps = max(walking, int(np.ceil(_size_gate(track, frame, track_reacquire) / log_step)))
        x_pred = track.predict(frame) + vr.PAD
        x_here = track.x + vr.PAD
        in_place = int(IN_PLACE_PX + 0.5 * gap + 4)
        poses = _likely_poses(track, gap, track_reacquire)
        seen = set()
        for k in range(-steps, steps + 1):
            if abs(k) > walking + 1 and (abs(k) - walking) % 2 == 0:
                continue    # big in-place size changes: every 2nd size (refine fills in)
            if track.lost and k % 2:
                continue    # lost: the wide search covers most of the stage, every 2nd size
            scale = snap(track.scale * GRID_STEP ** (LOCAL_GRID * k))
            if abs(k) <= (walking if poses is None else STEADY_WALK_STEPS):
                # Around where she was, both ways: she may have turned round
                centre = x_here if walker else x_pred
                job = (scale, centre, track.y, x_window, y_window, poses)
            else:
                # Stepping in place: the pivot drops as she grows
                job = (scale, x_here, int(round(_expected_y(track, scale))), in_place,
                       y_window, poses)
            if job not in seen:
                seen.add(job)
                jobs.append(job)
        if track.lost:
            # Out of sight she may have kept shrinking or growing on her row
            lost_steps = int(np.ceil(LOST_SEARCH_LOG_SCALE / log_step))
            for k in range(-lost_steps, lost_steps + 1, 2):
                if abs(k) <= walking:
                    continue
                scale = snap(track.scale * GRID_STEP ** (LOCAL_GRID * k))
                job = (scale, x_pred, track.y, x_window, int(_y_gate(track, gap)) + 10)
                if job not in seen:
                    seen.add(job)
                    jobs.append(job)
            # Wherever she went, she ends standing in a slot: look there too
            for slot in range(SLOT_COUNT):
                anchor = slot_x(slot) + SLOT_OFFSET + vr.PAD
                for scale in LOST_STAND_SCALES:
                    job = (snap(scale), anchor, STAND_PIVOT_Y, TERMINAL_X_WINDOW, LOST_GATE_Y // 2)
                    if job not in seen:
                        seen.add(job)
                        jobs.append(job)
        if track.matched >= 3:
            current_slot = min(
                range(SLOT_COUNT),
                key=lambda s: abs(x_pred - (slot_x(s) + SLOT_OFFSET + vr.PAD)))
            if track.vx > 1:
                terminal_slots = (current_slot + 1,)
            elif track.vx < -1:
                terminal_slots = (current_slot - 1,)
            else:
                terminal_slots = (current_slot - 1, current_slot + 1)
            for slot in terminal_slots:
                if not 0 <= slot < SLOT_COUNT:
                    continue
                anchor = slot_x(slot) + SLOT_OFFSET + vr.PAD
                if abs(anchor - x_pred) > TERMINAL_MAX_JUMP:
                    continue
                for k in (-2, 0, 2):
                    scale = snap(track.scale * GRID_STEP ** (LOCAL_GRID * k))
                    job = (scale, anchor, track.y, TERMINAL_X_WINDOW, y_window, poses)
                    if job not in seen:
                        seen.add(job)
                        jobs.append(job)
    jobs = list(dict.fromkeys(jobs))   # lost tracks share their slot searches
    squared = cv2.multiply(canvas, canvas)
    found = [h for h in pool.map(
                 lambda j: _search_one(canvas, bank, *j, canvas_squared=squared), jobs)
             if h is not None and h[0] >= TRACK_CANDIDATE_SEEN]
    found.sort(reverse=True)
    kept = []
    for hit in found:       # one per place and size
        if all(abs(hit[2] - k[2]) > 10 or abs(np.log(hit[1] / k[1])) > 0.08 for k in kept):
            kept.append(hit)

    def refine(hit):
        best = hit
        for k in (-2, -1, 1, 2):        # one grid step at a time, the matched pose only
            better = _search_one(canvas, bank, hit[1] * GRID_STEP ** k, hit[2], hit[3], 3, 3,
                                 poses=(hit[4],), canvas_squared=squared)
            if better is not None and better[0] > best[0]:
                best = better
        score, scale, x, y, pose = best
        return float(score), float(scale), float(x - vr.PAD), float(y), pose

    refined = sorted((hit for hit in pool.map(refine, kept) if hit[0] >= TRACK_SEEN),
                     reverse=True)
    # Refining can bring two coarse peaks onto the same Violetta (190834 round
    # 3: yellow was found at 0.84 and 0.79 a few px apart, and the missing blue
    # track took the second copy and rode yellow from then on): keep one
    unique = []
    for hit in refined:
        if not any(_same_detection(hit, other) for other in unique):
            unique.append(hit)
    return unique


def _cost(track, frame, hit, reacquire=False):
    score, scale, x, y = hit[:4]
    gap = max(1, frame - track.seen_at)
    dx = _path_miss(track, frame, x)
    moved = abs(x - track.x)
    # Keep observed row/scale as association gates; extrapolating them caused
    # missed tracks to drift onto adjacent Violettas in the saved replay.
    ds = abs(np.log(scale / track.scale))
    dy = abs(y - track.y)
    in_place = moved <= IN_PLACE_PX + 0.5 * gap
    dy_error = abs(y - _expected_y(track, scale)) if in_place else dy
    track_reacquire = reacquire or gap > 1 or track.missed_frames >= 2
    if in_place and gap > 1 and _hidden_mid_walk(track):
        # She vanished while walking: hidden behind a front walker, still
        # walking at her size. A size jump where she was last seen is someone
        # else (20261119-221411 round 3: hidden at 184 by a 1.59 walker, the
        # track took a 0.35 match there and sat on it; she came out at 88
        # still walking left, to slot 1)
        size_gate = GATE_SCALE_REACQUIRE_MOVING
    elif in_place:
        size_gate = _size_gate(track, frame, track_reacquire)
    elif track_reacquire:
        size_gate = min(
            GATE_SCALE_IN_PLACE,
            GATE_SCALE_REACQUIRE_MOVING
            + 0.035 * max(0, gap - 2))
    else:
        size_gate = GATE_SCALE_WALKING
    terminal_slot = _terminal_handoff(track, frame, hit)
    movement = x - track.x
    if terminal_slot is None and not _speed_ok(track, gap, movement, ds, dy):
        return None
    own_row = _on_own_row(track, gap, ds, dy)
    if (track.direction and movement * track.direction < 0 and not own_row
            and abs(movement) > GATE_X and score < TERMINAL_MIN_SCORE):
        return None
    if ((dx > _motion_gate(track, frame) and terminal_slot is None and not own_row)
            or dy_error > _y_gate(track, gap) or ds > size_gate):
        return None
    scale_cost = COST_SCALE
    if track_reacquire and moved > IN_PLACE_PX + 0.5 * gap:
        scale_cost = COST_SCALE_REACQUIRE
    return dx + scale_cost * ds + COST_Y * dy_error - 20 * (score - SEEN)


def _hidden_mid_walk(track):
    """True if she was walking (a confirmed walk, still at pace) when last seen."""
    return (track.walk_matches >= WALK_CONFIRM_MATCHES
            and abs(track.vx) >= WALK_MIN_SPEED)


STAGE_X = (slot_x(0) + SLOT_OFFSET - 30, slot_x(SLOT_COUNT - 1) + SLOT_OFFSET + 30)


def _path_miss(track, frame, x):
    """px from X to where she can be: anywhere from her last place up to where
    her pace takes her, since a walker may stop short at a slot (002959 round 4:
    the front walker stopped at slot 3 while her prediction walked on, and she
    was dropped as she stood there stepping out)."""
    predicted = float(np.clip(track.predict(frame), *STAGE_X))
    lo, hi = sorted((track.x, predicted))
    if lo <= x <= hi:
        return 0.0
    return min(abs(x - lo), abs(x - hi))


def _on_own_row(track, gap, ds, dy):
    """A walking match: same size and height as her row, away from standing size."""
    return (ds <= FAST_GATE_SCALE and dy <= FAST_GATE_Y + gap
            and abs(np.log(track.scale)) > STANDING_LOG_SCALE)


def _speed_ok(track, gap, movement, ds, dy):
    """Up to MAX_SPEED anything the other gates allow; up to FAST_WALK_SPEED
    only on her own row (same size and height: a walk, not a hop to another)."""
    if abs(movement) <= (MAX_SPEED + MOTION_SPEED_TOLERANCE) * gap + GATE_X:
        return True
    return (abs(movement) <= (FAST_WALK_SPEED + MOTION_SPEED_TOLERANCE) * gap + GATE_X
            and _on_own_row(track, gap, ds, dy))


def _lost_cost(track, frame, hit):
    """Cost of re-identifying a LOST track with a detection nobody else claims."""
    score, scale, x, y = hit[:4]
    if score < LOST_SEEN:
        return None
    gap = max(1, frame - track.seen_at)
    if abs(x - track.x) > (FAST_WALK_SPEED + MOTION_SPEED_TOLERANCE) * gap + GATE_X:
        return None
    ds = abs(np.log(scale / track.scale))
    standing = _terminal_slot(x) is not None and min(
        abs(np.log(scale / s)) for s in STANDING_SCALES) <= 0.05
    if ds > GATE_SCALE_IN_PLACE and not standing:
        return None
    if abs(y - track.y) > LOST_GATE_Y:
        return None
    # She comes back at the size she left at, unless she is already standing
    scale_cost = COST_SCALE if standing else COST_SCALE_REACQUIRE
    return _path_miss(track, frame, x) + scale_cost * ds - 20 * (score - SEEN)


def same_row(first, second):
    return (abs(np.log(first.scale / second.scale)) <= SAME_ROW_LOG_SCALE
            and abs(first.y - second.y) <= SAME_ROW_Y)


def order_kept(tracks, hits, pick):
    """False if PICK re-identifies only part of a same-row group of tracks, or
    swaps the left-to-right order of two of them."""
    for a in range(len(tracks)):
        for b in range(a + 1, len(tracks)):
            if not same_row(tracks[a], tracks[b]):
                continue
            if (pick[a] is None) != (pick[b] is None):
                return False        # wait until both are visible again
            if pick[a] is None:
                continue
            before = tracks[a].x - tracks[b].x
            after = hits[pick[a]][2] - hits[pick[b]][2]
            if before * after < 0:
                return False
    return True


def one_id_each(hits, pick):
    """False if two tracks would take detections of the same Violetta: one
    that is seen belongs to one ID, and the others stay missing."""
    chosen = [hits[h] for h in pick if h is not None]
    return not any(_same_detection(a, b)
                   for i, a in enumerate(chosen) for b in chosen[i + 1:])


def associate(tracks, frame, hits, reacquire=False, cost=None, valid=None,
              miss_cost=MISS_COST):
    """Return only stable matches across near-equal maximum-cardinality assignments."""
    if cost is None:
        costs = [[_cost(t, frame, h, reacquire) for h in hits] for t in tracks]
    else:
        costs = [[cost(t, frame, h) for h in hits] for t in tracks]
    candidates = []
    options = list(range(len(hits))) + [None] * len(tracks)
    for pick in set(itertools.permutations(options, len(tracks))):
        total, count, ok = 0.0, 0, True
        for k, h in enumerate(pick):
            if h is None:
                continue
            if costs[k][h] is None:
                ok = False
                break
            total += costs[k][h]
            count += 1
        if ok and (valid is None or valid(pick)):
            candidates.append((count, total, pick))
    if not candidates:
        return {}

    candidates = [(count, total + miss_cost * (len(tracks) - count), pick)
                  for count, total, pick in candidates]
    best_cost = min(total for _, total, _ in candidates)
    near_best = [pick for count, total, pick in candidates
                 if total <= best_cost + ASSOCIATION_AMBIGUITY_MARGIN]
    stable = {}
    for track_index in range(len(tracks)):
        choices = {pick[track_index] for pick in near_best}
        if len(choices) == 1:
            hit_index = next(iter(choices))
            if hit_index is not None:
                stable[track_index] = hit_index
    # Three strong IDs plus one unique leftover lets the missing track recover
    # without following a stale prediction onto a different Violetta.
    if cost is not None or len(tracks) != SLOT_COUNT:
        return stable
    missing_tracks = [i for i in range(len(tracks)) if i not in stable]
    if len(missing_tracks) != 1 or len(stable) != len(tracks) - 1:
        return stable
    if any(hits[i][0] < SEEN for i in stable.values()):
        return stable

    missing = missing_tracks[0]
    track = tracks[missing]
    if track.matched < 3 or track.missed_frames < RESIDUAL_REACQUIRE_MIN_MISSES:
        return stable

    used = set(stable.values())
    remaining = []
    for hit_index, hit in enumerate(hits):
        if hit_index in used or hit[0] < TRACK_SEEN:
            continue
        if any(_same_detection(hit, hits[known]) for known in used):
            continue
        cost = _residual_reacquire_cost(track, frame, hit)
        if cost is not None:
            remaining.append((hit_index, hit, cost))
    clusters = []
    for candidate in remaining:
        if not any(_same_detection(candidate[1], cluster[0][1]) for cluster in clusters):
            clusters.append([candidate])
        else:
            next(cluster for cluster in clusters
                 if _same_detection(candidate[1], cluster[0][1])).append(candidate)
    if len(clusters) == 1:
        hit_index, _, _ = min(clusters[0], key=lambda candidate: candidate[2])
        stable[missing] = hit_index
    return stable


def _same_detection(first, second):
    return (abs(first[2] - second[2]) <= 10
            and abs(np.log(first[1] / second[1])) <= 0.08
            and abs(first[3] - second[3]) <= GATE_Y)


def _residual_reacquire_cost(track, frame, hit):
    score, scale, x, y = hit[:4]
    gap = max(1, frame - track.seen_at)
    movement = x - track.x
    if not _speed_ok(track, gap, movement, abs(np.log(scale / track.scale)),
                     abs(y - track.y)):
        return None
    if track.direction and movement * track.direction < 0 and abs(movement) > GATE_X:
        return None
    if abs(y - track.y) > _y_gate(track, gap):
        return None
    scale_gate = min(
        GATE_SCALE_IN_PLACE,
        GATE_SCALE_REACQUIRE + GATE_SCALE_REACQUIRE_PER_FRAME * max(0, gap - 2))
    scale_delta = abs(np.log(scale / track.scale))
    if scale_delta > scale_gate:
        return None
    return (abs(x - track.predict(frame)) + COST_SCALE_REACQUIRE * scale_delta
            + COST_Y * abs(y - track.y) - 20 * (score - SEEN))


# A close pass is only a real identity risk if the two could have swapped
# there. The spotlight version (20261119-195802 round 1) sends a decoy straight
# across her at the same size; she walked on, the decoy walked on, and the
# tracker kept them apart - but the closeness alone marked her track untrusted
# and the solver answered her start slot, where the decoy ended. A swap shows
# in the motion: one of them turns round at the meeting, or one stops just as
# the other sets off (her pace handed to the other ID). Otherwise both kept
# their motion through it, and continuity already decided who is who.
PASS_WINDOW_FRAMES = 6      # frames either side of the meeting to read each one's motion
PASS_MIN_POINTS = 2         # matches needed on each side, else it stays ambiguous


def _motion_sign(path, first_frame, last_frame):
    """-1/0/+1: which way the matches in [first_frame, last_frame] went, or None."""
    points = [p for p in path if first_frame <= p[0] <= last_frame]
    if len(points) < PASS_MIN_POINTS:
        return None
    frames = points[-1][0] - points[0][0]
    if frames <= 0:
        return None
    speed = (points[-1][1] - points[0][1]) / frames
    return 0 if abs(speed) < WALK_MIN_SPEED else (1 if speed > 0 else -1)


def _passed_through(first, second, meet_start, meet_end):
    """True if both kept their motion through the meeting (no swap is possible)."""
    signs = []
    for track in (first, second):
        before = _motion_sign(track.path, meet_start - PASS_WINDOW_FRAMES, meet_start)
        after = _motion_sign(track.path, meet_end, meet_end + PASS_WINDOW_FRAMES)
        if before is None or after is None:
            return False
        if before * after < 0:
            return False        # turned round where they met
        signs.append((before, after))
    (b1, a1), (b2, a2) = signs
    handed_off = ((b1 and not a1 and not b2 and a2)
                  or (b2 and not a2 and not b1 and a1))
    return not handed_off


def _identity_ambiguities(tracks):
    """Mark tracks that came too close to another ID to keep identity certain."""
    ambiguous = [False] * len(tracks)
    paths = [{int(point[0]): point for point in track.path} for track in tracks]
    distance_squared = IDENTITY_AMBIGUITY_DISTANCE ** 2
    for first in range(len(tracks)):
        for second in range(first + 1, len(tracks)):
            met = []
            for frame, point in paths[first].items():
                for other_frame in range(
                        frame - IDENTITY_AMBIGUITY_FRAME_GAP,
                        frame + IDENTITY_AMBIGUITY_FRAME_GAP + 1):
                    other = paths[second].get(other_frame)
                    if other is None:
                        continue
                    dx, dy = point[1] - other[1], point[3] - other[3]
                    if (dx * dx + dy * dy <= distance_squared
                            and abs(np.log(point[2] / other[2]))
                            <= IDENTITY_AMBIGUITY_LOG_SCALE):
                        met.append(frame)
                        break
            # Each separate meeting (frames close together) is judged on its own
            meetings, met = [], sorted(met)
            for frame in met:
                if meetings and frame - meetings[-1][1] <= 2 * IDENTITY_AMBIGUITY_FRAME_GAP:
                    meetings[-1][1] = frame
                else:
                    meetings.append([frame, frame])
            if any(not _passed_through(tracks[first], tracks[second], start, end)
                   for start, end in meetings):
                ambiguous[first] = ambiguous[second] = True
    return ambiguous


class ViolettaTracker:
    """Feed it canvases (local-contrast window canvases) with their frame index."""

    def __init__(self, sprites):
        self.bank = vr._SizeBank(sprites)
        for scale in vr.SCALES:
            self.bank.get(scale)
        for scale in grid_scales():     # every size the local search can ask for
            self.bank.get(scale)
        self.pool, _ = vr._pools()
        self.tracks = []
        self.frames = 0
        self.last_frame = 0
        self.ms = 0.0
        self.local = True       # search only near each track (detect_near)
        self._all_missed_last_update = False

    def start(self, frame):
        """A new round: the four Violettas stand in their slots."""
        self.tracks = [Track(s, frame) for s in range(SLOT_COUNT)]
        self.frames, self.ms = 0, 0.0
        self.last_frame = frame
        self._all_missed_last_update = False

    def update(self, frame, canvas):
        if not self.tracks:
            self.start(frame)
        started = time.perf_counter()
        reacquire = self._all_missed_last_update
        if self.local:
            hits = detect_near(vr.pad(canvas), self.bank, self.pool, self.tracks, frame,
                               reacquire=reacquire)
        else:
            hits = detect(vr.pad(canvas), self.bank, self.pool)
        taken = self._associate_with_lost(frame, hits, reacquire)
        for k, track in enumerate(self.tracks):
            if k in taken:
                hit = hits[taken[k]]
                track.take(frame, hit, _terminal_handoff(track, frame, hit))
            else:
                track.wait(frame)
                # Not during a lights-out (nobody matched): everyone is missing then
                if taken and track.missed_frames >= LOST_AFTER_FRAMES:
                    track.lost = True
        if taken:
            for track in self.tracks:
                if (not track.lost and track.missed_frames >= GROUP_MISSED_FRAMES
                        and any(other.lost and same_row(track, other)
                                for other in self.tracks if other is not track)):
                    track.lost = True   # wait with her row-mate
        self._all_missed_last_update = not taken
        self.frames += 1
        self.last_frame = frame
        self.ms += 1000 * (time.perf_counter() - started)
        return hits

    def _associate_with_lost(self, frame, hits, reacquire):
        """
        {track index: hit index}. Tracks still being followed are matched as
        usual, without the lost ones (a lost track parked on its last place
        must not make a walker arriving there ambiguous). Then each lost track
        waits for a detection that no followed track claims or could claim,
        and takes it only if that choice is unambiguous.
        """
        active = [k for k, t in enumerate(self.tracks) if not t.lost]
        lost = [k for k, t in enumerate(self.tracks) if t.lost]
        taken = {active[i]: h for i, h in associate(
            [self.tracks[k] for k in active], frame, hits, reacquire=reacquire,
            valid=lambda pick: one_id_each(hits, pick)).items()}
        if not lost:
            return taken
        claimed = [hits[h] for h in taken.values()]
        waiting = [k for k in active if k not in taken]
        free = [h for h, hit in enumerate(hits)
                if h not in taken.values()
                and not any(_same_detection(hit, other) for other in claimed)
                and all(_cost(self.tracks[k], frame, hit, reacquire) is None
                        for k in waiting)]
        if not free:
            return taken
        lost_tracks = [self.tracks[k] for k in lost]
        free_hits = [hits[h] for h in free]
        # Where a lost Violetta went while hidden says little about which one
        # is back, so every lost track that can be matched is (no miss cost)
        found = associate(lost_tracks, frame, free_hits, cost=_lost_cost,
                          valid=lambda pick: (order_kept(lost_tracks, free_hits, pick)
                                              and one_id_each(free_hits, pick)),
                          miss_cost=LOST_MISS_COST)
        for i, h in found.items():
            taken[lost[i]] = free[h]
        return taken

    def snapshot(self):
        """Current per-track positions and match quality for the live overlay."""
        tracks = []
        ambiguous = _identity_ambiguities(self.tracks)
        for track, identity_ambiguous in zip(self.tracks, ambiguous):
            predicted_x, predicted_y, predicted_scale = track.predict_state(self.last_frame)
            max_step = max(
                (abs(b[1] - a[1]) for a, b in zip(track.path, track.path[1:])),
                default=0.0)
            max_speed = max(
                (abs(b[1] - a[1]) / max(1, b[0] - a[0])
                 for a, b in zip(track.path, track.path[1:])),
                default=0.0)
            tracks.append({
                'id': track.slot + 1,
                'slot': track.slot,
                'x': track.x,
                'y': track.y,
                'scale': track.scale,
                'pose': track.pose,
                'vy': track.vy,
                'vlog_scale': track.vlog_scale,
                'direction': track.direction,
                'score': float(track.path[-1][4]) if track.matched else None,
                'matched': track.matched,
                'missed_frames': track.missed_frames,
                'all_missed_last_update': self._all_missed_last_update,
                'seen': track.matched / max(1, self.frames),
                'max_step': float(max_step),
                'max_speed': float(max_speed),
                'identity_ambiguous': identity_ambiguous,
                'terminal_confirmed': track.terminal_confirmed,
                'lost': track.lost,
                'age': max(0, self.last_frame - track.seen_at),
                'vx': track.vx,
                'predicted_x': predicted_x,
                'predicted_y': predicted_y,
                'predicted_scale': predicted_scale,
            })
        return {
            'frames': self.frames,
            'last_frame': self.last_frame,
            'threshold': SEEN,
            'tracks': tracks,
        }

    def result(self):
        """({start slot: end slot}, report). The end slots are the tracks' last
        places matched to slots one-to-one."""
        ends = [t.x for t in self.tracks]
        lost = [t.lost for t in self.tracks]
        end_slots, error = end_slots_with_lost([x - SLOT_OFFSET for x in ends], lost)
        mapping = {t.slot: e for t, e in zip(self.tracks, end_slots)}
        seen = [t.matched / max(1, self.frames) for t in self.tracks]
        # Per-track motion and overlap evidence for identity trust in the solver.
        max_step = [max((abs(b[1] - a[1]) for a, b in zip(t.path, t.path[1:])), default=0.0)
                    for t in self.tracks]
        max_speed = [
            max((abs(b[1] - a[1]) / max(1, b[0] - a[0])
                 for a, b in zip(t.path, t.path[1:])), default=0.0)
            for t in self.tracks
        ]
        ambiguous = _identity_ambiguities(self.tracks)
        end_error = [abs(t.x - SLOT_OFFSET - slot_x(e)) for t, e in zip(self.tracks, end_slots)]
        nearest_end_slot = [
            min(range(SLOT_COUNT),
                key=lambda slot: abs(t.x - SLOT_OFFSET - slot_x(slot)))
            for t in self.tracks
        ]
        nearest_end_error = [
            abs(t.x - SLOT_OFFSET - slot_x(slot))
            for t, slot in zip(self.tracks, nearest_end_slot)
        ]
        report = {
            'end_x': [round(x) for x in ends],
            'worst_slot_error': round(float(error), 1),
            'seen': [round(s, 2) for s in seen],
            'max_step': [round(float(m), 1) for m in max_step],
            'max_speed': [round(float(speed), 1) for speed in max_speed],
            'ambiguous': ambiguous,
            'end_error': [round(float(e), 1) for e in end_error],
            'nearest_end_slot': nearest_end_slot,
            'nearest_end_error': [round(float(e), 1) for e in nearest_end_error],
            'terminal_confirmed': [t.terminal_confirmed for t in self.tracks],
            'lost': lost,
            'frames': self.frames,
            'ms_per_frame': round(self.ms / max(1, self.frames)),
            'consistent': bool(error < SLOT_STEP / 3 and min(seen) >= 0.5),
            'layout': 'tracked',
        }
        return mapping, report


class BackgroundTracker:
    """Runs a ViolettaTracker on its own thread with the newest canvas offered,
    so the solver's frame loop never waits for it."""

    def __init__(self, sprites):
        self.tracker = ViolettaTracker(sprites)
        self._lock = threading.Lock()
        self._latest = None
        self._wake = threading.Event()
        self._busy = False
        self._generation = 0
        self._snapshot = self.tracker.snapshot()
        threading.Thread(target=self._main, name='violetta tracker', daemon=True).start()

    def start(self, frame):
        with self._lock:
            self._generation += 1
            self._latest = None
            self.tracker.start(frame)
            self._snapshot = self.tracker.snapshot()

    def offer(self, frame, canvas):
        with self._lock:
            self._latest = (self._generation, frame, canvas)
        self._wake.set()

    def _main(self):
        while True:
            self._wake.wait()
            self._wake.clear()
            with self._lock:
                job, self._latest = self._latest, None
                if job is None or job[0] != self._generation:
                    continue
                self._busy = True
            try:
                self.tracker.update(job[1], np.asarray(job[2], np.float32))
            except Exception as error:
                print(f'[VIOLETTA] tracker failed on a frame: {error!r}')
            finally:
                snapshot = self.tracker.snapshot()
                with self._lock:
                    self._snapshot = snapshot
                    self._busy = False

    def snapshot(self):
        with self._lock:
            return self._snapshot

    def result(self, wait=0.3):
        """The tracker's reading; waits up to WAIT s for a frame in progress."""
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            with self._lock:
                if not self._busy:
                    break
            time.sleep(0.01)
        with self._lock:
            return self.tracker.result()


def source_stamp():
    """Newest modification time of the Violetta code (src/detection/violetta_*.py)."""
    import glob
    import os
    folder = os.path.dirname(os.path.abspath(__file__))
    return max((os.path.getmtime(path)
                for path in glob.glob(os.path.join(folder, 'violetta_*.py'))), default=0.0)


class ProcessTracker:
    """
    Like BackgroundTracker, but the tracker runs in its own process.

    2026-11-15: on a thread, the tracker and the solver's frame loop take turns
    on Python's GIL: the loop fell from 17 to 8.6 fps while tracking (the live
    game ran at 7-8 fps and missed most of each shuffle). A process has its own.

    Started with `python -m src.detection.violetta_tracker --serve` (not
    multiprocessing, which would re-run Auto Maple's main.py in the child).
    Messages are pickled over its stdin/stdout. One server is kept for the
    whole session (starting it costs ~1-2 s), see shared().
    """

    _shared = None

    @classmethod
    def shared(cls):
        # 2026-11-19 22:56: the tracker was fixed on disk at 22:25, but this
        # process had been started by the 22:14 game and kept the old code; it
        # lost round 1 (the cursor hid slot 4) and all four answers were wrong.
        # A replay with the new code read every round right. Restart it when
        # its code changes.
        if cls._shared is not None and not cls._shared._dead() \
                and cls._shared._source_stamp != source_stamp():
            print('[VIOLETTA] Tracker code changed on disk since its process started; '
                  'restarting the tracker process.')
            cls._shared.close()
            cls._shared = None
        if cls._shared is None or cls._shared._dead():
            cls._shared = cls()
        return cls._shared

    def close(self):
        try:
            self._process.kill()
            self._process.wait(timeout=2)
        except Exception:
            pass

    def __init__(self):
        import os
        import subprocess
        import sys
        self._source_stamp = source_stamp()
        self._process = subprocess.Popen(
            [sys.executable, '-m', 'src.detection.violetta_tracker', '--serve'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=os.getcwd(),
            # Above normal (it sleeps between games): heavy background work on
            # the PC must not slow the live tracking (see mini_games._Priority)
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
            | getattr(subprocess, 'ABOVE_NORMAL_PRIORITY_CLASS', 0))
        self._send_lock = threading.Lock()
        self._replies = {}
        self._reply_ready = threading.Condition()
        self._asked = 0
        self._snapshot_lock = threading.Lock()
        self._snapshot = None
        threading.Thread(target=self._read_replies, name='violetta tracker replies',
                         daemon=True).start()

    def _dead(self):
        return self._process.poll() is not None

    def _send(self, message):
        import pickle
        data = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
        with self._send_lock:
            self._process.stdin.write(len(data).to_bytes(4, 'little') + data)
            self._process.stdin.flush()

    def _read_replies(self):
        import pickle
        stream = self._process.stdout
        while True:
            size = stream.read(4)
            if len(size) < 4:
                return
            ask, reply = pickle.loads(stream.read(int.from_bytes(size, 'little')))
            if ask == 0:
                with self._snapshot_lock:
                    self._snapshot = reply
                continue
            with self._reply_ready:
                self._replies[ask] = reply
                self._reply_ready.notify_all()

    def snapshot(self):
        with self._snapshot_lock:
            return self._snapshot

    def start(self, frame):
        self._send(('start', frame))

    def offer(self, frame, canvas):
        self._send(('frame', frame, np.asarray(canvas, np.float16)))

    def result(self, wait=0.5):
        """The tracker's reading once it finishes the frame in hand (up to WAIT s more)."""
        self._asked += 1
        ask = self._asked
        self._send(('result', ask))
        deadline = time.monotonic() + max(wait, 0.05) + 0.5
        with self._reply_ready:
            while ask not in self._replies:
                left = deadline - time.monotonic()
                if left <= 0 or self._dead():
                    raise RuntimeError('the tracker process did not answer')
                self._reply_ready.wait(left)
            reply = self._replies.pop(ask)
        if isinstance(reply, Exception):
            raise reply
        return reply


# 2026-11-18 (violetta-20261118-021248): taking only the newest frame, the
# live tracker saw about every 2nd frame on a slow night. Walkers move ~30 px
# a frame, so a skipped frame or two put a box on a passing Violetta and three
# rounds were read wrong (a replay of every frame read the same recording
# right). Every frame is queued now and tracked in order; the solver waits
# for the queue to empty before it reads the answer.
SERVE_QUEUE_MAX = 400       # frames waiting at most (the oldest go first past this)


def _serve():
    """The ProcessTracker side: every frame of a round, in order."""
    import pickle
    import sys
    from src.detection.violetta_solver import Sprites

    import os
    tracker = ViolettaTracker(Sprites())
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    # Tests only: extra seconds per frame, to act like a slow PC (unset in play)
    test_delay = float(os.environ.get('AUTO_MAPLE_TRACKER_DELAY') or 0)
    lock = threading.Lock()
    wake = threading.Event()
    state = {'jobs': [], 'generation': 0, 'asks': [], 'closed': False}

    def read():
        while True:
            size = stdin.read(4)
            if len(size) < 4:
                break
            message = pickle.loads(stdin.read(int.from_bytes(size, 'little')))
            with lock:
                if message[0] == 'start':
                    state['generation'] += 1
                    state['jobs'] = [('start', message[1])]     # a new round: old frames go
                elif message[0] == 'frame':
                    jobs = state['jobs']
                    jobs.append(message)
                    if len(jobs) > SERVE_QUEUE_MAX:
                        first = 1 if jobs[0][0] == 'start' else 0
                        del jobs[first]                 # far behind: drop the oldest frame
                else:
                    state['asks'].append(message[1])
            wake.set()
        with lock:
            state['closed'] = True
        wake.set()

    threading.Thread(target=read, daemon=True).start()
    while True:
        wake.wait()
        wake.clear()
        while True:
            with lock:
                job = state['jobs'].pop(0) if state['jobs'] else None
                # (a result is answered once every frame sent before it is tracked)
                asks = []
                if not state['jobs']:
                    asks, state['asks'] = state['asks'], []
                closed = state['closed']
            if job is not None:
                try:
                    if job[0] == 'start':
                        tracker.start(job[1])
                    else:
                        tracker.update(job[1], np.asarray(job[2], np.float32))
                        if test_delay:
                            time.sleep(test_delay)
                except Exception as error:
                    print(f'[VIOLETTA] tracker failed on a frame: {error!r}', file=sys.stderr)
                snapshot = pickle.dumps((0, tracker.snapshot()), protocol=pickle.HIGHEST_PROTOCOL)
                stdout.write(len(snapshot).to_bytes(4, 'little') + snapshot)
                stdout.flush()
            for ask in asks:
                try:
                    reply = tracker.result()
                except Exception as error:
                    reply = error
                data = pickle.dumps((ask, reply), protocol=pickle.HIGHEST_PROTOCOL)
                stdout.write(len(data).to_bytes(4, 'little') + data)
                stdout.flush()
            if closed:
                return
            if job is None and not asks:
                break


if __name__ == '__main__':
    import sys
    if '--serve' in sys.argv:
        _serve()
