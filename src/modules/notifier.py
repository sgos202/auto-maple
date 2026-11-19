"""A module for detecting and notifying the user of dangerous in-game events."""

from src.common import config, utils
from src.detection.mini_game_alert import PopUpAnchorDetector, RisingEdgeAlert
from src.modules.mini_games import AUTO_SOLVE, GAMES, MiniGameRunner, TALLY
from src.modules.mini_game_alarm import ALARM
import glob
import time
import cv2
import threading
import numpy as np
import keyboard as kb
from src.routine.components import Point


# A rune's symbol on the minimap
RUNE_RANGES = (
    ((141, 148, 245), (146, 158, 255)),
)
rune_filtered = utils.filter_color(cv2.imread('assets/rune_template.png'), RUNE_RANGES)
RUNE_TEMPLATE = cv2.cvtColor(rune_filtered, cv2.COLOR_BGR2GRAY)

# Other players' symbols on the minimap
OTHER_RANGES = (
    ((0, 245, 215), (10, 255, 255)),
)
other_filtered = utils.filter_color(cv2.imread('assets/other_template.png'), OTHER_RANGES)
OTHER_TEMPLATE = cv2.cvtColor(other_filtered, cv2.COLOR_BGR2GRAY)

# The Elite Boss's warning sign
ELITE_TEMPLATE = cv2.imread('assets/elite_template.jpg', 0)

# Mini-game pop-ups that need the user's attention. Each is detected from small,
# distinctive pieces ("anchors") cropped from its reference image (assets/pop-up-N.png)
# and saved as assets/pop_up_anchors/<prefix>_*.png. Any one anchor matching counts,
# so the real pop-up can differ from the reference. Drop in more crops to add anchors.
# Where a solver exists it starts automatically (see src/modules/mini_games.py).
#
# Every pop-up rings the same alarm (the shape game's siren, looping) and shows a
# flashing button beside the game until it is clicked or the solver clears the
# game (mini_games.CLEARED_RESULTS); see src/modules/mini_game_alarm.py.
# WHITE_TEXT: the pop-up is drawn see-through over the game, so only its white
# text is matched (the picture behind it changes).
#   (name, prefix, white text, threshold)
POP_UPS = (
    ('Lie Detector (fading shape) mini-game', 'pop-up-1', False, 0.83),  # gameplay <= 0.77
    ('Violetta mini-game', 'pop-up-2', False, 0.83),
    ('Question/answer ("Answer within 10 seconds")', 'pop-up-3', True, 0.70),
    # pop-up-3: pop-up >= 0.80, gameplay <= 0.55 in testing
)
# A pop-up is ignored while any of its rivals is also on screen. Since 2026-11-19 the
# Violetta notice ("Lie Detector Guide", blue box) shows the same "W" guide NPC as the
# shape game's, so pop-up-1_npc matched it (0.97) and the shape solver ran instead.
# pop-up-2_v2_portrait catches that new notice. pop-up-2_mushroom was retired then
# (assets/pop_up_anchors/retired): it set off 5 of 5 false alarms on Explorer/viewer
# windows showing the sprites; pop-up-2_bubble now covers the variant only it caught.
POP_UP_OVERRIDES = {'pop-up-1': ('pop-up-2',)}
POP_UP_SCALES = tuple(round(0.8 + 0.05 * i, 2) for i in range(10))     # 0.80 - 1.25
POP_UP_CHECK_SECONDS = 0.5      # How often to scan the screen for pop-ups


class Notifier:
    def __init__(self):
        """Initializes this Notifier object's main thread."""

        self.ready = False
        self.thread = threading.Thread(target=self._main)
        self.thread.daemon = True

        self.room_change_threshold = 0.9
        self.rune_alert_delay = 270         # 4.5 minutes
        self.pop_ups = []
        for name, prefix, white_text, threshold in POP_UPS:
            anchors = [cv2.imread(path) for path in
                       sorted(glob.glob(f'assets/pop_up_anchors/{prefix}_*.png'))]
            detector = PopUpAnchorDetector(anchors, threshold=threshold, scales=POP_UP_SCALES,
                                           downscale=0.5, white_text=white_text)
            self.pop_ups.append((name, prefix, detector, RisingEdgeAlert(release_frames=2)))
        self.mini_games = MiniGameRunner()
        self.pop_up_thread = threading.Thread(target=self._pop_up_main, daemon=True)

    def start(self):
        """Starts this Notifier's thread."""

        print('\n[~] Started notifier')
        self.thread.start()
        self.pop_up_thread.start()

    def _main(self):
        self.ready = True
        prev_others = 0
        rune_start_time = time.time()
        while True:
            frame = config.capture.frame
            if config.enabled:
                height, width, _ = frame.shape
                minimap = config.capture.minimap['minimap']

                # Check for unexpected black screen
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                expected = time.monotonic() < getattr(config, 'map_transition_until', 0.0)
                if not expected and \
                        np.count_nonzero(gray < 15) / height / width > self.room_change_threshold:
                    self._alert()

                # Check for elite warning
                elite_frame = frame[height // 4:3 * height // 4, width // 4:3 * width // 4]
                elite = utils.multi_match(elite_frame, ELITE_TEMPLATE, threshold=0.9)
                if len(elite) > 0:
                    self._alert()

                # Check for other players entering the map
                filtered = utils.filter_color(minimap, OTHER_RANGES)
                others = len(utils.multi_match(filtered, OTHER_TEMPLATE, threshold=0.5))
                config.stage_fright = others > 0
                if others != prev_others:
                    if others > prev_others:
                        self._ping('ding')
                    prev_others = others

                # Check for rune
                now = time.time()
                if not config.bot.rune_active:
                    filtered = utils.filter_color(minimap, RUNE_RANGES)
                    matches = utils.multi_match(filtered, RUNE_TEMPLATE, threshold=0.9)
                    rune_start_time = now
                    if matches and config.routine.sequence:
                        abs_rune_pos = (matches[0][0], matches[0][1])
                        config.bot.rune_pos = utils.convert_to_relative(abs_rune_pos, minimap)
                        distances = list(map(distance_to_rune, config.routine.sequence))
                        index = np.argmin(distances)
                        config.bot.rune_closest_pos = config.routine[index].location
                        config.bot.rune_active = True
                        self._ping('rune_appeared')
                elif now - rune_start_time > self.rune_alert_delay:     # Alert if rune hasn't been solved
                    config.bot.rune_active = False
                    self._alert()
            time.sleep(0.05)

    def _pop_up_main(self):
        """Runs on its own thread so pop-up scanning never slows the checks above."""

        while True:
            frame = config.capture.frame if config.capture else None
            if frame is not None:
                try:
                    self._check_pop_ups(frame)
                except Exception as error:
                    print(f'\n[!] Pop-up check failed: {error}')
            time.sleep(POP_UP_CHECK_SECONDS)

    def _check_pop_ups(self, frame):
        """
        When one of POP_UPS appears on screen: rings the alarm (siren + flashing
        button until the user clicks it) and starts its mini-game solver, if any.
        """

        seen = {prefix: detector.detect(frame) for _, prefix, detector, _ in self.pop_ups}
        for prefix, rivals in POP_UP_OVERRIDES.items():
            if seen.get(prefix) and any(seen.get(rival) for rival in rivals):
                seen[prefix] = False
        for name, prefix, detector, edge in self.pop_ups:
            if edge.update(seen[prefix]):
                print(f'\n[!] {name} pop-up detected! Check the game window.')
                self._save_pop_up(prefix, frame, detector)
                ALARM.ring(name)
                started = self.mini_games.trigger(prefix)
                if not started and not (prefix in GAMES and AUTO_SOLVE.get(prefix, False)):
                    # No solver for this one: it is still a mini-game that came up
                    print(f'[~] {TALLY.record(name.split(" (")[0], "left to you")}')

    POP_UP_RECORD_DIR = 'logs/pop_ups'      # the frame behind each alert; None to disable
    POP_UP_RECORD_KEEP = 200

    def _save_pop_up(self, prefix, frame, detector):
        """Keeps the screenshot that set an alert off (was it a real pop-up?)."""
        directory = self.POP_UP_RECORD_DIR
        if not directory:
            return
        try:
            import os
            os.makedirs(directory, exist_ok=True)
            try:
                score = f'-score{detector.best_score(frame):.2f}'
            except Exception:
                score = ''
            name = f'{prefix}-{time.strftime("%Y%m%d-%H%M%S")}{score}.jpg'
            cv2.imwrite(os.path.join(directory, name), frame[:, :, :3],
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            files = sorted(f for f in os.listdir(directory) if f.endswith('.jpg'))
            for old in files[:-self.POP_UP_RECORD_KEEP]:
                os.remove(os.path.join(directory, old))
        except Exception as error:
            print(f'\n[!] Could not save the pop-up screenshot: {error}')

    def _alert(self):
        """
        Pauses Auto Maple for a dangerous event until the user acknowledges it.
        """

        config.enabled = False
        config.listener.enabled = False
        print(
            '\n[!] Safety alert detected; Auto Maple is paused. '
            'Press the Start/stop key to dismiss the alert.'
        )
        while not kb.is_pressed(config.listener.config['Start/stop']):
            time.sleep(0.1)
        time.sleep(2)
        config.listener.enabled = True

    def _ping(self, name):
        """Reports a non-dangerous event without playing a sound."""

        message = {
            'ding': 'Another player entered the map.',
            'rune_appeared': 'A rune was detected.'
        }.get(name, f'Notification: {name}.')
        print(f'\n[~] {message}')


#################################
#       Helper Functions        #
#################################
def distance_to_rune(point):
    """
    Calculates the distance from POINT to the rune.
    :param point:   The position to check.
    :return:        The distance from POINT to the rune, infinity if it is not a Point object.
    """

    if isinstance(point, Point):
        return utils.distance(config.bot.rune_pos, point.location)
    return float('inf')
