"""A module for detecting and notifying the user of dangerous in-game events."""

from src.common import config, utils
from src.detection.mini_game_alert import PopUpAnchorDetector, RisingEdgeAlert
from src.modules.mini_games import MiniGameRunner
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
# A solver starts automatically for each one (see src/modules/mini_games.py).
POP_UPS = (
    ('Lie Detector (fading shape) mini-game', 'pop-up-1'),
    ('Violetta mini-game', 'pop-up-2'),
)
POP_UP_THRESHOLD = 0.83         # Normal gameplay scored <= 0.77 in testing
POP_UP_SCALES = tuple(round(0.8 + 0.05 * i, 2) for i in range(10))     # 0.80 - 1.25
POP_UP_CHECK_SECONDS = 0.5      # How often to scan the screen for pop-ups
POP_UP_SOUND = 'assets/alerts/siren.mp3'
POP_UP_REPEAT_SECONDS = 5       # Replay the sound while the pop-up stays on screen


class Notifier:
    def __init__(self):
        """Initializes this Notifier object's main thread."""

        self.ready = False
        self.thread = threading.Thread(target=self._main)
        self.thread.daemon = True

        self.room_change_threshold = 0.9
        self.rune_alert_delay = 270         # 4.5 minutes
        self.pop_ups = []
        for name, prefix in POP_UPS:
            anchors = [cv2.imread(path) for path in
                       sorted(glob.glob(f'assets/pop_up_anchors/{prefix}_*.png'))]
            detector = PopUpAnchorDetector(anchors, threshold=POP_UP_THRESHOLD,
                                           scales=POP_UP_SCALES, downscale=0.5)
            self.pop_ups.append((name, prefix, detector, RisingEdgeAlert(release_frames=2)))
        self.mini_games = MiniGameRunner()
        self.pop_up_thread = threading.Thread(target=self._pop_up_main, daemon=True)
        self.pop_up_last_sound = 0
        self._sound = None
        try:
            import pygame
            pygame.mixer.init()
            self._sound = pygame.mixer.Sound(POP_UP_SOUND)
        except Exception as error:
            print(f'\n[!] Could not load pop-up alert sound, falling back to beep: {error}')

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
                if np.count_nonzero(gray < 15) / height / width > self.room_change_threshold:
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
        When pop-up-1.png or pop-up-2.png appears on screen: plays an alert and
        starts that mini-game's solver.
        """

        any_visible = False
        for name, prefix, detector, edge in self.pop_ups:
            visible = detector.detect(frame)
            any_visible = any_visible or visible
            if edge.update(visible):
                print(f'\n[!] {name} pop-up detected! Check the game window.')
                self._play_pop_up_sound()
                self.mini_games.trigger(prefix)
        if any_visible and time.time() - self.pop_up_last_sound >= POP_UP_REPEAT_SECONDS:
            self._play_pop_up_sound()
        elif not any(edge.active for _, _, _, edge in self.pop_ups) and self._sound is not None:
            self._sound.stop()      # Pop-ups are gone; silence the siren

    def _play_pop_up_sound(self):
        self.pop_up_last_sound = time.time()
        try:
            if self._sound is not None:
                if self._sound.get_num_channels() == 0:     # Don't overlap itself
                    self._sound.play()
                return
            import winsound
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except Exception as error:
            print(f'\n[!] Could not play the pop-up alert sound: {error}')

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
