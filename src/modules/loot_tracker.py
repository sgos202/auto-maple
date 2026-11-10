"""Template-based pickup of loot directly under the player."""

import os
import threading

import cv2

from src.common import config
from src.common.interfaces import Configurable
from src.modules.health import is_game_focused


class LootTracker(Configurable):
    DEFAULT_CONFIG = {
        'Enabled': False,
        'Alignment Y': 0,
        'Item count': 0,
        'Item alignments': ''
    }
    TEMPLATE_DIRECTORY = '.settings'
    LOOT_TEMPLATE_PATH = os.path.join(TEMPLATE_DIRECTORY, 'loot_target.png')
    DEFAULT_TEMPLATE_DIRECTORY = os.path.join('assets', 'loot_samples')
    DEFAULT_TEMPLATE_NAMES = (
        'drop_purple.png',
        'drop_coin_left.png',
        'drop_coin_center.png',
        'drop_coin_right.png',
        'drop_white.png'
    )
    LOOT_KEY = 'x'
    MATCH_INTERVAL = 8
    MATCH_THRESHOLD = 0.5
    HIGH_CONFIDENCE_THRESHOLD = 0.72
    MATCH_SCALES = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5)
    MAX_HORIZONTAL_OFFSET = 14
    VERTICAL_TOLERANCE = 12
    MAX_BOB_OFFSET = 16
    MIN_BOB_MOTION = 2
    DETECTION_CONFIRMATIONS = 2
    MISS_REARM_CONFIRMATIONS = 3

    def __init__(self):
        self._lock = threading.Lock()
        self._player_template = None
        self._default_templates = []
        self._custom_templates = []
        self._loot_templates = []
        self._pickup_requested = False
        self._nearby_frames = 0
        self._candidate_track = None
        self._missed_scans = 0
        self._latched_target = None
        self._frame_count = 0
        self._status = 'Built-in loot samples will load when the player badge is calibrated.'
        self._load_error = ''
        super().__init__('loot_tracker')
        self._load_templates()

    def settings(self):
        with self._lock:
            return self.config.copy()

    def update_settings(self, enabled):
        with self._lock:
            self.config['Enabled'] = bool(enabled)
            self._pickup_requested = False
            self._nearby_frames = 0
            self._candidate_track = None
            self._missed_scans = 0
            self._latched_target = None
            self.save_config()

    def set_templates(self, frame, player_bounds, loot_bounds, append=False):
        height, width = frame.shape[:2]
        if (len(loot_bounds) == 2
                and isinstance(loot_bounds[0][0], (int, float))):
            loot_bounds = [loot_bounds]
        if not loot_bounds:
            raise ValueError('Select at least one loot sample.')

        (px1, py1), (px2, py2) = player_bounds
        if (px1 < 0 or py1 < 0 or px2 > width or py2 > height
                or px2 - px1 < 20 or py2 - py1 < 6):
            raise ValueError('The detected player badge is invalid or outside the screen.')

        player_template = None
        capture = config.capture
        mob_tracker = getattr(capture, 'mob_tracker', None)
        if mob_tracker is not None:
            player_template = mob_tracker.player_template()
        if player_template is None:
            raise ValueError(
                'Calibrate the player badge with “Calibrate player + enemies” first.'
            )

        samples = []
        for index, bounds in enumerate(loot_bounds, start=1):
            (x1, y1), (x2, y2) = bounds
            if (x1 < 0 or y1 < 0 or x2 > width or y2 > height
                    or x2 - x1 < 8 or y2 - y1 < 8):
                raise ValueError(
                    f'The selected loot sample {index} is too small or outside the screen.'
                )
            sample = cv2.cvtColor(
                frame[y1:y2, x1:x2, :3],
                cv2.COLOR_BGR2GRAY
            )
            alignment_y = round((y1 + y2 - py1 - py2) / 2)
            samples.append((sample, alignment_y))

        os.makedirs(self.TEMPLATE_DIRECTORY, exist_ok=True)
        with self._lock:
            existing = self._custom_templates.copy() if append else []
        start_index = len(existing)
        for index, (sample, _) in enumerate(samples, start=start_index):
            path = self._loot_template_path(index)
            if not cv2.imwrite(path, sample):
                raise OSError(f'Could not save loot sample to {path}.')

        templates = existing + samples
        with self._lock:
            self._player_template = player_template
            self._custom_templates = templates
            self._loot_templates = self._default_templates + templates
            self.config['Item count'] = len(templates)
            self.config['Item alignments'] = ','.join(
                str(alignment) for _, alignment in templates
            )
            self.config['Alignment Y'] = templates[0][1]
            self._pickup_requested = False
            self._nearby_frames = 0
            self._candidate_track = None
            self._missed_scans = 0
            self._latched_target = None
            self._load_error = ''
            self._status = (
                f'{len(self._default_templates)} built-in and {len(templates)} '
                'custom loot sample(s) ready.'
            )
            self.save_config()
        return len(templates)

    @classmethod
    def _loot_template_path(cls, index):
        if index == 0:
            return cls.LOOT_TEMPLATE_PATH
        return os.path.join(cls.TEMPLATE_DIRECTORY, f'loot_target_{index}.png')

    def _load_templates(self):
        capture = config.capture
        mob_tracker = getattr(capture, 'mob_tracker', None)
        player_template = (
            mob_tracker.player_template() if mob_tracker is not None else None
        )
        with self._lock:
            try:
                item_count = max(0, int(self.config.get('Item count') or 0))
            except (TypeError, ValueError):
                item_count = 0
            alignment_string = self.config.get('Item alignments', '')
            try:
                alignments = [
                    int(value) for value in alignment_string.split(',') if value
                ]
            except (AttributeError, ValueError):
                alignments = []
            if not alignments:
                try:
                    alignments = [int(self.config.get('Alignment Y') or 0)]
                except (TypeError, ValueError):
                    alignments = [0]

        default_templates = []
        missing_defaults = []
        for name in self.DEFAULT_TEMPLATE_NAMES:
            path = os.path.join(self.DEFAULT_TEMPLATE_DIRECTORY, name)
            sample = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if sample is None:
                missing_defaults.append(name)
            else:
                default_templates.append((sample, None))

        custom_templates = []
        missing_samples = []
        for index in range(item_count):
            path = self._loot_template_path(index)
            sample = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if sample is None:
                missing_samples.append(os.path.basename(path))
                continue
            alignment = alignments[index] if index < len(alignments) else alignments[-1]
            custom_templates.append((sample, alignment))

        with self._lock:
            self._default_templates = default_templates
            self._custom_templates = custom_templates
            self._loot_templates = default_templates + custom_templates
            if player_template is None:
                self._load_error = (
                    'Calibrate the player badge with “Calibrate player + enemies” '
                    'to enable built-in loot detection.'
                )
                self._status = self._load_error
                return
            if not self._loot_templates:
                self._load_error = (
                    'No built-in loot samples could be loaded. '
                    'Check the assets/loot_samples folder.'
                )
                self._status = self._load_error
                return
            self._player_template = player_template
            if missing_defaults or missing_samples:
                missing = missing_defaults + missing_samples
                self._load_error = (
                    'Some loot samples could not be loaded: ' + ', '.join(missing)
                )
            else:
                self._load_error = ''
            self._status = (
                f'{len(default_templates)} built-in and '
                f'{len(custom_templates)} custom loot sample(s) loaded.'
            )

    def status(self):
        with self._lock:
            return self._status

    def consume_pickup(self):
        with self._lock:
            if not self._pickup_requested:
                return None
            self._pickup_requested = False
            return self.LOOT_KEY

    @staticmethod
    def _best_match(search, template):
        if search.size == 0:
            return None
        best_match = None
        for scale in LootTracker.MATCH_SCALES:
            width = max(1, round(template.shape[1] * scale))
            height = max(1, round(template.shape[0] * scale))
            if width > search.shape[1] or height > search.shape[0]:
                continue
            scaled_template = (
                template if scale == 1.0 else cv2.resize(
                    template, (width, height), interpolation=cv2.INTER_NEAREST
                )
            )
            result = cv2.matchTemplate(
                search, scaled_template, cv2.TM_CCOEFF_NORMED
            )
            _, score, _, top_left = cv2.minMaxLoc(result)
            if best_match is None or score > best_match[1]:
                best_match = (
                    top_left,
                    score,
                    (scaled_template.shape[0], scaled_template.shape[1])
                )
        return best_match

    def _record_loot_miss(self):
        self._missed_scans += 1
        self._candidate_track = None
        if self._missed_scans >= self.MISS_REARM_CONFIRMATIONS:
            self._latched_target = None

    def process_frame(self, frame, bot_enabled):
        with self._lock:
            self._frame_count += 1
            enabled = self.config['Enabled']
            if not enabled:
                self._pickup_requested = False
                self._nearby_frames = 0
                self._candidate_track = None
                self._missed_scans = 0
                self._latched_target = None
                self._status = 'Enable auto-pickup to detect sampled loot.'
                return
            if self._frame_count % self.MATCH_INTERVAL:
                return
            loot_templates = self._loot_templates.copy()
            player_template = self._player_template

        capture = config.capture
        mob_tracker = getattr(capture, 'mob_tracker', None)
        if mob_tracker is not None:
            shared_player_template = mob_tracker.player_template()
            if shared_player_template is not None:
                player_template = shared_player_template
                with self._lock:
                    self._player_template = shared_player_template

        if player_template is None or not loot_templates:
            with self._lock:
                self._status = self._load_error
                self._pickup_requested = False
            return
        if frame is None or frame.ndim != 3:
            return

        gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
        player_bounds = None
        if mob_tracker is not None:
            player_bounds = mob_tracker.find_player_bounds(frame)
        else:
            player_match = self._best_match(gray, player_template)
            if (player_match is None
                    or player_match[1] < self.MATCH_THRESHOLD):
                player_bounds = None
            else:
                player_x, player_y = player_match[0]
                matched_height, matched_width = player_match[2]
                player_bounds = (
                    (player_x, player_y),
                    (
                        player_x + matched_width,
                        player_y + matched_height
                    )
                )
        if player_bounds is None:
            with self._lock:
                self._status = 'Player badge not found; make sure it is visible.'
                self._nearby_frames = 0
                self._record_loot_miss()
            return

        (player_left, player_top), (player_right, player_bottom) = player_bounds
        player_x = (player_left + player_right) // 2
        player_y = (player_top + player_bottom) // 2
        badge_height = player_bottom - player_top
        candidates = []
        best_score = None
        for loot_template, offset_y in loot_templates:
            if offset_y is None:
                offset_y = -max(18, round(badge_height * 1.2))
            expected_loot_y = player_y + offset_y
            half_height = loot_template.shape[0] // 2
            vertical_search = max(self.VERTICAL_TOLERANCE, self.MAX_BOB_OFFSET)
            y1 = max(
                0,
                expected_loot_y - half_height - vertical_search
            )
            y2 = min(
                gray.shape[0],
                expected_loot_y + half_height + vertical_search
            )
            half_width = loot_template.shape[1] // 2
            x1 = max(0, player_x - self.MAX_HORIZONTAL_OFFSET - half_width)
            x2 = min(gray.shape[1], player_x + self.MAX_HORIZONTAL_OFFSET + half_width)
            match = self._best_match(gray[y1:y2, x1:x2], loot_template)
            if match is None:
                continue
            top, score, matched_size = match
            matched_height, matched_width = matched_size
            if best_score is None or score > best_score:
                best_score = score
            if score < self.MATCH_THRESHOLD:
                continue
            loot_x = x1 + top[0] + matched_width // 2
            loot_y = y1 + top[1] + matched_height // 2
            if (abs(loot_x - player_x) <= self.MAX_HORIZONTAL_OFFSET
                    and abs(loot_y - expected_loot_y) <= vertical_search):
                candidates.append((
                    score, loot_x, loot_y, loot_template, matched_size
                ))

        if not candidates:
            with self._lock:
                if best_score is None:
                    self._status = (
                        'No drop sample fits near the player; add a map-specific '
                        'sample if this drop looks different.'
                    )
                else:
                    self._status = (
                        f'No drop match near the player '
                        f'(best {best_score:.0%}, need {self.MATCH_THRESHOLD:.0%}). '
                        'Add a map-specific sample if needed.'
                    )
                self._nearby_frames = 0
                self._record_loot_miss()
            return

        score, loot_x, loot_y, loot_template, matched_size = max(
            candidates, key=lambda match: match[0]
        )
        if not bot_enabled:
            with self._lock:
                self._pickup_requested = False
                self._nearby_frames = 0
                self._candidate_track = None
                self._missed_scans = 0
                self._latched_target = None
                self._status = (
                    'Sampled loot detected under the player; start Auto Maple '
                    'to send pickup taps.'
                )
            return
        if not is_game_focused():
            with self._lock:
                self._status = 'Loot is under the player; waiting for MapleStory focus.'
                self._nearby_frames = 0
                self._candidate_track = None
            return

        with self._lock:
            target = (loot_x, loot_y)
            tracked_bobbing = False
            if self._candidate_track is None:
                track_count = 1
            else:
                previous_x, previous_y, previous_count, was_bobbing = (
                    self._candidate_track
                )
                same_item = (
                    abs(loot_x - previous_x) <= max(
                        self.MAX_HORIZONTAL_OFFSET * 2,
                        matched_size[1] // 2
                    )
                    and abs(loot_y - previous_y) <= self.MAX_BOB_OFFSET
                )
                if same_item:
                    track_count = previous_count + 1
                    tracked_bobbing = (
                        was_bobbing
                        or abs(loot_y - previous_y) >= self.MIN_BOB_MOTION
                    )
                else:
                    track_count = 1
            self._candidate_track = (
                loot_x, loot_y, track_count, tracked_bobbing
            )
            if self._latched_target is not None:
                previous_x, previous_y = self._latched_target
                same_target = (
                    abs(target[0] - previous_x) <= max(matched_size[1], 6)
                    and abs(target[1] - previous_y) <= max(
                        matched_size[0], self.MAX_BOB_OFFSET * 2
                    )
                )
                if same_target:
                    self._missed_scans = 0
                    self._status = 'Loot under the player; pickup already sent.'
                    return
                self._latched_target = None
                self._nearby_frames = 0

            self._missed_scans = 0
            self._nearby_frames = track_count
            confirmed = (
                track_count >= self.DETECTION_CONFIRMATIONS
                and (
                    score >= self.HIGH_CONFIDENCE_THRESHOLD
                    or tracked_bobbing
                )
            )
            if confirmed:
                self._pickup_requested = True
                self._latched_target = target
                self._nearby_frames = 0
                self._candidate_track = None
                motion = 'bobbing loot tracked' if tracked_bobbing else 'loot matched'
                self._status = (
                    f'{motion} under the player; six quick X taps requested.'
                )
            elif track_count >= self.DETECTION_CONFIRMATIONS:
                self._status = (
                    f'Loot-like shape at {score:.0%}; watching for its vertical '
                    'bob before pickup.'
                )
            else:
                self._status = (
                    f'Loot-like shape at {score:.0%}; tracking '
                    f'{track_count}/{self.DETECTION_CONFIRMATIONS}.'
                )
