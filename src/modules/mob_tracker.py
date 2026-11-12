"""Template-based tracking of nearby mobs on the player's current platform."""

import os
import threading
import time

import cv2
import numpy as np

from src.common.interfaces import Configurable
from src.modules.health import is_game_focused


class MobTracker(Configurable):
    DEFAULT_CONFIG = {
        'Enabled': False,
        'Range': 140,
        'Threshold': 78,
        'Alignment Y': 0,
        'Enemy count': 1,
        'Enemy alignments': '',
        'Enemy speeds': '',
        'Focus configured': False,
        'Focused enemies': ''
    }
    TEMPLATE_DIRECTORY = '.settings'
    PLAYER_TEMPLATE_PATH = os.path.join(TEMPLATE_DIRECTORY, 'mob_player_badge.png')
    MOB_TEMPLATE_PATH = os.path.join(TEMPLATE_DIRECTORY, 'mob_target.png')
    PLAYER_SEARCH_HEIGHT_RATIO = 0.88
    PLAYER_MATCH_MIN_THRESHOLD = 0.60
    MOB_MATCH_MIN_THRESHOLD = 0.52
    MOB_MATCH_THRESHOLD_MARGIN = 0.20
    # Game sprites are always drawn at their real size; trying other sizes only
    # produced false matches (small copies of a sample fit clouds and bushes).
    MOB_TEMPLATE_SCALES = (1.0,)
    # A sample is a box around the enemy, so it includes scenery that changes as the
    # player moves. Matching uses only the enemy's own pixels (see _sample_mask), then
    # checks the colours really are the enemy's: soft cloud shading correlates with a
    # round mushroom's shape, but it is the wrong colour.
    MASK_MIN_FRACTION = 0.2
    MASK_MAX_FRACTION = 0.9
    COLOR_TOLERANCE = 50        # mean absolute BGR difference over the enemy's pixels
    JUMP_HEIGHT = 130           # enemies this far above the ground line are mid-jump
    GROUND_MARGIN = 12
    MASK_MATCH_STEP = 2         # shape matching runs at 1/2 size for speed
    SAME_ENEMY_OVERLAP = 0.6    # box overlap that makes two samples' matches one enemy
    MATCH_INTERVAL = 6
    FAST_MATCH_INTERVAL = 3
    ATTACK_COOLDOWN = 0.4
    ATTACK_RANGE = 60
    VERTICAL_TOLERANCE = 16
    SEARCH_VERTICAL_TOLERANCE = 64
    GROUP_ATTACK_VERTICAL_TOLERANCE = 64
    TARGET_CONFIRMATIONS = 2
    TARGET_LOST_GRACE_SECONDS = 1.5
    TARGET_LOCK_RADIUS = 32
    TARGET_LOCK_X_RADIUS = 48    # a locked enemy may jump; follow it by horizontal position
    HP_BAR_SEARCH_HEIGHT = 42
    HP_BAR_MAX_WIDTH = 120
    HP_BAR_MAX_CENTER_OFFSET = 48
    GROUP_MIN_ENEMIES = 3
    GROUP_HORIZONTAL_RADIUS = 72
    GROUP_VERTICAL_RADIUS = 64
    FAST_VERTICAL_SEARCH_EXTRA = 48
    FAST_VERTICAL_ATTACK_TOLERANCE = 48
    FAST_LOCK_RADIUS = 64

    def __init__(self):
        self._lock = threading.Lock()
        self._player_template = None
        self._mob_templates = []
        self._mask_cache = {}
        self._attack_direction = None
        self._target_offset_x = None
        self._target_vertical_offset = None
        self._target_attackable = False
        self._group_count = 0
        self._attack_is_group = False
        self._locked_target = None
        self._pending_target = None
        self._target_missed_scans = 0
        self._target_missed_since = None
        self._last_attack_time = 0
        self._group_in_range = False
        self._player_position = None
        self._sweep_direction = 'right'
        self._frame_count = 0
        self._status = 'Calibrate the player badge and an enemy sample to begin.'
        self._load_error = ''
        super().__init__('mob_tracker')
        self._load_templates()

    def settings(self):
        with self._lock:
            return self.config.copy()

    def update_settings(self, enabled, distance, threshold):
        try:
            distance = int(distance)
            threshold = int(threshold)
        except (TypeError, ValueError) as error:
            raise ValueError('Mob range and match threshold must be whole numbers.') from error
        if not 20 <= distance <= 600:
            raise ValueError('Mob range must be between 20 and 600 pixels.')
        if not 50 <= threshold <= 99:
            raise ValueError('Match threshold must be between 50 and 99%.')

        with self._lock:
            self.config.update({
                'Enabled': bool(enabled),
                'Range': distance,
                'Threshold': threshold
            })
            self._attack_direction = None
            self._target_offset_x = None
            self._target_vertical_offset = None
            self._target_attackable = False
            self._group_count = 0
            self._attack_is_group = False
            self._locked_target = None
            self._pending_target = None
            self._target_missed_scans = 0
            self._target_missed_since = None
            self._last_attack_time = 0
            self._group_in_range = False
            self.save_config()

    def set_templates(self, frame, player_bounds, mob_bounds):
        height, width = frame.shape[:2]
        if (len(mob_bounds) == 2
                and isinstance(mob_bounds[0][0], (int, float))):
            mob_bounds = [mob_bounds]
        if not mob_bounds:
            raise ValueError('Select at least one enemy sample.')

        def crop(bounds, name, min_width, min_height):
            (x1, y1), (x2, y2) = bounds
            if (x1 < 0 or y1 < 0 or x2 > width or y2 > height
                    or x2 - x1 < min_width or y2 - y1 < min_height):
                raise ValueError(f'The selected {name} sample is too small or outside the screen.')
            return frame[y1:y2, x1:x2, :3].copy()

        player_template = cv2.cvtColor(
            crop(player_bounds, 'player badge', 20, 6),
            cv2.COLOR_BGR2GRAY
        )
        mob_templates = [
            (
                crop(bounds, f'enemy {index + 1}', 12, 12),
                round(
                    (bounds[0][1] + bounds[1][1]
                     - player_bounds[0][1] - player_bounds[1][1]) / 2
                )
            )
            for index, bounds in enumerate(mob_bounds)
        ]
        os.makedirs(self.TEMPLATE_DIRECTORY, exist_ok=True)
        if not cv2.imwrite(self.PLAYER_TEMPLATE_PATH, player_template):
            raise OSError(f'Could not save player badge sample to {self.PLAYER_TEMPLATE_PATH}.')
        for index, (mob_template, _) in enumerate(mob_templates):
            path = self._mob_template_path(index)
            if not cv2.imwrite(path, mob_template):
                raise OSError(f'Could not save enemy sample to {path}.')

        with self._lock:
            self._player_template = player_template
            self._mob_templates = mob_templates
            self.config['Focus configured'] = False
            self.config['Focused enemies'] = ''
            self.config['Alignment Y'] = mob_templates[0][1]
            self.config['Enemy count'] = len(mob_templates)
            self.config['Enemy alignments'] = ','.join(
                str(alignment_y) for _, alignment_y in mob_templates
            )
            self.config['Enemy speeds'] = ','.join(
                'normal' for _ in mob_templates
            )
            self._attack_direction = None
            self._target_offset_x = None
            self._target_vertical_offset = None
            self._target_attackable = False
            self._group_count = 0
            self._attack_is_group = False
            self._locked_target = None
            self._pending_target = None
            self._target_missed_scans = 0
            self._target_missed_since = None
            self._last_attack_time = 0
            self._group_in_range = False
            self._load_error = ''
            self._status = (
                f'{len(mob_templates)} enemy sample(s) saved. '
                'Enable auto-attack and press the Start/stop key to test.'
            )
            self.save_config()

    def has_player_sample(self):
        with self._lock:
            return self._player_template is not None

    def player_template(self):
        with self._lock:
            if self._player_template is None:
                return None
            return self._player_template.copy()

    def find_player_bounds(self, frame):
        template = self.player_template()
        if template is None or frame is None or frame.ndim != 3:
            return None

        with self._lock:
            threshold = max(
                self.PLAYER_MATCH_MIN_THRESHOLD,
                self.config['Threshold'] / 100 - 0.18
            )
        gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
        search_height = max(
            1, int(gray.shape[0] * self.PLAYER_SEARCH_HEIGHT_RATIO)
        )
        match = self._best_match(gray[:search_height], template)
        if match is None or match[1] < threshold:
            return None

        x, y = match[0]
        return ((x, y), (x + template.shape[1], y + template.shape[0]))

    def has_samples(self):
        with self._lock:
            return self._player_template is not None and bool(self._mob_templates)

    def enemy_samples(self):
        with self._lock:
            return [template.copy() for template, _ in self._mob_templates]

    def focused_enemy_samples(self):
        with self._lock:
            if not self.config.get('Focus configured'):
                return set(range(len(self._mob_templates)))
            try:
                indices = {
                    int(value)
                    for value in self.config.get('Focused enemies', '').split(',')
                    if value
                }
            except (AttributeError, ValueError):
                return set()
            return {
                index for index in indices
                if 0 <= index < len(self._mob_templates)
            }

    def enemy_speed_profiles(self):
        with self._lock:
            return self._enemy_speed_profiles_locked()

    def _enemy_speed_profiles_locked(self):
        values = self.config.get('Enemy speeds', '').split(',')
        return [
            values[index] if index < len(values) and values[index] in {'normal', 'fast'}
            else 'normal'
            for index in range(len(self._mob_templates))
        ]

    def set_enemy_speed_profile(self, index, profile):
        if profile not in {'normal', 'fast'}:
            raise ValueError("Enemy speed profile must be 'normal' or 'fast'.")
        with self._lock:
            if not 0 <= index < len(self._mob_templates):
                raise ValueError(f'No enemy sample exists at index {index}.')
            profiles = self._enemy_speed_profiles_locked()
            profiles[index] = profile
            self.config['Enemy speeds'] = ','.join(profiles)
            self._locked_target = None
            self._pending_target = None
            self.save_config()

    def set_sweep_direction(self, direction):
        if direction not in {'left', 'right'}:
            raise ValueError("Sweep direction must be 'left' or 'right'.")
        with self._lock:
            self._sweep_direction = direction

    def player_x(self):
        with self._lock:
            return None if self._player_position is None else self._player_position[0]

    def group_in_range(self):
        with self._lock:
            return self._group_in_range

    def set_focused_enemy_samples(self, indices):
        with self._lock:
            valid_indices = {
                int(index) for index in indices
                if 0 <= int(index) < len(self._mob_templates)
            }
            self.config['Focus configured'] = True
            self.config['Focused enemies'] = ','.join(
                str(index) for index in sorted(valid_indices)
            )
            self._locked_target = None
            self._pending_target = None
            self._target_missed_scans = 0
            self._target_missed_since = None
            self.save_config()

    @staticmethod
    def _preferred_candidates(positioned_candidates, focused_samples):
        focused_candidates = [
            item for item in positioned_candidates
            if item[0][1] in focused_samples
        ]
        return focused_candidates or positioned_candidates

    def add_enemy_template(self, frame, player_bounds, mob_bounds):
        height, width = frame.shape[:2]
        (x1, y1), (x2, y2) = mob_bounds
        if (x1 < 0 or y1 < 0 or x2 > width or y2 > height
                or x2 - x1 < 12 or y2 - y1 < 12):
            raise ValueError('The selected enemy sample is too small or outside the screen.')
        (px1, py1), (px2, py2) = player_bounds
        if (px1 < 0 or py1 < 0 or px2 > width or py2 > height
                or px2 - px1 < 20 or py2 - py1 < 6):
            raise ValueError('The selected player badge sample is too small or outside the screen.')

        mob_template = frame[y1:y2, x1:x2, :3].copy()
        with self._lock:
            if self._player_template is None:
                raise ValueError(
                    'Calibrate the player badge with “Calibrate player + enemy” first.'
                )
            sample_count = len(self._mob_templates)
            alignment_y = round((y1 + y2 - py1 - py2) / 2)

        path = self._mob_template_path(sample_count)
        if not cv2.imwrite(path, mob_template):
            raise OSError(f'Could not save enemy sample to {path}.')

        with self._lock:
            self._mob_templates.append((mob_template, alignment_y))
            self.config['Enemy count'] = len(self._mob_templates)
            if not self.config.get('Focus configured'):
                self.config['Focused enemies'] = ','.join(
                    str(index) for index in range(len(self._mob_templates))
                )
            self.config['Enemy alignments'] = ','.join(
                str(offset) for _, offset in self._mob_templates
            )
            profiles = self._enemy_speed_profiles_locked()
            profiles.append('normal')
            self.config['Enemy speeds'] = ','.join(profiles)
            self._attack_direction = None
            self._target_offset_x = None
            self._target_vertical_offset = None
            self._target_attackable = False
            self._locked_target = None
            self._pending_target = None
            self._target_missed_scans = 0
            self._target_missed_since = None
            self._group_in_range = False
            self._status = (
                f'{len(self._mob_templates)} enemy samples saved. '
                'Enable auto-attack and press the Start/stop key to test.'
            )
            self.save_config()
            return len(self._mob_templates)

    @classmethod
    def _mob_template_path(cls, index):
        if index == 0:
            return cls.MOB_TEMPLATE_PATH
        return os.path.join(cls.TEMPLATE_DIRECTORY, f'mob_target_{index}.png')

    def _load_templates(self):
        player_template = cv2.imread(self.PLAYER_TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
        with self._lock:
            try:
                sample_count = int(self.config.get('Enemy count') or 1)
            except (TypeError, ValueError):
                sample_count = 1
            if sample_count < 1:
                sample_count = 1
            alignment_string = self.config.get('Enemy alignments', '')
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

        mob_templates = []
        missing_samples = []
        for index in range(sample_count):
            path = self._mob_template_path(index)
            mob_template = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if mob_template is None:
                missing_samples.append(os.path.basename(path))
                continue
            alignment_y = alignments[index] if index < len(alignments) else alignments[-1]
            mob_templates.append((mob_template, alignment_y))

        with self._lock:
            if player_template is None or not mob_templates:
                self._load_error = 'Select the player badge and an enemy sample.'
                self._status = self._load_error
                return
            if missing_samples:
                self._load_error = (
                    'Enemy sample file(s) missing: '
                    + ', '.join(missing_samples)
                    + '. Recalibrate the enemy samples.'
                )
                self._status = self._load_error
                return
            self._player_template = player_template
            self._mob_templates = mob_templates
            self._load_error = ''
            self._status = (
                f'{len(mob_templates)} enemy sample(s) loaded. '
                'Waiting to detect the player and a nearby enemy.'
            )

    def status(self):
        with self._lock:
            return self._status

    def consume_attack(self):
        with self._lock:
            direction = self._attack_direction
            self._attack_direction = None
            self._attack_is_group = False
            return direction

    def consume_attack_action(self):
        with self._lock:
            action = (
                self._attack_direction,
                self._attack_is_group,
                self._group_count if self._attack_is_group else 0,
            )
            self._attack_direction = None
            self._attack_is_group = False
            return action

    def target(self):
        with self._lock:
            if self._target_offset_x is None:
                return None
            return self._target_offset_x, self._target_attackable

    def pursuit_target(self):
        with self._lock:
            if self._target_offset_x is None or self._target_vertical_offset is None:
                return None
            return (
                self._target_offset_x,
                self._target_attackable,
                self._target_vertical_offset,
            )

    def _hold_locked_target(self, player_x, player_y, enabled, bot_enabled):
        with self._lock:
            if self._locked_target is None:
                return False

            now = time.monotonic()
            if self._target_missed_since is None:
                self._target_missed_since = now
            self._target_missed_scans += 1
            missing_for = now - self._target_missed_since
            if missing_for > self.TARGET_LOST_GRACE_SECONDS:
                self._locked_target = None
                self._pending_target = None
                self._target_missed_scans = 0
                self._target_missed_since = None
                self._target_offset_x = None
                self._target_vertical_offset = None
                self._target_attackable = False
                return False

            template_index, locked_x, locked_y = self._locked_target
            self._target_offset_x = locked_x - player_x
            # A locked enemy was on the player's level when last seen (it may only
            # be hidden behind an effect or mid-jump), so only the distance matters.
            self._target_vertical_offset = 0
            self._target_attackable = (
                abs(self._target_offset_x) <= self.ATTACK_RANGE
            )
            self._attack_direction = None
            if (self._target_attackable and enabled and bot_enabled
                    and is_game_focused()
                    and time.monotonic() - self._last_attack_time
                    >= self.ATTACK_COOLDOWN):
                self._attack_direction = (
                    'left' if self._target_offset_x < 0 else 'right'
                )
                self._last_attack_time = time.monotonic()
            self._status = (
                f'Locked enemy temporarily hidden '
                f'({missing_for:.1f}/{self.TARGET_LOST_GRACE_SECONDS:.1f} sec); '
                f'{"continuing attack" if self._target_attackable else "holding target"}.'
            )
            return True

    @classmethod
    def _sample_mask(cls, template):
        """
        The enemy's own pixels inside its sample box (uint8, 255 = enemy), or None
        when the split is not believable (then the whole box is matched as before).
        """
        if template.ndim != 3 or min(template.shape[:2]) < 12:
            return None
        height, width = template.shape[:2]
        mask = np.zeros((height, width), np.uint8)
        try:
            cv2.grabCut(
                np.ascontiguousarray(template[:, :, :3]), mask,
                (2, 2, width - 4, height - 4),
                np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_RECT
            )
        except cv2.error:
            return None
        foreground = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
        foreground = cv2.morphologyEx(
            foreground, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        if not cls.MASK_MIN_FRACTION <= foreground.mean() <= cls.MASK_MAX_FRACTION:
            return None
        return foreground * 255

    def _mask_for(self, template):
        cache = self.__dict__.setdefault('_mask_cache', {})
        key = (template.shape, hash(template.tobytes()))
        if key not in cache:
            if len(cache) > 64:
                cache.clear()
            cache[key] = self._sample_mask(template)
        return cache[key]

    @classmethod
    def _masked_matching_locations(cls, search, template, mask, threshold):
        """Like _matching_locations, but on the enemy's pixels only plus a colour check."""
        if (search.size == 0 or template.shape[0] > search.shape[0]
                or template.shape[1] > search.shape[1]):
            return [], 0.0
        template = np.ascontiguousarray(template[:, :, :3])
        search = np.ascontiguousarray(search[:, :, :3])
        template_height, template_width = template.shape[:2]
        # The shape match runs at half size (masked matching is slow); the colour
        # check below runs at full size and also settles the exact position.
        step = cls.MASK_MATCH_STEP if min(template_height, template_width) >= 24 else 1
        if step > 1:
            small = (template_width // step, template_height // step)
            small_template = cv2.resize(template, small, interpolation=cv2.INTER_AREA)
            small_mask = cv2.resize(mask, small, interpolation=cv2.INTER_NEAREST)
            small_search = cv2.resize(
                search, (search.shape[1] // step, search.shape[0] // step),
                interpolation=cv2.INTER_AREA
            )
        else:
            small_template, small_mask, small_search = template, mask, search
        if (small_template.shape[0] > small_search.shape[0]
                or small_template.shape[1] > small_search.shape[1]):
            return [], 0.0
        result = cv2.matchTemplate(
            small_search, small_template, cv2.TM_CCOEFF_NORMED,
            mask=cv2.merge([small_mask] * 3)
        )
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        result[(result > 1.0) | (result < -1.0)] = 0.0
        maxima = cv2.dilate(result, np.ones(small_template.shape[:2], dtype=np.uint8))
        peak_y, peak_x = np.where((result >= threshold) & (result == maxima))
        order = np.argsort(result[peak_y, peak_x])[::-1]
        template_float = template.astype(np.float32)
        inside = mask > 0
        max_x = search.shape[1] - template_width
        max_y = search.shape[0] - template_height
        matches = []
        for index in order:
            score = float(result[peak_y[index], peak_x[index]])
            best = None
            for dy in range(0, step):
                for dx in range(0, step):
                    x = min(int(peak_x[index]) * step + dx, max_x)
                    y = min(int(peak_y[index]) * step + dy, max_y)
                    patch = search[y:y + template_height, x:x + template_width]
                    difference = np.abs(
                        patch.astype(np.float32) - template_float
                    )[inside].mean()
                    if best is None or difference < best[0]:
                        best = (difference, x, y)
            difference, x, y = best
            if difference > cls.COLOR_TOLERANCE:
                continue
            if any(
                abs(x - match[0][0]) < template_width
                and abs(y - match[0][1]) < template_height
                for match in matches
            ):
                continue
            matches.append(((x, y), score))
        if matches:
            best_score = matches[0][1]
        else:   # shape-only matches (wrong colours) don't count as near misses
            best_score = min(float(result.max()), threshold - 0.01) if result.size else 0.0
        return matches, best_score

    @classmethod
    def _merge_same_enemy(cls, positioned_candidates):
        """
        Different samples of one enemy match it at slightly different offsets.
        Keeps the best-scoring match of each set of heavily overlapping boxes, so
        one enemy is not counted twice (which would fake a group).
        """
        accepted = []
        for item in sorted(positioned_candidates, key=lambda c: c[0][0], reverse=True):
            candidate, center_x, center_y, _ = item
            height, width = candidate[2].shape[:2]
            duplicate = False
            for other in accepted:
                other_height, other_width = other[0][2].shape[:2]
                overlap_x = (min(center_x + width / 2, other[1] + other_width / 2)
                             - max(center_x - width / 2, other[1] - other_width / 2))
                overlap_y = (min(center_y + height / 2, other[2] + other_height / 2)
                             - max(center_y - height / 2, other[2] - other_height / 2))
                if overlap_x <= 0 or overlap_y <= 0:
                    continue
                intersection = overlap_x * overlap_y
                union = width * height + other_width * other_height - intersection
                if intersection / union >= cls.SAME_ENEMY_OVERLAP:
                    duplicate = True
                    break
            if not duplicate:
                accepted.append(item)
        return accepted

    @staticmethod
    def _ground_offset(mob_templates):
        """
        Where enemies' feet are relative to the player badge: the lowest bottom edge
        among the samples (a sample taken mid-jump sits higher and is ignored).
        """
        return max(
            alignment_y + template.shape[0] / 2
            for template, alignment_y in mob_templates
        )

    @staticmethod
    def _matching_locations(search, template, threshold):
        if (search.size == 0 or template.shape[0] > search.shape[0]
                or template.shape[1] > search.shape[1]):
            return [], 0.0
        if template.ndim == 3:
            channel_results = [
                cv2.matchTemplate(
                    search[:, :, channel],
                    template[:, :, channel],
                    cv2.TM_CCOEFF_NORMED
                )
                for channel in range(template.shape[2])
            ]
            result = np.mean(channel_results, axis=0, dtype=np.float32)
        else:
            result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        best_score = float(np.max(result))
        maxima = cv2.dilate(
            result,
            np.ones(template.shape[:2], dtype=np.uint8)
        )
        peak_y, peak_x = np.where((result >= threshold) & (result == maxima))
        order = np.argsort(result[peak_y, peak_x])[::-1]
        matches = []
        template_height, template_width = template.shape[:2]
        for index in order:
            x = int(peak_x[index])
            y = int(peak_y[index])
            if any(
                abs(x - match[0][0]) < template_width
                and abs(y - match[0][1]) < template_height
                for match in matches
            ):
                continue
            matches.append(((x, y), float(result[y, x])))
        return matches, best_score

    @staticmethod
    def _best_match(search, template):
        if (search.size == 0 or template.shape[0] > search.shape[0]
                or template.shape[1] > search.shape[1]):
            return None
        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, top_left = cv2.minMaxLoc(result)
        return top_left, score

    @classmethod
    def _nearest_enemy_group(cls, positioned_candidates, player_x):
        positioned_candidates = cls._deduplicate_positioned_candidates(
            positioned_candidates
        )
        enemies = [
            (
                item[1],
                item[2],
                item[0][2].shape[1],
                item[0][2].shape[0],
                item,
            )
            for item in positioned_candidates
        ]
        groups = []
        for anchor_x, anchor_y, _, _, _ in enemies:
            component = [
                enemy for enemy in enemies
                if abs(enemy[0] - anchor_x) <= cls.GROUP_HORIZONTAL_RADIUS
                and abs(enemy[1] - anchor_y) <= cls.GROUP_VERTICAL_RADIUS
            ]
            if len(component) < cls.GROUP_MIN_ENEMIES:
                continue
            if (
                max(enemy[0] for enemy in component)
                - min(enemy[0] for enemy in component)
                > cls.GROUP_HORIZONTAL_RADIUS
                or max(enemy[1] for enemy in component)
                - min(enemy[1] for enemy in component)
                > cls.GROUP_VERTICAL_RADIUS
            ):
                continue
            members = [enemy[4] for enemy in component]
            center_x = sum(enemy[0] for enemy in component) / len(component)
            groups.append((abs(center_x - player_x), -len(component), members))

        if not groups:
            return []
        return min(groups, key=lambda group: (group[0], group[1]))[2]

    @staticmethod
    def _deduplicate_positioned_candidates(positioned_candidates):
        accepted = []
        for item in sorted(
            positioned_candidates,
            key=lambda candidate: candidate[0][0],
            reverse=True,
        ):
            candidate, center_x, center_y, _ = item
            width = candidate[2].shape[1]
            height = candidate[2].shape[0]
            duplicate = False
            for previous in accepted:
                previous_candidate, previous_x, previous_y, _ = previous
                previous_width = previous_candidate[2].shape[1]
                previous_height = previous_candidate[2].shape[0]
                close_centers = (
                    abs(center_x - previous_x) < min(width, previous_width) * 0.25
                    and abs(center_y - previous_y) < min(height, previous_height) * 0.25
                )
                if close_centers:
                    duplicate = True
                    break
            if not duplicate:
                accepted.append(item)
        return accepted

    @staticmethod
    def _sweep_candidate(candidates, player_x, direction):
        if direction == 'right':
            forward = [item for item in candidates if item[1] >= player_x]
            if forward:
                return min(forward, key=lambda item: (item[1], -item[0][0]))
        elif direction == 'left':
            forward = [item for item in candidates if item[1] <= player_x]
            if forward:
                return max(forward, key=lambda item: (item[1], item[0][0]))
        else:
            raise ValueError(f'Unsupported sweep direction: {direction}')
        return min(
            candidates,
            key=lambda item: (abs(item[1] - player_x), -item[0][0]),
        )

    @classmethod
    def _group_in_attack_range(cls, group, player_x, player_y):
        enemies_in_range = sum(
            abs(item[1] - player_x) <= cls.ATTACK_RANGE
            and abs(item[2] - (player_y + item[0][3]))
            <= cls.GROUP_ATTACK_VERTICAL_TOLERANCE
            for item in group
        )
        return enemies_in_range >= cls.GROUP_MIN_ENEMIES

    @classmethod
    def _health_bar_ratio(cls, frame, mob_bounds):
        (mob_x1, mob_y1), (mob_x2, _) = mob_bounds
        mob_center_x = (mob_x1 + mob_x2) // 2
        y1 = max(0, mob_y1 - cls.HP_BAR_SEARCH_HEIGHT)
        y2 = max(y1, mob_y1 - 4)
        horizontal_margin = (
            cls.HP_BAR_MAX_WIDTH // 2 + cls.HP_BAR_MAX_CENTER_OFFSET
        )
        x1 = max(0, mob_center_x - horizontal_margin)
        x2 = min(frame.shape[1], mob_center_x + horizontal_margin + 1)
        region = frame[y1:y2, x1:x2, :3]
        if region.size == 0:
            return None

        blue, green, red = cv2.split(region)
        fill_mask = (green > 200) & (red < 80) & (blue < 80)
        best = None
        for row_index, mask_row in enumerate(fill_mask):
            fill_x = np.flatnonzero(mask_row)
            if fill_x.size < 2:
                continue
            runs = []
            run_start = previous = int(fill_x[0])
            for value in fill_x[1:]:
                x = int(value)
                if x > previous + 1:
                    if previous - run_start + 1 >= 2:
                        runs.append((run_start, previous))
                    run_start = x
                previous = x
            if previous - run_start + 1 >= 2:
                runs.append((run_start, previous))

            dark_row = np.all(region[row_index] < 75, axis=1)
            for fill_start, fill_end in runs:
                bar_start = fill_start
                while bar_start > 0 and dark_row[bar_start - 1]:
                    bar_start -= 1
                bar_end = fill_end
                while bar_end + 1 < dark_row.size and dark_row[bar_end + 1]:
                    bar_end += 1
                bar_width = bar_end - bar_start + 1
                if not fill_end - fill_start + 1 < bar_width - 2:
                    continue
                if bar_width > cls.HP_BAR_MAX_WIDTH:
                    continue

                fill_center = x1 + (fill_start + fill_end) // 2
                if abs(fill_center - mob_center_x) > cls.HP_BAR_MAX_CENTER_OFFSET:
                    continue
                ratio = (fill_end - fill_start + 1) / bar_width
                vertical_distance = abs((y1 + row_index) - (mob_y1 - 14))
                rank = (abs(fill_center - mob_center_x), vertical_distance)
                if best is None or rank < best[0]:
                    best = (rank, ratio)
        return best[1] if best is not None else None

    def process_frame(self, frame, bot_enabled):
        with self._lock:
            self._frame_count += 1
            enabled = self.config['Enabled']
            if not enabled or not bot_enabled:
                self._attack_direction = None
                self._attack_is_group = False
                self._group_count = 0
                self._group_in_range = False
                self._target_offset_x = None
                self._target_vertical_offset = None
                self._target_attackable = False
                self._status = 'Waiting for auto-attack to be enabled and started.'
                return
            speed_profiles = self._enemy_speed_profiles_locked()
            match_interval = (
                self.FAST_MATCH_INTERVAL
                if 'fast' in speed_profiles
                else self.MATCH_INTERVAL
            )
            if self._frame_count % match_interval:
                return
            self._attack_direction = None
            self._group_count = 0
            self._attack_is_group = False
            self._group_in_range = False
            player_template = self._player_template
            mob_templates = self._mob_templates.copy()
            threshold = max(
                self.MOB_MATCH_MIN_THRESHOLD,
                self.config['Threshold'] / 100
                - self.MOB_MATCH_THRESHOLD_MARGIN
            )
            player_threshold = max(
                self.PLAYER_MATCH_MIN_THRESHOLD,
                threshold - 0.18
            )
            distance = self.config['Range']

        focused_samples = self.focused_enemy_samples()
        if player_template is None or not mob_templates:
            with self._lock:
                self._status = self._load_error
            return
        if frame is None or frame.ndim != 3:
            return

        gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
        search_height = max(
            1, int(gray.shape[0] * self.PLAYER_SEARCH_HEIGHT_RATIO)
        )
        player_match = self._best_match(gray[:search_height], player_template)
        if player_match is None or player_match[1] < player_threshold:
            with self._lock:
                self._status = 'Player badge not found; make sure it is visible.'
                self._attack_direction = None
                self._attack_is_group = False
                self._group_count = 0
                self._target_offset_x = None
                self._target_vertical_offset = None
                self._target_attackable = False
            return

        player_top, player_score = player_match
        player_x = player_top[0] + player_template.shape[1] // 2
        player_y = player_top[1] + player_template.shape[0] // 2
        with self._lock:
            self._player_position = (player_x, player_y)
        candidates = []
        sample_best_scores = []
        # Every enemy on the player's platform stands on the same ground line, whatever
        # sample found it, and may be anywhere from there up to a jump above it.
        ground_offset = self._ground_offset(mob_templates)
        for template_index, (mob_template, _) in enumerate(mob_templates):
            mob_search = gray
            if mob_template.ndim == 3:
                mob_search = frame[:, :, :3]
            mask = self._mask_for(mob_template)
            sample_candidates = []
            sample_best_score = 0.0
            for scale in self.MOB_TEMPLATE_SCALES:
                scaled_mask = mask
                if scale == 1.0:
                    scaled_template = mob_template
                else:
                    scaled_size = (
                        max(2, round(mob_template.shape[1] * scale)),
                        max(2, round(mob_template.shape[0] * scale))
                    )
                    scaled_template = cv2.resize(
                        mob_template,
                        scaled_size,
                        interpolation=(
                            cv2.INTER_AREA if scale < 1.0
                            else cv2.INTER_CUBIC
                        )
                    )
                    if mask is not None:
                        scaled_mask = cv2.resize(
                            mask, scaled_size, interpolation=cv2.INTER_NEAREST
                        )

                template_height = scaled_template.shape[0]
                # Centre of this sample when its enemy stands on the ground line
                alignment_y = round(ground_offset - template_height / 2)
                ground_y = player_y + ground_offset
                y1 = max(
                    0,
                    round(ground_y - template_height - self.JUMP_HEIGHT)
                )
                y2 = min(
                    gray.shape[0],
                    round(ground_y + self.GROUND_MARGIN)
                )
                x1 = max(
                    0,
                    player_x - distance - scaled_template.shape[1]
                )
                x2 = min(gray.shape[1], player_x + distance)
                if scaled_mask is not None:
                    mob_matches, best_score = self._masked_matching_locations(
                        mob_search[y1:y2, x1:x2],
                        scaled_template,
                        scaled_mask,
                        threshold
                    )
                else:
                    mob_matches, best_score = self._matching_locations(
                        mob_search[y1:y2, x1:x2],
                        scaled_template,
                        threshold
                    )
                sample_best_score = max(sample_best_score, best_score)
                sample_candidates.extend(
                    (
                        mob_score,
                        template_index,
                        scaled_template,
                        alignment_y,
                        mob_top,
                        x1,
                        y1
                    )
                    for mob_top, mob_score in mob_matches
                )
            sample_best_scores.append(sample_best_score)

            sample_candidates.sort(key=lambda candidate: candidate[0], reverse=True)
            for candidate in sample_candidates:
                _, _, scaled_template, _, mob_top, x1, y1 = candidate
                center_x = x1 + mob_top[0] + scaled_template.shape[1] // 2
                center_y = y1 + mob_top[1] + scaled_template.shape[0] // 2
                if any(
                    existing[1] == template_index
                    and abs(
                        existing[5] + existing[4][0]
                        + existing[2].shape[1] // 2 - center_x
                    ) < min(existing[2].shape[1], scaled_template.shape[1]) // 2
                    and abs(
                        existing[6] + existing[4][1]
                        + existing[2].shape[0] // 2 - center_y
                    ) < min(existing[2].shape[0], scaled_template.shape[0]) // 2
                    for existing in candidates
                ):
                    continue
                candidates.append(candidate)

        if not candidates:
            with self._lock:
                self._attack_direction = None
                self._attack_is_group = False
                self._group_count = 0
            if self._hold_locked_target(
                player_x, player_y, enabled, bot_enabled
            ):
                return
            with self._lock:
                self._target_offset_x = None
                self._target_vertical_offset = None
                self._target_attackable = False
                self._group_count = 0
                score_summary = ', '.join(
                    f'{index + 1}:{score:.0%}'
                    for index, score in enumerate(sample_best_scores)
                )
                self._status = (
                    f'Player found ({player_score:.0%}); no nearby enemy match '
                    f'in {len(mob_templates)} sample(s); scores '
                    f'{score_summary}; threshold {threshold:.0%}.'
                )
            return

        positioned_candidates = []
        for candidate in candidates:
            mob_score, template_index, mob_template, alignment_y, mob_top, x1, y1 = candidate
            mob_x = x1 + mob_top[0] + mob_template.shape[1] // 2
            mob_y = y1 + mob_top[1] + mob_template.shape[0] // 2
            mob_bounds = (
                (x1 + mob_top[0], y1 + mob_top[1]),
                (
                    x1 + mob_top[0] + mob_template.shape[1],
                    y1 + mob_top[1] + mob_template.shape[0]
                )
            )
            health_ratio = self._health_bar_ratio(frame, mob_bounds)
            positioned_candidates.append((
                candidate, mob_x, mob_y, health_ratio
            ))

        positioned_candidates = self._merge_same_enemy(
            self._deduplicate_positioned_candidates(positioned_candidates)
        )
        target_candidates = self._preferred_candidates(
            positioned_candidates, focused_samples
        )
        group = self._nearest_enemy_group(positioned_candidates, player_x)
        group_count = len(group)
        with self._lock:
            self._group_count = group_count
            locked_target = self._locked_target
        selected = None
        if locked_target is not None:
            _, locked_x, locked_y = locked_target
            locked_candidates = [
                item for item in target_candidates
                if abs(item[1] - locked_x) <= (
                    self.FAST_LOCK_RADIUS
                    if locked_target[0] < len(speed_profiles)
                    and speed_profiles[locked_target[0]] == 'fast'
                    else self.TARGET_LOCK_X_RADIUS
                )
            ]
            if locked_candidates:
                selected = min(
                    locked_candidates,
                    key=lambda item: abs(item[1] - locked_x)
                    + 0.25 * abs(item[2] - locked_y)
                )
            else:
                if self._hold_locked_target(
                    player_x, player_y, enabled, bot_enabled
                ):
                    return

        if selected is None:
            with self._lock:
                sweep_direction = self._sweep_direction
            selected = self._sweep_candidate(
                target_candidates, player_x, sweep_direction
            )
            candidate, selected_x, selected_y, _ = selected
            template_index = candidate[1]
            with self._lock:
                pending = self._pending_target
                if (pending is not None
                        and abs(pending[1] - selected_x) <= self.TARGET_LOCK_RADIUS
                        and abs(pending[2] - selected_y) <= self.TARGET_LOCK_RADIUS):
                    confirmations = pending[3] + 1
                else:
                    confirmations = 1
                self._pending_target = (
                    template_index, selected_x, selected_y, confirmations
                )
                if confirmations < self.TARGET_CONFIRMATIONS:
                    self._target_offset_x = None
                    self._target_attackable = False
                    self._attack_direction = None
                    self._attack_is_group = False
                    self._status = (
                        f'Confirming enemy at ({selected_x}, {selected_y}) '
                        f'({confirmations}/{self.TARGET_CONFIRMATIONS}); '
                        f'match {candidate[0]:.0%}.'
                    )
                    return
                self._locked_target = (
                    template_index, selected_x, selected_y
                )
                self._pending_target = None

        candidate, _, _, health_ratio = selected
        mob_score, template_index, mob_template, alignment_y, mob_top, x1, y1 = candidate
        mob_x = x1 + mob_top[0] + mob_template.shape[1] // 2
        mob_y = y1 + mob_top[1] + mob_template.shape[0] // 2
        delta_x = mob_x - player_x
        delta_y = mob_y - (player_y + alignment_y)
        direction = 'left' if delta_x < 0 else 'right'
        # delta_y is 0 for an enemy standing on the player's ground line and negative
        # while it is in the air; it comes back down, so a jump doesn't end the fight.
        airborne = delta_y < -self.VERTICAL_TOLERANCE
        on_level = -self.JUMP_HEIGHT - self.GROUND_MARGIN <= delta_y <= self.VERTICAL_TOLERANCE
        attackable = abs(delta_x) <= self.ATTACK_RANGE and on_level
        group_direction = None
        group_in_range = False
        if group:
            group_center_x = sum(item[1] for item in group) / group_count
            group_direction = 'left' if group_center_x < player_x else 'right'
            group_in_range = self._group_in_attack_range(
                group, player_x, player_y
            )

        with self._lock:
            self._locked_target = (template_index, mob_x, mob_y)
            self._target_missed_scans = 0
            self._target_missed_since = None
            self._status = (
                f'Locked enemy {mob_score:.0%} among {len(mob_templates)} sample(s), '
                f'offset ({delta_x:+d}, {delta_y:+d}) px; '
                f'{f"HP ~{health_ratio:.0%}; " if health_ratio is not None else ""}'
                f'{"in attack range" if attackable else "approaching"}'
                f'{" (enemy is jumping)" if airborne and on_level else ""} '
                'on calibrated level.'
            )
            if group:
                self._status += (
                    f' Group of {group_count} detected; '
                    f'{"in group-attack range" if group_in_range else "approaching group"}.'
                )
            self._target_offset_x = group_center_x - player_x if group else delta_x
            self._target_vertical_offset = 0 if on_level else delta_y
            self._target_attackable = attackable or group_in_range
            self._group_in_range = group_in_range and group_count >= self.GROUP_MIN_ENEMIES
            if not enabled or not bot_enabled:
                self._attack_direction = None
                self._attack_is_group = False
            elif not is_game_focused():
                self._status += ' Waiting for MapleStory focus.'
                self._attack_direction = None
                self._attack_is_group = False
            elif time.monotonic() - self._last_attack_time >= self.ATTACK_COOLDOWN:
                if group_in_range:
                    self._attack_direction = group_direction
                    self._attack_is_group = True
                elif attackable:
                    self._attack_direction = direction
                    self._attack_is_group = False
                else:
                    self._attack_direction = None
                    self._attack_is_group = False
                if self._attack_direction is not None:
                    self._last_attack_time = time.monotonic()
