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
        'Enemy alignments': ''
    }
    TEMPLATE_DIRECTORY = '.settings'
    PLAYER_TEMPLATE_PATH = os.path.join(TEMPLATE_DIRECTORY, 'mob_player_badge.png')
    MOB_TEMPLATE_PATH = os.path.join(TEMPLATE_DIRECTORY, 'mob_target.png')
    PLAYER_SEARCH_HEIGHT_RATIO = 0.88
    PLAYER_MATCH_MIN_THRESHOLD = 0.60
    MOB_MATCH_MIN_THRESHOLD = 0.52
    MOB_MATCH_THRESHOLD_MARGIN = 0.20
    MOB_TEMPLATE_SCALES = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2)
    MATCH_INTERVAL = 6
    ATTACK_COOLDOWN = 0.4
    ATTACK_RANGE = 60
    VERTICAL_TOLERANCE = 16
    SEARCH_VERTICAL_TOLERANCE = 64
    GROUP_ATTACK_VERTICAL_TOLERANCE = 64
    TARGET_CONFIRMATIONS = 2
    TARGET_LOST_GRACE_SECONDS = 1.5
    TARGET_LOCK_RADIUS = 32
    HP_BAR_SEARCH_HEIGHT = 42
    HP_BAR_MAX_WIDTH = 70
    HP_BAR_MAX_CENTER_OFFSET = 24
    GROUP_MIN_ENEMIES = 3
    GROUP_HORIZONTAL_RADIUS = 72
    GROUP_VERTICAL_RADIUS = 64

    def __init__(self):
        self._lock = threading.Lock()
        self._player_template = None
        self._mob_templates = []
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
            self.config['Alignment Y'] = mob_templates[0][1]
            self.config['Enemy count'] = len(mob_templates)
            self.config['Enemy alignments'] = ','.join(
                str(alignment_y) for _, alignment_y in mob_templates
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
            self.config['Enemy alignments'] = ','.join(
                str(offset) for _, offset in self._mob_templates
            )
            self._attack_direction = None
            self._target_offset_x = None
            self._target_vertical_offset = None
            self._target_attackable = False
            self._locked_target = None
            self._pending_target = None
            self._target_missed_scans = 0
            self._target_missed_since = None
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

    def _hold_locked_target(self, player_x, enabled, bot_enabled):
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

            _, locked_x, _ = self._locked_target
            self._target_offset_x = locked_x - player_x
            self._target_attackable = (
                abs(self._target_offset_x) <= self.ATTACK_RANGE
                and self._target_vertical_offset is not None
                and abs(self._target_vertical_offset) <= self.VERTICAL_TOLERANCE
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
        enemies = []
        for item in sorted(
            positioned_candidates,
            key=lambda candidate: candidate[0][0],
            reverse=True,
        ):
            candidate, mob_x, mob_y, _ = item
            template = candidate[2]
            width, height = template.shape[1], template.shape[0]
            duplicate = any(
                abs(mob_x - existing[0]) < min(width, existing[2]) * 0.5
                and abs(mob_y - existing[1]) < min(height, existing[3]) * 0.5
                for existing in enemies
            )
            if not duplicate:
                enemies.append((mob_x, mob_y, width, height, item))

        groups = []
        remaining = set(range(len(enemies)))
        while remaining:
            component = []
            frontier = [remaining.pop()]
            while frontier:
                index = frontier.pop()
                component.append(index)
                x, y = enemies[index][:2]
                neighbors = [
                    other for other in remaining
                    if abs(enemies[other][0] - x) <= cls.GROUP_HORIZONTAL_RADIUS
                    and abs(enemies[other][1] - y) <= cls.GROUP_VERTICAL_RADIUS
                ]
                for other in neighbors:
                    remaining.remove(other)
                    frontier.append(other)
            if len(component) >= cls.GROUP_MIN_ENEMIES:
                members = [enemies[index][4] for index in component]
                center_x = (
                    sum(enemies[index][0] for index in component) / len(component)
                )
                groups.append((abs(center_x - player_x), -len(component), members))

        if not groups:
            return []
        return min(groups, key=lambda group: (group[0], group[1]))[2]

    @classmethod
    def _health_bar_ratio(cls, frame, mob_bounds):
        (mob_x1, mob_y1), (mob_x2, _) = mob_bounds
        mob_center_x = (mob_x1 + mob_x2) // 2
        y1 = max(0, mob_y1 - cls.HP_BAR_SEARCH_HEIGHT)
        y2 = max(y1, mob_y1 - 4)
        x1 = max(0, mob_center_x - cls.HP_BAR_MAX_CENTER_OFFSET - 8)
        x2 = min(frame.shape[1], mob_center_x + cls.HP_BAR_MAX_CENTER_OFFSET + 9)
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
                self._target_offset_x = None
                self._target_vertical_offset = None
                self._target_attackable = False
                self._status = 'Waiting for auto-attack to be enabled and started.'
                return
            if self._frame_count % self.MATCH_INTERVAL:
                return
            self._attack_direction = None
            self._group_count = 0
            self._attack_is_group = False
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
        candidates = []
        sample_best_scores = []
        for template_index, (mob_template, alignment_y) in enumerate(mob_templates):
            mob_search = gray
            if mob_template.ndim == 3:
                mob_search = frame[:, :, :3]
            sample_candidates = []
            sample_best_score = 0.0
            for scale in self.MOB_TEMPLATE_SCALES:
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

                expected_mob_y = player_y + alignment_y
                half_height = scaled_template.shape[0] // 2
                y1 = max(
                    0,
                    expected_mob_y - half_height - self.SEARCH_VERTICAL_TOLERANCE
                )
                y2 = min(
                    gray.shape[0],
                    expected_mob_y + half_height + self.SEARCH_VERTICAL_TOLERANCE
                )
                x1 = max(
                    0,
                    player_x - distance - scaled_template.shape[1]
                )
                x2 = min(gray.shape[1], player_x + distance)
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
            if self._hold_locked_target(player_x, enabled, bot_enabled):
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

        group = self._nearest_enemy_group(positioned_candidates, player_x)
        group_count = len(group)
        with self._lock:
            self._group_count = group_count
            locked_target = self._locked_target
        selected = None
        if locked_target is not None:
            _, locked_x, locked_y = locked_target
            locked_candidates = [
                item for item in positioned_candidates
                if abs(item[1] - locked_x) <= self.TARGET_LOCK_RADIUS
                and abs(item[2] - locked_y) <= self.TARGET_LOCK_RADIUS
            ]
            if locked_candidates:
                selected = min(
                    locked_candidates,
                    key=lambda item: abs(item[1] - locked_x)
                    + abs(item[2] - locked_y)
                )
            else:
                if (not group
                        and self._hold_locked_target(player_x, enabled, bot_enabled)):
                    return

        if selected is None:
            selected = min(
                positioned_candidates,
                key=lambda item: (
                    abs(item[1] - player_x),
                    -item[0][0]
                )
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
        attackable = (
            abs(delta_x) <= self.ATTACK_RANGE
            and abs(delta_y) <= self.VERTICAL_TOLERANCE
        )
        group_direction = None
        group_in_range = False
        if group:
            group_center_x = sum(item[1] for item in group) / group_count
            group_direction = 'left' if group_center_x < player_x else 'right'
            group_in_range = any(
                abs(item[1] - player_x) <= self.ATTACK_RANGE
                and abs(
                    item[2] - (
                        player_y + item[0][3]
                    )
                ) <= self.GROUP_ATTACK_VERTICAL_TOLERANCE
                for item in group
            )

        with self._lock:
            self._locked_target = (template_index, mob_x, mob_y)
            self._target_missed_scans = 0
            self._target_missed_since = None
            self._status = (
                f'Locked enemy {mob_score:.0%} among {len(mob_templates)} sample(s), '
                f'offset ({delta_x:+d}, {delta_y:+d}) px; '
                f'{f"HP ~{health_ratio:.0%}; " if health_ratio is not None else ""}'
                f'{"in attack range" if attackable else "approaching"} '
                'on calibrated level.'
            )
            if group:
                self._status += (
                    f' Group of {group_count} detected; '
                    f'{"in group-attack range" if group_in_range else "approaching group"}.'
                )
            self._target_offset_x = group_center_x - player_x if group else delta_x
            self._target_vertical_offset = delta_y
            self._target_attackable = attackable or group_in_range
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
