"""
Spots one particular monster (or any sprite) anywhere in the play area, from a
few sample crops: e.g. the Iron Hog that sometimes spawns on Pig Beach.

Matching works like the mob tracker's (only the sprite's own pixels, both
facing directions, then a colour check), but the exact position is settled
at full size: a shape pass at half size alone misses a sharp-edged sprite by a
pixel and its colour check then fails (seen on the Iron Hog clip).

A sighting only counts once it repeats (CONFIRM_HITS within CONFIRM_WINDOW), so
one stray match on scenery never sets anything off.
"""

import glob

import cv2
import numpy as np


class SpriteWatch:
    STEP = 3                    # shape pass at 1/3 size (~145 ms a scan, same hits as 1/2)
    COARSE_MARGIN = 0.15        # the small-size pass may score this much lower
    REFINE_PAD = 4              # full-size search this far round each coarse peak
    MAX_PEAKS = 6               # per sample and frame

    def __init__(self, name, samples, threshold=0.62, color_tolerance=40,
                 confirm_hits=2, confirm_window=2.0, region=(0.0, 0.12, 1.0, 0.88)):
        """
        :param samples:     BGR crops of the sprite, or a glob pattern for them.
        :param region:      Where to look, as fractions (left, top, right, bottom)
                            of the frame: skip the minimap and the status bar.
        """
        if isinstance(samples, str):
            samples = [cv2.imread(path) for path in sorted(glob.glob(samples))]
        self.name = name
        self.threshold = threshold
        self.color_tolerance = color_tolerance
        self.confirm_hits = confirm_hits
        self.confirm_window = confirm_window
        self.region = region
        self.variants = []
        for sample in samples:
            if sample is None:
                continue
            sample = np.ascontiguousarray(sample[:, :, :3])
            mask = self._mask(sample)
            self.variants.append(self._variant(sample, mask))
            self.variants.append(self._variant(
                np.ascontiguousarray(sample[:, ::-1]), np.ascontiguousarray(mask[:, ::-1])))
        self.hits = []
        self.last_seen = None       # (time, [(x, y, score), ...]) of the last sighting

    @property
    def ready(self):
        return bool(self.variants)

    @staticmethod
    def _mask(sample):
        from src.modules.mob_tracker import MobTracker
        mask = MobTracker._sample_mask(sample)
        return mask if mask is not None else MobTracker._oval_mask(*sample.shape[:2])

    def _variant(self, sample, mask):
        height, width = sample.shape[:2]
        size = (max(1, width // self.STEP), max(1, height // self.STEP))
        return {
            'image': sample, 'mask': mask, 'mask3': cv2.merge([mask] * 3),
            'inside': mask > 0,
            'small': cv2.resize(sample, size, interpolation=cv2.INTER_AREA),
            'small_mask3': cv2.merge([cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)] * 3),
        }

    def find(self, frame):
        """Every sighting in FRAME: [(centre x, centre y, score)] in frame pixels."""
        if frame is None or not self.variants:
            return []
        frame = frame[:, :, :3]
        height, width = frame.shape[:2]
        left, top = int(width * self.region[0]), int(height * self.region[1])
        right, bottom = int(width * self.region[2]), int(height * self.region[3])
        area = np.ascontiguousarray(frame[top:bottom, left:right])
        if area.size == 0:
            return []
        small_area = cv2.resize(area, (area.shape[1] // self.STEP, area.shape[0] // self.STEP),
                                interpolation=cv2.INTER_AREA)
        found = []
        for variant in self.variants:
            for x, y, score in self._find_variant(area, small_area, variant):
                sample_h, sample_w = variant['image'].shape[:2]
                centre = (left + x + sample_w / 2, top + y + sample_h / 2)
                if any(abs(centre[0] - f[0]) < sample_w * 0.6 and abs(centre[1] - f[1]) < sample_h * 0.6
                       for f in found):
                    continue
                found.append((centre[0], centre[1], score))
        return found

    def _find_variant(self, area, small_area, variant):
        small = variant['small']
        if small.shape[0] > small_area.shape[0] or small.shape[1] > small_area.shape[1]:
            return []
        coarse = cv2.matchTemplate(small_area, small, cv2.TM_CCOEFF_NORMED,
                                   mask=variant['small_mask3'])
        coarse = np.nan_to_num(coarse, nan=0.0, posinf=0.0, neginf=0.0)
        coarse[(coarse > 1.0) | (coarse < -1.0)] = 0.0
        floor = self.threshold - self.COARSE_MARGIN
        peaks = cv2.dilate(coarse, np.ones((small.shape[0], small.shape[1]), np.uint8))
        ys, xs = np.where((coarse >= floor) & (coarse == peaks))
        order = np.argsort(coarse[ys, xs])[::-1][:self.MAX_PEAKS]
        image = variant['image']
        sample_h, sample_w = image.shape[:2]
        pad = self.REFINE_PAD
        results = []
        for index in order:
            x0 = max(0, int(xs[index]) * self.STEP - pad)
            y0 = max(0, int(ys[index]) * self.STEP - pad)
            window = area[y0:y0 + sample_h + 2 * pad, x0:x0 + sample_w + 2 * pad]
            if window.shape[0] < sample_h or window.shape[1] < sample_w:
                continue
            fine = cv2.matchTemplate(window, image, cv2.TM_CCOEFF_NORMED, mask=variant['mask3'])
            fine = np.nan_to_num(fine, nan=0.0, posinf=0.0, neginf=0.0)
            fine[(fine > 1.0) | (fine < -1.0)] = 0.0
            _, score, _, (dx, dy) = cv2.minMaxLoc(fine)
            if score < self.threshold:
                continue
            patch = window[dy:dy + sample_h, dx:dx + sample_w]
            difference = np.abs(patch.astype(np.float32) - image.astype(np.float32))[variant['inside']].mean()
            if difference > self.color_tolerance:
                continue        # the right shape in the wrong colours (scenery)
            results.append((x0 + dx, y0 + dy, float(score)))
        return results

    def update(self, frame, now):
        """Scans FRAME; True once the sprite has been seen often enough lately."""
        found = self.find(frame)
        if found:
            self.hits.append(now)
            self.last_seen = (now, found)
        self.hits = [t for t in self.hits if now - t <= self.confirm_window]
        return len(self.hits) >= self.confirm_hits

    def reset(self):
        self.hits = []
