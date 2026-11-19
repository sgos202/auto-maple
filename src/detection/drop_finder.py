"""
Finds drops (mesos, items) lying on the ground near the character, so the bot
can walk to them, not just notice the ones right under its feet.

Samples are colour crops of drops: assets/loot_samples/<map>/*.png (e.g.
pig_beach: the gold and bronze mesos). They are matched through an oval mask
(drops are roundish, and the corners hold scenery) and must match in colour
too, so grass, fence posts and the character's own gold weapon don't count.
Only a band along the character's ground line is searched.
"""

import glob
import os

import cv2
import numpy as np


class DropFinder:
    SAMPLE_GLOB = os.path.join('assets', 'loot_samples', '*', '*.png')
    THRESHOLD = 0.70
    COLOR_TOLERANCE = 42        # mean absolute BGR difference inside the oval
    REACH = 360                 # px either side of the character
    BAND_ABOVE = 42             # band top: this far above the badge's top edge
    BAND_BELOW = 4              # band bottom: this far below it
    MERGE = 16                  # matches this close (px) are one drop

    def __init__(self, samples=None):
        if samples is None:
            samples = [cv2.imread(path) for path in sorted(glob.glob(self.SAMPLE_GLOB))]
        self.samples = []
        for sample in samples:
            if sample is None or min(sample.shape[:2]) < 8:
                continue
            sample = np.ascontiguousarray(sample[:, :, :3])
            height, width = sample.shape[:2]
            mask = np.zeros((height, width), np.uint8)
            cv2.ellipse(mask, (width // 2, height // 2),
                        (max(1, width // 2 - 2), max(1, height // 2 - 2)), 0, 0, 360, 255, -1)
            self.samples.append((sample, cv2.merge([mask] * 3), mask > 0))

    @property
    def ready(self):
        return bool(self.samples)

    def find(self, frame, player_bounds):
        """
        Drops near the character: [(x offset from the character's centre, y, score)],
        nearest first. PLAYER_BOUNDS is its name badge ((left, top), (right, bottom)).
        """
        if frame is None or player_bounds is None or not self.samples:
            return []
        (left, top), (right, bottom) = player_bounds
        player_x = (left + right) / 2
        x1 = max(0, int(player_x - self.REACH))
        x2 = min(frame.shape[1], int(player_x + self.REACH))
        y1 = max(0, top - self.BAND_ABOVE)
        y2 = min(frame.shape[0], top + self.BAND_BELOW)
        band = np.ascontiguousarray(frame[y1:y2, x1:x2, :3])
        found = []
        for sample, mask3, inside in self.samples:
            height, width = sample.shape[:2]
            if band.shape[0] < height or band.shape[1] < width:
                continue
            result = cv2.matchTemplate(band, sample, cv2.TM_CCOEFF_NORMED, mask=mask3)
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            result[(result > 1.0) | (result < -1.0)] = 0.0
            peaks = cv2.dilate(result, np.ones((height, width), np.uint8))
            ys, xs = np.where((result >= self.THRESHOLD) & (result == peaks))
            sample_float = sample.astype(np.float32)
            for y, x in zip(ys, xs):
                patch = band[y:y + height, x:x + width].astype(np.float32)
                if np.abs(patch - sample_float)[inside].mean() > self.COLOR_TOLERANCE:
                    continue
                centre_x = x1 + x + width / 2
                centre_y = y1 + y + height / 2
                score = float(result[y, x])
                near = [i for i, f in enumerate(found)
                        if abs(f[0] - centre_x) < self.MERGE and abs(f[1] - centre_y) < self.MERGE]
                if near:
                    if found[near[0]][2] < score:
                        found[near[0]] = (centre_x, centre_y, score)
                    continue
                found.append((centre_x, centre_y, score))
        drops = [(x - player_x, y, score) for x, y, score in found]
        return sorted(drops, key=lambda d: abs(d[0]))
