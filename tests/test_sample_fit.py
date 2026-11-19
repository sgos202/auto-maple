"""The "these samples are for another map" warning (pig-beach-attack-issue.mp4:
Pig Beach farmed with the Henesys snail samples, swinging at the portal)."""

import os
import pickle
import unittest
from unittest.mock import patch

import cv2

from src.modules import mob_tracker as tracker_module

CLIP = os.path.join('assets', 'pig-beach-attack-issue.mp4')
PIG_PROFILE = os.path.join('.settings', 'profiles', 'pig-profile')


def _tracker(clock):
    with patch.object(tracker_module.MobTracker, 'save_config', lambda self: None):
        tracker = tracker_module.MobTracker()
    tracker.__dict__.pop('_sample_fit', None)
    tracker.__dict__.pop('_last_bar_seen', None)
    return tracker


class SampleFitTests(unittest.TestCase):
    def run_minute(self, best, locked, bar_at=None):
        clock = [1000.0]
        said = []
        with patch.object(tracker_module.time, 'monotonic', lambda: clock[0]), \
                patch('builtins.print', lambda *a, **k: None):
            tracker = _tracker(clock)
            tracker._locked_target = (0, 0) if locked else None
            for _ in range(700):                # 70 s of scans, 10 a second
                clock[0] += 0.1
                if bar_at is not None and clock[0] >= bar_at:
                    tracker._last_bar_seen = clock[0]
                message = tracker._note_sample_fit(best)
                if message:
                    said.append(message)
        return said

    def test_weak_matches_and_no_hp_bar_for_a_minute_is_said_once(self):
        said = self.run_minute(best=0.70, locked=True)
        self.assertEqual(len(said), 1)
        self.assertIn('Load the profile for this map', said[0])

    def test_a_good_match_is_no_warning(self):
        self.assertEqual(self.run_minute(best=0.85, locked=True), [])

    def test_an_hp_bar_is_no_warning(self):
        self.assertEqual(self.run_minute(best=0.70, locked=True, bar_at=1030.0), [])

    def test_an_empty_map_is_no_warning(self):
        self.assertEqual(self.run_minute(best=0.60, locked=False), [])


@unittest.skipUnless(os.path.exists(CLIP) and os.path.isfile(os.path.join(PIG_PROFILE, 'mob_tracker')),
                     'clip or pig profile not available')
class PigBeachClipTests(unittest.TestCase):
    def test_with_the_pig_profile_the_portal_and_flowers_are_never_a_target(self):
        """The clip: 0-14 s only the portal, flowers and fence are in view (the
        snail samples locked onto them); the pigs walk in from the right at ~14.5 s."""
        from src.common import config, utils
        with open(os.path.join(PIG_PROFILE, 'mob_tracker'), 'rb') as file:
            loaded = pickle.load(file)
        dot = cv2.imread(os.path.join('assets', 'player_template.png'), 0)
        clock = [0.0]
        early, late = [], []
        saved = (config.player_pos, getattr(config, 'player_pos_time', None))
        video = cv2.VideoCapture(CLIP)
        try:
            with patch.object(tracker_module, 'is_game_focused', return_value=True), \
                    patch.object(tracker_module.time, 'monotonic', lambda: clock[0]), \
                    patch.object(tracker_module.MobTracker, 'save_config', lambda self: None):
                tracker = tracker_module.MobTracker()   # (the live name badge: same character)
                alignments = [int(v) for v in loaded['Enemy alignments'].split(',')]
                tracker._mob_templates = [
                    (cv2.imread(os.path.join(PIG_PROFILE, os.path.basename(
                        tracker._mob_template_path(index)))), alignment)
                    for index, alignment in enumerate(alignments)]
                tracker._mask_cache = {}
                tracker.config = dict(tracker.DEFAULT_CONFIG, **loaded)
                tracker.config.update({'Enabled': True, 'Magic missile': False,
                                       'Sweep patrol': False})
                tracker._ignored_spots = []
                index = 0
                while True:
                    ok, frame = video.read()
                    if not ok:
                        break
                    clock[0] = index / 30.0
                    index += 1
                    # the game is the right 1024 px; its top 5 rows are off the recording
                    game = cv2.copyMakeBorder(frame[:763, 228:1252], 5, 0, 0, 0,
                                              cv2.BORDER_CONSTANT)
                    minimap = game[60:200, 5:205]
                    found = utils.multi_match(minimap, dot, threshold=0.8)
                    if found:
                        config.player_pos = (found[0][0] / minimap.shape[1],
                                             found[0][1] / minimap.shape[0])
                        config.player_pos_time = clock[0]
                    tracker.process_frame(game, True)
                    if tracker._frame_count % 3 == 0 and tracker.target() is not None:
                        (early if clock[0] < 14.0 else late).append(
                            (round(clock[0], 1), tracker.target()))
        finally:
            config.player_pos, config.player_pos_time = saved
            video.release()
        self.assertEqual(early, [])
        self.assertGreater(len(late), 10)
        self.assertTrue(all(target[0] >= 0 for _, target in late), late)    # the pigs: to the right (0: hidden mid-hop)


if __name__ == '__main__':
    unittest.main()
