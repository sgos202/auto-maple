"""The answer buttons count as lit in any style: grey while disabled, anything else when lit."""

import unittest

import cv2
import numpy as np

from src.detection import violetta_solver as vs


def _solver():
    solver = vs.ViolettaSolver.__new__(vs.ViolettaSolver)
    solver.calibration = vs.Calibration(1.0, (0.0, 0.0), 1.0)
    return solver


def _frame(colours):
    """A window-sized frame with the four buttons painted in COLOURS (BGR)."""
    frame = np.full((vs.WINDOW_SIZE[1], vs.WINDOW_SIZE[0], 3), 120, np.uint8)
    for (cx, cy), colour in zip(vs.BUTTON_CENTRES, colours):
        cv2.rectangle(frame, (int(cx - 48), int(cy - 17)), (int(cx + 48), int(cy + 1)), colour, -1)
    return frame


GREY = (180, 180, 182)


class ButtonStyleTests(unittest.TestCase):
    def test_grey_buttons_are_not_lit(self):
        self.assertFalse(_solver().buttons_enabled(_frame([GREY] * 4)))

    def test_any_colour_or_white_counts_as_lit(self):
        for colour in [(230, 200, 40), (30, 140, 250), (60, 200, 60), (200, 60, 160), (250, 250, 250)]:
            solver = _solver()
            self.assertFalse(solver.buttons_enabled(_frame([GREY] * 4)))   # learns the grey
            self.assertTrue(solver.buttons_enabled(_frame([colour] * 4)), colour)

    def test_one_highlighted_button_is_not_enough(self):
        solver = _solver()
        solver.buttons_enabled(_frame([GREY] * 4))
        self.assertFalse(solver.buttons_enabled(_frame([GREY, GREY, (250, 250, 250), GREY])))

    def test_a_slightly_lighter_grey_is_not_lit(self):
        solver = _solver()
        solver.buttons_enabled(_frame([GREY] * 4))
        self.assertFalse(solver.buttons_enabled(_frame([(200, 200, 200)] * 4)))


if __name__ == '__main__':
    unittest.main()
