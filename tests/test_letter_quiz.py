"""Tests for the "Click all X's (N total)" quiz solver on saved screenshots."""

import unittest

import cv2
import numpy as np

from src.detection import letter_quiz


class TestLetterQuiz(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer = letter_quiz._TitleRenderer()
        cls.glyphs = letter_quiz.Glyphs()
        cls.items = letter_quiz.Items()

    def solve(self, path):
        frame = cv2.imread(path)
        self.assertIsNotNone(frame, path)
        return letter_quiz.solve(frame, self.renderer, self.glyphs, items=self.items)

    def test_square_tiles(self):
        answer = self.solve('assets/pop-up-3-quiz.png')
        self.assertEqual((answer.title.char, answer.title.count), ('X', 2))
        self.assertEqual(len(answer.tiles), 3)
        self.assertEqual(answer.chosen, [0, 1])
        self.assertEqual(answer.looks[2], 'M')
        # CONFIRM is at about (356-420, 259-276) in this screenshot
        x, y = answer.button
        self.assertTrue(356 <= x <= 420 and 259 <= y <= 276, answer.button)

    def test_snowy_letters_downscaled(self):
        answer = self.solve('assets/pop-up-3-quiz-2.png')
        self.assertEqual((answer.title.char, answer.title.count), ('X', 3))
        self.assertEqual(len(answer.tiles), 4)
        self.assertEqual(answer.chosen, [0, 2, 3])
        self.assertEqual(answer.looks[1], 'Y')
        x, y = answer.button
        self.assertTrue(255 <= x <= 298 and 125 <= y <= 138, answer.button)

    def test_item_question(self):
        # pop-up-quiz-3.mp4: "Click a popsicle" over a popsicle and a witch hat.
        answer = self.solve('assets/pop-up-3-quiz-3.png')
        self.assertEqual(answer.title.item, 'popsicle')
        self.assertEqual(answer.title.count, 1)
        self.assertIsNone(answer.button)        # no CONFIRM for item questions
        self.assertEqual(len(answer.chosen), 1)
        (x, y), = answer.points()
        self.assertTrue(390 <= x <= 470 and 175 <= y <= 260, (x, y))

    def test_picked_item_is_not_chosen_again(self):
        # Once clicked the popsicle turns green; clicking again would unpick it.
        answer = self.solve('assets/pop-up-3-quiz-3-picked.png')
        self.assertEqual(answer.title.item, 'popsicle')
        self.assertEqual(answer.chosen, [])

    def test_odd_one_out(self):
        # "Click the one that is different" over two red 3s and a cat face.
        answer = self.solve('assets/pop-up-3-quiz-4.png')
        self.assertTrue(answer.title.odd)
        self.assertEqual(len(answer.tiles), 3)
        self.assertEqual(answer.chosen, [2])
        self.assertFalse(answer.guessed)        # plain copies: no AI needed
        (x, y), = answer.points()
        self.assertTrue(255 <= x <= 335 and 80 <= y <= 150, (x, y))
        # CONFIRM is at about (280-343, 197-214) in this screenshot
        x, y = answer.button
        self.assertTrue(280 <= x <= 343 and 197 <= y <= 214, answer.button)

    def test_odd_one_out_with_a_number_tile(self):
        # pop-up-7: two straw hats and a grey "9" tile. The tile has no dark
        # outline, so it is found by comparing the panel's slots.
        answer = self.solve('assets/pop-up-7-quiz.png')
        self.assertTrue(answer.title.odd)
        self.assertEqual(len(answer.tiles), 3)
        self.assertEqual(answer.chosen, [1])
        self.assertFalse(answer.guessed)        # reasoned out, no AI
        (x, y), = answer.points()
        self.assertTrue(130 <= x <= 215 and 60 <= y <= 145, (x, y))
        # CONFIRM is at about (268-330, 185-198) in this screenshot
        x, y = answer.button
        self.assertTrue(268 <= x <= 330 and 185 <= y <= 198, answer.button)

    def test_odd_one_out_on_fresh_made_up_rows(self):
        """Rows never seen: random pictures on scenery, the copies mirrored or
        resized a little, the odd one anywhere in a row of 3 or 4."""
        base = cv2.imread('assets/pop-up-7-quiz.png')
        base = cv2.copyMakeBorder(base, 0, 0, 150, 150, cv2.BORDER_REPLICATE)  # room for 4
        title = letter_quiz.read_title(base, self.renderer)
        rng = np.random.default_rng(7)

        def picture(seed):
            r = np.random.default_rng(seed)
            image = np.zeros((60, 60, 3), np.uint8)
            for _ in range(6):
                colour = tuple(int(c) for c in r.integers(40, 255, 3))
                kind = r.integers(3)
                a, b = r.integers(5, 55, 2), r.integers(5, 55, 2)
                if kind == 0:
                    cv2.rectangle(image, tuple(map(int, a)), tuple(map(int, b)), colour, -1)
                elif kind == 1:
                    cv2.circle(image, tuple(map(int, a)), int(r.integers(5, 18)), colour, -1)
                else:
                    cv2.line(image, tuple(map(int, a)), tuple(map(int, b)), colour, 4)
            return image

        for trial in range(12):
            count = 3 if trial % 2 == 0 else 4
            frame = base.copy()
            # repaint the row with plain dimmed scenery
            frame[50:155, :] = (90 + rng.integers(-20, 20, (105, frame.shape[1], 3))).astype(np.uint8)
            same, odd = picture(1000 + trial), picture(2000 + trial)
            where = int(rng.integers(count))
            for k in range(count):
                art = odd if k == where else same
                if k != where and k % 2:
                    art = art[:, ::-1]                              # mirrored copy
                if k != where and k == count - 1:
                    art = cv2.resize(art, (64, 64))                 # resized copy
                cx = int(title.centre_x + (k - (count - 1) / 2) * letter_quiz.TILE_PITCH)
                cy = int(title.y + letter_quiz.ROW_CENTRE)
                h, w = art.shape[:2]
                frame[cy - h // 2:cy - h // 2 + h, cx - w // 2:cx - w // 2 + w] = art
            answer = letter_quiz.solve_odd(frame, title)
            self.assertIsNotNone(answer, trial)
            self.assertEqual(len(answer.tiles), count, trial)
            self.assertEqual(answer.chosen, [where], (trial, answer.looks))
            self.assertFalse(answer.guessed, (trial, answer.looks))

    def test_odd_one_needs_copies_to_be_sure(self):
        self.assertEqual(letter_quiz.odd_one([[1, .9, .3], [.9, 1, .2], [.3, .2, 1]]),
                         (2, True))
        # No two alike: still the best guess, but not sure.
        self.assertEqual(letter_quiz.odd_one([[1, .6, .5], [.6, 1, .4], [.5, .4, 1]]),
                         (2, False))
        self.assertEqual(letter_quiz.odd_one([[1, .2, .9, .9], [.2, 1, .3, .2],
                                              [.9, .3, 1, .9], [.9, .2, .9, 1]]), (1, True))

    def test_other_kinds_are_not_odd_one_out(self):
        for path in ('assets/pop-up-3-quiz.png', 'assets/pop-up-3-quiz-2.png',
                     'assets/pop-up-3-quiz-3.png'):
            self.assertFalse(self.solve(path).title.odd, path)

    def test_is_there_a_spear(self):
        # pop-up-5-6-quiz.png, quiz 5: a candy, wings and a red button; no spear.
        answer = self.solve('assets/pop-up-3-quiz-5.png')
        self.assertEqual((answer.title.ask, answer.title.about), ('there', 'spear'))
        self.assertEqual(len(answer.tiles), 5)          # 3 pictures, O, X
        self.assertEqual(answer.chosen, [4])            # X = no
        (x, y), = answer.points()
        self.assertTrue(190 <= x <= 240 and 150 <= y <= 180, (x, y))
        x, y = answer.button                            # CONFIRM: about (223-275, 212-226)
        self.assertTrue(223 <= x <= 275 and 210 <= y <= 228, answer.button)

    def test_is_this_a_wand(self):
        # pop-up-5-6-quiz.png, quiz 6: the wooden wand (assets/quiz_items/wand.png).
        answer = self.solve('assets/pop-up-3-quiz-6.png')
        self.assertEqual((answer.title.ask, answer.title.about), ('this', 'wand'))
        self.assertEqual(len(answer.tiles), 3)          # 1 picture, O, X
        self.assertEqual(answer.chosen, [1])            # O = yes
        self.assertFalse(answer.guessed)
        (x, y), = answer.points()
        self.assertTrue(50 <= x <= 110 and 185 <= y <= 220, (x, y))
        x, y = answer.button                            # CONFIRM: about (232-296, 256-272)
        self.assertTrue(232 <= x <= 296 and 254 <= y <= 274, answer.button)

    def test_known_picture_under_another_name_is_a_no(self):
        # The wand asked about as something else: its picture settles it.
        frame = cv2.imread('assets/pop-up-3-quiz-6.png')
        title = letter_quiz.read_title(frame, self.renderer)
        title.about = 'spear'
        answer = letter_quiz.solve_yesno(frame, title, self.items)
        self.assertEqual(answer.chosen, [2])            # X = no

    def test_same_item(self):
        self.assertTrue(letter_quiz.same_item('wand', 'wand'))
        self.assertTrue(letter_quiz.same_item('wings', 'wing'))
        self.assertTrue(letter_quiz.same_item('witch hat', 'hat'))
        self.assertFalse(letter_quiz.same_item('wand', 'spear'))
        self.assertFalse(letter_quiz.same_item('bat', 'hat'))

    def test_no_quiz(self):
        for path in ('assets/pop-up-1.png', 'assets/pop-up-2.png', 'assets/pop-up-3.png'):
            self.assertIsNone(self.solve(path), path)

    def test_choose_uses_count_and_look_alikes(self):
        a, b = np.zeros((32, 32)), np.zeros((32, 32))
        a[0, 0] = b[1, 1] = 1.0
        shapes = [a, a, b]
        # The second X reads badly, but looks just like the first one.
        scores = [{'X': 0.7, 'K': 0.5}, {'X': 0.4, 'K': 0.5}, {'X': 0.5, 'K': 0.6}]
        self.assertEqual(letter_quiz.choose(shapes, scores, 'X', 2), [0, 1])
        self.assertEqual(letter_quiz.choose(shapes, scores, 'X', None), [0])


if __name__ == '__main__':
    unittest.main()
