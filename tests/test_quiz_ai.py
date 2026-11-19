"""Tests for the quiz AI fallback (no network: the API client is faked)."""

import glob
import json
import os
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from src.detection import letter_quiz, quiz_ai

QUIZZES = sorted(glob.glob('assets/pop-up-3-quiz*.png')) + ['assets/pop-up-7-quiz.png']
COUNTDOWN = 'logs/letter_quiz/miss-20261117-223929'


class FakeMessages:
    def __init__(self, data, delay=0.0):
        self.data, self.delay, self.calls = data, delay, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        time.sleep(self.delay)
        return SimpleNamespace(stop_reason='end_turn',
                               content=[SimpleNamespace(type='text', text=json.dumps(self.data))])


def fake_ai(data, delay=0.0):
    with mock.patch.object(quiz_ai, '_settings', return_value={'api_key': 'test', 'backend': 'claude'}):
        ai = quiz_ai.QuizAI()
    ai.client = SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages(data, delay)))
    return ai


def answer(**changes):
    data = {'quiz_visible': True, 'countdown_seconds': -1, 'question': 'Click the cat',
            'reasoning': 'r', 'clicks': [{'x': 150, 'y': 90, 'what': 'cat'}],
            'press_confirm': True, 'confirm_x': 300, 'confirm_y': 450, 'sure': True}
    data.update(changes)
    return data


class TestFooter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer = letter_quiz._TitleRenderer()

    def test_found_on_every_kind_of_quiz(self):
        for path in QUIZZES:
            footer = letter_quiz.find_footer(cv2.imread(path), self.renderer)
            self.assertIsNotNone(footer, path)

    def test_panel_box_holds_the_title(self):
        frame = cv2.imread('assets/pop-up-7-quiz.png')
        x0, y0, x1, y1 = letter_quiz.panel_box(
            frame, letter_quiz.find_footer(frame, self.renderer))
        title = letter_quiz.read_title(frame, self.renderer)
        self.assertTrue(x0 <= title.x and y0 <= title.y and title.x + title.width <= x1, title)

    @unittest.skipUnless(os.path.isdir(COUNTDOWN), 'saved countdown frames not here')
    def test_not_found_during_the_countdown(self):
        for path in sorted(glob.glob(f'{COUNTDOWN}/0[01]*.jpg')):
            self.assertIsNone(letter_quiz.find_footer(cv2.imread(path), self.renderer), path)


class TestQuizAI(unittest.TestCase):
    def test_off_without_a_key(self):
        with mock.patch.object(quiz_ai, '_settings', return_value={'backend': 'claude'}), \
                mock.patch.dict(os.environ, {}, clear=True):
            ai = quiz_ai.QuizAI()
        self.assertFalse(ai.on)
        self.assertIn('--setup', ai.why_off)
        self.assertFalse(ai.ask(np.zeros((10, 10, 3), np.uint8)))

    def test_crop_coordinates_map_back_to_the_frame(self):
        frame = np.zeros((768, 1024, 3), np.uint8)
        ai = fake_ai(answer())
        reply = ai.solve(frame, crop=(100, 200, 500, 600))
        # crop enlarged 1.5x: model (150, 90) -> frame (100 + 100, 200 + 60)
        (x, y, what), = reply.points
        self.assertAlmostEqual(x, 200, delta=1)
        self.assertAlmostEqual(y, 260, delta=1)
        self.assertEqual(reply.confirm, (300, 500))
        self.assertTrue(reply.usable)
        call = ai.client.beta.messages.calls[0]
        self.assertEqual(call['model'], 'claude-opus-5-5')
        self.assertEqual(call['output_config']['format']['type'], 'json_schema')
        self.assertIn('600 x 600 pixels', call['messages'][0]['content'][1]['text'])

    def test_big_frames_are_shrunk_and_mapped_back(self):
        frame = np.zeros((1440, 2560, 3), np.uint8)
        png, offset, factor, w, h = quiz_ai.prepare(frame)
        self.assertLessEqual(max(w, h), quiz_ai.MAX_SIDE)
        self.assertLessEqual(w * h, quiz_ai.MAX_PIXELS * 1.01)
        reply = fake_ai(answer()).solve(frame)
        self.assertAlmostEqual(reply.points[0][0], 150 / factor[0], delta=1)

    def test_no_confirm_and_off_screen_clicks(self):
        reply = fake_ai(answer(press_confirm=False, clicks=[
            {'x': 5000, 'y': 5, 'what': 'off'}, {'x': 10, 'y': 10, 'what': 'on'}])).solve(
            np.zeros((100, 100, 3), np.uint8))
        self.assertIsNone(reply.confirm)
        self.assertEqual([p[2] for p in reply.points], ['on'])

    def test_countdown_is_not_usable(self):
        reply = fake_ai(answer(quiz_visible=False, countdown_seconds=12, clicks=[])).solve(
            np.zeros((100, 100, 3), np.uint8))
        self.assertFalse(reply.usable)
        self.assertEqual(reply.countdown, 12)

    def test_asks_in_the_background(self):
        ai = fake_ai(answer(), delay=0.3)
        self.assertTrue(ai.ask(np.zeros((100, 400, 3), np.uint8)))
        self.assertTrue(ai.busy)
        self.assertFalse(ai.ask(np.zeros((100, 400, 3), np.uint8)))    # one at a time
        self.assertIsNone(ai.take())
        ai._thread.join(2)
        self.assertIsInstance(ai.take(), quiz_ai.Reply)
        self.assertIsNone(ai.take())                                    # handed out once

    def test_forgotten_question_is_dropped(self):
        ai = fake_ai(answer(), delay=0.2)
        ai.ask(np.zeros((100, 400, 3), np.uint8))
        ai.forget()
        ai._thread.join(2)
        self.assertIsNone(ai.take())

    def test_refusal_is_an_error(self):
        ai = fake_ai(answer())
        ai.client.beta.messages.create = lambda **kw: SimpleNamespace(
            stop_reason='refusal', content=[])
        with self.assertRaises(RuntimeError):
            ai.solve(np.zeros((100, 100, 3), np.uint8))


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def json(self):
        return self.data

    def raise_for_status(self):
        pass


class TestOllama(unittest.TestCase):
    TAGS = {'models': [{'name': 'qwen2.5vl:7b', 'model': 'qwen2.5vl:7b'}]}

    def make(self, tags=TAGS):
        with mock.patch.object(quiz_ai, '_settings', return_value={}), \
                mock.patch('requests.get', return_value=FakeResponse(tags)), \
                mock.patch('requests.post', return_value=FakeResponse({})):
            ai = quiz_ai.QuizAI()
            time.sleep(0.05)            # let the warm-up thread finish under the mock
        return ai

    def test_ollama_is_the_default(self):
        ai = self.make()
        self.assertTrue(ai.on, ai.why_off)
        self.assertEqual((ai.backend, ai.model), ('ollama', 'qwen2.5vl:7b'))

    def test_off_when_not_running_or_not_pulled(self):
        with mock.patch.object(quiz_ai, '_settings', return_value={}), \
                mock.patch('requests.get', side_effect=OSError('refused')):
            ai = quiz_ai.QuizAI()
        self.assertFalse(ai.on)
        self.assertIn('not running', ai.why_off)
        ai = self.make({'models': []})
        self.assertFalse(ai.on)
        self.assertIn('ollama pull qwen2.5vl:7b', ai.why_off)

    def test_two_steps_pick_then_locate(self):
        ai = self.make()
        frame = np.zeros((768, 1024, 3), np.uint8)
        step1 = {'quiz_visible': True, 'countdown_seconds': -1,
                 'question': 'Click the one that is different',
                 'pictures': ['straw hat', 'grey 9 tile', 'straw hat'], 'kind': 'click',
                 'reasoning': 'The 9 is not a hat.', 'pick': [2], 'yes_no': 'none',
                 'press_confirm': True, 'confidence': 90}
        boxes = ('```json\n[{"bbox_2d": [100, 50, 200, 150], "label": "picture 2 of 3 from the '
                 'left: grey 9 tile"}, {"bbox_2d": [300, 400, 340, 420], "label": '
                 '"the CONFIRM button"}]\n```')
        replies = [FakeResponse({'message': {'content': json.dumps(step1)}}),
                   FakeResponse({'message': {'content': boxes}})]
        with mock.patch('requests.post', side_effect=replies) as post:
            reply = ai.solve(frame, crop=(100, 200, 500, 600))
        first, second = (c.kwargs['json'] for c in post.call_args_list)
        self.assertEqual(first['format'], quiz_ai.STEP1_SCHEMA)
        self.assertNotIn('format', second)
        # the follow-up continues the first conversation (so the image is reused)
        self.assertEqual(second['messages'][:2], first['messages'])
        text = first['messages'][1]['content']
        w, h = (int(v) for v in text.split('is ')[1].split(' pixels')[0].split(' x '))
        self.assertEqual((w % 28, h % 28), (0, 0), text)
        self.assertTrue(quiz_ai.OLLAMA_MIN_PIXELS <= w * h <= quiz_ai.OLLAMA_MAX_PIXELS, (w, h))
        (x, y, what), = reply.points
        self.assertAlmostEqual(x, 100 + 150 * 400 / w, delta=0.5)
        self.assertAlmostEqual(y, 200 + 100 * 400 / h, delta=0.5)
        self.assertIn('grey 9 tile', what)
        self.assertAlmostEqual(reply.confirm[0], 100 + 320 * 400 / w, delta=0.5)

    def test_yes_no_becomes_the_o_or_x_button(self):
        ai = self.make()
        step1 = {'quiz_visible': True, 'countdown_seconds': -1, 'question': 'Is this a wand?',
                 'pictures': ['wooden wand'], 'kind': 'yes_no', 'reasoning': 'It is.',
                 'pick': [], 'yes_no': 'yes', 'press_confirm': False, 'confidence': 70}
        boxes = '[{"bbox_2d": [10, 10, 30, 30], "label": "the O button"}]'
        with mock.patch('requests.post', side_effect=[
                FakeResponse({'message': {'content': json.dumps(step1)}}),
                FakeResponse({'message': {'content': boxes}})]):
            reply = ai.solve(np.zeros((300, 300, 3), np.uint8))
        self.assertEqual([p[2] for p in reply.points], ['the O button'])

    def test_picking_every_picture_is_no_answer(self):
        ai = self.make()
        step1 = {'quiz_visible': True, 'countdown_seconds': -1,
                 'question': 'Click the one that is different', 'pictures': ['a', 'b', 'c'],
                 'kind': 'click', 'reasoning': '?', 'pick': [1, 2, 3], 'yes_no': 'none',
                 'press_confirm': True, 'confidence': 100}
        with mock.patch('requests.post', side_effect=[
                FakeResponse({'message': {'content': json.dumps(step1)}})]) as post:
            reply = ai.solve(np.zeros((300, 300, 3), np.uint8))
        self.assertFalse(reply.usable)
        self.assertEqual(post.call_count, 1)        # nothing to locate

    def test_box_answers_are_read_in_any_wrapping(self):
        self.assertEqual(quiz_ai._boxes('{"bbox_2d": [1, 2, 3, 4], "label": "x"}'),
                         [('x', [1, 2, 3, 4])])
        self.assertEqual(quiz_ai._boxes('Here: [{"bbox_2d": [1, 2, 3, 4]}] done'),
                         [('', [1, 2, 3, 4])])
        self.assertEqual(quiz_ai._boxes('no boxes'), [])

class FakeScreen:
    def __init__(self, frame):
        self.frame = np.dstack([frame, np.full(frame.shape[:2], 255, np.uint8)])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def grab(self, region):
        return self.frame


class TestLiveFallback(unittest.TestCase):
    """play_live with a fake game window: a question nothing here reads, the AI's
    answer held until the timer is down to click_at, then clicked."""

    def run_live(self, click_at):
        import ctypes
        frame = cv2.imread('assets/pop-up-7-quiz.png')
        ai = fake_ai(answer(clicks=[{'x': 100, 'y': 100, 'what': 'nine'}]), delay=0.3)
        ai.click_at = click_at
        clicks = []
        start = time.monotonic()
        with mock.patch('mss.mss', return_value=FakeScreen(frame)),                 mock.patch('src.detection.violetta_solver._client_region',
                           return_value=(7, {'left': 0, 'top': 0, 'width': frame.shape[1],
                                             'height': frame.shape[0]})),                 mock.patch('src.detection.violetta_solver._click',
                           side_effect=lambda x, y: clicks.append((time.monotonic() - start, x, y))),                 mock.patch.object(ctypes.windll.user32, 'GetForegroundWindow', return_value=7),                 mock.patch.object(letter_quiz, 'solve', return_value=None),                 mock.patch.object(letter_quiz, '_quiz_ai', return_value=ai),                 mock.patch('src.detection.item_clip.available', return_value=False):
            result = letter_quiz.play_live(select=True, timeout=12.0)
        return result, clicks

    def test_answer_waits_for_the_timer(self):
        result, clicks = self.run_live(click_at=8.5)
        self.assertEqual(len(clicks), 2, clicks)          # the picture, then CONFIRM
        # footer first seen at ~0 s with 10 s on the clock: 8.5 s left comes ~1.5 s in
        self.assertGreater(clicks[0][0], 1.2)
        self.assertLess(clicks[0][0], 4.0)
        self.assertTrue(result.startswith('timed out (1 answered)') or
                        result.startswith('answered'), result)

    def test_answer_clicked_at_once_when_not_waiting(self):
        _, clicks = self.run_live(click_at=None)
        self.assertEqual(len(clicks), 2, clicks)
        self.assertLess(clicks[0][0], 2.5)


if __name__ == '__main__':
    unittest.main()
