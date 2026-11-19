"""
AI fallback for pop-up-3 quiz questions the built-in solvers can't read.

When the quiz panel is up (its "Remain : N Sec" footer is seen) and none of
the solvers in letter_quiz.py has an answer, the screenshot goes to a vision
model, which reads the question and says where to click. The reply usually
takes a few seconds of the 10 s timer. It runs on its own thread, so the
solvers keep trying meanwhile and win if they get there first.

Two backends, picked by "backend" in .settings/quiz_ai.json (not in git):
- "ollama" (the default): a free model running on this PC (Qwen2.5-VL 7B on
  the GPU), no key or account. Setup, once: install Ollama from ollama.com, then
      ollama pull qwen2.5vl:7b
  The model is loaded into the GPU when the quiz's countdown starts, so the
  question itself isn't kept waiting for that.
- "claude": Claude Opus 5.5 over the internet, cleverer with new kinds of
  question; needs an API key from https://console.anthropic.com (paid per
  use, about a cent a question):
      python -m src.detection.quiz_ai --setup sk-ant-...
  (an ANTHROPIC_API_KEY environment variable works too).
Switch with  python -m src.detection.quiz_ai --use ollama|claude

Check it on a screenshot (no clicking):
    python -m src.detection.quiz_ai SCREENSHOT [--output annotated.png]

Settings in .settings/quiz_ai.json: "backend" ("ollama"), "enabled" (true),
"ollama_model" ("qwen2.5vl:7b"), "ollama_url" ("http://127.0.0.1:11434"),
"api_key", "claude_model" ("claude-opus-5-5"), "effort" (Claude: "low", for
quicker replies), "timeout" (8 s),
"probe" (true: also ask now and then when no footer is seen, in case a new
kind of panel has none), "click_at_seconds" (2: the AI's answer is shown with
pink circles and only clicked once the quiz timer is down to this; null or 0
clicks as soon as the answer is in).
"""

import base64
import json
import os
import threading
import time

import cv2
import numpy as np

SETTINGS = '.settings/quiz_ai.json'
BACKEND = 'ollama'
MODEL = 'claude-opus-5-5'
OLLAMA_MODEL = 'qwen2.5vl:7b'
OLLAMA_URL = 'http://127.0.0.1:11434'
OLLAMA_KEEP = '30m'             # keep the model in the GPU this long after a question
# Ollama resizes every image for Qwen2.5-VL to 28 px multiples between
# 1024 and 1280 patches of 28 x 28 px (measured: a 504 x 308 crop came back
# with coordinates 2.3x too big). Images sent already inside that range are
# left alone, so the model's coordinates are pixels of the image sent.
OLLAMA_SIDE = 28
OLLAMA_MIN_PIXELS = 1024 * 28 * 28      # 802,816
OLLAMA_MAX_PIXELS = 1280 * 28 * 28      # 1,003,520
OLLAMA_TARGET_PIXELS = 900_000

# Small models do badly asked to reason and point in one go (0/18 on the saved
# quizzes), so Ollama is asked twice: first what to click, in words, then
# where those things are, in the box format Qwen2.5-VL was trained on.
STEP1_SCHEMA = {
    'type': 'object',
    'properties': {
        'quiz_visible': {'type': 'boolean'},
        'countdown_seconds': {'type': 'integer'},
        'question': {'type': 'string'},
        'pictures': {'type': 'array', 'items': {'type': 'string'}},
        'kind': {'type': 'string', 'enum': ['click', 'yes_no']},
        'reasoning': {'type': 'string'},
        'pick': {'type': 'array', 'items': {'type': 'integer'}},
        'yes_no': {'type': 'string', 'enum': ['yes', 'no', 'none']},
        'press_confirm': {'type': 'boolean'},
        'confidence': {'type': 'integer'},
    },
    'required': ['quiz_visible', 'countdown_seconds', 'question', 'pictures', 'kind',
                 'reasoning', 'pick', 'yes_no', 'press_confirm', 'confidence'],
}
STEP1_PROMPT = """Answer in steps:
1. "pictures": the pictures or letter tiles in the quiz panel, left to right, a few words each \
saying what it shows (not the O/X or CONFIRM buttons). They are numbered 1, 2, 3, ...
2. "kind": "yes_no" if the question asks yes or no ("Is there ...?", "Is this ...?"), else "click".
3. "reasoning": one or two sentences comparing the pictures with the question.
4. For "click": "pick" lists the NUMBERS of the pictures to click - only the ones the question \
asks for (e.g. [2] for the odd one out, [1, 3] for two matching tiles), never all of them unless \
all match. For "yes_no": "pick" is [] and "yes_no" is "yes" or "no".
5. "press_confirm": true if the panel has a CONFIRM button."""

EFFORT = 'low'                  # the timer is 10 s; low effort answers in a few seconds
TIMEOUT = 8.0                   # seconds; a later reply is no use
MAX_TOKENS = 4000
MAX_SIDE = 1568                 # the API shrinks bigger images; do it here to keep coordinates right
MAX_PIXELS = 1_150_000
CLICK_AT_SECONDS = 2            # first live tests: show the answer, click with 2 s left
CROP_SCALE = 1.5                # the panel crop is enlarged this much so small pictures read well

SYSTEM = """You solve the quiz pop-up of a MapleStory anti-bot check for the player.
You get a screenshot of the game. A quiz panel (translucent dark blue) shows a question at the top, \
some pictures, letter tiles or buttons, "Remain : N Sec" at the bottom left and a CONFIRM button at \
the bottom right. Known kinds, for reference (new kinds will appear, so read the question itself):
- "Click all X's (N total)": click every letter tile showing X (any art style), then CONFIRM.
- "Click a <item>": click the picture of that item.
- "Click the one that is different": click the picture unlike the others, then CONFIRM.
- "Is there a <item> here?" / "Is this a <item>?": click the O button for yes or the X button for no \
(they sit under the pictures), then CONFIRM.
Coordinates are pixels in the image you are given, x to the right and y down from its top-left \
corner, at the centre of the thing to click. Be precise: aim at the middle of each tile, picture or button.
If instead the screen shows "Question in N seconds" (the quiz has not started yet), set \
quiz_visible to false and countdown_seconds to N. If there is no quiz at all, quiz_visible is false and \
countdown_seconds is -1."""

SCHEMA = {
    'type': 'object',
    'properties': {
        'quiz_visible': {'type': 'boolean',
                         'description': 'A quiz question is on screen now.'},
        'countdown_seconds': {'type': 'integer',
                              'description': 'N from "Question in N seconds", else -1.'},
        'question': {'type': 'string', 'description': 'The question text as shown.'},
        'reasoning': {'type': 'string', 'description': 'One or two short sentences.'},
        'clicks': {
            'type': 'array',
            'description': 'What to click, in order, not counting CONFIRM.',
            'items': {
                'type': 'object',
                'properties': {
                    'x': {'type': 'integer'},
                    'y': {'type': 'integer'},
                    'what': {'type': 'string', 'description': 'What is clicked there.'},
                },
                'required': ['x', 'y', 'what'],
                'additionalProperties': False,
            },
        },
        'press_confirm': {'type': 'boolean',
                          'description': 'Click CONFIRM after the clicks (if the panel has one).'},
        'confirm_x': {'type': 'integer', 'description': 'CONFIRM centre x, -1 if none.'},
        'confirm_y': {'type': 'integer', 'description': 'CONFIRM centre y, -1 if none.'},
        'sure': {'type': 'boolean', 'description': 'Confident in the answer.'},
        'confidence': {'type': 'integer',
                       'description': 'How likely the answer is right, 0-100.'},
    },
    'required': ['quiz_visible', 'countdown_seconds', 'question', 'reasoning', 'clicks',
                 'press_confirm', 'confirm_x', 'confirm_y', 'sure', 'confidence'],
    'additionalProperties': False,
}


def _settings():
    try:
        with open(SETTINGS, encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}
    except Exception as error:
        print(f'[QUIZ-AI] Could not read {SETTINGS}: {error}')
        return {}


class Reply:
    """The model's answer, with coordinates in the frame that was sent."""

    def __init__(self, data, points, confirm, seconds):
        self.data = data
        self.visible = bool(data.get('quiz_visible'))
        self.countdown = int(data.get('countdown_seconds', -1))
        self.question = data.get('question', '')
        self.reasoning = data.get('reasoning', '')
        self.points = points            # [(x, y, what)] in frame pixels
        self.confirm = confirm          # (x, y) or None
        self.sure = bool(data.get('sure'))
        self.confidence = int(data.get('confidence', 100 if self.sure else 50))
        self.seconds = seconds          # how long the reply took

    @property
    def usable(self):
        return self.visible and bool(self.points)

    def summary(self):
        clicks = ', '.join(f'{what} at ({x:.0f}, {y:.0f})' for x, y, what in self.points)
        return (f'"{self.question}" -> {clicks or "nothing"}'
                f'{" + CONFIRM" if self.confirm else ""}'
                f' [{self.confidence}% sure, {self.seconds:.1f} s; {self.reasoning}]')


def prepare(frame, crop=None, side=1, max_pixels=MAX_PIXELS, pixels=None):
    """FRAME (or its CROP box x0, y0, x1, y1) as PNG bytes for the model, plus
    how to map the model's pixels back: frame = offset + model / (fx, fy).
    SIDE: width and height are made multiples of this (by stretching a little,
    so x and y can have slightly different factors). PIXELS: scale to about this
    many pixels instead (Ollama: OLLAMA_TARGET_PIXELS)."""
    x0, y0 = 0, 0
    image = frame[:, :, :3]
    factor = 1.0
    if crop is not None:
        x0, y0, x1, y1 = (int(v) for v in crop)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(frame.shape[1], x1), min(frame.shape[0], y1)
        image = image[y0:y1, x0:x1]
        factor = CROP_SCALE
    h, w = image.shape[:2]
    if pixels:
        factor = (pixels / (h * w)) ** 0.5
        width = max(side, int(round(w * factor / side)) * side)
        height = max(side, int(round(h * factor / side)) * side)
    else:
        factor = min(factor, MAX_SIDE / max(h, w), (max_pixels / (h * w)) ** 0.5)
        width = max(side, int(w * factor / side) * side)
        height = max(side, int(h * factor / side) * side)
    if (width, height) != (w, h):
        image = cv2.resize(image, (width, height),
                           interpolation=cv2.INTER_CUBIC if factor > 1 else cv2.INTER_AREA)
    ok, png = cv2.imencode('.png', np.ascontiguousarray(image))
    if not ok:
        raise RuntimeError('could not encode the screenshot')
    return png.tobytes(), (x0, y0), (width / w, height / h), width, height


class QuizAI:
    """Asks the model about one screenshot at a time, on a background thread."""

    def __init__(self):
        s = _settings()
        self.backend = s.get('backend', BACKEND)
        self.model = None
        self.effort = s.get('effort', EFFORT)
        self.timeout = float(s.get('timeout', TIMEOUT))
        self.probe = bool(s.get('probe', True))
        self.click_at = s.get('click_at_seconds', CLICK_AT_SECONDS)
        self.client = None
        self.why_off = None
        self._lock = threading.Lock()
        self._thread = None
        self._result = None
        self._ask_id = 0
        if not s.get('enabled', True):
            self.why_off = f'turned off in {SETTINGS}'
            return
        if self.backend == 'ollama':
            self.model = s.get('ollama_model', OLLAMA_MODEL)
            self.url = s.get('ollama_url', OLLAMA_URL).rstrip('/')
            self.why_off = self._ollama_check()
            if self.why_off is None:
                self.client = 'ollama'
                threading.Thread(target=self._ollama_warm, daemon=True).start()
            return
        if self.backend != 'claude':
            self.why_off = f'unknown backend {self.backend!r} in {SETTINGS} (ollama or claude)'
            return
        self.model = s.get('claude_model', s.get('model', MODEL))
        key = s.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
        if not key:
            self.why_off = ('no API key (run: python -m src.detection.quiz_ai --setup KEY, '
                            'or set ANTHROPIC_API_KEY)')
            return
        try:
            import anthropic
        except ImportError:
            self.why_off = 'the anthropic package is missing (pip install anthropic)'
            return
        # No retries: a second try would come back after the quiz timer ran out.
        self.client = anthropic.Anthropic(api_key=key, timeout=self.timeout, max_retries=0)

    # ------------------------------------------------------------ ollama --
    def _ollama_check(self):
        """None if Ollama is running and has the model, else why not."""
        import requests
        try:
            tags = requests.get(f'{self.url}/api/tags', timeout=2).json()
        except Exception:
            return (f'Ollama is not running at {self.url} (install it from ollama.com, '
                    f'then: ollama pull {self.model})')
        names = set()
        for m in tags.get('models', []):
            names.update((m.get('name'), m.get('model')))
        wanted = self.model if ':' in self.model else f'{self.model}:latest'
        if wanted not in names:
            return f'Ollama has no {self.model} model yet (run: ollama pull {self.model})'
        return None

    def _ollama_warm(self):
        """Loads the model into the GPU now, so the question isn't kept waiting."""
        import requests
        try:
            started = time.monotonic()
            requests.post(f'{self.url}/api/generate', json={
                'model': self.model, 'prompt': 'ok', 'stream': False,
                'keep_alive': OLLAMA_KEEP, 'options': {'num_predict': 1}}, timeout=180)
            print(f'[QUIZ-AI] {self.model} is loaded ({time.monotonic() - started:.1f} s).')
        except Exception as error:
            print(f'[QUIZ-AI] Could not load {self.model} in advance: {error}')

    def _ollama_chat(self, messages, deadline, schema=None):
        import requests
        left = deadline - time.monotonic()
        if left <= 0.3:
            raise TimeoutError('out of time')
        body = {'model': self.model, 'messages': messages, 'stream': False,
                'keep_alive': OLLAMA_KEEP, 'options': {'temperature': 0}}
        if schema:
            body['format'] = schema
        response = requests.post(f'{self.url}/api/chat', json=body, timeout=left)
        response.raise_for_status()
        return response.json()['message']['content']

    def _ask_ollama(self, png, text):
        """Two calls: what to click (in words), then where (boxes). Returns the
        same answer as the one-call Claude backend."""
        deadline = time.monotonic() + self.timeout
        image = base64.standard_b64encode(png).decode()
        messages = [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': f'{text}\n{STEP1_PROMPT}', 'images': [image]},
        ]
        said = self._ollama_chat(messages, deadline, STEP1_SCHEMA)
        step1 = json.loads(said)
        data = {k: step1.get(k) for k in ('quiz_visible', 'countdown_seconds', 'question',
                                           'reasoning', 'press_confirm', 'confidence')}
        data['countdown_seconds'] = int(data.get('countdown_seconds') or -1)
        data['pictures'] = pictures = [str(p) for p in step1.get('pictures', [])]
        if step1.get('kind') == 'yes_no' and step1.get('yes_no') in ('yes', 'no'):
            targets = ['the O button' if step1['yes_no'] == 'yes' else 'the X button']
        else:
            numbers = [n for n in step1.get('pick', []) if 1 <= n <= len(pictures)]
            if len(numbers) == len(pictures) and len(pictures) > 1 \
                    and 'all' not in data.get('question', '').lower():
                numbers = []            # picked everything: no real choice was made
            targets = [f'picture {n} of {len(pictures)} from the left: {pictures[n - 1]}'
                       for n in dict.fromkeys(numbers)]
        data['targets'] = targets
        data['sure'] = int(data.get('confidence') or 0) >= 80
        data.update(clicks=[], confirm_x=-1, confirm_y=-1)
        if not data['quiz_visible'] or not targets:
            return data
        wanted = targets + (['the CONFIRM button'] if data['press_confirm'] else [])
        listing = '\n'.join(f'{i + 1}. {t}' for i, t in enumerate(wanted))
        # Asked as a follow-up in the same conversation: Ollama reuses the image
        # it has already processed (1.5 s instead of 3 s).
        boxes = _boxes(self._ollama_chat(messages + [
            {'role': 'assistant', 'content': said},
            {'role': 'user', 'content':
                f'Locate each of these in the image and output their bbox coordinates in JSON '
                f'format, in this order, with "label" set to the text given:\n{listing}'},
        ], deadline))
        for target, box in _pair(wanted, boxes):
            if box is None:
                continue
            x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            if target == 'the CONFIRM button':
                data['confirm_x'], data['confirm_y'] = x, y
            else:
                data['clicks'].append({'x': x, 'y': y, 'what': target})
        return data

    # ------------------------------------------------------------ claude --
    def _ask_claude(self, png, text):
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            messages=[{'role': 'user', 'content': [
                {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png',
                                             'data': base64.standard_b64encode(png).decode()}},
                {'type': 'text', 'text': text},
            ]}],
            output_config={'effort': self.effort,
                           'format': {'type': 'json_schema', 'schema': SCHEMA}},
            # If the model declines, the API retries on a fallback model in the same call.
            betas=['server-side-fallback-2026-07-01'],
            fallbacks='default',
        )
        if response.stop_reason == 'refusal':
            raise RuntimeError('the model declined to answer')
        if response.stop_reason == 'max_tokens':
            raise RuntimeError('the reply was cut off')
        body = next((b.text for b in response.content if b.type == 'text'), None)
        if body is None:
            raise RuntimeError(f'no answer in the reply (stop: {response.stop_reason})')
        return json.loads(body)

    @property
    def on(self):
        return self.client is not None

    @property
    def busy(self):
        return self.on and self._thread is not None and self._thread.is_alive()

    def ask(self, frame, crop=None, note=''):
        """Starts a question about FRAME (only the CROP box of it, if given).
        Returns False if one is already running."""
        if not self.on or self.busy:
            return False
        self._ask_id += 1
        ask_id = self._ask_id
        with self._lock:
            self._result = None
        frame = np.ascontiguousarray(frame[:, :, :3]).copy()
        self._thread = threading.Thread(target=self._run, args=(ask_id, frame, crop, note),
                                        daemon=True)
        self._thread.start()
        return True

    def take(self):
        """The finished Reply (or Exception) once, else None."""
        if not self.on:
            return None
        with self._lock:
            result, self._result = self._result, None
        return result

    def forget(self):
        """Drops whatever the running question comes back with."""
        if self.on:
            self._ask_id += 1
            with self._lock:
                self._result = None

    def _run(self, ask_id, frame, crop, note):
        try:
            result = self.solve(frame, crop, note)
        except Exception as error:
            result = error
        with self._lock:
            if ask_id == self._ask_id:
                self._result = result

    def solve(self, frame, crop=None, note=''):
        """Asks the model and waits: a Reply. Raises on API errors."""
        if self.backend == 'ollama':
            png, (ox, oy), (fx, fy), width, height = prepare(
                frame, crop, OLLAMA_SIDE, pixels=OLLAMA_TARGET_PIXELS)
        else:
            png, (ox, oy), (fx, fy), width, height = prepare(frame, crop)
        text = f'The image is {width} x {height} pixels. Solve the quiz. {note}'.strip()
        started = time.monotonic()
        if self.backend == 'ollama':
            data = self._ask_ollama(png, text)
        else:
            data = self._ask_claude(png, text)
        seconds = time.monotonic() - started

        def back(x, y):
            return ox + x / fx, oy + y / fy

        fh, fw = frame.shape[:2]
        points = []
        for click in data.get('clicks', []):
            x, y = back(click['x'], click['y'])
            if 0 <= x < fw and 0 <= y < fh:
                points.append((x, y, click.get('what', '?')))
        confirm = None
        if data.get('press_confirm') and data.get('confirm_x', -1) >= 0 \
                and data.get('confirm_y', -1) >= 0:
            confirm = back(data['confirm_x'], data['confirm_y'])
        return Reply(data, points, confirm, seconds)


def _boxes(text):
    """[(label, [x0, y0, x1, y1])] from Qwen2.5-VL's box answer (JSON, often in
    a ```json block, sometimes one object instead of a list)."""
    if '```' in text:
        text = text.split('```json')[-1] if '```json' in text else text.split('```')[1]
        text = text.split('```')[0]
    try:
        found = json.loads(text)
    except ValueError:
        start, end = text.find('['), text.rfind(']')
        try:
            found = json.loads(text[start:end + 1]) if start >= 0 else []
        except ValueError:
            return []
    if isinstance(found, dict):
        found = [found]
    out = []
    for item in found if isinstance(found, list) else []:
        box = item.get('bbox_2d') if isinstance(item, dict) else None
        if isinstance(box, list) and len(box) == 4:
            out.append((str(item.get('label', '')), [float(v) for v in box]))
    return out


def _pair(wanted, boxes):
    """Each wanted text with its box: by label where the labels match, else in
    order (the model was asked to keep the order)."""
    def words(text):
        return set(text.lower().replace('the ', ' ').split())
    left = list(boxes)
    pairs = []
    for i, target in enumerate(wanted):
        match = next((b for b in left if words(b[0]) == words(target)), None)
        if match is None and len(boxes) == len(wanted):
            match = boxes[i] if boxes[i] in left else None
        if match is None and left and len(wanted) == 1:
            match = left[0]
        if match is not None:
            left.remove(match)
        pairs.append((target, match[1] if match else None))
    return pairs


def annotate(frame, reply):
    out = np.ascontiguousarray(frame[:, :, :3]).copy()
    for i, (x, y, what) in enumerate(reply.points):
        cv2.circle(out, (int(x), int(y)), 14, (180, 105, 255), 2)
        cv2.putText(out, f'{i + 1} {what}', (int(x) + 16, int(y) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 105, 255), 1)
    if reply.confirm:
        cv2.circle(out, (int(reply.confirm[0]), int(reply.confirm[1])), 10, (255, 255, 0), 2)
    cv2.putText(out, f'AI ({reply.confidence}%): {reply.question}'[:90], (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 0, 255), 2)
    return out


def setup(key):
    os.makedirs(os.path.dirname(SETTINGS), exist_ok=True)
    s = _settings()
    s.update({'api_key': key, 'enabled': True})
    s.setdefault('claude_model', MODEL)
    s.setdefault('effort', EFFORT)
    with open(SETTINGS, 'w', encoding='utf-8') as file:
        json.dump(s, file, indent=2)
    print(f'Saved to {SETTINGS}.')


def use(backend):
    os.makedirs(os.path.dirname(SETTINGS), exist_ok=True)
    s = _settings()
    s.update({'backend': backend, 'enabled': True})
    with open(SETTINGS, 'w', encoding='utf-8') as file:
        json.dump(s, file, indent=2)
    print(f'Backend set to {backend} in {SETTINGS}.')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='AI fallback for the pop-up-3 quiz.')
    parser.add_argument('image', nargs='?', help='Screenshot to ask about (no clicking).')
    parser.add_argument('--output', help='With IMAGE: save the answer drawn on it here.')
    parser.add_argument('--setup', metavar='KEY', help='Save an Anthropic API key.')
    parser.add_argument('--use', choices=('ollama', 'claude'), help='Pick the backend.')
    args = parser.parse_args()
    if args.setup:
        setup(args.setup)
    if args.use:
        use(args.use)
    if (args.setup or args.use) and not args.image:
        return
    if not args.image:
        parser.error('give a SCREENSHOT or --setup KEY')
    ai = QuizAI()
    if not ai.on:
        print(f'AI fallback is off: {ai.why_off}')
        return
    print(f'Backend: {ai.backend} ({ai.model}).')
    frame = cv2.imread(args.image)
    if frame is None:
        parser.error(f'cannot read {args.image}')
    from src.detection.letter_quiz import find_footer, panel_box, _TitleRenderer
    footer = find_footer(frame, _TitleRenderer())
    crop = panel_box(frame, footer) if footer else None
    print(f'Footer {"at " + str(footer[:2]) if footer else "not found; sending the whole frame"}.')
    reply = ai.solve(frame, crop)
    print(json.dumps(reply.data, indent=2))
    print(reply.summary())
    if args.output:
        cv2.imwrite(args.output, annotate(frame, reply))
        print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
