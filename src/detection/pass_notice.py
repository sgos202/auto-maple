"""
The notice the game shows after a mini-game is won:

    "You have successfully passed the Lie Detector Test.
     Thank you for participating!"                          [ OK ]

It stays on screen until OK is clicked, so the routine can't farm behind it
(reference: assets/post_violetta.png, 2026-11-17). find() locates it in a
screenshot from its text; click_ok() waits for it in the live game and clicks OK.

The two pictures matched are cut from the reference at the game's normal size:
assets/pass_notice/text.png (the two white lines) and ok.png (the button).

Check a screenshot:  python -m src.detection.pass_notice --image assets/post_violetta.png
Try it on the game:  python -m src.detection.pass_notice [--click]
"""

import time

import cv2
import numpy as np

ASSET_DIR = 'assets/pass_notice'
# The game's normal size first (the usual case, and find() stops at a sure match),
# then 0.80 - 1.25 in small steps: the tiny pixel text only matches close to its size.
SCALES = (1.0,) + tuple(s for s in (round(0.8 + 0.025 * i, 3) for i in range(19)) if s != 1.0)
TEXT_THRESHOLD = 0.65           # the reference scores 1.0; other screens <= 0.50 in testing
SURE_SCORE = 0.9                # no need to try the other sizes after this
OK_THRESHOLD = 0.6              # looser: the button brightens under the mouse
OK_OFFSET = (199.5, 86.5)       # OK's centre from the text picture's top-left, at scale 1
OK_SEARCH = 14                  # px around that spot to look for the button (scale 1)

WAIT_SECONDS = 15.0             # how long after the game the notice may take to appear
CHECK_SECONDS = 0.3
CLICK_TRIES = 3
GONE_SECONDS = 0.8              # after a click, the notice should have closed by now


class Notice:
    def __init__(self, score, scale, ok, ok_score):
        self.score = score          # text match, 0-1
        self.scale = scale
        self.ok = ok                # (x, y) centre of the OK button in the frame
        self.ok_score = ok_score    # 0 when the button was placed from the text alone

    def __repr__(self):
        return (f'Notice(score {self.score:.2f}, scale {self.scale}, '
                f'OK at {self.ok}, button {self.ok_score:.2f})')


_templates = None


def _load():
    global _templates
    if _templates is None:
        text = cv2.imread(f'{ASSET_DIR}/text.png', cv2.IMREAD_GRAYSCALE)
        ok = cv2.imread(f'{ASSET_DIR}/ok.png', cv2.IMREAD_GRAYSCALE)
        if text is None or ok is None:
            raise FileNotFoundError(f'{ASSET_DIR}/text.png and ok.png are needed.')
        _templates = (text, ok)
    return _templates


def _resized(image, scale):
    if scale == 1:
        return image
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)


def find(frame, scales=SCALES):
    """The pass notice in FRAME (BGR or BGRA) as a Notice, or None if it isn't there."""
    if frame is None or frame.size == 0:
        return None
    text, ok = _load()
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
    best = None
    for scale in scales:
        template = _resized(text, scale)
        if template.shape[0] > gray.shape[0] or template.shape[1] > gray.shape[1]:
            continue
        _, score, _, spot = cv2.minMaxLoc(cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED))
        if best is None or score > best[0]:
            best = (score, scale, spot)
        if score >= SURE_SCORE:
            break
    if best is None or best[0] < TEXT_THRESHOLD:
        return None
    score, scale, (x, y) = best

    # The button sits at a fixed place under the text; look for it right there
    # so the click lands on its middle even if the size was read a little off.
    cx, cy = x + OK_OFFSET[0] * scale, y + OK_OFFSET[1] * scale
    button = _resized(ok, scale)
    h, w = button.shape
    reach = round(OK_SEARCH * scale)
    x0, y0 = max(0, round(cx - w / 2) - reach), max(0, round(cy - h / 2) - reach)
    area = gray[y0:y0 + h + 2 * reach, x0:x0 + w + 2 * reach]
    ok_score = 0.0
    if area.shape[0] >= h and area.shape[1] >= w:
        _, found, _, (bx, by) = cv2.minMaxLoc(cv2.matchTemplate(area, button, cv2.TM_CCOEFF_NORMED))
        if found >= OK_THRESHOLD:
            ok_score = float(found)
            cx, cy = x0 + bx + w / 2, y0 + by + h / 2
    if not (0 <= cx < gray.shape[1] and 0 <= cy < gray.shape[0]):
        return None     # the button is off the edge of the picture: nothing to click
    return Notice(float(score), scale, (round(cx), round(cy)), ok_score)


def click_ok(title='MapleStory', wait=WAIT_SECONDS, stop=None, click=True, grab=None,
             press=None, sleep=time.sleep):
    """
    Waits up to WAIT seconds for the pass notice in the game window TITLE and
    clicks its OK button. Returns 'passed' once the notice has closed, 'stuck' if
    it is still there after CLICK_TRIES clicks, 'stopped', or None if it never
    showed (the game was not won, or the notice looks different).

    GRAB() -> (frame, (left, top)) and PRESS(x, y) replace the screen and the
    mouse in tests.
    """
    screen = None
    if grab is None:
        import ctypes
        import mss
        from src.detection.violetta_solver import _client_region
        ctypes.windll.user32.SetProcessDPIAware()
        screen = mss.mss()

        def grab():
            _, region = _client_region(title)
            return np.asarray(screen.grab(region))[:, :, :3], (region['left'], region['top'])
    if press is None:
        from src.detection.violetta_solver import _click as press

    try:
        clicks = 0
        clicked_at = None
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            if stop is not None and stop.is_set():
                return 'stopped'
            try:
                frame, (left, top) = grab()
            except Exception:       # window minimised or gone for a moment
                sleep(CHECK_SECONDS)
                continue
            notice = find(frame)
            if notice is None:
                if clicks:
                    print('[PASS NOTICE] Closed.')
                    return 'passed'
                sleep(CHECK_SECONDS)
                continue
            if not click:
                print(f'[PASS NOTICE] Found (not clicking): {notice}')
                return 'passed'
            if clicked_at is not None and time.monotonic() - clicked_at < GONE_SECONDS:
                sleep(0.1)          # give the last click time to close it
                continue
            if clicks >= CLICK_TRIES:
                break
            x, y = left + notice.ok[0], top + notice.ok[1]
            print(f'[PASS NOTICE] "Passed the Lie Detector Test" notice found '
                  f'(score {notice.score:.2f}); clicking OK at screen ({x}, {y}).')
            press(x, y)
            clicks += 1
            clicked_at = time.monotonic()
            deadline = max(deadline, clicked_at + GONE_SECONDS + 1.0)
        if clicks:
            print(f'[PASS NOTICE] Still on screen after {clicks} click(s) on OK.')
            return 'stuck'
        return None
    finally:
        if screen is not None:
            screen.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Find the "passed the Lie Detector Test" notice.')
    parser.add_argument('--image', help='Check a screenshot instead of the game.')
    parser.add_argument('--title', default='MapleStory', help='Game window title.')
    parser.add_argument('--click', action='store_true', help='Live: click OK.')
    parser.add_argument('--wait', type=float, default=WAIT_SECONDS, help='Live: seconds to wait.')
    args = parser.parse_args()
    if args.image:
        print(find(cv2.imread(args.image)))
    else:
        print(click_ok(args.title, wait=args.wait, click=args.click))


if __name__ == '__main__':
    main()
