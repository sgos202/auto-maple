"""
Telegram messages to your phone (with a screenshot of the game) when a
mini-game starts and ends, or when the bot pauses itself.

Setup, once:
 1. In Telegram, message @BotFather: /newbot, pick a name and a username
    ending in "bot". It replies with a token (keep it private).
 2. Open your new bot and send it any message ("hi").
 3. Run:  python -m src.modules.phone_alert --setup TOKEN
    It finds your chat, saves both in .settings/telegram.json (not in git) and
    sends a test message with a screenshot.

Test again any time:  python -m src.modules.phone_alert --test
Turn off: set "enabled": false in .settings/telegram.json (or delete the file).

Sending runs on its own thread, so the bot never waits on the network.
"""

import json
import os
import queue
import threading
import time

SETTINGS = '.settings/telegram.json'
API = 'https://api.telegram.org/bot{token}/{method}'
TIMEOUT = 15
JPEG_QUALITY = 80
MAX_WIDTH = 1280                # screenshots are shrunk to this width at most

_queue = queue.Queue(maxsize=20)
_thread = None
_settings = None
_warned = False


def _load():
    global _settings
    if _settings is None:
        try:
            with open(SETTINGS, encoding='utf-8') as file:
                _settings = json.load(file)
        except FileNotFoundError:
            _settings = {}
        except Exception as error:
            print(f'\n[!] Could not read {SETTINGS}: {error}')
            _settings = {}
    return _settings


def enabled():
    s = _load()
    return bool(s.get('token') and s.get('chat_id') and s.get('enabled', True))


def screenshot():
    """The game as JPEG bytes: the bot's latest capture, else the game window,
    else the whole screen. None if nothing could be taken."""
    import cv2
    import numpy as np
    frame = None
    try:
        from src.common import config
        capture = getattr(config, 'capture', None)
        frame = getattr(capture, 'frame', None)
    except Exception:
        pass
    if frame is None:
        try:
            import mss
            with mss.mss() as screen:
                try:
                    from src.detection.violetta_solver import _client_region
                    _, region = _client_region('MapleStory')
                except Exception:
                    region = screen.monitors[1]
                frame = np.asarray(screen.grab(region))[:, :, :3]
        except Exception:
            return None
    frame = np.ascontiguousarray(frame[:, :, :3])
    if frame.shape[1] > MAX_WIDTH:
        frame = cv2.resize(frame, None, fx=MAX_WIDTH / frame.shape[1],
                           fy=MAX_WIDTH / frame.shape[1], interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return jpeg.tobytes() if ok else None


def send(text, photo=False):
    """Queues TEXT (with a screenshot taken now if PHOTO) for your phone.
    Does nothing when Telegram isn't set up. Never blocks or raises."""
    global _thread
    if not enabled():
        return
    try:
        image = screenshot() if photo else None
        _queue.put_nowait((text, image))
    except queue.Full:
        return
    except Exception as error:
        print(f'\n[!] Phone alert not sent: {error}')
        return
    if _thread is None or not _thread.is_alive():
        _thread = threading.Thread(target=_main, name='phone alerts', daemon=True)
        _thread.start()


def _main():
    while True:
        text, image = _queue.get()
        _deliver(text, image)


def _deliver(text, image):
    """Sends now; True on success."""
    global _warned
    import requests
    s = _load()
    try:
        if image:
            response = requests.post(
                API.format(token=s['token'], method='sendPhoto'),
                data={'chat_id': s['chat_id'], 'caption': text[:1000]},
                files={'photo': ('game.jpg', image, 'image/jpeg')}, timeout=TIMEOUT)
        else:
            response = requests.post(
                API.format(token=s['token'], method='sendMessage'),
                data={'chat_id': s['chat_id'], 'text': text}, timeout=TIMEOUT)
        if response.ok:
            _warned = False
            return True
        reason = response.json().get('description', response.status_code)
    except Exception as error:
        reason = type(error).__name__       # (no token in the message)
    if not _warned:
        _warned = True
        print(f'\n[!] Telegram message failed ({reason}); the bot carries on.')
    return False


def setup(token):
    """Finds the chat of whoever last messaged the bot, saves the settings and
    sends a test message."""
    import requests
    global _settings
    token = token.strip()
    response = requests.get(API.format(token=token, method='getMe'), timeout=TIMEOUT)
    if not response.ok:
        print('That token was not accepted by Telegram - copy it again from @BotFather.')
        return False
    bot = response.json()['result']['username']
    updates = requests.get(API.format(token=token, method='getUpdates'),
                           timeout=TIMEOUT).json().get('result', [])
    chats = [u['message']['chat'] for u in updates if 'message' in u]
    if not chats:
        print(f'Found your bot @{bot}, but nobody has messaged it yet. Open '
              f'https://t.me/{bot} in Telegram, send it "hi", then run this again.')
        return False
    chat = chats[-1]
    os.makedirs(os.path.dirname(SETTINGS), exist_ok=True)
    _settings = {'token': token, 'chat_id': chat['id'], 'enabled': True,
                 'mini_game_start': True, 'mini_game_end': True, 'bot_paused': True}
    with open(SETTINGS, 'w', encoding='utf-8') as file:
        json.dump(_settings, file, indent=2)
    who = chat.get('first_name') or chat.get('title') or 'you'
    print(f'Saved: alerts from @{bot} go to {who}. Sending a test message...')
    return test()


def test():
    if not enabled():
        print(f'Telegram is not set up (run with --setup TOKEN), or is off in {SETTINGS}.')
        return False
    ok = _deliver('✅ Auto Maple alerts are working. Here is the screen right now.',
                  screenshot())
    print('Sent - check your phone.' if ok else 'Sending failed (see above).')
    return ok


def wants(kind):
    """Whether alerts of KIND ('mini_game_start', 'mini_game_end', 'bot_paused') are on."""
    return enabled() and _load().get(kind, True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Telegram alerts for Auto Maple.')
    parser.add_argument('--setup', metavar='TOKEN', help="the token from @BotFather")
    parser.add_argument('--test', action='store_true', help='send a test message')
    args = parser.parse_args()
    if args.setup:
        setup(args.setup)
    elif args.test:
        test()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
