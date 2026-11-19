"""A module for simulating low-level keyboard and mouse key presses."""

import ctypes
import time
from src.common import config, utils
from ctypes import wintypes
from random import random
import threading


user32 = ctypes.WinDLL('user32', use_last_error=True)

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_SCANCODE = 0x0008

MAPVK_VK_TO_VSC = 0

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSE_BUTTONS = {
    'left': (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
    'right': (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
}

#################################
#        Input settings         #
#################################
# True: keys are sent by their hardware scan code (KEYEVENTF_SCANCODE), the way
# a keyboard driver reports them, so games that read scan codes see the right
# key. False: the old behaviour (virtual-key code only), kept as a way back.
SCANCODE_MODE = True

# True: a held arrow key repeats its key-down (~30 a second after ~0.25 s) the
# way a real keyboard does; like a keyboard, only the last key pressed repeats.
# False: one key-down per hold (the old behaviour). The attack key has its own
# repeater (AttackHand) and is unaffected.
HOLD_REPEAT = False     # off since 19 Nov: ~30 extra key-downs a second per held
                        # arrow lagged the client and console, and fixed nothing
HOLD_REPEAT_DELAY = (0.22, 0.30)    # s after the press before repeats start
HOLD_REPEAT_GAP = (0.028, 0.040)    # s between repeats
HOLD_REPEAT_KEYS = {'left', 'right', 'up', 'down'}   # routine directions

# True: every SendInput call is logged as a [SEND] line showing exactly what was
# handed to the game client: the key, the action, and the raw INPUT record
# (wVk, wScan, dwFlags) plus SendInput's return value. Mouse events too.
LOG_SENDINPUT = False   # a debugging aid: 9,000+ lines a run buried the console
# Held arrow keys repeat ~30 times a second; set False to leave those out.
LOG_SENDINPUT_REPEATS = False
# Keys in QUIET_KEYS (the loot key) are pressed constantly; set False to leave
# them out of the [SEND] log.
LOG_SENDINPUT_QUIET_KEYS = True

# Keys a keyboard sends with an E0 prefix. Without KEYEVENTF_EXTENDEDKEY the
# arrow keys' scan codes are the numpad's (Left = numpad 4).
EXTENDED_KEYS = {
    'left', 'right', 'up', 'down',
    'insert', 'delete', 'home', 'end', 'page up', 'page down',
    'num lock',
}

# https://docs.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes?redirectedfrom=MSDN
KEY_MAP = {
    'left': 0x25,   # Arrow keys
    'up': 0x26,
    'right': 0x27,
    'down': 0x28,

    'backspace': 0x08,      # Special keys
    'tab': 0x09,
    'enter': 0x0D,
    'shift': 0x10,
    'ctrl': 0x11,
    'alt': 0x12,
    'caps lock': 0x14,
    'esc': 0x1B,
    'space': 0x20,
    'page up': 0x21,
    'page down': 0x22,
    'end': 0x23,
    'home': 0x24,
    'insert': 0x2D,
    'delete': 0x2E,

    '0': 0x30,      # Numbers
    '1': 0x31,
    '2': 0x32,
    '3': 0x33,
    '4': 0x34,
    '5': 0x35,
    '6': 0x36,
    '7': 0x37,
    '8': 0x38,
    '9': 0x39,

    'a': 0x41,      # Letters
    'b': 0x42,
    'c': 0x43,
    'd': 0x44,
    'e': 0x45,
    'f': 0x46,
    'g': 0x47,
    'h': 0x48,
    'i': 0x49,
    'j': 0x4A,
    'k': 0x4B,
    'l': 0x4C,
    'm': 0x4D,
    'n': 0x4E,
    'o': 0x4F,
    'p': 0x50,
    'q': 0x51,
    'r': 0x52,
    's': 0x53,
    't': 0x54,
    'u': 0x55,
    'v': 0x56,
    'w': 0x57,
    'x': 0x58,
    'y': 0x59,
    'z': 0x5A,

    'f1': 0x70,     # Functional keys
    'f2': 0x71,
    'f3': 0x72,
    'f4': 0x73,
    'f5': 0x74,
    'f6': 0x75,
    'f7': 0x76,
    'f8': 0x77,
    'f9': 0x78,
    'f10': 0x79,
    'f11': 0x7A,
    'f12': 0x7B,
    'num lock': 0x90,
    'scroll lock': 0x91,

    ';': 0xBA,      # Special characters
    '=': 0xBB,
    ',': 0xBC,
    '-': 0xBD,
    '.': 0xBE,
    '/': 0xBF,
    '`': 0xC0,
    '[': 0xDB,
    '\\': 0xDC,
    ']': 0xDD,
    "'": 0xDE
}

HORIZONTAL_MOVEMENT_KEYS = {
    'left': 'left',
    'right': 'right'
}
_INPUT_STATE_LOCK = threading.Lock()
_ACTIVE_INPUT_KEYS = set()
_LAST_INPUT_EVENT = 'No simulated key events yet.'
# Keys pressed so often that a [KEY] line for each would bury the log (the loot
# pickup key); whoever presses them logs a summary instead.
QUIET_KEYS = set()


def input_status():
    """Returns keys currently held by Auto Maple and its most recent key event."""

    with _INPUT_STATE_LOCK:
        return tuple(sorted(_ACTIVE_INPUT_KEYS)), _LAST_INPUT_EVENT


def _record_input_event(key, action, semantic_key=None):
    global _LAST_INPUT_EVENT
    with _INPUT_STATE_LOCK:
        if action == 'DOWN':
            _ACTIVE_INPUT_KEYS.add(key)
        else:
            _ACTIVE_INPUT_KEYS.discard(key)
        detail = f' (routine direction {semantic_key})' if semantic_key else ''
        _LAST_INPUT_EVENT = f'{action} {key.upper()}{detail}'
    # The [SEND] line written by _send_key_input already logs this event.


_KEY_FLAG_NAMES = (
    (KEYEVENTF_EXTENDEDKEY, 'EXTENDEDKEY'),
    (KEYEVENTF_KEYUP, 'KEYUP'),
    (KEYEVENTF_UNICODE, 'UNICODE'),
    (KEYEVENTF_SCANCODE, 'SCANCODE'),
)
_MOUSE_FLAG_NAMES = {
    MOUSEEVENTF_LEFTDOWN: 'LEFTDOWN',
    MOUSEEVENTF_LEFTUP: 'LEFTUP',
    MOUSEEVENTF_RIGHTDOWN: 'RIGHTDOWN',
    MOUSEEVENTF_RIGHTUP: 'RIGHTUP',
}


def _flag_text(flags):
    names = [name for bit, name in _KEY_FLAG_NAMES if flags & bit]
    return '|'.join(names) or 'none'


def _log_key_send(x, mapped_key, action, semantic_key, result):
    """Logs one keyboard SendInput call and the INPUT record it carried."""

    if not LOG_SENDINPUT:
        return
    if action == 'REPEAT' and not LOG_SENDINPUT_REPEATS:
        return
    if mapped_key in QUIET_KEYS and not LOG_SENDINPUT_QUIET_KEYS:
        return
    ki = x.ki
    detail = f' (routine direction {semantic_key})' if semantic_key else ''
    print(
        f'[SEND] KEY {action:<6} {mapped_key.upper()}{detail} -> '
        f'INPUT_KEYBOARD wVk=0x{ki.wVk:02X} wScan=0x{ki.wScan:02X} '
        f'dwFlags=0x{ki.dwFlags:04X} ({_flag_text(ki.dwFlags)}) | {result}'
    )


def set_horizontal_movement_keys(left, right):
    """Maps routine left/right movement directions to configured game keys."""

    left = left.lower()
    right = right.lower()
    if left not in KEY_MAP or right not in KEY_MAP or left == right:
        raise ValueError('Horizontal movement keys must be distinct supported keys.')
    HORIZONTAL_MOVEMENT_KEYS.update({'left': left, 'right': right})


#################################
#     C Struct Definitions      #
#################################
wintypes.ULONG_PTR = wintypes.WPARAM


class KeyboardInput(ctypes.Structure):
    _fields_ = (('wVk', wintypes.WORD),
                ('wScan', wintypes.WORD),
                ('dwFlags', wintypes.DWORD),
                ('time', wintypes.DWORD),
                ('dwExtraInfo', wintypes.ULONG_PTR))

    def __init__(self, *args, **kwargs):
        super(KeyboardInput, self).__init__(*args, **kwargs)
        if not self.dwFlags & KEYEVENTF_UNICODE and not self.wScan:
            self.wScan = user32.MapVirtualKeyExW(self.wVk, MAPVK_VK_TO_VSC, 0)


class MouseInput(ctypes.Structure):
    _fields_ = (('dx', wintypes.LONG),
                ('dy', wintypes.LONG),
                ('mouseData', wintypes.DWORD),
                ('dwFlags', wintypes.DWORD),
                ('time', wintypes.DWORD),
                ('dwExtraInfo', wintypes.ULONG_PTR))


class HardwareInput(ctypes.Structure):
    _fields_ = (('uMsg', wintypes.DWORD),
                ('wParamL', wintypes.WORD),
                ('wParamH', wintypes.WORD))


class Input(ctypes.Structure):
    class _Input(ctypes.Union):
        _fields_ = (('ki', KeyboardInput),
                    ('mi', MouseInput),
                    ('hi', HardwareInput))

    _anonymous_ = ('_input',)
    _fields_ = (('type', wintypes.DWORD),
                ('_input', _Input))


LPINPUT = ctypes.POINTER(Input)


def err_check(result, _, args):
    if result == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    else:
        return args


user32.SendInput.errcheck = err_check
user32.SendInput.argtypes = (wintypes.UINT, LPINPUT, ctypes.c_int)


#################################
#           Functions           #
#################################
def _key_input(mapped_key, up=False):
    """The INPUT record for a key-down (or key-up when UP) of MAPPED_KEY."""

    vk = KEY_MAP[mapped_key]
    flags = KEYEVENTF_KEYUP if up else 0
    if SCANCODE_MODE:
        scan = user32.MapVirtualKeyExW(vk, MAPVK_VK_TO_VSC, 0)
        if scan:
            flags |= KEYEVENTF_SCANCODE
            if mapped_key in EXTENDED_KEYS:
                flags |= KEYEVENTF_EXTENDEDKEY
            return Input(type=INPUT_KEYBOARD,
                         ki=KeyboardInput(wVk=0, wScan=scan, dwFlags=flags))
        # No scan code for this key on this layout: send it the old way.
    return Input(type=INPUT_KEYBOARD, ki=KeyboardInput(wVk=vk, dwFlags=flags))


def _send_key_input(mapped_key, up=False, action=None, semantic_key=None):
    """Sends one key event and logs exactly what went to SendInput. ACTION
    labels the log line (DOWN / UP / REPEAT); it defaults from UP."""

    x = _key_input(mapped_key, up)
    action = action or ('UP' if up else 'DOWN')
    try:
        sent = user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))
    except OSError as e:
        _log_key_send(x, mapped_key, action, semantic_key, f'REFUSED: {e}')
        raise
    _log_key_send(x, mapped_key, action, semantic_key, f'SendInput -> {sent}')


class _HoldRepeater:
    """
    Repeats the key-down of a held arrow key, like a keyboard's typematic
    repeat: after HOLD_REPEAT_DELAY, every HOLD_REPEAT_GAP, until the key is
    released or another key is pressed (a keyboard only repeats the last key
    pressed, and does not go back to an earlier one). Repeats stop while
    Auto Maple is paused.
    """

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._cond = threading.Condition()
        self._key = None            # mapped key being repeated
        self._due = 0.0
        self._thread = None
        self.repeats_sent = 0

    def pressed(self, semantic_key, mapped_key):
        with self._cond:
            if HOLD_REPEAT and semantic_key in HOLD_REPEAT_KEYS:
                if self._key == mapped_key:
                    return      # a re-sent key-down of the held key: keep the rhythm
                self._key = mapped_key
                self._due = self._clock() + _between(HOLD_REPEAT_DELAY)
                self._start()
            else:
                self._key = None
            self._cond.notify_all()

    def released(self, mapped_key):
        with self._cond:
            if self._key == mapped_key:
                self._key = None
                self._cond.notify_all()

    def stop(self):
        with self._cond:
            self._key = None
            self._cond.notify_all()

    @property
    def key(self):
        return self._key

    def _start(self):
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name='HoldRepeater',
                                            daemon=True)
            self._thread.start()

    def _run(self):
        with self._cond:
            while True:
                if self._key is None:
                    self._cond.wait()
                    continue
                wait = self._due - self._clock()
                if wait > 0:
                    self._cond.wait(wait)
                    continue
                # Sent under the lock: once released() returns, no repeat of
                # that key can still arrive after its key-up.
                if getattr(config, 'enabled', False):
                    try:
                        _send_key_input(self._key, action='REPEAT')
                        self.repeats_sent += 1
                    except Exception:
                        pass
                self._due = self._clock() + _between(HOLD_REPEAT_GAP)


def _between(span):
    return span[0] + (span[1] - span[0]) * random()


_REPEATER = _HoldRepeater()


def _send_key_down(key):
    key = key.lower()
    mapped_key = HORIZONTAL_MOVEMENT_KEYS.get(key, key)
    if mapped_key not in KEY_MAP:
        print(f"Invalid keyboard input: '{key}'.")
    else:
        semantic = key if mapped_key != key else None
        _send_key_input(mapped_key, semantic_key=semantic)
        _record_input_event(mapped_key, 'DOWN', semantic)
        _REPEATER.pressed(key, mapped_key)


def key_repeat(key):
    """
    Sends another key-down for a key that is already held, the way a real
    keyboard repeats a held key (~30 times a second). Logged as REPEAT
    (see LOG_SENDINPUT_REPEATS). Does nothing while Auto Maple is paused.
    """
    if not getattr(config, 'enabled', False):
        return
    key = key.lower()
    mapped_key = HORIZONTAL_MOVEMENT_KEYS.get(key, key)
    if mapped_key in KEY_MAP:
        _send_key_input(mapped_key, action='REPEAT')


@utils.run_if_enabled
def key_down(key):
    """
    Simulates a key-down action. Can be cancelled by Bot.toggle_enabled.
    :param key:     The key to press.
    :return:        None
    """

    _send_key_down(key)


def key_up(key):
    """
    Simulates a key-up action. Cannot be cancelled by Bot.toggle_enabled.
    This is to ensure no keys are left in the 'down' state when the program pauses.
    :param key:     The key to press.
    :return:        None
    """

    key = key.lower()
    mapped_key = HORIZONTAL_MOVEMENT_KEYS.get(key, key)
    if mapped_key not in KEY_MAP:
        print(f"Invalid keyboard input: '{key}'.")
    else:
        _REPEATER.released(mapped_key)      # before the key-up: no repeat after it
        semantic = key if mapped_key != key else None
        _send_key_input(mapped_key, up=True, semantic_key=semantic)
        _record_input_event(mapped_key, 'UP', semantic)


@utils.run_if_enabled
def press(key, n, down_time=0.05, up_time=0.1, allow_paused=False):
    """
    Presses KEY N times, holding it for DOWN_TIME seconds, and releasing for UP_TIME seconds.
    :param key:         The keyboard input to press.
    :param n:           Number of times to press KEY.
    :param down_time:   Duration of down-press (in seconds).
    :param up_time:     Duration of release (in seconds).
    :param allow_paused:    Whether this explicit key press may run while paused.
    :return:            None
    """

    for _ in range(n):
        if allow_paused:
            _send_key_down(key)
        else:
            key_down(key)
        time.sleep(down_time * (0.8 + 0.4 * random()))
        key_up(key)
        time.sleep(up_time * (0.8 + 0.4 * random()))


#################################
#             Mouse             #
#################################
_HELD_BUTTONS = set()


def _send_mouse(flags):
    """Sends a mouse button event through SendInput. A refused event is
    reported, not raised: the old mouse_event calls failed silently too."""

    x = Input(type=INPUT_MOUSE, mi=MouseInput(dwFlags=flags))
    name = _MOUSE_FLAG_NAMES.get(flags, 'flags')
    try:
        sent = user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))
    except OSError as e:
        print(f'[MOUSE] SendInput refused the event: {e}')
        if LOG_SENDINPUT:
            print(f'[SEND] MOUSE {name} -> INPUT_MOUSE dwFlags=0x{flags:04X} | REFUSED: {e}')
        return False
    if LOG_SENDINPUT:
        print(f'[SEND] MOUSE {name} -> INPUT_MOUSE dwFlags=0x{flags:04X} | SendInput -> {sent}')
    return True


def move_cursor(x, y, strict=False):
    """Puts the cursor on screen pixel (X, Y). STRICT raises if Windows refuses."""

    if not user32.SetCursorPos(int(x), int(y)):
        if strict:
            raise ctypes.WinError(ctypes.get_last_error())
        return False
    return True


def mouse_down(button='left'):
    _send_mouse(MOUSE_BUTTONS[button][0])
    with _INPUT_STATE_LOCK:
        _HELD_BUTTONS.add(button)


def mouse_up(button='left'):
    _send_mouse(MOUSE_BUTTONS[button][1])
    with _INPUT_STATE_LOCK:
        _HELD_BUTTONS.discard(button)


@utils.run_if_enabled
def click(position=None, button='left', settle=None, hold=None, allow_paused=False):
    """
    Simulate a mouse click with BUTTON at POSITION (screen pixels; None clicks
    where the cursor already is).
    :param position:    The (x, y) position at which to click, or None.
    :param button:      Either the left or right mouse button.
    :param settle:      (min, max) s to wait after moving before pressing, or None.
    :param hold:        (min, max) s to hold the button down, or None.
    :param allow_paused:    Whether this click may run while Auto Maple is paused
                        (mini-games and the game restart click while paused).
    :return:            None
    """

    if button not in MOUSE_BUTTONS:
        print(f"'{button}' is not a valid mouse button.")
        return
    if position is not None:
        move_cursor(position[0], position[1])
        if settle:
            time.sleep(_between(settle))
    mouse_down(button)
    try:
        if hold:
            time.sleep(_between(hold))
    finally:
        mouse_up(button)
    where = f' at ({int(position[0])}, {int(position[1])})' if position is not None else ''
    print(f'[MOUSE] {button.upper()} click{where} via SendInput.')


def release_all_keys():
    """Lets go of every key and mouse button Auto Maple is holding (after a
    stall or crash) and stops any held-key repeat."""
    _REPEATER.stop()
    with _INPUT_STATE_LOCK:
        held = list(_ACTIVE_INPUT_KEYS)
        buttons = list(_HELD_BUTTONS)
    for key in held:
        try:
            key_up(key)
        except Exception:
            pass
    for button in buttons:
        try:
            mouse_up(button)
        except Exception:
            pass
    return held + [f'mouse {b}' for b in buttons]
