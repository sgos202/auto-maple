"""Timed buffs the user sets up in the app (the Buffs page).

Each buff has a name, a key, how often to cast it again and how many times
to press the key. The bot thread casts due buffs between its own actions
(service(), from Bot._main), so a buff never lands in the middle of a held
attack or a jump, and never while a mini-game or a game restart has the
screen. Saved to .settings/buffs.json.
"""

import json
import os
import threading
import time
import uuid

from src.common import vkeys

SETTINGS_PATH = os.path.join('.settings', 'buffs.json')
MIN_INTERVAL = 5            # s: anything faster is an attack, not a buff
MAX_INTERVAL = 6 * 3600
MAX_PRESSES = 5
CAST_PAUSE = 0.6            # s after a cast: the skill's animation
RETRY_SECONDS = 2.0         # a buff that could not be cast is tried again this soon


def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f'{hours}h {minutes:02d}m'
    if minutes:
        return f'{minutes}m {secs:02d}s'
    return f'{secs}s'


class BuffManager:
    def __init__(self, path=SETTINGS_PATH):
        self.path = path
        self._lock = threading.Lock()
        self.enabled = True             # master switch
        self.buffs = []                 # dicts: id, name, key, interval, presses, enabled
        self._last_cast = {}            # id -> monotonic time of the last cast
        self._next_try = {}             # id -> no attempt before this (after a failed one)
        self._load()

    # ---- settings ----
    def _load(self):
        try:
            with open(self.path, encoding='utf-8') as file:
                data = json.load(file)
        except (OSError, ValueError):
            return
        self.enabled = bool(data.get('enabled', True))
        for raw in data.get('buffs', []):
            try:
                self.buffs.append(self._clean(raw, raw.get('id')))
            except ValueError:
                continue

    def _save_locked(self):
        data = {'enabled': self.enabled, 'buffs': self.buffs}
        try:
            os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
            temp = self.path + '.tmp'
            with open(temp, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=1)
            os.replace(temp, self.path)
        except OSError as error:
            print(f'\n[!] Could not save buffs: {error}')

    @staticmethod
    def _clean(raw, buff_id=None):
        name = str(raw.get('name', '')).strip()
        if not name:
            raise ValueError('Give the buff a name.')
        if len(name) > 40:
            raise ValueError('Keep the name under 40 characters.')
        key = str(raw.get('key', '')).strip().lower()
        if key not in vkeys.KEY_MAP:
            raise ValueError(f"'{key}' is not a key Auto Maple can press.")
        try:
            interval = int(round(float(raw.get('interval', 0))))
        except (TypeError, ValueError):
            raise ValueError('Recast time must be a number of seconds.') from None
        if not MIN_INTERVAL <= interval <= MAX_INTERVAL:
            raise ValueError(f'Recast time must be between {MIN_INTERVAL} s and '
                             f'{MAX_INTERVAL // 3600} hours.')
        try:
            presses = int(raw.get('presses', 1))
        except (TypeError, ValueError):
            raise ValueError('Presses must be a whole number.') from None
        if not 1 <= presses <= MAX_PRESSES:
            raise ValueError(f'Presses must be between 1 and {MAX_PRESSES}.')
        return {'id': buff_id or uuid.uuid4().hex[:8], 'name': name, 'key': key,
                'interval': interval, 'presses': presses,
                'enabled': bool(raw.get('enabled', True))}

    def list(self):
        with self._lock:
            return [dict(buff) for buff in self.buffs]

    def add(self, name, key, interval, presses=1):
        buff = self._clean({'name': name, 'key': key, 'interval': interval,
                            'presses': presses})
        with self._lock:
            self.buffs.append(buff)
            self._save_locked()
        return buff['id']

    def update(self, buff_id, **changes):
        with self._lock:
            for index, buff in enumerate(self.buffs):
                if buff['id'] == buff_id:
                    merged = dict(buff, **changes)
                    self.buffs[index] = self._clean(merged, buff_id)
                    if 'interval' in changes:
                        self._next_try.pop(buff_id, None)
                    self._save_locked()
                    return
        raise KeyError(buff_id)

    def remove(self, buff_id):
        with self._lock:
            self.buffs = [b for b in self.buffs if b['id'] != buff_id]
            self._last_cast.pop(buff_id, None)
            self._next_try.pop(buff_id, None)
            self._save_locked()

    def set_enabled(self, enabled):
        with self._lock:
            self.enabled = bool(enabled)
            self._save_locked()

    def recast_soon(self, buff_id):
        """Forget the last cast: the buff goes out at the next chance."""
        with self._lock:
            self._last_cast.pop(buff_id, None)
            self._next_try.pop(buff_id, None)

    # ---- timing ----
    def seconds_left(self, buff_id, now=None):
        """Seconds until BUFF_ID is due (0 = due now), or None if unknown."""
        now = time.monotonic() if now is None else now
        with self._lock:
            buff = next((b for b in self.buffs if b['id'] == buff_id), None)
            if buff is None:
                return None
            last = self._last_cast.get(buff_id)
            return 0.0 if last is None else max(0.0, last + buff['interval'] - now)

    def due(self, now=None):
        """The enabled buffs whose recast time has come, most overdue first."""
        now = time.monotonic() if now is None else now
        with self._lock:
            if not self.enabled:
                return []
            due = []
            for buff in self.buffs:
                if not buff['enabled'] or now < self._next_try.get(buff['id'], 0):
                    continue
                last = self._last_cast.get(buff['id'])
                late = float('inf') if last is None else now - last - buff['interval']
                if late >= 0:
                    due.append((late, dict(buff)))
            return [buff for _, buff in sorted(due, key=lambda item: -item[0])]

    def mark_cast(self, buff_id, now=None):
        with self._lock:
            self._last_cast[buff_id] = time.monotonic() if now is None else now
            self._next_try.pop(buff_id, None)

    def mark_failed(self, buff_id, now=None):
        now = time.monotonic() if now is None else now
        with self._lock:
            self._next_try[buff_id] = now + RETRY_SECONDS

    # ---- casting (bot thread) ----
    def service(self, before_cast=None):
        """
        Casts every due buff. BEFORE_CAST() is called once before the first
        one (the bot lets go of movement and attack keys). Returns how many
        were cast. Only when the game is focused and nothing else (a
        mini-game, a game restart) has the screen.
        """
        due = self.due()
        if not due or not _safe_to_cast():
            return 0
        if before_cast is not None:
            before_cast()
        cast = 0
        for buff in due:
            if not _safe_to_cast():
                self.mark_failed(buff['id'])
                break
            try:
                vkeys.press(buff['key'], buff['presses'])
            except Exception as error:
                self.mark_failed(buff['id'])
                print(f"\n[!] Buff {buff['name']} ({buff['key'].upper()}) failed: {error}")
                continue
            self.mark_cast(buff['id'])
            cast += 1
            print(f"\n[~] Buff: {buff['name']} ({buff['key'].upper()}); "
                  f"next in {format_duration(buff['interval'])}.")
            time.sleep(CAST_PAUSE)
        return cast


def _safe_to_cast():
    from src.common import config
    if not config.enabled:
        return False
    try:
        from src.modules.health import is_game_focused     # False mid-restart too
        if not is_game_focused():
            return False
    except Exception:
        return False
    try:
        from src.modules import mini_games
        runner = mini_games.ACTIVE_RUNNER
        if runner is not None and runner.busy:
            return False
    except Exception:
        pass
    return True


BUFFS = BuffManager()
