"""
Keeps long runs healthy: watches the machine's free commit memory (RAM + page
file) and this process's own memory.

- Free commit runs low (another program leaking, e.g. the game client after a
  day of running): pause the bot before allocations start failing, and say why.
  Every image the bot handles is a fresh allocation, so at ~0 free commit the
  capture fails, the minimap position is lost and the bot walks blind.
- This process keeps growing: report it (a leak in the bot itself).

Uses only the Windows API on this process and the system totals; it never opens
the game's process (the game's anti-cheat may not like that).
"""

import ctypes
import gc
import threading
import time
from ctypes import wintypes

from src.common import config


class _PerformanceInfo(ctypes.Structure):
    _fields_ = [
        ('cb', wintypes.DWORD),
        ('CommitTotal', ctypes.c_size_t), ('CommitLimit', ctypes.c_size_t),
        ('CommitPeak', ctypes.c_size_t), ('PhysicalTotal', ctypes.c_size_t),
        ('PhysicalAvailable', ctypes.c_size_t), ('SystemCache', ctypes.c_size_t),
        ('KernelTotal', ctypes.c_size_t), ('KernelPaged', ctypes.c_size_t),
        ('KernelNonpaged', ctypes.c_size_t), ('PageSize', ctypes.c_size_t),
        ('HandleCount', wintypes.DWORD), ('ProcessCount', wintypes.DWORD),
        ('ThreadCount', wintypes.DWORD),
    ]


class _ProcessMemory(ctypes.Structure):
    _fields_ = [
        ('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
        ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
        ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t),
        ('PrivateUsage', ctypes.c_size_t),
    ]


MB = 1024 * 1024


def system_commit():
    """(free commit MB, commit limit MB), or None if it can't be read."""
    try:
        info = _PerformanceInfo()
        info.cb = ctypes.sizeof(info)
        psapi = ctypes.windll.psapi
        psapi.GetPerformanceInfo.argtypes = [ctypes.c_void_p, wintypes.DWORD]
        if not psapi.GetPerformanceInfo(ctypes.byref(info), info.cb):
            return None
        page = info.PageSize
        return ((info.CommitLimit - info.CommitTotal) * page / MB,
                info.CommitLimit * page / MB)
    except Exception:
        return None


def own_memory_mb():
    """This process's private memory in MB, or None."""
    try:
        counters = _ProcessMemory()
        counters.cb = ctypes.sizeof(counters)
        kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return None
        return counters.PrivateUsage / MB
    except Exception:
        return None


def _process_table():
    """[(name, committed bytes, handle count)] for every process, from the
    system's own process table in one call (no process is opened)."""
    ntdll = ctypes.windll.ntdll
    size = wintypes.ULONG(1 << 20)
    for _ in range(6):
        buffer = ctypes.create_string_buffer(size.value)
        status = ntdll.NtQuerySystemInformation(
            5, buffer, size.value, ctypes.byref(size)) & 0xFFFFFFFF
        if status == 0:
            break
        if status != 0xC0000004:        # anything but "buffer too small"
            return []
        size = wintypes.ULONG(size.value + (1 << 18))
    else:
        return []
    rows = []
    offset = 0
    raw = buffer.raw
    while True:
        # 64-bit SYSTEM_PROCESS_INFORMATION: next entry, image name
        # (UNICODE_STRING at 0x38), handle count at 0x60, committed at 0xB8
        next_entry = int.from_bytes(raw[offset:offset + 4], 'little')
        length = int.from_bytes(raw[offset + 0x38:offset + 0x3A], 'little')
        pointer = int.from_bytes(raw[offset + 0x40:offset + 0x48], 'little')
        handles = int.from_bytes(raw[offset + 0x60:offset + 0x64], 'little')
        private = int.from_bytes(raw[offset + 0xB8:offset + 0xC0], 'little')
        name = ctypes.wstring_at(pointer, length // 2) if pointer and length else 'System'
        if name.lower().endswith('.exe'):
            name = name[:-4]
        rows.append((name, private, handles))
        if not next_entry:
            break
        offset += next_entry
    return rows


def game_handles(name='MapleStory'):
    """Open handles of the game client (None if it isn't running or the table
    can't be read). 2026-11-15: the client leaked ~72,000 handles an hour on
    a busy map, with the bot paused, and Windows ran out of memory; the
    processes themselves held only ~10 of the 31 GB committed."""
    try:
        counts = [handles for proc, _, handles in _process_table()
                  if proc.lower() == name.lower()]
    except Exception:
        return None
    return sum(counts) if counts else None


def top_memory_users(count=5):
    """
    The programs holding the most memory, as [(name, MB, processes)], biggest
    first, with a program's processes added up (a browser is dozens). Read
    from the system's own process table in one call: no process is opened
    (the game's anti-cheat is left alone). [] if it can't be read.
    """
    try:
        totals = {}
        for name, private, _ in _process_table():
            entry = totals.setdefault(name, [0, 0])
            entry[0] += private
            entry[1] += 1
        ranked = sorted(totals.items(), key=lambda item: item[1][0], reverse=True)
        return [(name, used / MB, processes) for name, (used, processes) in ranked[:count]]
    except Exception:
        return []


class MemoryGuard:
    CHECK_SECONDS = 15
    LOW_CHECK_SECONDS = 5       # while paused for memory: look more often
    # Pause below this much free commit. The bot itself needs little (a few
    # frames at a time); 1200 was needlessly early on a PC that sits near its
    # limit all day (16 GB RAM, editors and browsers open).
    PAUSE_FREE_MB = 600
    RESUME_FREE_MB = 1200       # back above this: farming resumes by itself...
    RESUME_CHECKS = 2           # ...once it has held for this many checks
    REPORT_SECONDS = 600        # a status line in the console this often...
    PAUSED_REPORT_SECONDS = 3600    # ...and this often while Auto Maple is paused
    IDLE_ALERT_SECONDS = 600    # paused this long after farming: warn (+ phone alert)
    OWN_GROWTH_WARN_MB = 400    # this much above the settled baseline: likely a leak
    # The game client's handle leak (see game_handles): warn this early, and
    # again every GAME_WARN_SECONDS while it is that high.
    GAME_HANDLES_WARN = 60000
    GAME_WARN_SECONDS = 900

    def __init__(self):
        self.thread = None
        self._baseline = None
        self._started = None
        self._last_report = 0.0
        self._paused_for_memory = False
        self._warned_growth = False
        self._recovered_checks = 0
        self._waiting_focus_said = False
        self._game_seen = None          # (time, handles) for the leak rate
        self._game_warned = float('-inf')
        self._paused_since = None
        self._ever_enabled = False
        self._idle_alerted = False

    def _game_note(self, now):
        """(' MapleStory holds N handles (+R/hour).', N), or ('', None)."""
        handles = game_handles()
        if handles is None:
            return '', None
        rate = None
        seen = self.__dict__.get('_game_seen')
        if seen is not None and now - seen[0] >= 60:
            rate = (handles - seen[1]) * 3600 / (now - seen[0])
        if seen is None or now - seen[0] >= 600:
            self._game_seen = (now, handles)
        text = f' MapleStory holds {handles:,} handles'
        if rate is not None:
            text += f' ({rate:+,.0f}/hour)'
        return text + '.', handles

    def _track_pause(self, now):
        """Seconds Auto Maple has been paused (None while it is farming)."""
        if getattr(config, 'enabled', False):
            self._ever_enabled = True
            self._paused_since = None
            self._idle_alerted = False
            return None
        if self._paused_since is None:
            self._paused_since = now
        return now - self._paused_since

    @staticmethod
    def _start_key():
        try:
            return config.listener.config['Start/stop'].upper()
        except Exception:
            return 'the Start/stop key'

    @staticmethod
    def _duration(seconds):
        minutes = int(seconds // 60)
        return f'{minutes // 60} h {minutes % 60:02d} min' if minutes >= 60 else f'{minutes} min'

    def _status_line(self, paused_for, commit, own, game_text):
        if paused_for is None:
            state = 'Status: farming.'
        elif not self._ever_enabled:
            state = f'Status: not started yet (press {self._start_key()} in the game).'
        else:
            reason = ' because Windows is low on memory' if self._paused_for_memory else ''
            state = (f'Status: PAUSED{reason} for {self._duration(paused_for)}; nothing '
                     f'is being farmed. Press {self._start_key()} in the game to carry on.')
        if commit:
            health = 'LOW' if commit[0] < self.RESUME_FREE_MB else 'OK'
            memory = (f'Memory {health}: {commit[0]:,.0f} of {commit[1]:,.0f} MB free')
        else:
            memory = 'Memory: unknown'
        if own is not None:
            memory += f', bot {own:.0f} MB'
        return f'{state} {memory}.{game_text}'

    def _alert_idle(self, paused_for):
        """Auto Maple stopped farming and nobody restarted it: say so loudly."""
        reason = ('Windows was low on memory and the game was not in front to resume'
                  if self._paused_for_memory else 'it was switched off')
        print(f'\n[!] Auto Maple has been paused for {self._duration(paused_for)} '
              f'({reason}). Nothing is being farmed until you press '
              f'{self._start_key()} in the game.')
        try:
            from src.modules import phone_alert
            if phone_alert.wants('bot_paused'):
                phone_alert.send(f'⏸ Auto Maple has been paused for '
                                 f'{self._duration(paused_for)} ({reason}). '
                                 'Nothing is being farmed.', photo=True)
        except Exception as error:
            print(f'\n[!] Could not send the phone alert: {error}')

    def user_toggled(self):
        """The Start/stop key was pressed: the user is in charge again, so a
        pause made here is no longer this guard's to undo."""
        self._paused_for_memory = False
        self._recovered_checks = 0

    def start(self):
        if self.thread is not None:
            return
        self._started = time.monotonic()
        self.thread = threading.Thread(target=self._main, daemon=True)
        self.thread.start()

    def _main(self):
        while True:
            try:
                self.check()
            except Exception as error:
                print(f'\n[!] Memory check failed: {error}')
            time.sleep(self.LOW_CHECK_SECONDS if self._paused_for_memory
                       else self.CHECK_SECONDS)

    def check(self, now=None):
        now = time.monotonic() if now is None else now
        commit = system_commit()
        own = own_memory_mb()
        if own is not None and self._baseline is None \
                and now - self._started >= 120:
            self._baseline = own    # after start-up (models, samples) settles
        game_text, handles = self._game_note(now)
        paused_for = self._track_pause(now)
        # 2026-11-15: the bot sat paused for 7 h while this line kept coming
        # every 10 min, which read like an ongoing memory problem. Say plainly
        # whether it is farming, and only report hourly while it is paused.
        interval = self.REPORT_SECONDS if paused_for is None else self.PAUSED_REPORT_SECONDS
        if now - self._last_report >= interval:
            self._last_report = now
            print('\n[~] ' + self._status_line(paused_for, commit, own, game_text))
        if paused_for is not None and paused_for >= self.IDLE_ALERT_SECONDS \
                and not self._idle_alerted and self._ever_enabled:
            self._idle_alerted = True
            self._alert_idle(paused_for)
        if (handles is not None and handles >= self.GAME_HANDLES_WARN
                and now - self.__dict__.get('_game_warned', float('-inf'))
                >= self.GAME_WARN_SECONDS):
            self._game_warned = now
            print(f'\n[!] The MapleStory client is leaking memory:{game_text} '
                  'That memory is only freed when the game closes. Restart '
                  'MapleStory when convenient, or Windows runs out of memory and '
                  'farming pauses.')
        if own is not None and self._baseline is not None and not self._warned_growth \
                and own - self._baseline >= self.OWN_GROWTH_WARN_MB:
            self._warned_growth = True
            gc.collect()
            print(f'\n[!] The bot has grown from {self._baseline:.0f} MB to {own:.0f} MB '
                  'since start-up; restart it when convenient and report this.')
        if commit is None:
            return
        free_mb = commit[0]
        # The game's leak has nearly filled memory: restart the game (it logs
        # back in and farming carries on) rather than wait for the pause below.
        # Only from the running guard thread, never from a bare check().
        if self.thread is not None:
            from src.modules import game_restart
            if game_restart.RUNNER.busy:
                return
            if game_restart.should_restart(free_mb, handles, config.enabled, now,
                                           game_restart.RUNNER.last) \
                    and game_restart.RUNNER.request(
                        f'{free_mb:,.0f} MB free, the game holds {handles:,} handles'):
                return
        if free_mb < self.PAUSE_FREE_MB:
            gc.collect()
            if config.enabled and not self._paused_for_memory:
                self._paused_for_memory = True
                self._recovered_checks = 0
                config.enabled = False      # the bot loop releases all held keys
                users = top_memory_users()
                using = '; '.join(
                    f'{name} {used:.0f} MB' + (f' ({n} processes)' if n > 1 else '')
                    for name, used, n in users)
                mine = f' The bot itself holds {own:.0f} MB.' if own is not None else ''
                mine += game_text
                print(
                    f'\n[!] Windows is almost out of memory ({free_mb:.0f} MB of '
                    f'{commit[1]:.0f} MB free). Paused the bot so it does not run '
                    f'blind; it resumes by itself once {self.RESUME_FREE_MB} MB is '
                    f'free.{mine}'
                    + (f' Using the most now: {using}. Close what you do not need.'
                       if using else '')
                )
            self._recovered_checks = 0
        elif self._paused_for_memory:
            if config.enabled:
                self._paused_for_memory = False     # resumed by hand meanwhile
            elif free_mb >= self.RESUME_FREE_MB:
                self._recovered_checks += 1
                if self._recovered_checks >= self.RESUME_CHECKS:
                    self._resume(free_mb)
            else:
                self._recovered_checks = 0

    def _resume(self, free_mb):
        """Memory is back: carry on farming, if the game is still in front."""
        try:
            from src.modules.health import is_game_focused
            focused = is_game_focused()
        except Exception:
            focused = True
        if not focused:
            if not self._waiting_focus_said:
                self._waiting_focus_said = True
                print(f'\n[~] Memory recovered ({free_mb:.0f} MB free). Click the '
                      'MapleStory window and the bot carries on.')
            return
        self._waiting_focus_said = False
        self._paused_for_memory = False
        self._recovered_checks = 0
        print(f'\n[~] Memory recovered ({free_mb:.0f} MB free); resuming.')
        config.enabled = True
        try:
            from src.common import utils
            utils.print_state()
        except Exception:
            pass


memory_guard = MemoryGuard()
