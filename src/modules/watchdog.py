"""
Stall watchdog for long runs.

Worker loops call `heartbeat(name)` every iteration. A supervisor thread checks
them; when one has been silent too long:
- every thread's Python stack is written to logs/hangs/hang_<time>.txt (so the
  exact line it is stuck on is known, no debugger needed) and summarised in the
  console,
- all keys the bot holds are released (a stalled bot must not keep walking),
- the loop's restart hook runs, if it has one (capture: a fresh capture thread;
  the stuck one exits by itself if it ever wakes up).

A loop is only watched while it is registered; `pause(name)` stops watching it
during waits that are expected to be long (e.g. a manual minimap selection).
"""

import os
import sys
import threading
import time
import traceback
from datetime import datetime


class Watchdog:
    CHECK_SECONDS = 2.0
    HANG_DIR = os.path.join('logs', 'hangs')
    MAX_DUMPS = 30              # oldest dumps are deleted beyond this

    def __init__(self):
        self._lock = threading.Lock()
        self._beats = {}        # name -> last heartbeat (monotonic)
        self._limits = {}       # name -> seconds of silence allowed
        self._hooks = {}        # name -> restart callable or None
        self._paused = set()
        self._reported = {}     # name -> heartbeat time already reported
        self._restarts = {}     # name -> [times]
        self._thread = None

    def register(self, name, limit_seconds, restart=None):
        with self._lock:
            self._beats[name] = time.monotonic()
            self._limits[name] = limit_seconds
            self._hooks[name] = restart
        self.start()

    def heartbeat(self, name):
        self._beats[name] = time.monotonic()    # (dict write is atomic)

    def pause(self, name):
        with self._lock:
            self._paused.add(name)

    def resume(self, name):
        with self._lock:
            self._paused.discard(name)
            self._beats[name] = time.monotonic()

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name='watchdog',
                                            daemon=True)
            self._thread.start()

    def _run(self):
        while True:
            time.sleep(self.CHECK_SECONDS)
            try:
                self.check()
            except Exception as error:
                print(f'\n[!] Watchdog check failed: {error}')

    def check(self, now=None):
        now = time.monotonic() if now is None else now
        with self._lock:
            stalled = [
                (name, now - beat) for name, beat in self._beats.items()
                if name not in self._paused
                and now - beat > self._limits[name]
                and self._reported.get(name) != beat
            ]
            for name, _ in stalled:
                self._reported[name] = self._beats[name]
        for name, silent in stalled:
            self._handle_stall(name, silent, now)
        return [name for name, _ in stalled]

    def _handle_stall(self, name, silent, now):
        path = dump_stacks(f'{name} stalled for {silent:.0f}s', self.HANG_DIR,
                           self.MAX_DUMPS)
        stuck_at = _thread_summary(name)
        print(f'\n[!] Watchdog: the {name} loop has not run for {silent:.0f}s'
              f'{f" (stuck at {stuck_at})" if stuck_at else ""}. '
              f'All thread stacks saved to {path}.')
        try:
            from src.common.vkeys import release_all_keys
            released = release_all_keys()
            if released:
                print(f'[!] Watchdog: released held keys {", ".join(released)}.')
        except Exception:
            pass
        hook = self._hooks.get(name)
        if hook is None:
            return
        recent = [t for t in self._restarts.get(name, []) if now - t < 600]
        recent.append(now)
        self._restarts[name] = recent
        if len(recent) > 5:
            print(f'[!] Watchdog: {name} stalled {len(recent)} times in 10 min; '
                  'not restarting it again. Restart Auto Maple and send the hang '
                  'files in logs/hangs.')
            return
        try:
            hook()
            self._beats[name] = time.monotonic()
            print(f'[~] Watchdog: started a fresh {name} thread.')
        except Exception as error:
            print(f'[!] Watchdog: restarting {name} failed: {error}')


def _thread_summary(name):
    """'file:line in function' at the top of the thread called NAME, if found."""
    frames = sys._current_frames()
    for thread in threading.enumerate():
        if thread.name == name and thread.ident in frames:
            stack = traceback.extract_stack(frames[thread.ident])
            if stack:
                top = stack[-1]
                return f'{os.path.basename(top.filename)}:{top.lineno} in {top.name}'
    return None


def dump_stacks(reason, directory, keep=30):
    """Writes every thread's stack to a new file; returns its path."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f'hang_{datetime.now():%Y%m%d_%H%M%S}.txt')
    frames = sys._current_frames()
    names = {thread.ident: thread.name for thread in threading.enumerate()}
    lines = [f'{datetime.now():%Y-%m-%d %H:%M:%S}  {reason}\n']
    for ident, frame in frames.items():
        lines.append(f'\n--- Thread {names.get(ident, "?")} ({ident}) ---\n')
        lines.extend(traceback.format_stack(frame))
    try:
        with open(path, 'w', encoding='utf-8') as file:
            file.writelines(lines)
        dumps = sorted(f for f in os.listdir(directory) if f.startswith('hang_'))
        for old in dumps[:-keep]:
            os.remove(os.path.join(directory, old))
    except OSError:
        pass
    return path


watchdog = Watchdog()
