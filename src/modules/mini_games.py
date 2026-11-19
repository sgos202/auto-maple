"""
Plays the mini-games that interrupt the routine.

The notifier spots the warning pop-ups; this module then solves the game that
follows on its own thread:

* pop-up-1 (the "Lie Detector" shape game): keep the mouse on the fading figure
  (src/detection/fading_figure.py). The game starts a few seconds after the pop-up.
* pop-up-2 (Violetta): find the Violetta wearing make-up and click her
  (src/detection/violetta_solver.py). A won game ends with a "You have
  successfully passed the Lie Detector Test" notice that stays up until its OK
  is clicked, so that is clicked too (src/detection/pass_notice.py).
* pop-up-3 ("Question in 11 seconds"): a "Click all X's (N total)" quiz follows;
  click the matching letter tiles, then CONFIRM (src/detection/letter_quiz.py).
  Or "Click the one that is different": click the odd picture, then CONFIRM.
  Or a "Click a popsicle" item quiz: click that item once, no CONFIRM (item
  pictures live in assets/quiz_items/).
  Or a yes/no question, "Is there a spear here?" / "Is this a wand?": click the
  O (yes) or the X (no), then CONFIRM.
  Any other question goes to a Claude vision model, which says what to click
  (src/detection/quiz_ai.py; off until set up with --setup KEY).

While a game runs the bot is paused so it doesn't press keys, and it is resumed
afterwards by itself if it was running before. Pressing Start/stop during a game
doesn't start the routine over the solver; it toggles whether the routine resumes
when the game ends (see handle_start_stop).

When a game is cleared the alarm (siren + flashing button) is switched off by
itself. It keeps going if the game was not cleared, so the user comes to look.
"""

import threading
import time
import traceback

from src.common import config, utils
from src.modules import phone_alert
from src.modules.mini_game_alarm import ALARM


# Set to False to only get the siren and solve the games by hand.
AUTO_SOLVE = {
    'pop-up-1': True,       # fading-figure shape game
    'pop-up-2': True,       # Violetta
    'pop-up-3': True,       # "Click all X's" letter quiz
}
GAME_WINDOW_TITLE = 'MapleStory'
# Seconds after pop-up-1 before looking for the game. 0: the window finder already
# ignores the 'will start soon' notice (it needs the brown play area under the title
# bar). 2026-11-19 01:24 the notice was missed and pop-up-1 matched the game itself at
# countdown '2'; the old 3 s wait swallowed the ~2.5 s white phase, the bot never
# locked on and the character was jailed (shape-20261119-012428).
SHAPE_START_DELAY = 0.0
SHOW_OVERLAY = True         # draw the solvers' tracking boxes over the game
VIOLETTA_RECORD_DIR = 'logs/violetta'   # what the solver saw, for replaying; None to disable
LETTER_QUIZ_RECORD_DIR = 'logs/letter_quiz'  # screenshot of each answer; None to disable
COOLDOWN_SECONDS = 20      # ignore the same pop-up again this soon after a game ends
# Results (their first words) that mean the game was cleared: the alarm stops by
# itself and the phone message gets a tick. Anything else leaves the alarm on.
CLEARED_RESULTS = ('answered', 'done', 'won', 'solved', 'passed')


def _shape_game(stop):
    from src.detection import fading_figure
    return fading_figure.play_live(GAME_WINDOW_TITLE, move=True, timeout=90.0,
                                   start_delay=SHAPE_START_DELAY, stop=stop,
                                   overlay=SHOW_OVERLAY)


def _violetta_game(stop):
    from src.detection import violetta_solver
    violetta_solver.play_live(GAME_WINDOW_TITLE, select=True, timeout=120.0,
                              overlay=SHOW_OVERLAY, record_dir=VIOLETTA_RECORD_DIR)
    if stop.is_set():
        return 'stopped'
    # Won: the "passed the Lie Detector Test" notice needs its OK clicked before
    # the character can farm again. No notice: the game was most likely failed.
    from src.detection import pass_notice
    closed = pass_notice.click_ok(GAME_WINDOW_TITLE, stop=stop)
    return {'passed': 'passed', 'stuck': 'passed but OK not clicked',
            'stopped': 'stopped'}.get(closed, 'no pass notice')


def _letter_quiz(stop):
    from src.detection import letter_quiz
    return letter_quiz.play_live(GAME_WINDOW_TITLE, select=True, timeout=60.0, stop=stop,
                                 record_dir=LETTER_QUIZ_RECORD_DIR, overlay=SHOW_OVERLAY,
                                 phone=True)


GAMES = {
    'pop-up-1': ('Shape (fading figure)', _shape_game),
    'pop-up-2': ('Violetta', _violetta_game),
    'pop-up-3': ('Letter quiz', _letter_quiz),
}


# 2026-11-18 19:53: a shape game failed while other heavy work ran on this PC
# (16-32 busy threads): the solver fell to 7.7 frames a second (40-50
# normally), its mouse trailed the moving figure by 50-160 px and the
# character was sent to the jail room. While a game is played the bot (and the
# Violetta tracker's process) run above normal priority so background load
# can't starve them. Not HIGH: the game client must keep drawing too.
ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
_PROCESS_SET_INFORMATION = 0x0200
_PROCESS_QUERY_INFORMATION = 0x0400


def _set_priority(priority, pid=None):
    """Sets a process's priority class; returns the old one (None on failure)."""
    import ctypes
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetPriorityClass.argtypes = [ctypes.c_void_p]
    kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    if pid is None:
        handle, opened = kernel32.GetCurrentProcess(), False
    else:
        handle = kernel32.OpenProcess(
            _PROCESS_SET_INFORMATION | _PROCESS_QUERY_INFORMATION, False, pid)
        opened = True
        if not handle:
            return None
    try:
        old = kernel32.GetPriorityClass(handle)
        if not old or not kernel32.SetPriorityClass(handle, priority):
            return None
        return old
    finally:
        if opened:
            kernel32.CloseHandle(handle)


class _Priority:
    """Above normal priority for this process while a game runs (the Violetta
    tracker's server process is started above normal by itself)."""

    def __enter__(self):
        self.old = _set_priority(ABOVE_NORMAL_PRIORITY_CLASS)
        return self

    def __exit__(self, *exc):
        if self.old is not None:
            _set_priority(self.old)
        return False


def _machine_lock():
    """A lock shared by every Auto Maple on this PC; None if another one holds it."""
    import ctypes
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    handle = kernel32.CreateMutexW(None, False, 'AutoMapleMiniGameSolver')
    if handle and ctypes.get_last_error() == 183:       # ERROR_ALREADY_EXISTS
        _release(handle)
        return None
    return handle or 0      # 0: could not create it; carry on without the lock


def _release(lock):
    if lock:
        import ctypes
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle(lock)


class MiniGameTally:
    """How many of each mini-game came up this session, and how they ended."""

    def __init__(self):
        self._lock = threading.Lock()
        self.games = {}             # name -> {result: count}
        self.started = time.time()

    def record(self, name, result):
        with self._lock:
            results = self.games.setdefault(name, {})
            results[result] = results.get(result, 0) + 1
            return self._summary_locked()

    def summary(self):
        with self._lock:
            return self._summary_locked()

    @property
    def total(self):
        with self._lock:
            return sum(sum(r.values()) for r in self.games.values())

    def short(self):
        """One line for the GUI: the total and how many of each game."""
        with self._lock:
            total = sum(sum(r.values()) for r in self.games.values())
            if not total:
                return 'Mini-games played: 0'
            parts = ', '.join(f'{name} {sum(results.values())}'
                              for name, results in self.games.items())
            return f'Mini-games played: {total} ({parts})'

    def _summary_locked(self):
        if not self.games:
            return 'Mini-games this session: none yet.'
        total = sum(sum(r.values()) for r in self.games.values())
        parts = []
        for name, results in self.games.items():
            count = sum(results.values())
            detail = ', '.join(f'{n} {result}' for result, n in results.items())
            parts.append(f'{name} x{count} ({detail})')
        minutes = (time.time() - self.started) / 60
        return (f'Mini-games this session: {total} in {minutes:.0f} min - '
                + ' | '.join(parts))


TALLY = MiniGameTally()


def _result_label(result):
    """A short outcome for the tally from whatever a solver returned."""
    if result in (None, True):
        return 'done'
    if result is False:
        return 'failed'
    text = str(result).strip().lower()
    return text if text and len(text) <= 30 else 'done'


ACTIVE_RUNNER = None    # the notifier's runner, for the Start/stop key


def _banner(text):
    utils.print_separator()
    print('#' * 34)
    print(f'#  {text:<30}#')
    print('#' * 34)


def handle_start_stop():
    """
    Start/stop was pressed. While a mini-game is being solved, flip whether the
    routine resumes after it (instead of starting the routine over the solver) and
    return True. Return False when no game is running, so the key works as usual.
    """
    runner = ACTIVE_RUNNER
    if runner is None or not runner.busy:
        return False
    if not runner.paused_routine:
        print('\n[~] A mini-game is being solved; press Start/stop again once it ends '
              'to start Auto Maple.')
        return True
    runner.resume_after = not runner.resume_after
    if runner.resume_after:
        print('\n[~] OK: Auto Maple will resume by itself when the mini-game ends.')
    else:
        print('\n[~] OK: Auto Maple will STAY STOPPED after the mini-game. '
              'Press Start/stop again to undo.')
    return True


class MiniGameRunner:
    def __init__(self):
        global ACTIVE_RUNNER
        ACTIVE_RUNNER = self
        self.paused_routine = False     # this game paused a running routine
        self.resume_after = False       # resume it when the game ends
        self.thread = None
        self.stop_event = threading.Event()
        self.current = None
        self._finished = {}     # prefix -> time the last game ended

    @property
    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def trigger(self, prefix):
        """Called when pop-up PREFIX appears. Returns True if a solver was started."""
        if prefix not in GAMES or not AUTO_SOLVE.get(prefix, False):
            return False
        if self.busy:
            return False
        if time.time() - self._finished.get(prefix, 0) < COOLDOWN_SECONDS:
            return False
        name, game = GAMES[prefix]
        self.stop_event.clear()
        self.current = prefix
        self.thread = threading.Thread(target=self._run, args=(prefix, name, game), daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.stop_event.set()

    def _run(self, prefix, name, game):
        was_enabled = config.enabled
        self.paused_routine = self.resume_after = was_enabled
        if was_enabled:
            config.enabled = False      # pause the routine; bot releases held keys when disabled
            _banner('PAUSED FOR MINI-GAME')
        print(f'\n[~] {name} mini-game: solving automatically.')
        if was_enabled:
            print('[~] The routine resumes by itself afterwards - no need to press Start/stop.')
        if phone_alert.wants('mini_game_start'):
            if prefix == 'pop-up-3' and phone_alert.wants('quiz_answer'):
                # The countdown screen says nothing; the screenshot comes once
                # the question opens, with the answer drawn on it.
                phone_alert.send(f'🎮 {name} coming - waiting for the question; '
                                 f'the answer screenshot follows.')
            else:
                phone_alert.send(f'🎮 {name} mini-game started - solving it automatically.',
                                 photo=True)
        started = time.time()
        result = 'error'
        lock = _machine_lock()
        try:
            if lock is None:
                # Two solvers moving one mouse fight each other and halve each
                # other's frame rate, which is enough to lose the game.
                result = 'left to the other Auto Maple'
                print(f'\n[!] Another Auto Maple on this PC is already solving the {name} '
                      f'mini-game; this one is staying out of it. Close the extra one.')
                self.stop_event.wait(30)    # stay paused while the other one plays
            else:
                with _Priority():
                    result = game(self.stop_event)
        except Exception:
            print(f'\n[!] {name} mini-game solver failed:')
            traceback.print_exc()
        finally:
            _release(lock)
            self._finished[prefix] = time.time()
            self.current = None
            print(f'\n[~] {name} mini-game finished ({result}).')
            label = _result_label(result)
            summary = TALLY.record(name, label)
            print(f'[~] Mini-game #{TALLY.total} this session. {summary}')
            good = label.startswith(CLEARED_RESULTS)
            if good:
                ALARM.dismiss()         # cleared: no need for the user to come and look
            if phone_alert.wants('mini_game_end'):
                phone_alert.send(f'{"✅" if good else "⚠️"} {name} mini-game finished '
                                 f'({result}) after {time.time() - started:.0f} s.\n'
                                 f'{summary}', photo=not good)
            if was_enabled and not config.enabled:
                time.sleep(1.0)         # let the result screen close
                if self.resume_after:
                    config.enabled = True
                    utils.print_state()
                    print('[~] Routine resumed after the mini-game.')
                else:
                    utils.print_state()
                    print('[~] Staying stopped after the mini-game, as asked.')
            self.paused_routine = self.resume_after = False
