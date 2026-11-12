"""
Plays the mini-games that interrupt the routine.

The notifier spots the warning pop-ups; this module then solves the game that
follows on its own thread:

* pop-up-1 (the "Lie Detector" shape game): keep the mouse on the fading figure
  (src/detection/fading_figure.py). The game starts a few seconds after the pop-up.
* pop-up-2 (Violetta): find the Violetta wearing make-up and click her
  (src/detection/violetta_solver.py).

While a game runs the bot is paused so it doesn't press keys, and it is resumed
afterwards if it was running before.
"""

import threading
import time
import traceback

from src.common import config, utils


# Set to False to only get the siren and solve the games by hand.
AUTO_SOLVE = {
    'pop-up-1': True,       # fading-figure shape game
    'pop-up-2': True,       # Violetta
}
GAME_WINDOW_TITLE = 'MapleStory'
SHAPE_START_DELAY = 3.0     # seconds after pop-up-1 before looking for the white figure
COOLDOWN_SECONDS = 20       # ignore the same pop-up again this soon after a game ends


def _shape_game(stop):
    from src.detection import fading_figure
    return fading_figure.play_live(GAME_WINDOW_TITLE, move=True, timeout=90.0,
                                   start_delay=SHAPE_START_DELAY, stop=stop)


def _violetta_game(stop):
    from src.detection import violetta_solver
    violetta_solver.play_live(GAME_WINDOW_TITLE, select=True, timeout=120.0)
    return 'done'


GAMES = {
    'pop-up-1': ('Shape (fading figure)', _shape_game),
    'pop-up-2': ('Violetta', _violetta_game),
}


class MiniGameRunner:
    def __init__(self):
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
        if was_enabled:
            config.enabled = False      # pause the routine; bot releases held keys when disabled
            utils.print_state()
        print(f'\n[~] {name} mini-game: solving automatically'
              f'{" (routine paused)" if was_enabled else ""}.')
        result = 'error'
        try:
            result = game(self.stop_event)
        except Exception:
            print(f'\n[!] {name} mini-game solver failed:')
            traceback.print_exc()
        finally:
            self._finished[prefix] = time.time()
            self.current = None
            print(f'\n[~] {name} mini-game finished ({result}).')
            if was_enabled and not config.enabled:
                time.sleep(1.0)         # let the result screen close
                config.enabled = True
                utils.print_state()
                print('[~] Routine resumed.')
