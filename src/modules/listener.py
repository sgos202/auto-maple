"""A keyboard listener to track user inputs."""

import time
import threading
import keyboard as kb
from src.common.interfaces import Configurable
from src.common import config, utils
from datetime import datetime
from src.modules.health import is_game_focused


class Listener(Configurable):
    DEFAULT_CONFIG = {
        'Start/stop': 'insert',
        'Reload routine': 'f6',
        'Record position': 'f7'
    }
    BLOCK_DELAY = 1         # Delay after blocking restricted button press

    def __init__(self):
        """Initializes this Listener object's main thread."""

        super().__init__('controls')
        config.listener = self

        self.enabled = False
        self.ready = False
        self.block_time = 0
        self.thread = threading.Thread(target=self._main)
        self.thread.daemon = True

    def start(self):
        """
        Starts listening to user inputs.
        :return:    None
        """

        print('\n[~] Started keyboard listener')
        self.thread.start()

    def _main(self):
        """
        Constantly listens for user inputs and updates variables in config accordingly.
        :return:    None
        """

        self.ready = True
        while True:
            if self.enabled:
                if kb.is_pressed(self.config['Start/stop']):
                    Listener.toggle_enabled()
                    self.wait_for_release('Start/stop')
                elif kb.is_pressed(self.config['Reload routine']):
                    Listener.reload_routine()
                    self.wait_for_release('Reload routine')
                elif self.restricted_pressed('Record position'):
                    Listener.record_position()
                    self.wait_for_release('Record position')
            time.sleep(0.01)

    def wait_for_release(self, action):
        """Prevents one held control key from triggering multiple actions."""

        while kb.is_pressed(self.config[action]):
            time.sleep(0.05)

    def restricted_pressed(self, action):
        """Returns whether the key bound to ACTION is pressed only if the bot is disabled."""

        if kb.is_pressed(self.config[action]):
            if not config.enabled:
                return True
            now = time.time()
            if now - self.block_time > Listener.BLOCK_DELAY:
                print(f"\n[!] Cannot use '{action}' while Auto Maple is enabled")
                self.block_time = now
        return False

    @staticmethod
    def toggle_enabled():
        """Starts or pauses Auto Maple's routine or standalone mode."""

        config.bot.rune_active = False

        if not config.enabled:
            standalone = (
                len(config.routine) == 0
                and config.capture.mob_tracker.settings()['Enabled']
            )
            if len(config.routine) == 0 and not standalone:
                print(
                    '\n[!] Load a routine, or enable sampled-enemy auto-attack '
                    'to start standalone patrol.'
                )
                return
            if standalone:
                if not config.capture.mob_tracker.has_samples():
                    print(
                        '\n[!] Calibrate the player badge and at least one enemy '
                        'sample before starting standalone patrol.'
                    )
                    return
                command_book = config.bot.command_book
                if command_book is None or 'cruelstab' not in command_book.dict:
                    print(
                        '\n[!] Load a command book that provides CruelStab before '
                        'starting standalone auto-attack.'
                    )
                    return
            if not config.capture.calibrated and not Listener.recalibrate_minimap():
                return
            print('\n[~] Auto Maple starts in 3 seconds. Focus the MapleStory window now.')
            for remaining in (3, 2, 1):
                print(f'[~] Auto Maple starts in {remaining}...')
                time.sleep(1)
            if not is_game_focused():
                print(
                    '\n[!] MapleStory is not focused; start cancelled. '
                    'Focus the game and try again.'
                )
                return

        config.enabled = not config.enabled
        utils.print_state()

    @staticmethod
    def reload_routine():
        if not Listener.recalibrate_minimap():
            return

        config.routine.load(config.routine.path)

    @staticmethod
    def recalibrate_minimap():
        config.capture.calibrated = False
        config.capture.manual_calibration_requested.set()
        deadline = time.monotonic() + 120
        while not config.capture.calibrated and time.monotonic() < deadline:
            time.sleep(0.01)
        if not config.capture.calibrated:
            print('\n[!] Minimap calibration failed; check the game window and minimap.')
            return False
        config.gui.edit.minimap.redraw()
        return True

    @staticmethod
    def record_position():
        if not config.capture.calibrated:
            print('\n[!] Cannot record a position until the minimap is calibrated.')
            return
        pos = tuple('{:.3f}'.format(round(i, 3)) for i in config.player_pos)
        now = datetime.now().strftime('%I:%M:%S %p')
        config.gui.edit.record.add_entry(now, pos)
        print(f'\n[~] Recorded position ({pos[0]}, {pos[1]}) at {now}')
        time.sleep(0.6)
