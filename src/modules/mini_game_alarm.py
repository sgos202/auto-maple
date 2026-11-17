"""
The alarm for a mini-game pop-up: the siren loops and a flashing button shows
until the user clicks the button, or until the solver clears the game
(mini_games.MiniGameRunner dismisses it then; a game that was not cleared
keeps ringing).

The notifier calls ALARM.ring() when any pop-up appears. The GUI thread draws the
button (Tk isn't thread-safe) by polling ALARM.active, and clicking it calls
ALARM.dismiss(), which stops the sound.

2026-11-15: the siren played on Windows' default output and the user, with four
outputs (speakers, PlayStation Link headset, monitor, digital out), didn't hear
it. So it plays on EVERY output device: one small helper process per device
(pygame can only open one device per process). A helper exits when its stdin
closes, so it also stops if Auto Maple is closed or crashes.

Try it: python -m src.modules.mini_game_alarm --test
"""

import os
import subprocess
import sys
import threading
import time

SOUND = os.environ.get('AUTO_MAPLE_ALARM_SOUND',     # for silent tests
                       'assets/alerts/siren.mp3')    # the shape game's alert, for every mini-game
DEVICES = 'all'                     # 'all' output devices, 'default', or a list of names
BEEP_SECONDS = 2.0                  # fallback beep interval if the mp3 can't play


def output_devices():
    """Names of the sound output devices pygame can open."""
    import pygame
    import pygame._sdl2.audio as sdl_audio
    started_here = not pygame.mixer.get_init()
    if started_here:
        pygame.mixer.init()
    try:
        return list(sdl_audio.get_audio_device_names(False))
    finally:
        if started_here:
            pygame.mixer.quit()


class MiniGameAlarm:
    def __init__(self):
        self._lock = threading.Lock()
        self.active = False
        self.name = ''
        self.started = 0.0
        self._players = []          # helper processes, one per device
        self._sound = None          # in-process fallback

    def ring(self, name):
        """Starts the alarm for mini-game NAME; does nothing more if it is already on."""
        with self._lock:
            self.name = name
            if self.active:
                return
            self.active = True
            self.started = time.time()
        threading.Thread(target=self._start_sound, daemon=True).start()

    def _start_sound(self):
        try:
            names = [None] if DEVICES == 'default' else \
                output_devices() if DEVICES == 'all' else list(DEVICES)
        except Exception as error:
            print(f'\n[!] Could not list sound devices ({error}); using the default one.')
            names = [None]
        players = []
        for device in names or [None]:
            try:
                command = [sys.executable, '-m', 'src.modules.mini_game_alarm', '--play']
                if device:
                    command.append(device)
                players.append(subprocess.Popen(
                    command, stdin=subprocess.PIPE, cwd=os.getcwd(),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)))
            except Exception as error:
                print(f'\n[!] Could not start the alarm sound on {device or "default"}: {error}')
        with self._lock:
            if not self.active:         # dismissed while starting
                _stop_players(players)
                return
            self._players = players
        if players:
            print(f'[~] Alarm sounding on {len(players)} output device(s): '
                  f'{", ".join(d or "default" for d in names)}.')
        else:
            self._play_here()

    def _play_here(self):
        """Fallback: play in this process on the default device, or beep."""
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            self._sound = pygame.mixer.Sound(SOUND)
            self._sound.play(loops=-1)
            return
        except Exception as error:
            print(f'\n[!] Could not play {SOUND}, beeping instead: {error}')
        import winsound
        while self.active:
            try:
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            except Exception:
                return
            time.sleep(BEEP_SECONDS)

    def dismiss(self):
        """Stops the sound; the GUI then removes the button."""
        with self._lock:
            if not self.active:
                return
            self.active = False
            players, self._players = self._players, []
        _stop_players(players)
        if self._sound is not None:
            try:
                self._sound.stop()
            except Exception:
                pass
        print('\n[~] Mini-game alarm dismissed.')


def _stop_players(players):
    for player in players:
        try:
            player.stdin.close()        # the helper exits on end of input...
        except Exception:
            pass
        try:
            player.terminate()          # ...and this makes sure of it
        except Exception:
            pass


ALARM = MiniGameAlarm()


def _play(device):
    """Helper process: loop the siren on DEVICE until stdin closes."""
    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
    import pygame
    try:
        pygame.mixer.init(devicename=device) if device else pygame.mixer.init()
        pygame.mixer.Sound(SOUND).play(loops=-1)
    except Exception as error:
        print(f'[!] Alarm sound failed on {device or "default"}: {error}')
        return
    try:
        sys.stdin.read()                # returns when the parent closes it or exits
    except Exception:
        pass


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == '--play':
        _play(sys.argv[2] if len(sys.argv) > 2 else None)
    elif len(sys.argv) >= 2 and sys.argv[1] == '--test':
        seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
        print('Output devices:', output_devices())
        print(f'Ringing for {seconds:.0f} s...')
        ALARM.ring('Test')
        time.sleep(seconds)
        ALARM.dismiss()
    else:
        print('Usage: python -m src.modules.mini_game_alarm --test [SECONDS]')
