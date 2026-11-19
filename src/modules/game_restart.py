"""
Restarts the MapleStory client by itself, as in assets/restart_process_maple.mp4.

Why: the game client leaks memory while farming (about 5-6 GB of commit an hour
on a busy map; 2026-11-17 it filled a 53 GB limit in 7 hours). Nothing outside
the game can give that memory back; it is only freed when the game closes. So
when free memory runs low the bot closes the game, starts it again and logs
back in. Only the game restarts - Auto Maple keeps running.

The steps, each recognised from a small crop of the screen (assets/ui/restart_*):

    1. MENU -> GAME MENU -> EXIT GAME -> "Are you sure you want to quit?" -> OK
    2. the game window closes; start it again through the Nexon Launcher
       (LAUNCH_COMMAND; PLAY is clicked if the launcher only opens). The
       launcher is first re-opened as administrator if it wasn't, so Windows'
       permission box (which no program can click) does not come up - see
       prepare_launcher
    3. LOG-IN screen (the launcher has already signed in): click Login
    4. Select a world/channel: click the world, double-click a channel
    5. Select a character: double-click the character
    6. the map is back (the MENU button is on screen): farming resumes

No password or PIN is typed anywhere. The character comes back on the map it
logged out from.

Screenshots of every step go to logs/game_restart/<time>/, so a restart that
went wrong can be looked at afterwards.

Try it by hand, with the game open and the character somewhere safe:
    python -m src.modules.game_restart            (the whole restart)
    python -m src.modules.game_restart --look     (only say which screen is showing)
"""

import ctypes
import os
import random
import subprocess
import threading
import time

import cv2
import numpy as np

AUTO_RESTART = True                 # False: only warn, never restart the game
# Restart when Windows has less than this much memory (commit) left and the
# game is what filled it. At ~6 GB an hour this leaves about half an hour.
RESTART_FREE_MB = 15000             # restart the game when free commit drops under ~15 GB
RESTART_MIN_HANDLES = 60000         # the client's leak shows as this many open handles
RESTART_MIN_GAP_SECONDS = 1800      # never twice within this long
GAME_TITLE = 'MapleStory'
LAUNCH_COMMAND = (r'D:\Nexon Launcher\nexon_launcher.exe', 'nxl://launch/59813')
LAUNCH_SHORTCUT = os.path.join(os.path.expanduser('~'), 'Desktop',
                               'MapleStory Classic World.lnk')   # used if the exe is gone
RECORD_DIR = 'logs/game_restart'
MATCH = 0.85                        # (other screens score at most 0.75 on these crops)

QUIT_WAIT = 40                      # the window closing after OK
WINDOW_WAIT = 300                   # launcher -> game window (it may patch first)
LOGIN_WAIT = 180                    # game window -> LOG-IN screen
STEP_WAIT = 45                      # each screen after a click
MAP_WAIT = 90                       # character chosen -> standing in the map

# Offsets in screen pixels (the game always draws at 1:1), measured on the recording.
EXIT_GAME_BELOW_CHANGE_CHANNEL = 130        # GAME MENU: EXIT GAME is 5 rows under it
WORLD_TAB_FROM_BADGE = (210, 58)            # first world tab, from "Select a world/channel"
GRID_FIRST_FROM_GO = (-431.5, 48)           # CH.1, from "Go to the selected world"
GRID_PITCH = (68, 32)
GRID_COLUMNS, GRID_ROWS = 8, 5
CHARACTER_FROM_SIGN = (-467.5, 200)         # first character, from the "Select a character" sign
OPEN_CHANNEL_SATURATION = 80                # a channel's load bar is coloured; a shut one is grey


def _template(name):
    image = cv2.imread(f'assets/ui/{name}.png', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f'\n[!] Missing assets/ui/{name}.png; the game restart will not work.')
    return image


TEMPLATES = {name: None for name in (
    'menu_button', 'menu_change_channel', 'restart_quit_ok', 'restart_login_badge',
    'restart_login_button', 'restart_world_badge', 'restart_world_go',
    'restart_char_sign', 'restart_launcher_play')}


def _load_templates():
    for name in TEMPLATES:
        if TEMPLATES[name] is None:
            TEMPLATES[name] = _template(name)
    return all(image is not None for image in TEMPLATES.values())


def find(frame, name):
    """Centre (x, y) of template NAME in FRAME (BGR), or None."""
    template = TEMPLATES.get(name)
    if frame is None or template is None:
        return None
    gray = cv2.cvtColor(np.ascontiguousarray(frame[:, :, :3]), cv2.COLOR_BGR2GRAY)
    if template.shape[0] > gray.shape[0] or template.shape[1] > gray.shape[1]:
        return None
    _, score, _, location = cv2.minMaxLoc(cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED))
    if score < MATCH:
        return None
    return location[0] + template.shape[1] / 2, location[1] + template.shape[0] / 2


def which_screen(frame):
    """The restart screen FRAME shows: 'quit', 'login', 'channels', 'world',
    'character', 'map', or None."""
    for screen, name in (('quit', 'restart_quit_ok'), ('login', 'restart_login_badge'),
                         ('character', 'restart_char_sign'), ('channels', 'restart_world_go'),
                         ('world', 'restart_world_badge'), ('map', 'menu_button')):
        if find(frame, name) is not None:
            return screen
    return None


def channel_cell(go, channel):
    """Screen-pixel centre of CHANNEL's (1..40) cell on the world screen."""
    index = channel - 1
    return (go[0] + GRID_FIRST_FROM_GO[0] + GRID_PITCH[0] * (index % GRID_COLUMNS),
            go[1] + GRID_FIRST_FROM_GO[1] + GRID_PITCH[1] * (index // GRID_COLUMNS))


def open_channels(frame, go):
    """The channels that can be entered: their load bar is coloured, not grey."""
    hsv = cv2.cvtColor(np.ascontiguousarray(frame[:, :, :3]), cv2.COLOR_BGR2HSV)
    found = []
    for channel in range(1, GRID_COLUMNS * GRID_ROWS + 1):
        x, y = channel_cell(go, channel)
        x, y = int(round(x)), int(round(y)) + 8         # the bar is under the name
        patch = hsv[max(0, y - 3):y + 4, max(0, x - 25):x + 26, 1]
        if patch.size and float(patch.mean()) >= OPEN_CHANNEL_SATURATION:
            found.append(channel)
    return found


def should_restart(free_mb, handles, enabled, now, last_restart):
    """Whether the memory guard should ask for a restart now."""
    return bool(AUTO_RESTART and enabled and free_mb is not None and handles is not None
                and free_mb < RESTART_FREE_MB and handles >= RESTART_MIN_HANDLES
                and now - last_restart >= RESTART_MIN_GAP_SECONDS)


# ------------------------------------------------------------- the screen ----

class Screen:
    """Looks at and clicks in the game window (or the whole desktop)."""

    def __init__(self, title=GAME_TITLE):
        import mss
        self.title = title
        self.user32 = ctypes.windll.user32
        self.user32.SetProcessDPIAware()
        self.grabber = mss.mss()
        self.origin = (0, 0)

    def window(self):
        handle = self.user32.FindWindowW(None, self.title)
        if handle and self.user32.IsWindowVisible(handle):
            return handle
        return None

    def focus(self):
        handle = self.window()
        if handle and self.user32.GetForegroundWindow() != handle:
            if self.user32.IsIconic(handle):
                self.user32.ShowWindow(handle, 9)       # SW_RESTORE
            self.user32.SetForegroundWindow(handle)
            time.sleep(0.4)
        return bool(handle) and self.user32.GetForegroundWindow() == handle

    def game(self):
        """The game window's picture (BGR), or None while there is no window.
        Clicks are then placed relative to it."""
        from ctypes import wintypes
        handle = self.window()
        if not handle or self.user32.IsIconic(handle):
            return None
        rect = wintypes.RECT()
        self.user32.GetClientRect(handle, ctypes.byref(rect))
        origin = wintypes.POINT(0, 0)
        self.user32.ClientToScreen(handle, ctypes.byref(origin))
        if rect.right < 200 or rect.bottom < 150:
            return None
        return self._grab({'left': origin.x, 'top': origin.y,
                           'width': rect.right, 'height': rect.bottom})

    def desktop(self):
        return self._grab(self.grabber.monitors[0])

    def _grab(self, region):
        try:
            frame = np.asarray(self.grabber.grab(region))[:, :, :3]
        except Exception:
            return None
        self.origin = (region['left'], region['top'])
        return frame

    def click(self, point, double=False):
        x, y = int(self.origin[0] + point[0]), int(self.origin[1] + point[1])
        from src.common import vkeys
        for _ in range(2 if double else 1):
            vkeys.click((x, y), settle=(0.05, 0.10), hold=(0.05, 0.09),
                        allow_paused=True)
            time.sleep(0.08 + 0.04 * random.random())

    def park(self):
        """Moves the mouse off the buttons (a hovered button looks different)."""
        self.user32.SetCursorPos(int(self.origin[0] + 300), int(self.origin[1] + 120))


LAUNCHER_PROCESSES = ('nexon_launcher.exe', 'nexon_runtime.exe', 'nexon_client.exe')
PERMISSION_HINT_SECONDS = 45        # no game window this long: Windows is probably asking


def is_admin():
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _pids(image):
    """Process ids of the running IMAGE (e.g. 'nexon_runtime.exe')."""
    try:
        output = subprocess.run(
            ['tasklist', '/FO', 'CSV', '/NH', '/FI', f'IMAGENAME eq {image}'],
            capture_output=True, text=True, timeout=15,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).stdout
    except Exception:
        return []
    pids = []
    for line in output.splitlines():
        parts = [part.strip('"') for part in line.split('","')]
        if len(parts) >= 2 and parts[0].lower() == image.lower() and parts[1].isdigit():
            pids.append(int(parts[1]))
    return pids


def _process_is_elevated(pid):
    """True/False for whether process PID runs as administrator; None if unknown."""
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    advapi32 = ctypes.WinDLL('advapi32', use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                          ctypes.POINTER(wintypes.HANDLE)]
    advapi32.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                             wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    process = kernel32.OpenProcess(0x1000, False, pid)     # QUERY_LIMITED_INFORMATION
    if not process:
        return None
    try:
        token = wintypes.HANDLE()
        if not advapi32.OpenProcessToken(process, 0x0008, ctypes.byref(token)):   # TOKEN_QUERY
            return None
        try:
            elevated, size = wintypes.DWORD(0), wintypes.DWORD(0)
            if not advapi32.GetTokenInformation(token, 20, ctypes.byref(elevated),  # TokenElevation
                                                ctypes.sizeof(elevated), ctypes.byref(size)):
                return None
            return bool(elevated.value)
        finally:
            kernel32.CloseHandle(token)
    finally:
        kernel32.CloseHandle(process)


def prepare_launcher(log=print):
    """
    MapleStory.exe needs administrator rights, so starting it brings up Windows'
    "Do you want to allow this app to make changes?" box. That box is on a
    protected screen no program can see or click (which is also why a screen
    recording does not show it), so the bot cannot answer it.

    It does not come up when the Nexon Launcher itself already runs as
    administrator. Auto Maple runs as administrator, and a launcher it starts
    inherits that. So: if a launcher is running without those rights (the one
    Windows started in the tray), close it here; launch_game then starts a new
    one. Only the launcher is touched, never the game. Returns False when the
    box will have to be answered by hand.
    """
    if not is_admin():
        log('Auto Maple is not running as administrator, so Windows will ask for '
            'permission to start the game: click Yes on that box.')
        return False
    running = [pid for image in LAUNCHER_PROCESSES for pid in _pids(image)]
    if not running:
        return True
    if all(_process_is_elevated(pid) for pid in running):
        return True                 # ours from an earlier restart: no question asked
    log('Closing the Nexon Launcher and starting it again as administrator, so '
        'Windows does not ask for permission to start the game.')
    for image in LAUNCHER_PROCESSES:
        try:
            subprocess.run(['taskkill', '/F', '/T', '/IM', image], capture_output=True,
                           timeout=20, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except Exception:
            pass
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and any(_pids(image) for image in LAUNCHER_PROCESSES):
        time.sleep(0.5)
    time.sleep(2.0)
    return True


def launch_game():
    if os.path.isfile(LAUNCH_COMMAND[0]):
        subprocess.Popen(list(LAUNCH_COMMAND), close_fds=True,
                         creationflags=getattr(subprocess, 'DETACHED_PROCESS', 0))
    elif os.path.isfile(LAUNCH_SHORTCUT):
        os.startfile(LAUNCH_SHORTCUT)
    else:
        raise FileNotFoundError(f'Neither {LAUNCH_COMMAND[0]} nor {LAUNCH_SHORTCUT} exists.')


# ------------------------------------------------------------ the restart ----

class Failed(Exception):
    pass


class Stopped(Failed):
    """Start/stop was pressed: give up at once, no retries."""


class GameRestart:
    def __init__(self, screen=None, launch=launch_game, clock=time.monotonic, sleep=time.sleep,
                 record_dir=RECORD_DIR, stop=None, log=None, prepare=None):
        self.screen = screen
        self.launch = launch
        # (the real launcher is only closed and re-opened for the real launch)
        self.prepare = prepare if prepare is not None else (
            prepare_launcher if launch is launch_game else (lambda log: True))
        self.clock, self.sleep = clock, sleep
        self.stop = stop
        self.log = log or (lambda text: print(f'[RESTART] {text}'))
        self.folder = None
        self.shots = 0
        if record_dir:
            self.folder = os.path.join(record_dir, time.strftime('%Y%m%d-%H%M%S'))

    # -- helpers
    def _check_stop(self):
        if self.stop is not None and self.stop.is_set():
            raise Stopped('stopped')

    def _shot(self, frame, label):
        if not self.folder or frame is None:
            return
        try:
            os.makedirs(self.folder, exist_ok=True)
            self.shots += 1
            cv2.imwrite(os.path.join(self.folder, f'{self.shots:02d}-{label}.jpg'), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
        except Exception:
            pass

    def _wait(self, names, seconds, what, every=0.5, grab=None, retry=None, retry_every=10.0):
        """Waits until one of the templates NAMES shows; returns (name, point,
        frame). RETRY() is called every RETRY_EVERY seconds meanwhile. Raises
        Failed after SECONDS."""
        grab = grab or self.screen.game
        started = last_retry = self.clock()
        frame = None
        while self.clock() - started < seconds:
            self._check_stop()
            frame = grab()
            for name in names:
                point = find(frame, name)
                if point is not None:
                    return name, point, frame
            if retry is not None and self.clock() - last_retry >= retry_every:
                last_retry = self.clock()
                retry(frame)
            self.sleep(every)
        self._shot(frame if frame is not None else self.screen.desktop(), 'stuck')
        raise Failed(f'{what} did not show within {seconds:.0f} s')

    # -- steps
    def quit_game(self):
        screen = self.screen
        if screen.window() is None:
            self.log('The game is not open; starting it.')
            return
        if not screen.focus():
            self.log('Could not bring the game to the front; clicking it anyway.')
        for attempt in range(3):
            frame = screen.game()
            if find(frame, 'restart_quit_ok') is None:
                menu = find(frame, 'menu_button')
                if menu is None:
                    # Not on the map (a log-in screen, or the window is covered):
                    # nothing to quit through, close what is there the plain way.
                    if which_screen(frame) in ('login', 'world', 'channels', 'character'):
                        self.log('The game is on its log-in screens already.')
                        return 'login'
                    self._shot(frame, 'no-menu')
                    raise Failed('the MENU button is not on screen (is the game covered?)')
                screen.click(menu)
                try:
                    _, item, _ = self._wait(('menu_change_channel',), 5, 'The GAME MENU')
                except Stopped:
                    raise
                except Failed:
                    continue        # the click closed an open menu: once more
                screen.click((item[0], item[1] + EXIT_GAME_BELOW_CHANGE_CHANNEL))
            try:
                _, ok, frame = self._wait(('restart_quit_ok',), 6, 'The quit question')
            except Stopped:
                raise
            except Failed:
                continue
            self._shot(frame, 'quit-question')
            screen.click(ok)
            break
        else:
            raise Failed('could not open GAME MENU -> EXIT GAME')
        started = self.clock()
        while screen.window() is not None:
            self._check_stop()
            if self.clock() - started > QUIT_WAIT:
                raise Failed(f'the game did not close within {QUIT_WAIT} s of OK')
            self.sleep(0.5)
        self.log('The game closed.')
        self.sleep(5.0)             # let the client and its anti-cheat finish exiting

    def start_game(self):
        unattended = self.prepare(self.log)
        self.launch()
        self.log('Asked the Nexon Launcher to start the game.')
        started = self.clock()
        last_play = float('-inf')
        hinted = False
        while self.screen.window() is None:
            self._check_stop()
            waited = self.clock() - started
            if waited > WINDOW_WAIT:
                self._shot(self.screen.desktop(), 'no-window')
                raise Failed(f'no game window within {WINDOW_WAIT} s of starting the launcher')
            if waited > PERMISSION_HINT_SECONDS and not hinted:
                hinted = True
                self.log('No game window yet. If Windows is showing its "Do you want to '
                         'allow this app to make changes?" box, click Yes - the bot '
                         'cannot see or click that box.')
                self._ask_for_help(unattended)
            # The launcher may just open and wait: press its PLAY button.
            if waited > 20 and self.clock() - last_play > 30:
                desktop = self.screen.desktop()
                play = find(desktop, 'restart_launcher_play')
                if play is not None:
                    last_play = self.clock()
                    self._shot(desktop, 'launcher-play')
                    self.screen.click(play)
                    self.log('Clicked PLAY in the Nexon Launcher.')
            self.sleep(1.0)
        self.log('The game window is open.')

    def _ask_for_help(self, unattended):
        try:
            from src.modules import phone_alert
            if phone_alert.wants('bot_paused'):
                phone_alert.send('⚠️ MapleStory is not opening after the restart. Windows may '
                                 'be asking for permission (Yes/No box) - click Yes.')
        except Exception:
            pass

    def log_in(self):
        screen = self.screen
        name, point, frame = self._wait(
            ('restart_login_button', 'restart_world_badge', 'restart_char_sign'),
            LOGIN_WAIT, 'The LOG-IN screen')
        if name == 'restart_login_button':
            self._shot(frame, 'login')
            self.sleep(1.0)
            screen.focus()
            screen.click(point)
            screen.park()

            def again(frame):
                button = find(frame, 'restart_login_button')
                if button is not None:
                    screen.click(button)
                    screen.park()
            name, point, frame = self._wait(('restart_world_badge', 'restart_char_sign'),
                                            STEP_WAIT, 'The world list', retry=again)
        if name == 'restart_char_sign':
            return
        # The world, then a channel.
        self._shot(frame, 'world')
        badge = point
        tried = []
        for attempt in range(4):
            self.sleep(0.8)
            screen.click((badge[0] + WORLD_TAB_FROM_BADGE[0], badge[1] + WORLD_TAB_FROM_BADGE[1]))
            screen.park()
            try:
                _, go, frame = self._wait(('restart_world_go',), 10, 'The channel list')
            except Stopped:
                raise
            except Failed:
                continue
            self.sleep(0.6)
            frame = screen.game()
            go = find(frame, 'restart_world_go') or go
            choices = [c for c in open_channels(frame, go) if c not in tried]
            if not choices:
                self._shot(frame, 'no-channel')
                raise Failed('no open channel found in the list')
            channel = random.choice(choices)
            tried.append(channel)
            self._shot(frame, f'channel-{channel}')
            self.log(f'Going to channel {channel}.')
            screen.click(channel_cell(go, channel), double=True)
            screen.park()
            try:
                self._wait(('restart_char_sign',), 25, 'The character screen')
                return
            except Stopped:
                raise
            except Failed:
                self.log(f'Channel {channel} did not open; trying another.')
        raise Failed('could not get from the world list to the character screen')

    def pick_character(self):
        screen = self.screen
        _, sign, frame = self._wait(('restart_char_sign',), STEP_WAIT, 'The character screen')
        self._shot(frame, 'character')
        character = (sign[0] + CHARACTER_FROM_SIGN[0], sign[1] + CHARACTER_FROM_SIGN[1])
        self.sleep(1.0)
        screen.click(character, double=True)
        screen.park()

        def again(frame):
            sign = find(frame, 'restart_char_sign')
            if sign is not None:        # still here: select, then the sign itself
                screen.click((sign[0] + CHARACTER_FROM_SIGN[0], sign[1] + CHARACTER_FROM_SIGN[1]))
                self.sleep(0.5)
                screen.click(sign)
                screen.park()
        _, _, frame = self._wait(('menu_button',), MAP_WAIT, 'The map', retry=again,
                                 retry_every=8.0)
        self.sleep(2.0)
        self._shot(screen.game(), 'map')

    def run(self):
        """'restarted', or 'failed: <why>'."""
        if not _load_templates():
            return 'failed: UI pictures missing'
        if self.screen is None:
            self.screen = Screen()
        try:
            self._check_stop()
            if self.quit_game() != 'login':
                self.start_game()
            self.log_in()
            self.pick_character()
            self.screen.focus()
        except Failed as error:
            self.log(f'Failed: {error}.'
                     + (f' Screenshots: {self.folder}' if self.folder and self.shots else ''))
            return f'failed: {error}'
        self.log('The game is back on the map.')
        return 'restarted'


# ------------------------------------------------------- inside Auto Maple ----

class RestartRunner:
    """Runs a restart on its own thread with the routine paused, then carries on
    farming. One at a time; the memory guard asks through request()."""

    def __init__(self):
        self.thread = None
        self.stop_event = threading.Event()
        self.last = float('-inf')       # monotonic time the last restart began
        self.paused_routine = False
        self.count = 0

    @property
    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def request(self, reason):
        """Starts a restart unless one is running or a mini-game is being solved."""
        if self.busy:
            return False
        try:
            from src.modules import mini_games
            runner = mini_games.ACTIVE_RUNNER
            if runner is not None and runner.busy:
                return False            # after the mini-game; the guard asks again
        except Exception:
            pass
        self.last = time.monotonic()
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, args=(reason,), daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.stop_event.set()

    def _run(self, reason):
        from src.common import config, utils
        from src.modules import phone_alert
        was_enabled = config.enabled
        self.paused_routine = was_enabled
        config.enabled = False          # the bot loop releases all held keys
        utils.print_separator()
        print(f'[~] Restarting MapleStory to free its memory ({reason}). Auto Maple keeps '
              'running and carries on farming afterwards - leave the mouse alone for a minute.')
        if phone_alert.wants('bot_paused'):
            phone_alert.send(f'🔄 Restarting MapleStory to free memory ({reason}).')
        time.sleep(2.0)
        result = 'failed: error'
        try:
            result = GameRestart(stop=self.stop_event,
                                 log=lambda text: print(f'[~] Restart: {text}')).run()
        except Exception as error:
            result = f'failed: {error}'
            import traceback
            traceback.print_exc()
        self.count += 1
        if result == 'restarted':
            if not self._await_minimap(config):
                print('\n[!] MapleStory restarted, but the minimap was not found in time. '
                      'Auto Maple stays paused; bring the game to the map and press '
                      'Start/stop after it is calibrated.')
                if phone_alert.wants('bot_paused'):
                    phone_alert.send('⚠️ MapleStory restarted, but the minimap was not '
                                     'found in time. Auto Maple stays paused.', photo=True)
            else:
                if was_enabled and not config.enabled and not self.stop_event.is_set():
                    config.enabled = True
                    utils.print_state()
                print(f'\n[~] MapleStory restarted (#{self.count} this session); '
                      f'{"farming again" if config.enabled else "Auto Maple stays stopped"}.')
                if phone_alert.wants('bot_paused'):
                    phone_alert.send('✅ MapleStory restarted; farming again.', photo=True)
        else:
            print(f'\n[!] The MapleStory restart did not finish ({result}). Auto Maple is '
                  f'PAUSED. Get the character back onto the map and press Start/stop. '
                  f'Screenshots of what it saw are in {RECORD_DIR}.')
            if phone_alert.wants('bot_paused'):
                phone_alert.send(f'⚠️ MapleStory restart did not finish ({result}). '
                                 'Auto Maple is paused.', photo=True)
        self.paused_routine = False

    @staticmethod
    def _await_minimap(config, seconds=60.0):
        """Has the capture thread find the minimap again in the new window (it
        can sit elsewhere, and the old area was measured in the old window),
        and gives it time to."""
        capture = getattr(config, 'capture', None)
        if capture is not None:
            capture.calibrated = False
            time.sleep(1.0)             # the capture loop notices and starts over
        started = time.monotonic()
        while capture is not None and time.monotonic() - started < seconds:
            if getattr(capture, 'calibrated', False):
                time.sleep(2.0)         # ...and to read the position once
                return True
            time.sleep(0.5)
        return capture is None


RUNNER = RestartRunner()


def handle_start_stop():
    """Start/stop was pressed. During a restart it calls the restart off (Auto
    Maple then stays stopped) and returns True; otherwise False."""
    if not RUNNER.busy:
        return False
    RUNNER.stop()
    print('\n[~] Restart called off; Auto Maple stays stopped. Press Start/stop again '
          'once the character is on the map.')
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Restart MapleStory and log back in.')
    parser.add_argument('--look', action='store_true',
                        help='Only say which screen the game is showing.')
    args = parser.parse_args()
    _load_templates()
    if args.look:
        screen = Screen()
        frame = screen.game()
        print('Game window:', 'none' if frame is None else f'{frame.shape[1]}x{frame.shape[0]}',
              '- screen:', which_screen(frame))
        desktop = screen.desktop()
        print('Launcher PLAY button on the desktop:',
              find(desktop, 'restart_launcher_play') is not None)
        return
    print('Restarting MapleStory in 5 seconds - leave the mouse and keyboard alone...')
    time.sleep(5)
    print(GameRestart().run())


if __name__ == '__main__':
    main()
