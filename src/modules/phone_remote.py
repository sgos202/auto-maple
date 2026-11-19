"""
Phone remote: control Auto Maple from Telegram, even when it isn't running.

A small program of its own that runs all the time (started at log-on by
Windows, see "Install Phone Remote.bat"), reads the commands you send to your
Telegram alert bot and answers them. Only messages from your own chat (the
one in .settings/telegram.json) are obeyed.

  /status        screenshot of the whole desktop + what Auto Maple is doing
  /start         start Auto Maple (if it isn't running)
  /restart       close Auto Maple and start it again
  /quit          close Auto Maple
  /go            start farming (like F3)        - Auto Maple must be running
  /pause         stop farming (like F3)         - Auto Maple must be running
  /restartgame   restart MapleStory             - Auto Maple must be running
  /help          this list

Auto Maple runs as Administrator. Started by Windows with "highest
privileges", the remote is too, so it starts Auto Maple without a UAC box
(no one could click it from a phone).

Run by hand (for a test):  venv\\Scripts\\python.exe -m src.modules.phone_remote
It talks to the running Auto Maple through files, see remote_link.py.
"""

import ctypes
import datetime
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.modules import phone_alert, remote_link        # noqa: E402

LOG = os.path.join(ROOT, 'logs', 'phone_remote.log')
PYTHON = os.path.join(ROOT, 'venv', 'Scripts', 'python.exe')
POLL_SECONDS = 25               # Telegram long-poll
STATE_FRESH_SECONDS = 20        # Auto Maple writes its state every 3 s
START_WAIT_SECONDS = 120        # for a started Auto Maple to report in
QUIT_WAIT_SECONDS = 15          # for Auto Maple to close itself before it is killed
STALE_MESSAGE_SECONDS = 180     # commands older than this (sent while the PC was off) are ignored
SCREEN_MAX_WIDTH = 1920
MUTEX = 'Local\\AutoMaplePhoneRemote'

COMMANDS = [
    ('status', 'Screenshot of the desktop + what Auto Maple is doing'),
    ('start', 'Start Auto Maple'),
    ('restart', 'Close Auto Maple and start it again'),
    ('quit', 'Close Auto Maple'),
    ('go', 'Start farming (like F3)'),
    ('pause', 'Stop farming (like F3)'),
    ('restartgame', 'Restart MapleStory'),
    ('help', 'List the commands'),
]


def log(text):
    line = f'{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {text}'
    try:
        print(line)
    except Exception:       # (a console that can't show emoji; none at all under pythonw)
        pass
    try:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with open(LOG, 'a', encoding='utf-8') as file:
            file.write(line + '\n')
    except OSError:
        pass


def is_admin():
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


# ---- Telegram ----

def api(method, http_timeout=15, **data):
    import requests
    s = phone_alert._load()
    response = requests.post(phone_alert.API.format(token=s['token'], method=method),
                             data=data, timeout=http_timeout)
    return response.json()


def reply(text, image=None):
    log(f'-> {text.splitlines()[0][:120]}')
    if not phone_alert._deliver(text, image):
        log('   (sending failed)')


# ---- Auto Maple's processes ----

def _processes():
    """[(pid, parent pid, name, command line)] of python/powershell processes."""
    script = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe' or Name='pythonw.exe' "
              "or Name='powershell.exe'\" | Select-Object ProcessId,ParentProcessId,Name,"
              "CommandLine | ConvertTo-Json -Compress")
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-Command', script],
                             capture_output=True, text=True, timeout=30,
                             creationflags=subprocess.CREATE_NO_WINDOW).stdout
        rows = json.loads(out) if out.strip() else []
    except Exception as error:
        log(f'process list failed: {error}')
        return []
    if isinstance(rows, dict):
        rows = [rows]
    return [(r['ProcessId'], r['ParentProcessId'], r['Name'], r.get('CommandLine') or '')
            for r in rows]


def auto_maple_pids():
    """(python pids running Auto Maple's main.py, their console windows' pids)."""
    procs = _processes()
    root = ROOT.lower()
    bots = [pid for pid, _, name, cmd in procs
            if name.lower().startswith('python') and 'main.py' in cmd.lower()
            and root in cmd.lower()]
    # venv\Scripts\python.exe is a launcher: the real interpreter is its child,
    # whose command line names the base Python, not this folder
    bots += [pid for pid, parent, name, cmd in procs
             if parent in bots and pid not in bots and name.lower().startswith('python')]
    # The PowerShell window "Start Auto Maple.bat" opens (-NoExit) stays after
    # python ends: close it too, or every restart leaves one more window
    shells = [pid for pid, _, name, cmd in procs
              if name.lower() == 'powershell.exe' and 'main.py' in cmd.lower()
              and 'auto-maple' in cmd.lower()]
    state = remote_link.read_state()
    if state and _fresh(state) and state.get('pid') and state['pid'] not in bots:
        bots.append(state['pid'])
    return bots, shells


def _fresh(state):
    return state is not None and time.time() - state.get('time', 0) <= STATE_FRESH_SECONDS


def running_state():
    """Auto Maple's latest state when it is running (reporting in), else None."""
    state = remote_link.read_state()
    return state if _fresh(state) else None


def start_auto_maple():
    """Opens Auto Maple in its own PowerShell window, like "Start Auto Maple.bat"."""
    # The same command as the .bat (single quotes only: no quoting trouble on
    # the way through two PowerShells; the folder has no spaces)
    verb = '' if is_admin() else ' -Verb RunAs'
    command = (f"Start-Process powershell{verb} -ArgumentList '-NoExit','-Command',"
               f"'cd {ROOT}; .\\venv\\Scripts\\python.exe main.py'")
    subprocess.Popen(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                      '-Command', command], creationflags=subprocess.CREATE_NO_WINDOW)
    log(f'started Auto Maple (remote elevated: {is_admin()})')


def wait_for_start(old_pid=None):
    deadline = time.time() + START_WAIT_SECONDS
    while time.time() < deadline:
        state = running_state()
        if state and state.get('pid') != old_pid:
            return state
        time.sleep(2)
    return None


def stop_auto_maple():
    """Asks Auto Maple to close (it lets go of every key first), then makes sure."""
    bots, shells = auto_maple_pids()
    if not bots and not shells:
        return False
    if running_state():
        remote_link.post('quit')
        deadline = time.time() + QUIT_WAIT_SECONDS
        while time.time() < deadline and auto_maple_pids()[0]:
            time.sleep(1)
    bots, shells = auto_maple_pids()
    for pid in bots + shells:
        subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'], capture_output=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    if bots:
        log(f'force-closed Auto Maple {bots}')
    try:
        os.remove(remote_link.STATE)
    except OSError:
        pass
    return True


# ---- answers ----

def desktop_screenshot():
    """The whole desktop (every monitor) as JPEG bytes, or None."""
    import cv2
    import mss
    import numpy as np
    try:
        with mss.mss() as screen:
            frame = np.asarray(screen.grab(screen.monitors[0]))[:, :, :3]
    except Exception as error:
        log(f'screenshot failed: {error}')
        return None
    if frame.shape[1] > SCREEN_MAX_WIDTH:
        scale = SCREEN_MAX_WIDTH / frame.shape[1]
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode('.jpg', np.ascontiguousarray(frame), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpeg.tobytes() if ok else None


def _duration(seconds):
    minutes = int(seconds // 60)
    return f'{minutes // 60} h {minutes % 60:02d} min' if minutes >= 60 else f'{minutes} min'


def _game_running():
    out = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq MapleStory.exe', '/NH'],
                         capture_output=True, text=True,
                         creationflags=subprocess.CREATE_NO_WINDOW).stdout
    return 'maplestory.exe' in out.lower()


def status_text():
    state = running_state()
    lines = []
    if state is None:
        bots, _ = auto_maple_pids()
        lines.append('🔴 Auto Maple is NOT running.' if not bots else
                     '🟠 Auto Maple is running but not reporting (starting up or stuck).')
        lines.append('Send /start to start it.' if not bots else 'Send /restart if it stays like this.')
    else:
        doing = ('🧩 solving a mini-game' if state.get('mini_game') else
                 '🔄 restarting MapleStory' if state.get('restarting_game') else
                 '▶️ farming' if state.get('enabled') else '⏸ paused')
        lines.append(f'🟢 Auto Maple is running ({_duration(time.time() - state["started"])}): {doing}.')
        if state.get('map'):
            lines.append(f'Map: {state["map"]}')
        if not state.get('minimap_found', True):
            lines.append('Minimap not found.')
        if state.get('free_mb') is not None:
            lines.append(f'Memory: {state["free_mb"]:,} of {state["limit_mb"]:,} MB free, '
                         f'bot {state.get("bot_mb", 0)} MB')
        if state.get('game_handles'):
            lines.append(f'MapleStory handles: {state["game_handles"]:,}')
    lines.append('MapleStory is running.' if _game_running() else '⚠️ MapleStory is NOT running.')
    return '\n'.join(lines)


def handle(text):
    word = text.strip().split()[0].lower().lstrip('/').split('@')[0] if text.strip() else ''
    word = {'restart_game': 'restartgame', 'screen': 'status', 'stop': 'pause',
            'resume': 'go', 'farm': 'go', 'commands': 'help'}.get(word, word)
    log(f'<- {text.strip()[:80]}')
    if word == 'status':
        reply(status_text(), desktop_screenshot())
    elif word == 'start':
        if running_state() or auto_maple_pids()[0]:
            reply('Auto Maple is already running. /restart to restart it, /go to farm.')
            return
        reply('Starting Auto Maple...')
        start_auto_maple()
        _report_start(None)
    elif word == 'restart':
        old = running_state()
        reply('Restarting Auto Maple...')
        stop_auto_maple()
        time.sleep(2)
        start_auto_maple()
        _report_start(old.get('pid') if old else None)
    elif word == 'quit':
        reply('Auto Maple closed.' if stop_auto_maple() else 'Auto Maple was not running.')
    elif word in ('go', 'pause', 'restartgame'):
        if not running_state():
            reply('Auto Maple is not running - send /start first.')
            return
        remote_link.post({'restartgame': 'restart_game'}.get(word, word))
        log(f'   passed "{word}" to Auto Maple')
    else:
        reply('Commands:\n' + '\n'.join(f'/{name} - {what}' for name, what in COMMANDS))


def _report_start(old_pid):
    state = wait_for_start(old_pid)
    if state:
        reply('✅ Auto Maple is up. Send /go to start farming (the game must be on the map).',
              desktop_screenshot())
    else:
        hint = '' if is_admin() else (' The remote is not running as Administrator, so '
                                      'Windows may be waiting for a UAC "Yes" on the PC.')
        reply(f'⚠️ Auto Maple did not report in within {START_WAIT_SECONDS} s.{hint}',
              desktop_screenshot())


# ---- main loop ----

def _single_instance():
    """False when another remote already runs (two would fight over the messages)."""
    ctypes.windll.kernel32.CreateMutexW(None, False, MUTEX)
    return ctypes.windll.kernel32.GetLastError() != 183      # ERROR_ALREADY_EXISTS


def main():
    if not _single_instance():
        log('another phone remote is already running; exiting')
        return
    if not phone_alert.enabled():
        log('Telegram is not set up (python -m src.modules.phone_alert --setup TOKEN); exiting')
        return
    chat = str(phone_alert._load()['chat_id'])
    log(f'phone remote started (Administrator: {is_admin()})')
    try:
        api('setMyCommands', commands=json.dumps(
            [{'command': name, 'description': what} for name, what in COMMANDS]))
    except Exception as error:
        log(f'could not set the command menu: {error}')
    offset = None
    failures = 0
    while True:
        try:
            data = {'timeout': POLL_SECONDS, 'allowed_updates': json.dumps(['message'])}
            if offset is not None:
                data['offset'] = offset
            result = api('getUpdates', http_timeout=POLL_SECONDS + 10, **data)
            if not result.get('ok'):
                raise RuntimeError(result.get('description', 'getUpdates failed'))
            failures = 0
            for update in result.get('result', []):
                offset = update['update_id'] + 1
                message = update.get('message') or {}
                if str(message.get('chat', {}).get('id')) != chat:
                    continue                    # only you
                if time.time() - message.get('date', 0) > STALE_MESSAGE_SECONDS:
                    log(f'ignored an old command: {message.get("text", "")[:40]!r}')
                    continue
                text = message.get('text') or ''
                if text:
                    try:
                        handle(text)
                    except Exception as error:
                        log(f'command failed: {error!r}')
                        reply(f'⚠️ That failed: {error}')
        except Exception as error:
            failures += 1
            if failures in (1, 10) or failures % 100 == 0:
                log(f'Telegram not reachable ({type(error).__name__}: {error}); retrying')
            time.sleep(min(60, 2 * failures))


if __name__ == '__main__':
    main()
