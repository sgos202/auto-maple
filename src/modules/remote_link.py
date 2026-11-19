"""
Auto Maple's side of the phone remote (src/modules/phone_remote.py).

The remote is a separate, always-running program: it reads your Telegram
commands and can start Auto Maple when it isn't running. Only one program
may read a Telegram bot's messages, so Auto Maple itself never does; the two
talk through files instead:

 - .settings/remote/state.json   written here every few seconds: pid,
   farming or not, map, memory. The remote reads it for /status and to tell
   whether Auto Maple is running.
 - .settings/remote/inbox/*.json commands for the running Auto Maple, dropped
   by the remote: go, pause, restart_game, quit. Replies go straight to the
   phone (phone_alert).
"""

import json
import os
import threading
import time

FOLDER = os.path.join('.settings', 'remote')
STATE = os.path.join(FOLDER, 'state.json')
INBOX = os.path.join(FOLDER, 'inbox')
STATE_SECONDS = 3.0
INBOX_SECONDS = 0.5
STALE_COMMAND_SECONDS = 120     # older commands (sent while Auto Maple was down) are dropped

_thread = None
_started = time.time()


def start():
    global _thread
    if _thread is None:
        os.makedirs(INBOX, exist_ok=True)
        _thread = threading.Thread(target=_main, name='remote link', daemon=True)
        _thread.start()


def post(command, folder=INBOX):
    """(Used by the remote) drops COMMAND for the running Auto Maple."""
    os.makedirs(folder, exist_ok=True)
    name = f'{time.time():.3f}-{os.getpid()}.json'
    temp = os.path.join(folder, name + '.tmp')
    with open(temp, 'w', encoding='utf-8') as file:
        json.dump({'command': command, 'time': time.time()}, file)
    os.replace(temp, os.path.join(folder, name))


def read_state(path=STATE):
    try:
        with open(path, encoding='utf-8') as file:
            return json.load(file)
    except Exception:
        return None


def _main():
    last_state = 0.0
    while True:
        try:
            now = time.monotonic()
            if now - last_state >= STATE_SECONDS:
                last_state = now
                _write_state()
            for command in _take_commands():
                _handle(command)
        except Exception as error:
            print(f'\n[!] Phone remote link: {error}')
        time.sleep(INBOX_SECONDS)


def state():
    """What the remote shows in /status."""
    from src.common import config
    info = {'pid': os.getpid(), 'time': time.time(), 'started': _started,
            'enabled': bool(getattr(config, 'enabled', False))}
    try:
        watcher = config.bot.map_events.watcher
        info['map'] = getattr(watcher.map, 'name', None)
    except Exception:
        info['map'] = None
    try:
        from src.modules import game_restart, mini_games
        info['restarting_game'] = game_restart.RUNNER.busy
        runner = mini_games.ACTIVE_RUNNER
        info['mini_game'] = bool(runner is not None and runner.busy)
    except Exception:
        pass
    try:
        from src.modules.memory_guard import game_handles, own_memory_mb, system_commit
        commit = system_commit()
        if commit:
            info['free_mb'], info['limit_mb'] = round(commit[0]), round(commit[1])
        info['bot_mb'] = round(own_memory_mb() or 0)
        info['game_handles'] = game_handles()
    except Exception:
        pass
    try:
        capture = config.capture
        info['minimap_found'] = bool(capture.calibrated)
        pos = getattr(config, 'player_pos', None)
        info['position'] = [round(v, 3) for v in pos] if pos else None
    except Exception:
        pass
    return info


def _write_state():
    temp = STATE + '.tmp'
    with open(temp, 'w', encoding='utf-8') as file:
        json.dump(state(), file)
    os.replace(temp, STATE)


def _take_commands():
    try:
        names = sorted(n for n in os.listdir(INBOX) if n.endswith('.json'))
    except FileNotFoundError:
        return []
    commands = []
    for name in names:
        path = os.path.join(INBOX, name)
        try:
            with open(path, encoding='utf-8') as file:
                message = json.load(file)
        except Exception:
            message = None
        try:
            os.remove(path)
        except OSError:
            pass
        if message and time.time() - message.get('time', 0) <= STALE_COMMAND_SECONDS:
            commands.append(message.get('command'))
    return commands


def _reply(text, photo=False):
    print(f'\n[~] Phone remote: {text}')
    try:
        from src.modules import phone_alert
        phone_alert.send(text, photo=photo)
    except Exception:
        pass


def _handle(command):
    handler = {'go': _go, 'pause': _pause, 'restart_game': _restart_game,
               'quit': _quit}.get(command)
    if handler is None:
        _reply(f'Unknown command for Auto Maple: {command!r}')
        return
    # (go counts down and waits for the game: not on this thread)
    threading.Thread(target=handler, name=f'remote {command}', daemon=True).start()


def _busy():
    """Why the bot can't be started or stopped from the phone right now, or None."""
    from src.modules import game_restart, mini_games
    if game_restart.RUNNER.busy:
        return 'MapleStory is being restarted; farming carries on by itself afterwards.'
    runner = mini_games.ACTIVE_RUNNER
    if runner is not None and runner.busy:
        return 'a mini-game is being solved right now; try again when it ends.'
    return None


def _go():
    from src.common import config
    from src.modules.game_restart import Screen
    from src.modules.listener import Listener
    if config.enabled:
        _reply('▶️ Already farming.', photo=True)
        return
    reason = _busy()
    if reason:
        _reply(f'Not started: {reason}')
        return
    if not Screen().focus():
        _reply('⚠️ Not started: the MapleStory window could not be brought to the front '
               '(is the game running? /restartgame starts it).', photo=True)
        return
    _reply('Starting Auto Maple (3 s countdown)...')
    # Listener.toggle_enabled checks the routine / enemy samples / minimap and
    # that the game is in front after its countdown, printing why it refused
    for _ in range(2):
        Listener.toggle_enabled()
        if config.enabled:
            break
        Screen().focus()        # something took the focus during the countdown: once more
    if config.enabled:
        _reply('▶️ Farming.', photo=True)
    else:
        _reply('⚠️ Auto Maple did not start; see its console (no routine/enemy samples '
               'loaded, minimap not found, or the game was not in front).', photo=True)


def _pause():
    from src.common import config
    from src.modules.listener import Listener
    if not config.enabled:
        _reply('⏸ Already paused.')
        return
    reason = _busy()
    if reason:
        _reply(f'Not paused: {reason}')
        return
    Listener.toggle_enabled()
    _reply('⏸ Paused.' if not config.enabled else '⚠️ Could not pause.', photo=True)


def _restart_game():
    from src.modules import game_restart
    if game_restart.RUNNER.busy:
        _reply('MapleStory is already being restarted.')
    elif game_restart.RUNNER.request('asked from the phone'):
        _reply('🔄 Restarting MapleStory; you get a message when it is back.')
    else:
        _reply('Not restarting MapleStory: a mini-game is being solved.')


def _quit():
    """Closes Auto Maple cleanly: farming off, every key let go, then exit."""
    from src.common import config
    from src.common.vkeys import release_all_keys
    print('\n[~] Phone remote: closing Auto Maple.')    # (the remote tells the phone)
    config.enabled = False      # the bot loop lets go of its keys
    time.sleep(1.0)
    try:
        release_all_keys()
    except Exception:
        pass
    try:
        os.remove(STATE)        # the remote sees it gone at once
    except OSError:
        pass
    os._exit(0)
