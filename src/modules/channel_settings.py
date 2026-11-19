"""
"Stay on this channel": never change channel on its own.

Normally the bot changes channel when the map is empty, when other players
crowd its platform, and to escape the Iron Hog. With this on (checkbox in the
View tab's Status panel, saved in .settings/channel):

* empty map / players nearby: it stays and keeps farming;
* Iron Hog: it hides on the rope until the hog has been gone for a while, then
  jumps off and goes back to farming (it pauses itself if the hog never leaves).
"""

import threading

from src.common.interfaces import Configurable


class ChannelSettings(Configurable):
    DEFAULT_CONFIG = {
        'Stay on channel': False,
        'Map': '',              # a MapProfile name, or '' to recognise the map
    }

    def __init__(self):
        super().__init__('channel')


_settings = None
_lock = threading.Lock()


def settings():
    global _settings
    with _lock:
        if _settings is None:
            try:
                _settings = ChannelSettings()
            except Exception as error:      # unreadable settings file: use the default
                print(f'\n[!] Could not load the channel settings ({error}); '
                      'channel changes stay on.')
                _settings = type('Defaults', (), {'config': dict(ChannelSettings.DEFAULT_CONFIG),
                                                  'save_config': lambda self: None})()
        return _settings


def stay_on_channel():
    return bool(settings().config.get('Stay on channel', False))


def set_stay_on_channel(stay):
    current = settings()
    current.config['Stay on channel'] = bool(stay)
    try:
        current.save_config()
    except Exception as error:
        print(f'\n[!] Could not save the channel setting: {error}')
    print(f"\n[~] Stay on this channel: {'ON - Auto Maple will not change channel' if stay else 'OFF'}.")


def chosen_map():
    """The map picked in the GUI ('' = recognise it from the minimap title)."""
    return str(settings().config.get('Map', '') or '')


def set_chosen_map(name):
    current = settings()
    current.config['Map'] = name or ''
    try:
        current.save_config()
    except Exception as error:
        print(f'\n[!] Could not save the map setting: {error}')
    print(f"\n[~] Map: {name or 'recognise it automatically'}.")
