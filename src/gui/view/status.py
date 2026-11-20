"""The Game page: which map, channel and client controls, mini-games and the session."""

import time
import tkinter as tk
from tkinter import ttk

from src.common import config
from src.gui import theme
from src.gui.interfaces import Tab
from src.modules.mini_games import TALLY
from src.modules import channel_settings


class Status(Tab):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, 'Game', **kwargs)
        self.curr_cb = tk.StringVar(value='—')
        self.curr_routine = tk.StringVar(value='—')
        self.app_state = tk.StringVar()

        # ---- map ----
        # Which map this is, for its spots and events (map_events.py). Maps
        # without a minimap title crop (A Split Road) are only known this way.
        from src.modules.map_events import MAPS
        body = theme.section(self, 'Map')
        auto = 'Recognise automatically'
        names = [auto] + [m.name for m in MAPS]
        chosen = channel_settings.chosen_map()
        self.map_choice = tk.StringVar(value=chosen if chosen in names else auto)
        self.map_menu = ttk.Combobox(body, textvariable=self.map_choice, values=names,
                                     state='readonly')
        self.map_menu.keep_enabled = True
        self.map_menu.grid(row=0, column=0, columnspan=2, sticky=tk.EW)
        self.map_menu.bind('<<ComboboxSelected>>', lambda e: channel_settings.set_chosen_map(
            '' if self.map_choice.get() == auto else self.map_choice.get()))
        # Never change channel on its own (empty map, players nearby, Iron Hog).
        # Safe to flip while running, so it is left unlocked (see GUI._sync_control_states).
        self.stay_on_channel = tk.BooleanVar(value=channel_settings.stay_on_channel())
        self.stay_check = ttk.Checkbutton(
            body, text='Stay on this channel (never change channel)',
            variable=self.stay_on_channel,
            command=lambda: channel_settings.set_stay_on_channel(self.stay_on_channel.get()))
        self.stay_check.keep_enabled = True
        self.stay_check.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(6, 0))

        # ---- client ----
        # The game client leaks memory that only closing it gives back; this
        # closes the game, starts it again and logs back in (game_restart.py).
        # The bot does the same by itself when memory runs low.
        body = theme.section(self, 'Game client')
        self.memory = tk.StringVar(value='')
        ttk.Label(body, text='Free memory').grid(row=0, column=0, sticky=tk.W)
        ttk.Label(body, textvariable=self.memory, style='Value.TLabel').grid(
            row=0, column=1, sticky=tk.E)
        self.restart_text = tk.StringVar(value='Restart MapleStory')
        self.restart_button = ttk.Button(body, textvariable=self.restart_text,
                                         command=self._restart_game)
        self.restart_button.keep_enabled = True
        self.restart_button.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(6, 0))
        ttk.Label(body, text='Closes the game, starts it again and logs back in '
                             '(frees its memory, about a minute).',
                  style='Muted.TLabel', wraplength=340).grid(
            row=2, column=0, columnspan=2, sticky=tk.W, pady=(4, 0))

        # ---- session ----
        body = theme.section(self, 'Session')
        self.session_rows = {}
        for row, name in enumerate(('Running for', 'EXP gained', 'Levels gained',
                                    'Mini-games')):
            ttk.Label(body, text=name).grid(row=row, column=0, sticky=tk.W, pady=1)
            var = tk.StringVar(value='—')
            ttk.Label(body, textvariable=var, style='Value.TLabel').grid(
                row=row, column=1, sticky=tk.E, pady=1)
            self.session_rows[name] = var
        self._run_started = None
        self._run_total = 0.0
        self._exp_start = None

        # ---- loaded files ----
        body = theme.section(self, 'Loaded')
        for row, (name, var) in enumerate((('Command book', self.curr_cb),
                                           ('Routine', self.curr_routine))):
            ttk.Label(body, text=name).grid(row=row, column=0, sticky=tk.W, pady=1)
            ttk.Label(body, textvariable=var, style='Value.TLabel').grid(
                row=row, column=1, sticky=tk.E, pady=1)
        ttk.Frame(self, height=theme.PAD).pack()
        self._refresh_app_state()

    def _restart_game(self):
        from tkinter import messagebox
        from src.modules import game_restart
        if game_restart.RUNNER.busy:
            game_restart.handle_start_stop()        # pressed again: call it off
            return
        if not messagebox.askyesno(
                'Restart MapleStory',
                'Close MapleStory, start it again and log back in?\n\n'
                'It takes about a minute. Leave the mouse and keyboard alone '
                'until the character is back on the map. Auto Maple keeps running'
                + (' and carries on farming afterwards.' if config.enabled else '.')):
            return
        if not game_restart.RUNNER.request('asked for in the app'):
            messagebox.showinfo('Restart MapleStory',
                                'Not now: a mini-game is being solved. Try again after it.')

    def _refresh_restart_button(self):
        from src.modules import game_restart
        if game_restart.RUNNER.busy:
            self.restart_text.set('Restarting MapleStory… (click to call it off)')
        else:
            self.restart_text.set('Restart MapleStory')
        try:
            from src.modules.memory_guard import system_commit
            commit = system_commit()
            self.memory.set(f'{commit[0] / 1024:.1f} GB' if commit is not None else '—')
        except Exception:
            self.memory.set('—')

    def _refresh_session(self):
        now = time.monotonic()
        if config.enabled and self._run_started is None:
            self._run_started = now
        elif not config.enabled and self._run_started is not None:
            self._run_total += now - self._run_started
            self._run_started = None
        total = self._run_total + (now - self._run_started if self._run_started else 0)
        hours, rest = divmod(int(total), 3600)
        self.session_rows['Running for'].set(f'{hours}:{rest // 60:02d}:{rest % 60:02d}')
        exp = getattr(config.capture, 'exp', None)
        percent = exp.current_percent() if exp else None
        if percent is not None:
            levels = exp.levels_gained()
            total_exp = levels * 100 + percent
            if self._exp_start is None:
                self._exp_start = total_exp
            self.session_rows['EXP gained'].set(f'{total_exp - self._exp_start:+.2f}%')
            self.session_rows['Levels gained'].set(str(levels))
        tally = TALLY.short().replace('Mini-games played: ', '').replace('Mini-games: ', '')
        self.session_rows['Mini-games'].set(tally or 'none yet')

    def _refresh_app_state(self):
        self._ticks = getattr(self, '_ticks', 0) + 1
        if self._ticks % 10 == 1:           # every 2 s
            self._refresh_restart_button()
        self.app_state.set('Auto Maple: ON' if config.enabled else 'Auto Maple: OFF')
        self._refresh_session()
        self.after(200, self._refresh_app_state)

    def set_cb(self, string):
        self.curr_cb.set(string or '—')

    def set_routine(self, string):
        self.curr_routine.set(string or '—')
