import tkinter as tk
from src.gui.interfaces import LabelFrame
from src.common import config
from src.modules.mini_games import TALLY
from src.modules import channel_settings


class Status(LabelFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, 'Status', **kwargs)

        self.grid_columnconfigure(1, weight=1)

        self.curr_cb = tk.StringVar()
        self.curr_routine = tk.StringVar()

        self.cb_label = tk.Label(self, text='Command Book:')
        self.cb_label.grid(row=0, column=0, padx=5, pady=(5, 0), sticky=tk.E)
        self.cb_entry = tk.Entry(self, textvariable=self.curr_cb, state=tk.DISABLED)
        self.cb_entry.grid(row=0, column=1, padx=(0, 5), pady=(5, 0), sticky=tk.EW)

        self.r_label = tk.Label(self, text='Routine:')
        self.r_label.grid(row=1, column=0, padx=5, pady=(0, 5), sticky=tk.E)
        self.r_entry = tk.Entry(self, textvariable=self.curr_routine, state=tk.DISABLED)
        self.r_entry.grid(row=1, column=1, padx=(0, 5), pady=(0, 5), sticky=tk.EW)

        self.app_state = tk.StringVar()
        self.app_state_label = tk.Label(
            self, textvariable=self.app_state, font=('', 10, 'bold')
        )
        self.app_state_label.grid(row=2, column=0, columnspan=2, pady=(4, 5))

        self.mini_games = tk.StringVar(value=TALLY.short())
        tk.Label(self, textvariable=self.mini_games, wraplength=260,
                 justify=tk.CENTER).grid(row=3, column=0, columnspan=2, padx=5, pady=(0, 5))

        # Never change channel on its own (empty map, players nearby, Iron Hog).
        # Safe to flip while running, so it is left unlocked (see GUI._sync_control_states).
        self.stay_on_channel = tk.BooleanVar(value=channel_settings.stay_on_channel())
        self.stay_check = tk.Checkbutton(
            self, text='Stay on this channel\n(never change channel)',
            variable=self.stay_on_channel,
            anchor=tk.W, justify=tk.LEFT,
            command=lambda: channel_settings.set_stay_on_channel(self.stay_on_channel.get()))
        self.stay_check.keep_enabled = True
        self.stay_check.grid(row=4, column=0, columnspan=2, padx=5, pady=(0, 5), sticky=tk.EW)
        # The game client leaks memory that only closing it gives back; this
        # closes the game, starts it again and logs back in (game_restart.py).
        # The bot does the same by itself when memory runs low.
        self.restart_text = tk.StringVar(value='Restart MapleStory\n(frees its memory)')
        self.restart_button = tk.Button(self, textvariable=self.restart_text,
                                        command=self._restart_game)
        self.restart_button.keep_enabled = True
        self.restart_button.grid(row=5, column=0, columnspan=2, padx=5, pady=(0, 5),
                                 sticky=tk.EW)
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
            self.restart_text.set('Restarting MapleStory...\n(click to call it off)')
            return
        text = 'Restart MapleStory\n(frees its memory)'
        try:
            from src.modules.memory_guard import system_commit
            commit = system_commit()
            if commit is not None:
                text = f'Restart MapleStory\n(memory free: {commit[0] / 1024:.1f} GB)'
        except Exception:
            pass
        self.restart_text.set(text)

    def _refresh_app_state(self):
        self._ticks = getattr(self, '_ticks', 0) + 1
        if self._ticks % 10 == 1:           # every 2 s
            self._refresh_restart_button()
        if config.enabled:
            self.app_state.set('Auto Maple: ON')
            self.app_state_label.configure(fg='#16803c')
        else:
            self.app_state.set('Auto Maple: OFF')
            self.app_state_label.configure(fg='#b42318')
        self.mini_games.set(TALLY.short())
        self.after(200, self._refresh_app_state)

    def set_cb(self, string):
        self.curr_cb.set(string)

    def set_routine(self, string):
        self.curr_routine.set(string)
