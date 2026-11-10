"""Controls for optional screen-based HP potion use."""

import tkinter as tk
from tkinter import messagebox, ttk

from src.common import config, vkeys
from src.gui.interfaces import Frame
from src.modules.health import HP_TRIGGER_PERCENT


class HealthPotion(Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.monitor = config.capture.health
        current = self.monitor.settings()

        self.enabled = tk.BooleanVar(value=current['Enabled'])
        self.key = tk.StringVar(value=current['Key'])

        ttk.Checkbutton(
            self,
            text='Auto HP potion',
            variable=self.enabled,
            command=self._apply_settings
        ).grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        ttk.Label(self, text=f'Use at or below {HP_TRIGGER_PERCENT}%').grid(
            row=0, column=1, sticky=tk.W
        )
        ttk.Label(self, text='Potion key').grid(
            row=0, column=2, sticky=tk.W, padx=(8, 2)
        )
        self.key_box = ttk.Combobox(
            self,
            textvariable=self.key,
            values=tuple(sorted(vkeys.KEY_MAP)),
            width=8,
            state='readonly'
        )
        self.key_box.grid(row=0, column=3, sticky=tk.W)
        self.key_box.bind('<<ComboboxSelected>>', self._apply_settings)

        self.status = tk.StringVar()
        ttk.Label(self, textvariable=self.status).grid(
            row=1, column=0, columnspan=4, sticky=tk.W, pady=(3, 0)
        )
        self._refresh_status()

    def _apply_settings(self, _event=None):
        if config.enabled:
            messagebox.showwarning(
                'Pause Auto Maple first',
                'Pause Auto Maple before changing HP potion settings.',
                parent=self.winfo_toplevel()
            )
            current = self.monitor.settings()
            self.enabled.set(current['Enabled'])
            self.key.set(current['Key'])
            return

        try:
            self.monitor.update_settings(
                self.enabled.get(),
                HP_TRIGGER_PERCENT,
                self.key.get()
            )
        except ValueError as error:
            messagebox.showerror(
                'Invalid HP potion settings',
                str(error),
                parent=self.winfo_toplevel()
            )
            current = self.monitor.settings()
            self.enabled.set(current['Enabled'])
            self.key.set(current['Key'])

    def _refresh_status(self):
        current = self.monitor.settings()
        percent = self.monitor.current_percent()
        if percent is None:
            text = 'HP bar not detected at the fixed bottom HUD position.'
        else:
            routine_state = 'running' if config.enabled else 'paused'
            action = self.monitor.last_action()
            action_status = f'   |   {action}' if action else ''
            mode = 'on' if current['Enabled'] else 'off'
            text = (
                f'HP: {percent:.0f}%   |   Auto potion {mode} at '
                f'{HP_TRIGGER_PERCENT}%   |   Auto Maple {routine_state}'
                f'{action_status}'
            )
        self.status.set(text)
        self.after(300, self._refresh_status)
