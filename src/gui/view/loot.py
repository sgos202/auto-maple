"""Controls for optional sampled-loot pickup."""

import tkinter as tk
from tkinter import ttk

from src.common import config
from src.gui.interfaces import Frame


class LootPickup(Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.tracker = config.capture.loot_tracker
        current = self.tracker.settings()

        self.enabled = tk.BooleanVar(value=current['Enabled'])

        ttk.Checkbutton(
            self,
            text='Auto-pick up loot under player (6 quick X taps)',
            variable=self.enabled,
            command=self._apply_settings
        ).grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        ttk.Button(
            self,
            text='Add map-specific loot sample',
            command=config.gui.add_loot_templates
        ).grid(row=0, column=1, sticky=tk.W)

        self.status = tk.StringVar()
        ttk.Label(self, textvariable=self.status, wraplength=480).grid(
            row=1, column=0, columnspan=2, sticky=tk.W, pady=(3, 0)
        )
        self._refresh_status()

    def _apply_settings(self):
        self.tracker.update_settings(self.enabled.get())

    def _refresh_status(self):
        mode = 'on' if self.tracker.settings()['Enabled'] else 'off'
        routine_state = 'running' if config.enabled else 'paused'
        self.status.set(
            f'Loot pickup {mode} | Auto Maple {routine_state} | {self.tracker.status()}'
        )
        self.after(300, self._refresh_status)
