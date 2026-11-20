"""Loot pickup controls (a section of the Supplies page)."""

import tkinter as tk
from tkinter import ttk

from src.common import config
from src.gui import theme


class LootPickup:
    def __init__(self, page):
        self.page = page
        self.tracker = config.capture.loot_tracker
        current = self.tracker.settings()
        self.enabled = tk.BooleanVar(value=current['Enabled'])

        body = theme.section(page, 'Loot pickup', hint='6 quick X taps')
        ttk.Checkbutton(body, text='Pick up loot under the character',
                        variable=self.enabled, command=self._apply_settings
                        ).grid(row=0, column=0, columnspan=2, sticky=tk.W)
        ttk.Button(body, text='Add map-specific loot sample',
                   command=config.gui.add_loot_templates
                   ).grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(6, 0))
        self.status = tk.StringVar()
        theme.status_label(body, self.status).grid(row=2, column=0, columnspan=2,
                                                   sticky=tk.EW, pady=(4, 0))
        self._refresh_status()

    def _apply_settings(self):
        self.tracker.update_settings(self.enabled.get())

    def _refresh_status(self):
        self.status.set(self.tracker.status())
        self.page.after(300, self._refresh_status)
