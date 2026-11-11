"""Controls for optional sampled-enemy detection and attacks."""

import tkinter as tk
from tkinter import messagebox, ttk

from src.common import config


class MobTracker(ttk.LabelFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, text='Sampled enemy auto-attack', padding=6, **kwargs)
        self.tracker = config.capture.mob_tracker
        current = self.tracker.settings()

        self.enabled = tk.BooleanVar(value=current['Enabled'])
        self.distance = tk.StringVar(value=str(current['Range']))

        ttk.Checkbutton(
            self,
            text='Enable auto-attack (works without a routine)',
            variable=self.enabled,
            command=self._apply_settings
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(self, text='Insert (3s); start on patrol level').grid(
            row=0, column=2, columnspan=2, sticky=tk.E
        )

        ttk.Button(
            self,
            text='Calibrate player + enemies',
            command=config.gui.select_mob_templates
        ).grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(4, 0), padx=(0, 4))
        ttk.Button(
            self,
            text='Add enemy sample',
            command=config.gui.add_mob_template
        ).grid(row=1, column=2, columnspan=2, sticky=tk.EW, pady=(4, 0))

        ttk.Label(self, text='Search range').grid(
            row=2, column=0, sticky=tk.W, pady=(5, 0)
        )
        self.distance_box = tk.Spinbox(
            self,
            from_=20,
            to=600,
            increment=10,
            width=5,
            textvariable=self.distance,
            command=self._apply_settings
        )
        self.distance_box.grid(row=2, column=1, sticky=tk.W, pady=(5, 0))
        self.distance_box.bind('<FocusOut>', self._apply_settings)
        self.distance_box.bind('<Return>', self._apply_settings)
        ttk.Label(self, text='px   2 hits / 0.4 sec; 3+ group: Q').grid(
            row=2, column=2, columnspan=2, sticky=tk.W, pady=(5, 0)
        )

        self.status = tk.StringVar()
        ttk.Label(self, textvariable=self.status, wraplength=460).grid(
            row=3, column=0, columnspan=4, sticky=tk.W, pady=(5, 0)
        )
        self._refresh_status()

    def _apply_settings(self, _event=None):
        try:
            current = self.tracker.settings()
            self.tracker.update_settings(
                self.enabled.get(),
                self.distance.get(),
                current['Threshold']
            )
        except ValueError as error:
            messagebox.showerror(
                'Invalid mob tracking settings',
                str(error),
                parent=self.winfo_toplevel()
            )
            current = self.tracker.settings()
            self.enabled.set(current['Enabled'])
            self.distance.set(str(current['Range']))

    def _refresh_status(self):
        if config.enabled:
            mode = (
                'Standalone patrol running'
                if len(config.routine) == 0
                else 'Routine running'
            )
        else:
            mode = 'Paused'
        self.status.set(f'{mode} | {self.tracker.status()}')
        self.after(300, self._refresh_status)
