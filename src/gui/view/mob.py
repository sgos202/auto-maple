"""Controls for optional sampled-enemy detection and attacks."""

import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk
from src.common import config


class MobTracker(ttk.Frame):
    SAMPLE_PREVIEW_SIZE = (72, 72)

    def __init__(self, parent, **kwargs):
        super().__init__(parent, padding=(0, 3), **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.tracker = config.capture.mob_tracker
        current = self.tracker.settings()
        self.sample_vars = []
        self.speed_vars = []
        self.sample_images = []

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
        ttk.Label(self, text='px   Hold attack; 3+ in-range group: Q').grid(
            row=2, column=2, columnspan=2, sticky=tk.W, pady=(5, 0)
        )

        self.status = tk.StringVar()
        ttk.Label(self, textvariable=self.status, wraplength=700).grid(
            row=3, column=0, columnspan=4, sticky=tk.EW, pady=(5, 0)
        )

        self.samples = ttk.LabelFrame(
            self, text='Enemy samples — focused samples are preferred', padding=5
        )
        self.samples.grid(
            row=4, column=0, columnspan=4, sticky=tk.EW, pady=(5, 0)
        )
        self.samples.grid_columnconfigure(0, weight=1)
        self.sample_canvas = tk.Canvas(
            self.samples, height=112, highlightthickness=0
        )
        self.sample_canvas.grid(row=0, column=0, sticky=tk.EW)
        self.sample_scroll = ttk.Scrollbar(
            self.samples, orient=tk.HORIZONTAL, command=self.sample_canvas.xview
        )
        self.sample_scroll.grid(row=1, column=0, sticky=tk.EW)
        self.sample_canvas.configure(xscrollcommand=self.sample_scroll.set)
        self.sample_container = ttk.Frame(self.sample_canvas)
        self.sample_window = self.sample_canvas.create_window(
            (0, 0), window=self.sample_container, anchor=tk.NW
        )
        self.sample_container.bind(
            '<Configure>',
            lambda _event: self.sample_canvas.configure(
                scrollregion=self.sample_canvas.bbox('all')
            )
        )
        self.sample_canvas.bind(
            '<Configure>',
            lambda event: self.sample_canvas.itemconfigure(
                self.sample_window, height=event.height
            )
        )
        self.refresh_samples()
        self._refresh_status()

    def refresh_samples(self):
        for widget in self.sample_container.winfo_children():
            widget.destroy()
        self.sample_vars.clear()
        self.speed_vars.clear()
        self.sample_images.clear()

        samples = self.tracker.enemy_samples()
        focused = self.tracker.focused_enemy_samples()
        speed_profiles = self.tracker.enemy_speed_profiles()
        if not samples:
            ttk.Label(
                self.sample_container,
                text='No enemy samples saved yet.'
            ).pack(side=tk.LEFT, padx=8, pady=8)
            return

        for index, sample in enumerate(samples):
            card = ttk.Frame(self.sample_container, padding=(4, 2))
            card.pack(side=tk.LEFT, padx=3)

            if sample.ndim == 2:
                preview = Image.fromarray(sample)
            elif sample.shape[2] == 4:
                preview = Image.fromarray(
                    cv2.cvtColor(sample, cv2.COLOR_BGRA2RGBA)
                )
            else:
                preview = Image.fromarray(
                    cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
                )
            preview.thumbnail(self.SAMPLE_PREVIEW_SIZE, Image.Resampling.LANCZOS)
            image = ImageTk.PhotoImage(preview)
            self.sample_images.append(image)
            ttk.Label(card, image=image).pack()

            variable = tk.BooleanVar(value=index in focused)
            self.sample_vars.append(variable)
            ttk.Checkbutton(
                card,
                text=f'Focus {index + 1}',
                variable=variable,
                command=self._apply_focus_samples
            ).pack()
            speed_variable = tk.BooleanVar(
                value=speed_profiles[index] == 'fast'
            )
            self.speed_vars.append(speed_variable)
            ttk.Checkbutton(
                card,
                text='Fast / jumping',
                variable=speed_variable,
                command=lambda sample_index=index, value=speed_variable:
                    self._apply_speed_profile(sample_index, value)
            ).pack()

    def _apply_focus_samples(self):
        self.tracker.set_focused_enemy_samples(
            index for index, variable in enumerate(self.sample_vars)
            if variable.get()
        )

    def _apply_speed_profile(self, index, variable):
        profile = 'fast' if variable.get() else 'normal'
        try:
            self.tracker.set_enemy_speed_profile(index, profile)
        except ValueError as error:
            messagebox.showerror(
                'Invalid enemy speed profile',
                str(error),
                parent=self.winfo_toplevel()
            )
            self.refresh_samples()

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
