import tkinter as tk
from src.gui.interfaces import LabelFrame
from src.common import config


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
        self._refresh_app_state()

    def _refresh_app_state(self):
        if config.enabled:
            self.app_state.set('Auto Maple: ON')
            self.app_state_label.configure(fg='#16803c')
        else:
            self.app_state.set('Auto Maple: OFF')
            self.app_state_label.configure(fg='#b42318')
        self.after(200, self._refresh_app_state)

    def set_cb(self, string):
        self.curr_cb.set(string)

    def set_routine(self, string):
        self.curr_routine.set(string)
