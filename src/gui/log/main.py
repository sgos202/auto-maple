"""Displays the application's console output inside the GUI."""

import tkinter as tk
from tkinter import ttk

from src.common.output import output_capture
from src.common import vkeys
from src.gui.interfaces import Tab


class Log(Tab):
    MAX_LINES = 3000

    def __init__(self, parent, **kwargs):
        super().__init__(parent, 'Log', **kwargs)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.output = tk.Text(self, wrap=tk.WORD, state=tk.DISABLED)
        self.scrollbar = ttk.Scrollbar(self, command=self.output.yview)
        self.output.configure(yscrollcommand=self.scrollbar.set)
        self.output.grid(row=0, column=0, sticky=tk.NSEW, padx=(8, 0), pady=8)
        self.scrollbar.grid(row=0, column=1, sticky=tk.NS, padx=(0, 8), pady=8)
        self.output.tag_configure('stderr', foreground='#b00020')

        self.key_status = tk.StringVar(value='')
        ttk.Label(self, textvariable=self.key_status).grid(
            row=1, column=0, columnspan=2, sticky=tk.W, padx=8
        )

        controls = ttk.Frame(self)
        controls.grid(row=2, column=0, columnspan=2, sticky=tk.EW, padx=8, pady=(4, 8))
        ttk.Label(
            controls,
            text='Application output is also shown in the console window.'
        ).pack(side=tk.LEFT)
        ttk.Button(controls, text='Clear', command=self.clear).pack(side=tk.RIGHT)

    def poll(self):
        held_keys, last_event = vkeys.input_status()
        held_text = ', '.join(key.upper() for key in held_keys) or 'none'
        self.key_status.set(
            f'Input overlay — held: {held_text}  |  last: {last_event} '
            '|  SendInput request only; game receipt is not verified.'
        )

        events = output_capture.drain()
        if events:
            follow_output = self.output.yview()[1] >= 0.99
            self.output.configure(state=tk.NORMAL)
            for stream, text in events:
                if stream == 'stderr':
                    self.output.insert(tk.END, text, 'stderr')
                else:
                    self.output.insert(tk.END, text)
            line_count = int(self.output.index('end-1c').split('.')[0])
            if line_count > self.MAX_LINES:
                self.output.delete('1.0', f'{line_count - self.MAX_LINES}.0')
            self.output.configure(state=tk.DISABLED)
            if follow_output:
                self.output.see(tk.END)

    def clear(self):
        output_capture.drain()
        self.output.configure(state=tk.NORMAL)
        self.output.delete('1.0', tk.END)
        self.output.configure(state=tk.DISABLED)
