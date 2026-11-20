"""The Log page: the application's console output inside the app."""

import tkinter as tk
from tkinter import ttk

from src.common.output import output_capture
from src.common import vkeys
from src.gui import theme
from src.gui.interfaces import Tab


class Log(Tab):
    MAX_LINES = 3000
    FILL = True             # takes the page's full height; the text scrolls itself

    def __init__(self, parent, **kwargs):
        super().__init__(parent, 'Log', **kwargs)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        bar = ttk.Frame(self)
        bar.grid(row=0, column=0, columnspan=2, sticky=tk.EW, padx=theme.PAD, pady=(8, 4))
        ttk.Label(bar, text='RUNTIME LOG', style='Section.TLabel').pack(side=tk.LEFT)
        self.follow = tk.BooleanVar(value=True)
        ttk.Button(bar, text='Clear', command=self.clear).pack(side=tk.RIGHT)
        follow = ttk.Checkbutton(bar, text='Follow', variable=self.follow)
        follow.keep_enabled = True
        follow.pack(side=tk.RIGHT, padx=(0, 8))

        self.output = tk.Text(self, wrap=tk.WORD, state=tk.DISABLED, bg='#0e1014',
                              fg='#c9cfdb', bd=0, padx=6, pady=4, font=theme.FONT_MONO_SMALL)
        self.output.keep_enabled = True
        self.scrollbar = ttk.Scrollbar(self, command=self.output.yview)
        self.output.configure(yscrollcommand=self.scrollbar.set)
        self.output.grid(row=1, column=0, sticky=tk.NSEW, padx=(theme.PAD, 0))
        self.scrollbar.grid(row=1, column=1, sticky=tk.NS, padx=(0, theme.PAD))
        self.output.tag_configure('stderr', foreground='#ff7b7f')
        self.output.tag_configure('warn', foreground=theme.WARN)
        self.output.tag_configure('violetta', foreground='#c9a7ff')

        self.key_status = tk.StringVar(value='')
        ttk.Label(self, textvariable=self.key_status, style='Muted.TLabel').grid(
            row=2, column=0, columnspan=2, sticky=tk.W, padx=theme.PAD, pady=(4, 8))

    def poll(self):
        held_keys, last_event = vkeys.input_status()
        held_text = ', '.join(key.upper() for key in held_keys) or 'none'
        self.key_status.set(f'Keys held: {held_text}   ·   last input: {last_event}')

        events = output_capture.drain()
        if events:
            self.output.configure(state=tk.NORMAL)
            for stream, text in events:
                if stream == 'stderr':
                    tag = 'stderr'
                elif '[!]' in text:
                    tag = 'warn'
                elif '[VIOLETTA]' in text:
                    tag = 'violetta'
                else:
                    tag = None
                if tag:
                    self.output.insert(tk.END, text, tag)
                else:
                    self.output.insert(tk.END, text)
            line_count = int(self.output.index('end-1c').split('.')[0])
            if line_count > self.MAX_LINES:
                self.output.delete('1.0', f'{line_count - self.MAX_LINES}.0')
            self.output.configure(state=tk.DISABLED)
            if self.follow.get():
                self.output.see(tk.END)

    def clear(self):
        output_capture.drain()
        self.output.configure(state=tk.NORMAL)
        self.output.delete('1.0', tk.END)
        self.output.configure(state=tk.DISABLED)
