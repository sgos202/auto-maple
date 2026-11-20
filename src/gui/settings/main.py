"""The Keys page: Auto Maple's hotkeys, in-game and class key bindings, pets."""

import tkinter as tk
from tkinter import ttk

from src.gui import theme
from src.gui.interfaces import KeyBindings, Tab, Frame
from src.gui.settings.pets import Pets
from src.common import config


class Settings(Tab):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, 'Keys', **kwargs)
        ttk.Label(self, text='Click a box, then press the key to bind. Save each group.',
                  style='Muted.TLabel').pack(anchor=tk.W, padx=theme.PAD, pady=(10, 0))

        self.column1 = Frame(self)
        self.column1.pack(fill=tk.X, padx=theme.PAD, pady=(8, 0))
        self.controls = KeyBindings(self.column1, 'Auto Maple controls', config.listener)
        self.controls.pack(side=tk.TOP, fill=tk.X)
        self.common_bindings = KeyBindings(self.column1, 'In-game key bindings', config.bot)
        self.common_bindings.pack(side=tk.TOP, fill=tk.X, pady=(10, 0))
        self.pets = Pets(self.column1)
        self.pets.pack(side=tk.TOP, fill=tk.X, pady=(10, 0))

        self.column2 = Frame(self)
        self.column2.pack(fill=tk.X, padx=theme.PAD, pady=(10, theme.PAD))
        self.class_bindings = KeyBindings(self.column2, 'No command book loaded', None)
        self.class_bindings.pack(side=tk.TOP, fill=tk.X)

    def update_class_bindings(self):
        self.class_bindings.destroy()
        class_name = config.bot.command_book.name.capitalize()
        self.class_bindings = KeyBindings(self.column2, f'{class_name} key bindings',
                                          config.bot.command_book)
        self.class_bindings.pack(side=tk.TOP, fill=tk.X)
