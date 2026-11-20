"""The Routine page: the loaded routine, the selected step's details, and the
file actions (command book, routines, the editor window)."""

import tkinter as tk
from tkinter import ttk

from src.common import config, utils
from src.gui import theme
from src.gui.interfaces import Tab


class Routine(Tab):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, 'Routine', **kwargs)
        from src.gui.menu.file import File

        body = theme.section(self, 'Files')
        grid = ttk.Frame(body)
        grid.grid(row=0, column=0, columnspan=2, sticky=tk.EW)
        grid.columnconfigure((0, 1), weight=1, uniform='f')
        self.cb_button = ttk.Button(grid, text='Load command book',
                                    command=utils.async_callback(self, File._load_commands))
        self.cb_button.grid(row=0, column=0, columnspan=2, sticky=tk.EW)
        # Disabled until a command book is loaded, like the old File menu
        self.routine_buttons = []
        for index, (text, action) in enumerate((('Load routine', File._load_routine),
                                                ('New routine', File._new_routine),
                                                ('Save routine', File._save_routine))):
            button = ttk.Button(grid, text=text, state=tk.DISABLED,
                                command=utils.async_callback(self, action))
            button.grid(row=1 + index // 2, column=index % 2, sticky=tk.EW,
                        padx=(0, 3) if index % 2 == 0 else (3, 0), pady=(6, 0))
            self.routine_buttons.append(button)
        self.editor_button = ttk.Button(grid, text='Open routine editor',
                                        command=lambda: config.gui.open_editor())
        self.editor_button.keep_enabled = True
        self.editor_button.grid(row=2, column=1, sticky=tk.EW, padx=(3, 0), pady=(6, 0))

        body = theme.section(self, 'Steps', hint='select one for details')
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0)
        self.listbox = tk.Listbox(body, height=8, listvariable=config.gui.routine_var,
                                  exportselection=False, activestyle='none',
                                  bg=theme.RAISED, fg=theme.TEXT, bd=0,
                                  selectbackground=theme.ACCENT,
                                  selectforeground=theme.ACCENT_TEXT,
                                  font=theme.FONT_MONO)
        self.listbox.keep_enabled = True
        for key in ('<Up>', '<Down>', '<Left>', '<Right>'):
            self.listbox.bind(key, lambda e: 'break')
        self.scroll = ttk.Scrollbar(body, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=self.scroll.set)
        self.listbox.grid(row=0, column=0, sticky=tk.EW)
        self.scroll.grid(row=0, column=1, sticky=tk.NS)

        self.details = Details(self)
        self.listbox.bind('<<ListboxSelect>>', self.details.show_details)
        ttk.Frame(self, height=theme.PAD).pack()

    def enable_routine_buttons(self):
        for button in self.routine_buttons:
            button.configure(state=tk.NORMAL)

    def select(self, i):
        self.listbox.selection_clear(0, 'end')
        self.listbox.selection_set(i)
        self.listbox.see(i)


class Details:
    """The selected step's name and settings."""

    def __init__(self, page):
        self.page = page
        self.name_var = tk.StringVar()
        body = theme.section(page, 'Details')
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0)
        ttk.Label(body, textvariable=self.name_var, style='Value.TLabel').grid(
            row=0, column=0, sticky=tk.W)
        self.text = tk.Text(body, height=6, width=1, state=tk.DISABLED, wrap=tk.WORD,
                            bg=theme.RAISED, fg=theme.TEXT, bd=0, padx=6, pady=4,
                            font=theme.FONT_MONO_SMALL)
        self.text.grid(row=1, column=0, sticky=tk.EW, pady=(4, 0))

    def show_details(self, e):
        """Callback for updating the Details section everytime Listbox selection changes."""
        selections = e.widget.curselection()
        if len(selections) > 0:
            self.display_info(int(selections[0]))

    def update_details(self):
        """Updates Details to show info about the current selection."""
        selects = self.page.listbox.curselection()
        if len(selects) > 0:
            self.display_info(int(selects[0]))
        else:
            self.clear_info()

    def display_info(self, index):
        """Shows info about the Component at position INDEX."""
        info = config.routine[index].info()
        self.name_var.set(info['name'])
        self.text.config(state=tk.NORMAL)
        self.text.delete(1.0, 'end')
        self.text.insert(1.0, '\n'.join(f'{key}: {value}'
                                        for key, value in info['vars'].items()))
        self.text.config(state=tk.DISABLED)

    def clear_info(self):
        self.name_var.set('')
        self.text.config(state=tk.NORMAL)
        self.text.delete(1.0, 'end')
        self.text.config(state=tk.DISABLED)
