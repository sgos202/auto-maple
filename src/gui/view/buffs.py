"""The Buffs page: named buffs with a key and a recast time, cast while running."""

import tkinter as tk
from tkinter import messagebox, ttk

from src.common import config, vkeys
from src.gui import theme
from src.gui.interfaces import Tab
from src.modules.buffs import BUFFS, MAX_PRESSES, format_duration

UNITS = {'sec': 1, 'min': 60}


class Buffs(Tab):
    REFRESH_MS = 500

    def __init__(self, parent, **kwargs):
        super().__init__(parent, 'Buffs', **kwargs)
        self.editing = None             # id of the buff in the form, or None for a new one
        self.rows = {}                  # id -> (countdown var, countdown label)

        body = theme.section(self, 'Buffs', hint='cast while Auto Maple runs')
        self.master_on = tk.BooleanVar(value=BUFFS.enabled)
        master = ttk.Checkbutton(body, text='Cast buffs automatically', variable=self.master_on,
                                 command=lambda: BUFFS.set_enabled(self.master_on.get()))
        master.keep_enabled = True
        master.grid(row=0, column=0, columnspan=2, sticky=tk.W)
        self.list_frame = ttk.Frame(body)
        self.list_frame.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(6, 0))
        self.list_frame.columnconfigure(0, weight=1)

        # ---- add / edit form ----
        self.form_body = theme.section(self, 'Add a buff')
        form = self.form_body
        self.name = tk.StringVar()
        self.key = tk.StringVar()
        self.amount = tk.StringVar(value='180')
        self.unit = tk.StringVar(value='sec')
        self.presses = tk.StringVar(value='1')

        ttk.Label(form, text='Name').grid(row=0, column=0, sticky=tk.W)
        name_entry = ttk.Entry(form, textvariable=self.name)
        name_entry.grid(row=0, column=1, sticky=tk.EW, pady=2)
        ttk.Label(form, text='Key').grid(row=1, column=0, sticky=tk.W)
        key_box = ttk.Combobox(form, textvariable=self.key, values=tuple(sorted(vkeys.KEY_MAP)),
                               width=10)
        key_box.grid(row=1, column=1, sticky=tk.W, pady=2)
        ttk.Label(form, text='Cast again every').grid(row=2, column=0, sticky=tk.W, padx=(0, 10))
        every = ttk.Frame(form)
        every.grid(row=2, column=1, sticky=tk.W, pady=2)
        amount_box = ttk.Spinbox(every, from_=1, to=3600, increment=5, width=7,
                                 textvariable=self.amount)
        amount_box.pack(side=tk.LEFT)
        unit_box = ttk.Combobox(every, textvariable=self.unit, values=tuple(UNITS), width=5,
                                state='readonly')
        unit_box.pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(form, text='Presses').grid(row=3, column=0, sticky=tk.W)
        presses_box = ttk.Spinbox(form, from_=1, to=MAX_PRESSES, width=4,
                                  textvariable=self.presses)
        presses_box.grid(row=3, column=1, sticky=tk.W, pady=2)
        ttk.Label(form, text='Set the recast a few seconds shorter than the buff lasts, '
                             'so it never runs out.',
                  style='Muted.TLabel', wraplength=340).grid(row=4, column=0, columnspan=2,
                                                            sticky=tk.W, pady=(4, 0))
        buttons = ttk.Frame(form)
        buttons.grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=(8, 0))
        buttons.columnconfigure((0, 1), weight=1, uniform='b')
        self.save_button = ttk.Button(buttons, text='Add buff', style='Accent.TButton',
                                      command=self._save)
        self.save_button.grid(row=0, column=0, columnspan=2, sticky=tk.EW)
        self.cancel_button = ttk.Button(buttons, text='Cancel', command=self._clear_form)
        for widget in (name_entry, key_box, amount_box, unit_box, presses_box,
                       self.save_button, self.cancel_button):
            widget.keep_enabled = True      # safe to change while running
        name_entry.bind('<Return>', lambda e: self._save())
        ttk.Frame(self, height=theme.PAD).pack()

        self._rebuild()
        self.after(self.REFRESH_MS, self._tick)

    # ---- list ----
    def _rebuild(self):
        for widget in self.list_frame.winfo_children():
            widget.destroy()
        self.rows.clear()
        buffs = BUFFS.list()
        if not buffs:
            ttk.Label(self.list_frame, text='No buffs yet. Add one below.',
                      style='Muted.TLabel').grid(row=0, column=0, sticky=tk.W, pady=4)
            return
        for index, buff in enumerate(buffs):
            self._row(index, buff)

    def _row(self, index, buff):
        card = tk.Frame(self.list_frame, bg=theme.RAISED, padx=8, pady=6)
        card.grid(row=index, column=0, sticky=tk.EW, pady=(0, 6))
        card.columnconfigure(1, weight=1)
        on = tk.BooleanVar(value=buff['enabled'])
        check = tk.Checkbutton(card, variable=on, bg=theme.RAISED, activebackground=theme.RAISED,
                               command=lambda: self._toggle(buff['id'], on.get()))
        check.keep_enabled = True
        check.grid(row=0, column=0, rowspan=2, sticky=tk.W)
        tk.Label(card, text=buff['name'], bg=theme.RAISED, fg=theme.TEXT,
                 font=theme.FONT_BOLD, anchor=tk.W).grid(row=0, column=1, sticky=tk.W)
        presses = f"  ×{buff['presses']}" if buff['presses'] > 1 else ''
        tk.Label(card, text=f"{buff['key'].upper()}{presses}  ·  every "
                            f"{format_duration(buff['interval'])}",
                 bg=theme.RAISED, fg=theme.MUTED, font=theme.FONT_SMALL,
                 anchor=tk.W).grid(row=1, column=1, sticky=tk.W)
        countdown = tk.StringVar()
        label = tk.Label(card, textvariable=countdown, bg=theme.RAISED, fg=theme.OK,
                         font=theme.FONT_MONO)
        label.grid(row=0, column=2, rowspan=2, padx=(6, 8))
        actions = tk.Frame(card, bg=theme.RAISED)
        actions.grid(row=0, column=3, rowspan=2)
        for text, tip, command in (('✎', 'Edit', lambda: self._edit(buff)),
                                   ('▶', 'Cast at the next chance',
                                    lambda: self._recast(buff['id'])),
                                   ('✕', 'Delete', lambda: self._delete(buff))):
            button = tk.Label(actions, text=text, bg=theme.RAISED, fg=theme.MUTED,
                              font=theme.FONT_TITLE, padx=4, cursor='hand2')
            button.pack(side=tk.LEFT)
            button.bind('<Button-1>', lambda e, c=command: c())
            button.bind('<Enter>', lambda e, b=button: b.configure(fg=theme.ACCENT))
            button.bind('<Leave>', lambda e, b=button: b.configure(fg=theme.MUTED))
        self.rows[buff['id']] = (countdown, label, buff['enabled'])

    def _tick(self):
        master = BUFFS.enabled
        for buff_id, (countdown, label, enabled) in self.rows.items():
            left = BUFFS.seconds_left(buff_id)
            if left is None:
                continue
            if not enabled or not master:
                text, colour = 'off', theme.FAINT
            elif left <= 0:
                text = 'due' if config.enabled else 'on start'
                colour = theme.WARN
            else:
                text, colour = format_duration(left), theme.OK
            if countdown.get() != text:
                countdown.set(text)
                label.configure(fg=colour)
        self.after(self.REFRESH_MS, self._tick)

    # ---- actions ----
    def _toggle(self, buff_id, enabled):
        BUFFS.update(buff_id, enabled=enabled)
        self._rebuild()

    def _recast(self, buff_id):
        BUFFS.recast_soon(buff_id)

    def _delete(self, buff):
        if messagebox.askyesno('Delete buff', f"Delete {buff['name']}?",
                               parent=self.winfo_toplevel()):
            BUFFS.remove(buff['id'])
            if self.editing == buff['id']:
                self._clear_form()
            self._rebuild()

    def _edit(self, buff):
        self.editing = buff['id']
        self.name.set(buff['name'])
        self.key.set(buff['key'])
        if buff['interval'] % 60 == 0 and buff['interval'] >= 60:
            self.amount.set(str(buff['interval'] // 60))
            self.unit.set('min')
        else:
            self.amount.set(str(buff['interval']))
            self.unit.set('sec')
        self.presses.set(str(buff['presses']))
        self.save_button.configure(text='Save changes')
        self.save_button.grid_configure(column=0, columnspan=1, padx=(0, 3))
        self.cancel_button.grid(row=0, column=1, sticky=tk.EW, padx=(3, 0))

    def _clear_form(self):
        self.editing = None
        self.name.set('')
        self.key.set('')
        self.amount.set('180')
        self.unit.set('sec')
        self.presses.set('1')
        self.save_button.configure(text='Add buff')
        self.save_button.grid_configure(column=0, columnspan=2, padx=0)
        self.cancel_button.grid_remove()

    def _save(self):
        try:
            interval = float(self.amount.get()) * UNITS.get(self.unit.get(), 1)
        except ValueError:
            messagebox.showerror('Buff', 'Recast time must be a number.',
                                 parent=self.winfo_toplevel())
            return
        try:
            if self.editing is None:
                BUFFS.add(self.name.get(), self.key.get(), interval, self.presses.get())
            else:
                BUFFS.update(self.editing, name=self.name.get(), key=self.key.get(),
                             interval=interval, presses=self.presses.get())
        except (ValueError, KeyError) as error:
            messagebox.showerror('Buff', str(error), parent=self.winfo_toplevel())
            return
        self._clear_form()
        self._rebuild()
