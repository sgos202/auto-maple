"""The Supplies page: HP / MP potions read from the screen, and loot pickup."""

import tkinter as tk
from tkinter import messagebox, ttk

from src.common import config, vkeys
from src.gui import theme
from src.gui.interfaces import Tab
from src.modules.health import HP_TRIGGER_PERCENT, MP_TRIGGER_PERCENT


class Supplies(Tab):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, 'Supplies', **kwargs)
        self.health_potion = HealthPotion(self)
        from src.gui.view.loot import LootPickup
        self.loot_pickup = LootPickup(self)
        ttk.Frame(self, height=theme.PAD).pack()


class HealthPotion:
    """Auto HP and MP potions. Builds its sections on PAGE."""

    def __init__(self, page):
        self.page = page
        self.monitor = config.capture.health
        current = self.monitor.settings()
        self.enabled = tk.BooleanVar(value=current['Enabled'])
        self.key = tk.StringVar(value=current['Key'])
        keys = tuple(sorted(vkeys.KEY_MAP))

        body = theme.section(page, 'HP potion', hint=f'at or below {HP_TRIGGER_PERCENT}%')
        ttk.Checkbutton(body, text='Drink automatically', variable=self.enabled,
                        command=self._apply_settings).grid(row=0, column=0, sticky=tk.W)
        key_row = ttk.Frame(body)
        key_row.grid(row=0, column=1, sticky=tk.E)
        ttk.Label(key_row, text='Key', style='Muted.TLabel').pack(side=tk.LEFT, padx=(0, 6))
        self.key_box = ttk.Combobox(key_row, textvariable=self.key, values=keys, width=7,
                                    state='readonly')
        self.key_box.pack(side=tk.LEFT)
        self.key_box.bind('<<ComboboxSelected>>', self._apply_settings)
        self.hp_status = tk.StringVar()
        theme.status_label(body, self.hp_status).grid(row=1, column=0, columnspan=2,
                                                      sticky=tk.EW, pady=(4, 0))

        # MP: same idea, its own key
        self.mana = getattr(config.capture, 'mana', None)
        mana = self.mana.settings() if self.mana else {'Enabled': False, 'Key': 't'}
        self.mp_enabled = tk.BooleanVar(value=mana['Enabled'])
        self.mp_key = tk.StringVar(value=mana['Key'])
        body = theme.section(page, 'MP potion',
                             hint=f'at or below {mana.get("Threshold", MP_TRIGGER_PERCENT)}%')
        ttk.Checkbutton(body, text='Drink automatically', variable=self.mp_enabled,
                        command=self._apply_mp_settings).grid(row=0, column=0, sticky=tk.W)
        key_row = ttk.Frame(body)
        key_row.grid(row=0, column=1, sticky=tk.E)
        ttk.Label(key_row, text='Key', style='Muted.TLabel').pack(side=tk.LEFT, padx=(0, 6))
        self.mp_key_box = ttk.Combobox(key_row, textvariable=self.mp_key, values=keys,
                                       width=7, state='readonly')
        self.mp_key_box.pack(side=tk.LEFT)
        self.mp_key_box.bind('<<ComboboxSelected>>', self._apply_mp_settings)
        self.mp_status = tk.StringVar()
        theme.status_label(body, self.mp_status).grid(row=1, column=0, columnspan=2,
                                                      sticky=tk.EW, pady=(4, 0))
        self._refresh_status()

    def winfo_toplevel(self):
        return self.page.winfo_toplevel()

    def _apply_mp_settings(self, _event=None):
        if self.mana is None:
            return
        current = self.mana.settings()
        if config.enabled:
            messagebox.showwarning(
                'Pause Auto Maple first',
                'Pause Auto Maple before changing MP potion settings.',
                parent=self.winfo_toplevel())
            self.mp_enabled.set(current['Enabled'])
            self.mp_key.set(current['Key'])
            return
        try:
            self.mana.update_settings(self.mp_enabled.get(), self.mp_key.get())
        except ValueError as error:
            messagebox.showerror('Invalid MP potion settings', str(error),
                                 parent=self.winfo_toplevel())
            self.mp_enabled.set(current['Enabled'])
            self.mp_key.set(current['Key'])

    def _apply_settings(self, _event=None):
        if config.enabled:
            messagebox.showwarning(
                'Pause Auto Maple first',
                'Pause Auto Maple before changing HP potion settings.',
                parent=self.winfo_toplevel())
            current = self.monitor.settings()
            self.enabled.set(current['Enabled'])
            self.key.set(current['Key'])
            return
        try:
            self.monitor.update_settings(self.enabled.get(), HP_TRIGGER_PERCENT,
                                         self.key.get())
        except ValueError as error:
            messagebox.showerror('Invalid HP potion settings', str(error),
                                 parent=self.winfo_toplevel())
            current = self.monitor.settings()
            self.enabled.set(current['Enabled'])
            self.key.set(current['Key'])

    def _refresh_status(self):
        percent = self.monitor.current_percent()
        action = self.monitor.last_action()
        if percent is None:
            text = 'HP bar not in view.'
        else:
            text = f'HP {percent:.0f}%' + (f'  ·  {action}' if action else '')
        self.hp_status.set(text)
        if self.mana is not None:
            mp = self.mana.current_percent()
            mp_action = self.mana.last_action()
            if mp is None:
                text = 'MP bar not in view.'
            else:
                text = f'MP {mp:.0f}%' + (f'  ·  {mp_action}' if mp_action else '')
            self.mp_status.set(text)
        self.page.after(300, self._refresh_status)
