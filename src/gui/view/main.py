"""Displays the current minimap as well as various information regarding the current routine."""

import tkinter as tk
from src.gui.view.details import Details
from src.gui.view.minimap import Minimap
from src.gui.view.routine import Routine
from src.gui.view.status import Status
from src.gui.view.health import HealthPotion
from src.gui.view.loot import LootPickup
from src.gui.view.mob import MobTracker
from src.gui.interfaces import Tab


class View(Tab):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, 'View', **kwargs)

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(4, weight=1)

        self.minimap = Minimap(self)
        self.minimap.grid(row=0, column=2, sticky=tk.NSEW, padx=(5, 10), pady=5)

        self.status = Status(self)
        self.status.grid(row=0, column=1, sticky=tk.NSEW, padx=5, pady=5)

        self.health_potion = HealthPotion(self)
        self.health_potion.grid(
            row=1, column=0, columnspan=3, sticky=tk.EW, padx=10, pady=(5, 0)
        )

        self.mob_tracker = MobTracker(self)
        self.mob_tracker.grid(
            row=2, column=0, columnspan=3, sticky=tk.EW, padx=10, pady=(5, 0)
        )

        self.loot_pickup = LootPickup(self)
        self.loot_pickup.grid(
            row=3, column=0, columnspan=3, sticky=tk.EW, padx=10, pady=(5, 0)
        )

        self.details = Details(self)
        self.details.grid(
            row=4, column=0, columnspan=3, sticky=tk.NSEW, padx=10, pady=(5, 10)
        )

        self.routine = Routine(self)
        self.routine.grid(row=0, column=0, sticky=tk.N, padx=(10, 5), pady=5)
