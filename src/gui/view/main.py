"""Creates the panel's main pages and keeps the names the rest of the bot uses
(config.gui.view.status, .routine, .details, .minimap, .mob_tracker)."""

from src.gui.view.buffs import Buffs
from src.gui.view.health import Supplies
from src.gui.view.mob import MobTracker
from src.gui.view.routine import Routine
from src.gui.view.status import Status


class View:
    def __init__(self, host, minimap):
        self.minimap = minimap
        self.mob_tracker = MobTracker(host)         # Combat
        self.supplies = Supplies(host)              # HP / MP potions, loot pickup
        self.health_potion = self.supplies.health_potion
        self.loot_pickup = self.supplies.loot_pickup
        self.buffs = Buffs(host)                    # timed buffs
        self.status = Status(host)                  # Game
        self.routine = Routine(host)
        self.details = self.routine.details
