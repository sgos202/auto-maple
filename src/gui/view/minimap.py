import cv2
import tkinter as tk
from PIL import ImageTk, Image
from src.gui.interfaces import LabelFrame
from src.common import config, utils
from src.routine.components import Point


class Minimap(LabelFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, 'Minimap', **kwargs)

        # Upper bounds only: the canvas is resized to the minimap's own aspect
        # ratio every frame, so flat, tall or square maps fill it without bars.
        self.WIDTH = 270
        self.HEIGHT = 150
        self.area = tk.Frame(self, width=self.WIDTH, height=self.HEIGHT)
        self.area.pack(expand=True, fill='both', padx=5, pady=5)
        self.area.pack_propagate(False)
        self.canvas = tk.Canvas(self.area, width=self.WIDTH, height=self.HEIGHT,
                                borderwidth=0, highlightthickness=0)
        self.canvas.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.container = None
        self._canvas_size = None

    def _fit(self, width, height):
        """The largest size with the minimap's aspect ratio that fits the panel."""
        avail_w = self.area.winfo_width()
        avail_h = self.area.winfo_height()
        if avail_w < 20 or avail_h < 20:            # not laid out yet
            avail_w, avail_h = self.WIDTH, self.HEIGHT
        ratio = min(avail_w / width, avail_h / height)
        return max(1, int(width * ratio)), max(1, int(height * ratio))

    def display_minimap(self):
        """Updates the Main page with the current minimap."""

        minimap = config.capture.minimap
        if minimap:
            rune_active = minimap['rune_active']
            rune_pos = minimap['rune_pos']
            path = minimap['path']
            player_pos = minimap['player_pos']

            img = cv2.cvtColor(minimap['minimap'], cv2.COLOR_BGR2RGB)
            height, width, _ = img.shape

            if width < 1 or height < 1:
                return

            # Scale the minimap to fill the panel while keeping its shape, and
            # size the canvas to match so there are no bars around it.
            new_width, new_height = self._fit(width, height)
            interpolation = cv2.INTER_AREA if new_width < width else cv2.INTER_LINEAR
            img = cv2.resize(img, (new_width, new_height), interpolation=interpolation)
            if self._canvas_size != (new_width, new_height):
                self.canvas.configure(width=new_width, height=new_height)
                if self.container is not None:
                    self.canvas.coords(self.container, 0, 0)
                self._canvas_size = (new_width, new_height)

            # Mark the position of the active rune
            if rune_active:
                cv2.circle(img,
                           utils.convert_to_absolute(rune_pos, img),
                           3,
                           (128, 0, 128),
                           -1)

            # Draw the current path that the program is taking
            if config.enabled and len(path) > 1:
                for i in range(len(path) - 1):
                    start = utils.convert_to_absolute(path[i], img)
                    end = utils.convert_to_absolute(path[i + 1], img)
                    cv2.line(img, start, end, (0, 255, 255), 1)

            # Draw each Point in the routine as a circle
            for p in config.routine.sequence:
                if isinstance(p, Point):
                    utils.draw_location(img,
                                        p.location,
                                        (0, 255, 0) if config.enabled else (255, 0, 0))

            # Display the current Layout
            if config.layout:
                config.layout.draw(img)

            # Draw the player's position on top of everything
            cv2.circle(img,
                       utils.convert_to_absolute(player_pos, img),
                       3,
                       (0, 0, 255),
                       -1)

            # Display the minimap in the Canvas
            img = ImageTk.PhotoImage(Image.fromarray(img))
            if self.container is None:
                self.container = self.canvas.create_image(0, 0, image=img, anchor=tk.NW)
            else:
                self.canvas.itemconfig(self.container, image=img)
            self._img = img                 # Prevent garbage collection
