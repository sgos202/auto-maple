"""The live minimap: the game's own, scaled up pixel-sharp, with the routine,
path, rune and character drawn on it, FPS and the character's position on top."""

import cv2
import tkinter as tk
from PIL import ImageTk, Image
from src.common import config, utils
from src.routine.components import Point
from src.gui import theme


class Minimap(tk.Frame):
    HEIGHT = 120            # the most the minimap may take; width follows the panel

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg='#0e1014', height=self.HEIGHT,
                         highlightthickness=1, highlightbackground=theme.BORDER, **kwargs)
        self.parent = parent
        self.pack_propagate(False)
        self.canvas = tk.Canvas(self, bg='#0e1014', borderwidth=0, highlightthickness=0)
        self.canvas.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.canvas.keep_enabled = True
        self.fps_text = tk.Label(self, text='', bg='#0e1014', fg=theme.OK,
                                 font=theme.FONT_MONO_SMALL)
        self.fps_text.place(x=4, y=2)
        self.pos_text = tk.Label(self, text='', bg='#0e1014', fg=theme.WARN,
                                 font=theme.FONT_MONO_SMALL)
        self.pos_text.place(relx=1.0, x=-4, y=2, anchor=tk.NE)
        self.empty_text = tk.Label(self, text='Waiting for the minimap…', bg='#0e1014',
                                   fg=theme.FAINT, font=theme.FONT_SMALL)
        self.empty_text.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.container = None
        self._canvas_size = None
        self._labels = (None, None)

    def _fit(self, width, height):
        """The largest size with the minimap's aspect ratio that fits the card."""
        avail_w = self.winfo_width() - 4
        avail_h = self.winfo_height() - 4
        if avail_w < 20 or avail_h < 20:            # not laid out yet
            avail_w, avail_h = 380, self.HEIGHT
        ratio = min(avail_w / width, avail_h / height)
        return max(1, int(width * ratio)), max(1, int(height * ratio))

    def _set_labels(self, minimap, player_pos):
        fps = getattr(config.capture, 'fps', 0) or 0
        fps_text = f'FPS {fps:.1f}' if fps else ''
        pos_text = ''
        if player_pos is not None and minimap is not None:
            x, y = utils.convert_to_absolute(player_pos, minimap)
            pos_text = f'({x}, {y})'
        if (fps_text, pos_text) != self._labels:
            self._labels = (fps_text, pos_text)
            self.fps_text.configure(text=fps_text)
            self.pos_text.configure(text=pos_text)

    def display_minimap(self):
        """Redraws the minimap (called on the Tk thread by the GUI)."""

        minimap = config.capture.minimap
        if not minimap:
            return
        rune_active = minimap['rune_active']
        rune_pos = minimap['rune_pos']
        path = minimap['path']
        player_pos = minimap['player_pos']
        raw = minimap['minimap']
        if raw is None or raw.size == 0:
            return

        img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape
        if width < 1 or height < 1:
            return
        if self.empty_text.winfo_ismapped():
            self.empty_text.place_forget()
        self._set_labels(raw, player_pos)

        # Pixel-sharp when enlarged (the "pixel map" look), smooth when shrunk
        new_width, new_height = self._fit(width, height)
        interpolation = cv2.INTER_AREA if new_width < width else cv2.INTER_NEAREST
        img = cv2.resize(img, (new_width, new_height), interpolation=interpolation)
        if self._canvas_size != (new_width, new_height):
            self.canvas.configure(width=new_width, height=new_height)
            self._canvas_size = (new_width, new_height)

        if rune_active:
            cv2.circle(img, utils.convert_to_absolute(rune_pos, img), 4, (180, 90, 255), -1)

        if config.enabled and len(path) > 1:
            for i in range(len(path) - 1):
                start = utils.convert_to_absolute(path[i], img)
                end = utils.convert_to_absolute(path[i + 1], img)
                cv2.line(img, start, end, (255, 210, 60), 1)

        for p in config.routine.sequence:
            if isinstance(p, Point):
                utils.draw_location(img, p.location,
                                    (63, 185, 80) if config.enabled else (229, 72, 77))

        if config.layout:
            config.layout.draw(img)

        # The character on top: an orange dot with a dark rim
        if player_pos is not None:
            centre = utils.convert_to_absolute(player_pos, img)
            cv2.circle(img, centre, 5, (20, 22, 27), -1)
            cv2.circle(img, centre, 3, (255, 138, 61), -1)

        img = ImageTk.PhotoImage(Image.fromarray(img))
        if self.container is None:
            self.container = self.canvas.create_image(0, 0, image=img, anchor=tk.NW)
        else:
            self.canvas.itemconfig(self.container, image=img)
        self._img = img                 # Prevent garbage collection
