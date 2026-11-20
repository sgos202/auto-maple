"""The always-visible top of the panel: HP / MP / EXP bars and the enemy radar."""

import tkinter as tk

from src.common import config
from src.gui import theme


def _duration(seconds):
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes = rest // 60
    if hours >= 100:
        return f'{hours}h'
    if hours:
        return f'{hours}h {minutes:02d}m'
    return f'{minutes}m'


class Bars(tk.Canvas):
    """HP, MP and EXP read from the game's bottom bar, like the game shows them."""

    BAR_HEIGHT = 18
    GAP = 6
    ROWS = (('HP', theme.HP), ('MP', theme.MP), ('EXP', theme.EXP))
    REFRESH_MS = 250

    def __init__(self, parent, **kwargs):
        height = len(self.ROWS) * (self.BAR_HEIGHT + self.GAP) - self.GAP + 16
        super().__init__(parent, height=height, bg=theme.BG, highlightthickness=0,
                         bd=0, **kwargs)
        self.keep_enabled = True
        self._values = {}
        self.bind('<Configure>', lambda e: self._draw())
        self.after(self.REFRESH_MS, self._refresh)

    def _read(self):
        capture = config.capture
        values = {}
        for name, attr in (('HP', 'health'), ('MP', 'mana'), ('EXP', 'exp')):
            monitor = getattr(capture, attr, None)
            try:
                values[name] = monitor.current_percent() if monitor else None
            except Exception:
                values[name] = None
        values['rate'] = self._rate_text(getattr(capture, 'exp', None))
        return values

    @staticmethod
    def _rate_text(exp):
        if exp is None:
            return ''
        try:
            rate = exp.rate_per_hour()
            eta = exp.seconds_to_level()
        except Exception:
            return ''
        if exp.current_percent() is None:
            return 'EXP bar not in view'
        if rate is None:
            return 'EXP/hour: measuring…'
        parts = [f'+{rate:.2f}% / hour']
        if eta is not None and rate > 0:
            parts.append(f'level in {_duration(eta)}')
        return '  ·  '.join(parts)

    def _refresh(self):
        values = self._read()
        if values != self._values:
            self._values = values
            self._draw()
        self.after(self.REFRESH_MS, self._refresh)

    def _draw(self):
        self.delete('all')
        width = self.winfo_width()
        if width < 40:
            return
        label_w = 30
        for index, (name, colour) in enumerate(self.ROWS):
            top = index * (self.BAR_HEIGHT + self.GAP)
            bottom = top + self.BAR_HEIGHT
            left, right = label_w, width - 1
            self.create_text(0, (top + bottom) / 2, text=name, anchor=tk.W,
                             fill=theme.MUTED, font=theme.FONT_BOLD)
            self.create_rectangle(left, top, right, bottom, fill=theme.TRACK, width=0)
            value = self._values.get(name)
            if value is not None:
                fill_right = left + (right - left) * max(0.0, min(100.0, value)) / 100
                if fill_right > left:
                    self.create_rectangle(left, top, fill_right, bottom, fill=colour,
                                          width=0)
                    # a lighter top half, like the game's glossy bars
                    self.create_rectangle(left, top, fill_right, top + self.BAR_HEIGHT // 3,
                                          fill=_lighten(colour), width=0)
                text = f'{value:.2f}%' if name == 'EXP' else f'{value:.0f}%'
            else:
                text = '—'
            self.create_text((left + right) / 2 + 1, (top + bottom) / 2 + 1, text=text,
                             fill='#000000', font=theme.FONT_BOLD)
            self.create_text((left + right) / 2, (top + bottom) / 2, text=text,
                             fill='#ffffff', font=theme.FONT_BOLD)
        rate = self._values.get('rate', '')
        bottom = len(self.ROWS) * (self.BAR_HEIGHT + self.GAP) - self.GAP
        self.create_text(label_w, bottom + 8, text=rate, anchor=tk.W,
                         fill=theme.MUTED, font=theme.FONT_SMALL)


def _lighten(colour, amount=0.25):
    r, g, b = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
    r, g, b = (int(c + (255 - c) * amount) for c in (r, g, b))
    return f'#{r:02x}{g:02x}{b:02x}'


class Radar(tk.Canvas):
    """
    A pixel map of the game screen, always on: the character (orange) and
    every enemy the tracker sees (red; dark red when hurt), the locked target
    ringed in yellow. One cell is CELL px; the game frame is shrunk to fit.
    """

    CELLS = (40, 30)        # 4:3 like the game window
    CELL = 3
    REFRESH_MS = 120
    GRID = '#1f232b'

    def __init__(self, parent, **kwargs):
        w, h = self.CELLS[0] * self.CELL, self.CELLS[1] * self.CELL
        super().__init__(parent, width=w + 2, height=h + 2, bg='#0e1014',
                         highlightthickness=1, highlightbackground=theme.BORDER,
                         bd=0, **kwargs)
        self.keep_enabled = True
        self._last = None
        self._draw_grid()
        self.after(self.REFRESH_MS, self._refresh)

    def _draw_grid(self):
        cols, rows = self.CELLS
        c = self.CELL
        for x in range(0, cols, 4):
            for y in range(0, rows, 4):
                self.create_rectangle(1 + x * c, 1 + y * c, 1 + x * c + 1, 1 + y * c + 1,
                                      fill=self.GRID, width=0, tags='grid')

    def _cell(self, x, y, size):
        cols, rows = self.CELLS
        cx = max(0, min(cols - 1, int(x * cols / size[0])))
        cy = max(0, min(rows - 1, int(y * rows / size[1])))
        return cx, cy

    def _block(self, cx, cy, colour, span=1, tag='live'):
        c = self.CELL
        x0, y0 = 1 + cx * c, 1 + cy * c
        self.create_rectangle(x0, y0, x0 + c * span, y0 + c * span, fill=colour,
                              width=0, tags=tag)

    def _refresh(self):
        tracker = getattr(config.capture, 'mob_tracker', None)
        try:
            radar = tracker.radar() if tracker is not None else None
        except Exception:
            radar = None
        key = None if radar is None else (radar['player'], tuple(radar['enemies']),
                                          radar['target'])
        if key != self._last:
            self._last = key
            self._draw(radar)
        self.after(self.REFRESH_MS, self._refresh)

    def _draw(self, radar):
        self.delete('live')
        cols, rows = self.CELLS
        c = self.CELL
        if radar is None:
            self.create_text(1 + cols * c / 2, 1 + rows * c / 2, text='no scan',
                             fill=theme.FAINT, font=theme.FONT_SMALL, tags='live')
            return
        size = radar['size']
        px, py = self._cell(*radar['player'], size)
        # the ground the character stands on
        ground = min(rows - 1, py + 2)
        for x in range(cols):
            self._block(x, ground, '#262b35')
        target = radar['target']
        tcell = self._cell(*target, size) if target else None
        for x, y, health in radar['enemies']:
            cx, cy = self._cell(x, y, size)
            hurt = health is not None and health < 0.999
            self._block(cx - 1 if cx else cx, cy, '#8c1d22' if hurt else theme.HP, span=2)
        if tcell is not None:
            tx, ty = tcell
            x0, y0 = 1 + (tx - 2) * c, 1 + (ty - 1) * c
            self.create_rectangle(x0, y0, x0 + 4 * c, y0 + 4 * c, outline=theme.WARN,
                                  width=1, tags='live')
        self._block(px - 1 if px else px, py, theme.ACCENT, span=2)
        count = len(radar['enemies'])
        self.create_text(cols * c - 2, 3, text=f'{count}', anchor=tk.NE,
                         fill=theme.HP if count else theme.FAINT,
                         font=theme.FONT_MONO_SMALL, tags='live')
