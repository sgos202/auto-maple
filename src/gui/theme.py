"""The app's look: a dark, compact panel with the orange-mushroom accent.

One palette and a few building blocks (sections, toggles, a scrollable page
host and a tab strip) so every page is laid out the same way.
"""

import tkinter as tk
from tkinter import ttk

# ---- palette ----
BG = '#14161b'          # window
SURFACE = '#1b1e25'     # cards and pages
RAISED = '#232731'      # inputs, buttons
HOVER = '#2c313d'
BORDER = '#2f3442'
TEXT = '#e7e9ef'
MUTED = '#8a92a4'
FAINT = '#5b6273'
ACCENT = '#ff8a3d'      # the mushroom's orange
ACCENT_HOVER = '#ff9d5c'
ACCENT_TEXT = '#1a1206'
HP = '#e5484d'
MP = '#3e8ef7'
EXP = '#c4dc2c'         # the game's yellow-green EXP fill
OK = '#3fb950'
WARN = '#f5a524'
DANGER = '#e5484d'
TRACK = '#2a2e38'       # empty part of a bar

FONT = ('Segoe UI', 9)
FONT_SMALL = ('Segoe UI', 8)
FONT_BOLD = ('Segoe UI Semibold', 9)
FONT_TITLE = ('Segoe UI Semibold', 10)
FONT_MONO = ('Consolas', 9)
FONT_MONO_SMALL = ('Consolas', 8)

PAD = 10                # page side padding


def apply(root):
    """Styles ttk widgets and sets defaults for the classic tk ones."""
    root.configure(bg=BG)
    # Classic tk widgets (the routine editor, key bindings, dialogs) read these
    for pattern, value in (
            ('*Background', SURFACE), ('*Foreground', TEXT),
            ('*Font', FONT), ('*activeBackground', HOVER),
            ('*activeForeground', TEXT), ('*selectBackground', ACCENT),
            ('*selectForeground', ACCENT_TEXT), ('*highlightThickness', 0),
            ('*insertBackground', TEXT), ('*disabledForeground', FAINT),
            ('*Entry.Background', RAISED), ('*Entry.relief', 'flat'),
            ('*Entry.disabledBackground', RAISED), ('*Entry.readonlyBackground', RAISED),
            ('*Spinbox.Background', RAISED), ('*Spinbox.relief', 'flat'),
            ('*Spinbox.buttonBackground', RAISED), ('*Spinbox.disabledBackground', RAISED),
            ('*Listbox.Background', RAISED), ('*Listbox.relief', 'flat'),
            ('*Text.Background', '#111318'), ('*Text.relief', 'flat'),
            ('*Button.Background', RAISED), ('*Button.relief', 'flat'),
            ('*Button.borderWidth', 0), ('*Button.padX', 10), ('*Button.padY', 4),
            ('*Checkbutton.selectColor', RAISED), ('*Radiobutton.selectColor', RAISED),
            ('*Menubutton.Background', RAISED), ('*Menubutton.relief', 'flat'),
            ('*Menu.Background', RAISED), ('*Menu.relief', 'flat'),
            ('*Menu.activeBackground', ACCENT), ('*Menu.activeForeground', ACCENT_TEXT),
            ('*Scrollbar.Background', RAISED), ('*Scrollbar.troughColor', SURFACE),
            ('*TCombobox*Listbox.Background', RAISED),
            ('*TCombobox*Listbox.Foreground', TEXT),
            ('*TCombobox*Listbox.selectBackground', ACCENT),
            ('*TCombobox*Listbox.selectForeground', ACCENT_TEXT)):
        root.option_add(pattern, value)

    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure('.', background=SURFACE, foreground=TEXT, font=FONT,
                    bordercolor=BORDER, darkcolor=SURFACE, lightcolor=SURFACE,
                    troughcolor=BG, focuscolor=ACCENT, fieldbackground=RAISED,
                    selectbackground=ACCENT, selectforeground=ACCENT_TEXT,
                    insertcolor=TEXT)
    style.map('.', foreground=[('disabled', FAINT)])
    style.configure('TFrame', background=SURFACE)
    style.configure('Bg.TFrame', background=BG)
    style.configure('TLabel', background=SURFACE, foreground=TEXT)
    style.configure('Muted.TLabel', foreground=MUTED, font=FONT_SMALL)
    style.configure('Section.TLabel', foreground=ACCENT, font=FONT_BOLD)
    style.configure('Value.TLabel', foreground=TEXT, font=FONT_BOLD)
    style.configure('TLabelframe', background=SURFACE, bordercolor=BORDER,
                    relief='solid', borderwidth=1)
    style.configure('TLabelframe.Label', background=SURFACE, foreground=MUTED,
                    font=FONT_SMALL)

    style.configure('TButton', background=RAISED, foreground=TEXT, borderwidth=0,
                    focusthickness=0, padding=(10, 4), relief='flat')
    style.map('TButton',
              background=[('disabled', '#1e2129'), ('pressed', BORDER), ('active', HOVER)],
              foreground=[('disabled', FAINT)])
    style.configure('Accent.TButton', background=ACCENT, foreground=ACCENT_TEXT,
                    font=FONT_BOLD)
    style.map('Accent.TButton',
              background=[('disabled', RAISED), ('pressed', '#e5772e'),
                          ('active', ACCENT_HOVER)],
              foreground=[('disabled', FAINT)])
    style.configure('Danger.TButton', background='#3a1d22', foreground='#ff9ea1')
    style.map('Danger.TButton', background=[('active', '#4a242a')])

    _checkbox_images(root, style)
    style.configure('TCheckbutton', background=SURFACE, foreground=TEXT,
                    indicatorbackground=RAISED, indicatorforeground=ACCENT_TEXT,
                    indicatormargin=(0, 0, 6, 0), padding=(0, 2),
                    upperbordercolor=BORDER, lowerbordercolor=BORDER, focusthickness=0)
    style.map('TCheckbutton',
              background=[('active', SURFACE)],
              indicatorbackground=[('selected', ACCENT), ('active', HOVER)],
              foreground=[('disabled', FAINT)])

    for name in ('TEntry', 'TCombobox', 'TSpinbox'):
        style.configure(name, fieldbackground=RAISED, foreground=TEXT,
                        background=RAISED, bordercolor=BORDER, lightcolor=RAISED,
                        darkcolor=RAISED, arrowcolor=MUTED, padding=(5, 2),
                        insertcolor=TEXT)
        style.map(name,
                  fieldbackground=[('readonly', RAISED), ('disabled', SURFACE)],
                  foreground=[('disabled', FAINT)],
                  bordercolor=[('focus', ACCENT)],
                  arrowcolor=[('active', TEXT)])
    style.configure('TScrollbar', background=RAISED, troughcolor=SURFACE,
                    bordercolor=SURFACE, arrowcolor=MUTED, lightcolor=RAISED,
                    darkcolor=RAISED, gripcount=0, arrowsize=10)
    style.map('TScrollbar', background=[('active', HOVER)])
    style.configure('TNotebook', background=SURFACE, borderwidth=0)
    style.configure('TNotebook.Tab', background=RAISED, foreground=MUTED,
                    padding=(10, 4), borderwidth=0)
    style.map('TNotebook.Tab', background=[('selected', SURFACE)],
              foreground=[('selected', TEXT)])
    return style


def _checkbox_images(root, style):
    """Rounded tick boxes (clam's own draws an X) in the accent colour."""
    from PIL import Image, ImageDraw, ImageTk
    scale, box, width = 4, 14, 20         # drawn 4x then shrunk: smooth edges

    def draw(fill, border, tick):
        size = (width * scale, box * scale)
        image = Image.new('RGBA', size, (0, 0, 0, 0))
        pen = ImageDraw.Draw(image)
        pen.rounded_rectangle((scale, scale, (box - 1) * scale, (box - 1) * scale),
                              radius=3 * scale, fill=fill, outline=border, width=scale)
        if tick:
            pen.line([(3.6 * scale, 7.2 * scale), (6.0 * scale, 9.6 * scale),
                      (10.4 * scale, 4.6 * scale)], fill=tick, width=int(1.9 * scale),
                     joint='curve')
        return ImageTk.PhotoImage(image.resize((width, box), Image.Resampling.LANCZOS),
                                  master=root)

    images = {
        'off': draw(RAISED, '#454b5a', None),
        'hover': draw(HOVER, ACCENT, None),
        'on': draw(ACCENT, ACCENT, ACCENT_TEXT),
        'on_disabled': draw('#7a4a2c', '#7a4a2c', '#2a2420'),
        'off_disabled': draw(SURFACE, BORDER, None),
    }
    root._checkbox_images = images        # keep them alive
    style.element_create('Mushroom.indicator', 'image', images['off'],
                         ('disabled', 'selected', images['on_disabled']),
                         ('selected', images['on']),
                         ('disabled', images['off_disabled']),
                         ('active', images['hover']), sticky='w')
    style.layout('TCheckbutton', [('Checkbutton.padding', {'sticky': 'nswe', 'children': [
        ('Mushroom.indicator', {'side': 'left', 'sticky': ''}),
        ('Checkbutton.focus', {'side': 'left', 'sticky': 'w', 'children': [
            ('Checkbutton.label', {'sticky': 'nswe'})]})]})])


# ---- building blocks ----

def section(parent, title, row=None, hint=None):
    """A titled group on a page: an orange heading, optional grey hint, and a
    body frame (returned) to put the controls in."""
    box = ttk.Frame(parent)
    if row is None:
        box.pack(fill=tk.X, padx=PAD, pady=(10, 0))
    else:
        box.grid(row=row, column=0, sticky=tk.EW, padx=PAD, pady=(10, 0))
    head = ttk.Frame(box)
    head.pack(fill=tk.X)
    ttk.Label(head, text=title.upper(), style='Section.TLabel').pack(side=tk.LEFT)
    if hint:
        ttk.Label(head, text=hint, style='Muted.TLabel').pack(side=tk.RIGHT)
    tk.Frame(box, height=1, bg=BORDER).pack(fill=tk.X, pady=(3, 6))
    body = ttk.Frame(box)
    body.pack(fill=tk.X)
    body.columnconfigure(1, weight=1)
    return body


def status_label(parent, variable, **grid):
    """Grey wrapping text for live status lines."""
    label = ttk.Label(parent, textvariable=variable, style='Muted.TLabel',
                      wraplength=360, justify=tk.LEFT)
    label.bind('<Configure>', lambda e: label.configure(wraplength=max(120, e.width - 4)))
    return label


class PageHost(ttk.Frame):
    """
    Shows one page at a time inside a scrolling area. Pages are created with
    this as their parent and register with add(page, text) (src.gui.interfaces.Tab
    does it). A page with FILL = True gets the full height and no scrolling
    (the log, which scrolls itself).
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.canvas = tk.Canvas(self, bg=SURFACE, highlightthickness=0, bd=0)
        self.scroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self._on_scroll)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.window = self.canvas.create_window(0, 0, anchor=tk.NW)
        self.pages = {}
        self.order = []
        self.current = None
        self.listeners = []
        self.canvas.bind('<Configure>', self._fit)
        self.bind_all('<MouseWheel>', self._wheel, add='+')

    def add(self, page, text=''):
        self.pages[text] = page
        self.order.append(text)
        page.bind('<Configure>', lambda e: self._update_region(), add='+')
        if self.current is None:
            self.show(text)

    def show(self, text):
        page = self.pages[text]
        self.current = text
        page.lift(self.canvas)
        self.canvas.itemconfigure(self.window, window=page)
        self.canvas.yview_moveto(0)
        self._fit()
        for listener in self.listeners:
            listener(text)

    def _fill(self):
        page = self.pages.get(self.current)
        return bool(getattr(page, 'FILL', False))

    def _fit(self, _event=None):
        width = self.canvas.winfo_width()
        if width <= 1:
            return
        if self._fill():
            self.canvas.itemconfigure(self.window, width=width,
                                      height=self.canvas.winfo_height())
        else:
            self.canvas.itemconfigure(self.window, width=width, height=0)
        self._update_region()

    def _update_region(self):
        page = self.pages.get(self.current)
        if page is None:
            return
        if self._fill():
            self.canvas.configure(scrollregion=(0, 0, 0, 0))
            self.scroll.pack_forget()
            return
        height = max(page.winfo_reqheight(), 1)
        self.canvas.configure(scrollregion=(0, 0, self.canvas.winfo_width(), height))
        if height > self.canvas.winfo_height():
            if not self.scroll.winfo_ismapped():
                self.scroll.pack(side=tk.RIGHT, fill=tk.Y, before=self.canvas)
        elif self.scroll.winfo_ismapped():
            self.scroll.pack_forget()
            self.canvas.yview_moveto(0)

    def _on_scroll(self, first, last):
        self.scroll.set(first, last)

    def _wheel(self, event):
        if self._fill() or not self.scroll.winfo_ismapped():
            return
        widget = event.widget
        try:
            if widget.winfo_toplevel() is not self.winfo_toplevel():
                return
            if widget.winfo_class() in ('Text', 'Listbox', 'TCombobox'):
                return
            # Only when the pointer is over the page area
            x, y = self.winfo_pointerxy()
            left, top = self.winfo_rootx(), self.winfo_rooty()
            if not (left <= x < left + self.winfo_width()
                    and top <= y < top + self.winfo_height()):
                return
        except (tk.TclError, AttributeError, KeyError):
            return
        self.canvas.yview_scroll(int(-event.delta / 120) * 3, 'units')


class NavBar(tk.Frame):
    """A flat tab strip for a PageHost: text buttons, the open one underlined."""

    def __init__(self, parent, host, **kwargs):
        super().__init__(parent, bg=BG, **kwargs)
        self.host = host
        self.items = {}
        host.listeners.append(self._mark)

    def build(self):
        for column, name in enumerate(self.host.order):
            self.columnconfigure(column, weight=1, uniform='nav')
            cell = tk.Frame(self, bg=BG)
            cell.grid(row=0, column=column, sticky=tk.NSEW)
            label = tk.Label(cell, text=name, bg=BG, fg=MUTED, font=FONT,
                             padx=0, pady=6, cursor='hand2')
            label.pack(fill=tk.X)
            bar = tk.Frame(cell, height=2, bg=BG)
            bar.pack(fill=tk.X)
            for widget in (cell, label):
                widget.bind('<Button-1>', lambda e, n=name: self.host.show(n))
                widget.bind('<Enter>', lambda e, n=name: self._hover(n, True))
                widget.bind('<Leave>', lambda e, n=name: self._hover(n, False))
            self.items[name] = (label, bar)
        self._mark(self.host.current)

    def _hover(self, name, inside):
        label, _ = self.items[name]
        if name != self.host.current:
            label.configure(fg=TEXT if inside else MUTED)

    def _mark(self, current):
        for name, (label, bar) in self.items.items():
            selected = name == current
            label.configure(fg=TEXT if selected else MUTED,
                            font=FONT_BOLD if selected else FONT)
            bar.configure(bg=ACCENT if selected else BG)


class WindowHost(tk.Toplevel):
    """A separate window that holds one Tab (the routine editor). Closing it
    only hides it, so code that updates the page keeps working."""

    def __init__(self, parent, title, geometry, **kwargs):
        super().__init__(parent, bg=SURFACE, **kwargs)
        self.title(title)
        self.geometry(geometry)
        self.withdraw()
        self.protocol('WM_DELETE_WINDOW', self.withdraw)

    def add(self, page, text=''):
        page.pack(fill=tk.BOTH, expand=True)

    def open(self):
        self.deiconify()
        self.lift()
        self.focus_set()
