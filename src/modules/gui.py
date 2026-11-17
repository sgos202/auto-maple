"""User friendly GUI to interact with Auto Maple."""

import os
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from src.common import config, settings
from src.gui import Menu, View, Edit, Settings, Log


class GUI:
    DISPLAY_FRAME_RATE = 30
    EDITABLE_WIDGET_CLASSES = {
        'Button', 'Canvas', 'Checkbutton', 'Combobox', 'Entry', 'Listbox',
        'Menubutton', 'Radiobutton', 'Scale', 'Spinbox', 'TButton',
        'TCheckbutton', 'TCombobox', 'TEntry', 'TMenubutton', 'TRadiobutton',
        'TScale', 'TSpinbox', 'Text'
    }
    RESOLUTIONS = {
        'DEFAULT': '800x800',
        'Edit': '1400x800'
    }

    def __init__(self):
        config.gui = self

        self.root = tk.Tk()
        self.root.title('Auto Maple')
        icon = tk.PhotoImage(file='assets/icon.png')
        self.root.iconphoto(False, icon)
        self.root.geometry(GUI.RESOLUTIONS['DEFAULT'])
        self.root.resizable(False, False)

        # Initialize GUI variables
        self.routine_var = tk.StringVar()

        # Build the GUI
        self.menu = Menu(self.root)
        self.root.config(menu=self.menu)

        self.navigation = ttk.Notebook(self.root)

        self.view = View(self.navigation)
        self.edit = Edit(self.navigation)
        self.settings = Settings(self.navigation)
        self.log = Log(self.navigation)

        self.navigation.pack(expand=True, fill='both')
        self.navigation.bind('<<NotebookTabChanged>>', self._resize_window)
        self._control_states = {}
        self._menu_states = []
        self._controls_locked = None
        self._alarm_win = None          # the flashing mini-game button, while it shows
        self._alarm_button = None
        self._alarm_flash = False
        self._alarm_click_through = None
        self.root.after(100, self._sync_control_states)
        self.root.focus()

    def set_routine(self, arr):
        self.routine_var.set(arr)

    def clear_routine_info(self):
        """
        Clears information in various GUI elements regarding the current routine.
        Does not clear Listboxes containing routine Components, as that is handled by Routine.
        """

        self.view.details.clear_info()
        self.view.status.set_routine('')

        self.edit.minimap.redraw()
        self.edit.routine.commands.clear_contents()
        self.edit.routine.commands.update_display()
        self.edit.editor.reset()

    def _resize_window(self, e):
        """Callback to resize entire Tkinter window every time a new Page is selected."""

        nav = e.widget
        curr_id = nav.select()
        nav.nametowidget(curr_id).focus()      # Focus the current Tab
        page = nav.tab(curr_id, 'text')
        if self.root.state() != 'zoomed':
            if page in GUI.RESOLUTIONS:
                self.root.geometry(GUI.RESOLUTIONS[page])
            else:
                self.root.geometry(GUI.RESOLUTIONS['DEFAULT'])

    # Loaded on start-up so there is no setup step after a restart. Load
    # another one from File > Load Command Book whenever needed.
    DEFAULT_COMMAND_BOOK = 'begginer.py'

    def _preload_command_book(self):
        if config.bot is None or config.bot.command_book is not None:
            return
        path = os.path.join(config.RESOURCES_DIR, 'command_books', GUI.DEFAULT_COMMAND_BOOK)
        if not os.path.isfile(path):
            print(f'\n[!] Default command book not found: {path}')
            return
        try:
            config.bot.load_commands(path)
        except Exception as error:
            print(f'\n[!] Could not preload command book {path}: {error}')

    def start(self):
        """Starts the GUI as well as any scheduled functions."""

        self.root.after(50, self._preload_command_book)
        self.root.after(100, self._check_minimap_selection)
        self.root.after(100, self._poll_log)
        self.root.after(250, self._poll_alarm)

        # Tk is not thread-safe: drawing from a worker thread can freeze the
        # whole window at random. The minimap is redrawn on the Tk thread.
        self.root.after(100, self._display_minimap)

        layout_thread = threading.Thread(target=self._save_layout)
        layout_thread.daemon = True
        layout_thread.start()

        self.root.mainloop()

    def _poll_log(self):
        self.log.poll()
        self.root.after(100, self._poll_log)

    def _sync_control_states(self):
        """Lock editing controls while Auto Maple is running."""

        locked = bool(config.enabled)
        if locked != self._controls_locked:
            if locked:
                self._control_states.clear()
                for widget in self._iter_widgets(self.root):
                    if widget.winfo_class() not in GUI.EDITABLE_WIDGET_CLASSES:
                        continue
                    if widget is self._alarm_button or getattr(widget, 'keep_enabled', False):
                        continue        # stays usable while running (alarm, stay-on-channel)
                    state = widget.cget('state')
                    if state != tk.DISABLED:
                        self._control_states[widget] = state
                        widget.configure(state=tk.DISABLED)

                self._menu_states = []
                for index in range(self.menu.index('end') + 1):
                    state = self.menu.entrycget(index, 'state')
                    self._menu_states.append(state)
                    self.menu.entryconfigure(index, state=tk.DISABLED)
            else:
                for widget, state in self._control_states.items():
                    if widget.winfo_exists():
                        widget.configure(state=state)
                self._control_states.clear()

                for index, state in enumerate(self._menu_states):
                    self.menu.entryconfigure(index, state=state)
                self._menu_states = []

            self._controls_locked = locked

        self.root.after(200, self._sync_control_states)

    #################################
    #   Mini-game alarm (screen)    #
    #################################
    ALARM_SIZE = (520, 170)
    ALARM_COLOURS = (('#d00000', 'white'), ('#ffd400', 'black'))
    ALARM_FLASH_MS = 400
    ALARM_GAME_TITLE = 'MapleStory'

    def _poll_alarm(self):
        """Shows, flashes and removes the mini-game alarm button."""

        from src.modules.mini_game_alarm import ALARM
        try:
            if ALARM.active:
                if self._alarm_win is None or not self._alarm_win.winfo_exists():
                    self._show_alarm()
                self._update_alarm(ALARM.name)
            elif self._alarm_win is not None:
                self._hide_alarm()
        except Exception as error:
            print(f'\n[!] Mini-game alarm button failed: {error}')
            self._alarm_win = self._alarm_button = None
        self.root.after(GUI.ALARM_FLASH_MS, self._poll_alarm)

    def _show_alarm(self):
        from src.modules.mini_game_alarm import ALARM
        win = tk.Toplevel(self.root)
        win.overrideredirect(True)                  # no title bar
        win.attributes('-topmost', True)
        win.attributes('-alpha', 0.95)              # makes it a layered window (click-through)
        w, h = GUI.ALARM_SIZE

        def clicked():
            ALARM.dismiss()
            self._hide_alarm()

        button = tk.Button(win, font=('Segoe UI', 17, 'bold'), relief=tk.FLAT, bd=0,
                           wraplength=w - 30, command=clicked, cursor='hand2')
        button.pack(fill='both', expand=True)
        self._alarm_win, self._alarm_button = win, button
        self._alarm_click_through = None
        self._alarm_geometry = None
        self._alarm_overlaps = False
        self._place_alarm()
        win.update_idletasks()
        self._set_alarm_style(click_through=False)

    def _place_alarm(self):
        """
        As near the middle of the main screen as possible WITHOUT covering the
        game: MapleStory draws its own mouse cursor, so anything on top of the game
        hides the cursor (2026-11-15: the user couldn't see it to take over).
        """
        win = self._alarm_win
        w, h = GUI.ALARM_SIZE
        x, y, overlaps = alarm_spot(win.winfo_screenwidth(), win.winfo_screenheight(),
                                    w, h, _game_rect(GUI.ALARM_GAME_TITLE))
        geometry = f'{w}x{h}+{x}+{y}'
        if geometry != self._alarm_geometry:
            win.geometry(geometry)
            self._alarm_geometry = geometry
        self._alarm_overlaps = overlaps

    def _set_alarm_style(self, click_through):
        """
        The button never takes focus from the game, is left out of screenshots (the
        solvers capture the screen), and - only if it has to sit on top of the game
        - lets clicks through while a solver is using the mouse there.
        """
        if click_through == self._alarm_click_through:
            return
        self._alarm_click_through = click_through
        try:
            import ctypes
            user32 = ctypes.windll.user32
            hwnd = int(self._alarm_win.wm_frame(), 16)
            GWL_EXSTYLE = -20
            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            style |= 0x08000000 | 0x00000080 | 0x00080000    # noactivate, toolwindow, layered
            if click_through:
                style |= 0x00000020                          # transparent to the mouse
            else:
                style &= ~0x00000020
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
            user32.SetWindowDisplayAffinity(hwnd, 0x11)      # WDA_EXCLUDEFROMCAPTURE
        except Exception as error:
            print(f'\n[!] Could not set up the alarm window: {error}')

    def _update_alarm(self, name):
        from src.modules import mini_games
        self._place_alarm()                 # follow the game window if it moves
        runner = mini_games.ACTIVE_RUNNER
        solving = runner is not None and runner.busy
        blocked = solving and self._alarm_overlaps
        self._set_alarm_style(click_through=blocked)
        self._alarm_flash = not self._alarm_flash
        bg, fg = GUI.ALARM_COLOURS[self._alarm_flash]
        game = name.split(' (')[0] if name else 'Mini-game'
        if blocked:
            text = f'⚠ {game} ⚠\nBeing solved automatically...\n(click here to silence once it ends)'
        elif solving:
            text = f'⚠ {game} ⚠\nBeing solved automatically\nCLICK HERE to stop the alarm'
        else:
            text = f'⚠ {game} ⚠\nCLICK HERE to stop the alarm'
        self._alarm_button.configure(text=text, bg=bg, fg=fg, activebackground=bg,
                                     activeforeground=fg, state=tk.NORMAL)

    def _hide_alarm(self):
        if self._alarm_win is not None:
            try:
                self._alarm_win.destroy()
            except tk.TclError:
                pass
        self._alarm_win = self._alarm_button = None

    @staticmethod
    def _iter_widgets(parent):
        for widget in parent.winfo_children():
            yield widget
            yield from GUI._iter_widgets(widget)

    def _check_minimap_selection(self):
        capture = config.capture
        if capture.selection_requested.is_set():
            capture.selection_requested.clear()
            self._show_screen_region_selection(
                capture.selection_frame,
                'Select minimap area',
                'Drag around the minimap map area, including the player marker but not the title bar; then press Enter.',
                capture.complete_minimap_selection,
                region=GUI.MINIMAP_SELECTION_REGION
            )
        self.root.after(100, self._check_minimap_selection)

    def select_mob_templates(self):
        if config.enabled:
            messagebox.showwarning(
                'Pause routine first',
                'Pause Auto Maple before calibrating mob tracking.',
                parent=self.root
            )
            return
        frame = config.capture.frame
        if frame is None:
            messagebox.showwarning(
                'Screen not ready',
                'Wait for the game screen capture to start, then try again.',
                parent=self.root
            )
            return
        frame = frame.copy()

        def save_mob_template(player_bounds):
            if player_bounds is None:
                return

            def save_samples(mob_bounds_list):
                if not mob_bounds_list:
                    return
                try:
                    config.capture.mob_tracker.set_templates(
                        frame, player_bounds, mob_bounds_list
                    )
                except (OSError, ValueError) as error:
                    messagebox.showerror(
                        'Could not save mob samples',
                        str(error),
                        parent=self.root
                    )
                    return
                self.view.mob_tracker.refresh_samples()
                print(
                    f'\n[~] Saved player badge and '
                    f'{len(mob_bounds_list)} enemy sample(s) from one screen.'
                )

            self._show_screen_region_selection(
                frame,
                'Select enemy samples',
                'Drag a tight box around each complete enemy sprite. Select as many as you want, then choose Finish.',
                save_samples,
                min_size=(12, 12),
                multiple=True
            )

        self._show_screen_region_selection(
            frame,
            'Select player name badge',
            'Select the player name badge only, then press Enter.',
            save_mob_template,
            min_size=(20, 6)
        )

    def add_mob_template(self):
        if config.enabled:
            messagebox.showwarning(
                'Pause routine first',
                'Pause Auto Maple before adding an enemy sample.',
                parent=self.root
            )
            return
        tracker = config.capture.mob_tracker
        if not tracker.has_player_sample():
            messagebox.showwarning(
                'Calibrate player badge first',
                'Use “Calibrate player + enemy” before adding more enemy samples.',
                parent=self.root
            )
            return
        frame = config.capture.frame
        if frame is None:
            messagebox.showwarning(
                'Screen not ready',
                'Wait for the game screen capture to start, then try again.',
                parent=self.root
            )
            return
        frame = frame.copy()
        player_bounds = tracker.find_player_bounds(frame)
        if player_bounds is None:
            messagebox.showwarning(
                'Player badge not found',
                'Keep the saved player name badge visible, or calibrate the badge again.',
                parent=self.root
            )
            return

        def save_enemy(mob_bounds):
            if mob_bounds is None:
                return
            try:
                count = tracker.add_enemy_template(
                    frame, player_bounds, mob_bounds
                )
            except (OSError, ValueError) as error:
                messagebox.showerror(
                    'Could not add enemy sample',
                    str(error),
                    parent=self.root
                )
                return
            self.view.mob_tracker.refresh_samples()
            print(f'\n[~] Saved enemy sample {count}.')

        self._show_screen_region_selection(
            frame,
            'Select another enemy',
            'Drag tightly around one complete enemy sprite, excluding nearby drops and background, then press Enter.',
            save_enemy,
            min_size=(12, 12)
        )

    def select_loot_templates(self):
        if config.enabled:
            messagebox.showwarning(
                'Pause Auto Maple first',
                'Pause Auto Maple before calibrating loot pickup.',
                parent=self.root
            )
            return
        mob_tracker = config.capture.mob_tracker
        if not mob_tracker.has_player_sample():
            messagebox.showwarning(
                'Calibrate player badge first',
                'Calibrate the player badge with “Calibrate player + enemies” first.',
                parent=self.root
            )
            return
        frame = config.capture.frame
        if frame is None:
            messagebox.showwarning(
                'Screen not ready',
                'Wait for the game screen capture to start, then try again.',
                parent=self.root
            )
            return
        frame = frame.copy()
        player_bounds = mob_tracker.find_player_bounds(frame)
        if player_bounds is None:
            messagebox.showwarning(
                'Player badge not found',
                'Keep the player badge visible and recalibrate it if needed.',
                parent=self.root
            )
            return

        def save_loot_templates(loot_bounds):
            if not loot_bounds:
                return
            try:
                count = config.capture.loot_tracker.set_templates(
                    frame, player_bounds, loot_bounds
                )
            except (OSError, ValueError) as error:
                messagebox.showerror(
                    'Could not save loot samples',
                    str(error),
                    parent=self.root
                )
                return
            print(f'\n[~] Saved {count} loot sample(s).')

        self._show_screen_region_selection(
            frame,
            'Select loot samples',
            'Box one whole loot icon tightly; exclude ground, nearby items, and effects. Sample items on this platform, then choose Finish.',
            save_loot_templates,
            min_size=(8, 8),
            multiple=True
        )

    def add_loot_templates(self):
        if config.enabled:
            messagebox.showwarning(
                'Pause Auto Maple first',
                'Pause Auto Maple before adding loot samples.',
                parent=self.root
            )
            return
        tracker = config.capture.mob_tracker
        if not tracker.has_player_sample():
            messagebox.showwarning(
                'Calibrate player badge first',
                'Calibrate the player badge with “Calibrate player + enemies” first.',
                parent=self.root
            )
            return
        frame = config.capture.frame
        if frame is None:
            messagebox.showwarning(
                'Screen not ready',
                'Wait for the game screen capture to start, then try again.',
                parent=self.root
            )
            return
        frame = frame.copy()
        player_bounds = tracker.find_player_bounds(frame)
        if player_bounds is None:
            messagebox.showwarning(
                'Player badge not found',
                'Keep the player badge visible and recalibrate it if needed.',
                parent=self.root
            )
            return

        def save_samples(bounds):
            if not bounds:
                return
            try:
                count = config.capture.loot_tracker.set_templates(
                    frame, player_bounds, bounds, append=True
                )
            except (OSError, ValueError) as error:
                messagebox.showerror(
                    'Could not add loot samples',
                    str(error),
                    parent=self.root
                )
                return
            print(f'\n[~] Saved {count} loot sample(s).')

        self._show_screen_region_selection(
            frame,
            'Add map-specific loot sample',
            'Optional: box each unusual loot icon tightly, excluding ground, nearby items, and effects. Built-in samples remain active.',
            save_samples,
            min_size=(8, 8),
            multiple=True
        )

    # The part of the game screen shown (zoomed in) when selecting the minimap,
    # as fractions of the screen: MapleStory keeps it in the top-left corner.
    # Wide enough for long, flat maps; the dialog can still show the whole screen.
    MINIMAP_SELECTION_REGION = (0.0, 0.0, 0.55, 0.45)
    SELECTION_MAX_ZOOM = 3.0

    def _show_screen_region_selection(
            self, frame, title, instructions, on_selected, min_size=(10, 10),
            multiple=False, region=None, zoomed=True):
        """REGION (x0, y0, x1, y1 as fractions of FRAME) shows only that part,
        enlarged to fill the dialog, with a button to switch to the whole screen.
        Selections are always returned in full-frame coordinates."""
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.transient(self.root)
        dialog.grab_set()

        frame_h, frame_w = frame.shape[:2]
        offset_x, offset_y = 0, 0
        view = frame
        if region is not None and zoomed:
            x0 = max(0, min(frame_w - 1, int(region[0] * frame_w)))
            y0 = max(0, min(frame_h - 1, int(region[1] * frame_h)))
            x1 = max(x0 + 1, min(frame_w, int(round(region[2] * frame_w))))
            y1 = max(y0 + 1, min(frame_h, int(round(region[3] * frame_h))))
            view = frame[y0:y1, x0:x1]
            offset_x, offset_y = x0, y0

        image = Image.fromarray(view[:, :, :3][:, :, ::-1].copy())
        # Fit the dialog to the monitor it opens on (big screens get a big view).
        max_width = max(400, min(1600, self.root.winfo_screenwidth() - 120))
        max_height = max(300, min(1000, self.root.winfo_screenheight() - 220))
        max_zoom = GUI.SELECTION_MAX_ZOOM if view is not frame else 1
        scale = min(max_zoom, max_width / image.width, max_height / image.height)
        display_size = (round(image.width * scale), round(image.height * scale))
        display_image = ImageTk.PhotoImage(image.resize(display_size, Image.Resampling.LANCZOS))

        instruction_label = tk.Label(
            dialog,
            text=instructions
        )
        instruction_label.pack(padx=8, pady=6)

        canvas = tk.Canvas(
            dialog,
            width=display_size[0],
            height=display_size[1],
            cursor='crosshair',
            highlightthickness=0
        )
        canvas.pack(padx=8, pady=4)
        canvas.create_image(0, 0, image=display_image, anchor=tk.NW)
        canvas.image = display_image

        selection = {
            'start': None,
            'rectangle': None,
            'bounds': [],
            'rectangles': []
        }
        selection_status = tk.StringVar(value='')
        if multiple:
            tk.Label(dialog, textvariable=selection_status).pack()

        def start_selection(event):
            selection['start'] = (event.x, event.y)
            if not multiple:
                for rectangle in selection['rectangles']:
                    canvas.delete(rectangle)
                selection['bounds'].clear()
                selection['rectangles'].clear()
            selection['rectangle'] = canvas.create_rectangle(
                event.x, event.y, event.x, event.y, outline='red', width=2
            )

        def update_selection(event):
            if selection['start'] is not None:
                canvas.coords(
                    selection['rectangle'],
                    *selection['start'],
                    event.x,
                    event.y
                )

        def finish_selection(event):
            if selection['start'] is not None:
                x1, y1 = selection['start']
                bounds = (
                    min(x1, event.x),
                    min(y1, event.y),
                    max(x1, event.x),
                    max(y1, event.y)
                )
                image_bounds = tuple(
                    max(0, min(limit, round(value / scale))) + offset
                    for value, limit, offset in zip(
                        bounds,
                        (
                            image.width, image.height,
                            image.width, image.height
                        ),
                        (offset_x, offset_y, offset_x, offset_y)
                    )
                )
                if (image_bounds[2] - image_bounds[0] >= min_size[0]
                        and image_bounds[3] - image_bounds[1] >= min_size[1]):
                    selection['bounds'].append(image_bounds)
                    selection['rectangles'].append(selection['rectangle'])
                    if multiple:
                        selection_status.set(
                            f'{len(selection["bounds"])} sample(s) selected.'
                        )
                else:
                    canvas.delete(selection['rectangle'])
                    selection_status.set(
                        f'Selection is too small; minimum is '
                        f'{min_size[0]} x {min_size[1]} pixels.'
                    )
                selection['rectangle'] = None
                selection['start'] = None

        def close_with_selection(_event=None):
            if multiple:
                if not selection['bounds']:
                    selection_status.set('Select at least one sample.')
                    return
                result = [
                    ((bounds[0], bounds[1]), (bounds[2], bounds[3]))
                    for bounds in selection['bounds']
                ]
            else:
                if not selection['bounds']:
                    on_selected(None)
                    dialog.destroy()
                    return
                bounds = selection['bounds'][-1]
                result = ((bounds[0], bounds[1]), (bounds[2], bounds[3]))
            on_selected(result)
            dialog.destroy()

        def cancel_selection():
            on_selected(None)
            dialog.destroy()

        canvas.bind('<ButtonPress-1>', start_selection)
        canvas.bind('<B1-Motion>', update_selection)
        canvas.bind('<ButtonRelease-1>', finish_selection)
        dialog.bind('<Return>', close_with_selection)
        dialog.bind('<Escape>', lambda _event: cancel_selection())
        dialog.protocol('WM_DELETE_WINDOW', cancel_selection)
        buttons = ttk.Frame(dialog)
        buttons.pack(pady=(4, 8))
        ttk.Button(
            buttons,
            text='Finish' if multiple else 'Use selection',
            command=close_with_selection
        ).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(buttons, text='Cancel', command=cancel_selection).pack(
            side=tk.LEFT, padx=4
        )
        if region is not None:
            def toggle_zoom():
                dialog.destroy()
                self._show_screen_region_selection(
                    frame, title, instructions, on_selected, min_size=min_size,
                    multiple=multiple, region=region, zoomed=not zoomed
                )

            ttk.Button(
                buttons,
                text='Show whole screen' if zoomed else 'Zoom to top-left',
                command=toggle_zoom
            ).pack(side=tk.LEFT, padx=4)

    def _display_minimap(self):
        try:
            self.view.minimap.display_minimap()
        except Exception as error:
            now = time.monotonic()
            if now - getattr(self, '_display_error_at', 0) > 30:
                self._display_error_at = now
                print(f'\n[!] Minimap display failed: {error}')
        self.root.after(max(1, int(1000 / GUI.DISPLAY_FRAME_RATE)), self._display_minimap)

    def _save_layout(self):
        """Periodically saves the current Layout object."""

        while True:
            if config.layout is not None and settings.record_layout:
                config.layout.save()
            time.sleep(5)


if __name__ == '__main__':
    gui = GUI()
    gui.start()


def _game_rect(title):
    """The game window's (left, top, right, bottom) on screen, or None."""
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, title)
        if not hwnd or user32.IsIconic(hwnd):
            return None
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None
        return rect.left, rect.top, rect.right, rect.bottom
    except Exception:
        return None


def alarm_spot(screen_w, screen_h, w, h, game, margin=12):
    """
    Top-left (x, y) for a W x H box as close to the screen centre as possible
    without overlapping GAME (left, top, right, bottom), and whether it still
    overlaps (no room anywhere: then it goes to the top edge).
    """
    cx, cy = (screen_w - w) / 2, (screen_h - h) / 2

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    if game is None:
        return int(cx), int(cy), False
    left, top, right, bottom = game
    left, top, right, bottom = left - margin, top - margin, right + margin, bottom + margin

    def overlaps(x, y):
        return x < right and x + w > left and y < bottom and y + h > top

    if not overlaps(cx, cy):
        return int(cx), int(cy), False
    spots = []
    x_any, y_any = clamp(cx, 0, screen_w - w), clamp(cy, 0, screen_h - h)
    if bottom <= screen_h - h:
        spots.append((x_any, clamp(cy, bottom, screen_h - h)))     # below the game
    if top - h >= 0:
        spots.append((x_any, clamp(cy, 0, top - h)))                # above
    if left - w >= 0:
        spots.append((clamp(cx, 0, left - w), y_any))               # left of it
    if right <= screen_w - w:
        spots.append((clamp(cx, right, screen_w - w), y_any))       # right of it
    spots = [s for s in spots if not overlaps(*s)]
    if not spots:
        return int(x_any), 0, True
    x, y = min(spots, key=lambda s: (s[0] - cx) ** 2 + (s[1] - cy) ** 2)
    return int(x), int(y), False
