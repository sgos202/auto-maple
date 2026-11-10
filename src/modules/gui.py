"""User friendly GUI to interact with Auto Maple."""

import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from src.common import config, settings
from src.gui import Menu, View, Edit, Settings, Log


class GUI:
    DISPLAY_FRAME_RATE = 30
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

    def start(self):
        """Starts the GUI as well as any scheduled functions."""

        self.root.after(100, self._check_minimap_selection)
        self.root.after(100, self._poll_log)

        display_thread = threading.Thread(target=self._display_minimap)
        display_thread.daemon = True
        display_thread.start()

        layout_thread = threading.Thread(target=self._save_layout)
        layout_thread.daemon = True
        layout_thread.start()

        self.root.mainloop()

    def _poll_log(self):
        self.log.poll()
        self.root.after(100, self._poll_log)

    def _check_minimap_selection(self):
        capture = config.capture
        if capture.selection_requested.is_set():
            capture.selection_requested.clear()
            self._show_screen_region_selection(
                capture.selection_frame,
                'Select minimap area',
                'Drag around the minimap map area, including the player marker but not the title bar; then press Enter.',
                capture.complete_minimap_selection
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

        def select_enemy(player_bounds):
            if player_bounds is None:
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
                print(f'\n[~] Saved enemy sample {count}.')

            self._show_screen_region_selection(
                frame,
                'Select another enemy',
                'Drag tightly around one complete enemy sprite, excluding nearby drops and background, then press Enter.',
                save_enemy,
                min_size=(12, 12)
            )

        self._show_screen_region_selection(
            frame,
            'Select player name badge',
            'Select the current player name badge, then press Enter.',
            select_enemy,
            min_size=(20, 6)
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

    def _show_screen_region_selection(
            self, frame, title, instructions, on_selected, min_size=(10, 10),
            multiple=False):
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.transient(self.root)
        dialog.grab_set()

        image = Image.fromarray(frame[:, :, :3][:, :, ::-1].copy())
        max_width, max_height = 1050, 600
        scale = min(1, max_width / image.width, max_height / image.height)
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
                    max(0, min(limit, round(value / scale)))
                    for value, limit in zip(
                        bounds,
                        (
                            image.width, image.height,
                            image.width, image.height
                        )
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

    def _display_minimap(self):
        delay = 1 / GUI.DISPLAY_FRAME_RATE
        while True:
            self.view.minimap.display_minimap()
            time.sleep(delay)

    def _save_layout(self):
        """Periodically saves the current Layout object."""

        while True:
            if config.layout is not None and settings.record_layout:
                config.layout.save()
            time.sleep(5)


if __name__ == '__main__':
    gui = GUI()
    gui.start()
