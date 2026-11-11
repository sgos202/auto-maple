"""An interpreter that reads and executes user-created routines."""

import threading
import time
import git
import cv2
import inspect
import importlib
import traceback
from os.path import splitext, basename
from src.common import config, utils
from src.detection import detection
from src.routine import components
from src.routine.routine import Routine
from src.command_book.command_book import CommandBook
from src.routine.components import Point
from src.common.vkeys import press, click, key_down, key_up
from src.common.interfaces import Configurable
from src.modules.health import is_game_focused


# The rune's buff icon
RUNE_BUFF_TEMPLATE = cv2.imread('assets/rune_buff_template.jpg', 0)


class Bot(Configurable):
    """A class that interprets and executes user-defined routines."""

    PATROL_TURN_INTERVAL = 2.5
    DEFAULT_CONFIG = {
        'Interact': 'y',
        'Feed pet': '9'
    }

    def __init__(self):
        """Loads a user-defined routine on start up and initializes this Bot's main thread."""

        super().__init__('keybindings')
        config.bot = self

        self.rune_active = False
        self.rune_pos = (0, 0)
        self.rune_closest_pos = (0, 0)      # Location of the Point closest to rune
        self.submodules = []
        self.command_book = None            # CommandBook instance
        # self.module_name = None
        # self.buff = components.Buff()

        # self.command_book = {}
        # for c in (components.Wait, components.Walk, components.Fall,
        #           components.Move, components.Adjust, components.Buff):
        #     self.command_book[c.__name__.lower()] = c

        config.routine = Routine()

        self.ready = False
        self._auto_attack_movement = None
        self._patrol_direction = None
        self._patrol_until = 0
        self.thread = threading.Thread(target=self._main)
        self.thread.daemon = True

    def start(self):
        """
        Starts this Bot object's thread.
        :return:    None
        """

        self.update_submodules()
        print('\n[~] Started main bot loop')
        self.thread.start()

    def _main(self):
        """
        The main body of Bot that executes the user's routine.
        :return:    None
        """

        print('\n[~] Initializing detection algorithm:\n')
        try:
            model = detection.load_model()
        except FileNotFoundError as error:
            model = None
            print(f'[!] Rune solver disabled: {error}')
            print('[!] The routine will pause when a rune is detected.')
        else:
            print('\n[~] Initialized detection algorithm')

        self.ready = True
        config.listener.enabled = True
        last_fed = time.time()
        while True:
            if config.enabled and len(config.routine) == 0:
                self._run_standalone_mob_attack()
            elif config.enabled and len(config.routine) > 0:
                # Buff and feed pets
                self.command_book.buff.main()
                pet_settings = config.gui.settings.pets
                auto_feed = pet_settings.auto_feed.get()
                num_pets = pet_settings.num_pets.get()
                now = time.time()
                if auto_feed and now - last_fed > 1200 / num_pets:
                    press(self.config['Feed pet'], 1)
                    last_fed = now

                self._service_loot_pickup()
                if self._service_mob_attack():
                    continue

                # Highlight the current Point
                config.gui.view.routine.select(config.routine.index)
                config.gui.view.details.display_info(config.routine.index)

                # Execute next Point in the routine
                element = config.routine[config.routine.index]
                if self.rune_active and isinstance(element, Point) \
                        and element.location == self.rune_closest_pos:
                    if model is None:
                        config.enabled = False
                        print(
                            '\n[!] Rune detected. Handle it manually, then '
                            'resume the routine with the Start/stop key.'
                        )
                        continue
                    self._solve_rune(model)
                element.execute()
                config.routine.step()
            else:
                self._set_auto_attack_movement(None)
                time.sleep(0.01)

    def _run_standalone_mob_attack(self):
        tracker = config.capture.mob_tracker
        if not tracker.settings()['Enabled']:
            self._set_auto_attack_movement(None)
            time.sleep(0.05)
            return
        if not is_game_focused():
            self._set_auto_attack_movement(None)
            time.sleep(0.05)
            return

        self._service_loot_pickup()
        target = tracker.target()
        if target is not None:
            offset_x, attackable = target
            if attackable:
                self._set_auto_attack_movement(None)
                self._service_mob_attack()
            else:
                self._set_auto_attack_movement(
                    'left' if offset_x < 0 else 'right'
                )
        else:
            self._service_mob_attack()
            self._patrol_for_mob()
        time.sleep(0.02)

    def _patrol_for_mob(self):
        now = time.monotonic()
        if self._patrol_direction is None:
            direction = 'right'
        elif now >= self._patrol_until:
            direction = (
                'left' if self._patrol_direction == 'right'
                else 'right'
            )
        else:
            direction = self._patrol_direction

        if direction != self._patrol_direction:
            self._patrol_until = now + self.PATROL_TURN_INTERVAL
        self._patrol_direction = direction
        self._set_auto_attack_movement(direction)

    def _set_auto_attack_movement(self, direction):
        if direction == self._auto_attack_movement:
            return
        if self._auto_attack_movement is not None:
            key_up(self._auto_attack_movement)
        self._auto_attack_movement = direction
        if direction is not None and config.enabled:
            key_down(direction)

    def _service_loot_pickup(self):
        key = config.capture.loot_tracker.consume_pickup()
        if key is None:
            return False
        if not is_game_focused():
            print('\n[!] Loot detected, but MapleStory is not focused; pickup cancelled.')
            return False
        print(
            f'\n[~] Loot is under the player; sending six quick '
            f'{key.upper()} pickup taps.'
        )
        press(key, 6, down_time=0.04, up_time=0.05)
        return True

    def _service_mob_attack(self, movement_direction=None):
        direction, group_attack, group_count = (
            config.capture.mob_tracker.consume_attack_action()
        )
        if direction is None:
            return False
        if (not group_attack and (
                self.command_book is None
                or 'cruelstab' not in self.command_book.dict)):
            print(
                '\n[!] Nearby enemy detected, but the loaded command book '
                'does not provide CruelStab.'
            )
            return False

        if not is_game_focused():
            print('\n[!] Enemy detected, but MapleStory is not focused; attack cancelled.')
            return False

        if movement_direction:
            key_up(movement_direction)
        if group_attack:
            print(
                f'\n[~] Group of {group_count} nearby enemies detected on the '
                f'{direction}; turning and using Q.'
            )
            try:
                key_down(direction)
                time.sleep(0.06)
            finally:
                key_up(direction)
            press('q', 1, down_time=0.08, up_time=0.1)
        else:
            print(
                f'\n[~] Nearby enemy detected on the {direction}; executing CruelStab.'
            )
            self.command_book['cruelstab'](
                direction,
                attacks=2,
                repetitions=1
            ).execute()
        return True

    @utils.run_if_enabled
    def _solve_rune(self, model):
        """
        Moves to the position of the rune and solves the arrow-key puzzle.
        :param model:   The TensorFlow model to classify with.
        :param sct:     The mss instance object with which to take screenshots.
        :return:        None
        """

        move = self.command_book['move']
        move(*self.rune_pos).execute()
        adjust = self.command_book['adjust']
        adjust(*self.rune_pos).execute()
        time.sleep(0.2)
        press(self.config['Interact'], 1, down_time=0.2)        # Inherited from Configurable

        print('\nSolving rune:')
        inferences = []
        for _ in range(15):
            frame = config.capture.frame
            solution = detection.merge_detection(model, frame)
            if solution:
                print(', '.join(solution))
                if solution in inferences:
                    print('Solution found, entering result')
                    for arrow in solution:
                        press(arrow, 1, down_time=0.1)
                    time.sleep(1)
                    for _ in range(3):
                        time.sleep(0.3)
                        frame = config.capture.frame
                        rune_buff = utils.multi_match(frame[:frame.shape[0] // 8, :],
                                                      RUNE_BUFF_TEMPLATE,
                                                      threshold=0.9)
                        if rune_buff:
                            rune_buff_pos = min(rune_buff, key=lambda p: p[0])
                            target = (
                                round(rune_buff_pos[0] + config.capture.window['left']),
                                round(rune_buff_pos[1] + config.capture.window['top'])
                            )
                            click(target, button='right')
                    self.rune_active = False
                    break
                elif len(solution) == 4:
                    inferences.append(solution)

    def load_commands(self, file):
        try:
            self.command_book = CommandBook(file)
            config.gui.settings.update_class_bindings()
        except ValueError:
            pass    # TODO: UI warning popup, say check cmd for errors
        #
        # utils.print_separator()
        # print(f"[~] Loading command book '{basename(file)}':")
        #
        # ext = splitext(file)[1]
        # if ext != '.py':
        #     print(f" !  '{ext}' is not a supported file extension.")
        #     return False
        #
        # new_step = components.step
        # new_cb = {}
        # for c in (components.Wait, components.Walk, components.Fall):
        #     new_cb[c.__name__.lower()] = c
        #
        # # Import the desired command book file
        # module_name = splitext(basename(file))[0]
        # target = '.'.join(['resources', 'command_books', module_name])
        # try:
        #     module = importlib.import_module(target)
        #     module = importlib.reload(module)
        # except ImportError:     # Display errors in the target Command Book
        #     print(' !  Errors during compilation:\n')
        #     for line in traceback.format_exc().split('\n'):
        #         line = line.rstrip()
        #         if line:
        #             print(' ' * 4 + line)
        #     print(f"\n !  Command book '{module_name}' was not loaded")
        #     return
        #
        # # Check if the 'step' function has been implemented
        # step_found = False
        # for name, func in inspect.getmembers(module, inspect.isfunction):
        #     if name.lower() == 'step':
        #         step_found = True
        #         new_step = func
        #
        # # Populate the new command book
        # for name, command in inspect.getmembers(module, inspect.isclass):
        #     new_cb[name.lower()] = command
        #
        # # Check if required commands have been implemented and overridden
        # required_found = True
        # for command in [components.Buff]:
        #     name = command.__name__.lower()
        #     if name not in new_cb:
        #         required_found = False
        #         new_cb[name] = command
        #         print(f" !  Error: Must implement required command '{name}'.")
        #
        # # Look for overridden movement commands
        # movement_found = True
        # for command in (components.Move, components.Adjust):
        #     name = command.__name__.lower()
        #     if name not in new_cb:
        #         movement_found = False
        #         new_cb[name] = command
        #
        # if not step_found and not movement_found:
        #     print(f" !  Error: Must either implement both 'Move' and 'Adjust' commands, "
        #           f"or the function 'step'")
        # if required_found and (step_found or movement_found):
        #     self.module_name = module_name
        #     self.command_book = new_cb
        #     self.buff = new_cb['buff']()
        #     components.step = new_step
        #     config.gui.menu.file.enable_routine_state()
        #     config.gui.view.status.set_cb(basename(file))
        #     config.routine.clear()
        #     print(f" ~  Successfully loaded command book '{module_name}'")
        # else:
        #     print(f" !  Command book '{module_name}' was not loaded")

    def update_submodules(self, force=False):
        """
        Pulls updates from the submodule repositories. If FORCE is True,
        rebuilds submodules by overwriting all local changes.
        """

        utils.print_separator()
        print('[~] Retrieving latest submodules:')
        self.submodules = []
        repo = git.Repo.init()
        with open('.gitmodules', 'r') as file:
            lines = file.readlines()
            i = 0
            while i < len(lines):
                if lines[i].startswith('[') and i < len(lines) - 2:
                    path = lines[i + 1].split('=')[1].strip()
                    url = lines[i + 2].split('=')[1].strip()
                    self.submodules.append(path)
                    try:
                        repo.git.clone(url, path)       # First time loading submodule
                        print(f" -  Initialized submodule '{path}'")
                    except git.exc.GitCommandError:
                        sub_repo = git.Repo(path)
                        if not force:
                            sub_repo.git.stash()        # Save modified content
                        sub_repo.git.fetch('origin', 'main')
                        sub_repo.git.reset('--hard', 'FETCH_HEAD')
                        if not force:
                            try:                # Restore modified content
                                sub_repo.git.checkout('stash', '--', '.')
                                print(f" -  Updated submodule '{path}', restored local changes")
                            except git.exc.GitCommandError:
                                print(f" -  Updated submodule '{path}'")
                        else:
                            print(f" -  Rebuilt submodule '{path}'")
                        sub_repo.git.stash('clear')
                    i += 3
                else:
                    i += 1
