"""Captures application stdout and stderr for the GUI log viewer.

Writing to the console window must never block the bot. On Windows a click
inside the console starts a QuickEdit text selection, and from then on every
write blocks until the selection ends. Any thread that prints (the capture
loop, the bot, key logging) froze with it (14:30 log: minimap position lost,
the bot stood still and could not be restarted). So:
- QuickEdit is switched off for this console at start-up, and
- console writes go through a queue drained by one background thread; a stuck
  console only delays (and at worst drops) console text, never the caller.
"""

from collections import deque
from datetime import datetime
import queue
import sys
import threading


class OutputCapture:
    def __init__(self, max_events=5000):
        self._events = deque(maxlen=max_events)
        self._lock = threading.Lock()
        self._installed = False

    def install(self):
        if self._installed:
            return
        disable_console_quick_edit()
        writer = ConsoleWriter()
        self.file_log = FileLog()
        sys.stdout = TeeStream(sys.stdout, self, 'stdout', writer, self.file_log)
        sys.stderr = TeeStream(sys.stderr, self, 'stderr', writer, self.file_log)
        self.file_log.install_crash_hooks()
        self._installed = True

    def publish(self, stream, text):
        if text:
            with self._lock:
                self._events.append((stream, text))

    def drain(self):
        with self._lock:
            events = list(self._events)
            self._events.clear()
        return events


def timestamp():
    """Wall-clock time for log lines, e.g. '20:16:42.318'."""
    return datetime.now().strftime('%H:%M:%S.%f')[:-3]


def disable_console_quick_edit():
    """Turns QuickEdit off for this process's console (no-op without one)."""
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-10)         # STD_INPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        ENABLE_QUICK_EDIT_MODE, ENABLE_EXTENDED_FLAGS = 0x0040, 0x0080
        new_mode = (mode.value & ~ENABLE_QUICK_EDIT_MODE) | ENABLE_EXTENDED_FLAGS
        return bool(kernel32.SetConsoleMode(handle, new_mode))
    except Exception:
        return False


class ConsoleWriter:
    """Writes to the real console on its own thread."""

    MAX_PENDING = 20000

    def __init__(self):
        self._queue = queue.Queue(maxsize=self.MAX_PENDING)
        self._dropped = 0
        self._thread = threading.Thread(target=self._run, name='console-writer',
                                        daemon=True)
        self._thread.start()

    def write(self, stream, text):
        try:
            self._queue.put_nowait((stream, text))
        except queue.Full:
            self._dropped += 1      # console stuck for a long time: drop, don't wait

    def _run(self):
        while True:
            stream, text = self._queue.get()
            try:
                if self._dropped:
                    dropped, self._dropped = self._dropped, 0
                    stream.write(f'\n[!] {dropped} console lines were dropped '
                                 '(the console was not accepting text).\n')
                stream.write(text)
                if self._queue.empty():
                    stream.flush()
            except Exception:
                pass


class TeeStream:
    """
    Writes to the real stream and the GUI log, starting every non-blank line with
    the time, so console output can be lined up with recordings and other logs.
    """

    def __init__(self, stream, capture, name, writer=None, file_log=None):
        self._stream = stream
        self._capture = capture
        self._name = name
        self._writer = writer
        self._file_log = file_log
        self._line_start = True
        self._write_lock = threading.Lock()

    def write(self, text):
        if not text:
            return 0
        with self._write_lock:
            pieces = []
            for piece in text.splitlines(keepends=True):
                if self._line_start and piece.strip():
                    pieces.append(f'{timestamp()} ')
                pieces.append(piece)
                if piece.strip() or piece.endswith('\n'):
                    self._line_start = piece.endswith(('\n', '\r'))
            stamped = ''.join(pieces)
        if self._writer is not None:
            self._writer.write(self._stream, stamped)
        else:
            self._stream.write(stamped)
        if self._file_log is not None:
            self._file_log.write(stamped)
        self._capture.publish(self._name, stamped)
        return len(text)

    def flush(self):
        if self._writer is None:
            return self._stream.flush()
        return None     # the writer thread flushes; never wait on the console here

    def __getattr__(self, name):
        return getattr(self._stream, name)


class FileLog:
    """
    Everything printed also goes to logs/console/run-<start time>.log, so after
    a long run there is a record of how it ended (the console window is gone
    once it closes, and 2026-11-15's "did it die?" could not be answered).
    - Written on its own thread: a slow disk never blocks the bot.
    - A file rolls over at MAX_BYTES; only the newest KEEP_FILES are kept.
    - Native crashes (OpenCV, out of memory) are dumped by faulthandler to
      logs/console/crash-<start time>.txt; uncaught errors in any thread are
      printed (so logged); a normal exit writes an "exiting" line, so a log
      that just stops means the process was killed or crashed hard.
    """

    MAX_BYTES = 20 * 1024 * 1024
    KEEP_FILES = 12
    MAX_PENDING = 20000

    def __init__(self, folder=None):
        import os
        self._os = os
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.folder = folder or os.path.join(root, 'logs', 'console')
        self._stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self._part = 0
        self._file = None
        self._size = 0
        self._crash_file = None
        self._queue = queue.Queue(maxsize=self.MAX_PENDING)
        try:
            os.makedirs(self.folder, exist_ok=True)
            self._prune()
            self._open()
        except Exception:
            self._file = None
        self._thread = threading.Thread(target=self._run, name='file-log', daemon=True)
        self._thread.start()

    def path(self):
        suffix = f'-{self._part}' if self._part else ''
        return self._os.path.join(self.folder, f'run-{self._stamp}{suffix}.log')

    def _open(self):
        self._file = open(self.path(), 'a', encoding='utf-8', errors='replace')
        self._size = 0

    def _prune(self):
        os = self._os
        names = sorted((n for n in os.listdir(self.folder)
                        if n.startswith(('run-', 'crash-'))),
                       key=lambda n: os.path.getmtime(os.path.join(self.folder, n)))
        for name in names[:-self.KEEP_FILES] if len(names) > self.KEEP_FILES else []:
            try:
                os.remove(os.path.join(self.folder, name))
            except OSError:
                pass

    def write(self, text):
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass

    def _run(self):
        while True:
            text = self._queue.get()
            if self._file is None:
                continue
            try:
                self._file.write(text)
                self._size += len(text)
                if self._queue.empty():
                    self._file.flush()
                if self._size >= self.MAX_BYTES:
                    self._file.close()
                    self._part += 1
                    self._prune()
                    self._open()
            except Exception:
                pass

    def install_crash_hooks(self):
        import atexit
        import faulthandler
        import traceback
        try:
            path = self._os.path.join(self.folder, f'crash-{self._stamp}.txt')
            self._crash_file = open(path, 'w')
            faulthandler.enable(file=self._crash_file, all_threads=True)
        except Exception:
            pass

        def thread_hook(args):
            if args.exc_type is SystemExit:
                return
            name = args.thread.name if args.thread is not None else '?'
            print(f'\n[!] Thread "{name}" crashed:', file=sys.stderr)
            traceback.print_exception(args.exc_type, args.exc_value,
                                      args.exc_traceback, file=sys.stderr)

        def main_hook(exc_type, exc_value, tb):
            print('\n[!] Auto Maple crashed:', file=sys.stderr)
            traceback.print_exception(exc_type, exc_value, tb, file=sys.stderr)

        threading.excepthook = thread_hook
        sys.excepthook = main_hook
        atexit.register(self._on_exit)
        print(f'[~] Console log: {self.path()}')

    def _on_exit(self):
        self.write(f'{timestamp()} [~] Auto Maple exiting normally.\n')
        deadline = datetime.now().timestamp() + 1.0
        while not self._queue.empty() and datetime.now().timestamp() < deadline:
            threading.Event().wait(0.05)
        try:
            if self._crash_file is not None:
                empty = self._crash_file.tell() == 0
                self._crash_file.close()
                if empty:   # no native crash: don't leave an empty file
                    self._os.remove(self._crash_file.name)
        except Exception:
            pass


output_capture = OutputCapture()
