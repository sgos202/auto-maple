"""Captures application stdout and stderr for the GUI log viewer."""

from collections import deque
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
        sys.stdout = TeeStream(sys.stdout, self, 'stdout')
        sys.stderr = TeeStream(sys.stderr, self, 'stderr')
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


class TeeStream:
    def __init__(self, stream, capture, name):
        self._stream = stream
        self._capture = capture
        self._name = name

    def write(self, text):
        written = self._stream.write(text)
        self._capture.publish(self._name, text)
        return written

    def flush(self):
        return self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


output_capture = OutputCapture()
