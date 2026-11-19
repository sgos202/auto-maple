"""
Starts tools/LLMHREmover/LLMHREmover.exe alongside Auto Maple.

LLMHREmover installs low-level keyboard and mouse hooks that clear the
"injected" flag on synthetic input (SendInput), so the bot's key presses and
clicks carry the same flags as real ones. It runs until Enter arrives on its
stdin, so we hold a pipe open to it:
 - Auto Maple closes normally  -> atexit sends Enter, it unhooks and exits
 - Auto Maple is killed        -> it sits in a job object that Windows closes
   when Auto Maple's process ends, which ends LLMHREmover too (a broken
   stdin pipe alone does NOT stop it)

It inherits Auto Maple's Administrator rights, which a low-level hook needs to
see input aimed at an elevated MapleStory window.

Turn off: delete tools/LLMHREmover (or rename the .exe). Its output goes to
logs/llmhremover.log.
"""

import atexit
import ctypes
import os
import subprocess
from ctypes import wintypes

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXE = os.path.join(ROOT, 'tools', 'LLMHREmover', 'LLMHREmover.exe')
LOG = os.path.join(ROOT, 'logs', 'llmhremover.log')

_process = None
_job = None


def _kill_with_us(pid):
    """Puts PID in a job that ends it when this process ends (any way)."""
    global _job
    k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.OpenProcess.restype = wintypes.HANDLE

    class BASIC(ctypes.Structure):
        _fields_ = [('PerProcessUserTimeLimit', ctypes.c_int64), ('PerJobUserTimeLimit', ctypes.c_int64),
                    ('LimitFlags', wintypes.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t),
                    ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', wintypes.DWORD),
                    ('Affinity', ctypes.c_size_t), ('PriorityClass', wintypes.DWORD),
                    ('SchedulingClass', wintypes.DWORD)]

    class IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ('r', 'w', 'o', 'rb', 'wb', 'ob')]

    class EXTENDED(ctypes.Structure):
        _fields_ = [('Basic', BASIC), ('Io', IO), ('ProcessMemoryLimit', ctypes.c_size_t),
                    ('JobMemoryLimit', ctypes.c_size_t), ('PeakProcessMemoryUsed', ctypes.c_size_t),
                    ('PeakJobMemoryUsed', ctypes.c_size_t)]

    job = k32.CreateJobObjectW(None, None)
    info = EXTENDED()
    info.Basic.LimitFlags = 0x2000          # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
        raise OSError(ctypes.get_last_error(), 'SetInformationJobObject')
    handle = k32.OpenProcess(0x0101, False, pid)     # PROCESS_TERMINATE | PROCESS_SET_QUOTA
    try:
        if not k32.AssignProcessToJobObject(job, handle):
            raise OSError(ctypes.get_last_error(), 'AssignProcessToJobObject')
    finally:
        k32.CloseHandle(handle)
    _job = job                               # held (never closed) until we exit


def _kill_leftovers():
    """Ends LLMHREmover copies left by an earlier run (two would hook twice)."""
    subprocess.run(['taskkill', '/IM', 'LLMHREmover.exe', '/F'], capture_output=True,
                   creationflags=subprocess.CREATE_NO_WINDOW)


def start():
    global _process
    if _process is not None and _process.poll() is None:
        return
    if not os.path.isfile(EXE):
        print(f'\n[~] LLMHREmover not found ({EXE}); carrying on without it.')
        return
    _kill_leftovers()
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    log = open(LOG, 'a', encoding='utf-8')
    _process = subprocess.Popen([EXE], cwd=os.path.dirname(EXE), stdin=subprocess.PIPE,
                                stdout=log, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        _kill_with_us(_process.pid)
    except Exception as error:
        print(f'\n[!] LLMHREmover will not close by itself if Auto Maple is killed: {error}')
    atexit.register(stop)
    print(f'\n[~] LLMHREmover started (pid {_process.pid})')


def stop():
    global _process
    if _process is None or _process.poll() is not None:
        return
    try:
        _process.stdin.write(b'\r\n')
        _process.stdin.flush()
        _process.wait(timeout=3)
    except Exception:
        _process.kill()
    _process = None


def running():
    return _process is not None and _process.poll() is None
