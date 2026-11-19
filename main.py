"""The central program that ties all the modules together."""

import os

# Before anything imports numpy: its maths library (OpenBLAS) reserves memory
# for every CPU thread at import, about 750 MB on a 32-thread PC, and the bot
# never uses it (image work is OpenCV's). With one thread it reserves ~10 MB.
# This is most of what the bot used to hold, and it matters on long runs: the
# 16:52 pause was Windows running out of memory with the bot at 1.2 GB.
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import time
from src.common.output import output_capture

output_capture.install()

try:    # input-flag hook (tools/LLMHREmover), before the bot sends any input
    from src.modules import hook_remover
    hook_remover.start()
except Exception as _error:
    print(f'\n[!] LLMHREmover not started: {_error}')

from src.modules.bot import Bot
from src.modules.capture import Capture
from src.modules.notifier import Notifier
from src.modules.listener import Listener
from src.modules.gui import GUI


bot = Bot()
capture = Capture()
notifier = Notifier()
listener = Listener()

bot.start()
while not bot.ready:
    time.sleep(0.01)

capture.start()
while not capture.ready:
    time.sleep(0.01)

notifier.start()
while not notifier.ready:
    time.sleep(0.01)

listener.start()
while not listener.ready:
    time.sleep(0.01)

try:    # status + commands for the phone remote (src/modules/phone_remote.py)
    from src.modules import remote_link
    remote_link.start()
except Exception as _error:
    print(f'\n[!] Phone remote link not started: {_error}')

print('\n[~] Successfully initialized Auto Maple')

gui = GUI()
gui.start()
