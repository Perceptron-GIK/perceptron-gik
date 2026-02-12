"""Inference calibration gate.

Provides a simple mechanism so that real-time inference only begins
once the operator presses the **spacebar**, allowing the user to
position their hands before data collection starts (Issue #7).

Usage
-----
>>> from src.keyboard.calibration import wait_for_spacebar
>>> import asyncio
>>> asyncio.run(wait_for_spacebar())   # blocks until space is pressed
"""

import asyncio
import sys


async def wait_for_spacebar() -> None:
    """Block until the user presses the spacebar.

    Works in a terminal environment by reading stdin in a background
    thread so that the asyncio event loop is not blocked.
    """
    print("Press SPACEBAR to start inference...", flush=True)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _blocking_wait_for_space)
    print("Spacebar pressed – inference starting.", flush=True)


def _blocking_wait_for_space() -> None:
    """Read stdin character-by-character until a space is detected."""
    # Try to use raw terminal mode for immediate key detection
    try:
        import tty
        import termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.read(1)
                if ch == ' ':
                    return
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except (ImportError, AttributeError, termios.error):
        # Fallback: read line-by-line (user types space then Enter)
        while True:
            line = input()
            if ' ' in line or line == '':
                return
