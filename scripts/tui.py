"""Terminal plumbing shared by the two manual-review interfaces.

Both `thematic_filter.py` and `thematic_label.py` are single-keypress tools
driven for hours at a time, so the two ways a Windows console can silently
defeat them are worth solving once, here, rather than twice and differently.

## Reading a key on a non-Latin layout

`msvcrt.getch()` hands back a byte in the *console* codepage, which on a
Russian-locale Windows is 866, not UTF-8. Pressing the physical `k` key with
a Cyrillic layout active yields `b'\\xab'` -- `л` -- and decoding that as
UTF-8 produces a replacement character that matches no branch. The interface
then redraws the same item forever and looks frozen while behaving exactly as
written.

`msvcrt.getwch()` returns a `str` and skips the codepage entirely, so the
character arrives intact. It then still has to be *understood*: `л` is what
the `k` key produces, and the assessor pressing "keep" means keep. So every
letter is mapped back to the Latin key at the same physical position on a
ЙЦУКЕН layout. Both layouts work, and neither has to be the active one.

## Escape sequences

A console without virtual-terminal processing prints `\\x1b[2J\\x1b[H`
literally instead of clearing, so screens pile up and the header scrolls out
of view. `enable_ansi()` turns the flag on where it can; `clear()` falls back
to `cls`/`clear` where it cannot, so the caller never has to care.
"""

from __future__ import annotations

import os
import subprocess
import sys

ANSI_CLEAR = "\x1b[2J\x1b[H"

# Cyrillic character -> the Latin key at the same physical position (ЙЦУКЕН).
# Only letters are listed; digits and Enter are layout-independent already.
LAYOUT = dict(
    zip(
        "йцукенгшщзхъфывапролджэячсмитьбю",
        "qwertyuiop[]asdfghjkl;'zxcvbnm,.",
        strict=True,  # a typo in either row should fail loudly, not shift the map
    )
)
# ЙЦУКЕН puts `.` and `,` on the physical `/?` key, so an assessor reaching
# for "unsure" sends one of those. Neither has another meaning in these tools.
LAYOUT["."] = "/"
LAYOUT[","] = "?"

_ansi_ok: bool | None = None


def normalize_key(ch: str) -> str:
    """Fold a keypress to the Latin key at that physical position, lowercased.

    Pure: no I/O, so the layout table can be checked without a terminal.
    """
    if not ch:
        return ""
    ch = ch.lower()
    return LAYOUT.get(ch, ch)


def enable_ansi() -> bool:
    """Ask the console for virtual-terminal processing. True if escapes work."""
    global _ansi_ok
    if _ansi_ok is not None:
        return _ansi_ok
    if os.name != "nt":
        _ansi_ok = sys.stdout.isatty()
        return _ansi_ok
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            _ansi_ok = False
            return _ansi_ok
        enable_vt = 0x0004  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        _ansi_ok = bool(kernel32.SetConsoleMode(handle, mode.value | enable_vt))
    except (OSError, AttributeError, ImportError):
        _ansi_ok = False
    return _ansi_ok


def clear() -> None:
    """Blank the screen and home the cursor, by whichever means works."""
    if enable_ansi():
        print(ANSI_CLEAR, end="")
        return
    if not sys.stdout.isatty():  # piped or captured: a form feed keeps logs readable
        print("\f", end="")
        return
    subprocess.run("cls" if os.name == "nt" else "clear", shell=True, check=False)


def read_key() -> str:
    """One keypress, no Enter, folded through `normalize_key`.

    The platform module is imported lazily so this file still loads on the
    other operating system.
    """
    if os.name == "nt":
        import msvcrt

        ch = msvcrt.getwch()  # wide-char: bypasses the console codepage
        if ch in ("\x00", "\xe0"):  # function/arrow key: consume the second unit
            msvcrt.getwch()
            return ""
        if ch == "\x03":
            raise KeyboardInterrupt
        return normalize_key(ch)

    import termios
    import tty

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
    if ch == "\x03":
        raise KeyboardInterrupt
    return normalize_key(ch)
