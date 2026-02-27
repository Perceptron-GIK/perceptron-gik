"""Shared character vocabulary and keyboard-mapping constants."""

from typing import Dict, List, Tuple

LETTER_CHARS = "abcdefghijklmnopqrstuvwxyz"
DIGIT_CHARS = "0123456789"
SPECIAL_CHARS = (" ", "\n", "\b", "\t")

CHAR_TO_INDEX: Dict[str, int] = {}
_idx = 0
for c in LETTER_CHARS:
    CHAR_TO_INDEX[c] = _idx
    _idx += 1
for c in DIGIT_CHARS:
    CHAR_TO_INDEX[c] = _idx
    _idx += 1
for c in SPECIAL_CHARS:
    CHAR_TO_INDEX[c] = _idx
    _idx += 1

INDEX_TO_CHAR = {v: k for k, v in CHAR_TO_INDEX.items()}
NUM_CLASSES = len(CHAR_TO_INDEX)

SPECIAL_KEY_MAP = {
    "enter": "\n",
    "space": " ",
    "tab": "\t",
    "backspace": "\b",
}

# Refined coordinates for letters and digits, mapped to a QWERTY-like layout.
KEY_COORDS: Dict[str, Tuple[float, float]] = {
    "1": (0.0, 0.0), "2": (1.0, 0.0), "3": (2.0, 0.0), "4": (3.0, 0.0), "5": (4.0, 0.0),
    "6": (5.0, 0.0), "7": (6.0, 0.0), "8": (7.0, 0.0), "9": (8.0, 0.0), "0": (9.0, 0.0),
    "q": (0.5, 1.0), "w": (1.5, 1.0), "e": (2.5, 1.0), "r": (3.5, 1.0), "t": (4.5, 1.0),
    "y": (5.5, 1.0), "u": (6.5, 1.0), "i": (7.5, 1.0), "o": (8.5, 1.0), "p": (9.5, 1.0),
    "a": (0.8, 2.0), "s": (1.8, 2.0), "d": (2.8, 2.0), "f": (3.8, 2.0), "g": (4.8, 2.0),
    "h": (5.8, 2.0), "j": (6.8, 2.0), "k": (7.8, 2.0), "l": (8.8, 2.0),
    "z": (1.3, 3.0), "x": (2.3, 3.0), "c": (3.3, 3.0), "v": (4.3, 3.0), "b": (5.3, 3.0),
    "n": (6.3, 3.0), "m": (7.3, 3.0),
}

# Approximate coordinates for non-alphanumeric keys.
SPECIAL_COORDS: Dict[str, Tuple[float, float]] = {
    "\n": (12.0, 2.0),  # Enter, to the right of L
    "\b": (12.0, 0.0),  # Backspace, to the right of P
    "\t": (-0.5, 1.0),  # Tab, to the left of Q
}

# Multiple anchors to mimic a long spacebar.
SPACE_ANCHORS: List[Tuple[float, float]] = [
    (3.3, 4.0),  # under C
    (4.3, 4.0),  # under V
    (5.3, 4.0),  # under B
    (6.3, 4.0),  # under N
    (7.3, 4.0),  # under M
]

ALL_CHARS = [INDEX_TO_CHAR[i] for i in range(NUM_CLASSES)]

FULL_COORDS: Dict[str, Tuple[float, float]] = {}
for _ch in ALL_CHARS:
    if _ch in KEY_COORDS:
        FULL_COORDS[_ch] = KEY_COORDS[_ch]
    elif _ch in SPECIAL_COORDS:
        FULL_COORDS[_ch] = SPECIAL_COORDS[_ch]
    elif _ch == " ":
        # Spacebar center (length is represented by SPACE_ANCHORS).
        FULL_COORDS[_ch] = (5.3, 4.0)
    else:
        # Fallback for any unexpected character.
        FULL_COORDS[_ch] = (5.3, 4.0)
