"""Shared character vocabulary and keyboard-mapping constants."""

from typing import Dict, List, Tuple

LETTER_CHARS = "abcdefghijklmnopqrstuvwxyz"
DIGIT_CHARS = "0123456789"
SPECIAL_CHARS = (" ", "\n", "\b", "\t")

# CHAR_TO_INDEX: Dict[str, int] = {}
# _idx = 0
# for c in LETTER_CHARS:
#     CHAR_TO_INDEX[c] = _idx
#     _idx += 1
# # for c in DIGIT_CHARS:
# #     CHAR_TO_INDEX[c] = _idx
# #     _idx += 1
# for c in SPECIAL_CHARS:
#     CHAR_TO_INDEX[c] = _idx
#     _idx += 1

CHAR_TO_INDEX = {
    'q': 0, 'a': 0, 'z': 0,
    'w': 1, 's': 1, 'x': 1,
    'e': 2, 'd': 2, 'c': 2,
    'r': 3, 'f': 3, 'v': 3,
    't': 4, 'g': 4, 'b': 4,
    'y': 5, 'h': 5, 'n': 5,
    'u': 6, 'j': 6, 'm': 6,
    'i': 7, 'k': 7,
    'o': 8, 'l': 8,
    'p': 9,
    ' ': 0,  # sentinel for "no previous char" in inference
}

# INDEX_TO_CHAR = {v: k for k, v in CHAR_TO_INDEX.items()}
INDEX_TO_CHAR = {
    0: "qaz",
    1: "wsx",
    2: "edc",
    3: "rfv",
    4: "tgb",
    5: "yhn",
    6: "ujm",
    7: "ik",
    8: "ol",
    9: "p"
}
# NUM_CLASSES = len(CHAR_TO_INDEX)
NUM_CLASSES = 10

# --- Full char mapping: letters + special chars (no digits) ---
# One class per character: a-z (26) + space, newline, tab, backspace (4) = 30
CHAR_TO_INDEX_CHARS: Dict[str, int] = {}
for i, c in enumerate("abcdefghijklmnopqrstuvwxyz"):
    CHAR_TO_INDEX_CHARS[c] = i
CHAR_TO_INDEX_CHARS[" "] = 26
CHAR_TO_INDEX_CHARS["\n"] = 27
CHAR_TO_INDEX_CHARS["\t"] = 28
CHAR_TO_INDEX_CHARS["\b"] = 29

INDEX_TO_CHAR_CHARS = {v: k for k, v in CHAR_TO_INDEX_CHARS.items()}
NUM_CLASSES_CHARS = len(CHAR_TO_INDEX_CHARS)

ALL_CHARS_FULL = list(INDEX_TO_CHAR_CHARS[i] for i in range(NUM_CLASSES_CHARS))

# --- 4-class row-based mapping (optional) ---
# 0: top row (qwertyuiop), 1: home row (asdfghjkl), 2: bottom row (zxcvbnm), 3: space
CHAR_TO_INDEX_4: Dict[str, int] = {}
for c in "qwertyuiop":
    CHAR_TO_INDEX_4[c] = 0
for c in "asdfghjkl":
    CHAR_TO_INDEX_4[c] = 1
for c in "zxcvbnm":
    CHAR_TO_INDEX_4[c] = 2
CHAR_TO_INDEX_4[" "] = 3
CHAR_TO_INDEX_4["\n"] = 3  # enter/space-like sentinel

INDEX_TO_CHAR_4 = {
    0: "qwertyuiop",
    1: "asdfghjkl",
    2: "zxcvbnm",
    3: " ",
}
NUM_CLASSES_4 = 4

ALL_CHARS_4 = [INDEX_TO_CHAR_4[i] for i in range(NUM_CLASSES_4)]

# --- 10-class diagonal mapping (top-left to bottom-right diagonals on QWERTY) ---
# Diag 0: q | Diag 1: wa | Diag 2: esz | Diag 3: rdx | Diag 4: tfc | Diag 5: ygv | Diag 6: uhb | Diag 7: ijn | Diag 8: okm | Diag 9: pl
CHAR_TO_INDEX_DIAGONAL: Dict[str, int] = {}
for c in "q":
    CHAR_TO_INDEX_DIAGONAL[c] = 0
for c in "wa":
    CHAR_TO_INDEX_DIAGONAL[c] = 1
for c in "esz":
    CHAR_TO_INDEX_DIAGONAL[c] = 2
for c in "rdx":
    CHAR_TO_INDEX_DIAGONAL[c] = 3
for c in "tfc":
    CHAR_TO_INDEX_DIAGONAL[c] = 4
for c in "ygv":
    CHAR_TO_INDEX_DIAGONAL[c] = 5
for c in "uhb":
    CHAR_TO_INDEX_DIAGONAL[c] = 6
for c in "ijn":
    CHAR_TO_INDEX_DIAGONAL[c] = 7
for c in "okm":
    CHAR_TO_INDEX_DIAGONAL[c] = 8
for c in "pl":
    CHAR_TO_INDEX_DIAGONAL[c] = 9
CHAR_TO_INDEX_DIAGONAL[" "] = 0  # sentinel for "no previous char"
CHAR_TO_INDEX_DIAGONAL["\n"] = 0

INDEX_TO_CHAR_DIAGONAL = {
    0: "q",
    1: "wa",
    2: "esz",
    3: "rdx",
    4: "tfc",
    5: "ygv",
    6: "uhb",
    7: "ijn",
    8: "okm",
    9: "pl",
}
NUM_CLASSES_DIAGONAL = 10

ALL_CHARS_DIAGONAL = [INDEX_TO_CHAR_DIAGONAL[i] for i in range(NUM_CLASSES_DIAGONAL)]

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

FULL_COORDS_4: Dict[str, Tuple[float, float]] = {}
for _ch in ALL_CHARS_4:
    if _ch in KEY_COORDS:
        FULL_COORDS_4[_ch] = KEY_COORDS[_ch]
    elif _ch in SPECIAL_COORDS:
        FULL_COORDS_4[_ch] = SPECIAL_COORDS[_ch]
    elif _ch == " ":
        FULL_COORDS_4[_ch] = (5.3, 4.0)
    else:
        FULL_COORDS_4[_ch] = (5.3, 4.0)

# Coords for full-char mode (letters + space; \n \t \b use space position)
FULL_COORDS_CHARS: Dict[str, Tuple[float, float]] = {}
for _ch in ALL_CHARS_FULL:
    if _ch in KEY_COORDS:
        FULL_COORDS_CHARS[_ch] = KEY_COORDS[_ch]
    elif _ch in SPECIAL_COORDS:
        FULL_COORDS_CHARS[_ch] = SPECIAL_COORDS[_ch]
    elif _ch == " ":
        FULL_COORDS_CHARS[_ch] = (5.3, 4.0)
    else:
        FULL_COORDS_CHARS[_ch] = (5.3, 4.0)

FULL_COORDS_DIAGONAL: Dict[str, Tuple[float, float]] = {}
for _ch in ALL_CHARS_DIAGONAL:
    if _ch in KEY_COORDS:
        FULL_COORDS_DIAGONAL[_ch] = KEY_COORDS[_ch]
    elif _ch in SPECIAL_COORDS:
        FULL_COORDS_DIAGONAL[_ch] = SPECIAL_COORDS[_ch]
    elif _ch == " ":
        FULL_COORDS_DIAGONAL[_ch] = (5.3, 4.0)
    elif len(_ch) == 1:
        FULL_COORDS_DIAGONAL[_ch] = KEY_COORDS.get(_ch, (5.3, 4.0))
    else:
        # Centroid of constituent keys for multi-char diagonal (e.g. "wa", "esz")
        pts = [KEY_COORDS[c] for c in _ch if c in KEY_COORDS]
        if pts:
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            FULL_COORDS_DIAGONAL[_ch] = (cx, cy)
        else:
            FULL_COORDS_DIAGONAL[_ch] = (5.3, 4.0)

INITIAL_KEY_COORDS: Dict[str, Tuple[float, float, float]] = { # initial guess for imu position, with z=0.0 for all keys
    'base_L': (0.0, 0.0, 0.0),
    'thumb_L': (0.065, 0.055, 0.0),
    'index_L': (0.047, 0.088, 0.0),
    'middle_L': (0.020, 0.044, 0.0),
    'ring_L': (0.001, 0.088, 0.0),
    'pinky_L': (-0.015, 0.080, 0.0),
    'base_R': (0.17, 0.0, 0.0),
    'thumb_R': (0.11, 0.043, 0.0),
    'index_R': (0.14, 0.090, 0.0),
    'middle_R': (0.168, 0.096, 0.0),
    'ring_R': (0.190, 0.085, 0.0),
    'pinky_R': (0.205, 0.073, 0.0),
}
