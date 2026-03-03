from spellchecker import SpellChecker   # pip install pyspellchecker
import keyboard                         # pip install keyboard
import pyautogui                        # pip install pyautogui


class LiveAutoCorrect:
    def __init__(self, language: str = "en"):
        self.spell = SpellChecker(language=language)
        self.buffer = []  # our shadow of what we've sent to OS

    @property
    def text(self) -> str:
        return "".join(self.buffer)

    def _last_word_bounds(self):
        s = self.text
        if not s:
            return None, None
        end = len(s)
        while end > 0 and s[end - 1] == " ":
            end -= 1
        if end == 0:
            return None, None
        start = s.rfind(" ", 0, end) + 1
        return start, end

    def _correct_last_word(self):
        start, end = self._last_word_bounds()
        if start is None:
            return 0, ""
        s = self.text
        last_word = s[start:end]
        if not last_word:
            return 0, ""
        corrected = self.spell.correction(last_word) or last_word
        orig_len = end - start
        # update buffer: remove old word, add corrected
        for _ in range(orig_len):
            self.buffer.pop()
        self.buffer.extend(corrected)
        return orig_len, corrected

    def handle_key(self, key: str):
        """
        key: 'a', 'b', 'space', 'backspace', etc.
        Returns list of tokens to send to OS: normal chars or 'BACKSPACE'.
        """
        actions = []

        if key == "space":
            # autocorrect previous word
            orig_len, corrected = self._correct_last_word()
            # delete original word
            for _ in range(orig_len):
                actions.append("BACKSPACE")
            # type corrected
            for c in corrected:
                actions.append(c)
                self.buffer.append(c)
            # type space
            actions.append(" ")
            self.buffer.append(" ")

        elif key == "backspace":
            if self.buffer:
                self.buffer.pop()
            actions.append("BACKSPACE")

        elif len(key) == 1:
            # simple printable character
            self.buffer.append(key)
            actions.append(key)

        # ignore all other keys
        return actions


def send_tokens(tokens):
    for t in tokens:
        if t == "BACKSPACE":
            pyautogui.press("backspace")
        else:
            pyautogui.typewrite(t)


def main():
    ac = LiveAutoCorrect()
    print("Global autocorrect ON.")
    print("1) Run this in a terminal.")
    print("2) Click into another text field (editor, browser, etc.).")
    print("3) Type there; on SPACE, previous word is autocorrected via backspaces.")
    print("4) Press ESC to quit.\n")

    def on_event(e):
        if e.event_type != keyboard.KEY_DOWN:
            return
        name = e.name

        if name == "esc":
            keyboard.unhook_all()
            print("\nStopped.")
            return

        if name in ("space", "backspace") or len(name) == 1:
            tokens = ac.handle_key(name)
            if tokens:
                send_tokens(tokens)

    keyboard.hook(on_event)
    keyboard.wait("esc")


if __name__ == "__main__":
    main()


