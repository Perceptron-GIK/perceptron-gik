from spellchecker import SpellChecker
from neuspell import BertChecker
import sys

valid_checkers = ["pyspell", "neuspell_bert"]
pyspell_checker = SpellChecker()
neuspell_bert = BertChecker().from_pretrained("bert_base")

class AutoCorrector:
    def __init__(self, checker_type="pyspell", max_len=10):
        self.buffer = ""
        self.checker_type = checker_type
        self.max_len = max_len
    
    def correct_word(self):
        if not self.buffer:
            return
        
        if self.checker_type == "pyspell":
            corrected = pyspell_checker.correction(self.buffer)
        elif self.checker_type == "neuspell-bert":
            corrected = neuspell_bert.correct(self.buffer)
        else:
            raise ValueError(f"Invalid checker type. Choose from {valid_checkers}")

        if corrected != self.buffer:
            n = len(self.buffer)

            sys.stdout.write("\b"*n)
            sys.stdout.write(" "*n)
            sys.stdout.write("\b"*n)
            sys.stdout.write(corrected)
        
        self.buffer = ""

    def process_char(self, c):
        if c.isalnum():
            self.buffer += c
            if len(self.buffer) >= self.max_len:
                self.correct_word()
        else:
            self.correct_word()
