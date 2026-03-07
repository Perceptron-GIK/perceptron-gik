import sys, neuspell
from spellchecker import SpellChecker
from neuspell import BertChecker

valid_checkers = ["pyspell", "neuspell-bert"]

pyspell_checker = SpellChecker()

neuspell_bert_path = neuspell.seq_modeling.downloads.download_pretrained_model("subwordbert-probwordnoise")
neuspell_bert = BertChecker()
neuspell_bert.from_pretrained(neuspell_bert_path)

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
            try:
                n = len(self.buffer)
                sys.stdout.write("\b"*n)
                sys.stdout.write(" "*n)
                sys.stdout.write("\b"*n)
                sys.stdout.write(corrected)
                sys.stdout.write(" ")
            except:
                pass
        
        self.buffer = ""

    def process_char(self, c):
        self.buffer += c
        if c.isalnum():
            if len(self.buffer) >= self.max_len:
                self.correct_word()
        else:
            self.correct_word()
