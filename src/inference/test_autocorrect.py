import platform, sys
from autocorrect import AutoCorrector

AUTOCORRECTOR = AutoCorrector(checker_type="pyspell", max_len=10)

def get_char():
    if platform.system() == "Windows":
        import msvcrt
        return msvcrt.getwch()
    else:
        import tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    
print("Autocorrector is ready, start typing")

while True:
    c = get_char()
    sys.stdout.write(c)
    sys.stdout.flush()
    AUTOCORRECTOR.process_char(c)
