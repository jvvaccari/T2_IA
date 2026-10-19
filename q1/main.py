import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from q1.cli import play_game


if __name__ == "__main__":
    play_game()
