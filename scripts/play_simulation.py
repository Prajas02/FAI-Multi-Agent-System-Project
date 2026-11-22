"""Launch the interactive saved-plan animation."""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.visualization.animation import run_pygame


def main(argv=None):
    parser = argparse.ArgumentParser(description="Play a saved multi-robot allocation bundle.")
    parser.add_argument("--bundle", required=True, action="append",
                        help="saved playback JSON bundle (repeat to select scenarios in the window)")
    parser.add_argument("--algorithm", default=None)
    args = parser.parse_args(argv)
    run_pygame([Path(path) for path in args.bundle], args.algorithm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
