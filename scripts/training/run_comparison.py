#!/usr/bin/env python3
"""CLI entrypoint for independent comparison training runs."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.comparison.runner import main


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, KeyError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
