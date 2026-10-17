#!/usr/bin/env python3
"""Run Weaver with the full Lookahead checkpoint contract; arguments are Weaver's own.

    python scripts/training/weaver_full_state.py <weaver arguments>
"""
from pathlib import Path
import sys

# Replace this script's directory, whose weaver.py would shadow the Weaver package, with scripts/.
sys.path[0] = str(Path(__file__).resolve().parents[1])

from training.optimizer_contract import install  # noqa: E402


def main():
    install()
    from weaver.train import main as weaver_main
    return weaver_main()


if __name__ == '__main__':
    main()
