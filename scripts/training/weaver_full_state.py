#!/usr/bin/env python3
"""Run Weaver so that a restarted run continues exactly; all other arguments are Weaver's own.

Always: the full Lookahead checkpoint contract (``optimizer_contract.py``). With ``--cursor-data-epoch``:
train on a slice of that epoch's data stream (``data_cursor.py``), so that consecutive processes with the
same ``--seed`` reproduce one uninterrupted epoch.

    python scripts/training/weaver_full_state.py [--cursor-data-epoch E --cursor-skip N --cursor-take M
        --cursor-state-in PATH --cursor-state-out PATH] <weaver arguments>
"""
import argparse
from pathlib import Path
import sys

# Replace this script's directory, whose weaver.py would shadow the Weaver package, with scripts/.
sys.path[0] = str(Path(__file__).resolve().parents[1])

from training import data_cursor, optimizer_contract  # noqa: E402


def cursor_arguments(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--cursor-data-epoch', type=int)
    parser.add_argument('--cursor-skip', type=int, default=0)
    parser.add_argument('--cursor-take', type=int)
    parser.add_argument('--cursor-state-in')
    parser.add_argument('--cursor-state-out')
    return parser.parse_known_args(argv)


def main():
    cursor, weaver_argv = cursor_arguments(sys.argv[1:])
    optimizer_contract.install()
    if cursor.cursor_data_epoch is not None:
        data_cursor.install(cursor.cursor_data_epoch, cursor.cursor_skip, cursor.cursor_take,
                            cursor.cursor_state_in, cursor.cursor_state_out)
    elif any(vars(cursor).values()):
        raise SystemExit('--cursor-* options require --cursor-data-epoch')
    sys.argv = [sys.argv[0], *weaver_argv]
    from weaver.train import main as weaver_main
    return weaver_main()


if __name__ == '__main__':
    main()
