#!/usr/bin/env python3
"""Write diagnostics for a Weaver/PyTorch optimizer checkpoint."""

import argparse
import json
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from training.pbt.state.optimizer_state import (  # noqa: E402
    OPTIMIZER_STATE_MODES,
    atomic_torch_save,
    load_optimizer_state,
    normalize_optimizer_state_mode,
    summarize_optimizer_state,
    transform_optimizer_state,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("optimizer", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--preview-mode",
        choices=sorted(OPTIMIZER_STATE_MODES),
        default="raw",
        help="Include the selected transform settings in the report.",
    )
    parser.add_argument("--damping", type=float, default=0.1)
    parser.add_argument(
        "--write-transformed",
        type=Path,
        help="Optionally write a transformed optimizer checkpoint for inspection.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mode = normalize_optimizer_state_mode(args.preview_mode)
    state = load_optimizer_state(args.optimizer)
    summary = summarize_optimizer_state(state, top_k=args.top_k)
    summary.update(
        {
            "source_optimizer": str(args.optimizer),
            "transform_preview": {
                "mode": mode,
                "damping_factor": args.damping if mode == "damped" else None,
            },
        }
    )

    if args.write_transformed:
        transformed = transform_optimizer_state(
            state,
            mode=mode,
            damping_factor=args.damping,
        )
        atomic_torch_save(transformed, args.write_transformed)
        summary["transformed_optimizer"] = str(args.write_transformed)

    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
