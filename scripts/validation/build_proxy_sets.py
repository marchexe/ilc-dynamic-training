#!/usr/bin/env python3
"""Build deterministic representative, hard, and mixed proxy sets."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from training.pbt.execution.weaver_command import make_tiered_evaluation_command
from training.runtime import project_path
from validation.proxy_qualification import build_proxy_sets


DEFAULT_RUN_DIR = Path("runs/eval/proxy_qualification_v1")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/20250711_ilc_nnqq_sgv_10m_3cat_parquet"))
    parser.add_argument("--source-suffix", default="val50k_tail")
    parser.add_argument(
        "--anchor-checkpoint", type=Path,
        default=Path("runs/pbt/windowed_pbt_v2_100epochs/checkpoints/global_best_state.pt"),
    )
    parser.add_argument("--anchor-predictions", type=Path, default=None)
    parser.add_argument("--data-config", type=Path, default=Path("checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/data_config.auto.yaml"))
    parser.add_argument("--network-config", type=Path, default=Path("networks/pretrained_sgv_particle_transformer.py"))
    parser.add_argument("--sizes", default="15000,30000,60000")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--mixed-hard-fraction", type=float, default=0.30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--fetch-step", default="0.01")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--host", default="iutgpu01")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_anchor_predictions(args, prediction_path):
    if prediction_path.is_file() and not args.force:
        return
    run_dir = project_path(args.run_dir)
    log_path = run_dir / "cache" / "anchor_predictions.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "shared": {
            "data_config": str(project_path(args.data_config)),
            "network_config": str(project_path(args.network_config)),
            "data_extension": "parquet",
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "fetch_step": args.fetch_step,
            "use_amp": True,
            "amp_dtype": "fp16",
            "deterministic": True,
            "data_audit": True,
            "save_predictions": True,
        }
    }
    slot = {"host": args.host, "gpu": str(args.gpu), "label": f"{args.host}:{args.gpu}"} if args.host else str(args.gpu)
    command, _ = make_tiered_evaluation_command(
        config,
        slot,
        project_path(args.anchor_checkpoint),
        project_path(args.dataset),
        args.source_suffix,
        log_path,
    )
    result = subprocess.run(command, cwd=project_path("."), check=False)
    if result.returncode:
        raise SystemExit(result.returncode)
    generated = log_path.with_suffix(".predictions.parquet")
    if not generated.is_file():
        raise RuntimeError(f"anchor inference completed without prediction output: {generated}")
    if generated.resolve() != prediction_path.resolve():
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        generated.replace(prediction_path)


def main():
    args = parse_args()
    run_dir = project_path(args.run_dir)
    prediction_path = project_path(args.anchor_predictions) if args.anchor_predictions else run_dir / "cache" / "anchor_predictions.parquet"
    ensure_anchor_predictions(args, prediction_path)
    manifest = build_proxy_sets(
        dataset=args.dataset,
        source_suffix=args.source_suffix,
        output_root=run_dir / "proxy_datasets",
        manifest_output=run_dir / "proxy_sets.json",
        anchor_checkpoint=args.anchor_checkpoint,
        anchor_predictions=prediction_path,
        sizes=[int(value) for value in args.sizes.split(",") if value],
        seed=args.seed,
        mixed_hard_fraction=args.mixed_hard_fraction,
        force=args.force,
    )
    print(json.dumps({"proxy_sets": str(run_dir / "proxy_sets.json"), "candidates": [item["id"] for item in manifest["candidates"]]}, indent=2))


if __name__ == "__main__":
    main()
