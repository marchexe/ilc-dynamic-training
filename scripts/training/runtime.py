#!/usr/bin/env python3
"""Shared training launcher utilities."""

import ast
import hashlib
import json
import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
SUPPORTED_DATA_EXTENSIONS = {"root", "parquet", "h5", "awkd"}
# (Weaver input name, jet-flavor file stem)
FLAVOR_SAMPLE_GROUPS = (("nnbb", "bb"), ("nncc", "cc"), ("nndd", "dd"))
DATA_SPLITS = (("train", "train800k"), ("val", "val50k"))
BKG_REJECTION_EFFICIENCY_POINTS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def weaver_executable():
    return PROJECT_DIR / ".venv" / "bin" / "weaver"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def normalize_data_extension(value):
    extension = str(value or "root").strip().lstrip(".").lower()
    if extension not in SUPPORTED_DATA_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_DATA_EXTENSIONS))
        raise ValueError(f"Unsupported data_extension: {extension}. Expected one of: {supported}")
    return extension


def required_sample_patterns(data_extension):
    data_extension = normalize_data_extension(data_extension)
    return [
        f"*_{flavor}_{suffix}.{data_extension}"
        for _, suffix in DATA_SPLITS
        for _, flavor in FLAVOR_SAMPLE_GROUPS
    ]


def data_paths(dataset, data_extension):
    dataset = Path(dataset)
    data_extension = normalize_data_extension(data_extension)
    return {
        split: [
            f"{label}:{dataset}/*_{flavor}_{suffix}.{data_extension}"
            for label, flavor in FLAVOR_SAMPLE_GROUPS
        ]
        for split, suffix in DATA_SPLITS
    }


def data_command_args(dataset, data_extension):
    paths = data_paths(dataset, data_extension)
    return ["--data-train", *paths["train"], "--data-val", *paths["val"]]


def validate_dataset(dataset, data_extension):
    dataset = Path(dataset)
    if not dataset.is_dir():
        raise FileNotFoundError(f"dataset not found: {dataset}")
    missing = [
        pattern
        for pattern in required_sample_patterns(data_extension)
        if not any(dataset.glob(pattern))
    ]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            f"dataset is missing required samples: {missing_list}"
        )


def git_metadata():
    def run(*args):
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_DIR,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _parse_bkg_rejection_at_eff(text):
    pattern = re.compile(
        r"- bkg_rejection_at_eff:\s*\n(?P<payload>\{.*?\})(?=\n\s+- |\n\[|\Z)",
        re.DOTALL,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    curves = ast.literal_eval(matches[-1].group("payload"))
    parsed = {pair: [float(value) for value in values] for pair, values in curves.items()}
    point_count = max((len(values) for values in parsed.values()), default=0)
    efficiencies = list(BKG_REJECTION_EFFICIENCY_POINTS[:point_count])
    return {
        "efficiencies": efficiencies,
        "pairs": parsed,
    }


def _curve_value(pairs, pair, index):
    values = pairs.get(pair) or []
    return values[index] if index < len(values) else None


def _bkg_rejection_lookup(curves):
    if not curves:
        return None
    pairs = curves["pairs"]
    efficiencies = curves["efficiencies"]
    lookup = {}
    for target_eff in (0.5, 0.8, 0.9, 1.0):
        if target_eff not in efficiencies:
            continue
        index = efficiencies.index(target_eff)
        lookup[f"b_tag_eff_{target_eff:.2f}"] = {
            "c_bkg_rejection": _curve_value(pairs, "bc", index),
            "d_bkg_rejection": _curve_value(pairs, "bd", index),
        }
        lookup[f"c_tag_eff_{target_eff:.2f}"] = {
            "b_bkg_rejection": _curve_value(pairs, "cb", index),
            "d_bkg_rejection": _curve_value(pairs, "cd", index),
        }
    return lookup or None


def read_metrics(path):
    log_path = path / "train.log" if path.is_dir() else path
    if not log_path.is_file():
        return None
    text = log_path.read_text(errors="replace")
    loss_accuracy = re.findall(r"Eval AvgLoss: ([0-9.]+), AvgAcc: ([0-9.]+)", text)
    auc = re.findall(r"roc_auc_score:\s*\n([0-9.]+)", text)
    if not loss_accuracy:
        return None
    loss, accuracy = loss_accuracy[-1]
    metrics = {
        "validation_loss": float(loss),
        "validation_accuracy": float(accuracy),
        "validation_auc": float(auc[-1]) if auc else None,
        "validation_shutdown_warning": "cannot schedule new futures after shutdown" in text,
    }
    for name in (
        "bkg_rejection_bc_score",
        "bkg_rejection_bd_score",
        "bkg_rejection_cb_score",
        "bkg_rejection_cd_score",
        "b_tag_rejection_score",
        "c_tag_rejection_score",
        "bkg_rejection_score",
    ):
        values = re.findall(rf"{name}:\s*\n([0-9.eE+-]+)", text)
        metrics[f"validation_{name}"] = float(values[-1]) if values else None
    curves = _parse_bkg_rejection_at_eff(text)
    if curves:
        metrics["validation_bkg_rejection_at_eff"] = curves
        metrics["validation_bkg_rejection_at_eff_lookup"] = _bkg_rejection_lookup(curves)
    return metrics


def terminate(processes):
    for process in processes.values():
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10
    for process in processes.values():
        if process.poll() is None:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
