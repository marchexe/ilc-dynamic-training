#!/usr/bin/env python3
"""Shared training launcher utilities."""

import ast
import hashlib
import json
import math
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
DEFAULT_TRAIN_SUFFIX = "train800k"
DEFAULT_VALIDATION_SUFFIX = "val50k"
DATA_SPLITS = (("train", DEFAULT_TRAIN_SUFFIX), ("val", DEFAULT_VALIDATION_SUFFIX))
BKG_REJECTION_EFFICIENCY_POINTS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
# Fine-tuning selection points: b-tag is judged at 80/90%, c-tag at the
# reference table working points 50/80%. Pair names are <tag><background>.
WORKING_POINT_DEFINITION = (
    ("bc", 0.8),
    ("bd", 0.8),
    ("bc", 0.9),
    ("bd", 0.9),
    ("cb", 0.5),
    ("cd", 0.5),
    ("cb", 0.8),
    ("cd", 0.8),
)
CTAG_REFERENCE_WORKING_POINTS = (
    ("cb", 0.5),
    ("cd", 0.5),
    ("cb", 0.8),
    ("cd", 0.8),
)


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


def expand_path_aliases(value):
    text = str(value)
    aliases = {
        "PROJECT_DIR": PROJECT_DIR,
        "PART_ROOT": PROJECT_DIR.parent,
        "PART_DATA_DIR": PROJECT_DIR.parent / "data",
    }
    for name, path in aliases.items():
        for token in (f"${{{name}}}", f"${name}", f"<{name}>"):
            if text == token:
                return str(path)
            if text.startswith(f"{token}/"):
                return str(path / text[len(token) + 1:])
    return os.path.expandvars(text)


def project_path(value):
    path = Path(expand_path_aliases(value))
    return path if path.is_absolute() else PROJECT_DIR / path


def normalize_data_extension(value):
    extension = str(value or "root").strip().lstrip(".").lower()
    if extension not in SUPPORTED_DATA_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_DATA_EXTENSIONS))
        raise ValueError(f"Unsupported data_extension: {extension}. Expected one of: {supported}")
    return extension


def split_suffixes(train_suffix=None, validation_suffix=None):
    return {
        "train": str(train_suffix or DEFAULT_TRAIN_SUFFIX),
        "val": str(validation_suffix or DEFAULT_VALIDATION_SUFFIX),
    }


def required_sample_patterns(data_extension, split=None, train_suffix=None, validation_suffix=None):
    data_extension = normalize_data_extension(data_extension)
    suffixes = split_suffixes(train_suffix, validation_suffix)
    splits = tuple(suffixes) if split is None else (split,)
    if any(name not in suffixes for name in splits):
        raise ValueError(f"unknown data split: {split}")
    return [
        f"*_{flavor}_{suffixes[name]}.{data_extension}"
        for name in splits
        for _, flavor in FLAVOR_SAMPLE_GROUPS
    ]


def data_paths(dataset, data_extension, validation_dataset=None, train_suffix=None, validation_suffix=None):
    train_dataset = Path(dataset)
    val_dataset = Path(validation_dataset) if validation_dataset else train_dataset
    data_extension = normalize_data_extension(data_extension)
    suffixes = split_suffixes(train_suffix, validation_suffix)
    return {
        "train": [
            f"{label}:{train_dataset}/*_{flavor}_{suffixes['train']}.{data_extension}"
            for label, flavor in FLAVOR_SAMPLE_GROUPS
        ],
        "val": [
            f"{label}:{val_dataset}/*_{flavor}_{suffixes['val']}.{data_extension}"
            for label, flavor in FLAVOR_SAMPLE_GROUPS
        ],
    }


def data_command_args(dataset, data_extension, validation_dataset=None, train_suffix=None, validation_suffix=None):
    paths = data_paths(dataset, data_extension, validation_dataset, train_suffix, validation_suffix)
    return ["--data-train", *paths["train"], "--data-val", *paths["val"]]


def validate_dataset(dataset, data_extension, validation_dataset=None, train_suffix=None, validation_suffix=None):
    dataset = Path(dataset)
    if not dataset.is_dir():
        raise FileNotFoundError(f"dataset not found: {dataset}")
    missing = [
        pattern
        for pattern in required_sample_patterns(data_extension, split="train", train_suffix=train_suffix)
        if not any(dataset.glob(pattern))
    ]
    if validation_dataset:
        validation_dataset = Path(validation_dataset)
        if not validation_dataset.is_dir():
            raise FileNotFoundError(f"validation_dataset not found: {validation_dataset}")
        missing.extend(
            f"validation_dataset:{pattern}"
            for pattern in required_sample_patterns(
                data_extension,
                split="val",
                validation_suffix=validation_suffix,
            )
            if not any(validation_dataset.glob(pattern))
        )
    else:
        missing.extend(
            pattern
            for pattern in required_sample_patterns(
                data_extension,
                split="val",
                validation_suffix=validation_suffix,
            )
            if not any(dataset.glob(pattern))
        )
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


def _parse_metric_payload(text, name):
    pattern = re.compile(
        rf"- {name}:\s*\n(?P<payload>\{{.*?\}})(?=\n\s+- |\n\[|\Z)",
        re.DOTALL,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    payload = matches[-1].group("payload")
    try:
        return ast.literal_eval(payload)
    except ValueError:
        return eval(payload, {"__builtins__": {}}, {"nan": float("nan")})


def _parse_bkg_rejection_at_eff(text):
    curves = _parse_metric_payload(text, "bkg_rejection_at_eff")
    if not curves:
        return None
    parsed = {pair: [float(value) for value in values] for pair, values in curves.items()}
    point_count = max((len(values) for values in parsed.values()), default=0)
    efficiencies = list(BKG_REJECTION_EFFICIENCY_POINTS[:point_count])
    return {
        "efficiencies": efficiencies,
        "pairs": parsed,
    }


def _parse_bkg_rejection_at_eff_counts(text):
    counts = _parse_metric_payload(text, "bkg_rejection_at_eff_counts")
    if not counts:
        return None
    parsed = {}
    for pair, rows in counts.items():
        parsed[pair] = []
        for row in rows:
            parsed[pair].append(
                {
                    "signal_efficiency": float(row["signal_efficiency"]),
                    "background_passed": None if row["background_passed"] is None else int(row["background_passed"]),
                    "background_total": int(row["background_total"]),
                    "background_efficiency": None if row["background_efficiency"] is None else float(row["background_efficiency"]),
                }
            )
    return parsed


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


def geometric_mistag_score(x1, x2, x3, x4):
    """(x1 * x2 * x3 * x4) ** 0.25 -- the four-case geometric-mean mistag
    score used by anchor_copy_lr_recenter's model selection. Lower is
    better, same convention as every other mistag-percent metric in this
    file. All four inputs must be finite, non-negative numbers (a mistag
    percentage is never negative and never inf/NaN in a valid result) --
    reject anything else rather than silently coercing it. Zero is a
    legitimate input (a perfect working point) and needs no epsilon
    substitution: plain multiplication makes the product (and therefore the
    score) exactly 0.0 without ever dividing by or taking the log of an
    input, so there is no zero-related singularity to guard against here.
    """
    values = (x1, x2, x3, x4)
    if any(value is None for value in values):
        raise ValueError("geometric_mistag_score requires four non-None values")
    numeric = [float(value) for value in values]
    for value in numeric:
        if not math.isfinite(value):
            raise ValueError("geometric_mistag_score inputs must be finite (no NaN/inf)")
        if value < 0:
            raise ValueError("geometric_mistag_score inputs must be non-negative")
    product = numeric[0] * numeric[1] * numeric[2] * numeric[3]
    return product, product ** 0.25


# The four working points anchor_copy_lr_recenter's mistag_score is built
# from, in x1..x4 order -- a subset of CTAG_REFERENCE_WORKING_POINTS below,
# all c-tag mistag percentages (same unit, same tagger), never mixed with
# b-tag values.
GEOMETRIC_MISTAG_WORKING_POINTS = (("cb", 0.5), ("cd", 0.5), ("cb", 0.8), ("cd", 0.8))


def _mistags_for_working_points(curves, working_points):
    pairs = curves["pairs"]
    efficiencies = curves["efficiencies"]
    mistags = []
    out = {}
    for pair, eff in working_points:
        if eff not in efficiencies:
            continue
        index = efficiencies.index(eff)
        rejection = _curve_value(pairs, pair, index)
        key = f"validation_{pair}_mistag_eff_{eff:.2f}_percent"
        if rejection is None or rejection <= 0 or not math.isfinite(rejection):
            out[key] = None
            continue
        mistag = 100.0 / rejection
        if not math.isfinite(mistag):
            out[key] = None
            continue
        out[key] = mistag
        mistags.append(mistag)
    return out, mistags


def _working_point_metrics(curves):
    if not curves:
        return {}
    out, mistags = _mistags_for_working_points(curves, WORKING_POINT_DEFINITION)
    _, ctag_mistags = _mistags_for_working_points(curves, CTAG_REFERENCE_WORKING_POINTS)
    out["validation_working_point_mistag_percent"] = (
        sum(mistags) / len(mistags) if mistags else None
    )
    out["validation_ctag_reference_mistag_percent"] = (
        sum(ctag_mistags) / len(ctag_mistags) if ctag_mistags else None
    )

    # x1..x4 for anchor_copy_lr_recenter's geometric mistag_score, aliased
    # from the per-point values already computed above (GEOMETRIC_MISTAG_
    # WORKING_POINTS is a subset of WORKING_POINT_DEFINITION) so the x1-x4
    # mapping is self-documenting in the metrics dict itself, not something
    # a reader has to cross-reference against WORKING_POINT_DEFINITION's
    # ordering to recover. A working point genuinely absent from this
    # checkpoint's curves (out[key] is None) is a normal partial-eval
    # outcome here, not invalid input -- skip the score rather than raise.
    x_keys = [
        f"validation_{pair}_mistag_eff_{eff:.2f}_percent" for pair, eff in GEOMETRIC_MISTAG_WORKING_POINTS
    ]
    x_values = [out.get(key) for key in x_keys]
    for label, value in zip(("x1", "x2", "x3", "x4"), x_values):
        out[f"validation_mistag_geometric_score_{label}_percent"] = value
    if all(value is not None for value in x_values):
        product, score = geometric_mistag_score(*x_values)
        out["validation_mistag_geometric_score_product"] = product
        out["validation_mistag_geometric_score_percent"] = score
    else:
        out["validation_mistag_geometric_score_product"] = None
        out["validation_mistag_geometric_score_percent"] = None
    return out


def _count_mistag_uncertainty(counts, pair, eff):
    rows = (counts or {}).get(pair) or []
    for row in rows:
        if abs(float(row.get("signal_efficiency", -1.0)) - float(eff)) > 1.0e-6:
            continue
        total = row.get("background_total")
        passed = row.get("background_passed")
        if total is None or passed is None:
            return None
        total = int(total)
        passed = int(passed)
        if total <= 0:
            return None
        background_efficiency = passed / total
        variance = max(background_efficiency * (1.0 - background_efficiency), 0.0) / total
        return 100.0 * math.sqrt(variance)
    return None


def _working_point_uncertainty_metrics(counts):
    if not counts:
        return {}

    def combined_uncertainty(working_points):
        values = [
            _count_mistag_uncertainty(counts, pair, eff)
            for pair, eff in working_points
        ]
        values = [value for value in values if value is not None and math.isfinite(value)]
        if not values:
            return None, 0
        return math.sqrt(sum(value * value for value in values)) / len(values), len(values)

    working_point_uncertainty, working_point_count = combined_uncertainty(WORKING_POINT_DEFINITION)
    ctag_uncertainty, ctag_count = combined_uncertainty(CTAG_REFERENCE_WORKING_POINTS)
    return {
        "validation_working_point_mistag_percent_uncertainty": working_point_uncertainty,
        "validation_working_point_mistag_percent_uncertainty_points": working_point_count,
        "validation_ctag_reference_mistag_percent_uncertainty": ctag_uncertainty,
        "validation_ctag_reference_mistag_percent_uncertainty_points": ctag_count,
    }


def read_metrics(path):
    log_path = path / "train.log" if path.is_dir() else path
    if not log_path.is_file():
        return None
    text = log_path.read_text(errors="replace")
    loss_accuracy = re.findall(r"Eval AvgLoss: ([0-9.eE+-]+), AvgAcc: ([0-9.eE+-]+)", text)
    train_loss_accuracy = re.findall(r"Train AvgLoss: ([0-9.eE+-]+), AvgAcc: ([0-9.eE+-]+)", text)
    grad_norms = re.findall(r"Max Grad Norm: ([0-9.eE+-]+)", text)
    amp_skipped = re.findall(r"AMP skipped optimizer steps: (\d+)", text)
    cuda_memory = re.findall(r"Max CUDA memory: ([0-9.eE+-]+) MB", text)
    loaded_lrs = re.findall(r"Overrode loaded optimizer learning rate: [0-9.eE+-]+ -> ([0-9.eE+-]+)", text)
    auc = re.findall(r"roc_auc_score:\s*\n([0-9.eE+-]+)", text)
    if not loss_accuracy:
        return None
    loss, accuracy = loss_accuracy[-1]
    metrics = {
        "validation_loss": float(loss),
        "validation_accuracy": float(accuracy),
        "validation_auc": float(auc[-1]) if auc else None,
        "validation_shutdown_warning": "cannot schedule new futures after shutdown" in text,
    }
    if train_loss_accuracy:
        train_loss, train_accuracy = train_loss_accuracy[-1]
        metrics["train_loss"] = float(train_loss)
        metrics["train_accuracy"] = float(train_accuracy)
    if grad_norms:
        metrics["train_max_grad_norm"] = float(grad_norms[-1])
    if amp_skipped:
        metrics["train_amp_skipped_optimizer_steps"] = int(amp_skipped[-1])
    if cuda_memory:
        metrics["train_max_cuda_memory_mb"] = float(cuda_memory[-1])
    if loaded_lrs:
        metrics["train_loaded_optimizer_lr"] = float(loaded_lrs[-1])
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
        metrics.update(_working_point_metrics(curves))
    counts = _parse_bkg_rejection_at_eff_counts(text)
    if counts:
        metrics["validation_bkg_rejection_at_eff_counts"] = counts
        metrics.update(_working_point_uncertainty_metrics(counts))
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
