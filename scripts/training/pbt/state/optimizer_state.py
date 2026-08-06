#!/usr/bin/env python3
"""Optimizer-state diagnostics and safe resume transforms for PBT runs."""

import copy
import math
import os
import shutil
from collections import Counter
from pathlib import Path


try:
    import torch
except ImportError:  # pragma: no cover - reported at call sites with context.
    torch = None


OPTIMIZER_STATE_MODES = {"raw", "copy", "damped", "reset"}
MOMENTUM_KEYS = {"exp_avg", "momentum_buffer"}
SECOND_MOMENT_KEYS = {"exp_avg_sq", "max_exp_avg_sq"}


def _require_torch():
    if torch is None:
        raise RuntimeError("optimizer-state transforms require torch")
    return torch


def normalize_optimizer_state_mode(mode):
    normalized = str(mode or "raw").strip().lower()
    if normalized not in OPTIMIZER_STATE_MODES:
        raise ValueError(
            "initial_optimizer_mode must be one of: "
            + ", ".join(sorted(OPTIMIZER_STATE_MODES))
        )
    return normalized


def atomic_copy(source, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".pbt-tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def atomic_torch_save(payload, destination):
    _torch = _require_torch()
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".pbt-tmp")
    _torch.save(payload, temporary)
    os.replace(temporary, destination)


def load_optimizer_state(path):
    _torch = _require_torch()
    # weights_only=False: PyTorch 2.6 defaults to True, which real Weaver
    # optimizer checkpoints do not satisfy (confirmed against an actual
    # checkpoint -- see tests/test_pbt_anchor_copy_lr_recenter.py's
    # WeaverCheckpointFormatTest). These are locally-produced training
    # checkpoints, not untrusted third-party files.
    return _torch.load(path, map_location="cpu", weights_only=False)


def set_optimizer_state_lr(state, new_lr):
    """Rewrite every param_group's `lr` in-place to `new_lr`, mirroring what
    Weaver's own `--override-load-lr` does to an in-memory optimizer at
    resume time (weaver-core/weaver/train.py:806-831) -- except applied
    directly to the persisted checkpoint, so a copied optimizer.pt never
    holds a donor's stale LR at rest between the copy and the member's next
    training run. Flat-sets every group to the same value rather than
    Weaver's proportional rescale: this project's own optimizer setup
    (weaver-core train.py's decay/no-decay param groups) always starts
    every group at the identical LR, so the two are equivalent here, and a
    flat set is simpler. Handles both plain-float and tensor LR storage,
    same as Weaver's own override code.
    """
    for group in state.get("param_groups", []):
        current_lr = group.get("lr")
        if hasattr(current_lr, "fill_"):
            current_lr.fill_(float(new_lr))
        else:
            group["lr"] = float(new_lr)
    return state


def atomic_set_optimizer_lr(path, new_lr):
    state = load_optimizer_state(path)
    set_optimizer_state_lr(state, new_lr)
    atomic_torch_save(state, path)


def _zero_like_scalar_or_tensor(value):
    _torch = _require_torch()
    if _torch.is_tensor(value):
        return _torch.zeros_like(value)
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return 0
    if isinstance(value, float):
        return 0.0
    return value


def transform_optimizer_state(state, mode="raw", damping_factor=0.1):
    """Return a transformed copy of a torch optimizer state_dict.

    ``raw``/``copy`` preserve the checkpoint exactly.
    ``damped`` scales first-moment slots, keeping second moments and steps.
    ``reset`` clears first/second moments and restarts per-parameter steps.
    """

    _torch = _require_torch()
    normalized_mode = normalize_optimizer_state_mode(mode)
    damping_factor = float(damping_factor)
    if not 0.0 <= damping_factor <= 1.0:
        raise ValueError("initial_optimizer_damping must be in [0, 1]")

    transformed = copy.deepcopy(state)
    if normalized_mode in {"raw", "copy"}:
        return transformed

    for param_state in transformed.get("state", {}).values():
        if not isinstance(param_state, dict):
            continue
        for key, value in list(param_state.items()):
            if normalized_mode == "damped" and key in MOMENTUM_KEYS and _torch.is_tensor(value):
                value.mul_(damping_factor)
            elif normalized_mode == "reset" and key in MOMENTUM_KEYS | SECOND_MOMENT_KEYS:
                if _torch.is_tensor(value):
                    value.zero_()
                elif isinstance(value, (int, float)):
                    param_state[key] = type(value)(0)
            elif normalized_mode == "reset" and key == "step":
                param_state[key] = _zero_like_scalar_or_tensor(value)
    return transformed


def prepare_initial_optimizer(source, destination, *, mode="raw", damping_factor=0.1):
    normalized_mode = normalize_optimizer_state_mode(mode)
    if normalized_mode in {"raw", "copy"}:
        atomic_copy(source, destination)
        return {
            "mode": normalized_mode,
            "damping_factor": None,
            "transformed": False,
        }

    state = load_optimizer_state(source)
    transformed = transform_optimizer_state(
        state,
        mode=normalized_mode,
        damping_factor=damping_factor,
    )
    atomic_torch_save(transformed, destination)
    return {
        "mode": normalized_mode,
        "damping_factor": float(damping_factor),
        "transformed": True,
    }


def _tensor_norm(value):
    _torch = _require_torch()
    if value.numel() == 0:
        return 0.0
    return float(_torch.linalg.vector_norm(value.detach().float()).item())


def _step_value(value):
    _torch = _require_torch()
    if _torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return float(value.detach().float().max().item())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _stats(values):
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    finite.sort()
    return {
        "count": len(finite),
        "min": finite[0],
        "max": finite[-1],
        "mean": sum(finite) / len(finite),
        "median": finite[len(finite) // 2],
    }


def summarize_optimizer_state(state, *, top_k=10):
    _torch = _require_torch()
    key_counts = Counter()
    steps = []
    momentum_norms = []
    second_moment_norms = []
    adaptive_direction_norms = []
    top_adaptive = []
    tensor_slots = 0

    for param_id, param_state in state.get("state", {}).items():
        if not isinstance(param_state, dict):
            continue
        key_counts.update(param_state.keys())
        tensor_slots += sum(1 for value in param_state.values() if _torch.is_tensor(value))

        step = _step_value(param_state.get("step"))
        if step is not None:
            steps.append(step)

        exp_avg = param_state.get("exp_avg")
        exp_avg_sq = param_state.get("exp_avg_sq")
        if _torch.is_tensor(exp_avg):
            momentum_norms.append(_tensor_norm(exp_avg))
        if _torch.is_tensor(exp_avg_sq):
            second_moment_norms.append(_tensor_norm(exp_avg_sq.sqrt()))
        if _torch.is_tensor(exp_avg) and _torch.is_tensor(exp_avg_sq):
            direction = exp_avg.detach().float() / (exp_avg_sq.detach().float().sqrt() + 1.0e-12)
            norm = _tensor_norm(direction)
            adaptive_direction_norms.append(norm)
            top_adaptive.append(
                {
                    "param_id": str(param_id),
                    "shape": list(exp_avg.shape),
                    "adaptive_direction_norm": norm,
                    "momentum_norm": _tensor_norm(exp_avg),
                    "sqrt_second_moment_norm": _tensor_norm(exp_avg_sq.sqrt()),
                }
            )

    top_adaptive.sort(key=lambda item: item["adaptive_direction_norm"], reverse=True)
    top_k = max(0, int(top_k))
    return {
        "schema_version": 1,
        "param_states": len(state.get("state", {})),
        "param_groups": len(state.get("param_groups", [])),
        "tensor_slots": tensor_slots,
        "keys": dict(sorted(key_counts.items())),
        "steps": _stats(steps),
        "momentum_norm": _stats(momentum_norms),
        "sqrt_second_moment_norm": _stats(second_moment_norms),
        "adaptive_direction_norm": _stats(adaptive_direction_norms),
        "top_adaptive_directions": top_adaptive[:top_k],
    }
