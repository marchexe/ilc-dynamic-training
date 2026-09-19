"""Shared result checks; legacy callers retain their acceptance policies."""
import math

from training.runtime import sha256


def finite_metric_ok(metrics, metric_name):
    """True only if `metric_name` is present and a finite (non-NaN/inf) number.

    A non-finite value (e.g. NaN from a zero-count fixed-WP ratio) must never
    reach ranking/exploit/controller decisions -- treat it the same as a
    missing metric: the worker is marked failed rather than silently
    poisoning the population's ranking with a NaN comparison.
    """
    if metrics is None:
        return False
    value = metrics.get(metric_name)
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def require_checkpoint_result(record, checkpoint, expected_sha256, label):
    """Preserve the final-evaluation status and checkpoint-identity contract."""
    if record.get("status") != "completed" or sha256(checkpoint) != expected_sha256:
        raise RuntimeError(f"Required final evaluation failed or checkpoint changed: {label}")
    record["checkpoint_sha256"] = expected_sha256
    return record
