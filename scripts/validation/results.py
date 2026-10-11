"""Shared result checks; legacy callers retain their acceptance policies."""
import math
import hashlib
from collections.abc import Mapping
from pathlib import Path

from training.runtime import read_metrics, sha256


def finite_metric_ok(metrics, metric_name):
    """True only if `metric_name` is present and a finite (non-NaN/inf) number.

    A non-finite value (e.g. NaN from a zero-count fixed-WP ratio) must never
    reach ranking/exploit/controller decisions -- treat it the same as a
    missing metric: the worker is marked failed rather than silently
    poisoning the population's ranking with a NaN comparison.
    """
    if not isinstance(metrics, Mapping):
        return False
    value = metrics.get(metric_name)
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def require_evaluation_metrics(record, metric_name, label):
    """The required final-tier contract, also used before publication."""
    if not isinstance(record, dict):
        raise RuntimeError(f"Required final evaluation failed: {label}; invalid result record")
    if (record.get("status") != "completed" or record.get("returncode", 0) != 0
            or not finite_metric_ok(record.get("metrics"), metric_name)
            or isinstance(record['metrics'].get(metric_name), bool)):
        record.update(status="failed", error=f"Required final evaluation failed: {label}; invalid {metric_name} or process status")
        raise RuntimeError(record["error"])
    return record


def require_checkpoint_result(record, checkpoint, expected_sha256, label, metric_name):
    """Require finite primary evidence bound to the evaluated checkpoint.

    Older records may omit artifact paths, returncode and audit data. When those
    fields are present, they must agree; absent legacy fields are not invented.
    """
    require_evaluation_metrics(record, metric_name, label)
    checkpoint = Path(checkpoint)
    try:
        if sha256(checkpoint) != expected_sha256:
            raise ValueError("checkpoint changed")
        if record.get("checkpoint") and Path(record["checkpoint"]).resolve() != checkpoint.resolve():
            raise ValueError("evaluation names a different checkpoint")
        if record.get("checkpoint_sha256", expected_sha256) != expected_sha256:
            raise ValueError("evaluation checkpoint hash disagrees")
        if record.get("log"):
            logged = read_metrics(Path(record["log"]))
            if not finite_metric_ok(logged, metric_name) or float(logged[metric_name]) != float(record["metrics"][metric_name]):
                raise ValueError("evaluation log and recorded primary metric disagree")
        audit = record["metrics"].get("validation_data_audit")
        if audit is not None:
            if record.get("log") and logged.get("validation_data_audit") != audit:
                raise ValueError("evaluation log and recorded traversal disagree")
            consumed = audit.get("consumed") or {}
            if (audit.get("exhausted") is not True or not consumed.get("count", 0) > 0
                    or consumed.get("unique_ids") != consumed.get("count")
                    or consumed.get("repeated_ids") != 0):
                raise ValueError("incomplete or inconsistent validation traversal")
            for traversal in audit.get("traversal", []):
                if traversal.get("exhausted") is not True or traversal.get("wraps") != 0:
                    raise ValueError("inconsistent validation traversal artifact")
            traversals = audit.get("traversal") or []
            if traversals and all(isinstance(t.get('accepted'), Mapping) for t in traversals):
                if sum(t['accepted']['count'] for t in traversals) != consumed['count']:
                    raise ValueError("validation traversal and consumed row counts disagree")
        if record.get("prediction_path"):
            import awkward as ak
            import numpy as np
            predictions = ak.from_parquet(record["prediction_path"])
            scores = np.asarray(ak.to_numpy(predictions['scores']), dtype='<f4')
            if scores.ndim != 2 or not np.isfinite(scores).all():
                raise ValueError("invalid prediction scores")
            if audit:
                # Weaver hashes row-major float32 scores, not Parquet bytes.
                prediction_hash = hashlib.sha256(scores.tobytes()).hexdigest()
                if len(scores) != audit['consumed']['count'] or prediction_hash != audit.get("prediction_sha256"):
                    raise ValueError("prediction artifact hash or row count disagrees")
    except (OSError, ValueError, TypeError, AttributeError, KeyError, IndexError, OverflowError) as error:
        record.update(status="failed", error=f"Required final evaluation failed or checkpoint changed: {label}; {error}")
        raise RuntimeError(record["error"]) from error
    record["checkpoint_sha256"] = expected_sha256
    return record
