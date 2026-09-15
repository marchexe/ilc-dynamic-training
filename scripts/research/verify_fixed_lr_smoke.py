#!/usr/bin/env python3
"""Read-only verification of the five-arm, one-full-epoch production smoke.

Prints JSON and exits nonzero for incomplete or inconsistent evidence. Never
launches inference/training or writes into the run. Different-LR predictions
are expected to differ; training/validation input sequences must match.
"""

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "runs/pbt/foundation_fixed_lr_smoke_20260916/study"
ARMS = dict(zip(["lr_3e-6", "lr_5_75e-6", "lr_8_5e-6", "lr_11_25e-6", "lr_14e-6"],
                [3e-6, 5.75e-6, 8.5e-6, 11.25e-6, 1.4e-5]))
INITIAL_HASHES = {
    "state": "ae4928aa088b73538597f23b78b51678298e59d23552c9cd2c2e849fb3ced501",
    "optimizer": "a1909dd493b3ed07bea41d044fa6cb6862180709a33ad4cd691a094754519f09",
}
METRIC = "validation_total_reference_mistag_geomean_percent"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value):
    return isinstance(value, (float, int)) and math.isfinite(value)


def elapsed(record):
    try:
        return (datetime.fromisoformat(record["finished_at"]) -
                datetime.fromisoformat(record["started_at"])).total_seconds()
    except (KeyError, TypeError, ValueError):
        return None


def verify(run):
    manifest = json.loads((run / "manifest.json").read_text())
    failures, arms = [], []

    def check(label, condition):
        if not condition:
            failures.append(label)

    config = manifest.get("config", {})
    shared, pbt = config.get("shared", {}), config.get("pbt", {})
    check("run completed", manifest.get("status") == "completed")
    check("exact five LR arms", {m["name"]: m["start_lr"] for m in config.get("population", [])} == ARMS)
    check("raw epoch-17 initialization", shared.get("initial_optimizer_mode") == "raw" and shared.get("initial_epoch") == 17)
    check("one full generation", shared.get("generations") == shared.get("weaver_epochs_per_generation") == 1
          and shared.get("samples_per_epoch") is None and shared.get("samples_per_epoch_val") is None)
    check("deterministic audited validation", shared.get("deterministic") and shared.get("data_audit"))
    check("production training settings", shared.get("batch_size") == 1024 and shared.get("optimizer") == "ranger"
          and shared.get("freeze_batch_norm") and shared.get("use_amp") and shared.get("amp_dtype") == "fp16"
          and shared.get("num_workers") == 1 and shared.get("seed") == 12345
          and shared.get("fetch_step") == 0.01 and shared.get("prefetch_factor") == 4)
    check("adaptive mechanisms disabled", pbt.get("strategy") == "fixed_lr_grid"
          and pbt.get("rollback_fraction") == 0 and pbt.get("baseline_guard_action") == "observe"
          and pbt.get("early_stop_degraded_generations") == 0
          and (pbt.get("dynamic_controller") or {}).get("mode") == "disabled"
          and all(not pbt.get(k) for k in ["anchor_copy_lr_recenter", "lr_radius", "lr_controller", "population_lr_policy"])
          and not shared.get("training_controller") and shared.get("lr_scheduler") == "none")
    generations = manifest.get("generations", [])
    check("exactly one completed epoch-18 generation", len(generations) == 1
          and generations[0].get("epoch") == 18 and generations[0].get("status") == "completed")
    for generation in generations:
        check("no adaptive actions", generation.get("exploit") == []
              and not any(generation.get(k) for k in ["controller_actions", "controller_lr_changes",
                                                     "anchor_copy_lr_recenter", "early_stop_triggered"]))
    generation = generations[0] if generations else {}
    initial = manifest.get("initial_evaluation", {})
    check("initial standalone evaluation completed", initial.get("status") == "completed"
          and initial.get("checkpoint_sha256") == INITIAL_HASHES["state"])
    reference = initial.get("metrics", {}).get("validation_data_audit", {})

    def validation(label, audit):
        consumed = audit.get("consumed", {})
        check(label + ": 150k distinct events", consumed.get("count") == consumed.get("unique_ids") == 150000
              and consumed.get("repeated_ids") == 0)
        check(label + ": identical validation fingerprint", bool(consumed.get("sequence_sha256"))
              and consumed == reference.get("consumed")
              and bool(audit.get("dataset", {}).get("fingerprint"))
              and audit.get("dataset", {}).get("fingerprint") == reference.get("dataset", {}).get("fingerprint"))
        traversal = audit.get("traversal", [])
        check(label + ": finite no-wrap validation", audit.get("exhausted") and bool(traversal)
              and all(t.get("wraps") == 0 for t in traversal))

    validation("initial", reference)
    final = manifest.get("final_evaluations", {}).get("control", {})
    training_fingerprints = []
    for name, lr in ARMS.items():
        worker = generation.get("workers", {}).get(name, {})
        metrics = worker.get("metrics") or {}
        train, val = metrics.get("train_data_audit", {}), metrics.get("validation_data_audit", {})
        consumed = train.get("consumed", {})
        traversal = train.get("traversal", [])
        scanned = sum(t.get("scanned", {}).get("count", 0) for t in traversal)
        accepted = sum(t.get("accepted", {}).get("count", 0) for t in traversal)
        wraps = sum(t.get("wraps", 0) for t in traversal)
        checkpoint = run / name / "net_epoch-18_state.pt"
        check(name + ": completed", worker.get("status") == "completed" and worker.get("returncode") == 0)
        check(name + ": complete 2392232-row traversal", train.get("exhausted") and bool(traversal)
              and scanned == train.get("dataset", {}).get("total_rows") == 2392232)
        check(name + ": no wrapping or repeated source rows", bool(traversal)
              and all(t.get("wraps") == 0 and t.get("scanned", {}).get("repeated_ids") == 0 for t in traversal))
        check(name + ": all accepted rows consumed once", accepted > 0
              and accepted == consumed.get("count") == consumed.get("unique_ids") and consumed.get("repeated_ids") == 0)
        check(name + ": optimizer steps recorded", isinstance(train.get("optimizer_steps"), int)
              and 0 < train["optimizer_steps"] <= train.get("batches", 0))
        check(name + ": intended LR actually loaded", worker.get("lr") == metrics.get("train_loaded_optimizer_lr") == lr
              and manifest.get("members", {}).get(name, {}).get("lr") == lr)
        for component, expected in INITIAL_HASHES.items():
            path = run / name / f"net_epoch-17_{component}.pt"
            check(name + ": untouched initial " + component, path.is_file() and sha256(path) == expected)
        for component in ["state", "optimizer", "scaler"]:
            path = run / name / f"net_epoch-18_{component}.pt"
            check(name + ": saved " + component, path.is_file() and path.stat().st_size > 0)
        for key in ["validation_loss", METRIC]:
            check(name + ": finite " + key, finite(metrics.get(key)))
        training_fingerprints.append((consumed.get("sequence_sha256"), consumed.get("id_set_sha256"),
                                      train.get("dataset", {}).get("fingerprint")))
        validation(name, val)
        standalone = final.get(name, {})
        standalone_metrics = standalone.get("metrics") or {}
        standalone_audit = standalone_metrics.get("validation_data_audit", {})
        check(name + ": final standalone checkpoint", standalone.get("status") == "completed"
              and checkpoint.is_file() and standalone.get("checkpoint_sha256") == sha256(checkpoint))
        validation(name + "/standalone", standalone_audit)
        check(name + ": checkpoint prediction parity", bool(val.get("prediction_sha256"))
              and val.get("prediction_sha256") == standalone_audit.get("prediction_sha256"))
        for key in ["validation_loss", METRIC]:
            check(name + ": standalone " + key, finite(standalone_metrics.get(key))
                  and standalone_metrics.get(key) == metrics.get(key))
        arms.append(dict(name=name, status=worker.get("status", "missing"), source_rows=scanned,
                         accepted_rows=accepted, consumed_rows=consumed.get("count"),
                         optimizer_steps=train.get("optimizer_steps"), amp_skipped_steps=metrics.get("train_amp_skipped_optimizer_steps"),
                         wraps=wraps if traversal else None, training_sequence=consumed.get("sequence_sha256"),
                         validation_events=val.get("consumed", {}).get("count"),
                         validation_sequence=val.get("consumed", {}).get("sequence_sha256"),
                         lr_used=metrics.get("train_loaded_optimizer_lr") if finite(metrics.get("train_loaded_optimizer_lr")) else None,
                         checkpoint=str(checkpoint),
                         validation_loss=metrics.get("validation_loss") if finite(metrics.get("validation_loss")) else None,
                         mistag=metrics.get(METRIC) if finite(metrics.get(METRIC)) else None,
                         started_at=worker.get("started_at"), finished_at=worker.get("finished_at"),
                         elapsed_seconds=elapsed(worker)))
    check("identical training sequences and dataset across all five arms", all(all(v) for v in training_fingerprints)
          and len(set(training_fingerprints)) == 1)
    selected = final.get("selected_best", {})
    best_path = Path((manifest.get("best") or {}).get("state_path", ""))
    check("selected-best standalone checkpoint", selected.get("status") == "completed"
          and best_path.is_file() and selected.get("checkpoint_sha256") == sha256(best_path))
    validation("selected_best", (selected.get("metrics") or {}).get("validation_data_audit", {}))
    for key in ["validation_loss", METRIC]:
        check("selected_best: finite " + key, finite((selected.get("metrics") or {}).get(key)))
    return dict(passed=not failures, failures=failures, run_status=manifest.get("status"), arms=arms,
                started_at=manifest.get("started_at"), finished_at=manifest.get("finished_at"),
                elapsed_seconds=elapsed(manifest))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?", default=DEFAULT_RUN)
    args = parser.parse_args()
    try:
        result = verify(args.run.resolve())
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = dict(passed=False, failures=[f"Missing/malformed smoke evidence: {error}"])
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
