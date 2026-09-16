#!/usr/bin/env python3
"""Read-only evidence verification for full-epoch fixed-LR studies.

Arms, horizons, datasets and initialization hashes come from the run manifest.
Requires raw optimizer continuation and one epoch per generation: the manifest
only retains the last epoch's audit for multi-epoch generations. Never launches
training/inference or writes into the run. --through checks an ongoing prefix;
final standalone parity is required when checking the whole completed run.
"""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.pbt.state.checkpointing import checkpoint_paths, epoch_for_generation
from training.runtime import sha256


def verify(run, through=None):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    config = manifest["config"]
    shared, pbt = config["shared"], config["pbt"]
    arms = {m["name"]: m["start_lr"] for m in config["population"]}
    limit = shared["generations"] if through is None else through
    if not 1 <= limit <= shared["generations"]:
        raise ValueError("--through must be within the configured generation range")
    failures, results = [], []

    def check(label, condition):
        if not condition:
            failures.append(label)

    def finite(value):
        return isinstance(value, (int, float)) and math.isfinite(value)

    def saved(label, path, digest=None):
        check(label, path.is_file() and path.stat().st_size > 0
              and (digest is None or sha256(path) == digest))

    check("nonempty unique member names", bool(arms) and len(arms) == len(config["population"]))
    check("full-epoch audited deterministic configuration", shared.get("weaver_epochs_per_generation") == 1
          and shared.get("samples_per_epoch") is None and shared.get("samples_per_epoch_val") is None
          and shared.get("deterministic") and shared.get("data_audit"))
    check("raw optimizer continuation", shared.get("initial_optimizer_mode") == "raw")
    check("adaptive mechanisms disabled", pbt.get("strategy") == "fixed_lr_grid"
          and pbt.get("rollback_fraction") == 0 and pbt.get("baseline_guard_action") == "observe"
          and pbt.get("early_stop_degraded_generations") == 0
          and (pbt.get("dynamic_controller") or {}).get("mode", "disabled") == "disabled"
          and all(not pbt.get(k) for k in ["anchor_copy_lr_recenter", "lr_radius", "lr_controller", "population_lr_policy"])
          and not shared.get("training_controller") and shared.get("lr_scheduler") == "none")
    generations = manifest.get("generations", [])
    if through is None:
        check("run completed", manifest.get("status") == "completed" and len(generations) == limit)
    initial = manifest.get("initial_evaluation") or {}
    resume = manifest.get("initial_resume") or {}
    reference = (initial.get("metrics") or {}).get("validation_data_audit", {})
    check("initial evaluation", initial.get("status") == "completed"
          and bool(resume.get("state_sha256")) and initial.get("checkpoint_sha256") == resume.get("state_sha256"))
    metric_keys = {"validation_loss", pbt["metric"]}
    metric_keys.update(k for k in initial.get("metrics", {}) if "_mistag_eff_" in k)

    def dataset(label, data, split):
        files = data.get("files", [])
        expected = {f for group in manifest["datasets"]["resolved_files"][split] for f in group["files"]}
        check(label + ": dataset evidence", bool(files) and bool(data.get("fingerprint"))
              and {f["path"] for f in files} == expected
              and data.get("total_rows", 0) > 0
              and sum(f["rows"] for f in files) == data.get("total_rows"))

    dataset("validation", reference.get("dataset", {}), "val")

    def traversal(label, audit):
        items, consumed = audit.get("traversal", []), audit.get("consumed", {})
        check(label + ": full finite traversal", audit.get("exhausted") and bool(items)
              and sum(t.get("scanned", {}).get("count", 0) for t in items) == audit.get("dataset", {}).get("total_rows")
              and all(t.get("exhausted") and t.get("wraps") == 0
                      and t.get("scanned", {}).get("repeated_ids") == 0 for t in items))
        accepted = sum(t.get("accepted", {}).get("count", 0) for t in items)
        check(label + ": all accepted consumed once", accepted > 0
              and accepted == consumed.get("count") == consumed.get("unique_ids")
              and consumed.get("repeated_ids") == 0
              and bool(consumed.get("sequence_sha256")) and bool(consumed.get("id_set_sha256")))

    def validation(label, audit):
        traversal(label, audit)
        check(label + ": matched validation", audit.get("dataset") == reference.get("dataset")
              and audit.get("consumed") == reference.get("consumed")
              and audit.get("consumed", {}).get("count") == reference.get("dataset", {}).get("total_rows"))

    validation("initial", reference)
    initial_epoch = shared["initial_epoch"]
    check("initial checkpoint epoch", resume.get("epoch") == initial_epoch)
    for name in arms:
        for component in ("state", "optimizer"):
            digest = resume.get(component + "_sha256")
            check(name + ": recorded initial " + component, bool(digest))
            saved(name + ": raw initial " + component, run / name / f"net_epoch-{initial_epoch}_{component}.pt", digest)
        # Legacy checkpoints have no scaler. When supplied, its state must copy too.
        source_scaler = Path(shared["initial_optimizer"].replace("_optimizer.pt", "_scaler.pt"))
        if source_scaler.is_file():
            saved(name + ": initial scaler", run / name / f"net_epoch-{initial_epoch}_scaler.pt", sha256(source_scaler))
    train_reference = None
    for index in range(limit):
        matches = [g for g in generations if g.get("index") == index]
        check(f"generation {index}: unique completed record", len(matches) == 1 and matches[0].get("status") == "completed")
        if not matches:
            continue
        generation = matches[0]
        epoch = epoch_for_generation(config, index)
        check(f"generation {index}: checkpoint epoch", generation.get("epoch") == epoch)
        check(f"generation {index}: no adaptive actions", generation.get("exploit") == []
              and not any(generation.get(k) for k in ["controller_actions", "controller_lr_changes",
                                                     "anchor_copy_lr_recenter", "early_stop_triggered"]))
        sequences = []
        check(f"generation {index}: configured workers", set(generation.get("workers", {})) == set(arms))
        for name, lr in arms.items():
            worker = generation.get("workers", {}).get(name, {})
            metrics = worker.get("metrics") or {}
            train, val = metrics.get("train_data_audit", {}), metrics.get("validation_data_audit", {})
            label = f"epoch {epoch}/{name}"
            check(label + ": worker/LR", worker.get("status") == "completed" and worker.get("returncode") == 0
                  and worker.get("lr") == metrics.get("train_loaded_optimizer_lr") == lr)
            if train_reference is None:
                train_reference = train.get("dataset", {})
                dataset("training", train_reference, "train")
            check(label + ": matched training dataset", train.get("dataset") == train_reference)
            traversal(label, train)
            check(label + ": optimizer steps", 0 < train.get("optimizer_steps", 0) <= train.get("batches", 0))
            sequences.append(json.dumps([train.get("consumed"), train.get("traversal")], sort_keys=True))
            validation(label + "/validation", val)
            for key in metric_keys:
                check(label + ": finite " + key, finite(metrics.get(key)))
            state, optimizer = checkpoint_paths(run / name, epoch)
            for path in (state, optimizer):
                saved(label + ": retained " + path.name, path)
            if shared.get("use_amp") and shared.get("amp_dtype") == "fp16":
                saved(label + ": retained scaler", run / name / f"net_epoch-{epoch}_scaler.pt")
            if index == limit - 1:
                results.append(dict(name=name, lr=lr, full_epoch=index + 1, checkpoint_epoch=epoch,
                                    loss=metrics.get("validation_loss"), metric=metrics.get(pbt["metric"])))
        check(f"generation {index}: matched training sequences", len(set(sequences)) == 1)
    if through is None:
        final = manifest.get("final_evaluations", {}).get("control", {})
        for name in [*arms, "selected_best"]:
            record = final.get(name, {})
            best = manifest.get("best") or {}
            path = Path(best.get("state_path", "")) if name == "selected_best" else checkpoint_paths(
                run / name, epoch_for_generation(config, limit - 1))[0]
            training = best.get("metrics", {}) if name == "selected_best" else next(
                (g.get("workers", {}).get(name, {}).get("metrics", {}) for g in generations if g.get("index") == limit - 1), {})
            check(name + ": final evaluation completed", record.get("status") == "completed" and bool(record.get("checkpoint_sha256")))
            saved(name + ": final checkpoint hash", path, record.get("checkpoint_sha256"))
            metrics = record.get("metrics") or {}
            audit = metrics.get("validation_data_audit", {})
            validation(name + "/standalone", audit)
            prediction = training.get("validation_data_audit", {}).get("prediction_sha256")
            check(name + ": checkpoint prediction parity", bool(prediction) and prediction == audit.get("prediction_sha256"))
            for key in metric_keys:
                check(name + ": standalone " + key, finite(metrics.get(key)) and metrics.get(key) == training.get(key))
    return dict(passed=not failures, failures=failures, through_full_epoch=limit,
                run_status=manifest.get("status"), results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--through", type=int, metavar="EPOCH")
    args = parser.parse_args()
    try:
        result = verify(args.run.resolve(), args.through)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = dict(passed=False, failures=[f"Missing/malformed run evidence: {error}"])
    print(json.dumps(result, indent=2, allow_nan=False))
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
