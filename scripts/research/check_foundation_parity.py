"""Re-evaluate an immutable pilot checkpoint; keep evidence separate from the pilot.

The optional counter intervention isolates SequenceTrimmer warmup from weights,
data, and inference settings. It is diagnostic only, never a training option.
"""
import argparse
import copy
import json
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.pbt.execution.weaver_command import _test_mode_command, wrap_remote_command
from training.runtime import data_paths, read_metrics, sha256


def summarize(run):
    import hashlib
    import awkward as ak
    import numpy as np
    from verify_foundation_pilot import verify

    manifest = json.loads((run / "manifest.json").read_text())
    original = verify(run)
    # Preserve the historical failure report; corrected evidence is additive.
    historical_parity = {name + ": identical standalone predictions"
                         for name in ("identical_a", "identical_b", "selected_best")}
    checks = [c for c in original["checks"] if c["check"] not in historical_parity]

    def check(name, passed):
        checks.append(dict(check=name, passed=bool(passed)))

    reference = manifest["generations"][-1]["workers"]["identical_a"]["metrics"]
    old_standalone = manifest["final_evaluations"]["control"]["identical_a"]
    arrays, audits, before_metric_differences = {}, {}, {}
    metric_keys = [k for k in reference if k.startswith("validation_")
                   and k not in ("validation_data_audit", "validation_shutdown_warning")]
    for label in ("before_cold", "before_warm", "after_cold", "after_warm"):
        directory = run / "parity" / label
        result = json.loads((directory / "result.json").read_text())
        audit = result["metrics"]["validation_data_audit"]
        audits[label] = audit
        check(label + ": successful immutable checkpoint evaluation",
              result["returncode"] == 0 and result["checkpoint_sha256"] ==
              old_standalone["checkpoint_sha256"] == sha256(Path(result["checkpoint"])))
        check(label + ": same 150k IDs and dataset",
              audit["consumed"] == reference["validation_data_audit"]["consumed"]
              and audit["dataset"] == reference["validation_data_audit"]["dataset"]
              and audit["consumed"]["count"] == audit["consumed"]["unique_ids"] == 150000)
        check(label + ": full finite validation", audit["exhausted"] and
              all(t["wraps"] == 0 for t in audit["traversal"]))
        expected_metrics = old_standalone["metrics"] if label == "before_cold" else reference
        check(label + ": identical recorded loss and all evaluation metrics",
              all(result["metrics"].get(k) == expected_metrics[k] for k in metric_keys))
        if label == "before_cold":
            before_metric_differences = {k: dict(training=reference[k], standalone=result["metrics"].get(k))
                                         for k in metric_keys if result["metrics"].get(k) != reference[k]}
        source = (run / "parity/original_network.py" if label.startswith("before") else
                  Path(__file__).resolve().parents[2] / "networks/pretrained_sgv_particle_transformer.py")
        check(label + ": expected code and counter intervention",
              result["network_source_sha256"] == sha256(source) == sha256(directory / "network_source.py")
              and result["counter"] == (5 if label.endswith("warm") else None))
        scores = ak.to_numpy(ak.from_parquet(directory / "predictions.parquet")["scores"])
        arrays[label] = scores
        check(label + ": persisted predictions match audit", scores.shape == (150000, 3)
              and hashlib.sha256(np.asarray(scores, dtype="<f4").tobytes()).hexdigest()
              == audit["prediction_sha256"])
        expected = (old_standalone["metrics"]["validation_data_audit"] if label == "before_cold"
                    else reference["validation_data_audit"])
        check(label + ": reproduces expected prediction hash",
              audit["prediction_sha256"] == expected["prediction_sha256"])
    check("fix removes counter dependence exactly",
          np.array_equal(arrays["after_cold"], arrays["after_warm"]))
    check("fix preserves warmed training-time predictions exactly",
          np.array_equal(arrays["after_cold"], arrays["before_warm"]))
    difference = np.abs(arrays["before_cold"] - arrays["before_warm"])
    changed = np.flatnonzero(np.any(difference != 0, axis=1))
    result = dict(ready=all(c["passed"] for c in checks), checks=checks,
                  original_pilot_checks=original["checks"], metric_keys=metric_keys,
                  before_metric_differences=before_metric_differences,
                  audits=audits, before_max_abs=float(difference.max()),
                  before_changed_events=len(changed), before_last_changed_event=int(changed[-1]),
                  after_max_abs=float(np.max(np.abs(arrays["after_cold"] - arrays["after_warm"]))),
                  scope="Re-evaluation of saved epoch-19 checkpoint; original two-epoch training artifacts reverified, not retrained.")
    (run / "parity" / "verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(ready=result["ready"], checks=len(checks),
                         failed=[c for c in checks if not c["passed"]],
                         before_max_abs=result["before_max_abs"], after_max_abs=result["after_max_abs"]), indent=2))
    return 0 if result["ready"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--label")
    parser.add_argument("--network-source", type=Path)
    parser.add_argument("--counter", type=int)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--host", default="iutgpu01")
    args = parser.parse_args()
    run = args.run.resolve()
    if args.summarize:
        return summarize(run)
    if not args.label or args.network_source is None:
        parser.error("Evaluation requires --label and --network-source")
    manifest = json.loads((run / "manifest.json").read_text())
    shared = copy.deepcopy(manifest["config"]["shared"])
    checkpoint = run / "identical_a/net_epoch-19_state.pt"
    output = run / "parity" / args.label
    output.mkdir(parents=True, exist_ok=False)
    source = args.network_source.resolve()
    snapshot = output / "network_source.py"
    snapshot.write_bytes(source.read_bytes())
    wrapper = output / "network.py"
    wrapper.write_text(
        f"exec(compile(open({str(snapshot)!r}).read(), {str(snapshot)!r}, 'exec'))\n"
        + ("_original_get_model = get_model\n"
           "def get_model(*args, **kwargs):\n"
           "    model, info = _original_get_model(*args, **kwargs)\n"
           "    for module in model.modules():\n"
           "        if isinstance(module, SequenceTrimmer):\n"
           f"            module._counter = {args.counter}\n"
           "    return model, info\n" if args.counter is not None else "")
    )
    shared["network_config"] = str(wrapper)
    paths = data_paths(shared["dataset"], shared.get("data_extension", "root"),
                       validation_dataset=shared.get("validation_dataset"),
                       validation_suffix=shared["validation_suffix"])["val"]
    log = output / "evaluation.log"
    command = _test_mode_command(shared, args.gpu, paths, checkpoint, log)
    command += ["--predict-output", str(output / "predictions.parquet")]
    command = wrap_remote_command(command, dict(host=args.host, gpu=args.gpu))
    record = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                  network_source_sha256=sha256(snapshot), counter=args.counter, command=command)
    with (output / "console.log").open("w") as stream:
        record["returncode"] = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT).returncode
    assert sha256(checkpoint) == record["checkpoint_sha256"], "Checkpoint changed"
    record["metrics"] = read_metrics(log)
    (output / "result.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(label=args.label, returncode=record["returncode"],
                         audit=record["metrics"].get("validation_data_audit")), indent=2))
    return record["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
