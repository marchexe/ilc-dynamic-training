"""Verify the two-arm foundation pilot from recorded evidence; never train.

Numerical tolerances are declared here before inspecting the GPU result.
Requires exact consumed-ID sequences and exact evaluation prediction hashes.
"""
import argparse
import json
from pathlib import Path

import torch


def compare_tree(left, right, *, atol=1e-7, rtol=1e-6):
    differences = []
    tensor_count, max_abs = 0, 0.0

    def visit(a, b, path):
        nonlocal tensor_count, max_abs
        if torch.is_tensor(a):
            tensor_count += 1
            if not torch.is_tensor(b) or a.shape != b.shape or a.dtype != b.dtype:
                differences.append(path + ": shape/type")
                return
            difference = float((a.double() - b.double()).abs().max()) if a.numel() else 0.0
            max_abs = max(max_abs, difference)
            if not torch.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False):
                differences.append(path)
        elif isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                differences.append(path + ": keys")
                return
            for key in a:
                visit(a[key], b[key], f"{path}/{key}")
        elif isinstance(a, (tuple, list)):
            if not isinstance(b, (tuple, list)) or len(a) != len(b):
                differences.append(path + ": length")
                return
            for i, (x, y) in enumerate(zip(a, b)):
                visit(x, y, f"{path}/{i}")
        elif a != b:
            differences.append(path)

    visit(left, right, "root")
    return dict(matches=not differences, tensor_count=tensor_count,
                max_abs_difference=max_abs, mismatch_paths=differences[:20], atol=atol, rtol=rtol)


def verify(run):
    manifest = json.loads((run / "manifest.json").read_text())
    checks, details = [], []

    def check(name, ok):
        checks.append(dict(check=name, passed=bool(ok)))

    check("run completed", manifest["status"] == "completed")
    check("two epochs", len(manifest["generations"]) == 2)
    names = list(manifest["members"])
    check("two arms", len(names) == 2)
    if len(names) != 2 or manifest["status"] != "completed":
        return dict(ready=False, checks=checks)
    initial = manifest["initial_evaluation"]
    check("initial checkpoint actually evaluated", initial["status"] == "completed" and initial.get("checkpoint_sha256"))
    reference = initial["metrics"]["validation_data_audit"]

    def validation_check(label, audit):
        check(label + ": same dataset", audit["dataset"]["fingerprint"] == reference["dataset"]["fingerprint"])
        check(label + ": same ID order", audit["consumed"] == reference["consumed"])
        check(label + ": 150k distinct IDs", audit["consumed"]["count"] == audit["consumed"]["unique_ids"] == 150000)
        check(label + ": naturally exhausted", audit["exhausted"] and all(t["wraps"] == 0 for t in audit["traversal"]))

    validation_check("initial", reference)
    for generation in manifest["generations"]:
        epoch = generation["epoch"]
        workers = [generation["workers"][name] for name in names]
        metrics = [w["metrics"] for w in workers]
        audits = [m["train_data_audit"] for m in metrics]
        check(f"epoch {epoch}: same LR", workers[0]["lr"] == workers[1]["lr"])
        check(f"epoch {epoch}: same training sequence", audits[0]["consumed"] == audits[1]["consumed"])
        check(f"epoch {epoch}: same training dataset", audits[0]["dataset"] == audits[1]["dataset"])
        for name, audit, metric in zip(names, audits, metrics):
            scanned = sum(t["scanned"]["count"] for t in audit["traversal"])
            accepted = sum(t["accepted"]["count"] for t in audit["traversal"])
            check(f"epoch {epoch}/{name}: full source traversal", audit["exhausted"] and scanned == audit["dataset"]["total_rows"] == 2392232)
            check(f"epoch {epoch}/{name}: no source repeats", all(t["wraps"] == 0 and t["scanned"]["repeated_ids"] == 0 for t in audit["traversal"]))
            check(f"epoch {epoch}/{name}: all accepted rows consumed", accepted == audit["consumed"]["count"])
            check(f"epoch {epoch}/{name}: every optimizer step applied", audit["optimizer_steps"] == audit["batches"])
            validation_check(f"epoch {epoch}/{name}", metric["validation_data_audit"])
        scalar_keys = [k for k in metrics[0] if k.startswith("validation_") and isinstance(metrics[0][k], float)]
        check(f"epoch {epoch}: scalar evaluation metrics", all(abs(metrics[0][k] - metrics[1][k]) <= 1e-10 for k in scalar_keys))
        comparisons = {}
        for component in ("state", "optimizer", "scaler"):
            paths = [run / n / f"net_epoch-{epoch}_{component}.pt" for n in names]
            payloads = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
            comparisons[component] = compare_tree(*payloads)
            check(f"epoch {epoch}: matching {component}", comparisons[component]["matches"])
        details.append(dict(epoch=epoch, training=audits[0], validation=metrics[0]["validation_data_audit"], comparisons=comparisons))
    initial_epoch = manifest["config"]["shared"]["initial_epoch"]
    for component in ("state", "optimizer"):
        payloads = [torch.load(run / n / f"net_epoch-{initial_epoch}_{component}.pt", map_location="cpu", weights_only=False) for n in names]
        check("identical initial " + component, compare_tree(*payloads)["max_abs_difference"] == 0)
    initial_state = torch.load(run / names[0] / f"net_epoch-{initial_epoch}_state.pt", map_location="cpu", weights_only=False)
    final_state = torch.load(run / names[0] / f"net_epoch-{manifest['generations'][-1]['epoch']}_state.pt", map_location="cpu", weights_only=False)
    check("nonzero model update", compare_tree(initial_state, final_state)["max_abs_difference"] > 0)
    final = manifest["final_evaluations"]["control"]
    for name in [*names, "selected_best"]:
        record = final[name]
        check(name + ": standalone checkpoint hash", record["status"] == "completed" and record.get("checkpoint_sha256"))
        audit = record["metrics"]["validation_data_audit"]
        validation_check("standalone/" + name, audit)
        if name == "selected_best":
            best = manifest["best"]
            matched = manifest["generations"][best["generation"]]["workers"][best["member"]]["metrics"]
        else:
            matched = manifest["generations"][-1]["workers"][name]["metrics"]
        train_val = matched["validation_data_audit"]
        check(name + ": identical standalone predictions", train_val["prediction_sha256"] == audit["prediction_sha256"])
        check(name + ": standalone loss agrees", abs(train_val["loss"] - audit["loss"]) <= 1e-10)
    result = dict(ready=all(c["passed"] for c in checks), checks=checks, epochs=details,
                  initial=reference, final_evaluations=final)
    (run / "foundation_verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    result = verify(parser.parse_args().run)
    print(json.dumps({"ready": result["ready"], "checks": len(result["checks"]),
                      "failed": [c for c in result["checks"] if not c["passed"]]}, indent=2))
    raise SystemExit(0 if result["ready"] else 1)
