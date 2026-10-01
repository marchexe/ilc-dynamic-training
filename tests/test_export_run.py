import hashlib
import csv
import json
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.publish.export_run import export


class ExportRunTest(unittest.TestCase):
    @staticmethod
    def csv_rows(path):
        with path.open(newline="") as stream:
            return list(csv.DictReader(stream))

    def test_exports_selected_models_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            run.mkdir()
            member = run / "member_0"
            member.mkdir()
            protected = run / "checkpoints/protected/window-001"
            protected.mkdir(parents=True)
            checkpoints = run / "checkpoints"
            for path, value in (
                (checkpoints / "global_best_state.pt", b"best"),
                (protected / "net_epoch-2_state.pt", b"protected"),
                (protected / "net_epoch-2_optimizer.pt", b"optimizer"),
                (protected / "net_epoch-2_scaler.pt", b"scaler"),
                (member / "net_epoch-3_state.pt", b"final"),
            ):
                path.write_bytes(value)
            manifest = {
                "status": "completed", "method": "windowed_pbt_v2", "run": {"seed": 7},
                "best": {"state_path": str(checkpoints / "global_best_state.pt"), "epoch": 1,
                         "metric_value": 0.4},
                "protected_best": {"state_path": str(protected / "net_epoch-2_state.pt"),
                                   "optimizer_path": str(protected / "net_epoch-2_optimizer.pt"),
                                   "scaler_path": str(protected / "net_epoch-2_scaler.pt"),
                                   "epoch": 2, "single_epoch_metric": 0.5},
                "generations": [{"index": 0, "epoch": 3, "ranking": ["member_0"],
                                 "workers": {"member_0": {"metrics": {"validation_total_reference_mistag_geomean_percent": 0.6}}}}],
            }
            manifest_path = run / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            before = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            output = root / "published"
            docs = root / "docs"
            target = export(Namespace(run=str(run), slug="test-run", title="Test run", purpose="Test.",
                                      related_run=[], output_root=str(output), docs_dir=str(docs),
                                      resume_bundle="protected_best", release_tag="test-run",
                                      upload_release_assets=False))
            self.assertEqual(before, hashlib.sha256(manifest_path.read_bytes()).hexdigest())
            self.assertTrue((target / "models/best_single_state.pt").is_file())
            self.assertTrue((target / "models/protected_best_optimizer.pt").is_file())
            self.assertTrue((target / "models/final_best_state.pt").is_file())
            provenance = json.loads((target / "provenance.json").read_text())
            self.assertEqual(3, len(provenance["models"]))
            self.assertTrue((docs / "test-run.rst").is_file())
            commands = (target / "release-command.txt").read_text()
            self.assertIn("gh release create test-run", commands)
            self.assertIn("gh release upload test-run", commands)

    def test_exports_cadenced_histories_models_and_docs_deterministically(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "cadenced"
            run.mkdir()
            checkpoints = run / "checkpoints"
            checkpoints.mkdir()
            members = ["lr_3e-6", "lr_5_75e-6", "lr_8_5e-6", "lr_11_25e-6", "lr_14e-6"]
            starts = dict(zip(members, (3e-6, 5.75e-6, 8.5e-6, 11.25e-6, 14e-6)))
            for name, value in (
                ("global_best_state.pt", b"global-state"),
                ("global_best_optimizer.pt", b"global-optimizer"),
                ("global_best_scaler.pt", b"global-scaler"),
            ):
                (checkpoints / name).write_bytes(value)
            for member in members:
                directory = run / member
                directory.mkdir()
                (directory / "net_epoch-20_state.pt").write_bytes(f"final-{member}".encode())

            generations = []
            event_rows = []
            metrics_rows = []
            for generation in range(3):
                completed = generation + 1
                weaver = 18 + generation
                scores = {member: .4 + starts[member] * 100 + generation * .001 for member in members}
                ranking = sorted(members, key=scores.get)
                decision = {
                    "completed_epoch": completed, "copy_opportunity": generation == 1,
                    "lr_mutation_opportunity": generation == 1,
                    "warmup_active_during_training": generation < 2,
                    "terminal": generation == 2,
                    "reason": "metric_gap_exceeds_margin" if generation == 1 else
                              ("terminal_generation" if generation == 2 else "warmup"),
                    "donor": "lr_3e-6", "recipient": "lr_14e-6", "donor_lr": 3e-6,
                    "old_lr": 14e-6, "new_lr": 3.6e-6 if generation == 1 else 14e-6,
                    "mutation_factor": 1.2 if generation == 1 else None,
                    "mutation_applied": generation == 1, "mutation_reason": "mutated" if generation == 1 else "not_attempted",
                    "metric_gap": .01, "decision_margin": .002,
                }
                exploit = []
                if generation == 1:
                    exploit = [{
                        "event_id": "synthetic-exploit", "donor": "lr_3e-6", "recipient": "lr_14e-6",
                        "donor_lr": 3e-6, "recipient_lr": 14e-6, "new_lr": 3.6e-6,
                        "mutation_applied": True, "mutation_factor": 1.2, "mutation_reason": "mutated",
                        "metric_gap": .01, "decision_margin": .002, "rejected_mutations": [],
                    }]
                    event_rows.append({"event_type": "exploit", "generation": generation, "applied": True,
                                       "donor": "lr_3e-6", "recipient": "lr_14e-6", "new_lr": 3.6e-6,
                                       "mutation": 1.2})
                event_rows.append({
                    "event_type": "cadenced_pbt_decision", "generation": generation,
                    "completed_epoch": completed, "reason": decision["reason"],
                    "donor": decision["donor"], "recipient": decision["recipient"],
                    "terminal": decision["terminal"],
                })
                workers = {}
                for member in members:
                    lr = 3.6e-6 if generation == 2 and member == "lr_14e-6" else starts[member]
                    workers[member] = {"metrics": {"validation_total_reference_mistag_geomean_percent": scores[member]}}
                    metrics_rows.append({"generation": generation, "trial": member, "LR": lr,
                                         "optimization_metric_value": scores[member]})
                generations.append({"index": generation, "epoch": weaver, "ranking": ranking,
                                    "workers": workers, "cadenced_pbt_v1": decision, "exploit": exploit})
            manifest = {
                "status": "completed", "method": "cadenced_pbt_v1", "fingerprint": "config-fingerprint",
                "git": {"commit": "source-commit", "dirty": False},
                "run": {"seed": 12345, "git": {"commit": "source-commit", "dirty": False}},
                "members": {member: {"name": member} for member in members},
                "best": {"state_path": str(checkpoints / "global_best_state.pt"),
                         "optimizer_path": str(checkpoints / "global_best_optimizer.pt"),
                         "scaler_path": str(checkpoints / "global_best_scaler.pt"),
                         "epoch": 19, "generation": 1, "member": "lr_3e-6", "lr": 3e-6,
                         "metric": "validation_total_reference_mistag_geomean_percent", "metric_value": .401},
                "generations": generations,
            }
            (run / "manifest.json").write_text(json.dumps(manifest))
            (run / "summary.json").write_text(json.dumps({
                "best": manifest["best"],
                "final_best": {"generation": 2, "trial": generations[-1]["ranking"][0],
                               "LR": starts[generations[-1]["ranking"][0]],
                               "optimization_metric_name": "validation_total_reference_mistag_geomean_percent",
                               "optimization_metric_value": min(scores.values())},
                "scientific_endpoint": {"mean": .4},
                "checkpoint_selection": {"agrees_with_best_physics": True},
            }))
            (run / "resolved_config.yaml").write_text("schema_version: 1\n")
            with (run / "metrics.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(metrics_rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(metrics_rows)
            (run / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in event_rows))
            source_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in run.rglob("*") if path.is_file()}

            def fake_plot(_data, _manifest_or_path, path=None):
                target = path if path is not None else _manifest_or_path
                target.write_bytes(b"deterministic-png")

            output = root / "published"
            docs = root / "docs"
            args = Namespace(run=str(run), slug="cadenced-test", title="Cadenced test", purpose="Test.",
                             related_run=[], output_root=str(output), docs_dir=str(docs),
                             resume_bundle=None, release_tag="cadenced-test", upload_release_assets=False)
            with patch("scripts.publish.cadenced_pbt_v1.plot_performance", side_effect=fake_plot), \
                 patch("scripts.publish.cadenced_pbt_v1.plot_population_lrs", side_effect=fake_plot), \
                 patch("scripts.publish.cadenced_pbt_v1.plot_global_best_lineage", side_effect=fake_plot):
                target = export(args)
                first_hashes = {path.relative_to(target): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in target.rglob("*") if path.is_file() and "models" not in path.parts}
                export(args)
                second_hashes = {path.relative_to(target): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in target.rglob("*") if path.is_file() and "models" not in path.parts}
            self.assertEqual(first_hashes, second_hashes)
            self.assertEqual(source_hashes, {path: hashlib.sha256(path.read_bytes()).hexdigest()
                                             for path in source_hashes})
            self.assertEqual(15, len(self.csv_rows(target / "lr_history.csv")))
            self.assertEqual(3, len(self.csv_rows(target / "decision_history.csv")))
            self.assertEqual(1, len(self.csv_rows(target / "exploit_history.csv")))
            for name in ("global_best_state.pt", "global_best_optimizer.pt",
                         "global_best_scaler.pt", "final_best_state.pt"):
                self.assertTrue((target / "models" / name).is_file())
            summary = json.loads((target / "summary.json").read_text())
            self.assertEqual("cadenced_pbt_v1", summary["method"])
            self.assertEqual(1, summary["pbt_dynamics"]["exploit_count"])
            page = (docs / "cadenced-test.rst").read_text()
            self.assertIn("decision_history.csv", page)
            self.assertIn("03_global_best_lr_lineage.png", page)


if __name__ == "__main__":
    unittest.main()
