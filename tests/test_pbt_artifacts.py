import csv
import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import PROJECT_DIR  # noqa: F401
from training.pbt import strategy
from training.pbt.artifacts import append_event, run_contract, write_canonical_outputs


METRIC = "validation_working_point_mistag_percent"


def fixed_curve_metrics(metric_value, accuracy, auc, train_loss):
    return {
        METRIC: metric_value,
        "validation_accuracy": accuracy,
        "validation_auc": auc,
        "train_loss": train_loss,
        "validation_bkg_rejection_at_eff": {
            "efficiencies": [0.5, 0.8, 0.9],
            "pairs": {
                "bc": [800.0, 500.0, 40.0],
                "bd": [900.0, 700.0, 120.0],
                "cb": [250.0, 80.0, 20.0],
                "cd": [900.0, 120.0, 12.0],
            },
        },
    }


def synthetic_manifest(measured_baseline=True, configured_baseline=1.7, dataset_size=1000):
    shared = {
        "samples_per_epoch": 100,
        "epochs_per_generation": 1,
        "samples_per_epoch_val": 3000,
    }
    if dataset_size is not None:
        shared["training_dataset_size"] = dataset_size
    pbt_config = {
        "metric": METRIC,
        "mode": "min",
        "strategy": "anchored_lr_sweep",
        "baseline_metric_value": configured_baseline,
    }
    if measured_baseline:
        pbt_config["configured_baseline_metric_value"] = configured_baseline
        pbt_config["runtime_baseline_metric_value"] = 1.5
        pbt_config["baseline_metric_value"] = 1.5
    manifest = {
        "schema_version": 1,
        "experiment": "synthetic_publication_run",
        "status": "completed",
        "method": "anchored_lr_sweep",
        "run": {
            "method_name": "anchored_lr_sweep",
            "datasets": {"validation_dataset": "synthetic_proxy", "validation_suffix": "val5k_tail"},
            "schedule": {
                "training_interval": {"epochs_per_generation": 1, "samples_per_epoch": 100, "samples_per_trial_chunk": 100},
                "evaluation_interval": {"training_chunks": 1, "epochs": 1, "samples_per_trial": 100, "samples_per_epoch_val": 3000},
                "exploit_interval": {"enabled": True, "training_chunks": 1, "epochs": 1, "samples_per_trial": 100},
            },
        },
        "datasets": {"validation_dataset": "synthetic_proxy", "validation_suffix": "val5k_tail"},
        "checkpoint": {"path": "checkpoint.pt", "sha256": "abc123"},
        "initial_evaluation": {
            "status": "completed" if measured_baseline else "skipped",
            "checkpoint": "checkpoint.pt",
            "metrics": {METRIC: 1.5, "validation_accuracy": 0.86, "validation_auc": 0.96} if measured_baseline else {},
        },
        "config": {
            "shared": shared,
            "pbt": pbt_config,
        },
        "members": {
            "trial_a": {"name": "trial_a", "lr": 1.0e-4, "parent": None},
            "trial_b": {"name": "trial_b", "lr": 5.0e-5, "parent": "trial_a"},
        },
        "generations": [
            {
                "index": 0,
                "epoch": 18,
                "status": "completed",
                "workers": {
                    "trial_a": {"status": "completed", "lr": 1.0e-4, "metrics": fixed_curve_metrics(1.1, 0.88, 0.97, 0.32)},
                    "trial_b": {"status": "completed", "lr": 5.0e-5, "metrics": fixed_curve_metrics(1.3, 0.87, 0.965, 0.35)},
                },
                "ranking": ["trial_a", "trial_b"],
                "exploit": [],
            },
            {
                "index": 1,
                "epoch": 19,
                "status": "completed",
                "workers": {
                    "trial_a": {"status": "completed", "lr": 1.0e-4, "metrics": fixed_curve_metrics(1.0, 0.89, 0.972, 0.28)},
                    "trial_b": {"status": "completed", "lr": 7.5e-5, "metrics": fixed_curve_metrics(0.9, 0.9, 0.975, 0.27)},
                },
                "ranking": ["trial_b", "trial_a"],
                "exploit": [],
            },
        ],
        "best": {
            "generation": 1,
            "epoch": 19,
            "member": "trial_b",
            "metric": METRIC,
            "metric_value": 0.9,
            "lr": 7.5e-5,
            "state_path": "checkpoints/global_best_state.pt",
            "optimizer_path": "checkpoints/global_best_optimizer.pt",
            "metadata_path": "checkpoints/global_best_metadata.json",
        },
    }
    return manifest


class PBTArtifactsTest(unittest.TestCase):
    def test_run_contract_records_resolved_inputs_and_configured_intervals(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_DIR) as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            for suffix in ("trainTiny", "valTiny"):
                for flavor in ("bb", "cc", "dd"):
                    (dataset / f"sample_{flavor}_{suffix}.root").write_bytes(b"root")
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            config = {
                "shared": {
                    "checkpoint": str(checkpoint),
                    "dataset": str(dataset),
                    "data_extension": "root",
                    "train_suffix": "trainTiny",
                    "validation_suffix": "valTiny",
                    "seed": 123,
                    "epochs_per_generation": 2,
                    "samples_per_epoch": 50,
                    "samples_per_epoch_val": 30,
                },
                "pbt": {
                    "metric": METRIC,
                    "mode": "min",
                    "strategy": "exploit_mutate",
                    "exploit_fraction": 0.25,
                    "evaluation_interval_generations": 2,
                    "exploit_interval_generations": 3,
                },
                "slots": [{"gpu": "0", "label": "0"}],
            }

            contract = run_contract(config, ["python", "runner.py"], "local_weaver")

            self.assertEqual(contract["command"], ["python", "runner.py"])
            self.assertIn("commit", contract["git"])
            self.assertIn("dirty", contract["git"])
            self.assertEqual(contract["schedule"]["training_interval"]["samples_per_trial_chunk"], 100)
            self.assertEqual(contract["schedule"]["evaluation_interval"]["training_chunks"], 2)
            self.assertEqual(contract["schedule"]["exploit_interval"]["training_chunks"], 3)
            train_files = contract["datasets"]["resolved_files"]["train"]
            val_files = contract["datasets"]["resolved_files"]["val"]
            self.assertEqual(sum(len(row["files"]) for row in train_files), 3)
            self.assertEqual(sum(len(row["files"]) for row in val_files), 3)

    def test_canonical_outputs_from_synthetic_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            manifest = synthetic_manifest()
            append_event(
                run_dir,
                "exploit",
                {
                    "generation": 0,
                    "donor": "trial_a",
                    "recipient": "trial_b",
                    "donor_metric": 1.1,
                    "recipient_metric": 1.3,
                    "weight_source": "anchor",
                    "optimizer_source": "anchor",
                    "old_lr": 5.0e-5,
                    "new_lr": 7.5e-5,
                    "mutation": 0.75,
                },
            )

            artifacts = write_canonical_outputs(run_dir, manifest)

            for relative in (
                "manifest.json",
                "resolved_config.yaml",
                "events.jsonl",
                "metrics.csv",
                "summary.json",
                "report.md",
            ):
                self.assertTrue((run_dir / relative).is_file(), relative)
            for name in (
                "physics_metric_over_time.png",
                "learning_rate_over_time.png",
                "pbt_lineage.png",
                "best_model_progress.png",
                "final_summary.png",
                "report/physics_performance.png",
                "diagnostics/background_efficiency_curves.png",
                "report/btag_mistag_tables.csv",
                "report/ctag_mistag_tables.csv",
            ):
                self.assertTrue((run_dir / "plots" / name).is_file(), name)
            self.assertEqual(artifacts["report"], str(run_dir / "report.md"))

            with (run_dir / "metrics.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["trial"], "trial_a")
            self.assertEqual(rows[0]["training_chunk"], "0")
            self.assertEqual(rows[0]["samples_seen"], "100")
            self.assertEqual(rows[0]["epoch_fraction"], "0.1")
            self.assertEqual(rows[-1]["samples_seen"], "200")
            self.assertEqual(rows[-1]["epoch_fraction"], "0.2")
            self.assertEqual(rows[-1]["best_so_far"], "0.9")

            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary["winning_trial"], "trial_b")
            self.assertEqual(summary["baseline"]["kind"], "measured")
            self.assertEqual(summary["configured_baseline"]["kind"], "configured")
            self.assertEqual(summary["configured_baseline"]["metric_value"], 1.7)
            self.assertAlmostEqual(summary["best_improvement_vs_baseline"], (1.5 - 0.9) / 1.5)
            self.assertIn("physics_performance", summary["plots"])
            self.assertIn("anchored_lr_sweep", (run_dir / "report.md").read_text())

    def test_configured_baseline_is_not_measured_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            manifest = synthetic_manifest(measured_baseline=False, configured_baseline=1.5)

            write_canonical_outputs(run_dir, manifest)

            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertIsNone(summary["baseline"])
            self.assertEqual(summary["configured_baseline"]["metric_value"], 1.5)
            self.assertIsNone(summary["best_improvement_vs_baseline"])
            report = (run_dir / "report.md").read_text()
            self.assertIn("Measured baseline: n/a", report)
            self.assertIn("Configured baseline/reference: 1.5", report)

    def test_epoch_fraction_is_null_when_training_dataset_size_is_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            manifest = synthetic_manifest(dataset_size=None)

            write_canonical_outputs(run_dir, manifest)

            with (run_dir / "metrics.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["training_chunk"], "0")
            self.assertEqual(rows[0]["samples_seen"], "100")
            self.assertEqual(rows[0]["epoch_fraction"], "")

    def test_apply_exploit_records_structured_copy_and_lr_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("strong", "weak"):
                (root / name).mkdir()
            strong_state, strong_optimizer = strategy.checkpoint_paths(root / "strong", 0)
            weak_state, weak_optimizer = strategy.checkpoint_paths(root / "weak", 0)
            strong_state.write_bytes(b"strong-state")
            strong_optimizer.write_bytes(b"strong-optimizer")
            weak_state.write_bytes(b"weak-state")
            weak_optimizer.write_bytes(b"weak-optimizer")
            manifest_path = root / "manifest.json"
            manifest = {
                "config": {
                    "shared": {},
                    "pbt": {"metric": METRIC, "mode": "min"},
                },
                "members": {
                    "strong": {"lr": 1.0e-4, "parent": None},
                    "weak": {"lr": 1.5e-4, "parent": None},
                },
            }
            generation = {
                "index": 0,
                "epoch": 0,
                "workers": {
                    "strong": {"metrics": {METRIC: 0.9}},
                    "weak": {"metrics": {METRIC: 1.4}},
                },
                "exploit": [
                    {
                        "donor": "strong",
                        "recipient": "weak",
                        "recipient_lr": 1.5e-4,
                        "donor_lr": 1.0e-4,
                        "mutation_factor": 0.8,
                        "new_lr": 8.0e-5,
                        "applied": False,
                    }
                ],
            }

            strategy.apply_exploit(root, manifest, generation, manifest_path)

            self.assertEqual(weak_state.read_bytes(), b"strong-state")
            self.assertEqual(weak_optimizer.read_bytes(), b"strong-optimizer")
            events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
            self.assertEqual([event["event_type"] for event in events], ["exploit", "weight_copy", "optimizer_copy", "lr_change"])
            self.assertEqual(events[0]["donor_metric"], 0.9)
            self.assertEqual(events[0]["recipient_metric"], 1.4)
            self.assertEqual(events[-1]["old_lr"], 1.5e-4)
            self.assertEqual(events[-1]["new_lr"], 8.0e-5)


if __name__ == "__main__":
    unittest.main()
