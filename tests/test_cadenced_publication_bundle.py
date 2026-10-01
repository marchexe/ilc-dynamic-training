import csv
import json
from pathlib import Path
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
BUNDLE = PROJECT_DIR / "published/experiments/cadenced-pbt-v1-50"


def csv_rows(name):
    with (BUNDLE / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


class CadencedPublicationBundleTest(unittest.TestCase):
    def test_bundle_is_complete_and_records_registered_results(self):
        required = {
            "README.md", "config.yaml", "metrics.csv", "lr_history.csv",
            "decision_history.csv", "exploit_history.csv", "summary.json",
            "provenance.json", "release-command.txt",
            "plots/01_performance_progression.png",
            "plots/02_population_lr_trajectories.png",
            "plots/03_global_best_lr_lineage.png",
            "plots/physics_performance.png",
            "plots/background_efficiency_curves.png",
        }
        self.assertEqual([], sorted(path for path in required if not (BUNDLE / path).is_file()))
        summary = json.loads((BUNDLE / "summary.json").read_text())
        self.assertEqual("Cadenced PBT v1 — 50 epochs", summary["title"])
        self.assertEqual("cadenced_pbt_v1", summary["method"])
        self.assertEqual(50, summary["horizon"])
        self.assertEqual(0.3323913294200741, summary["final10_current_best_mean"])
        self.assertEqual(0.3309306510368319, summary["best"]["metric_value"])
        self.assertEqual(42, summary["global_best_completed_epoch"])
        self.assertEqual("lr_5_75e-6", summary["best"]["member"])
        self.assertEqual(1.9200000000000003e-05, summary["best"]["lr"])
        self.assertEqual("lr_8_5e-6", summary["final_best"]["member"])

    def test_histories_are_complete_and_consistent(self):
        lr_rows = csv_rows("lr_history.csv")
        decisions = csv_rows("decision_history.csv")
        exploits = csv_rows("exploit_history.csv")
        self.assertEqual(250, len(lr_rows))
        self.assertEqual(250, len({(row["completed_epoch"], row["member"]) for row in lr_rows}))
        self.assertEqual(set(range(1, 51)), {int(row["completed_epoch"]) for row in lr_rows})
        self.assertTrue(all(len({row["lr_training"] for row in lr_rows
                                 if int(row["completed_epoch"]) == epoch}) == 5
                            for epoch in range(1, 51)))
        self.assertEqual(50, len(decisions))
        self.assertEqual(32, len(exploits))
        self.assertEqual(28, sum(row["mutation_type"] == "copy_plus_mutation" for row in exploits))
        self.assertEqual(4, sum(row["copy_only"] == "true" for row in exploits))
        self.assertEqual(17, sum(row["mutation_factor"] == "0.8" for row in exploits))
        self.assertEqual(11, sum(row["mutation_factor"] == "1.2" for row in exploits))
        self.assertEqual({20, 25, 26, 39},
                         {int(row["completed_epoch"]) for row in exploits if row["copy_only"] == "true"})

    def test_provenance_models_and_docs_registration(self):
        provenance = json.loads((BUNDLE / "provenance.json").read_text())
        expected = {
            "global_best_state.pt",
            "global_best_optimizer.pt",
            "global_best_scaler.pt",
            "final_best_state.pt",
        }
        self.assertEqual(expected, {model["file"] for model in provenance["models"]})
        models = BUNDLE / "models"
        if models.is_dir():  # Release assets are intentionally excluded from Git.
            self.assertEqual([], sorted(name for name in expected if not (models / name).is_file()))
        index = (PROJECT_DIR / "docs/experiments.rst").read_text()
        page = PROJECT_DIR / "docs/experiments/cadenced-pbt-v1-50.rst"
        self.assertTrue(page.is_file())
        self.assertIn("Cadenced PBT v1 — 50 epochs", index)
        self.assertIn("experiments/cadenced-pbt-v1-50", index)


if __name__ == "__main__":
    unittest.main()
