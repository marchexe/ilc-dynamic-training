import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.helpers import PROJECT_DIR  # Sets up script imports.
from scripts.reports import plot_pbt_interim_milestone as milestone


def fixture_manifest(seed, interval, trajectory, exploits):
    names = [f"member_{index}" for index in range(5)]
    generations = []
    for index, current_best in enumerate(trajectory):
        workers = {
            name: {
                "status": "completed",
                "returncode": 0,
                "metrics": {milestone.METRIC: current_best + member_index * .001},
            }
            for member_index, name in enumerate(names)
        }
        event = []
        if index in exploits:
            recipient_index = max(1, 4 - list(exploits).index(index))
            event = [{"donor": names[0], "recipient": names[recipient_index], "applied": True}]
        generations.append({
            "index": index,
            "status": "completed",
            "workers": workers,
            "exploit": event,
            "cadenced_pbt_v1": {"terminal": index == 49},
        })
    best_value = min(trajectory)
    return {
        "status": "completed",
        "next_generation": 50,
        "config": {
            "experiment_name": f"seed{seed}_cadence{interval}",
            "shared": {"seed": seed, "weaver_epochs_per_generation": 1},
            "pbt": {
                "strategy": "cadenced_pbt_v1",
                "metric": milestone.METRIC,
                "exploit_interval_generations": interval,
            },
        },
        "members": {name: {} for name in names},
        "generations": generations,
        "final_evaluations": {"control": {
            name: {"status": "completed"} for name in [*names, "selected_best"]
        }},
        "best": {"metric_value": best_value, "member": names[0], "generation": trajectory.index(best_value)},
        "git": {"commit": "a" * 40, "dirty": False},
        "fingerprint": "b" * 64,
        "run": {"source_hashes": {"strategy": {"sha256": "c" * 64}}},
        "datasets": {"fingerprints": {"train": "d" * 64, "val": "e" * 64}},
        "checkpoint": {"sha256": "f" * 64},
    }


class PBTInterimMilestoneTest(unittest.TestCase):
    def write_manifest(self, root, seed, interval, trajectory, exploits=()):
        path = root / f"seed{seed}_cadence{interval}" / "manifest.json"
        path.parent.mkdir()
        path.write_text(json.dumps(fixture_manifest(seed, interval, trajectory, exploits)))
        return path

    def test_summary_uses_current_best_final10_and_replays_lineage(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_manifest(
                Path(temporary), 12345, 1, [1 - index / 100 for index in range(50)],
                exploits=range(4),
            )
            summary = milestone.summarize_run(path, 12345, 1)
        self.assertAlmostEqual(summary["final10_current_best_mean"], sum(.60 - i / 100 for i in range(10)) / 10)
        self.assertEqual(summary["global_best"], .51)
        self.assertEqual(summary["exploit_count"], 4)
        self.assertEqual(summary["lineage_collapse_epoch"], 4)

    def test_pair_counts_trajectory_wins_and_reports_cadence5_minus_cadence1(self):
        cadence1 = {"training_seed": 22345, "trajectory": [.3, .4, .5], "final10_current_best_mean": .4}
        cadence5 = {"training_seed": 22345, "trajectory": [.4, .4, .45], "final10_current_best_mean": .45}
        result = milestone.compare_pair(cadence1, cadence5)
        self.assertEqual(result["trajectory_wins"], {"cadence1": 1, "cadence5": 1, "ties": 1})
        self.assertAlmostEqual(result["cadence5_minus_cadence1"], .05)
        self.assertAlmostEqual(result["cadence5_relative_difference_percent"], 12.5)

    def test_export_has_fixed_presentation_artifact_names(self):
        fake = {
            "metric": milestone.METRIC,
            "lower_is_better": True,
            "runs": {(seed, cadence): {
                "training_seed": seed,
                "final10_current_best_mean": .33 + cadence / 1000,
                "trajectory": [.34] * 50,
                "exploit_count": 10,
                "lineage_collapse_epoch": 20,
            } for seed in (12345, 22345) for cadence in (1, 5)},
            "pairs": {seed: {
                "cadence5_minus_cadence1": .001,
                "trajectory_wins": {"cadence1": 40, "cadence5": 5, "ties": 5},
            } for seed in (12345, 22345)},
        }
        with tempfile.TemporaryDirectory() as temporary, patch.object(milestone, "collect", return_value=fake):
            paths = milestone.export(Path(temporary))
            self.assertEqual([path.name for path in paths], list(milestone.OUTPUT_NAMES))
            self.assertTrue(all(path.is_file() for path in paths))


if __name__ == "__main__":
    unittest.main()
