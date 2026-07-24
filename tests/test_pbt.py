import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

import run_pbt  # noqa: E402


class PBTTest(unittest.TestCase):
    def smoke_config(self):
        return run_pbt.load_config(
            SimpleNamespace(
                config=PROJECT_DIR / "configs/experiments/pp_pbt.yaml",
                experiment_name="unit_test",
                gpus="0,2",
                smoke=True,
            )
        )

    def test_smoke_config_and_resume_command(self):
        config = self.smoke_config()
        self.assertEqual(config["shared"]["generations"], 2)
        self.assertEqual(len(config["population"]), 2)

        member = {"name": "member_00", "lr": 9.0e-5}
        command, log_path, target_epoch = run_pbt.make_command(
            config,
            member,
            "0",
            PROJECT_DIR / "runs/pbt/unit_test/member_00",
            generation=1,
        )

        self.assertEqual(target_epoch, 1)
        self.assertEqual(command[command.index("--load-epoch") + 1], "0")
        self.assertIn("--override-load-lr", command)
        self.assertEqual(command[command.index("--start-lr") + 1], "9e-05")
        self.assertEqual(command[command.index("--seed") + 1], "12346")
        self.assertEqual(log_path.name, "generation-001.log")

    def test_ranking_and_exploit_plan_are_deterministic(self):
        config = self.smoke_config()
        generation = {
            "index": 0,
            "workers": {
                "member_00": {"metrics": {"validation_bkg_rejection_score": 2.0}},
                "member_01": {"metrics": {"validation_bkg_rejection_score": 1.0}},
            },
        }
        members = {
            "member_00": {"lr": 7.5e-5},
            "member_01": {"lr": 1.0e-4},
        }

        ranking, plan = run_pbt.ranking_and_plan(config, generation, members)

        self.assertEqual(ranking, ["member_00", "member_01"])
        self.assertEqual(plan[0]["donor"], "member_00")
        self.assertEqual(plan[0]["recipient"], "member_01")
        self.assertGreaterEqual(plan[0]["new_lr"], config["pbt"]["min_lr"])
        self.assertLessEqual(plan[0]["new_lr"], config["pbt"]["max_lr"])

    def test_exploit_copies_both_states_and_updates_lineage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("strong", "weak"):
                (root / name).mkdir()
            strong_state, strong_optimizer = run_pbt.checkpoint_paths(root / "strong", 0)
            weak_state, weak_optimizer = run_pbt.checkpoint_paths(root / "weak", 0)
            strong_state.write_bytes(b"strong-state")
            strong_optimizer.write_bytes(b"strong-optimizer")
            weak_state.write_bytes(b"weak-state")
            weak_optimizer.write_bytes(b"weak-optimizer")
            manifest_path = root / "manifest.json"
            manifest = {
                "members": {
                    "strong": {"lr": 1.0e-4, "parent": None},
                    "weak": {"lr": 1.5e-4, "parent": None},
                }
            }
            generation = {
                "index": 0,
                "epoch": 0,
                "exploit": [
                    {
                        "donor": "strong",
                        "recipient": "weak",
                        "new_lr": 8.0e-5,
                        "applied": False,
                    }
                ],
            }

            run_pbt.apply_exploit(root, manifest, generation, manifest_path)

            self.assertEqual(weak_state.read_bytes(), b"strong-state")
            self.assertEqual(weak_optimizer.read_bytes(), b"strong-optimizer")
            self.assertTrue(generation["exploit"][0]["applied"])
            self.assertEqual(manifest["members"]["weak"]["parent"], "strong")
            self.assertEqual(manifest["members"]["weak"]["lr"], 8.0e-5)


if __name__ == "__main__":
    unittest.main()
