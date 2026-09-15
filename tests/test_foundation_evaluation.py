import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.helpers import namespace, PROJECT_DIR
from training.pbt.config import load_config
from training.pbt.runner import initial_evaluation_enabled, run_final_checkpoint_evaluations
from training.pbt.execution.weaver_command import make_command
from training.pbt.state.optimizer_state import atomic_copy
from training.pbt.state.checkpointing import atomic_copy_pair


class FoundationEvaluationTest(unittest.TestCase):
    def test_full_pass_config_has_no_caps_and_initial_evaluation_without_controller(self):
        config = load_config(namespace(config=PROJECT_DIR / "configs/experiments/foundation_correctness_pilot.yaml",
                                       experiment_name="unit_foundation", gpus="0,1", slots=None, smoke=False))
        self.assertTrue(initial_evaluation_enabled(config))
        self.assertEqual(config["pbt"]["dynamic_controller"]["mode"], "disabled")
        with tempfile.TemporaryDirectory() as tmp:
            command, _, _ = make_command(config, dict(name="identical_a", lr=8.5e-6), "0", Path(tmp) / "identical_a", 0)
        self.assertNotIn("--samples-per-epoch", command)
        self.assertNotIn("--samples-per-epoch-val", command)
        self.assertIn("--data-audit", command)

    def fixture(self, root):
        for name in ("a", "b"):
            (root / name).mkdir()
            (root / name / "net_epoch-19_state.pt").write_bytes(name.encode())
        best = root / "selected.pt"
        best.write_bytes(b"an earlier selected checkpoint")
        config = dict(pbt=dict(evaluate_final_checkpoints=True),
                      shared=dict(dataset="data", validation_suffix="val", proxy_validation={}))
        manifest = dict(members={"a": {}, "b": {}}, generations=[dict(index=1, epoch=19)],
                        best=dict(state_path=str(best)))
        return config, manifest

    def test_final_evaluates_actual_arms_and_separate_selected_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, manifest = self.fixture(root)
            def evaluate(*args):
                checkpoints = args[6]
                self.assertEqual(set(checkpoints), {"a", "b", "selected_best"})
                self.assertEqual(checkpoints["a"].read_bytes(), b"a")
                self.assertEqual(checkpoints["selected_best"].read_bytes(), b"an earlier selected checkpoint")
                return {name: dict(status="completed", metrics={}) for name in checkpoints}
            with patch("training.pbt.runner.run_tiered_evaluation", side_effect=evaluate):
                run_final_checkpoint_evaluations(config, manifest, root, root / "manifest.json", root / "log")
            self.assertTrue(manifest["final_evaluations"]["control"]["selected_best"]["checkpoint_sha256"])

    def test_final_failure_is_not_silently_successful(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, manifest = self.fixture(root)
            with patch("training.pbt.runner.run_tiered_evaluation", return_value={}):
                with self.assertRaisesRegex(RuntimeError, "Required final evaluation failed"):
                    run_final_checkpoint_evaluations(config, manifest, root, root / "manifest.json", root / "log")

    def test_optimizer_copies_propagate_scaler_and_remove_stale_legacy_scaler(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, dst = root / "source_optimizer.pt", root / "dest_optimizer.pt"
            scaler, destination_scaler = root / "source_scaler.pt", root / "dest_scaler.pt"
            src.write_bytes(b"optimizer")
            scaler.write_bytes(b"scaler")
            atomic_copy_pair([(src, dst)])
            self.assertEqual(destination_scaler.read_bytes(), b"scaler")
            scaler.unlink()
            atomic_copy(src, dst)
            self.assertFalse(destination_scaler.exists())
