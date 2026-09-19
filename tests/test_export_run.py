import hashlib
import json
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

from scripts.publish.export_run import export


class ExportRunTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
