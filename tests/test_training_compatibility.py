"""Compatibility boundaries for the architecture cleanup; no training workers."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.helpers import pbt_smoke_config
from tests.test_pbt_artifacts import synthetic_manifest
from training import checkpoints
from training.members import MemberState
from training.pbt.execution.backend import LocalWeaverBackend, finite_metric_ok as legacy_metric_ok
from training.pbt.execution.ray_backend import RayWeaverBackend
from training.pbt.reporting import write_canonical_outputs
from training.pbt.reporting.plots import write_existing_physics_reports
from training.pbt.state import checkpointing, optimizer_state
from validation.results import finite_metric_ok, require_checkpoint_result
from validation.evaluate_checkpoint_fixed_wp import write_reports


class MemberCompatibilityTest(unittest.TestCase):
    def test_mutated_lr_preserves_identity_commands_and_legacy_metadata(self):
        config = pbt_smoke_config()
        member = dict(name="lr_3e-6", lr=1.344e-5, parent="lr_14e-6", extra={"history": [1]})
        before = copy.deepcopy(member)
        state = MemberState.from_legacy(member)
        self.assertEqual(state.member_id, "lr_3e-6")
        self.assertEqual(state.current_lr, member["lr"])
        self.assertEqual(state.to_legacy(), dict(name=member["name"], lr=member["lr"]))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / state.member_id
            for backend in (LocalWeaverBackend(), RayWeaverBackend()):
                for generation in (0, 1, 9):
                    with self.subTest(backend=backend.name, generation=generation):
                        legacy = backend.command_for(config, member, "0", directory, generation)
                        explicit = backend.command_for(config, state, "0", directory, generation)
                        self.assertEqual(legacy, explicit)
                        command, log, _ = explicit
                        self.assertEqual(float(command[command.index("--start-lr") + 1]), state.current_lr)
                        self.assertEqual(log.parent.name, state.member_id)
            self.assertEqual(list(Path(temporary).iterdir()), [])
        self.assertEqual(member, before)


class CheckpointCompatibilityTest(unittest.TestCase):
    def test_legacy_exports_use_shared_copy_operations(self):
        self.assertIs(checkpointing.atomic_copy_pair, checkpoints.atomic_copy_pair)
        self.assertIs(checkpointing.checkpoint_paths, checkpoints.checkpoint_paths)
        self.assertIs(optimizer_state.atomic_copy, checkpoints.atomic_copy)
        self.assertIs(optimizer_state.copy_optimizer_companion, checkpoints.copy_optimizer_companion)

    def test_bundle_copy_preserves_all_bytes_and_rejects_changed_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = checkpoints.bundle_paths(Path(temporary) / "donor", 4)
            destination = checkpoints.bundle_paths(Path(temporary) / "recipient", 9)
            source["state"].parent.mkdir()
            for part, path in source.items():
                path.write_bytes(part.encode())
            identity = checkpoints.bundle_identity(source)
            copied = checkpoints.copy_bundle(source, destination)
            checkpoints.check_bundle(copied)
            self.assertEqual(checkpoints.bundle_identity(source), identity)
            for part in source:
                self.assertEqual(destination[part].read_bytes(), source[part].read_bytes())
            destination["scaler"].write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Checkpoint identity changed"):
                checkpoints.check_bundle(copied)

    def test_staging_failure_leaves_recipient_and_scaler_untouched_then_legacy_copy_removes_scaler(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = checkpoints.bundle_paths(root / "donor", 1)
            destination = checkpoints.bundle_paths(root / "recipient", 1)
            source["state"].parent.mkdir()
            destination["state"].parent.mkdir()
            source["state"].write_bytes(b"donor")
            for path in destination.values():
                path.write_bytes(b"recipient")
            pairs = [(source[k], destination[k]) for k in ("state", "optimizer")]
            with self.assertRaises(FileNotFoundError):
                checkpoints.atomic_copy_pair(pairs)
            for path in destination.values():
                self.assertEqual(path.read_bytes(), b"recipient")
            self.assertEqual(list(destination["state"].parent.glob("*.pbt-tmp")), [])
            source["optimizer"].write_bytes(b"legacy optimizer")
            checkpoints.atomic_copy_pair(pairs)
            self.assertEqual(destination["state"].read_bytes(), b"donor")
            self.assertEqual(destination["optimizer"].read_bytes(), b"legacy optimizer")
            self.assertFalse(destination["scaler"].exists())


class ResultCompatibilityTest(unittest.TestCase):
    def test_metric_acceptance_matches_legacy_policy(self):
        self.assertIs(legacy_metric_ok, finite_metric_ok)
        for value, expected in [(0, True), ("1.2", True), (None, False),
                                (float("nan"), False), (float("inf"), False),
                                ("bad", False), ([], False)]:
            with self.subTest(value=value):
                self.assertEqual(finite_metric_ok({"metric": value}, "metric"), expected)
        self.assertFalse(finite_metric_ok(None, "metric"))
        self.assertFalse(finite_metric_ok({}, "metric"))

    def test_final_result_requires_completed_status_and_unchanged_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.pt"
            path.write_bytes(b"checkpoint")
            digest = checkpoints.sha256(path)
            record = dict(status="completed", metrics={"loss": 0.5})
            self.assertIs(require_checkpoint_result(record, path, digest, "control/member"), record)
            self.assertEqual(record["checkpoint_sha256"], digest)
            for invalid in ({}, {"status": "failed"}):
                with self.assertRaisesRegex(RuntimeError, "control/member"):
                    require_checkpoint_result(invalid, path, digest, "control/member")
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "checkpoint changed"):
                require_checkpoint_result(record, path, digest, "control/member")


class ReadOnlyReportingTest(unittest.TestCase):
    def test_direct_physics_bridge_uses_snapshot_without_touching_disk_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "manifest.json"
            # Deliberately unusable disk content: plots must use the supplied view.
            path.write_bytes(b"disk manifest must not be read or overwritten")
            original, stamp = path.read_bytes(), path.stat().st_mtime_ns
            manifest = synthetic_manifest()
            before = copy.deepcopy(manifest)
            outputs = write_existing_physics_reports(root, manifest)
            self.assertEqual(outputs["checkpoint_selection_metadata"]["member"], "trial_b")
            self.assertTrue(Path(outputs["physics_performance"]).is_file())
            self.assertEqual(manifest, before)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(path.stat().st_mtime_ns, stamp)

    def test_standalone_checkpoint_reports_preserve_manifest_and_summary_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "manifest.json"
            path.write_text(json.dumps(synthetic_manifest(), indent=3))
            original, stamp = path.read_bytes(), path.stat().st_mtime_ns
            summary_path = write_reports(path)
            summary = json.loads(summary_path.read_text())
            self.assertTrue(summary["plots"])
            for output in summary["plots"].values():
                self.assertTrue(Path(output).is_file())
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(path.stat().st_mtime_ns, stamp)

    def test_report_failure_does_not_change_caller_or_pinned_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "manifest.json"
            path.write_bytes(b"pinned evidence")
            stamp = path.stat().st_mtime_ns
            manifest = synthetic_manifest()
            before = copy.deepcopy(manifest)

            def failed_plot(_root, snapshot):
                snapshot["members"].clear()
                raise RuntimeError("plot failed")

            with patch("training.pbt.reporting.canonical.write_existing_physics_reports", side_effect=failed_plot):
                with self.assertRaisesRegex(RuntimeError, "plot failed"):
                    write_canonical_outputs(root, manifest)
            self.assertEqual(manifest, before)
            self.assertEqual(path.read_bytes(), b"pinned evidence")
            self.assertEqual(path.stat().st_mtime_ns, stamp)
