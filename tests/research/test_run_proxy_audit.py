import json
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from tests.helpers import SCRIPTS_DIR  # noqa: F401

from research.process_cleanup import AuditShutdownRequested
from research.run_proxy_audit import (
    build_checkpoint_metrics_rows,
    dedupe_checkpoints,
    handle_shutdown,
    poll_remote_processes_clear,
    provenance_for,
    unique_hosts,
    write_csv_rows,
    write_outputs,
)
from training.runtime import sha256
from tests.helpers import namespace


class DedupeCheckpointsTest(unittest.TestCase):
    def test_identical_files_are_deduplicated_by_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            first = temporary / "a.pt"
            second = temporary / "b.pt"
            first.write_bytes(b"same weights")
            second.write_bytes(b"same weights")
            checkpoints = [
                {"id": "first", "path": str(first)},
                {"id": "second", "path": str(second)},
            ]
            distinct, duplicates = dedupe_checkpoints(checkpoints)

        self.assertEqual(len(distinct), 1)
        self.assertEqual(distinct[0]["id"], "first")
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(duplicates[0]["id"], "second")
        self.assertEqual(duplicates[0]["duplicate_of"], "first")
        self.assertEqual(duplicates[0]["reason"], "identical_sha256")

    def test_distinct_files_are_both_kept(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            first = temporary / "a.pt"
            second = temporary / "b.pt"
            first.write_bytes(b"weights one")
            second.write_bytes(b"weights two")
            checkpoints = [
                {"id": "first", "path": str(first)},
                {"id": "second", "path": str(second)},
            ]
            distinct, duplicates = dedupe_checkpoints(checkpoints)

        self.assertEqual(len(distinct), 2)
        self.assertEqual(duplicates, [])

    def test_missing_checkpoint_file_is_recorded_not_silently_skipped(self):
        checkpoints = [{"id": "missing", "path": "/nonexistent/path/net.pt"}]
        distinct, duplicates = dedupe_checkpoints(checkpoints)
        self.assertEqual(distinct, [])
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(duplicates[0]["reason"], "checkpoint_file_missing")

    def test_evaluation_does_not_modify_the_checkpoint_file(self):
        """dedupe_checkpoints only reads the file to hash it -- confirms the
        hash is stable across repeated calls, i.e. nothing about this audit
        code path writes to a checkpoint file."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "a.pt"
            path.write_bytes(b"weights")
            before = sha256(path)
            dedupe_checkpoints([{"id": "a", "path": str(path)}])
            after = sha256(path)
        self.assertEqual(before, after)


class ProvenanceForTest(unittest.TestCase):
    def test_epoch_parsed_from_checkpoint_filename(self):
        entry = {"path": "runs/pbt/example/lr_6e-6/net_epoch-42_state.pt", "provenance": None}
        result = provenance_for(entry)
        self.assertEqual(result["epoch"], 42)
        self.assertIsNone(result["generation"])
        self.assertIsNone(result["lr"])

    def test_global_best_metadata_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            metadata_path = Path(temporary) / "global_best_metadata.json"
            metadata_path.write_text(json.dumps({"epoch": 85, "generation": 67, "lr": 7.9e-6, "metric_value": 0.95, "member": "lr_6e-6"}))
            entry = {
                "path": str(Path(temporary) / "global_best_state.pt"),
                "provenance": {"type": "global_best_metadata", "path": str(metadata_path)},
            }
            result = provenance_for(entry)

        self.assertEqual(result["epoch"], 85)
        self.assertEqual(result["generation"], 67)
        self.assertAlmostEqual(result["lr"], 7.9e-6)
        self.assertAlmostEqual(result["pbt_recorded_metric_value"], 0.95)
        self.assertEqual(result["member"], "lr_6e-6")

    def test_manifest_member_provenance_finds_matching_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "members": {"lr_14e-6": {"lr": 1.3e-5}},
                        "generations": [
                            {
                                "index": 24,
                                "epoch": 42,
                                "workers": {
                                    "lr_14e-6": {
                                        "status": "completed",
                                        "metrics": {"validation_working_point_mistag_percent": 1.1},
                                    }
                                },
                            }
                        ],
                    }
                )
            )
            entry = {
                "path": "runs/pbt/example/lr_14e-6/net_epoch-42_state.pt",
                "provenance": {"type": "manifest_member", "path": str(manifest_path), "member": "lr_14e-6"},
            }
            result = provenance_for(entry)

        self.assertEqual(result["epoch"], 42)
        self.assertEqual(result["generation"], 24)
        self.assertAlmostEqual(result["lr"], 1.3e-5)
        self.assertAlmostEqual(result["pbt_recorded_metric_value"], 1.1)

    def test_manifest_member_generation_unavailable_when_no_matching_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            manifest_path.write_text(json.dumps({"members": {"lr_14e-6": {"lr": 1.3e-5}}, "generations": []}))
            entry = {
                "path": "runs/pbt/example/lr_14e-6/net_epoch-42_state.pt",
                "provenance": {"type": "manifest_member", "path": str(manifest_path), "member": "lr_14e-6"},
            }
            result = provenance_for(entry)

        self.assertEqual(result["epoch"], 42)
        self.assertIsNone(result["generation"])
        self.assertAlmostEqual(result["lr"], 1.3e-5)


class CheckpointMetricsRowsTest(unittest.TestCase):
    def test_min_max_orientation_does_not_affect_row_construction(self):
        """build_checkpoint_metrics_rows records raw metric values verbatim
        -- ranking/orientation is applied downstream by proxy_statistics,
        not here, so this must not silently reinterpret min/max."""
        entry = {"id": "ckpt", "path": "x.pt", "sha256": "abc", "source_run": None}
        provenance_by_id = {"ckpt": {"member": None, "epoch": 17, "generation": None, "lr": None, "pbt_recorded_metric_value": None}}
        control_results = {"ckpt": {"status": "completed", "metrics": {"validation_working_point_mistag_percent": 1.23}, "log": "c.log"}}
        full_results = {"ckpt": {"status": "completed", "metrics": {"validation_working_point_mistag_percent": 4.56}, "log": "f.log"}}
        rows = build_checkpoint_metrics_rows([entry], provenance_by_id, control_results, full_results, 1)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["control_proxy_50k_working_point_mistag_percent"], 1.23)
        self.assertAlmostEqual(rows[0]["full_validation_working_point_mistag_percent"], 4.56)

    def test_nan_metric_is_recorded_as_not_finite_not_dropped(self):
        entry = {"id": "ckpt", "path": "x.pt", "sha256": "abc", "source_run": None}
        provenance_by_id = {"ckpt": {"member": None, "epoch": 17, "generation": None, "lr": None, "pbt_recorded_metric_value": None}}
        control_results = {"ckpt": {"status": "completed", "metrics": {"validation_working_point_mistag_percent": float("nan")}, "log": "c.log"}}
        full_results = {"ckpt": {"status": "failed", "metrics": None, "log": "f.log"}}
        rows = build_checkpoint_metrics_rows([entry], provenance_by_id, control_results, full_results, 1)
        self.assertFalse(rows[0]["control_proxy_50k_metric_finite"])
        self.assertEqual(rows[0]["full_validation_status"], "failed")
        self.assertIsNone(rows[0]["full_validation_working_point_mistag_percent"])


class WriteCsvRowsTest(unittest.TestCase):
    def test_reproducible_round_trip(self):
        rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "out.csv"
            write_csv_rows(path, rows)
            first_write = path.read_text()
            write_csv_rows(path, rows)
            second_write = path.read_text()
        self.assertEqual(first_write, second_write)
        self.assertIn("a,b", first_write)

    def test_empty_rows_still_produce_a_file_not_an_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "out.csv"
            write_csv_rows(path, [])
            self.assertTrue(path.is_file())

    def test_rows_with_different_keys_do_not_crash(self):
        """A partial/interrupted audit can produce rows with different
        columns (e.g. one checkpoint failed before a field was ever
        computed) -- the writer must union the fieldnames, not crash."""
        rows = [{"a": 1}, {"a": 2, "b": 3}]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "out.csv"
            write_csv_rows(path, rows)
            content = path.read_text()
        self.assertIn("a,b", content)


class WriteOutputsIncrementalTest(unittest.TestCase):
    """Covers the incremental-results safeguard: write_outputs must be
    callable after control_proxy_50k alone (full_results={}) and produce a
    real, readable partial checkpoint_metrics.csv/summary.json -- not just
    at the very end after both tiers -- so a killed/crashed process still
    leaves something recoverable."""

    def setUp(self):
        self.entry = {"id": "ckpt", "path": "x.pt", "sha256": "abc", "source_run": None}
        self.provenance_by_id = {"ckpt": {"member": None, "epoch": 17, "generation": None, "lr": None, "pbt_recorded_metric_value": None}}
        self.audit_config = {"checkpoints": [self.entry], "proxy_manifest": "/nonexistent/manifest.json"}
        self.args = namespace(config=Path("configs/research/nightly_proxy_audit.yaml"))

    def test_partial_write_after_control_only_has_no_full_validation_status(self):
        control_results = {"ckpt": {"status": "completed", "metrics": {"validation_working_point_mistag_percent": 1.0}}}
        with tempfile.TemporaryDirectory() as temporary:
            experiment_dir = Path(temporary)
            summary = write_outputs(
                experiment_dir, self.args, self.audit_config, [self.entry], [], self.provenance_by_id,
                control_results, {}, 1, "validation_working_point_mistag_percent", "min",
                {"control_proxy_50k": 30.0, "full_validation": None, "total": 30.0},
                "partial_control_only", "test_run",
            )
            self.assertTrue((experiment_dir / "checkpoint_metrics.csv").is_file())
            self.assertTrue((experiment_dir / "summary.json").is_file())
            rows = json.loads(json.dumps(summary))  # sanity: summary is JSON-serializable
            csv_text = (experiment_dir / "checkpoint_metrics.csv").read_text()

        self.assertEqual(summary["run_status"], "partial_control_only")
        self.assertIsNone(summary["runtime_seconds"]["full_validation"])
        self.assertIn("ckpt", csv_text)
        self.assertIn(",,", csv_text)  # full_validation_status column empty for this row
        self.assertIsInstance(rows, dict)

    def test_final_write_overwrites_partial_write_not_appends(self):
        """The second call (after full_validation completes) must replace
        the partial file, not leave two files or stale rows behind."""
        control_results = {"ckpt": {"status": "completed", "metrics": {"validation_working_point_mistag_percent": 1.0}}}
        full_results = {"ckpt": {"status": "completed", "metrics": {"validation_working_point_mistag_percent": 1.1}}}
        with tempfile.TemporaryDirectory() as temporary:
            experiment_dir = Path(temporary)
            write_outputs(
                experiment_dir, self.args, self.audit_config, [self.entry], [], self.provenance_by_id,
                control_results, {}, 1, "validation_working_point_mistag_percent", "min",
                {"control_proxy_50k": 30.0, "full_validation": None, "total": 30.0},
                "partial_control_only", "test_run",
            )
            final_summary = write_outputs(
                experiment_dir, self.args, self.audit_config, [self.entry], [], self.provenance_by_id,
                control_results, full_results, 1, "validation_working_point_mistag_percent", "min",
                {"control_proxy_50k": 30.0, "full_validation": 200.0, "total": 230.0},
                "completed", "test_run",
            )
            csv_files = list(experiment_dir.glob("checkpoint_metrics*.csv"))
            summary_files = list(experiment_dir.glob("summary*.json"))

        self.assertEqual(len(csv_files), 1)
        self.assertEqual(len(summary_files), 1)
        self.assertEqual(final_summary["run_status"], "completed")
        self.assertAlmostEqual(final_summary["runtime_seconds"]["full_validation"], 200.0)

    def test_no_leftover_tmp_files(self):
        control_results = {"ckpt": {"status": "completed", "metrics": {"validation_working_point_mistag_percent": 1.0}}}
        with tempfile.TemporaryDirectory() as temporary:
            experiment_dir = Path(temporary)
            write_outputs(
                experiment_dir, self.args, self.audit_config, [self.entry], [], self.provenance_by_id,
                control_results, {}, 1, "validation_working_point_mistag_percent", "min",
                {"control_proxy_50k": 30.0, "full_validation": None, "total": 30.0},
                "partial_control_only", "test_run",
            )
            tmp_files = list(experiment_dir.glob("*.tmp"))
        self.assertEqual(tmp_files, [])


class HandleShutdownTest(unittest.TestCase):
    """handle_shutdown is what main() calls on SIGTERM/SIGINT/timeout. Uses
    a real, freshly-spawned, childless subprocess's own PID as the "owning
    process" passed to terminate_child_processes -- it has zero children
    of its own, so cleanup is a real, safe no-op here, and this stays
    completely isolated from the actual test-runner process tree (we must
    never call terminate_child_processes(os.getpid()) in these tests)."""

    def setUp(self):
        self.entry = {"id": "ckpt", "path": "x.pt", "sha256": "abc", "source_run": None}
        self.provenance_by_id = {"ckpt": {"member": None, "epoch": 17, "generation": None, "lr": None, "pbt_recorded_metric_value": None}}
        self.audit_config = {"checkpoints": [self.entry], "proxy_manifest": "/nonexistent/manifest.json", "slots": []}
        self.args = namespace(config=Path("configs/research/nightly_proxy_audit.yaml"), termination_grace_period_seconds=1.0)
        self.childless_proc = subprocess.Popen(["sleep", "20"], start_new_session=True)
        time.sleep(0.2)

    def tearDown(self):
        self.childless_proc.kill()
        self.childless_proc.wait()

    def test_partial_outputs_remain_readable_after_simulated_sigterm(self):
        control_results = {"ckpt": {"status": "completed", "metrics": {"validation_working_point_mistag_percent": 1.0}}}
        with tempfile.TemporaryDirectory() as temporary:
            experiment_dir = Path(temporary)
            # Files main() writes before the try/except block, and which
            # handle_shutdown must never touch.
            (experiment_dir / "resolved_config.yaml").write_text("shared: {}\n")
            (experiment_dir / "environment.json").write_text(json.dumps({"hostname": "test"}))
            write_outputs(
                experiment_dir, self.args, self.audit_config, [self.entry], [], self.provenance_by_id,
                control_results, {}, 1, "validation_working_point_mistag_percent", "min",
                {"control_proxy_50k": 30.0, "full_validation": None, "total": 30.0},
                "partial_control_only", "test_run",
            )
            resolved_config_before = (experiment_dir / "resolved_config.yaml").read_text()
            environment_before = (experiment_dir / "environment.json").read_text()

            shutdown = AuditShutdownRequested("signal", signal_name="SIGTERM")
            run_status = handle_shutdown(
                shutdown, experiment_dir, experiment_dir / "logs" / "audit.log", self.args, self.audit_config,
                "test_run", self.childless_proc.pid, [self.entry], [], self.provenance_by_id, control_results, {},
                1, "validation_working_point_mistag_percent", "min", 30.0, time.monotonic() - 30.0,
            )

            resolved_config_after = (experiment_dir / "resolved_config.yaml").read_text()
            environment_after = json.loads((experiment_dir / "environment.json").read_text())
            summary = json.loads((experiment_dir / "summary.json").read_text())
            csv_text = (experiment_dir / "checkpoint_metrics.csv").read_text()

        self.assertEqual(run_status, "interrupted")
        self.assertEqual(resolved_config_before, resolved_config_after)
        self.assertEqual(environment_before, json.dumps(environment_after))
        self.assertEqual(summary["run_status"], "interrupted")
        self.assertEqual(summary["shutdown"]["received_signal"], "SIGTERM")
        self.assertEqual(summary["shutdown"]["reason"], "signal")
        self.assertTrue(summary["shutdown"]["cleanup_attempted"])
        self.assertEqual(summary["shutdown"]["child_processes_terminated"], 0)
        self.assertEqual(summary["shutdown"]["cleanup_failures"], [])
        self.assertIn("ckpt", csv_text)

    def test_timeout_reason_produces_timed_out_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment_dir = Path(temporary)
            shutdown = AuditShutdownRequested("timeout")
            run_status = handle_shutdown(
                shutdown, experiment_dir, experiment_dir / "logs" / "audit.log", self.args, self.audit_config,
                "test_run", self.childless_proc.pid, [self.entry], [], self.provenance_by_id, {}, {},
                1, "validation_working_point_mistag_percent", "min", None, time.monotonic() - 5,
            )
            summary = json.loads((experiment_dir / "summary.json").read_text())
        self.assertEqual(run_status, "timed_out")
        self.assertEqual(summary["run_status"], "timed_out")
        self.assertIsNone(summary["shutdown"]["received_signal"])
        self.assertEqual(summary["shutdown"]["reason"], "timeout")

    def test_no_slots_means_no_remote_verification_attempted(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment_dir = Path(temporary)
            shutdown = AuditShutdownRequested("signal", signal_name="SIGINT")
            handle_shutdown(
                shutdown, experiment_dir, experiment_dir / "logs" / "audit.log", self.args, self.audit_config,
                "test_run", self.childless_proc.pid, [self.entry], [], self.provenance_by_id, {}, {},
                1, "validation_working_point_mistag_percent", "min", None, time.monotonic() - 5,
            )
            summary = json.loads((experiment_dir / "summary.json").read_text())
        self.assertEqual(summary["shutdown"]["remote_verification"], [])


class UniqueHostsTest(unittest.TestCase):
    def test_deduplicates_hosts_across_slots(self):
        audit_config = {"slots": [{"host": "iutgpu01", "gpu": "0"}, {"host": "iutgpu01", "gpu": "1"}, {"host": "iutgpu02", "gpu": "0"}]}
        self.assertEqual(unique_hosts(audit_config), ["iutgpu01", "iutgpu02"])

    def test_empty_slots_gives_empty_hosts(self):
        self.assertEqual(unique_hosts({"slots": []}), [])
        self.assertEqual(unique_hosts({}), [])


class PollRemoteProcessesClearTest(unittest.TestCase):
    def test_unreachable_host_reports_error_without_retrying_forever(self):
        """An SSH/connection error is not the same as "confirmed clear" --
        callers must be able to tell the difference (an empty result with
        error=None means we checked and found nothing; a non-None error
        means we couldn't check at all)."""
        started = time.monotonic()
        pids, error, retries_used = poll_remote_processes_clear(
            "nonexistent-test-host-xyz.invalid", "some-run-id", max_retries=3, poll_interval_seconds=0.1
        )
        elapsed = time.monotonic() - started
        self.assertIsNone(pids)
        self.assertIsNotNone(error)
        self.assertEqual(retries_used, 0)  # gives up immediately on a hard error, doesn't retry a broken connection
        self.assertLess(elapsed, 5.0)


if __name__ == "__main__":
    unittest.main()
