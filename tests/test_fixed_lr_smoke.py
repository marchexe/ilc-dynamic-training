import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.helpers import PROJECT_DIR, namespace
from training import fixed_lr_smoke as launcher
from training.pbt.config import load_config
from scripts.research import verify_fixed_lr_smoke as verifier


class FixedLRSmokeTest(unittest.TestCase):
    def test_smoke_inherits_exact_production_training_settings(self):
        production = load_config(namespace(config=PROJECT_DIR / "configs/experiments/foundation_fixed_lr_20epochs.yaml",
                                           experiment_name=None, gpus="0,1,2,3,4", slots=None, smoke=False))
        smoke = launcher.configuration()
        expected = dict(production["shared"], generations=1)
        self.assertEqual(smoke["shared"], expected)
        self.assertEqual(smoke["pbt"], production["pbt"])
        self.assertEqual(smoke["population"], production["population"])
        self.assertEqual(smoke["shared"]["initial_optimizer_mode"], "raw")
        self.assertNotEqual(smoke["output_root"], production["output_root"])
        self.assertNotIn("--smoke", launcher.launch_command())

    def gpu_rows(self):
        return "\n".join(f"{i}, GPU-{i}, 0, 40536, 0" for i in range(5))

    def test_gpu_availability_refuses_any_occupied_or_missing_slot(self):
        for rows, apps in [(self.gpu_rows(), "GPU-2, 1234"),
                           (self.gpu_rows().replace("0, 40536, 0", "2000, 38536, 0"), ""),
                           (self.gpu_rows().splitlines()[0], "")]:
            with self.subTest(rows=rows, apps=apps), contextlib.redirect_stdout(io.StringIO()):
                with patch.object(launcher.subprocess, "check_output", side_effect=[rows, apps]):
                    with self.assertRaises(RuntimeError):
                        launcher.available_gpus()
        with patch.object(launcher.subprocess, "check_output", side_effect=[self.gpu_rows(), ""]):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(launcher.available_gpus(), [f"GPU-{i}" for i in range(5)])

    def test_launcher_uses_production_runner_and_separate_output_without_executing_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "smoke"
            config = dict(output_root=str(output), experiment_name="study")
            with patch.object(launcher, "OUTPUT", output), patch.object(launcher, "configuration", return_value=config), \
                 patch.object(launcher, "validate_inputs"), patch.object(launcher.socket, "gethostname", return_value="iutgpu01"), \
                 patch.object(launcher, "available_gpus", return_value=[f"GPU-{i}" for i in range(5)]), \
                 patch.object(launcher.subprocess, "Popen") as popen, contextlib.redirect_stdout(io.StringIO()):
                popen.return_value.pid = 1234
                launcher.start()
                self.assertFalse((output / "study").exists())
                self.assertEqual(popen.call_args.args[0], launcher.launch_command())
                env = popen.call_args.kwargs["env"]
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-0,GPU-1,GPU-2,GPU-3,GPU-4")
                self.assertEqual(json.loads((output / "launcher.json").read_text())["token"], env[launcher.TOKEN_KEY])
                with self.assertRaises(FileExistsError):
                    launcher.start()
                self.assertEqual(popen.call_count, 1)

    def test_process_ownership_requires_exact_token_and_uid(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp)
            record = dict(uid=os.getuid(), token="abc")
            (proc / "environ").write_bytes(f"{launcher.TOKEN_KEY}=abcdef\0".encode())
            self.assertFalse(launcher.owns_process(proc, record))
            (proc / "environ").write_bytes(f"{launcher.TOKEN_KEY}=abc\0".encode())
            self.assertTrue(launcher.owns_process(proc, record))
            self.assertFalse(launcher.owns_process(proc, dict(record, uid=os.getuid() + 1)))

    def test_stop_rechecks_identity_after_opening_pidfd(self):
        with patch.object(launcher, "metadata", return_value=dict(pid=1234)), \
             patch.object(launcher, "live_pids", return_value=[1234]), \
             patch.object(launcher.os, "pidfd_open", return_value=99), \
             patch.object(launcher, "owns_process", return_value=False), \
             patch.object(launcher.os, "close"), patch.object(launcher.signal, "pidfd_send_signal") as send, \
             contextlib.redirect_stdout(io.StringIO()):
            launcher.stop()
            send.assert_not_called()

    def fixture(self, run):
        config = launcher.configuration()
        rows = dict(count=150000, unique_ids=150000, repeated_ids=0,
                    sequence_sha256="v" * 64, id_set_sha256="s" * 64)
        val = dict(consumed=rows, dataset=dict(fingerprint="d" * 64), exhausted=True,
                   traversal=[dict(wraps=0)], prediction_sha256="p" * 64)
        train = dict(consumed=dict(rows, count=2151623, unique_ids=2151623, sequence_sha256="t" * 64),
                     dataset=dict(fingerprint="f" * 64, total_rows=2392232), exhausted=True,
                     traversal=[dict(wraps=0, scanned=dict(count=2392232, repeated_ids=0), accepted=dict(count=2151623))],
                     optimizer_steps=2102, batches=2102)
        workers, final, hashes = {}, {}, {}
        for name, lr in verifier.ARMS.items():
            folder = run / name
            folder.mkdir()
            for component in ["state", "optimizer"]:
                path = folder / f"net_epoch-17_{component}.pt"
                path.write_bytes(component.encode())
                hashes[component] = verifier.sha256(path)
            for component in ["state", "optimizer", "scaler"]:
                (folder / f"net_epoch-18_{component}.pt").write_bytes(b"new")
            metrics = dict(train_data_audit=copy.deepcopy(train), validation_data_audit=copy.deepcopy(val),
                           train_loaded_optimizer_lr=lr, validation_loss=0.3, **{verifier.METRIC: 0.4})
            workers[name] = dict(status="completed", returncode=0, lr=lr, metrics=metrics)
            final[name] = dict(status="completed", checkpoint_sha256=verifier.sha256(folder / "net_epoch-18_state.pt"),
                               metrics=copy.deepcopy(metrics))
        first = next(iter(verifier.ARMS))
        final["selected_best"] = copy.deepcopy(final[first])
        return dict(config=config, status="completed", members={n: dict(lr=lr) for n, lr in verifier.ARMS.items()},
                    initial_evaluation=dict(status="completed", checkpoint_sha256=hashes["state"],
                                            metrics=dict(validation_data_audit=copy.deepcopy(val))),
                    generations=[dict(epoch=18, status="completed", exploit=[], workers=workers)],
                    final_evaluations=dict(control=final), best=dict(state_path=str(run / first / "net_epoch-18_state.pt"))), hashes

    def test_verifier_accepts_complete_evidence_and_rejects_required_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            good, hashes = self.fixture(run)
            def verify(payload):
                (run / "manifest.json").write_text(json.dumps(payload))
                with patch.object(verifier, "INITIAL_HASHES", hashes):
                    return verifier.verify(run)
            self.assertTrue(verify(good)["passed"])
            mutations = [
                lambda m, w: w["metrics"]["train_data_audit"]["traversal"][0]["scanned"].update(count=119808),
                lambda m, w: w["metrics"]["train_data_audit"]["traversal"][0].update(wraps=1),
                lambda m, w: w["metrics"]["train_data_audit"]["consumed"].update(sequence_sha256="different"),
                lambda m, w: w["metrics"]["validation_data_audit"]["consumed"].update(sequence_sha256="different"),
                lambda m, w: m["generations"][0].update(exploit=[dict(action="rewind")]),
                lambda m, w: w["metrics"].pop(verifier.METRIC),
                lambda m, w: w["metrics"].update(train_loaded_optimizer_lr=1e-3),
                lambda m, w: m.update(status="failed"),
            ]
            for mutation in mutations:
                payload = copy.deepcopy(good)
                mutation(payload, payload["generations"][0]["workers"][next(iter(verifier.ARMS))])
                self.assertFalse(verify(payload)["passed"])
            (run / next(iter(verifier.ARMS)) / "net_epoch-18_scaler.pt").unlink()
            self.assertFalse(verify(good)["passed"])
