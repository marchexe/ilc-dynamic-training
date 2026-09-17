import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.helpers import PROJECT_DIR
from scripts.launch import experiment as launcher
from scripts.validation import verify_fixed_lr as verifier


class ExperimentLauncherTest(unittest.TestCase):
    def test_resume_refuses_live_mismatched_and_completed_runs_and_appends_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = dict(output_root=str(root / 'runs'), experiment_name='unit', config_path=str(root / 'config.yaml'),
                          gpus=['5'], slots=[dict(gpu='5')], population=[dict(name='a')])
            record = dict(token='a' * 32, host=launcher.socket.gethostname(), uid=os.getuid(), run=str(root / 'runs/unit'))
            with patch.object(launcher, 'PROJECT_DIR', root), patch.object(launcher, 'validate_inputs'), \
                 patch.object(launcher, 'contract_fingerprint', return_value='expected'), \
                 patch.object(launcher, 'available_gpus', return_value=['GPU-5']), \
                 patch.object(launcher, 'live_pids', return_value=[]) as live, \
                 patch.object(launcher.subprocess, 'Popen') as popen, contextlib.redirect_stdout(io.StringIO()):
                run, output = launcher.locations(config)
                run.mkdir(parents=True); output.mkdir(parents=True)
                (output / 'launcher.json').write_text(json.dumps(record))
                (output / 'main.log').write_text('previous log\n')
                for payload in (dict(fingerprint='wrong', status='failed'), dict(fingerprint='expected', status='completed')):
                    (run / 'manifest.json').write_text(json.dumps(payload))
                    with self.assertRaises(ValueError):
                        launcher.start(config, resume=True)
                (run / 'manifest.json').write_text(json.dumps(dict(fingerprint='expected', status='failed')))
                live.return_value = [42]
                with self.assertRaises(RuntimeError):
                    launcher.start(config, resume=True)
                popen.assert_not_called()
                live.return_value = []
                popen.return_value.pid = 1234
                launcher.start(config, resume=True)
                self.assertEqual(popen.call_args.args[0][-1], '--resume')
                self.assertEqual(popen.call_args.kwargs['env'][launcher.TOKEN_KEY], record['token'])
                self.assertEqual((output / 'main.log').read_text(), 'previous log\n')

    def test_full_epoch_smoke_preserves_production_contract(self):
        production = launcher.configuration(PROJECT_DIR / "configs/experiments/foundation_fixed_lr_20epochs.yaml")
        smoke = launcher.configuration(PROJECT_DIR / "configs/experiments/foundation_fixed_lr_smoke.yaml")
        self.assertEqual(smoke["shared"], dict(production["shared"], generations=1))
        self.assertEqual(smoke["pbt"], production["pbt"])
        self.assertEqual(smoke["population"], production["population"])
        self.assertNotEqual(launcher.locations(smoke), launcher.locations(production))
        self.assertNotIn("--smoke", launcher.launch_command(smoke, smoke["gpus"]))

    def test_gpu_availability_refuses_busy_missing_or_duplicate_devices(self):
        rows = "2, GPU-2, 0, 40000, 0\n5, GPU-5, 0, 40000, 0"
        cases = [(rows, "GPU-2, 1234"), (rows.replace("0, 40000", "2000, 38000"), ""),
                 (rows.replace("40000", "20000"), ""), (rows.replace("40000, 0", "40000, 90"), ""),
                 (rows.splitlines()[0], "")]
        for devices, apps in cases:
            with patch.object(launcher.subprocess, "check_output", side_effect=[devices, apps]), \
                 contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                launcher.available_gpus([5, 2])
        with patch.object(launcher.subprocess, "check_output", side_effect=[rows, "GPU-9, 999"]), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.available_gpus([5, 2]), ["GPU-5", "GPU-2"])
        with self.assertRaises(ValueError):
            launcher.available_gpus([2, 2])

    def test_start_uses_production_runner_and_refuses_collisions_without_executing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = dict(output_root=str(root / "runs"), experiment_name="unit", config_path=str(root / "config.yaml"),
                          gpus=["5", "2"], slots=[dict(gpu="5"), dict(gpu="2")], population=[dict(name="a"), dict(name="b")])
            with patch.object(launcher, "PROJECT_DIR", root), patch.object(launcher, "validate_inputs"), \
                 patch.object(launcher, "available_gpus", return_value=["GPU-5", "GPU-2"]), \
                 patch.object(launcher.subprocess, "Popen") as popen, contextlib.redirect_stdout(io.StringIO()):
                popen.return_value.pid = 1234
                run, output = launcher.locations(config)
                run.mkdir(parents=True)
                with self.assertRaises(FileExistsError):
                    launcher.start(config)
                popen.assert_not_called()
                run.rmdir()
                launcher.start(config)
                self.assertFalse(run.exists())
                command = popen.call_args.args[0]
                self.assertEqual(command, launcher.launch_command(config, config["gpus"]))
                self.assertEqual(command[-1], "0,1")
                env = popen.call_args.kwargs["env"]
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-5,GPU-2")
                self.assertEqual(json.loads((output / "launcher.json").read_text())["token"], env[launcher.TOKEN_KEY])
                with self.assertRaises(FileExistsError):
                    launcher.start(config)
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
            launcher.stop(Path("unused"))
            send.assert_not_called()


class FixedLRVerifierTest(unittest.TestCase):
    def fixture(self, run):
        population = [dict(name="a", start_lr=1e-5), dict(name="b", start_lr=2e-5)]
        metric = "validation_score"
        shared = dict(generations=2, initial_epoch=3, weaver_epochs_per_generation=1,
                      initial_optimizer_mode="raw", initial_optimizer=str(run / "source_optimizer.pt"),
                      deterministic=True, data_audit=True, lr_scheduler="none", use_amp=True, amp_dtype="fp16")
        pbt = dict(strategy="fixed_lr_grid", metric=metric, rollback_fraction=0,
                   baseline_guard_action="observe", early_stop_degraded_generations=0)

        def audit(split, count):
            rows = dict(count=count, unique_ids=count, repeated_ids=0, sequence_sha256=split, id_set_sha256=split)
            data = dict(files=[dict(path=split, rows=count)], total_rows=count, fingerprint=split)
            return dict(consumed=rows, dataset=data, exhausted=True, optimizer_steps=1, batches=1,
                        traversal=[dict(wraps=0, exhausted=True, scanned=rows, accepted=rows)], prediction_sha256="prediction")

        val, train = audit("val", 3), audit("train", 7)
        resume = dict(epoch=shared["initial_epoch"])
        for member in population:
            folder = run / member["name"]
            folder.mkdir()
            for component in ["state", "optimizer"]:
                path = folder / f"net_epoch-{shared['initial_epoch']}_{component}.pt"
                path.write_bytes(component.encode())
                resume[component + "_sha256"] = verifier.sha256(path)
        generations, final = [], {}
        for index in range(shared["generations"]):
            epoch = shared["initial_epoch"] + index + 1
            workers = {}
            for member in population:
                name, lr = member["name"], member["start_lr"]
                metrics = dict(train_data_audit=copy.deepcopy(train), validation_data_audit=copy.deepcopy(val),
                               train_loaded_optimizer_lr=lr, validation_loss=0.3, validation_score=0.4,
                               validation_bc_mistag_eff_0_8_percent=0.5)
                workers[name] = dict(status="completed", returncode=0, lr=lr, metrics=metrics)
                for component in ["state", "optimizer", "scaler"]:
                    path = run / name / f"net_epoch-{epoch}_{component}.pt"
                    path.write_bytes(f"{name}-{epoch}-{component}".encode())
                state = run / name / f"net_epoch-{epoch}_state.pt"
                final[name] = dict(status="completed", checkpoint_sha256=verifier.sha256(state), metrics=copy.deepcopy(metrics))
            generations.append(dict(index=index, epoch=epoch, status="completed", exploit=[], workers=workers))
        final["selected_best"] = copy.deepcopy(final["a"])
        return dict(config=dict(shared=shared, pbt=pbt, population=population), status="completed", initial_resume=resume,
                    datasets=dict(resolved_files={s: [dict(files=[s])] for s in ["train", "val"]}),
                    initial_evaluation=dict(status="completed", checkpoint_sha256=resume["state_sha256"],
                                            metrics=copy.deepcopy(final["a"]["metrics"])),
                    generations=generations, final_evaluations=dict(control=final),
                    best=dict(state_path=str(run / "a" / f"net_epoch-{epoch}_state.pt"), metrics=copy.deepcopy(final["a"]["metrics"])))

    def test_complete_evidence_and_failure_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            good = self.fixture(run)

            def verify(payload, through=None):
                (run / "manifest.json").write_text(json.dumps(payload))
                return verifier.verify(run, through)

            self.assertTrue(verify(good)["passed"])
            mutations = [
                lambda m, w: w["metrics"]["train_data_audit"]["traversal"][0].update(scanned=dict(count=2)),
                lambda m, w: w["metrics"]["train_data_audit"]["traversal"][0].update(wraps=1),
                lambda m, w: w["metrics"]["train_data_audit"]["consumed"].update(sequence_sha256="different"),
                lambda m, w: w["metrics"]["validation_data_audit"]["consumed"].update(sequence_sha256="different"),
                lambda m, w: m["generations"][0].update(exploit=[dict(action="rewind")]),
                lambda m, w: w["metrics"].pop("validation_score"),
                lambda m, w: w["metrics"].update(train_loaded_optimizer_lr=1),
                lambda m, w: m.update(status="failed"),
                lambda m, w: m["generations"][0].update(epoch=99),
                lambda m, w: m["generations"].pop(),
                lambda m, w: m["initial_resume"].update(optimizer_sha256="bad"),
                lambda m, w: m["final_evaluations"]["control"]["a"]["metrics"]["validation_data_audit"].update(prediction_sha256="bad"),
                lambda m, w: m["final_evaluations"]["control"]["selected_best"]["metrics"].update(validation_loss=9),
                lambda m, w: m["final_evaluations"]["control"]["a"]["metrics"].update(validation_bc_mistag_eff_0_8_percent=9),
            ]
            for mutation in mutations:
                payload = copy.deepcopy(good)
                mutation(payload, payload["generations"][0]["workers"]["a"])
                self.assertFalse(verify(payload)["passed"])
            prefix = copy.deepcopy(good)
            prefix.update(status="running", final_evaluations={})
            prefix["generations"].pop()
            self.assertTrue(verify(prefix, through=1)["passed"])
            with self.assertRaises(ValueError):
                verify(good, through=3)
            epoch = good["generations"][0]["epoch"]
            (run / "a" / f"net_epoch-{epoch}_scaler.pt").unlink()
            self.assertFalse(verify(good)["passed"])
