"""Small CPU fixtures for the new one-full-epoch PBT strategy; no real training."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests.helpers import PROJECT_DIR
from scripts.launch.experiment import configuration
from scripts.research.dry_run_cadenced_pbt_v1 import synthetic_schedule
from training.checkpoints import bundle_identity, bundle_paths, check_bundle
from training.pbt.execution.backend import LocalWeaverBackend
from training.pbt.models.config import ResolvedPBTConfig
from training.pbt.planning import cadenced_pbt_v1 as strategy
from training.pbt.planning.dispatch import plan_for_strategy
from training.pbt import runner
from training.pbt.reporting.events import record_cadenced_decision
from training.pbt.reporting.io import append_event, read_events
from training.pbt.reporting.markdown_report import (
    _cadenced_decision_summary_lines,
    _final_window_current_best,
)
from training.pbt.state.checkpointing import epoch_for_generation
from training.pbt.state.optimizer_state import load_optimizer_state
from training.runtime import atomic_json
from weaver.utils.nn.optimizer.ranger import Ranger


CONFIG_PATH = PROJECT_DIR / "configs/experiments/cadenced_pbt_v1.yaml"
CADENCE5_PATH = PROJECT_DIR / "configs/experiments/cadenced_pbt_v1_cadence5.yaml"
CADENCE5_SMOKE_PATH = PROJECT_DIR / "configs/experiments/cadenced_pbt_v1_cadence5_smoke.yaml"
SCRATCH_PATH = PROJECT_DIR / "configs/experiments/cadenced_pbt_v1_scratch.yaml"
SCRATCH_SMOKE_PATH = PROJECT_DIR / "configs/experiments/cadenced_pbt_v1_scratch_smoke.yaml"


def generation(config, members, index, scores=None):
    names = list(members)
    scores = scores or [0.400, 0.401, 0.402, 0.403, 0.410]
    return {
        "index": index,
        "epoch": epoch_for_generation(config, index),
        "workers": {
            name: {"status": "completed", "returncode": 0, "lr": members[name]["lr"],
                   "metrics": {config["pbt"]["metric"]: scores[position]}}
            for position, name in enumerate(names)
        },
    }


def tiny_bundle(run, name, epoch, lr):
    paths = bundle_paths(run / name, epoch)
    paths["state"].parent.mkdir(parents=True, exist_ok=True)
    model = torch.nn.Linear(1, 1)
    marker = sum((position + 1) * ord(character) for position, character in enumerate(name))
    with torch.no_grad():
        model.weight.fill_(float(marker))
        model.bias.fill_(float(epoch))
    optimizer = torch.optim.AdamW([
        {"params": [model.weight], "lr": lr},
        {"params": [model.bias], "lr": lr},
    ])
    model(torch.ones(1, 1)).sum().backward()
    optimizer.step()
    torch.save(model.state_dict(), paths["state"])
    torch.save(optimizer.state_dict(), paths["optimizer"])
    torch.save({"scale": 4096.0 + marker, "growth_tracker": epoch}, paths["scaler"])
    return paths


class CadencedPBTTest(unittest.TestCase):
    def setUp(self):
        self.config = configuration(CONFIG_PATH)
        self.names = [item["name"] for item in self.config["population"]]
        self.members = {
            item["name"]: {"name": item["name"], "lr": item["start_lr"]}
            for item in self.config["population"]
        }

    def plan(self, index, scores=None):
        record = generation(self.config, self.members, index, scores)
        ranking, events = plan_for_strategy(self.config, record, self.members)
        record["ranking"] = ranking
        record["exploit"] = events
        return record

    def test_two_epoch_warmup_then_every_epoch_copy_and_mutation_opportunity(self):
        for index in range(4):
            record = self.plan(index)
            decision = record[strategy.STRATEGY]
            self.assertEqual(decision["completed_epoch"], index + 1)
            self.assertTrue(decision["validation_complete"])
            self.assertEqual(decision["warmup_active_during_training"], index < 2)
            self.assertEqual(decision["copy_opportunity"], index >= 1)
            self.assertEqual(decision["lr_mutation_opportunity"], index >= 1)
            self.assertEqual(len(record["exploit"]), 0 if index == 0 else 1)
            if index >= 1:
                self.assertEqual(record["exploit"][0]["mutation_factor"] in (0.8, 1.2), True)
        self.assertEqual(self.plan(1)[strategy.STRATEGY]["completed_epoch"], 2)

    def test_cadence5_changes_only_the_generic_adaptation_interval(self):
        cadence5 = configuration(CADENCE5_PATH)
        self.assertEqual(cadence5["experiment_name"], "cadenced_pbt_v1_cadence5_50epochs")
        self.assertEqual(cadence5["shared"], self.config["shared"])
        self.assertEqual(cadence5["population"], self.config["population"])
        self.assertEqual(cadence5["slots"], self.config["slots"])
        self.assertEqual(cadence5["gpus"], self.config["gpus"])
        self.assertEqual(cadence5["output_root"], self.config["output_root"])
        baseline_pbt = copy.deepcopy(self.config["pbt"])
        control_pbt = copy.deepcopy(cadence5["pbt"])
        self.assertEqual(baseline_pbt.pop("exploit_interval_generations"), 1)
        self.assertEqual(control_pbt.pop("exploit_interval_generations"), 5)
        self.assertEqual(control_pbt, baseline_pbt)

    def test_cadence5_global_boundaries_warmup_off_cadence_and_terminal(self):
        config = configuration(CADENCE5_PATH)
        members = {
            item["name"]: {"name": item["name"], "lr": item["start_lr"]}
            for item in config["population"]
        }
        records = []
        for index in range(50):
            record = generation(config, members, index)
            ranking, events = plan_for_strategy(config, record, members)
            record.update(ranking=ranking, exploit=events)
            records.append(record)
        decisions = [record[strategy.STRATEGY] for record in records]
        expected = list(range(5, 50, 5))
        self.assertEqual(
            [item["completed_epoch"] for item in decisions if item["copy_opportunity"]],
            expected,
        )
        self.assertEqual(
            [record[strategy.STRATEGY]["completed_epoch"] for record in records if record["exploit"]],
            expected,
        )
        self.assertEqual(decisions[0]["reason"], "warmup")
        self.assertEqual([item["reason"] for item in decisions[1:4]], ["off_cadence"] * 3)
        self.assertTrue(decisions[4]["cadence_boundary"])
        self.assertEqual(decisions[4]["reason"], "metric_gap_exceeds_margin")
        self.assertTrue(decisions[-1]["cadence_boundary"])
        self.assertTrue(decisions[-1]["terminal"])
        self.assertEqual(decisions[-1]["reason"], "terminal_generation")
        self.assertFalse(decisions[-1]["copy_opportunity"])
        self.assertEqual(records[-1]["exploit"], [])

    def test_cadence5_eligible_boundary_matches_cadence1_selection_and_mutation(self):
        cadence5 = configuration(CADENCE5_PATH)
        members = copy.deepcopy(self.members)
        scores = [0.412, 0.404, 0.403, 0.402, 0.399]
        baseline = generation(self.config, members, 4, scores)
        control = generation(cadence5, members, 4, scores)
        baseline_ranking, baseline_events = plan_for_strategy(self.config, baseline, members)
        control_ranking, control_events = plan_for_strategy(cadence5, control, members)
        self.assertEqual(control_ranking, baseline_ranking)
        self.assertEqual(control_events, baseline_events)
        self.assertEqual(control[strategy.STRATEGY]["donor"], baseline[strategy.STRATEGY]["donor"])
        self.assertEqual(control[strategy.STRATEGY]["recipient"], baseline[strategy.STRATEGY]["recipient"])
        self.assertEqual(control[strategy.STRATEGY]["metric_gap"], baseline[strategy.STRATEGY]["metric_gap"])
        self.assertEqual(control[strategy.STRATEGY]["new_lr"], baseline[strategy.STRATEGY]["new_lr"])

    def test_cadence5_interrupted_boundary_replays_and_resumes_next_generation(self):
        config = configuration(CADENCE5_PATH)
        members = {
            item["name"]: {"name": item["name"], "lr": item["start_lr"]}
            for item in config["population"]
        }
        record = generation(config, members, 4)
        ranking, events = plan_for_strategy(config, record, members)
        record.update(ranking=ranking, exploit=events)
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            for name in members:
                tiny_bundle(run, name, record["epoch"], members[name]["lr"])
            strategy.prepare_boundary(run, record)
            manifest_path = run / "manifest.json"
            manifest = {"config": config, "members": copy.deepcopy(members), "generations": [record]}
            with patch.object(strategy, "atomic_set_optimizer_lr", side_effect=RuntimeError("cadence5 interruption")):
                with self.assertRaisesRegex(RuntimeError, "cadence5 interruption"):
                    strategy.apply_cadenced_exploits(run, manifest, record, manifest_path)
            self.assertFalse(record["exploit"][0]["applied"])
            check_bundle(record["exploit"][0]["donor_archive"])
            check_bundle(record["exploit"][0]["pre_copy_archive"])
            strategy.apply_cadenced_exploits(run, manifest, record, manifest_path)
            event = record["exploit"][0]
            self.assertTrue(event["applied"])
            command, _, _ = LocalWeaverBackend().command_for(
                config, manifest["members"][event["recipient"]], "0",
                run / event["recipient"], generation=5,
            )
            self.assertEqual(command[command.index("--load-epoch") + 1], str(record["epoch"]))
            self.assertEqual(float(command[command.index("--start-lr") + 1]), event["new_lr"])
            before = manifest_path.read_bytes()
            strategy.apply_cadenced_exploits(run, manifest, record, manifest_path)
            self.assertEqual(manifest_path.read_bytes(), before)

    def test_cadence5_smoke_reaches_boundary_and_subsequent_generation(self):
        production = configuration(CADENCE5_PATH)
        smoke = configuration(CADENCE5_SMOKE_PATH)
        self.assertEqual(smoke["experiment_name"], "cadenced_pbt_v1_cadence5_full_reference_smoke")
        self.assertEqual(smoke["shared"]["generations"], 6)
        comparable_shared = copy.deepcopy(smoke["shared"])
        comparable_shared["generations"] = production["shared"]["generations"]
        self.assertEqual(comparable_shared, production["shared"])
        self.assertEqual(smoke["population"], production["population"])
        self.assertEqual(smoke["pbt"], production["pbt"])
        self.assertEqual(smoke["pbt"][strategy.STRATEGY]["decision_margin"], 0.002)
        rows = list(synthetic_schedule(smoke, 6))
        self.assertEqual([row["completed_epoch"] for row in rows if row["copy_planned"]], [5])
        self.assertTrue(rows[4]["cadence_boundary"])
        self.assertFalse(rows[5]["copy_planned"])
        self.assertTrue(rows[5]["terminal"])

    def test_one_epoch_best_worst_strict_margin_noop_and_one_recipient(self):
        record = self.plan(1)
        self.assertEqual(record["ranking"][0], self.names[0])
        self.assertEqual(record["ranking"][-1], self.names[-1])
        self.assertEqual([event["recipient"] for event in record["exploit"]], [self.names[-1]])
        self.assertNotEqual(record["exploit"][0]["recipient"], record["ranking"][0])
        tied = self.plan(1, [0.400, 0.4005, 0.401, 0.4015, 0.402])
        self.assertEqual(tied["exploit"], [])
        self.assertEqual(tied[strategy.STRATEGY]["reason"], "within_margin")
        self.assertEqual(tied[strategy.STRATEGY]["new_lr"], tied[strategy.STRATEGY]["old_lr"])
        self.config["pbt"][strategy.STRATEGY]["decision_margin"] = 0.011
        self.assertEqual(self.plan(1)["exploit"], [])

    def test_no_action_until_all_five_current_epoch_validations_complete(self):
        record = generation(self.config, self.members, 1)
        record["workers"][self.names[-1]]["status"] = "pending"
        with self.assertRaisesRegex(ValueError, "completed validation"):
            plan_for_strategy(self.config, record, self.members)
        record["workers"][self.names[-1]]["status"] = "completed"
        record["workers"][self.names[-1]]["metrics"].clear()
        with self.assertRaisesRegex(ValueError, "finite reference metric"):
            plan_for_strategy(self.config, record, self.members)

    def test_terminal_generation_suppresses_copy_even_above_margin(self):
        self.config["shared"]["generations"] = 2
        record = self.plan(1)
        self.assertEqual(record["exploit"], [])
        self.assertTrue(record[strategy.STRATEGY]["terminal"])
        self.assertFalse(record[strategy.STRATEGY]["copy_opportunity"])

    def test_collision_suppresses_mutation_but_not_weight_copy(self):
        for name, lr in zip(self.names, [10e-6, 8e-6, 12e-6, 6e-6, 14e-6]):
            self.members[name]["lr"] = lr
        record = self.plan(1)
        self.assertEqual(len(record["exploit"]), 1)
        event = record["exploit"][0]
        self.assertEqual(event["new_lr"], 14e-6)
        self.assertIsNone(event.get("mutation_factor"))
        self.assertFalse(event["mutation_applied"])
        self.assertEqual(event["mutation_reason"], "lr_collision")
        self.assertTrue(record[strategy.STRATEGY]["copy_planned"])
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            for name in self.names:
                tiny_bundle(run, name, record["epoch"], self.members[name]["lr"])
            donor_before = bundle_identity(bundle_paths(run / event["donor"], record["epoch"]))
            recipient_before = bundle_identity(bundle_paths(run / event["recipient"], record["epoch"]))
            strategy.prepare_boundary(run, record)
            manifest = {"config": self.config, "members": copy.deepcopy(self.members), "generations": [record]}
            strategy.apply_cadenced_exploits(run, manifest, record, run / "manifest.json")
            self.assertTrue(event["applied"])
            self.assertTrue(record[strategy.STRATEGY]["copy_applied"])
            self.assertEqual(event["post_copy"]["state"]["sha256"], donor_before["state"]["sha256"])
            self.assertEqual(event["post_copy"]["scaler"]["sha256"], donor_before["scaler"]["sha256"])
            self.assertNotEqual(event["post_copy"]["state"]["sha256"], recipient_before["state"]["sha256"])
            self.assertEqual(manifest["members"][event["recipient"]]["lr"], event["recipient_lr"])
            optimizer = load_optimizer_state(bundle_paths(run / event["recipient"], record["epoch"])["optimizer"])
            self.assertTrue(all(group["lr"] == event["recipient_lr"] for group in optimizer["param_groups"]))
            check_bundle(event["pre_copy_archive"])

    def test_one_valid_factor_and_deterministic_separation_choice(self):
        self.members[self.names[0]]["lr"] = 10e-6
        self.members[self.names[1]]["lr"] = 8e-6
        selected = strategy.select_mutation(self.config, self.members, self.names[0], self.names[-1])
        self.assertEqual(selected["mutation_factor"], 1.2)
        self.assertAlmostEqual(selected["new_lr"], 12e-6)
        first = strategy.select_mutation(self.config, self.members, self.names[0], self.names[-1])
        second = strategy.select_mutation(self.config, dict(reversed(list(self.members.items()))), self.names[0], self.names[-1])
        self.assertEqual(first, second)
        for name, lr in zip(self.names, [10e-6, 3e-6, 5e-6, 15e-6, 20e-6]):
            self.members[name]["lr"] = lr
        both_valid = strategy.select_mutation(self.config, self.members, self.names[0], self.names[-1])
        self.assertEqual(both_valid["mutation_factor"], 0.8)
        self.assertAlmostEqual(both_valid["new_lr"], 8e-6)

    def test_unchanged_recipient_lr_is_not_a_valid_mutation(self):
        first = self.plan(1)["exploit"][0]
        self.assertEqual(first["mutation_factor"], 0.8)
        recipient = first["recipient"]
        self.members[recipient]["lr"] = first["new_lr"]
        next_event = self.plan(2)["exploit"][0]
        self.assertEqual(next_event["mutation_factor"], 1.2)
        self.assertAlmostEqual(next_event["new_lr"], 3.6e-6)
        self.assertTrue(next_event["mutation_applied"])
        self.assertEqual(next_event["mutation_reason"], "mutated")
        self.assertEqual(next_event["rejected_mutations"][0]["reason"], "unchanged_recipient_lr")

        # If the only distinct factor collides, exploitation still copies
        # weights and retains the recipient's previous LR.
        self.members[self.names[1]]["lr"] = next_event["new_lr"]
        blocked = self.plan(2)["exploit"][0]
        self.assertFalse(blocked["mutation_applied"])
        self.assertEqual(blocked["mutation_reason"], "lr_collision")
        self.assertEqual(blocked["new_lr"], self.members[recipient]["lr"])
        self.assertEqual(
            {item["reason"] for item in blocked["rejected_mutations"]},
            {"unchanged_recipient_lr", "lr_collision"},
        )

    def test_bundle_copy_preserves_pre_action_evidence_and_resume_lr(self):
        record = self.plan(1)
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            for name in self.names:
                tiny_bundle(run, name, record["epoch"], self.members[name]["lr"])
            donor = record["exploit"][0]["donor"]
            recipient = record["exploit"][0]["recipient"]
            donor_before = bundle_identity(bundle_paths(run / donor, record["epoch"]))
            recipient_before = bundle_identity(bundle_paths(run / recipient, record["epoch"]))
            strategy.prepare_boundary(run, record)
            event = record["exploit"][0]
            self.assertEqual(event["pre_copy"], recipient_before)
            self.assertEqual(
                {key: value["sha256"] for key, value in event["pre_copy_archive"].items()},
                {key: value["sha256"] for key, value in recipient_before.items()},
            )
            manifest = {"config": self.config, "members": copy.deepcopy(self.members), "generations": [record]}
            strategy.apply_cadenced_exploits(run, manifest, record, run / "manifest.json")
            self.assertTrue(event["applied"])
            self.assertTrue(record[strategy.STRATEGY]["copy_applied"])
            self.assertEqual(bundle_identity(bundle_paths(run / donor, record["epoch"])), donor_before)
            check_bundle(event["pre_copy_archive"])
            post = bundle_paths(run / recipient, record["epoch"])
            for part in ("state", "scaler"):
                self.assertEqual(event["post_copy"][part]["sha256"], event["donor_archive"][part]["sha256"])
            donor_optimizer = load_optimizer_state(event["donor_archive"]["optimizer"]["path"])
            resumed_optimizer = load_optimizer_state(post["optimizer"])
            self.assertEqual(donor_optimizer["state"].keys(), resumed_optimizer["state"].keys())
            for key in donor_optimizer["state"]:
                for slot in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(donor_optimizer["state"][key][slot], resumed_optimizer["state"][key][slot])
            self.assertTrue(all(group["lr"] == event["new_lr"] for group in resumed_optimizer["param_groups"]))
            fresh = torch.nn.Linear(1, 1)
            optimizer = torch.optim.AdamW([{"params": [fresh.weight]}, {"params": [fresh.bias]}])
            optimizer.load_state_dict(resumed_optimizer)
            self.assertTrue(all(group["lr"] == event["new_lr"] for group in optimizer.param_groups))
            command, _, _ = LocalWeaverBackend().command_for(
                self.config, manifest["members"][recipient], "0", run / recipient, generation=2,
            )
            self.assertEqual(command[command.index("--load-epoch") + 1], str(record["epoch"]))
            self.assertEqual(float(command[command.index("--start-lr") + 1]), event["new_lr"])
            self.assertIn("--override-load-lr", command)

    def test_runner_plans_after_validation_then_applies_at_boundary(self):
        record = generation(self.config, self.members, 1)
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            for name in self.names:
                tiny_bundle(run, name, record["epoch"], self.members[name]["lr"])
            recipient = self.names[-1]
            before = bundle_identity(bundle_paths(run / recipient, record["epoch"]))
            manifest = {"config": self.config, "members": copy.deepcopy(self.members), "generations": [record]}
            with patch.object(runner, "update_global_best", return_value=False):
                runner._plan_generation_exploit(
                    self.config, manifest, record, 1, False,
                    run, run / "manifest.json", run / "pbt.log",
                )
            self.assertEqual(len(record["exploit"]), 1)
            self.assertEqual(record[strategy.STRATEGY]["completed_epoch"], 2)
            self.assertFalse(record["exploit"][0]["applied"])
            self.assertEqual(bundle_identity(bundle_paths(run / recipient, record["epoch"])), before)
            runner._finalize_generation(
                self.config, manifest, record, run, run / "manifest.json", run / "pbt.log", 1,
            )
            self.assertTrue(record["exploit"][0]["applied"])
            self.assertEqual(manifest["next_generation"], 2)
            self.assertNotEqual(bundle_identity(bundle_paths(run / recipient, record["epoch"])), before)

    def test_interrupted_copy_replays_from_immutable_archive(self):
        record = self.plan(1)
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            for name in self.names:
                tiny_bundle(run, name, record["epoch"], self.members[name]["lr"])
            strategy.prepare_boundary(run, record)
            archive = copy.deepcopy(record["exploit"][0]["pre_copy_archive"])
            manifest = {"config": self.config, "members": copy.deepcopy(self.members), "generations": [record]}
            with patch.object(strategy, "atomic_set_optimizer_lr", side_effect=RuntimeError("injected interruption")):
                with self.assertRaisesRegex(RuntimeError, "injected interruption"):
                    strategy.apply_cadenced_exploits(run, manifest, record, run / "manifest.json")
            self.assertFalse(record["exploit"][0]["applied"])
            check_bundle(archive)
            strategy.apply_cadenced_exploits(run, manifest, record, run / "manifest.json")
            self.assertTrue(record["exploit"][0]["applied"])
            check_bundle(archive)
            before = (run / "manifest.json").read_bytes()
            strategy.apply_cadenced_exploits(run, manifest, record, run / "manifest.json")
            self.assertEqual((run / "manifest.json").read_bytes(), before)

    def test_scratch_and_pretrained_command_contracts_are_separate(self):
        scratch = configuration(SCRATCH_PATH)
        self.assertEqual(scratch["shared"]["initialization_mode"], "scratch")
        self.assertIsNone(scratch["shared"].get("checkpoint"))
        self.assertFalse(scratch["shared"]["freeze_batch_norm"])
        self.assertIsNone(scratch["shared"].get("initial_state"))
        command, _, target = LocalWeaverBackend().command_for(
            scratch, {"name": scratch["population"][0]["name"], "lr": scratch["population"][0]["start_lr"]},
            "0", Path("/tmp/cadenced-scratch-member"), generation=0,
        )
        self.assertEqual(target, 0)
        self.assertNotIn("--load-epoch", command)
        self.assertNotIn("--load-model-weights", command)
        self.assertNotIn("--freeze-batch-norm", command)
        self.assertIn("--use-amp", command)
        pretrained, _, target = LocalWeaverBackend().command_for(
            self.config, {"name": self.names[0], "lr": self.members[self.names[0]]["lr"]},
            "0", Path("/tmp/cadenced-pretrained-member"), generation=0,
        )
        self.assertEqual(target, 18)
        self.assertEqual(pretrained[pretrained.index("--load-epoch") + 1], "17")
        self.assertIn("--freeze-batch-norm", pretrained)
        checkpoint_only = copy.deepcopy(self.config)
        checkpoint_only["shared"]["initial_epoch"] = None
        checkpoint_only["shared"]["initial_state"] = None
        checkpoint_only["shared"]["initial_optimizer"] = None
        command, _, _ = LocalWeaverBackend().command_for(
            checkpoint_only, {"name": self.names[0], "lr": self.members[self.names[0]]["lr"]},
            "0", Path("/tmp/cadenced-checkpoint-only-member"), generation=0,
        )
        self.assertIn("--load-model-weights", command)

    def test_scratch_smoke_keeps_full_population_and_real_validation_contract(self):
        smoke = configuration(SCRATCH_SMOKE_PATH)
        self.assertEqual(smoke["experiment_name"], "cadenced_pbt_v1_scratch_full_reference_smoke")
        self.assertEqual(smoke["shared"]["generations"], 3)
        self.assertEqual(len(smoke["population"]), 5)
        self.assertEqual(
            [member["start_lr"] for member in smoke["population"]],
            [3e-6, 5.75e-6, 8.5e-6, 11.25e-6, 14e-6],
        )
        self.assertEqual(smoke["shared"]["initialization_mode"], "scratch")
        self.assertIsNone(smoke["shared"].get("checkpoint"))
        self.assertIsNone(smoke["shared"].get("initial_state"))
        self.assertIsNone(smoke["shared"].get("initial_optimizer"))
        self.assertFalse(smoke["shared"]["freeze_batch_norm"])
        self.assertIsNone(smoke["shared"].get("samples_per_epoch"))
        self.assertIsNone(smoke["shared"].get("samples_per_epoch_val"))
        self.assertIsNone(smoke["shared"].get("proxy_validation"))
        self.assertFalse(smoke["pbt"]["evaluate_initial_checkpoint"])
        self.assertEqual(smoke["pbt"][strategy.STRATEGY]["warmup_epochs"], 2)
        self.assertEqual(smoke["pbt"][strategy.STRATEGY]["decision_margin"], 0.0)
        command, _, target = LocalWeaverBackend().command_for(
            smoke,
            {"name": smoke["population"][0]["name"], "lr": smoke["population"][0]["start_lr"]},
            "0", Path("/tmp/cadenced-scratch-smoke-member"), generation=0,
        )
        self.assertEqual(target, 0)
        self.assertNotIn("--load-epoch", command)
        self.assertNotIn("--load-model-weights", command)
        self.assertNotIn("--freeze-batch-norm", command)

    def test_schema_rejects_proxy_controller_subepoch_and_scratch_checkpoint(self):
        for section, key, value in (
            ("shared", "weaver_epochs_per_generation", 2),
            ("shared", "samples_per_epoch", 120000),
            ("shared", "lr_scheduler", "steps"),
            ("pbt", "exploit_interval_generations", 0),
            ("pbt", "mutation_factors", [0.9, 1.1]),
        ):
            bad = copy.deepcopy(self.config)
            bad[section][key] = value
            with self.assertRaises(ValueError, msg=key):
                ResolvedPBTConfig.model_validate(bad)
        scratch = configuration(SCRATCH_PATH)
        scratch["shared"]["checkpoint"] = str(PROJECT_DIR / "checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/net_epoch-17_state.pt")
        with self.assertRaises(ValueError):
            ResolvedPBTConfig.model_validate(scratch)
        for option, value in (("warmup_epochs", 1), ("max_recipients", 2)):
            bad = copy.deepcopy(self.config)
            bad["pbt"][strategy.STRATEGY][option] = value
            with self.assertRaises(ValueError, msg=option):
                ResolvedPBTConfig.model_validate(bad)
        scratch = configuration(SCRATCH_PATH)
        scratch["shared"]["freeze_batch_norm"] = True
        with self.assertRaises(ValueError):
            ResolvedPBTConfig.model_validate(scratch)

    def test_manifest_decision_survives_roundtrip(self):
        record = self.plan(1)
        reloaded = json.loads(json.dumps(record))
        self.assertEqual(reloaded[strategy.STRATEGY], record[strategy.STRATEGY])
        self.assertEqual(len(reloaded["exploit"]), 1)

    def test_repeated_exploits_allow_donor_recipient_role_reversal(self):
        first = self.plan(1, [0.400, 0.401, 0.402, 0.403, 0.410])
        self.assertEqual((first["exploit"][0]["donor"], first["exploit"][0]["recipient"]),
                         (self.names[0], self.names[-1]))
        self.members[first["exploit"][0]["recipient"]]["lr"] = first["exploit"][0]["new_lr"]
        second = self.plan(2, [0.412, 0.404, 0.403, 0.402, 0.399])
        self.assertEqual((second["exploit"][0]["donor"], second["exploit"][0]["recipient"]),
                         (self.names[-1], self.names[0]))
        third = self.plan(3, [0.398, 0.404, 0.403, 0.402, 0.411])
        self.assertEqual((third["exploit"][0]["donor"], third["exploit"][0]["recipient"]),
                         (self.names[0], self.names[-1]))
        self.assertEqual(len({item["exploit"][0]["event_id"] for item in (first, second, third)}), 3)

    def test_ranger_restart_contract_restores_radam_but_resets_lookahead(self):
        model = torch.nn.Linear(2, 1)
        optimizer = Ranger(model.parameters(), lr=1e-3)
        for _ in range(3):
            optimizer.zero_grad()
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
        saved = copy.deepcopy(optimizer.state_dict())
        self.assertEqual(optimizer.step_counter, 3)
        self.assertNotIn("cached_params", next(iter(saved["state"].values())))

        resumed_model = torch.nn.Linear(2, 1)
        resumed_model.load_state_dict(model.state_dict())
        resumed = Ranger(resumed_model.parameters(), lr=9e-4)
        self.assertEqual(resumed.step_counter, 0)
        resumed.load_state_dict(saved)
        self.assertEqual(resumed.step_counter, 0)
        for parameter in resumed_model.parameters():
            torch.testing.assert_close(resumed.state[parameter]["cached_params"], parameter.data)
        resumed_state = resumed.state_dict()
        self.assertEqual(saved["state"].keys(), resumed_state["state"].keys())
        for key in saved["state"]:
            for slot in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(saved["state"][key][slot], resumed_state["state"][key][slot])
        self.assertEqual(self.config["shared"]["optimizer"], "ranger")
        windowed = configuration(PROJECT_DIR / "configs/experiments/windowed_pbt_v2.yaml")
        self.assertEqual(windowed["shared"]["optimizer"], "ranger")

    def test_event_replay_deduplicates_ids_and_tolerates_only_truncated_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            payload = {"event_id": "stable-event", "generation": 1, "reason": "fixture"}
            first = append_event(run, "cadenced_pbt_decision", payload)
            second = append_event(run, "cadenced_pbt_decision", payload)
            self.assertEqual(first, second)
            path = run / "events.jsonl"
            with path.open("a", encoding="utf-8") as stream:
                stream.write('{"event_id":"truncated"')
            self.assertEqual(read_events(run), [first])
            append_event(run, "cadenced_pbt_decision", {"event_id": "after-repair", "generation": 2})
            self.assertEqual([event["event_id"] for event in read_events(run)], ["stable-event", "after-repair"])
            with path.open("a", encoding="utf-8") as stream:
                stream.write("{broken}\n")
                stream.write(json.dumps({"event_id": "later"}) + "\n")
            with self.assertRaisesRegex(ValueError, "Corrupt event log line"):
                read_events(run)

    def test_interruption_after_event_persistence_replays_once(self):
        record = self.plan(1)
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            for name in self.names:
                tiny_bundle(run, name, record["epoch"], self.members[name]["lr"])
            strategy.prepare_boundary(run, record)
            manifest_path = run / "manifest.json"
            manifest = {"config": self.config, "members": copy.deepcopy(self.members), "generations": [record]}
            atomic_json(manifest_path, manifest)
            persisted_before = json.loads(manifest_path.read_text())
            with patch.object(strategy, "atomic_json", side_effect=RuntimeError("after event persistence")):
                with self.assertRaisesRegex(RuntimeError, "after event persistence"):
                    strategy.apply_cadenced_exploits(run, manifest, record, manifest_path)
            self.assertEqual(len(read_events(run)), 4)

            replay = persisted_before
            replay_record = replay["generations"][0]
            strategy.apply_cadenced_exploits(run, replay, replay_record, manifest_path)
            self.assertTrue(replay_record["exploit"][0]["applied"])
            self.assertEqual(len(read_events(run)), 4)
            strategy.apply_cadenced_exploits(run, replay, replay_record, manifest_path)
            self.assertEqual(len(read_events(run)), 4)

    def test_missing_or_corrupt_applied_post_copy_checkpoint_fails_safe(self):
        for damaged_part, remove in (("state", True), ("optimizer", False)):
            with self.subTest(damaged_part=damaged_part), tempfile.TemporaryDirectory() as temporary:
                record = self.plan(1)
                run = Path(temporary)
                for name in self.names:
                    tiny_bundle(run, name, record["epoch"], self.members[name]["lr"])
                strategy.prepare_boundary(run, record)
                manifest = {"config": self.config, "members": copy.deepcopy(self.members), "generations": [record]}
                strategy.apply_cadenced_exploits(run, manifest, record, run / "manifest.json")
                damaged = Path(record["exploit"][0]["post_copy"][damaged_part]["path"])
                if remove:
                    damaged.unlink()
                    expected = (FileNotFoundError,)
                else:
                    damaged.write_bytes(b"corrupt")
                    expected = (ValueError,)
                before = copy.deepcopy(manifest)
                with self.assertRaises(expected):
                    strategy.apply_cadenced_exploits(run, manifest, record, run / "manifest.json")
                self.assertEqual(manifest, before)

    def test_decision_reporting_covers_policy_audit_and_primary_endpoint(self):
        records = []
        for index, scores in enumerate((
            [0.400, 0.401, 0.402, 0.403, 0.410],
            [0.400, 0.4005, 0.401, 0.4015, 0.402],
            [0.412, 0.404, 0.403, 0.402, 0.399],
        )):
            record = self.plan(index, scores)
            records.append(record)
        manifest = {"config": self.config, "generations": records}
        lines = _cadenced_decision_summary_lines(manifest)
        report = "\n".join(lines)
        for text in ("warm-up", "eligible", "margin", "reason", "donor", "recipient",
                     "copy+mutation", "pre-copy", "post-copy", "LR before", "LR after"):
            self.assertIn(text, report)
        metric_rows = [
            {"generation": generation_index, "optimization_metric_value": value}
            for generation_index in range(12)
            for value in (1.0 - generation_index * 0.01, 1.2)
        ]
        endpoint = _final_window_current_best(metric_rows, "min")
        self.assertTrue(endpoint["complete"])
        self.assertEqual(endpoint["generations"], list(range(2, 12)))
        self.assertAlmostEqual(endpoint["mean"], sum(1.0 - i * 0.01 for i in range(2, 12)) / 10)

    def test_cadenced_decision_events_include_noop_and_checkpoint_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            warmup = self.plan(0)
            record_cadenced_decision(run, warmup)
            exploit = self.plan(1)
            for name in self.names:
                tiny_bundle(run, name, exploit["epoch"], self.members[name]["lr"])
            strategy.prepare_boundary(run, exploit)
            record_cadenced_decision(run, exploit)
            events = read_events(run)
            self.assertEqual([event["action"] for event in events], ["no_op", "copy_and_mutation"])
            self.assertEqual(events[0]["reason"], "warmup")
            self.assertIsNotNone(events[1]["pre_copy_checkpoint"])
            self.assertEqual(events[1]["decision_margin"], 0.002)


if __name__ == "__main__":
    unittest.main()
