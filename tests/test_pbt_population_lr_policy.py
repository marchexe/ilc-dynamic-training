import tempfile
import unittest
from pathlib import Path

from tests.helpers import pbt_smoke_config
from training.pbt import planning
from training.pbt.state import checkpointing, transitions


def _members(lrs):
    return {name: {"name": name, "lr": lr, "parent": None} for name, lr in lrs.items()}


def _monitor_round(generation, values, uncertainty=0.05):
    return {
        "generation": generation,
        "tier": "monitor",
        "members": {
            name: {
                "status": "completed",
                "metrics": {
                    "validation_bkg_rejection_score": value,
                    "validation_bkg_rejection_score_uncertainty": uncertainty,
                },
            }
            for name, value in values.items()
        },
    }


def _generation_record(index, epoch, members, control_values=None):
    control_values = control_values or {name: 1.0 for name in members}
    return {
        "index": index,
        "epoch": epoch,
        "workers": {
            name: {"status": "completed", "metrics": {"validation_bkg_rejection_score": value}}
            for name, value in control_values.items()
        },
        "ranking": None,
    }


def _policy_config(**overrides):
    policy = {
        "mode": "active",
        "eval_tier": "monitor",
        "up_factor": 1.1,
        "down_factor": 0.9,
        "direction_sigma": 1.0,
    }
    policy.update(overrides)
    return policy


class PopulationLrPolicyDirectionTest(unittest.TestCase):
    def _config(self, **policy_overrides):
        config = pbt_smoke_config()
        config["pbt"].update(
            min_lr=1.0e-6,
            max_lr=1.0e-3,
            population_lr_policy=_policy_config(**policy_overrides),
        )
        return config

    def _members(self):
        return _members(
            {
                "m_low_a": 1.0e-5,
                "m_low_b": 2.0e-5,
                "m_high_a": 8.0e-5,
                "m_high_b": 9.0e-5,
            }
        )

    def test_higher_lr_wins_increases_all_lrs(self):
        config = self._config()
        members = self._members()
        manifest = {
            "members": members,
            "generations": [],
            "tiered_evaluations": [
                _monitor_round(
                    0,
                    {"m_low_a": 5.0, "m_low_b": 5.2, "m_high_a": 6.0, "m_high_b": 6.3},
                )
            ],
        }
        generation = _generation_record(0, 5, members)

        ranking, plan = planning.population_lr_policy_plan(config, generation, members, manifest)

        self.assertEqual(len(plan), 4)
        for event in plan:
            self.assertEqual(event["direction"], "up")
            self.assertEqual(event["donor"], "m_high_b")
            self.assertAlmostEqual(event["new_lr"], members[event["recipient"]]["lr"] * 1.1)
        self.assertEqual(generation["population_lr_policy"]["decision"], "up")
        self.assertEqual(generation["population_lr_policy"]["donor"], "m_high_b")

    def test_lower_lr_wins_decreases_all_lrs(self):
        config = self._config()
        members = self._members()
        manifest = {
            "members": members,
            "generations": [],
            "tiered_evaluations": [
                _monitor_round(
                    0,
                    {"m_low_a": 7.0, "m_low_b": 7.5, "m_high_a": 5.0, "m_high_b": 5.5},
                )
            ],
        }
        generation = _generation_record(0, 5, members)

        ranking, plan = planning.population_lr_policy_plan(config, generation, members, manifest)

        self.assertEqual(len(plan), 4)
        for event in plan:
            self.assertEqual(event["direction"], "down")
            self.assertEqual(event["donor"], "m_low_b")
            self.assertAlmostEqual(event["new_lr"], members[event["recipient"]]["lr"] * 0.9)

    def test_no_clear_winner_keeps_lr_unchanged(self):
        config = self._config()
        members = self._members()
        manifest = {
            "members": members,
            "generations": [],
            "tiered_evaluations": [
                _monitor_round(
                    0,
                    {"m_low_a": 6.00, "m_low_b": 6.02, "m_high_a": 6.01, "m_high_b": 6.03},
                )
            ],
        }
        generation = _generation_record(0, 5, members)
        original_lrs = {name: member["lr"] for name, member in members.items()}

        ranking, plan = planning.population_lr_policy_plan(config, generation, members, manifest)

        self.assertEqual(plan, [])
        self.assertEqual(generation["population_lr_policy"]["decision"], "keep")
        self.assertEqual({name: member["lr"] for name, member in members.items()}, original_lrs)

    def test_no_fresh_monitor_round_is_a_no_op(self):
        config = self._config()
        members = self._members()
        manifest = {"members": members, "generations": [], "tiered_evaluations": []}
        generation = _generation_record(0, 5, members)

        ranking, plan = planning.population_lr_policy_plan(config, generation, members, manifest)

        self.assertEqual(plan, [])
        self.assertNotIn("population_lr_policy", generation)

    def test_decision_uses_monitor_tier_not_control_tier(self):
        # Control-tier (per-worker) metrics point "down" (low-LR half
        # winning); the monitor-tier round attached to this generation
        # points "up". Only the monitor round may drive the decision.
        config = self._config()
        members = self._members()
        manifest = {
            "members": members,
            "generations": [],
            "tiered_evaluations": [
                _monitor_round(
                    0,
                    {"m_low_a": 5.0, "m_low_b": 5.2, "m_high_a": 6.0, "m_high_b": 6.3},
                )
            ],
        }
        control_values = {"m_low_a": 9.0, "m_low_b": 9.5, "m_high_a": 1.0, "m_high_b": 1.2}
        generation = _generation_record(0, 5, members, control_values=control_values)

        ranking, plan = planning.population_lr_policy_plan(config, generation, members, manifest)

        self.assertTrue(plan)
        for event in plan:
            self.assertEqual(event["direction"], "up")

    def test_copy_direction_ignores_exploit_significance_sigma(self):
        # exploit_mutate's significance gate must never be consulted by this
        # strategy -- set it to a value that would block virtually any
        # exploit_mutate copy and confirm the decision/plan is unaffected.
        config = self._config()
        config["pbt"]["exploit_significance_sigma"] = 100.0
        members = self._members()
        manifest = {
            "members": members,
            "generations": [],
            "tiered_evaluations": [
                _monitor_round(
                    0,
                    {"m_low_a": 5.0, "m_low_b": 5.2, "m_high_a": 6.0, "m_high_b": 6.3},
                )
            ],
        }
        generation = _generation_record(0, 5, members)

        ranking, plan = planning.population_lr_policy_plan(config, generation, members, manifest)

        self.assertEqual(len(plan), 4)
        self.assertTrue(all(event["donor"] == "m_high_b" for event in plan))


class PopulationLrPolicyApplyTest(unittest.TestCase):
    def _config(self):
        config = pbt_smoke_config()
        config["pbt"].update(min_lr=1.0e-6, max_lr=1.0e-3, population_lr_policy=_policy_config())
        return config

    def test_accepted_decision_copies_weight_and_optimizer_to_every_recipient(self):
        config = self._config()
        members = _members({"m_low_a": 1.0e-5, "m_low_b": 2.0e-5, "m_high_a": 8.0e-5, "m_high_b": 9.0e-5})
        manifest = {
            "members": members,
            "generations": [],
            "tiered_evaluations": [
                _monitor_round(0, {"m_low_a": 5.0, "m_low_b": 5.2, "m_high_a": 6.0, "m_high_b": 6.3})
            ],
        }
        generation = _generation_record(0, 5, members)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contents = {}
            for name in members:
                (root / name).mkdir()
                state_path, optimizer_path = checkpointing.checkpoint_paths(root / name, 5)
                contents[name] = f"{name}-state".encode()
                state_path.write_bytes(contents[name])
                optimizer_path.write_bytes(f"{name}-optimizer".encode())

            ranking, plan = planning.population_lr_policy_plan(config, generation, members, manifest)
            generation["exploit"] = plan
            manifest_path = root / "manifest.json"
            transitions.apply_exploit(root, manifest, generation, manifest_path)

            for name in members:
                self.assertTrue(all(event["applied"] for event in generation["exploit"] if event["recipient"] == name))
                state_path, optimizer_path = checkpointing.checkpoint_paths(root / name, 5)
                self.assertEqual(state_path.read_bytes(), contents["m_high_b"])
                self.assertEqual(optimizer_path.read_bytes(), b"m_high_b-optimizer")
                # Pre-copy state must survive under the snapshot path so a
                # later rollback has something distinct from the donor to
                # restore -- this is what the non-donor recipients lost by
                # having net_epoch-5_* overwritten above.
                snapshot_state, snapshot_optimizer = checkpointing.population_lr_policy_snapshot_paths(
                    root / name, 5
                )
                self.assertEqual(snapshot_state.read_bytes(), contents[name])
            self.assertAlmostEqual(manifest["members"]["m_low_a"]["lr"], 1.0e-5 * 1.1)
            self.assertAlmostEqual(manifest["members"]["m_high_b"]["lr"], 9.0e-5 * 1.1)

    def test_rollback_restores_previous_weights_optimizer_and_lr_when_all_worse(self):
        config = self._config()
        members = _members({"m_low_a": 1.0e-5, "m_low_b": 2.0e-5, "m_high_a": 8.0e-5, "m_high_b": 9.0e-5})
        original_lrs = {name: member["lr"] for name, member in members.items()}
        manifest = {
            "members": members,
            "generations": [],
            "tiered_evaluations": [
                _monitor_round(0, {"m_low_a": 5.0, "m_low_b": 5.2, "m_high_a": 6.0, "m_high_b": 6.3})
            ],
        }
        generation0 = _generation_record(0, 5, members)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original_contents = {}
            for name in members:
                (root / name).mkdir()
                state_path, optimizer_path = checkpointing.checkpoint_paths(root / name, 5)
                original_contents[name] = f"{name}-gen0-state".encode()
                state_path.write_bytes(original_contents[name])
                optimizer_path.write_bytes(f"{name}-gen0-optimizer".encode())

            manifest_path = root / "manifest.json"

            ranking0, plan0 = planning.population_lr_policy_plan(config, generation0, members, manifest)
            generation0["exploit"] = plan0
            transitions.apply_exploit(root, manifest, generation0, manifest_path)
            manifest["generations"].append(generation0)
            post_decision_lrs = {name: member["lr"] for name, member in members.items()}
            self.assertNotEqual(post_decision_lrs, original_lrs)

            # One monitor interval later: every member trained further from
            # the copied state and got worse across the board.
            for name in members:
                state_path, optimizer_path = checkpointing.checkpoint_paths(root / name, 9)
                state_path.write_bytes(f"{name}-gen1-state".encode())
                optimizer_path.write_bytes(f"{name}-gen1-optimizer".encode())
            manifest["tiered_evaluations"].append(
                _monitor_round(1, {"m_low_a": 3.0, "m_low_b": 3.1, "m_high_a": 3.2, "m_high_b": 3.3})
            )
            generation1 = _generation_record(1, 9, members)

            ranking1, plan1 = planning.population_lr_policy_plan(config, generation1, members, manifest)
            self.assertTrue(all(event["source"] == "population_lr_policy_resolution" for event in plan1))
            self.assertTrue(all(event["outcome"] == "rolled_back" for event in plan1))

            generation1["exploit"] = plan1
            transitions.apply_exploit(root, manifest, generation1, manifest_path)

            for name in members:
                state_path, optimizer_path = checkpointing.checkpoint_paths(root / name, 9)
                self.assertEqual(state_path.read_bytes(), original_contents[name])
                self.assertEqual(optimizer_path.read_bytes(), f"{name}-gen0-optimizer".encode())
                self.assertAlmostEqual(manifest["members"][name]["lr"], original_lrs[name])
                # The snapshot's only consumer (this rollback) has used it;
                # it must not linger on disk indefinitely.
                snapshot_state, snapshot_optimizer = checkpointing.population_lr_policy_snapshot_paths(
                    root / name, 5
                )
                self.assertFalse(snapshot_state.exists())
                self.assertFalse(snapshot_optimizer.exists())

    def test_accepted_resolution_is_a_no_op(self):
        config = self._config()
        members = _members({"m_low_a": 1.0e-5, "m_low_b": 2.0e-5, "m_high_a": 8.0e-5, "m_high_b": 9.0e-5})
        manifest = {
            "members": members,
            "generations": [],
            "tiered_evaluations": [
                _monitor_round(0, {"m_low_a": 5.0, "m_low_b": 5.2, "m_high_a": 6.0, "m_high_b": 6.3})
            ],
        }
        generation0 = _generation_record(0, 5, members)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in members:
                (root / name).mkdir()
                state_path, optimizer_path = checkpointing.checkpoint_paths(root / name, 5)
                state_path.write_bytes(f"{name}-gen0-state".encode())
                optimizer_path.write_bytes(f"{name}-gen0-optimizer".encode())
            manifest_path = root / "manifest.json"

            ranking0, plan0 = planning.population_lr_policy_plan(config, generation0, members, manifest)
            generation0["exploit"] = plan0
            transitions.apply_exploit(root, manifest, generation0, manifest_path)
            manifest["generations"].append(generation0)
            lrs_after_decision = {name: member["lr"] for name, member in members.items()}

            for name in members:
                state_path, optimizer_path = checkpointing.checkpoint_paths(root / name, 9)
                state_path.write_bytes(f"{name}-gen1-state".encode())
                optimizer_path.write_bytes(f"{name}-gen1-optimizer".encode())
            # Everyone improved on metric_before (6.3) this round, but the
            # high/low halves are too close to each other to also start a
            # fresh forward decision -- isolates the accept/no-op path from
            # the (separately tested) "accept, then decide again" cascade.
            manifest["tiered_evaluations"].append(
                _monitor_round(1, {"m_low_a": 6.40, "m_low_b": 6.42, "m_high_a": 6.41, "m_high_b": 6.43})
            )
            generation1 = _generation_record(1, 9, members)

            ranking1, plan1 = planning.population_lr_policy_plan(config, generation1, members, manifest)
            self.assertEqual(len(plan1), 4)
            self.assertTrue(all(event["source"] == "population_lr_policy_resolution" for event in plan1))
            self.assertTrue(all(event["outcome"] == "accepted" for event in plan1))

            generation1["exploit"] = plan1
            transitions.apply_exploit(root, manifest, generation1, manifest_path)

            for name in members:
                state_path, _ = checkpointing.checkpoint_paths(root / name, 9)
                self.assertEqual(state_path.read_bytes(), f"{name}-gen1-state".encode())
                self.assertAlmostEqual(manifest["members"][name]["lr"], lrs_after_decision[name])
                # Accept is also terminal for the snapshot -- it must be
                # cleaned up here too, not only on the rollback path.
                snapshot_state, snapshot_optimizer = checkpointing.population_lr_policy_snapshot_paths(
                    root / name, 5
                )
                self.assertFalse(snapshot_state.exists())
                self.assertFalse(snapshot_optimizer.exists())


class PopulationLrPolicyResumeSafetyTest(unittest.TestCase):
    """A crashed run resumes by re-entering the exact same generation and
    calling apply_exploit again on the same persisted plan (runner.py only
    skips re-planning when generation_record["exploit"] is already a real
    list -- it never skips re-applying). These tests simulate that by
    partially applying a plan (mutating a prefix of its events in place,
    exactly as a real partial apply_exploit run would leave them) and then
    calling apply_exploit again on the *full* plan, verifying the second
    call finishes correctly and produces the same end state as an
    uninterrupted run.
    """

    def _config(self):
        config = pbt_smoke_config()
        config["pbt"].update(min_lr=1.0e-6, max_lr=1.0e-3, population_lr_policy=_policy_config())
        return config

    def test_forward_decision_resumes_correctly_after_a_partial_apply(self):
        config = self._config()
        members = _members({"m_low_a": 1.0e-5, "m_low_b": 2.0e-5, "m_high_a": 8.0e-5, "m_high_b": 9.0e-5})
        manifest = {
            "members": members,
            "generations": [],
            "tiered_evaluations": [
                _monitor_round(0, {"m_low_a": 5.0, "m_low_b": 5.2, "m_high_a": 6.0, "m_high_b": 6.3})
            ],
        }
        generation = _generation_record(0, 5, members)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contents = {}
            for name in members:
                (root / name).mkdir()
                state_path, optimizer_path = checkpointing.checkpoint_paths(root / name, 5)
                contents[name] = f"{name}-state".encode()
                state_path.write_bytes(contents[name])
                optimizer_path.write_bytes(f"{name}-optimizer".encode())

            ranking, plan = planning.population_lr_policy_plan(config, generation, members, manifest)
            self.assertEqual(len(plan), 4)
            manifest_path = root / "manifest.json"

            # "Crash": only the first two events of this generation's plan
            # ever got applied and persisted before the process died.
            generation["exploit"] = plan
            partial = dict(generation)
            partial["exploit"] = plan[:2]
            transitions.apply_exploit(root, manifest, partial, manifest_path)
            self.assertTrue(plan[0]["applied"])
            self.assertTrue(plan[1]["applied"])
            self.assertFalse(plan[2]["applied"])
            self.assertFalse(plan[3]["applied"])

            # "Resume": runner.py reloads the manifest (exploit is already a
            # real list, so it is never rebuilt) and calls apply_exploit
            # again on the full, mixed-progress plan.
            transitions.apply_exploit(root, manifest, generation, manifest_path)

            for name in members:
                self.assertTrue(all(event["applied"] for event in plan if event["recipient"] == name))
                state_path, optimizer_path = checkpointing.checkpoint_paths(root / name, 5)
                self.assertEqual(state_path.read_bytes(), contents["m_high_b"])
                self.assertEqual(optimizer_path.read_bytes(), b"m_high_b-optimizer")
                snapshot_state, _ = checkpointing.population_lr_policy_snapshot_paths(root / name, 5)
                self.assertEqual(snapshot_state.read_bytes(), contents[name])
            self.assertAlmostEqual(manifest["members"]["m_low_a"]["lr"], 1.0e-5 * 1.1)

    def test_reapplying_a_fully_applied_plan_is_a_pure_noop(self):
        config = self._config()
        members = _members({"m_low_a": 1.0e-5, "m_low_b": 2.0e-5, "m_high_a": 8.0e-5, "m_high_b": 9.0e-5})
        manifest = {
            "members": members,
            "generations": [],
            "tiered_evaluations": [
                _monitor_round(0, {"m_low_a": 5.0, "m_low_b": 5.2, "m_high_a": 6.0, "m_high_b": 6.3})
            ],
        }
        generation = _generation_record(0, 5, members)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in members:
                (root / name).mkdir()
                state_path, optimizer_path = checkpointing.checkpoint_paths(root / name, 5)
                state_path.write_bytes(f"{name}-state".encode())
                optimizer_path.write_bytes(f"{name}-optimizer".encode())

            ranking, plan = planning.population_lr_policy_plan(config, generation, members, manifest)
            generation["exploit"] = plan
            manifest_path = root / "manifest.json"
            transitions.apply_exploit(root, manifest, generation, manifest_path)
            lrs_after_first_pass = {name: member["lr"] for name, member in members.items()}
            state_bytes_after_first_pass = {
                name: checkpointing.checkpoint_paths(root / name, 5)[0].read_bytes() for name in members
            }

            # Simulates runner.py re-entering an already-fully-applied
            # generation on resume (e.g. it crashed later in the same
            # generation loop iteration, after apply_exploit but before
            # next_generation advanced).
            transitions.apply_exploit(root, manifest, generation, manifest_path)

            for name in members:
                self.assertAlmostEqual(manifest["members"][name]["lr"], lrs_after_first_pass[name])
                state_path, _ = checkpointing.checkpoint_paths(root / name, 5)
                self.assertEqual(state_path.read_bytes(), state_bytes_after_first_pass[name])

    def test_preexisting_snapshot_from_a_crashed_attempt_is_not_overwritten(self):
        # Simulates: snapshot step completed and was persisted to disk, but
        # the process crashed before the main donor copy ran (event still
        # applied=False in the last-persisted manifest). On a real resume
        # the snapshot would already exactly equal the recipient's current
        # (still pre-copy) checkpoint, so re-snapshotting would be harmless
        # -- this test instead pins the guard's *invariant* directly by
        # deliberately pre-seeding a snapshot whose content differs from the
        # current recipient checkpoint, so any accidental re-snapshot would
        # be immediately visible as a changed snapshot below.
        config = self._config()
        members = _members({"m_low_a": 1.0e-5, "m_low_b": 2.0e-5, "m_high_a": 8.0e-5, "m_high_b": 9.0e-5})
        manifest = {
            "members": members,
            "generations": [],
            "tiered_evaluations": [
                _monitor_round(0, {"m_low_a": 5.0, "m_low_b": 5.2, "m_high_a": 6.0, "m_high_b": 6.3})
            ],
        }
        generation = _generation_record(0, 5, members)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in members:
                (root / name).mkdir()
                state_path, optimizer_path = checkpointing.checkpoint_paths(root / name, 5)
                state_path.write_bytes(f"{name}-state".encode())
                optimizer_path.write_bytes(f"{name}-optimizer".encode())

            ranking, plan = planning.population_lr_policy_plan(config, generation, members, manifest)
            generation["exploit"] = plan
            manifest_path = root / "manifest.json"

            # Pre-create m_low_a's snapshot exactly as the pre-crash attempt
            # would have left it, before ever calling apply_exploit.
            snapshot_state, snapshot_optimizer = checkpointing.population_lr_policy_snapshot_paths(
                root / "m_low_a", 5
            )
            snapshot_state.write_bytes(b"preexisting-snapshot-state")
            snapshot_optimizer.write_bytes(b"preexisting-snapshot-optimizer")

            transitions.apply_exploit(root, manifest, generation, manifest_path)

            self.assertEqual(snapshot_state.read_bytes(), b"preexisting-snapshot-state")
            self.assertEqual(snapshot_optimizer.read_bytes(), b"preexisting-snapshot-optimizer")
            state_path, optimizer_path = checkpointing.checkpoint_paths(root / "m_low_a", 5)
            self.assertEqual(state_path.read_bytes(), b"m_high_b-state")
            self.assertEqual(optimizer_path.read_bytes(), b"m_high_b-optimizer")


class PopulationLrPolicyLegacyBehaviorTest(unittest.TestCase):
    def test_disabled_by_default_and_strategy_dispatch_unchanged(self):
        config = pbt_smoke_config()
        self.assertIsNone(planning.population_lr_policy_config(config))
        self.assertEqual(config["pbt"].get("strategy", "exploit_mutate"), "exploit_mutate")
        self.assertIs(planning.STRATEGY_PLANNERS["exploit_mutate"], planning.exploit_mutate_plan)
        self.assertIs(planning.STRATEGY_PLANNERS["population_lr_policy"], planning.population_lr_policy_plan)


if __name__ == "__main__":
    unittest.main()
