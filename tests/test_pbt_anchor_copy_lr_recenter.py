import math
import tempfile
import unittest
from pathlib import Path

import torch

from tests.helpers import pbt_smoke_config, PROJECT_DIR, namespace
from training.pbt import config as config_module
from training.pbt.planning import dispatch
from training.pbt.planning.anchor_copy_lr_recenter import (
    anchor_copy_lr_recenter_config,
    anchor_copy_lr_recenter_plan,
    detect_spread_collapse,
    should_apply_exploit_for_strategy,
)
from training.pbt.planning.rollbacks import strategy_uses_population_rollbacks
from training.pbt.models.manifest import PBTManifest
from training.pbt.state import anchor as anchor_module
from training.pbt.state import checkpointing, transitions

METRIC = "validation_working_point_mistag_percent"


def _members(lrs):
    return {name: {"name": name, "lr": lr, "parent": None} for name, lr in lrs.items()}


def _generation_record(index, epoch, metrics_by_member):
    return {
        "index": index,
        "epoch": epoch,
        "workers": {
            name: {"status": "completed", "metrics": {METRIC: value}}
            for name, value in metrics_by_member.items()
        },
        "ranking": None,
    }


def _anchor(member, metric_value, lr_center, generation=-1):
    return {
        "member": member,
        "generation": generation,
        "metric": METRIC,
        "mode": "min",
        "metric_value": metric_value,
        "lr_center": lr_center,
        "eval_tier": "control",
    }


def _write_optimizer_state(path, *, lr, marker):
    """A minimal, genuinely torch-serializable optimizer checkpoint --
    real content is required now that apply_exploit's anchor_copy_lr_recenter
    path loads and rewrites param_groups[*]['lr'] on every copy (see
    optimizer_state.py::atomic_set_optimizer_lr), not just raw bytes."""
    torch.save({"marker": marker, "state": {}, "param_groups": [{"lr": lr}]}, path)


def _read_optimizer_state(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _policy_config(**overrides):
    policy = {
        "mode": "active",
        "accept_tolerance": 0.01,
        "spread_multipliers": [0.80, 0.90, 1.00, 1.20],
    }
    policy.update(overrides)
    return policy


def _config(**policy_overrides):
    config = pbt_smoke_config()
    # This strategy's real preset never sets training_controller (matching
    # the base 10m proxy-control shared preset) -- pbt_smoke_config()'s
    # default is a real, truthy path, which would make apply_exploit's
    # generic per-member logic expect a controller checkpoint file no test
    # here creates. Match real usage instead of adding unrelated fixture
    # files.
    config["shared"]["training_controller"] = None
    config["pbt"].update(
        metric=METRIC,
        mode="min",
        min_lr=1.0e-6,
        max_lr=1.0e-3,
        anchor_copy_lr_recenter=_policy_config(**policy_overrides),
    )
    return config


class AnchorCopyLrRecenterPlannerTest(unittest.TestCase):
    def test_highest_lr_stream_wins_moves_center_upward(self):
        config = _config()
        members = _members({"m_low": 1.0e-5, "m_mid": 5.0e-5, "m_high": 9.0e-5})
        generation = _generation_record(1, 6, {"m_low": 1.2, "m_mid": 1.1, "m_high": 0.9})  # min mode: m_high has best (lowest) metric
        manifest = {"members": members, "generations": [], "anchor": _anchor("m_mid", 1.15, 5.0e-5)}

        ranking, plan = anchor_copy_lr_recenter_plan(config, generation, members, manifest)

        self.assertEqual(ranking[0], "m_high")
        info = generation["anchor_copy_lr_recenter"]
        self.assertEqual(info["decision"], "accepted_new_anchor")
        self.assertAlmostEqual(info["previous_lr_center"], 5.0e-5)
        # new_lr_center = winner_lr exactly -- no damping toward it.
        self.assertAlmostEqual(info["new_lr_center"], 9.0e-5)
        self.assertAlmostEqual(info["winner_lr"], 9.0e-5)
        self.assertGreater(info["new_lr_center"], info["previous_lr_center"])
        recipients = {event["recipient"] for event in plan if event["recipient"] != "__anchor__"}
        self.assertEqual(recipients, set(members))
        self.assertTrue(all(event["donor"] == "m_high" for event in plan))

    def test_lowest_lr_stream_wins_moves_center_downward(self):
        config = _config()
        members = _members({"m_low": 1.0e-5, "m_mid": 5.0e-5, "m_high": 9.0e-5})
        generation = _generation_record(1, 6, {"m_low": 0.9, "m_mid": 1.1, "m_high": 1.2})  # m_low wins
        manifest = {"members": members, "generations": [], "anchor": _anchor("m_mid", 1.15, 5.0e-5)}

        ranking, plan = anchor_copy_lr_recenter_plan(config, generation, members, manifest)

        self.assertEqual(ranking[0], "m_low")
        info = generation["anchor_copy_lr_recenter"]
        self.assertEqual(info["decision"], "accepted_new_anchor")
        self.assertAlmostEqual(info["new_lr_center"], 1.0e-5)
        self.assertLess(info["new_lr_center"], info["previous_lr_center"])

    def test_middle_stream_wins_center_lands_exactly_on_winner_lr(self):
        config = _config()
        members = _members({"m_low": 1.0e-5, "m_mid": 5.0e-5, "m_high": 9.0e-5})
        generation = _generation_record(1, 6, {"m_low": 1.2, "m_mid": 0.9, "m_high": 1.1})  # m_mid wins
        # Prior center far below the winner's LR: the new center must land
        # exactly on m_mid's own LR, not partway there -- no damping
        # fraction exists in this strategy's spec.
        manifest = {"members": members, "generations": [], "anchor": _anchor("m_low", 1.15, 1.0e-6)}

        ranking, plan = anchor_copy_lr_recenter_plan(config, generation, members, manifest)

        self.assertEqual(ranking[0], "m_mid")
        info = generation["anchor_copy_lr_recenter"]
        self.assertAlmostEqual(info["new_lr_center"], 5.0e-5)
        self.assertAlmostEqual(info["winner_lr"], 5.0e-5)

    def test_small_metric_difference_still_selects_a_winner_and_still_copies(self):
        """No significance gate: even a tie-zone (within accept_tolerance)
        result must still select a single winner and still build a copy
        event for every member -- distribution is never conditional on the
        margin being large."""
        config = _config(accept_tolerance=0.05)
        members = _members({"m_a": 1.0e-5, "m_b": 8.0e-5})
        generation = _generation_record(1, 6, {"m_a": 1.001, "m_b": 1.000})  # tiny difference
        manifest = {"members": members, "generations": [], "anchor": _anchor("m_a", 1.0005, 3.0e-5)}

        ranking, plan = anchor_copy_lr_recenter_plan(config, generation, members, manifest)

        self.assertEqual(ranking[0], "m_b")
        recipients = {event["recipient"] for event in plan if event["recipient"] != "__anchor__"}
        self.assertEqual(recipients, set(members))
        # within tolerance of the anchor -> reused, not accepted -- but the
        # copy still happens identically either way.
        self.assertEqual(generation["anchor_copy_lr_recenter"]["decision"], "reused_previous_anchor")

    def test_reused_previous_anchor_keeps_old_anchor_metric_but_still_moves_center(self):
        config = _config(accept_tolerance=0.05)
        members = _members({"m_a": 1.0e-5, "m_b": 8.0e-5})
        generation = _generation_record(1, 6, {"m_a": 1.02, "m_b": 0.99})  # m_b wins, within 5% of anchor's 1.0
        manifest = {"members": members, "generations": [], "anchor": _anchor("m_a", 1.0, 2.0e-5)}

        ranking, plan = anchor_copy_lr_recenter_plan(config, generation, members, manifest)

        info = generation["anchor_copy_lr_recenter"]
        self.assertEqual(info["decision"], "reused_previous_anchor")
        self.assertAlmostEqual(info["anchor_metric_value"], 1.0)  # old anchor's metric preserved, not the winner's 0.99
        self.assertAlmostEqual(info["new_lr_center"], 8.0e-5)  # == winner_lr exactly, no damping
        self.assertGreater(info["new_lr_center"], info["previous_lr_center"])  # center still moves despite reuse
        # manifest["anchor"] itself is untouched by the planner (only apply_exploit's
        # special case mutates it) -- confirmed unchanged here.
        self.assertEqual(manifest["anchor"]["metric_value"], 1.0)
        self.assertEqual(manifest["anchor"]["lr_center"], 2.0e-5)

    def test_all_streams_worse_than_anchor_rewinds_and_restores_center(self):
        config = _config(accept_tolerance=0.01)
        members = _members({"m_a": 1.0e-5, "m_b": 8.0e-5})
        generation = _generation_record(1, 6, {"m_a": 2.0, "m_b": 1.8})  # both much worse than anchor's 1.0
        manifest = {"members": members, "generations": [], "anchor": _anchor("m_prev", 1.0, 3.0e-5)}

        ranking, plan = anchor_copy_lr_recenter_plan(config, generation, members, manifest)

        info = generation["anchor_copy_lr_recenter"]
        self.assertEqual(info["decision"], "rewound_to_previous_anchor")
        self.assertAlmostEqual(info["new_lr_center"], 3.0e-5)  # restored, not moved
        self.assertAlmostEqual(info["previous_lr_center"], 3.0e-5)
        recipients = {event["recipient"] for event in plan if event["recipient"] != "__anchor__"}
        self.assertEqual(recipients, set(members))
        self.assertTrue(all(event["donor"] == "m_prev" for event in plan))
        # not just the whole-population fraction -- every member is a recipient.
        self.assertEqual(len(recipients), len(members))

    def test_stream_better_than_anchor_becomes_new_anchor(self):
        config = _config(accept_tolerance=0.01)
        members = _members({"m_a": 1.0e-5, "m_b": 8.0e-5})
        generation = _generation_record(1, 6, {"m_a": 1.5, "m_b": 0.5})  # m_b clearly better than anchor's 1.0
        manifest = {"members": members, "generations": [], "anchor": _anchor("m_prev", 1.0, 3.0e-5)}

        ranking, plan = anchor_copy_lr_recenter_plan(config, generation, members, manifest)

        self.assertEqual(ranking[0], "m_b")
        info = generation["anchor_copy_lr_recenter"]
        self.assertEqual(info["decision"], "accepted_new_anchor")
        self.assertAlmostEqual(info["anchor_metric_value"], 0.5)
        recipients = {event["recipient"] for event in plan if event["recipient"] != "__anchor__"}
        self.assertEqual(recipients, set(members))
        self.assertTrue(all(event["donor"] == "m_b" for event in plan))

    def test_nan_or_inf_stream_excluded_and_cannot_win(self):
        config = _config()
        members = _members({"m_nan": 1.0e-5, "m_inf": 5.0e-5, "m_ok": 9.0e-5})
        generation = _generation_record(0, 5, {"m_nan": float("nan"), "m_inf": float("inf"), "m_ok": 1.0})
        manifest = {"members": members, "generations": [], "anchor": None}

        ranking, plan = anchor_copy_lr_recenter_plan(config, generation, members, manifest)

        self.assertEqual(ranking, ["m_ok"])
        self.assertNotIn("m_nan", ranking)
        self.assertNotIn("m_inf", ranking)
        self.assertEqual(generation["anchor_copy_lr_recenter"]["winner"], "m_ok")

    def test_all_streams_non_finite_triggers_no_action_and_leaves_anchor_untouched(self):
        config = _config()
        members = _members({"m_nan": 1.0e-5, "m_inf": 5.0e-5})
        generation = _generation_record(0, 5, {"m_nan": float("nan"), "m_inf": float("inf")})
        existing_anchor = _anchor("someone", 1.0, 3.0e-5)
        manifest = {"members": members, "generations": [], "anchor": dict(existing_anchor)}

        ranking, plan = anchor_copy_lr_recenter_plan(config, generation, members, manifest)

        self.assertEqual(ranking, [])
        self.assertEqual(plan, [])
        self.assertEqual(manifest["anchor"], existing_anchor)
        self.assertEqual(generation["anchor_copy_lr_recenter"]["decision"], "no_finite_metric")

    def test_missing_worker_metric_excluded_same_as_nan(self):
        config = _config()
        members = _members({"m_missing": 1.0e-5, "m_ok": 5.0e-5})
        generation = {
            "index": 0,
            "epoch": 5,
            "workers": {
                "m_missing": {"status": "failed", "metrics": None},
                "m_ok": {"status": "completed", "metrics": {METRIC: 1.0}},
            },
            "ranking": None,
        }
        manifest = {"members": members, "generations": [], "anchor": None}

        ranking, plan = anchor_copy_lr_recenter_plan(config, generation, members, manifest)

        self.assertEqual(ranking, ["m_ok"])

    def test_every_stream_receives_a_new_lr_around_the_common_center(self):
        multipliers = [0.80, 0.90, 1.00, 1.20]
        config = _config(spread_multipliers=multipliers)
        members = _members({"m_a": 1.0e-5, "m_b": 5.0e-5, "m_c": 8.0e-5, "m_d": 9.0e-5})
        generation = _generation_record(0, 5, {"m_a": 1.0, "m_b": 1.1, "m_c": 1.2, "m_d": 1.3})
        manifest = {"members": members, "generations": [], "anchor": None}

        ranking, plan = anchor_copy_lr_recenter_plan(config, generation, members, manifest)

        center = generation["anchor_copy_lr_recenter"]["new_lr_center"]
        member_events = {event["recipient"]: event for event in plan if event["recipient"] != "__anchor__"}
        self.assertEqual(len(member_events), len(members))
        for name, multiplier in zip(members, multipliers):
            self.assertAlmostEqual(member_events[name]["new_lr"], center * multiplier)
        assigned = [event["new_lr"] for event in member_events.values()]
        # Spread must include values below center, exactly at center (the
        # 1.0 multiplier), and above center.
        self.assertTrue(any(lr < center for lr in assigned))
        self.assertTrue(any(math.isclose(lr, center) for lr in assigned))
        self.assertTrue(any(lr > center for lr in assigned))

    def test_lr_bounds_are_respected(self):
        config = _config(spread_multipliers=[0.1, 5.0])
        config["pbt"].update(min_lr=1.0e-5, max_lr=1.0e-4)
        members = _members({"m_a": 5.0e-5, "m_b": 6.0e-5})
        generation = _generation_record(0, 5, {"m_a": 1.0, "m_b": 1.1})
        manifest = {"members": members, "generations": [], "anchor": None}

        ranking, plan = anchor_copy_lr_recenter_plan(config, generation, members, manifest)

        member_events = [event for event in plan if event["recipient"] != "__anchor__"]
        for event in member_events:
            self.assertGreaterEqual(event["new_lr"], 1.0e-5)
            self.assertLessEqual(event["new_lr"], 1.0e-4)
        anchor_event = next(event for event in plan if event["recipient"] == "__anchor__")
        self.assertGreaterEqual(anchor_event["lr_center"], 1.0e-5)
        self.assertLessEqual(anchor_event["lr_center"], 1.0e-4)


class AnchorCopyLrRecenterMomentumTest(unittest.TestCase):
    """recenter_momentum_fraction: an extra push past winner_lr itself, on
    a genuine accepted_new_anchor only."""

    def test_low_lr_winner_pushes_center_further_down(self):
        config = _config(recenter_momentum_fraction=0.10)
        members = _members({"m_low": 1.0e-5, "m_high": 9.0e-5})
        generation = _generation_record(1, 6, {"m_low": 0.5, "m_high": 1.5})  # m_low wins, clearly better
        manifest = {"members": members, "generations": [], "anchor": _anchor("m_prev", 1.0, 5.0e-5, generation=0)}

        anchor_copy_lr_recenter_plan(config, generation, members, manifest)

        info = generation["anchor_copy_lr_recenter"]
        self.assertEqual(info["decision"], "accepted_new_anchor")
        # winner_lr (1.0e-5) pushed a further 10% below itself, not just
        # landed on it -- direction matches "low LR won, go lower still".
        self.assertAlmostEqual(info["new_lr_center"], 1.0e-5 * 0.90)
        self.assertLess(info["new_lr_center"], info["winner_lr"])

    def test_high_lr_winner_pushes_center_further_up(self):
        config = _config(recenter_momentum_fraction=0.10)
        members = _members({"m_low": 1.0e-5, "m_high": 9.0e-5})
        generation = _generation_record(1, 6, {"m_low": 1.5, "m_high": 0.5})  # m_high wins
        manifest = {"members": members, "generations": [], "anchor": _anchor("m_prev", 1.0, 5.0e-5, generation=0)}

        anchor_copy_lr_recenter_plan(config, generation, members, manifest)

        info = generation["anchor_copy_lr_recenter"]
        self.assertAlmostEqual(info["new_lr_center"], 9.0e-5 * 1.10)
        self.assertGreater(info["new_lr_center"], info["winner_lr"])

    def test_no_push_when_winner_lr_equals_previous_center(self):
        config = _config(recenter_momentum_fraction=0.10)
        members = _members({"m_mid": 5.0e-5, "m_other": 9.0e-5})
        generation = _generation_record(1, 6, {"m_mid": 0.5, "m_other": 1.5})  # m_mid wins, lr == prev center
        manifest = {"members": members, "generations": [], "anchor": _anchor("m_prev", 1.0, 5.0e-5, generation=0)}

        anchor_copy_lr_recenter_plan(config, generation, members, manifest)

        # No direction to extrapolate -- lands exactly on winner_lr, same
        # as momentum disabled.
        self.assertAlmostEqual(generation["anchor_copy_lr_recenter"]["new_lr_center"], 5.0e-5)

    def test_no_push_on_the_very_first_ever_accept(self):
        config = _config(recenter_momentum_fraction=0.10)
        members = _members({"m_a": 1.0e-5, "m_b": 9.0e-5})
        generation = _generation_record(0, 5, {"m_a": 1.5, "m_b": 0.5})
        manifest = {"members": members, "generations": [], "anchor": None}

        anchor_copy_lr_recenter_plan(config, generation, members, manifest)

        info = generation["anchor_copy_lr_recenter"]
        self.assertEqual(info["decision"], "accepted_new_anchor")
        self.assertAlmostEqual(info["new_lr_center"], 9.0e-5)  # no spurious push off an arbitrary start

    def test_no_push_on_reused_previous_anchor_tie_zone(self):
        config = _config(accept_tolerance=0.05, recenter_momentum_fraction=0.10)
        members = _members({"m_a": 1.0e-5, "m_b": 8.0e-5})
        generation = _generation_record(1, 6, {"m_a": 1.02, "m_b": 0.99})  # within tolerance -- reuse, not accept
        manifest = {"members": members, "generations": [], "anchor": _anchor("m_a", 1.0, 2.0e-5, generation=0)}

        anchor_copy_lr_recenter_plan(config, generation, members, manifest)

        info = generation["anchor_copy_lr_recenter"]
        self.assertEqual(info["decision"], "reused_previous_anchor")
        self.assertAlmostEqual(info["new_lr_center"], 8.0e-5)  # == winner_lr exactly, momentum not applied

    def test_momentum_disabled_by_default_matches_original_behavior(self):
        config = _config()  # recenter_momentum_fraction defaults to 0.0
        members = _members({"m_low": 1.0e-5, "m_high": 9.0e-5})
        generation = _generation_record(1, 6, {"m_low": 0.5, "m_high": 1.5})
        manifest = {"members": members, "generations": [], "anchor": _anchor("m_prev", 1.0, 5.0e-5, generation=0)}

        anchor_copy_lr_recenter_plan(config, generation, members, manifest)

        self.assertAlmostEqual(generation["anchor_copy_lr_recenter"]["new_lr_center"], 1.0e-5)


class AnchorCopyLrRecenterPlateauEscapeTest(unittest.TestCase):
    """plateau_escape_after_generations: force-accept the winner instead of
    rewinding forever once a run has gone that many generations with no
    genuine accepted_new_anchor."""

    def test_rewinds_normally_below_the_threshold(self):
        config = _config(plateau_escape_after_generations=8)
        members = _members({"m_a": 1.0e-5, "m_b": 8.0e-5})
        generation = _generation_record(5, 6, {"m_a": 2.0, "m_b": 1.8})  # both worse than anchor
        # Anchor last accepted at generation 0 -- generation 5 is only 5
        # generations later, below the threshold of 8.
        manifest = {"members": members, "generations": [], "anchor": _anchor("m_prev", 1.0, 3.0e-5, generation=0)}

        anchor_copy_lr_recenter_plan(config, generation, members, manifest)

        self.assertEqual(generation["anchor_copy_lr_recenter"]["decision"], "rewound_to_previous_anchor")

    def test_force_accepts_once_the_threshold_is_reached(self):
        config = _config(plateau_escape_after_generations=8)
        members = _members({"m_a": 1.0e-5, "m_b": 8.0e-5})
        generation = _generation_record(8, 6, {"m_a": 2.0, "m_b": 1.8})  # still worse than anchor
        # Anchor last accepted at generation 0 -- generation 8 is exactly 8
        # generations later, at the threshold.
        manifest = {"members": members, "generations": [], "anchor": _anchor("m_prev", 1.0, 3.0e-5, generation=0)}

        ranking, plan = anchor_copy_lr_recenter_plan(config, generation, members, manifest)

        info = generation["anchor_copy_lr_recenter"]
        self.assertEqual(info["decision"], "plateau_escape_accepted")
        # Forced accept lands on the winner's own LR (m_b, the better of
        # the two even though both are worse than anchor), not restored.
        self.assertEqual(ranking[0], "m_b")
        self.assertAlmostEqual(info["new_lr_center"], 8.0e-5)
        self.assertTrue(all(event["donor"] == "m_b" for event in plan if event["recipient"] != "__anchor__"))

    def test_disabled_by_default_rewinds_indefinitely(self):
        config = _config()  # plateau_escape_after_generations defaults to 0 (disabled)
        members = _members({"m_a": 1.0e-5, "m_b": 8.0e-5})
        generation = _generation_record(50, 6, {"m_a": 2.0, "m_b": 1.8})
        manifest = {"members": members, "generations": [], "anchor": _anchor("m_prev", 1.0, 3.0e-5, generation=0)}

        anchor_copy_lr_recenter_plan(config, generation, members, manifest)

        self.assertEqual(generation["anchor_copy_lr_recenter"]["decision"], "rewound_to_previous_anchor")

    def test_widens_spread_for_the_escape_generation_only(self):
        multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
        config = _config(
            spread_multipliers=multipliers, plateau_escape_after_generations=8, plateau_escape_widen_factor=2.0,
        )
        members = _members({"m_a": 1.0e-5, "m_b": 8.0e-5, "m_c": 3.0e-5, "m_d": 4.0e-5, "m_e": 5.0e-5})
        generation = _generation_record(8, 6, {"m_a": 2.0, "m_b": 1.8, "m_c": 2.1, "m_d": 2.2, "m_e": 2.3})
        manifest = {"members": members, "generations": [], "anchor": _anchor("m_prev", 1.0, 3.0e-5, generation=0)}

        anchor_copy_lr_recenter_plan(config, generation, members, manifest)

        info = generation["anchor_copy_lr_recenter"]
        self.assertEqual(info["decision"], "plateau_escape_accepted")
        center = info["new_lr_center"]
        # 2x widen_factor doubles each multiplier's deviation from 1.0:
        # 0.5 -> 0.0, 1.5 -> 2.0 -- the widest assigned LR should be
        # noticeably further from center than the un-widened 1.5x would
        # give.
        widest = max(info["unclamped_lrs"].values())
        self.assertAlmostEqual(widest, center * 2.0, places=12)

    def test_generations_since_accept_recorded_every_generation(self):
        config = _config()
        members = _members({"m_a": 1.0e-5, "m_b": 8.0e-5})
        generation = _generation_record(3, 6, {"m_a": 2.0, "m_b": 1.8})
        manifest = {"members": members, "generations": [], "anchor": _anchor("m_prev", 1.0, 3.0e-5, generation=0)}

        anchor_copy_lr_recenter_plan(config, generation, members, manifest)

        self.assertEqual(generation["anchor_copy_lr_recenter"]["generations_since_accept"], 3)


class AnchorCopyLrRecenterApplyTest(unittest.TestCase):
    def _config(self):
        return _config(accept_tolerance=0.01)

    def _write_member_checkpoint(self, root, name, epoch, state_bytes, optimizer_lr):
        (root / name).mkdir(exist_ok=True)
        state_path, optimizer_path = checkpointing.checkpoint_paths(root / name, epoch)
        state_path.write_bytes(state_bytes)
        _write_optimizer_state(optimizer_path, lr=optimizer_lr, marker=name)
        return state_path, optimizer_path

    def test_accepted_decision_copies_winner_weights_and_optimizer_to_every_stream(self):
        config = self._config()
        members = _members({"m_a": 1.0e-5, "m_b": 8.0e-5})
        generation = _generation_record(0, 5, {"m_a": 1.5, "m_b": 0.5})  # m_b wins
        manifest = {"config": config, "members": members, "generations": [], "anchor": None}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in members:
                self._write_member_checkpoint(root, name, 5, f"{name}-state".encode(), members[name]["lr"])

            ranking, plan = anchor_copy_lr_recenter_plan(config, generation, members, manifest)
            generation["exploit"] = plan
            manifest_path = root / "manifest.json"
            transitions.apply_exploit(root, manifest, generation, manifest_path)

            assigned = generation["anchor_copy_lr_recenter"]["assigned_lrs"]
            for name in members:
                state_path, optimizer_path = checkpointing.checkpoint_paths(root / name, 5)
                self.assertEqual(state_path.read_bytes(), b"m_b-state")
                optimizer_state = _read_optimizer_state(optimizer_path)
                # Weights/optimizer both genuinely sourced from the winner...
                self.assertEqual(optimizer_state["marker"], "m_b")
                # ...but every member's own optimizer LR matches *its own*
                # newly assigned spread LR, not the donor's original 8.0e-5
                # (requirement: a copied optimizer must not keep the donor
                # LR while the manifest reports a different member LR).
                self.assertAlmostEqual(optimizer_state["param_groups"][0]["lr"], assigned[name])

            paths = anchor_module.anchor_paths(root)
            self.assertEqual(Path(paths["state_path"]).read_bytes(), b"m_b-state")
            anchor_optimizer_state = _read_optimizer_state(Path(paths["optimizer_path"]))
            self.assertEqual(anchor_optimizer_state["marker"], "m_b")
            # The anchor bundle itself is never LR-patched -- it's a direct
            # snapshot of the winner's own real optimizer, LR included.
            self.assertAlmostEqual(anchor_optimizer_state["param_groups"][0]["lr"], 8.0e-5)
            self.assertEqual(manifest["anchor"]["member"], "m_b")
            self.assertEqual(manifest["anchor"]["metric_value"], 0.5)
            self.assertTrue(all(event["applied"] for event in generation["exploit"]))

    def test_rewind_restores_saved_anchor_to_every_stream_without_touching_anchor_files(self):
        config = self._config()
        members = _members({"m_a": 1.0e-5, "m_b": 8.0e-5})

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # Pre-existing anchor already on disk (as if accepted at an earlier generation).
            paths = anchor_module.anchor_paths(root)
            Path(paths["state_path"]).write_bytes(b"anchor-state")
            _write_optimizer_state(Path(paths["optimizer_path"]), lr=3.0e-5, marker="anchor")
            manifest = {
                "config": config,
                "members": members,
                "generations": [],
                "anchor": _anchor("m_prev", metric_value=0.5, lr_center=3.0e-5),
            }
            for name in members:
                self._write_member_checkpoint(root, name, 6, f"{name}-diverged-state".encode(), members[name]["lr"])

            generation = _generation_record(1, 6, {"m_a": 2.0, "m_b": 1.8})  # both much worse than anchor
            ranking, plan = anchor_copy_lr_recenter_plan(config, generation, members, manifest)
            self.assertEqual(generation["anchor_copy_lr_recenter"]["decision"], "rewound_to_previous_anchor")
            generation["exploit"] = plan
            manifest_path = root / "manifest.json"
            transitions.apply_exploit(root, manifest, generation, manifest_path)

            assigned = generation["anchor_copy_lr_recenter"]["assigned_lrs"]
            for name in members:
                state_path, optimizer_path = checkpointing.checkpoint_paths(root / name, 6)
                self.assertEqual(state_path.read_bytes(), b"anchor-state")
                optimizer_state = _read_optimizer_state(optimizer_path)
                self.assertEqual(optimizer_state["marker"], "anchor")
                self.assertAlmostEqual(optimizer_state["param_groups"][0]["lr"], assigned[name])

            # Anchor bundle and record are byte-for-byte/value-for-value unchanged.
            self.assertEqual(Path(paths["state_path"]).read_bytes(), b"anchor-state")
            anchor_optimizer_state = _read_optimizer_state(Path(paths["optimizer_path"]))
            self.assertEqual(anchor_optimizer_state["marker"], "anchor")
            self.assertAlmostEqual(anchor_optimizer_state["param_groups"][0]["lr"], 3.0e-5)
            self.assertEqual(manifest["anchor"]["member"], "m_prev")
            self.assertEqual(manifest["anchor"]["metric_value"], 0.5)
            self.assertEqual(manifest["anchor"]["lr_center"], 3.0e-5)

    def test_plateau_escape_writes_the_anchor_bundle_like_a_genuine_accept(self):
        config = _config(accept_tolerance=0.01, plateau_escape_after_generations=8)
        members = _members({"m_a": 1.0e-5, "m_b": 8.0e-5})

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = anchor_module.anchor_paths(root)
            Path(paths["state_path"]).write_bytes(b"anchor-state")
            _write_optimizer_state(Path(paths["optimizer_path"]), lr=3.0e-5, marker="anchor")
            manifest = {
                "config": config,
                "members": members,
                "generations": [],
                "anchor": _anchor("m_prev", metric_value=0.5, lr_center=3.0e-5, generation=0),
            }
            for name in members:
                self._write_member_checkpoint(root, name, 8, f"{name}-diverged-state".encode(), members[name]["lr"])

            # Both still worse than anchor's 0.5, but 8 generations have
            # passed with no genuine accept -- at the plateau threshold.
            generation = _generation_record(8, 8, {"m_a": 2.0, "m_b": 1.8})
            ranking, plan = anchor_copy_lr_recenter_plan(config, generation, members, manifest)
            self.assertEqual(generation["anchor_copy_lr_recenter"]["decision"], "plateau_escape_accepted")
            generation["exploit"] = plan
            manifest_path = root / "manifest.json"
            transitions.apply_exploit(root, manifest, generation, manifest_path)

            # Same real write as accepted_new_anchor: the anchor bundle now
            # holds the winner's own state, not the old anchor's.
            self.assertEqual(Path(paths["state_path"]).read_bytes(), b"m_b-diverged-state")
            anchor_optimizer_state = _read_optimizer_state(Path(paths["optimizer_path"]))
            self.assertEqual(anchor_optimizer_state["marker"], "m_b")
            self.assertEqual(manifest["anchor"]["member"], "m_b")
            self.assertEqual(manifest["anchor"]["metric_value"], 1.8)
            self.assertEqual(manifest["anchor"]["generation"], 8)


class AnchorCopyLrRecenterResumeTest(unittest.TestCase):
    def test_resume_reproduces_the_same_anchor_lr_center_and_next_plan(self):
        config = _config(accept_tolerance=0.01)
        members = _members({"m_a": 1.0e-5, "m_b": 8.0e-5})
        generation = _generation_record(1, 6, {"m_a": 1.5, "m_b": 0.5})

        manifest_before = {
            "schema_version": 1,
            "experiment": "unit_anchor_resume",
            "fingerprint": "abc",
            "status": "running",
            "next_generation": 1,
            "config": config,
            "members": members,
            "generations": [],
            "best": None,
            "anchor": _anchor("m_prev", metric_value=1.0, lr_center=3.0e-5),
        }

        # Simulate exactly what runner.py's resume path does: round-trip the
        # whole manifest through the typed model.
        resumed = PBTManifest.parse_payload(manifest_before).to_runtime_dict()

        ranking_direct, plan_direct = anchor_copy_lr_recenter_plan(
            config, dict(generation), dict(members), manifest_before
        )
        ranking_resumed, plan_resumed = anchor_copy_lr_recenter_plan(
            config, dict(generation), dict(resumed["members"]), resumed
        )

        self.assertEqual(ranking_direct, ranking_resumed)
        self.assertEqual(
            [{"recipient": e["recipient"], "donor": e["donor"], "new_lr": e["new_lr"], "decision": e["decision"]} for e in plan_direct],
            [{"recipient": e["recipient"], "donor": e["donor"], "new_lr": e["new_lr"], "decision": e["decision"]} for e in plan_resumed],
        )
        self.assertEqual(resumed["anchor"], manifest_before["anchor"])


class ExistingStrategiesUnchangedTest(unittest.TestCase):
    def test_disabled_by_default_and_dispatch_unchanged_for_other_strategies(self):
        config = pbt_smoke_config()
        self.assertIsNone(anchor_copy_lr_recenter_config(config))

        config["pbt"]["anchor_copy_lr_recenter"] = {"mode": "disabled"}
        self.assertIsNone(anchor_copy_lr_recenter_config(config))

    def test_all_previously_existing_strategies_still_dispatch(self):
        for name in ("exploit_mutate", "anchored_lr_sweep", "fixed_lr_grid", "population_lr_policy"):
            self.assertIn(name, dispatch.STRATEGY_PLANNERS)
        self.assertIn("anchor_copy_lr_recenter", dispatch.STRATEGY_PLANNERS)

    def test_population_rollbacks_unchanged_for_every_other_strategy(self):
        for name in ("exploit_mutate", "anchored_lr_sweep", "population_lr_policy"):
            config = pbt_smoke_config()
            config["pbt"]["strategy"] = name
            self.assertTrue(strategy_uses_population_rollbacks(config))
        config = pbt_smoke_config()
        config["pbt"]["strategy"] = "fixed_lr_grid"
        self.assertFalse(strategy_uses_population_rollbacks(config))
        config["pbt"]["strategy"] = "anchor_copy_lr_recenter"
        self.assertFalse(strategy_uses_population_rollbacks(config))


class ExperimentConfigResolutionTest(unittest.TestCase):
    def test_anchor_copy_lr_recenter_50k_config_resolves_correctly(self):
        config = config_module.load_config(
            namespace(
                config=PROJECT_DIR / "configs/experiments/anchor_copy_lr_recenter_50k.yaml",
                experiment_name="unit_anchor_copy_lr_recenter_50k",
                gpus="0,1,2,3",
                slots=None,
                smoke=False,
            )
        )

        shared = config["shared"]
        self.assertEqual(shared["validation_suffix"], "val50k_tail")
        self.assertEqual(shared["samples_per_epoch_val"], 150000)
        proxy = shared["proxy_validation"]
        self.assertEqual(proxy["control_rows_per_class"], 50000)
        self.assertNotIn("full_dataset", proxy)
        self.assertNotIn("full_suffix", proxy)
        self.assertEqual(proxy["full_holdout_suffix"], "val_holdout")

        pbt = config["pbt"]
        self.assertEqual(pbt["strategy"], "anchor_copy_lr_recenter")
        self.assertEqual(pbt["burn_in_generations"], 0)
        self.assertEqual(pbt["exploit_interval_generations"], 1)
        self.assertEqual(pbt["rollback_fraction"], 0.0)
        self.assertEqual(pbt["dynamic_controller"]["mode"], "disabled")
        tiered = pbt["tiered_validation"]
        self.assertNotIn("monitor_interval_generations", tiered)
        self.assertEqual(tiered["full_interval_generations"], 16)


class FinalGenerationOverrideTest(unittest.TestCase):
    """should_apply_exploit_for_strategy: the final-generation exception is
    scoped to anchor_copy_lr_recenter only -- every other strategy must see
    should_apply_exploit's original, unmodified behavior (skip on the final
    generation)."""

    def test_anchor_copy_lr_recenter_applies_on_final_generation(self):
        config = _config()
        config["pbt"]["strategy"] = "anchor_copy_lr_recenter"
        self.assertTrue(
            should_apply_exploit_for_strategy(config, 2, is_final_generation=True, early_stop_triggered=False)
        )

    def test_other_strategies_still_skip_the_final_generation(self):
        for name in ("exploit_mutate", "anchored_lr_sweep", "fixed_lr_grid", "population_lr_policy"):
            config = _config()
            config["pbt"]["strategy"] = name
            self.assertFalse(
                should_apply_exploit_for_strategy(config, 2, is_final_generation=True, early_stop_triggered=False),
                msg=f"strategy={name} should still skip the final generation",
            )

    def test_burn_in_still_blocks_anchor_copy_lr_recenter_on_the_final_generation(self):
        config = _config()
        config["pbt"]["strategy"] = "anchor_copy_lr_recenter"
        config["pbt"]["burn_in_generations"] = 5
        self.assertFalse(
            should_apply_exploit_for_strategy(config, 0, is_final_generation=True, early_stop_triggered=False)
        )

    def test_early_stop_still_blocks_anchor_copy_lr_recenter_on_the_final_generation(self):
        config = _config()
        config["pbt"]["strategy"] = "anchor_copy_lr_recenter"
        self.assertFalse(
            should_apply_exploit_for_strategy(config, 2, is_final_generation=True, early_stop_triggered=True)
        )


class AnchorCopyLrRecenterFinalGenerationApplyTest(unittest.TestCase):
    """Requirement: the complete plan (the "__anchor__" event AND every
    member's event) must physically apply on the run's final generation,
    unlike every other strategy. Exercises apply_exploit directly against a
    plan built exactly as runner.py would build it once
    should_apply_exploit_for_strategy has told it will_exploit=True (the
    live smoke test separately confirms runner.py's own generation loop
    actually reaches this point on a real final generation)."""

    def _config(self):
        config = _config(accept_tolerance=0.01)
        config["pbt"]["strategy"] = "anchor_copy_lr_recenter"
        return config

    def _write_member_checkpoint(self, root, name, epoch, state_bytes, optimizer_lr):
        (root / name).mkdir(exist_ok=True)
        state_path, optimizer_path = checkpointing.checkpoint_paths(root / name, epoch)
        state_path.write_bytes(state_bytes)
        _write_optimizer_state(optimizer_path, lr=optimizer_lr, marker=name)
        return state_path, optimizer_path

    def _apply_and_assert_complete(self, config, manifest, generation, root):
        self.assertTrue(
            should_apply_exploit_for_strategy(config, generation["index"], is_final_generation=True, early_stop_triggered=False),
            msg="should_apply_exploit_for_strategy must say yes on the final generation for this strategy",
        )
        manifest_path = root / "manifest.json"
        transitions.apply_exploit(root, manifest, generation, manifest_path)

        anchor_events = [e for e in generation["exploit"] if e["recipient"] == "__anchor__"]
        member_events = [e for e in generation["exploit"] if e["recipient"] != "__anchor__"]
        self.assertEqual(len(anchor_events), 1)
        self.assertEqual(len(member_events), len(manifest["members"]))
        self.assertTrue(all(event["applied"] for event in generation["exploit"]))

    def test_accepted_decision_on_final_generation_applies_completely(self):
        config = self._config()
        members = _members({"m_a": 1.0e-5, "m_b": 8.0e-5})
        generation = _generation_record(2, 7, {"m_a": 1.5, "m_b": 0.5})  # m_b beats the prior anchor
        manifest = {"config": config, "members": members, "generations": [], "anchor": _anchor("m_prev", 1.0, 3.0e-5)}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in members:
                self._write_member_checkpoint(root, name, 7, f"{name}-state".encode(), members[name]["lr"])
            paths = anchor_module.anchor_paths(root)
            Path(paths["state_path"]).write_bytes(b"prev-anchor-state")
            _write_optimizer_state(Path(paths["optimizer_path"]), lr=3.0e-5, marker="prev-anchor")

            ranking, plan = anchor_copy_lr_recenter_plan(config, generation, members, manifest)
            self.assertEqual(generation["anchor_copy_lr_recenter"]["decision"], "accepted_new_anchor")
            generation["exploit"] = plan

            self._apply_and_assert_complete(config, manifest, generation, root)

            assigned = generation["anchor_copy_lr_recenter"]["assigned_lrs"]
            for name in members:
                state_path, optimizer_path = checkpointing.checkpoint_paths(root / name, 7)
                self.assertEqual(state_path.read_bytes(), b"m_b-state")
                optimizer_state = _read_optimizer_state(optimizer_path)
                self.assertEqual(optimizer_state["marker"], "m_b")
                self.assertAlmostEqual(optimizer_state["param_groups"][0]["lr"], assigned[name])
            self.assertEqual(manifest["anchor"]["member"], "m_b")
            self.assertAlmostEqual(manifest["anchor"]["metric_value"], 0.5)
            self.assertAlmostEqual(manifest["anchor"]["lr_center"], generation["anchor_copy_lr_recenter"]["new_lr_center"])

    def test_reused_decision_on_final_generation_applies_completely(self):
        config = self._config()
        config["pbt"]["anchor_copy_lr_recenter"]["accept_tolerance"] = 0.05
        members = _members({"m_a": 1.0e-5, "m_b": 8.0e-5})
        generation = _generation_record(2, 7, {"m_a": 1.02, "m_b": 0.99})  # within tolerance of anchor's 1.0
        manifest = {"config": config, "members": members, "generations": [], "anchor": _anchor("m_prev", 1.0, 2.0e-5)}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in members:
                self._write_member_checkpoint(root, name, 7, f"{name}-state".encode(), members[name]["lr"])
            paths = anchor_module.anchor_paths(root)
            Path(paths["state_path"]).write_bytes(b"prev-anchor-state")
            _write_optimizer_state(Path(paths["optimizer_path"]), lr=2.0e-5, marker="prev-anchor")

            ranking, plan = anchor_copy_lr_recenter_plan(config, generation, members, manifest)
            self.assertEqual(generation["anchor_copy_lr_recenter"]["decision"], "reused_previous_anchor")
            generation["exploit"] = plan

            self._apply_and_assert_complete(config, manifest, generation, root)

            # Anchor bundle unchanged (still the *previous* anchor's content)...
            self.assertEqual(Path(paths["state_path"]).read_bytes(), b"prev-anchor-state")
            anchor_optimizer_state = _read_optimizer_state(Path(paths["optimizer_path"]))
            self.assertEqual(anchor_optimizer_state["marker"], "prev-anchor")
            self.assertAlmostEqual(anchor_optimizer_state["param_groups"][0]["lr"], 2.0e-5)
            self.assertEqual(manifest["anchor"]["member"], "m_prev")
            self.assertAlmostEqual(manifest["anchor"]["metric_value"], 1.0)
            # ...but every member was still physically copied from it (with
            # its own LR patched in), and lr_center still moved.
            assigned = generation["anchor_copy_lr_recenter"]["assigned_lrs"]
            for name in members:
                state_path, optimizer_path = checkpointing.checkpoint_paths(root / name, 7)
                self.assertEqual(state_path.read_bytes(), b"prev-anchor-state")
                optimizer_state = _read_optimizer_state(optimizer_path)
                self.assertEqual(optimizer_state["marker"], "prev-anchor")
                self.assertAlmostEqual(optimizer_state["param_groups"][0]["lr"], assigned[name])
            self.assertAlmostEqual(manifest["anchor"]["lr_center"], generation["anchor_copy_lr_recenter"]["new_lr_center"])
            self.assertNotAlmostEqual(manifest["anchor"]["lr_center"], 2.0e-5)

    def test_rewound_decision_on_final_generation_applies_completely(self):
        config = self._config()
        members = _members({"m_a": 1.0e-5, "m_b": 8.0e-5})
        generation = _generation_record(2, 7, {"m_a": 2.0, "m_b": 1.8})  # both much worse than the anchor
        manifest = {"config": config, "members": members, "generations": [], "anchor": _anchor("m_prev", 1.0, 3.0e-5)}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in members:
                self._write_member_checkpoint(root, name, 7, f"{name}-diverged-state".encode(), members[name]["lr"])
            paths = anchor_module.anchor_paths(root)
            Path(paths["state_path"]).write_bytes(b"prev-anchor-state")
            _write_optimizer_state(Path(paths["optimizer_path"]), lr=3.0e-5, marker="prev-anchor")

            ranking, plan = anchor_copy_lr_recenter_plan(config, generation, members, manifest)
            self.assertEqual(generation["anchor_copy_lr_recenter"]["decision"], "rewound_to_previous_anchor")
            generation["exploit"] = plan

            self._apply_and_assert_complete(config, manifest, generation, root)

            assigned = generation["anchor_copy_lr_recenter"]["assigned_lrs"]
            for name in members:
                state_path, optimizer_path = checkpointing.checkpoint_paths(root / name, 7)
                self.assertEqual(state_path.read_bytes(), b"prev-anchor-state")
                optimizer_state = _read_optimizer_state(optimizer_path)
                self.assertEqual(optimizer_state["marker"], "prev-anchor")
                self.assertAlmostEqual(optimizer_state["param_groups"][0]["lr"], assigned[name])
            self.assertEqual(manifest["anchor"]["member"], "m_prev")
            self.assertAlmostEqual(manifest["anchor"]["lr_center"], 3.0e-5)  # restored, not moved
            self.assertAlmostEqual(generation["anchor_copy_lr_recenter"]["new_lr_center"], 3.0e-5)


class WeaverCheckpointFormatTest(unittest.TestCase):
    """Verify -- against a real on-disk checkpoint and against Weaver's own
    save/load source -- exactly what the copied bundle contains. Do not
    claim complete state restoration (scaler/scheduler/step) unless the
    serialized content proves it: one test per separately-stored state
    component, plus explicit checks for the components that do NOT exist
    anywhere in this codebase's checkpoint format."""

    STATE_PATH = PROJECT_DIR / "checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/net_epoch-17_state.pt"
    OPTIMIZER_PATH = PROJECT_DIR / "checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/net_epoch-17_optimizer.pt"
    WEAVER_TRAIN_PY = PROJECT_DIR / "weaver-core/weaver/train.py"

    def test_model_state_component_is_present_and_is_a_plain_state_dict(self):
        import torch

        state = torch.load(self.STATE_PATH, map_location="cpu", weights_only=False)
        self.assertGreater(len(state), 0)
        self.assertTrue(all(hasattr(value, "shape") for value in state.values()))

    def test_optimizer_state_component_is_present_with_the_standard_pytorch_shape(self):
        import torch

        optimizer_state = torch.load(self.OPTIMIZER_PATH, map_location="cpu", weights_only=False)
        self.assertIn("state", optimizer_state)
        self.assertIn("param_groups", optimizer_state)
        self.assertIn("lr", optimizer_state["param_groups"][0])

    def test_optimizer_state_component_contains_no_scaler_or_scheduler_key(self):
        import torch

        optimizer_state = torch.load(self.OPTIMIZER_PATH, map_location="cpu", weights_only=False)
        self.assertEqual(set(optimizer_state.keys()), {"state", "param_groups"})

    def test_no_separate_scaler_scheduler_or_step_checkpoint_file_exists(self):
        checkpoint_dir = self.STATE_PATH.parent
        for pattern in ("*_scaler.pt", "*_scheduler.pt", "*_step.pt", "*_global_step.pt"):
            matches = list(checkpoint_dir.glob(pattern))
            self.assertEqual(matches, [], f"unexpected {pattern} file(s): {matches}")

    def test_weaver_save_code_only_persists_state_optimizer_and_optional_controller(self):
        """Regression guard: if a future Weaver change starts persisting
        scaler/scheduler state, this is the test that should catch it,
        since the anchor/exploit copy code would otherwise silently
        continue to not copy it."""
        source = self.WEAVER_TRAIN_PY.read_text()
        self.assertIn('torch.save(unwrap_model(model).state_dict(), f"{ckpt_base_name}_state.pt")', source)
        self.assertIn('torch.save(opt.state_dict(), f"{ckpt_base_name}_optimizer.pt")', source)
        self.assertIn('torch.save(training_controller.state_dict(), f"{ckpt_base_name}_controller.pt")', source)
        self.assertNotIn("grad_scaler.state_dict()", source)
        self.assertNotIn("scaler.state_dict()", source)
        self.assertNotIn("scheduler.state_dict()", source)

    def test_weaver_load_code_only_restores_model_and_optimizer_state(self):
        source = self.WEAVER_TRAIN_PY.read_text()
        start = source.index("def load_checkpoint(")
        end = source.index("\ndef ", start + 1)
        load_checkpoint_source = source[start:end]
        self.assertIn("model.load_state_dict", load_checkpoint_source)
        self.assertIn("opt.load_state_dict", load_checkpoint_source)
        self.assertNotIn("scaler", load_checkpoint_source.lower())
        self.assertNotIn("scheduler.load_state_dict", load_checkpoint_source)


class ControllerStateCopyTest(unittest.TestCase):
    """The third (optional) state component: controller.pt, copied
    alongside state+optimizer as one atomic bundle when
    shared.training_controller is configured (this strategy's real preset
    leaves it unset, so this exercises the code path generically, the same
    way update_global_best's own controller handling is generic)."""

    def test_controller_state_is_copied_atomically_alongside_state_and_optimizer(self):
        config = _config(accept_tolerance=0.01)
        config["shared"]["training_controller"] = "configs/controllers/linucb_lr_pp_active.yaml"
        members = _members({"m_a": 1.0e-5, "m_b": 8.0e-5})
        generation = _generation_record(0, 5, {"m_a": 1.5, "m_b": 0.5})
        manifest = {"config": config, "members": members, "generations": [], "anchor": None}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in members:
                (root / name).mkdir()
                state_path, optimizer_path = checkpointing.checkpoint_paths(root / name, 5)
                state_path.write_bytes(f"{name}-state".encode())
                _write_optimizer_state(optimizer_path, lr=members[name]["lr"], marker=name)
                controller_path = checkpointing.controller_checkpoint_path(root / name, 5)
                controller_path.write_bytes(f"{name}-controller".encode())

            ranking, plan = anchor_copy_lr_recenter_plan(config, generation, members, manifest)
            generation["exploit"] = plan
            manifest_path = root / "manifest.json"
            transitions.apply_exploit(root, manifest, generation, manifest_path)

            for name in members:
                controller_path = checkpointing.controller_checkpoint_path(root / name, 5)
                self.assertEqual(controller_path.read_bytes(), b"m_b-controller")
            paths = anchor_module.anchor_paths(root)
            self.assertEqual(Path(paths["controller_path"]).read_bytes(), b"m_b-controller")
            self.assertIsNotNone(manifest["anchor"]["sha256_controller"])


class SpreadCollapseTest(unittest.TestCase):
    def test_no_collapse_when_every_assigned_lr_is_distinct(self):
        collapsed, groups = detect_spread_collapse({"a": 1e-5, "b": 2e-5, "c": 3e-5})
        self.assertFalse(collapsed)
        self.assertEqual(groups, [])

    def test_collapse_detected_for_one_duplicate_pair(self):
        collapsed, groups = detect_spread_collapse({"a": 1e-5, "b": 1e-5, "c": 3e-5})
        self.assertTrue(collapsed)
        self.assertEqual(groups, [["a", "b"]])

    def test_collapse_detected_for_multiple_duplicate_groups(self):
        collapsed, groups = detect_spread_collapse({"a": 1e-5, "b": 1e-5, "c": 3e-5, "d": 3e-5, "e": 5e-5})
        self.assertTrue(collapsed)
        self.assertEqual(groups, [["a", "b"], ["c", "d"]])

    def test_planner_records_collapse_when_clamping_collapses_the_grid(self):
        # min_lr/max_lr are narrow enough that the extreme multipliers all
        # clamp onto the same two bounds.
        config = _config(spread_multipliers=[0.1, 0.2, 5.0, 6.0])
        config["pbt"].update(min_lr=1.0e-5, max_lr=1.0e-4)
        members = _members({"m_a": 5.0e-5, "m_b": 6.0e-5, "m_c": 7.0e-5, "m_d": 8.0e-5})
        generation = _generation_record(0, 5, {"m_a": 1.0, "m_b": 1.1, "m_c": 1.2, "m_d": 1.3})  # m_a wins -> center = 5.0e-5
        manifest = {"members": members, "generations": [], "anchor": None}

        ranking, plan = anchor_copy_lr_recenter_plan(config, generation, members, manifest)

        info = generation["anchor_copy_lr_recenter"]
        self.assertTrue(info["spread_collapsed"])
        self.assertEqual(info["duplicate_lr_groups"], [["m_a", "m_b"], ["m_c", "m_d"]])
        for event in plan:
            self.assertTrue(event["spread_collapsed"])

    def test_planner_records_no_collapse_for_a_healthy_spread(self):
        config = _config(spread_multipliers=[0.80, 0.95, 1.05, 1.20])
        members = _members({"m_a": 5.0e-5, "m_b": 6.0e-5, "m_c": 7.0e-5, "m_d": 8.0e-5})
        generation = _generation_record(0, 5, {"m_a": 1.0, "m_b": 1.1, "m_c": 1.2, "m_d": 1.3})
        manifest = {"members": members, "generations": [], "anchor": None}

        ranking, plan = anchor_copy_lr_recenter_plan(config, generation, members, manifest)

        info = generation["anchor_copy_lr_recenter"]
        self.assertFalse(info["spread_collapsed"])
        self.assertEqual(info["duplicate_lr_groups"], [])
        for event in plan:
            self.assertFalse(event["spread_collapsed"])


class SpreadMultiplierValidatorTest(unittest.TestCase):
    """anchor_copy_lr_recenter.spread_multipliers must include exactly one
    1.0 (so exactly one member continues at the exact new_lr_center), plus
    the pre-existing positive/below-and-above-1.0 checks."""

    def _build(self, multipliers):
        from training.pbt.models.config import AnchorCopyLrRecenterConfig

        return AnchorCopyLrRecenterConfig(mode="active", spread_multipliers=multipliers)

    def test_accepts_a_valid_spread_with_exactly_one_1_0(self):
        self._build([0.80, 0.90, 1.00, 1.20])

    def test_rejects_a_spread_with_no_1_0(self):
        with self.assertRaises(ValueError):
            self._build([0.80, 0.95, 1.05, 1.20])

    def test_rejects_a_spread_with_1_0_repeated(self):
        with self.assertRaises(ValueError):
            self._build([1.00, 1.00, 0.80, 1.20])

    def test_rejects_a_spread_with_no_value_below_1_0(self):
        with self.assertRaises(ValueError):
            self._build([1.00, 1.05, 1.20])

    def test_rejects_a_spread_with_no_value_above_1_0(self):
        with self.assertRaises(ValueError):
            self._build([0.80, 0.90, 1.00])

    def test_rejects_a_non_positive_multiplier(self):
        with self.assertRaises(ValueError):
            self._build([-0.1, 1.00, 1.20])


class MomentumAndPlateauEscapeFieldValidationTest(unittest.TestCase):
    """Bounds on the three new AnchorCopyLrRecenterConfig fields -- and
    that their disabling defaults (0.0 / 0 / 1.0) require no changes to
    any config that predates them."""

    def _build(self, **overrides):
        from training.pbt.models.config import AnchorCopyLrRecenterConfig

        fields = {"mode": "active", "spread_multipliers": [0.5, 0.75, 1.0, 1.25, 1.5]}
        fields.update(overrides)
        return AnchorCopyLrRecenterConfig(**fields)

    def test_defaults_are_disabling_values(self):
        policy = self._build()
        self.assertEqual(policy.recenter_momentum_fraction, 0.0)
        self.assertEqual(policy.plateau_escape_after_generations, 0)
        self.assertEqual(policy.plateau_escape_widen_factor, 1.0)

    def test_accepts_valid_values(self):
        self._build(recenter_momentum_fraction=0.075, plateau_escape_after_generations=8, plateau_escape_widen_factor=1.5)

    def test_rejects_momentum_fraction_of_1_0_or_more(self):
        with self.assertRaises(ValueError):
            self._build(recenter_momentum_fraction=1.0)

    def test_rejects_negative_momentum_fraction(self):
        with self.assertRaises(ValueError):
            self._build(recenter_momentum_fraction=-0.01)

    def test_rejects_negative_plateau_escape_after_generations(self):
        with self.assertRaises(ValueError):
            self._build(plateau_escape_after_generations=-1)

    def test_rejects_widen_factor_below_1_0(self):
        with self.assertRaises(ValueError):
            self._build(plateau_escape_widen_factor=0.5)


class OptimizerLrPatchTest(unittest.TestCase):
    """Requirement: a copied optimizer must not keep the donor LR internally
    while the manifest reports a different member LR -- verifies the patch
    helper directly (state/optimizer_state.py::set_optimizer_state_lr /
    atomic_set_optimizer_lr), independent of the apply_exploit integration
    already covered by AnchorCopyLrRecenterApplyTest /
    AnchorCopyLrRecenterFinalGenerationApplyTest."""

    def test_set_optimizer_state_lr_rewrites_every_param_group(self):
        from training.pbt.state.optimizer_state import set_optimizer_state_lr

        state = {"state": {}, "param_groups": [{"lr": 8.0e-5}, {"lr": 8.0e-5}]}
        set_optimizer_state_lr(state, 1.12e-5)
        self.assertAlmostEqual(state["param_groups"][0]["lr"], 1.12e-5)
        self.assertAlmostEqual(state["param_groups"][1]["lr"], 1.12e-5)

    def test_set_optimizer_state_lr_handles_tensor_valued_lr(self):
        from training.pbt.state.optimizer_state import set_optimizer_state_lr

        state = {"state": {}, "param_groups": [{"lr": torch.tensor(8.0e-5)}]}
        set_optimizer_state_lr(state, 1.12e-5)
        self.assertAlmostEqual(float(state["param_groups"][0]["lr"]), 1.12e-5, places=9)

    def test_atomic_set_optimizer_lr_round_trips_through_disk(self):
        from training.pbt.state.optimizer_state import atomic_set_optimizer_lr

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "optimizer.pt"
            _write_optimizer_state(path, lr=8.0e-5, marker="donor")

            atomic_set_optimizer_lr(path, 1.12e-5)

            state = _read_optimizer_state(path)
            self.assertEqual(state["marker"], "donor")  # only lr changes, nothing else
            self.assertAlmostEqual(state["param_groups"][0]["lr"], 1.12e-5)


if __name__ == "__main__":
    unittest.main()
