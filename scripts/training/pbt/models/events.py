#!/usr/bin/env python3
"""Typed exploit events for PBT checkpoint transitions."""

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from training.pbt.state.checkpointing import population_lr_policy_snapshot_paths


class ExploitEventBase(BaseModel):
    model_config = ConfigDict(extra="allow")

    recipient: str
    donor: str
    recipient_lr: float | None = None
    donor_lr: float | None = None
    new_lr: float
    applied: bool = False

    def donor_paths(self, experiment_dir, recipient_dir, epoch, manifest):
        donor_dir = Path(experiment_dir) / self.donor
        prefix = donor_dir / f"net_epoch-{epoch}"
        return (
            Path(f"{prefix}_state.pt"),
            Path(f"{prefix}_optimizer.pt"),
            donor_dir / f"net_epoch-{epoch}_controller.pt",
        )

    def parent_update(self):
        return {
            "parent": self.donor,
            "parent_source": self.source,
        }

    def error_donor_name(self):
        return self.donor


class PopulationExploitEvent(ExploitEventBase):
    source: Literal["population"] = "population"
    mutation_factor: float | None = None


class AnchoredLrSweepEvent(ExploitEventBase):
    source: Literal["anchored_lr_sweep"]
    anchor_member: str
    anchor_metric: float
    lr_factor: float
    lr_radius: float | None = None


class GlobalBestRollbackEvent(ExploitEventBase):
    source: Literal["global_best"]
    mutation_factor: float = 1.0
    reason: str = "rollback_from_global_best"
    global_best_generation: int

    def donor_paths(self, experiment_dir, recipient_dir, epoch, manifest):
        experiment_dir = Path(experiment_dir)
        best = manifest.get("best") or {}
        return (
            Path(best.get("state_path") or experiment_dir / "checkpoints" / "global_best_state.pt"),
            Path(best.get("optimizer_path") or experiment_dir / "checkpoints" / "global_best_optimizer.pt"),
            Path(best.get("controller_path") or experiment_dir / "checkpoints" / "global_best_controller.pt"),
        )


class PopulationLrPolicyEvent(ExploitEventBase):
    """One recipient's copy of a population_lr_policy directional decision:
    weights+optimizer come from `donor` (the best member in the winning
    LR-direction half), LR is the recipient's *own* previous LR scaled by
    `factor` (not the donor's), so the population keeps a spread of LRs to
    compare again at the next decision round. Every field needed to resolve
    (accept/roll back) this decision later is carried here rather than in
    separate mutable manifest state, so history can be reconstructed by
    scanning `generation_record["exploit"]` the same way previous_anchor()/
    last_action_epoch_fraction() already do for other planners.
    """

    source: Literal["population_lr_policy"] = "population_lr_policy"
    direction: Literal["up", "down"]
    factor: float
    margin_sigma: float | None = None
    decision_generation: int
    decision_epoch: int
    metric_before: float
    eval_tier: str


class PopulationLrPolicyResolutionEvent(ExploitEventBase):
    """Resolves the population_lr_policy decision made at `decision_generation`:
    "accepted" is a no-op copy (donor == recipient, same epoch) that just
    confirms the LR already in effect; "rolled_back" restores the recipient's
    own pre-decision checkpoint (`rollback_epoch`) and pre-decision LR.
    """

    source: Literal["population_lr_policy_resolution"] = "population_lr_policy_resolution"
    outcome: Literal["accepted", "rolled_back"]
    decision_generation: int
    rollback_epoch: int
    metric_before: float
    metric_after: float

    def donor_paths(self, experiment_dir, recipient_dir, epoch, manifest):
        if self.outcome != "rolled_back":
            return super().donor_paths(experiment_dir, recipient_dir, epoch, manifest)
        # The plain net_epoch-{rollback_epoch}_* path is the *donor copy's*
        # destination from the decision being undone, not the recipient's
        # own pre-decision state -- that was snapshotted separately (see
        # apply_exploit's population_lr_policy special case) precisely
        # because this same path is what the forward decision overwrote.
        snapshot_state, snapshot_optimizer = population_lr_policy_snapshot_paths(
            Path(recipient_dir), self.rollback_epoch
        )
        return (
            snapshot_state,
            snapshot_optimizer,
            Path(recipient_dir) / f"net_epoch-{self.rollback_epoch}_controller.pt",
        )

    def parent_update(self):
        if self.outcome != "rolled_back":
            return {"parent": self.donor, "parent_source": self.source}
        return {"parent": None, "parent_source": "population_lr_policy_rollback"}


class AnchorCopyEvent(ExploitEventBase):
    """One recipient's every-generation copy for the anchor_copy_lr_recenter
    strategy: weights+optimizer+controller always come from the single
    persisted population anchor (manifest["anchor"]), never from a member
    directory -- `donor` here is a human-readable label (the member that
    originally produced the current anchor state), not a path source; the
    real source is resolved by donor_paths() below regardless of that
    label's value, exactly like GlobalBestRollbackEvent resolves from
    manifest["best"] rather than from `donor`. Emitted for every member
    every generation, including the current winner -- unlike
    population_lr_policy's two-phase decision/resolution split, this
    strategy resolves accept-or-rewind within the same generation it
    evaluates (the control-tier metric this decision needs already exists
    every generation), so one event type is enough; `decision` records
    which of the four outcomes produced this copy (plateau_escape_accepted
    is a forced accept after too many consecutive rewinds -- see
    AnchorCopyLrRecenterConfig.plateau_escape_after_generations -- handled
    identically to accepted_new_anchor everywhere a real anchor-bundle
    write is needed, distinguished only for reporting).
    """

    source: Literal["anchor_copy_lr_recenter"] = "anchor_copy_lr_recenter"
    decision: Literal[
        "accepted_new_anchor", "reused_previous_anchor", "rewound_to_previous_anchor", "plateau_escape_accepted",
    ]
    anchor_generation: int
    anchor_metric_value: float
    winner: str
    winner_metric_value: float
    winner_lr: float
    lr_center: float
    # The pre-clamp value of `new_lr` (== new_lr when clamping was a
    # no-op) -- recorded per event so a collapsed spread can be explained
    # from any single event/the manifest alone, without recomputing
    # center * multiplier from the resolved config.
    unclamped_lr: float
    # True when min_lr/max_lr clamping collapsed two or more members onto
    # the exact same assigned LR this generation -- see
    # planning/anchor_copy_lr_recenter.py::detect_spread_collapse. Recorded
    # on every event of the generation (not just a summary field) so it
    # survives being read back from any single event during
    # history-reconstruction, the same way decision/lr_center do.
    spread_collapsed: bool = False

    def donor_paths(self, experiment_dir, recipient_dir, epoch, manifest):
        experiment_dir = Path(experiment_dir)
        anchor = manifest.get("anchor") or {}
        return (
            Path(anchor.get("state_path") or experiment_dir / "checkpoints" / "anchor_state.pt"),
            Path(anchor.get("optimizer_path") or experiment_dir / "checkpoints" / "anchor_optimizer.pt"),
            Path(anchor.get("controller_path") or experiment_dir / "checkpoints" / "anchor_controller.pt"),
        )


class InitialResumeRollbackEvent(ExploitEventBase):
    source: Literal["initial_resume"]
    mutation_factor: float
    reason: str = "rollback_from_initial_resume_baseline_guard"
    initial_epoch: int
    metric: str
    metric_value: float
    baseline_metric: float
    baseline_guard_tolerance: float

    def donor_paths(self, experiment_dir, recipient_dir, epoch, manifest):
        initial_epoch = int(self.initial_epoch)
        prefix = Path(recipient_dir) / f"net_epoch-{initial_epoch}"
        return (
            Path(f"{prefix}_state.pt"),
            Path(f"{prefix}_optimizer.pt"),
            Path(recipient_dir) / f"net_epoch-{initial_epoch}_controller.pt",
        )

    def parent_update(self):
        return {
            "parent": None,
            "parent_source": "initial_resume",
        }


ExploitEvent = Annotated[
    PopulationExploitEvent
    | AnchoredLrSweepEvent
    | GlobalBestRollbackEvent
    | InitialResumeRollbackEvent
    | PopulationLrPolicyEvent
    | PopulationLrPolicyResolutionEvent
    | AnchorCopyEvent,
    Field(discriminator="source"),
]
ExploitEventAdapter = TypeAdapter(ExploitEvent)


def parse_exploit_event(payload: Any):
    event_payload = dict(payload)
    event_payload.setdefault("source", "population")
    try:
        return ExploitEventAdapter.validate_python(event_payload)
    except ValidationError as error:
        raise ValueError(str(error)) from error


def dump_exploit_event(event):
    if isinstance(event, BaseModel):
        return event.model_dump(mode="json", exclude_none=True)
    return parse_exploit_event(event).model_dump(mode="json", exclude_none=True)


def normalize_exploit_plan(plan):
    return [dump_exploit_event(event) for event in plan]
