#!/usr/bin/env python3
"""Typed exploit events for PBT checkpoint transitions."""

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError


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
        return (
            experiment_dir / "global_best_state.pt",
            experiment_dir / "global_best_optimizer.pt",
            experiment_dir / "global_best_controller.pt",
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
    | InitialResumeRollbackEvent,
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
