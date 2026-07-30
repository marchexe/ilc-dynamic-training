#!/usr/bin/env python3
"""Typed PBT YAML and resolved runtime configuration schemas."""

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from training.pbt.optimizer_state import normalize_optimizer_state_mode


MEMBER_NAME_RE = re.compile(r"[A-Za-z0-9_.-]+")

PBT_METRICS = {
    "validation_accuracy",
    "validation_auc",
    "validation_loss",
    "validation_bkg_rejection_bc_score",
    "validation_bkg_rejection_bd_score",
    "validation_bkg_rejection_cb_score",
    "validation_bkg_rejection_cd_score",
    "validation_b_tag_rejection_score",
    "validation_c_tag_rejection_score",
    "validation_bkg_rejection_score",
    "validation_working_point_mistag_percent",
    "validation_ctag_reference_mistag_percent",
}


class StrictSectionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WeaverSharedSection(BaseModel):
    """Shared training options allow Weaver pass-through fields."""

    model_config = ConfigDict(extra="allow")


class ExperimentSection(StrictSectionModel):
    name: str | None = None
    output_root: str


class ResourcesSection(StrictSectionModel):
    gpus: list[int | str] | None = None


class PopulationMember(StrictSectionModel):
    name: str
    start_lr: float | None = None


class LrRadiusConfig(StrictSectionModel):
    initial: float = Field(ge=0.0)
    minimum: float = Field(ge=0.0)
    shrink_factor: float = Field(gt=0.0, le=1.0)
    shrink_after_inner_wins: int = Field(ge=1)
    keep_if_edge_wins: bool = True

    @model_validator(mode="after")
    def validate_radius_bounds(self):
        if self.minimum > self.initial:
            raise ValueError("anchored_lr_sweep lr_radius.minimum must be <= initial")
        return self


class SharedSection(WeaverSharedSection):
    dataset: str
    checkpoint: str
    data_config: str
    network_config: str
    seed: int
    generations: int = Field(gt=0)
    epochs_per_generation: int = Field(gt=0)
    samples_per_epoch: int = Field(gt=0)
    samples_per_epoch_val: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    optimizer: str
    lr_scheduler: str
    num_workers: int = Field(gt=0)
    fetch_step: float
    use_amp: bool
    amp_dtype: str
    no_remake_weights: bool
    data_extension: str | None = None
    training_controller: str | None = None
    initial_epoch: int | None = None
    initial_state: str | None = None
    initial_optimizer: str | None = None
    initial_controller: str | None = None
    initial_optimizer_mode: Literal["raw", "copy", "damped", "reset"] | None = None
    initial_optimizer_damping: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("initial_optimizer_mode", mode="before")
    @classmethod
    def validate_initial_optimizer_mode(cls, value):
        if value is None:
            return value
        return normalize_optimizer_state_mode(value)

    @model_validator(mode="after")
    def validate_initial_resume(self):
        initial_values = (
            self.initial_epoch,
            self.initial_state,
            self.initial_optimizer,
            self.initial_controller,
        )
        if any(value is not None for value in initial_values):
            if not self.initial_state or not self.initial_optimizer:
                raise ValueError("initial_state and initial_optimizer must be configured together")
            if self.initial_epoch is None:
                raise ValueError("initial_epoch is required when initial_state/initial_optimizer are configured")
        return self


class PBTSection(StrictSectionModel):
    metric: str
    mode: Literal["max", "min"]
    exploit_fraction: float = Field(gt=0.0, le=0.5)
    mutation_factors: list[float] = Field(min_length=1)
    min_lr: float = Field(gt=0.0)
    max_lr: float = Field(gt=0.0)
    seed: int
    degradation_tolerance: float = Field(default=0.02, ge=0.0, lt=1.0)
    degradation_window: int = Field(default=3, ge=1)
    early_stop_degraded_generations: int = Field(default=0, ge=0)
    rollback_fraction: float = Field(default=0.0, ge=0.0, le=0.5)
    controller_state_on_exploit: Literal["copy", "reset"] | None = None
    backend: Literal["local_weaver", "ray_weaver", "ray_tune"] | None = None
    strategy: Literal["exploit_mutate", "anchored_lr_sweep", "fixed_lr_grid"] | None = None
    base_start_lr: float | None = None
    lr_factors: list[float] | None = None
    lr_radius: LrRadiusConfig | None = None
    baseline_metric_value: float | None = None
    baseline_guard_tolerance: float | None = Field(default=None, ge=0.0, lt=1.0)
    baseline_guard_action: Literal["observe", "rollback_to_initial"] | None = None
    baseline_guard_lr_factor: float | None = Field(default=None, gt=0.0, le=1.0)
    baseline_guard_reject_global_best: bool | None = None

    @field_validator("metric")
    @classmethod
    def validate_metric(cls, value):
        if value not in PBT_METRICS:
            raise ValueError("Unsupported PBT metric")
        return value

    @field_validator("mutation_factors")
    @classmethod
    def validate_mutation_factors(cls, values):
        if any(value <= 0 for value in values):
            raise ValueError("mutation_factors must contain positive values")
        return values

    @field_validator("lr_factors")
    @classmethod
    def validate_lr_factors(cls, values):
        if values is not None and any(value <= 0 for value in values):
            raise ValueError("anchored_lr_sweep lr_factors must be positive")
        return values

    @model_validator(mode="after")
    def validate_pbt_shape(self):
        if self.min_lr >= self.max_lr:
            raise ValueError("Expected 0 < min_lr < max_lr")
        if self.strategy == "anchored_lr_sweep":
            if self.base_start_lr is None:
                raise ValueError("anchored_lr_sweep requires base_start_lr")
            if self.lr_radius is None and self.lr_factors is None:
                raise ValueError("anchored_lr_sweep requires lr_radius or lr_factors")
        return self


class PBTYamlConfig(StrictSectionModel):
    schema_version: Literal[1]
    experiment: ExperimentSection
    shared: SharedSection
    resources: ResourcesSection
    population: list[PopulationMember]
    pbt: PBTSection

    @classmethod
    def parse_payload(cls, payload: Any):
        if not isinstance(payload, dict):
            raise ValueError("Expected a schema_version: 1 PBT configuration")
        try:
            return cls.model_validate(payload)
        except ValidationError as error:
            raise ValueError(str(error)) from error

    def runtime_sections(self):
        return {
            "experiment": self.experiment.model_dump(exclude_unset=True),
            "shared": self.shared.model_dump(exclude_unset=True),
            "resources": self.resources.model_dump(exclude_unset=True),
            "population": [
                member.model_dump(exclude_unset=True)
                for member in self.population
            ],
            "pbt": self.pbt.model_dump(exclude_unset=True),
        }


class GpuSlot(StrictSectionModel):
    host: str | None = None
    gpu: str
    label: str

    @field_validator("gpu", "label")
    @classmethod
    def validate_non_empty(cls, value):
        if not str(value).strip():
            raise ValueError("GPU slots must be non-empty")
        return str(value)


class ResolvedSharedSection(SharedSection):
    data_extension: str = "root"

    @field_validator("lr_scheduler")
    @classmethod
    def validate_no_lr_scheduler(cls, value):
        if value != "none":
            raise ValueError("PBT learning-rate mutation requires lr_scheduler: none")
        return value


class ResolvedPBTSection(PBTSection):
    controller_state_on_exploit: Literal["copy", "reset"] = "copy"
    backend: Literal["local_weaver", "ray_weaver", "ray_tune"] = "local_weaver"
    strategy: Literal["exploit_mutate", "anchored_lr_sweep", "fixed_lr_grid"] = "exploit_mutate"

    @model_validator(mode="after")
    def validate_resolved_strategy_shape(self):
        if self.strategy == "anchored_lr_sweep" and self.lr_factors is not None and len(self.lr_factors) < 2:
            raise ValueError("anchored_lr_sweep requires at least two lr_factors")
        return self


class ResolvedPopulationMember(StrictSectionModel):
    name: str
    start_lr: float = Field(gt=0.0)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value):
        if not MEMBER_NAME_RE.fullmatch(value):
            raise ValueError("Every population member requires a filesystem-safe name")
        return value


def lr_factors_from_radius(radius, count):
    radius = float(radius)
    if count < 2:
        raise ValueError("anchored_lr_sweep requires at least two members")
    if count == 2:
        offsets = (radius, -radius)
    elif count == 4:
        offsets = (radius, radius / 2, -radius / 2, -radius)
    else:
        step = 2 * radius / (count - 1)
        offsets = [radius - index * step for index in range(count)]
    return [1.0 + offset for offset in offsets]


def anchored_lr_factors(pbt, population_size):
    if pbt.get("lr_radius"):
        return lr_factors_from_radius(pbt["lr_radius"]["initial"], population_size)

    factors = list(pbt.get("lr_factors") or [])
    if len(factors) < population_size:
        raise ValueError("anchored_lr_sweep requires at least one lr_factor per member")
    if population_size == 2 and len(factors) >= 2:
        return [factors[0], factors[-1]]
    return factors[:population_size]


class ResolvedPBTConfig(StrictSectionModel):
    schema_version: Literal[1]
    config_path: str
    experiment_name: str
    output_root: str
    shared: ResolvedSharedSection
    gpus: list[str]
    slots: list[GpuSlot]
    population: list[ResolvedPopulationMember] = Field(min_length=2)
    pbt: ResolvedPBTSection
    smoke: bool

    @classmethod
    def from_sections(
        cls,
        *,
        config_path,
        experiment_name,
        output_root,
        shared,
        slots,
        population,
        pbt,
        smoke,
    ):
        pbt_payload = dict(pbt)
        population_payload = [dict(member) for member in population]
        if pbt_payload.get("strategy", "exploit_mutate") == "anchored_lr_sweep":
            factors = anchored_lr_factors(pbt_payload, len(population_payload))
            base_start_lr = float(pbt_payload["base_start_lr"])
            for member, factor in zip(population_payload, factors):
                member["start_lr"] = base_start_lr * factor

        try:
            return cls.model_validate(
                {
                    "schema_version": 1,
                    "config_path": str(config_path),
                    "experiment_name": experiment_name,
                    "output_root": output_root,
                    "shared": shared,
                    "gpus": [slot["gpu"] for slot in slots],
                    "slots": slots,
                    "population": population_payload,
                    "pbt": pbt_payload,
                    "smoke": smoke,
                }
            )
        except ValidationError as error:
            raise ValueError(str(error)) from error

    @field_validator("experiment_name")
    @classmethod
    def validate_experiment_name(cls, value):
        if not MEMBER_NAME_RE.fullmatch(value):
            raise ValueError("Experiment requires a filesystem-safe name")
        return value

    @model_validator(mode="after")
    def validate_runtime_contract(self):
        names = [member.name for member in self.population]
        if len(set(names)) != len(names):
            raise ValueError("Population member names must be unique")
        if not self.slots:
            raise ValueError("At least one GPU slot is required")
        labels = [slot.label for slot in self.slots]
        if len(set(labels)) != len(labels):
            raise ValueError("GPU slots must be unique")
        if any(not self.pbt.min_lr <= member.start_lr <= self.pbt.max_lr for member in self.population):
            raise ValueError("Population start_lr values must lie within PBT LR bounds")
        if (
            self.pbt.baseline_guard_action == "rollback_to_initial"
            and not self.shared.initial_state
        ):
            raise ValueError("baseline_guard_action: rollback_to_initial requires initial_state/initial_optimizer")
        return self

    def to_runtime_dict(self):
        return self.model_dump(mode="json", exclude_none=True)
