#!/usr/bin/env python3
"""Typed PBT YAML and resolved runtime configuration schemas."""

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from training.pbt.models.controller import ControllerActionName, DEFAULT_CONTROLLER_ACTIONS
from training.pbt.state.optimizer_state import normalize_optimizer_state_mode


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
    "validation_ctag_reference_mistag_geomean_percent",
    "validation_btag_reference_mistag_geomean_percent",
    "validation_total_reference_mistag_geomean_percent",
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


class SmoothLrControllerConfig(StrictSectionModel):
    """Smooth the anchored LR sweep center so fine-tuning avoids abrupt LR jumps."""

    mode: Literal["smooth"] = "smooth"
    smoothing: float = Field(default=0.25, gt=0.0, le=1.0)
    max_center_increase: float = Field(default=1.25, ge=1.0)
    max_center_decrease: float = Field(default=0.80, gt=0.0, le=1.0)
    max_member_increase: float = Field(default=1.35, ge=1.0)
    max_member_decrease: float = Field(default=0.75, gt=0.0, le=1.0)
    decay_bias: float = Field(default=1.0, gt=0.0, le=1.0)


class DynamicControllerConfig(StrictSectionModel):
    """Dynamic-control policy that can override planned LR actions after each generation."""

    mode: Literal["disabled", "active"] = "active"
    evaluate_initial_checkpoint: bool = False
    allowed_actions: list[ControllerActionName] = Field(
        default_factory=lambda: list(DEFAULT_CONTROLLER_ACTIONS),
        min_length=1,
    )
    metric_delta_tolerance: float = Field(default=0.0, ge=0.0)
    ema_beta: float = Field(default=0.7, ge=0.0, lt=1.0)
    trend_window: int = Field(default=3, ge=2)
    noisy_metric_threshold: float | None = Field(default=None, gt=0.0)
    min_delta_sigma_for_action: float | None = Field(default=1.0, gt=0.0)
    eval_interval_fraction: float = Field(default=0.20, gt=0.0)
    action_interval_fraction: float = Field(default=0.20, gt=0.0)
    generation_epoch_fraction: float | None = Field(default=None, gt=0.0)
    max_cumulative_lr_factor_per_epoch: float = Field(default=2.0, ge=1.0)

    @field_validator("allowed_actions")
    @classmethod
    def validate_unique_actions(cls, values):
        if len(set(values)) != len(values):
            raise ValueError("dynamic_controller.allowed_actions must be unique")
        return values


class PopulationLrPolicyConfig(StrictSectionModel):
    """Bidirectional, population-comparative LR policy (pbt.strategy ==
    "population_lr_policy"): infer an up/down LR direction each time a fresh
    proxy-tier round is available by comparing the best member above the
    population's median LR against the best member below it, unconditionally
    copy weights+optimizer from the winning half's best member to everyone
    (no significance gate), and roll back to the pre-decision checkpoint if
    the next round's best result is worse for the whole population than it
    was before the change. Mutually exclusive with dynamic_controller/
    exploit_mutate at runtime -- selecting this strategy leaves those
    modules' code paths completely uninvoked, so the legacy
    exploit_mutate + dynamic_controller behavior is unchanged when this
    section is absent or mode: disabled.
    """

    mode: Literal["disabled", "active"] = "disabled"
    # Which proxy_validation tier's already-scheduled tiered_evaluations
    # round to decide on. "monitor" is the existing, much-larger-sample
    # (e.g. 50k rows/class vs control's 5k) tier that this policy was
    # designed around; using it costs nothing new -- it's already computed
    # on tiered_validation.monitor_interval_generations cadence.
    eval_tier: Literal["monitor", "full"] = "monitor"
    up_factor: float = Field(default=1.1, gt=1.0)
    down_factor: float = Field(default=0.9, gt=0.0, lt=1.0)
    # Minimum combined-uncertainty sigma the winning half must clear before
    # a direction is chosen; None disables the check (pure nominal
    # comparison). Distinct from exploit_significance_sigma, which this
    # strategy never reads -- the copy itself is never gated once a
    # direction is chosen, only the direction choice is.
    direction_sigma: float | None = Field(default=1.0, ge=0.0)


class AnchorCopyLrRecenterConfig(StrictSectionModel):
    """Isolated strategy (pbt.strategy == "anchor_copy_lr_recenter"):
    every generation, unconditionally (no significance gate) select the
    best-finite-metric member as winner, classify it against the single
    persisted population anchor within accept_tolerance (strictly better ->
    accepted_new_anchor, persist winner's full state as the new anchor;
    within tolerance -> reused_previous_anchor, keep the old anchor's
    weights/optimizer but still move the LR center toward the winner's LR;
    strictly worse -> rewound_to_previous_anchor, keep the old anchor AND
    restore the old LR center, undoing this generation's LR movement), copy
    the (possibly-just-updated) anchor to every stream including the
    winner, and assign a fresh deterministic LR spread around the
    resulting center. Mutually exclusive with dynamic_controller/
    exploit_mutate/population_lr_policy at runtime, the same way
    population_lr_policy is -- selecting this strategy leaves those
    modules' code paths completely uninvoked.
    """

    mode: Literal["disabled", "active"] = "disabled"
    # Always the per-generation control-tier metric (control_proxy_50k in
    # this project's convention) -- deliberately not tier-selectable like
    # population_lr_policy.eval_tier: this strategy is defined around
    # consuming the metric every worker already produces every generation,
    # never a periodic separate tiered_evaluations round.
    #
    # No damping field here: new_lr_center is always exactly the winner's
    # own LR (never a blend toward it) -- direction (up/down/flat) emerges
    # naturally from where that LR sits relative to the old center, per the
    # strategy's canonical spec. A previous center_step_fraction field that
    # damped this movement has been removed rather than left unused.
    # Relative/fractional, exactly like degradation_tolerance -- reuses
    # metrics.py::metric_is_worse_than_reference's existing orientation-safe
    # comparison (current worse-than-reference by more than this fraction),
    # applied symmetrically in both directions to get the three-way
    # accept/reuse/rewind split, rather than a new absolute-units comparator.
    accept_tolerance: float = Field(
        default=0.0, ge=0.0, lt=1.0,
        description="Fractional tolerance band for the accept/reuse/rewind classification (same convention as degradation_tolerance).",
    )
    spread_multipliers: list[float] = Field(
        min_length=1,
        description="Deterministic per-member LR multipliers applied to the new center, e.g. [0.80, 0.90, 1.00, 1.20] for 4 members. Length must equal the population size; must include exactly one 1.0 so exactly one member continues at the exact winning LR.",
    )

    @field_validator("spread_multipliers")
    @classmethod
    def validate_spread_multipliers(cls, values):
        if any(value <= 0 for value in values):
            raise ValueError("anchor_copy_lr_recenter.spread_multipliers must be positive")
        if not (min(values) < 1.0 < max(values)):
            raise ValueError(
                "anchor_copy_lr_recenter.spread_multipliers must include values both below and above 1.0 "
                "so the spread has at least one member below and one above the center"
            )
        if values.count(1.0) != 1:
            raise ValueError(
                "anchor_copy_lr_recenter.spread_multipliers must include exactly one 1.0 entry, "
                "so exactly one member continues at the exact new_lr_center -- "
                f"got {values.count(1.0)}"
            )
        return values


class ProxyValidationConfig(StrictSectionModel):
    """Proxy-validation datasets used for physics-aware high-frequency control."""

    manifest: str
    active_subset: Literal["control", "monitor", "full", "full_holdout"] = "control"
    control_dataset: str
    monitor_dataset: str | None = None
    full_dataset: str | None = None
    # full_holdout = full validation with the control+monitor tail windows
    # excluded -- zero overlap with either, used for control<->full
    # correlation/ranking-agreement diagnostics (unlike plain "full", which
    # contains the exact control/monitor events and so isn't an independent
    # check). Never used to drive decisions, same as monitor/full.
    full_holdout_dataset: str | None = None
    train_suffix: str | None = None
    control_suffix: str | None = None
    monitor_suffix: str | None = None
    full_suffix: str | None = None
    full_holdout_suffix: str | None = None
    control_rows_per_class: int | None = Field(default=None, gt=0)
    monitor_rows_per_class: int | None = Field(default=None, gt=0)
    full_rows_per_class: int | None = Field(default=None, gt=0)
    full_holdout_rows_per_class: int | None = Field(default=None, gt=0)
    strategy: str | None = None


class TieredValidationConfig(StrictSectionModel):
    """Schedule for automatic, read-only monitor/full proxy-tier evaluation.

    Control-tier evaluation is unconditional (every generation, via the
    normal per-worker Weaver eval) and is the only tier allowed to drive
    ranking/exploit/controller decisions. Monitor and full are diagnostic
    only: evaluated on their own cadence, for every population member (so
    ranking-agreement/correlation analysis has paired observations), and
    never read by planning.py or controller.py.
    """

    monitor_interval_generations: int | None = Field(default=None, gt=0)
    full_interval_generations: int | None = Field(default=None, gt=0)
    evaluate_initial_checkpoint_all_tiers: bool = False


class SharedSection(WeaverSharedSection):
    dataset: str
    validation_dataset: str | None = None
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
    # Every experiment in this repo resumes from a checkpoint (pretrained or
    # a prior PBT run's global_best) -- never a random init -- so BatchNorm
    # running stats accumulated over the source training are always worth
    # preserving by default. model.train() would otherwise let BN momentum
    # (0.1) overwrite them within ~20 minibatches (confirmed regression:
    # see bn_freeze_diag_baseline/frozen.yaml, 1.14% -> 7-9% mistag).
    # Configs that genuinely want BN to adapt can set `freeze_batch_norm:
    # false` explicitly (e.g. bn_freeze_diag_baseline.yaml).
    freeze_batch_norm: bool = True
    proxy_validation: ProxyValidationConfig | None = None
    data_extension: str | None = None
    train_suffix: str | None = None
    validation_suffix: str | None = None
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
    strategy: Literal[
        "exploit_mutate", "anchored_lr_sweep", "fixed_lr_grid", "population_lr_policy",
        "anchor_copy_lr_recenter",
    ] | None = None
    confidence_aware_selection: bool = True
    selection_uncertainty_sigma: float | None = Field(default=1.0, gt=0.0)
    exploit_significance_sigma: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "If set, a donor->recipient exploit copy is only executed when the donor "
            "beats the recipient by at least this many combined-uncertainty sigma "
            "(exploit_mutate only). Missing uncertainty is treated as inconclusive "
            "(skip, don't copy), not as a nominal-comparison fallback."
        ),
    )
    burn_in_generations: int = Field(default=0, ge=0)
    # Real cadence gate: PBT ranking is still computed every generation (for
    # health/global-best bookkeeping), but the exploit plan is only applied
    # every exploit_interval_generations generations. Unset (None) means
    # every generation, same as before this field existed. NOTE: this used
    # to also be readable as `exploit_interval` in configured_intervals()'s
    # *reporting* output, but that value was never actually consulted by the
    # runtime loop -- this field is what makes the interval real.
    exploit_interval_generations: int | None = Field(default=None, gt=0)
    exploit_replacement_policy: Literal["fraction", "elitist"] = Field(
        default="fraction",
        description=(
            "How ranking_and_plan() picks donors/recipients each exploit "
            "cycle (exploit_mutate only). 'fraction' (default, unchanged "
            "legacy behavior): classic truncation PBT -- the top "
            "exploit_fraction of the ranking donates to the bottom "
            "exploit_fraction; everyone else is left untouched. 'elitist': "
            "the single best-ranked member donates to every other member "
            "(each still independently LR-mutated and independently gated "
            "by exploit_significance_sigma when set); exploit_fraction is "
            "required by the schema but ignored under this policy."
        ),
    )
    anchored_weight_source: Literal["anchor", "self"] = "anchor"
    base_start_lr: float | None = None
    lr_factors: list[float] | None = None
    lr_radius: LrRadiusConfig | None = None
    lr_controller: SmoothLrControllerConfig | None = None
    dynamic_controller: DynamicControllerConfig | None = None
    population_lr_policy: PopulationLrPolicyConfig | None = None
    anchor_copy_lr_recenter: AnchorCopyLrRecenterConfig | None = None
    tiered_validation: TieredValidationConfig | None = None
    baseline_metric_value: float | None = None
    baseline_guard_tolerance: float | None = Field(default=None, ge=0.0, lt=1.0)
    baseline_guard_action: Literal["observe", "rollback_to_initial"] | None = None
    baseline_guard_lr_factor: float | None = Field(default=None, gt=0.0, le=1.0)
    baseline_guard_reject_global_best: bool | None = None
    baseline_guard_seed_initial_best: bool | None = None

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
    strategy: Literal[
        "exploit_mutate", "anchored_lr_sweep", "fixed_lr_grid", "population_lr_policy",
        "anchor_copy_lr_recenter",
    ] = "exploit_mutate"

    @model_validator(mode="after")
    def validate_resolved_strategy_shape(self):
        if self.lr_controller is not None and self.strategy != "anchored_lr_sweep":
            raise ValueError("lr_controller is only supported with anchored_lr_sweep")
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
        if self.pbt.baseline_guard_seed_initial_best:
            if not self.shared.initial_state or not self.shared.initial_optimizer:
                raise ValueError("baseline_guard_seed_initial_best requires initial_state/initial_optimizer")
            if self.pbt.baseline_metric_value is None:
                raise ValueError("baseline_guard_seed_initial_best requires baseline_metric_value")
        policy = self.pbt.anchor_copy_lr_recenter
        if policy is not None and policy.mode == "active":
            if len(policy.spread_multipliers) != len(self.population):
                raise ValueError(
                    "anchor_copy_lr_recenter.spread_multipliers must have exactly one entry "
                    f"per population member ({len(self.population)}), got {len(policy.spread_multipliers)}"
                )
        return self

    def to_runtime_dict(self):
        return self.model_dump(mode="json", exclude_none=True)
