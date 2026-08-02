#!/usr/bin/env python3
"""Typed contracts for physics-aware dynamic training controller I/O."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


ControllerActionName = Literal[
    "keep",
    "lr_mul_0_95",
    "lr_mul_0_9",
    "lr_mul_1_05",
    "lr_mul_1_1",
    "flag_review",
]

ControllerStateLabel = Literal["improving", "flat", "degraded", "noisy", "unsafe"]
ControllerSafetyCheck = Literal["passed", "blocked", "clamped", "cooldown"]
DEFAULT_CONTROLLER_ACTIONS = (
    "keep",
    "lr_mul_0_95",
    "lr_mul_0_9",
    "lr_mul_1_05",
    "lr_mul_1_1",
    "flag_review",
)


class StrictControllerModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ControllerObservation(StrictControllerModel):
    schema_version: Literal[1] = 1
    generation: int = Field(ge=0)
    member: str
    epoch: int | None = Field(default=None, ge=0)
    epoch_fraction: float | None = Field(default=None, ge=0.0)
    step: int | None = Field(default=None, ge=0)
    lr: float = Field(gt=0.0)
    epoch_start_lr: float | None = Field(default=None, gt=0.0)
    cumulative_lr_factor: float | None = Field(default=None, gt=0.0)
    metric_name: str
    metric_value: float
    previous_metric_value: float | None = None
    metric_delta: float | None = None
    metric_ema: float | None = None
    previous_metric_ema: float | None = None
    metric_ema_delta: float | None = None
    metric_trend: float | None = None
    metric_noise: float | None = Field(default=None, ge=0.0)
    metric_uncertainty: float | None = Field(default=None, ge=0.0)
    previous_metric_uncertainty: float | None = Field(default=None, ge=0.0)
    metric_delta_sigma: float | None = None
    trend_window: int | None = Field(default=None, ge=1)
    baseline_metric_value: float | None = None
    baseline_delta: float | None = None
    global_best_metric_value: float | None = None
    global_best_delta: float | None = None
    train_loss: float | None = Field(default=None, ge=0.0)
    train_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    train_loss_ema: float | None = Field(default=None, ge=0.0)
    train_loss_ema_delta: float | None = None
    grad_norm: float | None = Field(default=None, ge=0.0)
    amp_skipped_optimizer_steps: int | None = Field(default=None, ge=0)
    max_cuda_memory_mb: float | None = Field(default=None, ge=0.0)
    optimizer_step: float | None = Field(default=None, ge=0.0)
    optimizer_param_groups: int | None = Field(default=None, ge=0)
    optimizer_lr_mean: float | None = Field(default=None, gt=0.0)
    optimizer_lr_min: float | None = Field(default=None, gt=0.0)
    optimizer_lr_max: float | None = Field(default=None, gt=0.0)
    optimizer_weight_decay_mean: float | None = Field(default=None, ge=0.0)
    momentum_norm: float | None = Field(default=None, ge=0.0)
    second_moment_norm: float | None = Field(default=None, ge=0.0)
    adaptive_direction_norm: float | None = Field(default=None, ge=0.0)
    adaptive_direction_norm_max: float | None = Field(default=None, ge=0.0)
    action_ready: bool = False
    cooldown_remaining: float | None = Field(default=None, ge=0.0)
    allowed_actions: list[ControllerActionName] = Field(min_length=1)


class ControllerAction(StrictControllerModel):
    schema_version: Literal[1] = 1
    generation: int = Field(ge=0)
    member: str
    state_label: ControllerStateLabel
    confidence: float = Field(ge=0.0, le=1.0)
    action: ControllerActionName
    reason: str
    safety_check: ControllerSafetyCheck
    applied: bool = False
    lr_before: float | None = Field(default=None, gt=0.0)
    proposed_lr: float | None = Field(default=None, gt=0.0)
    bounded_lr: float | None = Field(default=None, gt=0.0)
    action_ready: bool = False
    cooldown_remaining: float | None = Field(default=None, ge=0.0)


def parse_controller_observation(payload: Any):
    try:
        return ControllerObservation.model_validate(payload)
    except ValidationError as error:
        raise ValueError(str(error)) from error


def dump_controller_observation(model_or_payload):
    if isinstance(model_or_payload, BaseModel):
        return model_or_payload.model_dump(mode="json", exclude_none=True)
    return parse_controller_observation(model_or_payload).model_dump(
        mode="json",
        exclude_none=True,
    )


def parse_controller_action(payload: Any):
    try:
        return ControllerAction.model_validate(payload)
    except ValidationError as error:
        raise ValueError(str(error)) from error


def dump_controller_action(model_or_payload):
    if isinstance(model_or_payload, BaseModel):
        return model_or_payload.model_dump(mode="json", exclude_none=True)
    return parse_controller_action(model_or_payload).model_dump(
        mode="json",
        exclude_none=True,
    )
