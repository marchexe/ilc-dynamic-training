#!/usr/bin/env python3
"""PBT ranking, LR planning, and rollback policy.

Split by concern: `ranking` (population ordering + cadence gates, no
strategy-specific mutation), one module per `pbt.strategy` value
(`exploit_mutate`, `anchored_lr_sweep`, `fixed_lr_grid` -- mirrors the 3
strategy presets under configs/presets/pbt/), `dispatch` (strategy-name ->
planner), and `rollbacks` (population/baseline-guard rollback injection,
applied after the strategy planner runs).
"""

from training.pbt.planning.anchored_lr_sweep import (
    adaptive_lr_radius_state,
    anchored_lr_sweep_plan,
    factors_from_radius,
    lr_factors_for_population,
    previous_lr_controller_record,
    previous_lr_radius_record,
    smooth_lr_controller_state,
)
from training.pbt.planning.dispatch import STRATEGY_PLANNERS, plan_for_strategy
from training.pbt.planning.exploit_mutate import exploit_mutate_plan, exploit_significance, ranking_and_plan
from training.pbt.planning.fixed_lr_grid import fixed_lr_grid_plan
from training.pbt.planning.ranking import (
    confidence_aware_ranking,
    in_burn_in,
    metric_uncertainty,
    raw_metric_ranking,
    should_apply_exploit,
)
from training.pbt.planning.rollbacks import (
    add_baseline_guard_rollbacks,
    add_global_best_rollbacks,
    strategy_uses_population_rollbacks,
)

__all__ = [
    "STRATEGY_PLANNERS",
    "adaptive_lr_radius_state",
    "add_baseline_guard_rollbacks",
    "add_global_best_rollbacks",
    "anchored_lr_sweep_plan",
    "confidence_aware_ranking",
    "exploit_mutate_plan",
    "exploit_significance",
    "factors_from_radius",
    "fixed_lr_grid_plan",
    "in_burn_in",
    "lr_factors_for_population",
    "metric_uncertainty",
    "plan_for_strategy",
    "previous_lr_controller_record",
    "previous_lr_radius_record",
    "ranking_and_plan",
    "raw_metric_ranking",
    "should_apply_exploit",
    "smooth_lr_controller_state",
    "strategy_uses_population_rollbacks",
]
