#!/usr/bin/env python3
"""Strategy-name -> planner dispatch."""

from training.pbt.models.events import normalize_exploit_plan
from training.pbt.planning.anchored_lr_sweep import anchored_lr_sweep_plan
from training.pbt.planning.exploit_mutate import exploit_mutate_plan
from training.pbt.planning.fixed_lr_grid import fixed_lr_grid_plan
from training.pbt.planning.population_lr_policy import population_lr_policy_plan

STRATEGY_PLANNERS = {
    "anchored_lr_sweep": anchored_lr_sweep_plan,
    "fixed_lr_grid": fixed_lr_grid_plan,
    "exploit_mutate": exploit_mutate_plan,
    "population_lr_policy": population_lr_policy_plan,
}


def plan_for_strategy(config, generation_record, members, manifest=None):
    strategy_name = config["pbt"].get("strategy", "exploit_mutate")
    try:
        planner = STRATEGY_PLANNERS[strategy_name]
    except KeyError as error:
        raise ValueError(f"Unsupported PBT strategy: {strategy_name}") from error
    ranking, plan = planner(config, generation_record, members, manifest)
    return ranking, normalize_exploit_plan(plan)
