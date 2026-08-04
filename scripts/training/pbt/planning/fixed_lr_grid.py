#!/usr/bin/env python3
"""The fixed_lr_grid strategy: rank the population, never mutate LR."""

from training.pbt.planning.exploit_mutate import ranking_and_plan


def fixed_lr_grid_plan(config, generation_record, members, manifest=None):
    ranking, _ = ranking_and_plan(config, generation_record, members)
    return ranking, []
