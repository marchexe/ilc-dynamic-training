#!/usr/bin/env python3
"""The fixed_lr_grid strategy: rank the population, never mutate LR."""

from training.pbt.planning.exploit_mutate import ranking_and_plan
from training.pbt.planning.ranking import confidence_aware_ranking


def fixed_lr_grid_plan(config, generation_record, members, manifest=None):
    if len(members) == 1:
        generation_record["skipped_exploits"] = []
        return confidence_aware_ranking(config, generation_record, members, manifest), []
    ranking, _ = ranking_and_plan(config, generation_record, members)
    return ranking, []
