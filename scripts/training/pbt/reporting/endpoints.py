"""Scientific endpoints computed only from recorded validation trajectories."""
import math
from statistics import mean


def population_best(values, mode="min"):
    values = [float(value) for value in values]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("Current-best endpoint requires finite member metrics")
    if mode not in {"min", "max"}:
        raise ValueError(f"Unknown metric mode: {mode}")
    return (min if mode == "min" else max)(values)


def current_best_trajectory(member_values, mode="min"):
    """Choose a member separately at every generation, before averaging."""
    lengths = {len(values) for values in member_values.values()}
    if len(lengths) != 1 or not lengths or not next(iter(lengths)):
        raise ValueError("Current-best endpoint requires aligned member trajectories")
    return [population_best(values, mode) for values in zip(*member_values.values())]


def final_window_mean(trajectory, window=10):
    if window <= 0 or len(trajectory) < window:
        raise ValueError(f"Current-best endpoint requires {window} generations")
    selected = trajectory[-window:]
    if any(not math.isfinite(value) for value in selected):
        raise ValueError("Current-best endpoint requires finite metrics")
    return mean(selected)
