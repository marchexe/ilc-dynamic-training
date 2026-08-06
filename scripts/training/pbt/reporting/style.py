#!/usr/bin/env python3
"""Shared matplotlib style/helper system for the report-facing plots
(report_plots.py, and the checkpoint-selection bridge in plots.py) -- one
place for member color assignment, the shared rcParams block, and small
display helpers, so every report-facing figure reads as one visual system
rather than several independently-styled ones."""

from training.pbt.reporting.constants import CB_PALETTE

REPORT_PLOT_RCPARAMS = {
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def plot_setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(REPORT_PLOT_RCPARAMS)
    return plt


# Member-identity colors -- CB_PALETTE minus "black" (reserved for the
# winner marker) and "grey" (reserved for neutral/population-context) so a
# member's own color is never confusable with those two semantic roles.
# Populations beyond this many members cycle (index % len(...)) rather than
# fail or silently borrow an unrelated palette.
MEMBER_COLOR_CYCLE = tuple(color for name, color in CB_PALETTE.items() if name not in ("black", "grey"))


def member_order(manifest):
    """Stable member ordering, used for both color assignment and legend
    order: by configured start_lr (smallest to largest -- the convention
    the old dashboard plots already used for their own per-member legend
    ordering), then by name for members with no configured start_lr.
    Source of truth for "who is in the population": manifest["members"]
    keys, the same set build_summary()'s own "population" field uses."""
    start_lr_by_member = {
        member["name"]: member.get("start_lr") for member in (manifest.get("config", {}).get("population") or [])
    }
    return tuple(
        sorted(
            manifest.get("members", {}),
            key=lambda name: (start_lr_by_member.get(name) is None, start_lr_by_member.get(name), name),
        )
    )


def member_color(member_name, ordered_members):
    if member_name not in ordered_members:
        return CB_PALETTE["grey"]
    index = ordered_members.index(member_name)
    return MEMBER_COLOR_CYCLE[index % len(MEMBER_COLOR_CYCLE)]


def compact_trial(name):
    return str(name).replace("member_", "m")
