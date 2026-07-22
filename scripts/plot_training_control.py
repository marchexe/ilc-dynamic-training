#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


parser = argparse.ArgumentParser()
parser.add_argument("--active-dir", type=Path, default=ROOT / "runs/sgv_3cat/active")
parser.add_argument("--baseline-dir", type=Path, default=ROOT / "runs/sgv_3cat/baseline")
parser.add_argument("--output", type=Path, default=ROOT / "runs/sgv_3cat/training_control_comparison.png")
args = parser.parse_args()
ACTIVE_DIR = args.active_dir
BASELINE_DIR = args.baseline_dir
OUTPUT = args.output


def read_metrics(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    loss_acc = re.search(r"Eval AvgLoss: ([0-9.]+), AvgAcc: ([0-9.]+)", text)
    auc = re.search(r"roc_auc_score:\s*\n([0-9.]+)", text)
    if loss_acc is None or auc is None:
        raise RuntimeError(f"Missing validation metrics in {path}")
    seed = re.search(r"Using random seed (\d+)", text)
    train_worker_seed = re.search(r"DataIter train_worker0, seed=(\d+)", text)
    val_worker_seed = re.search(r"DataIter val_worker0, seed=(\d+)", text)
    return {
        "loss": float(loss_acc.group(1)),
        "accuracy": float(loss_acc.group(2)),
        "auc": float(auc.group(1)),
        "seed": None if seed is None else int(seed.group(1)),
        "train_worker_seed": None if train_worker_seed is None else int(train_worker_seed.group(1)),
        "val_worker_seed": None if val_worker_seed is None else int(val_worker_seed.group(1)),
    }


events = [
    json.loads(line)
    for line in (ACTIVE_DIR / "net_controller.jsonl").read_text().splitlines()
    if line.strip()
]
baseline = read_metrics(BASELINE_DIR / "train.log")
active = read_metrics(ACTIVE_DIR / "train.log")

steps = [0] + [event["global_step"] for event in events]
active_lr = [events[0]["old_lr"]] + [event["new_lr"] for event in events]
last_step = events[-1]["steps_per_epoch"]

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
    }
)
fig, (ax_trace, ax_metric) = plt.subplots(
    1, 2, figsize=(10.2, 4.2), gridspec_kw={"width_ratios": [1.65, 1]}
)

decision_steps = [event["global_step"] for event in events]
loss_ema = [event["loss_ema"] for event in events]
ax_trace.plot(
    decision_steps,
    loss_ema,
    color="black",
    marker="o",
    markersize=4,
    linewidth=1.2,
    label=r"training loss EMA ($\beta=0.98$)",
)
ax_trace.set_xlim(0, last_step + 30)
loss_span = max(loss_ema) - min(loss_ema)
ax_trace.set_ylim(min(loss_ema) - 0.06 * loss_span, max(loss_ema) + 0.16 * loss_span)
ax_trace.set_xlabel("optimizer step")
ax_trace.set_ylabel("loss EMA")
ax_trace.grid(axis="both", color="0.88", linewidth=0.6)

ax_lr = ax_trace.twinx()
ax_lr.spines["right"].set_visible(True)
ax_lr.step(steps, active_lr, where="post", linewidth=1.5, color="#2166ac", label="LinUCB LR")
ax_lr.hlines(1e-3, 0, last_step, color="0.45", linewidth=1.2, linestyle="--", label="fixed LR")
lr_span = max(active_lr) - min(active_lr)
ax_lr.set_ylim(min(active_lr) - 0.06 * lr_span, max(active_lr) + 0.10 * lr_span)
ax_lr.set_ylabel("learning rate", color="#2166ac")
ax_lr.tick_params(axis="y", labelcolor="#2166ac")
ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

annotation_stride = max(1, (len(events) + 3) // 4)
for event_idx, event in enumerate(events):
    step = event["global_step"]
    factor = event["action_factor"]
    reward = event["reward"]
    reward_text = "r_prev=n/a" if reward is None else f"r_prev={reward:+.3f}"
    ax_trace.axvline(step, color="0.82", linewidth=0.7, linestyle=":", zorder=0)
    if event_idx % annotation_stride == 0 or event_idx == len(events) - 1:
        ax_trace.annotate(
            f"a=×{factor:g}\n{reward_text}",
            (step, event["loss_ema"]),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="0.2",
        )

lines_1, labels_1 = ax_trace.get_legend_handles_labels()
lines_2, labels_2 = ax_lr.get_legend_handles_labels()
ax_trace.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", frameon=False)
ax_trace.set_title("(a) Controller trace", loc="left")

metric_names = ["Validation accuracy", "Validation AUC"]
baseline_values = [baseline["accuracy"], baseline["auc"]]
active_values = [active["accuracy"], active["auc"]]
y_positions = [1, 0]

for y, fixed, controlled in zip(y_positions, baseline_values, active_values):
    ax_metric.plot([fixed, controlled], [y, y], color="0.72", linewidth=1.0, zorder=1)
    ax_metric.scatter(fixed, y, facecolor="white", edgecolor="0.25", s=42, label="fixed LR" if y == 1 else None, zorder=3)
    ax_metric.scatter(controlled, y, color="#2166ac", s=42, label="LinUCB" if y == 1 else None, zorder=3)
    ax_metric.annotate(f"{fixed:.5f}", (fixed, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=7)
    ax_metric.annotate(f"{controlled:.5f}", (controlled, y), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=7, color="#2166ac")

ax_metric.set_yticks(y_positions, metric_names)
ax_metric.set_ylim(-0.45, 1.45)
ax_metric.set_xlim(0.81, 0.95)
ax_metric.set_xlabel("metric value")
ax_metric.set_title("(b) End-of-run validation", loc="left")
ax_metric.grid(axis="x", color="0.88", linewidth=0.6)
ax_metric.legend(loc="lower right", frameon=False)
matched_random_streams = (
    baseline["seed"], baseline["train_worker_seed"], baseline["val_worker_seed"]
) == (
    active["seed"], active["train_worker_seed"], active["val_worker_seed"]
)
if matched_random_streams and baseline["seed"] is not None:
    comparison_note = (
        f"Ntrain=153,600; Nval=14,848; matched seed={baseline['seed']}\n"
        f"matched train/val worker seeds={baseline['train_worker_seed']}/{baseline['val_worker_seed']}"
    )
else:
    comparison_note = "Ntrain=153,600; Nval=14,848\nsingle run; random streams not verified as matched"
ax_metric.text(
    0.0,
    -0.18,
    comparison_note,
    transform=ax_metric.transAxes,
    ha="left",
    fontsize=7,
    color="0.35",
)

fig.tight_layout()
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
print(OUTPUT)
