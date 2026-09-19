# Prepared overnight continuation — not launched

Base commit: `9fc26e0a0d59ed63e1ece1829c787584b5ecdc1e`.
The working tree additionally allows one-member `fixed_lr_grid` runs and returns
an empty donor plan for them; other strategies still require two members. The
training loop, checkpoint format, and all existing multi-member paths are unchanged.
These three source/test files are not committed. Preparation stays ignored here.

## Continuation contract

Two independent production-runner instances, one member each. This avoids sharing
initial model/optimizer/scaler states and isolates arm failures. Each YAML inherits
the validated 50-epoch config, substitutes its own `net_epoch-67` state/optimizer,
sets initial_epoch=67, and advances the seed by 50 to 12395. LR stays unchanged.
Generation 0: load epoch 67, train checkpoint epoch 68 (full epoch 51), seed 12395.
Generation 49: load epoch 116, train checkpoint epoch 117 (full epoch 100), seed 12444.
Raw optimizer bootstrap copies its scaler companion; no damping/reset is applied.
All 50 new model/optimizer/scaler sets per arm and metrics remain on disk.

`preflight.json` records exact model/optimizer/scaler SHA256, scaler values,
optimizer step counters, fingerprints, 433 checks, and disk sizes. `commands.json`
contains the 100 resolved training commands. All 350 old 20/50-epoch commands and
fingerprints still match the prior verified snapshot. Full suite: 418 passed,
1 skipped. Prepared resume/evaluation/report tools passed CPU/mock checks only.
No GPU training or checkpoint inference was run during preparation.

No scheduler, controller, rollback, recenter, anchor copy, plateau escape, or
adaptive early stop is enabled. The retained best aliases are reporting artifacts;
they never become continuation sources.

## Allocation and estimates

Use iutgpu01 physical GPU 1 for 14e-6, GPU 3 for 8.5e-6. GPUs 0 and 2 reevaluate
original checkpoints concurrently. GPU 4 was occupied by an unrelated process;
do not use it. GPU availability is checked again by the launcher/evaluator.
Unused GPUs remain idle. Training devices are UUID-pinned and mapped to local GPU 0
inside each independent process.

Measured full epoch wall time: ~226–234 s, including ~40–48 s validation
(mean 43.85 s). Do not add validation a second time. 50 additional epochs:
3h08m–3h15m before initial/final standalone checks and reports; allow 3.2–3.5 h
wall time for both concurrent arms (~6.4–7 GPU-hours).

100 new checkpoint triples: 1,073,327,700 bytes, about 1.00 GiB. Reserve 3 GiB
for initial copies, best aliases, logs, audits, metrics, reports, and reevaluation
prediction files. Checked free space: about 15.58 TiB.

Original 0–19 models all load on CPU with the canonical 144-tensor schema.
All SHA256 values are recorded in `original_checkpoint_hashes.json`.
Current standalone evaluations measured 44–61 s/checkpoint; allow 10–15 minutes
on two GPUs. `originals.py` uses the existing deterministic evaluation command
builder, fixed 150k input sequence, FP16 and current eight working-point metrics.
It writes prediction parquet files outside the source checkpoint directory.
It never trains or writes model/optimizer/scaler files. Results/ranks are pending
execution; the epoch-17 result must match the old study's initial evaluation.

## Proxy feasibility

The completed 50-epoch study has no event-level .parquet/.root/.npy/.npz predictions.
Its hashes, aggregate curves and working-point counts cannot recover arbitrary
subsets. Four pilot parity prediction files cover a single pilot checkpoint;
old pretrained ROOT predictions cover a different historical sample, not the 250
LR-study checkpoints. Neither permits the requested trajectory calibration.

Future proposed nested subsets: 6k/12k/24k/30k total, exactly 2k/4k/8k/10k per class,
using one fixed seeded permutation per class. Freeze IDs before comparing results.
Compare each checkpoint's composite and eight WPs against full150k; report per-epoch
LR rankings/winners, pairwise reversal rates, raw and rolling trend directions, and
late-window Spearman/Pearson correlations (not only correlations dominated by the
common early-training improvement). Exact ties should remain ties.

No proxy is declared reliable: at 0.05% mistag, 30k balanced events yield only about
five passing background events per class; 6k yields roughly one. Full150k remains
authoritative. No proxy dataset/training-loop changes or proxy GPU job is prepared.

Estimated separate calibration for all 250 saved checkpoints:
- Evaluate the largest 30k once and save scores: ~19–24 s/checkpoint including
  process startup, ~1.3–1.7 GPU-hours, ~40–55 minutes on two GPUs. Derive all four
  nested proxies from those same predictions. Subset-only batches may differ
  numerically from full150k batches; quantify this before interpreting close ranks.
- Four separate subset passes: roughly 4–5.5 GPU-hours due to repeated startup.
- Stronger parity reference: reevaluate full150k once per checkpoint with prediction
  output, ~3–4.2 GPU-hours (~1.5–2.1 h on two GPUs); then derive every nested proxy
  and compare against identical full-event scores without subset-batching ambiguity.
These are estimates, not timings measured on the proposed subsets.

## Commands (run on iutgpu01, in /data/suehara/part/march)

```bash
P=configs/prepared/fixed_lr_continuation_20260917

# Launch continuations (user only); production engine, never --smoke.
for arm in lr_14e-6 lr_8_5e-6; do
  .venv/bin/python scripts/launch/experiment.py start --config "$P/$arm.yaml"
done

# Launch original-checkpoint evaluation only (user only).
nohup .venv/bin/python -u "$P/originals.py" run --gpus 0,2 \
  > "$P/original-evaluations.console.log" 2>&1 < /dev/null &

# Status: manifests plus identity-checked live processes.
.venv/bin/python "$P/review.py" status
for arm in lr_14e-6 lr_8_5e-6; do
  .venv/bin/python scripts/launch/experiment.py status --config "$P/$arm.yaml"
done

# Logs / GPU watch.
tail -F logs/experiments/fixed_lr_continuation_20260917_lr_{14e-6,8_5e-6}/main.log \
  "$P/original-evaluations.console.log"
ML_PFA_CLUSTER_NODES=iutgpu01 bash scripts/cluster/watch.sh 10

# Resume stopped/interrupted arms; refuses live tokens or occupied GPUs.
# Choose one arm instead of both if the other is still running.
.venv/bin/python "$P/resume.py" both

# Morning verification and corrected original-checkpoint ranking.
.venv/bin/python "$P/review.py" verify
.venv/bin/python "$P/originals.py" summarize

# Morning analysis (requires complete verified continuations).
.venv/bin/python "$P/review.py" analyze
```

Same-config resume uses the runner's persisted next_generation cursor and skips
workers already recorded complete. An interrupted/unrecorded epoch is retried from
the preceding complete checkpoint; this is epoch-boundary resume, not batch resume.
The resume helper reuses the original ownership token and log, refuses live jobs,
checks the fingerprint and GPU UUID, and adds only the existing runner --resume flag.
No completed historical run is reused as a writable destination.

Morning verification invokes the generic verifier independently for both new runs,
then checks their data schedules against each other, initial prediction parity
against each corresponding completed epoch-50 result, and untouched source artifacts.

Morning analysis writes epochs.csv (1–100), summary.csv (1–100 and 51–100),
rolling_means.csv (5/10/20), three plots, and summary.json with all eight WPs,
slopes/variability, crossovers, first threshold attainment, best individual checkpoint,
and best observed final-20 mean. Compare convergence speed separately from eventual
performance; plateau/degradation labels are descriptive, not statistical proof.
Original summary.json/metrics.csv report every checkpoint, corrected best, epoch-17
rank, absolute gap and relative differences. A different pretrained winner changes
starting-checkpoint selection interpretation, not the validity of the matched
14e-6 versus 8.5e-6 continuation comparison. It does not prove a better eventual
trajectory without a later experiment. No PBT recommendation is inferred in advance.
