# Stage 2B: train8m preparation and bounded GPU readiness smoke

Stage 2B makes the ten-times larger `train8m` input usable by the existing
training system and proves it with one bounded GPU run. It changes no model,
optimizer, PBT policy, physics selection, working point or historical artifact.
It is infrastructure readiness, not a scientific comparison.

## One command

```bash
bash scripts/experiments/run_stage2b_smoke.sh
```

The launcher checks the repository revision (tracked tree clean, Stage 2B files
committed), the frozen source identities and disk space, discovers idle GPUs
with the Stage 2A probe, and submits one detached remote **tmux** supervisor.
That supervisor runs, in order and without further input:

1. `[DATASET]` preflight → conversion of any missing flavour (three parallel
   CPU workers, `nice 10`) → full verification (rows, schema, labels, SHA256,
   unique Weaver glob) → realized manifest → Parquet isolation re-check;
2. `[GPU]` lease of two idle GPUs → production runner with the smoke config →
   deliberate SIGINT inside generation 1 training → `--resume` in a new runner
   process → completion with final evaluations;
3. evidence verification → real `--dry-run` of both Stage 3 drafts → parity →
   `report.md`.

`--status` prints dataset-conversion progress (rows, ETA per flavour)
separately from GPU-smoke progress. `--dry-run` performs the read-only checks
and GPU discovery only. `--prepare-only` stops after the dataset phase.
Reruns are safe: a verified dataset is reused, completed and verified campaigns
return immediately, a live tmux session is reported instead of duplicated, and
each smoke attempt gets a new run directory (failed attempts are preserved).
Exit codes: 0 submitted/completed, 1 smoke/verification failure, 2 invalid
inputs or configuration, 69 no GPU/disk/remote submission, 75 locked.

## Verified dataset facts (read-only, `PROVENANCE.md`, `provenance.json`)

| Flavour | Label | train8m rows | train800k prefix | val1000k overlap | legacy val50k / val5k inside train8m |
|---|---:|---:|---|---:|---|
| bb | 5 | 7,989,060 | exact, 798,977 rows | 0 | 49,928/49,928, 4,995/4,995 |
| cc | 4 | 7,976,224 | exact, 797,569 rows | 0 | 49,861/49,861, 4,980/4,980 |
| dd | 2 | 7,956,165 | exact, 795,686 rows | 0 | 49,729/49,729, 4,970/4,970 |

* Total **23,921,449** rows = ROOT entries (jets) of
  `/data/suehara/mldata/flavortag/20250711_ilc_nnqq_sgv_10m/ml_flavtag_6cat_{bb,cc,dd}_train8m.root`,
  produced by `divide_energy_10m.C` with `Entry$<8000000` and the common cut
  `pfcand_px[0]>-250&&pfcand_pz[0]>-250&&pfcand_e[0]<250&&neu_pfcand_pz[0]<60&&neu_pfcand_e[0]<120`
  (raw window 24,000,000; 99.67% pass). The cut is already applied in the ROOT
  files; conversion copies rows 1:1 and never re-selects.
* Each file holds a single label (b=5, c=4, d=2; Weaver classes b,c,d); all
  135 branches have the same schema as train800k and val1000k; every scalar
  record is unique within train8m.
* train800k (used by every production run) is the exact ordered prefix of
  train8m. val1000k (`Entry$` 16M–17M, source of `val50k_tail`, `val5k_tail`
  and the Stage 2A holdout) shares no scalar record with train8m.
* Legacy `val5k`/`val50k` lie inside train8m and are rejected by the Stage 2B
  split guard, as are `val_holdout`, `val1000k` and `train800k`. Selection uses
  `val50k_tail`. Historical note: legacy `val5k` is also inside train800k; all
  current production configurations select on `val50k_tail`.
* Expected accepted rows per full epoch after the existing flat reweighting
  (0.8988/0.8992/0.9000): about 21.51 M (train800k: 2.15 M measured).
* No physical event ID is stored (`event_nbh` is 0). Isolation is established
  by Entry windows and scalar identity, not by event lineage.

## Conversion design (`dataset_spec.json`)

The spec pins source size, mtime, entry count and branch-schema digest; output
`datasets/20250711_ilc_nnqq_sgv_10m_3cat_train8m_parquet/` (new directory; the
existing train800k directory is never written). Per flavour one Parquet file
`ml_flavtag_6cat_<f>_train8m.parquet`, so Weaver's existing glob and file-level
loading semantics are unchanged. Entries are streamed in 100k chunks
(`awkward.to_parquet_row_groups`, lz4) into a hidden `.partial-<host>-<pid>`
file with per-chunk label/finiteness checks, fsynced, verified (row count,
row-group layout, schema digest equal to the train800k conversion, labels),
hashed and atomically renamed. Per-flavour `flock` leases prevent concurrent
writers; interrupted partials are discarded on rerun; a published file without
a valid record is re-verified or set aside as `.invalid-*` evidence. The
realized manifest (`MANIFEST.json`, `datasets/manifests/<name>.json`) records
file and source SHA256 and Weaver's audit fingerprint.

Row groups are 10,000 rows instead of 1,000: at 8 M rows, 1,000-row groups give
8,000 groups and a measured 113 MB footer that Weaver re-parses for every 1%
fetch (0.9–1.5 s per open); 10,000 keeps ~800 groups per file like train800k.
Content, order and Weaver row IDs do not depend on the layout (tested).

## Bounded GPU smoke

`configs/experiments/stage2b_train8m_smoke.yaml` inherits the fixed-LR baseline
contract unchanged (epoch-17 checkpoint, raw Ranger state, FP16 AMP, frozen BN,
deterministic, audited data, batch 1024, `val50k_tail`) and differs only in
data (train8m), horizon (3 generations), members (lr 3e-6 and 1.4e-5) and a
256-step cap (`samples_per_epoch` 262,144, more than one 1% fetch). The
windowed and cadenced schemas forbid sample caps, so the smoke uses
`fixed_lr_grid` (Stage 3 arm A); arm B is checked by a real `--dry-run` against
the prepared data, the parity checker and the existing CPU policy tests.

Verified per worker and generation: train8m command globs only; Weaver audit
fingerprint, rows and file SHA256 equal to the realized manifest (a train800k
or synthetic substitution fails); 256 batches of unique rows; applied steps =
batches − AMP skips (≤10%); finite loss, gradient norm, primary metric and all
eight working points; complete 150,000-row validation; resume from epoch E−1
with model, Ranger state, LR override and the previous epoch's AMP scaler in a
new process; finite updated weights; per-parameter RAdam step increase equal to
the applied steps; finite scaler; isolated members; Stage 1 selected-best
bundle; final evaluations bound to recorded hashes; interruption during
training with exit 130, terminated workers and no checkpoint of that epoch;
resume exit 0; no surviving token-carrying process; no foreign GPU process;
CUDA memory and run disk within budget.

Budgets (`smoke_policy.json`): two GPUs on one authorized node (≥40 GB, idle:
no process, <500 MiB, <5%), 4 h wall and 8 GPU-hours hard caps, 36,000 MiB GPU
memory cap, 10 GiB run disk, 30 min bounded GPU wait. GPU and campaign leases
are inherited by the runner. A foreign process on a leased GPU stops only this
smoke. Outputs: `runs/eval/stage2b_train8m_readiness_v1/` (`status.json`,
`campaign.log`, `supervisor.log`, `logs/`, `attempts/attempt-NNN/`,
`pbt/<experiment>/`, `report.md`, `readiness.json`, `report_integrity.json`).

## Training compatibility on train8m

Unchanged by size: model/optimizer/scaler checkpoint size, epoch accounting
(`initial_epoch + generation + 1`), one Weaver process per generation, VRAM
(inputs padded to 75 particles; 21.5 GB max allocated measured on train800k),
validation, PBT decision code. Code-level limitations that do change:

* no intra-epoch checkpoint: an interruption costs up to one train8m epoch per
  member (~30–45 min instead of ~4 min);
* `--data-audit` hashes all training files at every worker start: 64 GB per
  member and generation (page cache on 1.5–2 TB nodes; cold cost measured by
  the smoke as startup time);
* `fetch_step 0.01` is a fraction: one fetch is ~240k rows instead of ~24k,
  i.e. a ten-times larger shuffle window and roughly 8–12 GB host RAM per
  worker (estimate; the smoke records peak RSS; GPU nodes have ≥1.5 TB);
* restart artefacts of the frozen contract (Lookahead slow weights/counter
  reset, five untrimmed SequenceTrimmer steps, fresh DataLoader) now occur
  every ~21k steps instead of ~2.1k: the definition is unchanged, the density
  per sample is ten times lower, so train8m and train800k epochs are not
  interchangeable units.

## Resources

Measured: ROOT read 10–16k rows/s and Parquet write ~33k rows/s per process
(~1.1 GB RSS for a 50k chunk); 2,675 bytes/row (lz4) → ~64 GB output;
provenance verification 1 min 56 s, 2.8 GB RSS; train800k production worker:
12,400 accepted rows/s, 173 s training, 36 s validation, 21,540 MB CUDA.
Estimated: conversion 20–40 min wall with three workers plus source/output
hashing (NFS-dependent); preflight requires ~86 GiB free (66 GiB data with 10%
margin + 20 GiB); smoke ~30–60 min on two GPUs; one full train8m epoch per
member 29–48 min (21.5 M rows at the recorded 8,188–13,032 rows/s plus
startup/validation); Stage 3 drafts 48–80 GPU-hours per arm, 10–16 h wall with
five GPUs per arm.

## Stage 3 drafts (inactive)

See `configs/drafts/stage3_train8m_v1/README.md`. `stage2b_campaign.py
--parity` verifies programmatically that arm A (fixed LR) and arm B (frozen
windowed_pbt_v2) share data, splits, checkpoint, optimizer/restart contract,
batch, seeds, metric, evaluation, population and member-epoch budget.

## Tests

`.venv/bin/python -m unittest tests.test_stage2b_train8m` (CPU) and, on a host
mounting the source, `STAGE2B_CPU_INTEGRATION=1 .venv/bin/python -m unittest
tests.test_stage2b_cpu_integration` (real train8m rows through the real Weaver:
train, checkpoint, resume in a new process, validate).
