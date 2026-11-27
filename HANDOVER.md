# Handover for new sessions

Read this first. The repository and the original records (`runs/eval/*/report.md`, `analysis.json`,
`configs/prepared/*`) are the source of truth; this file only points to them. Last updated 2026-11-28 (after the visual revision: overview figure, pair scatter, comparison tables).

## State

- Research July–November 2026 is complete: 33 numbered trials. No runs are active. Do not start GPU runs
  without the user's approval (budget: about 5 GPUs at a time).
- Published: <https://marchexe.github.io/ilc-dynamic-training/> (Sphinx index),
  `overview.html` (main research report, for the supervisor), `log.html` (experiment log, one record per trial).
- Deploy: push to `main` with changes under `docs/**` or `published/**` → GitHub Actions `docs.yml`
  (Sphinx, furo theme, `-W`) → GitHub Pages.

## Scientific findings (keep this wording and evidence level)

| Finding | Evidence |
|---|---|
| The initial PBT advantage was reduced after tuning the constant-LR baseline. Subsequent experiments did not establish a consistent advantage over tuned constant-LR training. | Sep: +1.36% (#1 vs #3). Nov, seed 12345: +0.51% (#13), +1.00% (#14, preregistered) vs fast PBT #15. Nov 9, seed 22345 **and** new grid 6×10⁻⁵–1.6×10⁻⁴ (both changed together): −0.47% (#16). #18 PBT vs tuned constant-LR grid: numerically lower on all 3 seeds (0.65%, 0.03%, 0.94%), but seed 22345 is negligible (95% interval 0.74% better to 0.57% worse; seeds 12345/32345: 0.07–1.51%, 0.21–1.65% better; `runs/eval/holdout_block_bootstrap_v1/block100.json`). Say "numerical improvement on every seed", never an established, practically meaningful advantage. |
| Training (800k): PBT better than standard ParT by 0.81% (0.67, 0.85, 0.91% per seed; 95% Test-sample interval 0.34–1.34%). | #18, 3 seeds |
| Training (8M): PBT 12.4% worse than standard ParT; probes + final decay + higher LR range (changed together) reduce it to 2.7%. #23 Test is **0.1811** (exact 0.181150; older text said 0.1812). | #22–#26, **one seed**; reference best of 6 models vs PBT 5; probes +12%/+9% training |
| Training (8M): ensemble of two standard ParT models 5.01% better than the better single model (4.86–5.14% per seed; interval 4.17–5.84%). | #27 (#24 epoch 45 + #22 1e-3 epoch 50), #28 (seed runs, epochs 50+45 and 49+48); 3 seeds. Reference = better model on Validation, also better on Test. |
| Ensemble diversity (exploratory, Validation (1M), Figure 4): disagreement alone does not predict the gain; every 8M pair gaining >1% has its weaker model ≤6% worse; no pair with the weaker model >9% worse gains >1%, with or without PBT. Pairs within one run never gain. | #29, #30; `figure_data.json` → `pairs` |

Rules for reader-facing text:
- Keep observation, hypothesis and established result separate. "800k overfits, 8M does not" and "PBT lowers
  the LR earlier than standard ParT" are observations; that the short decision horizon or overfitting *explains*
  why PBT helps only at 800k is a **hypothesis, not tested** (README, results.rst, report.rst and the report all
  say so since 2026-11-28).
- Say Training (800k / 8M), Validation (150k / 1M), Test (2.8M / 9.0M). Never "holdout". The names are source
  entries per flavour file, not jets: Training (800k) = 2,392,232 jets, Training (8M) = 23,921,449 (22,921,450 after
  removing Validation (1M)). Never write "800k training jets".
- Use "x% better/worse" in words, not signed percentages; intervals as "0.24–1.43% better" or "1.50% better to
  0.05% worse".
- Test (2.8M) was evaluated in 15 campaigns (Oct 12–Nov 25) and guided the next experiments; only the new seeds of
  #18 and #28 are clean replications. Keep this limitation in the setup and conclusions.
- Plain terms for the supervisor; define every term; no internal run names in prose; no AI attribution lines.
- Improvement percentages are relative changes of the composite mistag; compute them from full-precision values
  in `analysis.json` (6-decimal values in `report.md` can round differently, e.g. #16 gives −0.46% vs exact −0.465%).

## Report architecture (do not redesign)

| Output | Generator | Inputs |
|---|---|---|
| `docs/_extra/overview.html` | `scripts/reports/build_report.py` + `report_template.html` | `PHASES` (trial rows), `PHASE_TEXT` (interpretation, figures, captions), `report_cards.py` (`CARD`, `PHASE_SUMMARY` Why/Changed/Found/Next, `None` = not shown, `KEY`/`BASE`), `OVERVIEW_CAPTION`, `table_18`/`table_28` (per-seed tables from `figure_data.json`) |
| `docs/_extra/log.html` | `scripts/reports/build_log.py` | same trial texts as the report, plus run tables, the #18/#28 tables, ensemble models (run, member, checkpoint file), phase navigation and a text filter |
| `docs/_extra/pub/fig*.svg` | `scripts/reports/publication_figures.py` | tracked data only (`published/results/report/figure_data.json`: `overview` (Figure 1, from `extract_overview()`), `pairs` (Figure 4, `extract_all_pairs()`), `trajectories_8m_seed12345`; `published/results/sel1m/*`); `--refresh` re-extracts everything from `runs/` |
| `docs/report_trials.inc`, `report_runs.inc` | `scripts/reports/export_report_trials.py` | Sphinx tables of all trials |

Rebuild after any change:

```bash
.venv/bin/python scripts/reports/publication_figures.py   # FIG_PREVIEW=<dir> also writes PNG previews
.venv/bin/python scripts/reports/build_report.py
.venv/bin/python scripts/reports/build_log.py
.venv/bin/sphinx-build -W -q -b html -D html_theme=alabaster docs <scratch>/html   # furo is not installed locally
```

- Figure numbering in the report: Figure 1 = `fig1_overview.svg` (the research trajectory, stages A–E, in the
  Summary; wider than the text column, class `overview`), 2 = `fig5_mechanism.svg` (LR and Validation per epoch, 8M),
  3 = `fig4_regimes.svg` (training/Validation loss), 4 = `fig6_ensemble.svg` (every scored pair: improvement vs
  disagreement and vs quality gap). `fig3_learning_rate.svg` was removed (now panel B of Figure 1). SVGs are
  deterministic (`svg.hashsalt`).
- Encoding in every figure (`MARK` in `publication_figures.py`): standard ParT blue filled circle, constant LR grey
  open circle, PBT orange square, ensemble green diamond. Overview rule: each stage has its own axes; never connect
  values of different stages (different samples, comparators, seeds).
- Design decision (2026-11-28): vertical trajectory chosen over a single time axis (would join incomparable values)
  and over a small-multiples grid (loses the reason for each step).
- Asset URLs carry content hashes (`asset()` in `build_report.py`); rebuild the report after regenerating figures.
- Anchors are explicit IDs: `summary`, `setup-definitions`, `timeline`, `conclusions`, `fig-1`…`fig-4`,
  `phase-0`…`phase-5`, `setup`, `t1`…`t33` (both pages link to each other's `t<n>`); the log adds `logsearch`, `nomatch`.
  Changing headings never changes anchors; check with `grep -o 'id="[^"]*"' docs/_extra/*.html` before/after.
- Primary trials visible; supporting trials (`SUPPORTING` in `build_report.py`) in a collapsed `<details>`.
- Browser check: snap Firefox cannot read `/tmp`; copy the built HTML under `$HOME` and run
  `firefox --headless --profile <dir> --no-remote --window-size=1280,14000 --screenshot <png> file://…`.
- HTML reading experience is the priority; PDF/print layout is not.

- Report header has two dates in `report_template.html`: "Research status: 26 November 2026" (last run
  finished; change only if new results arrive) and "Report revised: <date>" (update on every content revision).
- After deploying, check that Pages serves the new text and figures: `curl` the live `overview.html` (add a
  `?nc=<random>` query) and compare the `fig*.svg?v=<hash>` and the SVG bytes with the local files.

## Outstanding

- None from the report corrections of 2026-11-28.
- Optional, needs approval: GPU equivalence test of full Lookahead state / sub-epoch (0.2-epoch) adaptation
  (`docs/ongoing.rst`, `sub-epoch-adaptation`).
