# Handover for new sessions

Read this first. The repository and the original records (`runs/eval/*/report.md`, `analysis.json`,
`configs/prepared/*`) are the source of truth; this file only points to them. Last updated 2026-11-28 (P4 layout: question-led chapters, one idea per figure; approved by the user).

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
| Ensemble diversity (exploratory, Validation (1M), model-pair figure `#fig-4`): disagreement alone does not predict the gain; every 8M pair gaining >1% has its weaker model ≤6% worse; no pair with the weaker model >9% worse gains >1%, with or without PBT. Pairs with a shared training history (two checkpoints of one model, or two models of one PBT population) never gain. | #29, #30; `figure_data.json` → `pairs` |

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
- Test (2.8M) was evaluated in 15 campaigns (Oct 12–Nov 25) and guided the next experiments. The new seeds of #18
  (32345) and #28 (22345, 32345) are "replications using new training seeds under fixed protocols" on the same Test
  sample; never "clean replications". Keep this limitation in the setup and conclusions.
- Test dates differ from run dates for #1–#4 (Test measured Oct 12, #8) and #10 (Test (9.0M) opened Nov 2); every
  other campaign ran within a day of its runs. The timeline and chapter 1 say so.
- Ensembles: say "independently trained" models (no weights copied) versus a "shared training history" (checkpoints
  of one model, models of one PBT population). A computational run is not a model: the #28 run trains the two
  ensemble members side by side. Gains are always against the better of the two models.
- #17 fallback copies: kept from #18 on because they remove a stuck member (skipped ~24 times in #15; documented in
  `configs/experiments/windowed_pbt_v2_fast_fallback.yaml`), not because they helped (0.78% worse, one seed). Do not
  infer the decision process (e.g. from commit timestamps); the user asked that this not be stated (2026-11-28).
- Plain terms for the supervisor; define every term; no internal run names in prose; no AI attribution lines.
- Improvement percentages are relative changes of the composite mistag; compute them from full-precision values
  in `analysis.json` (6-decimal values in `report.md` can round differently, e.g. #16 gives −0.46% vs exact −0.465%).

## Report architecture (P4, approved 2026-11-28; refine, do not redesign)

| Output | Generator | Inputs |
|---|---|---|
| `docs/_extra/overview.html` | `scripts/reports/build_report.py` (rendering) + `report_template.html` (CSS) | `report_text.py` (all reader-facing text: findings, chapters as ordered blocks, figure captions, compact trial rows, conclusions, setup), `report_cards.py` (`CARD` for supporting trials, `PHASE_SUMMARY[k][3]` = "Next decision"), `table_18`/`table_28` in `build_report.py` |
| `docs/_extra/log.html` | `scripts/reports/build_log.py` | `PHASES` in `build_report.py` (full configuration/result/interpretation per trial), `CARD`, run manifests, the #18/#28 tables |
| `docs/_extra/pub/fig*.svg` | `scripts/reports/publication_figures.py` | tracked data only: `published/results/report/figure_data.json` (`overview` incl. `grid_test`, `trajectories_8m_seed12345`, `trajectories_800k_seed12345`, `pairs`), `published/results/sel1m/train8m_followups.json`; `--refresh` re-extracts from `runs/` |
| `docs/report_trials.inc`, `report_runs.inc` | `scripts/reports/export_report_trials.py` | Sphinx tables of all trials |

Rebuild after any change:

```bash
.venv/bin/python scripts/reports/publication_figures.py   # FIG_PREVIEW=<dir> also writes PNG previews
.venv/bin/python scripts/reports/build_report.py
.venv/bin/python scripts/reports/build_log.py
.venv/bin/sphinx-build -q -b html -D html_theme=alabaster docs <scratch>/html   # furo is not installed locally
prototypes/shoot_prod.sh   # (untracked helper) 1440 px full page + one crop per chapter
```

- Page: summary (question, four findings, contents as the five questions) → timeline (Figure 1) → five chapters
  (`phase-1`…`phase-5`), each: date/trials line → question (h2) → answer → figure with caption beside it → prose →
  primary table with note beside it → supporting experiments (collapsed) → "Next decision" → conclusions and limits →
  setup and definitions. Supporting analyses (loss curves, model pairs) are collapsed `<details>` in chapters 4 and 5.
- Figures, in reading order (numbers computed by `build_report._figure_numbers`): 1 `fig_timeline` (`#fig-1`),
  2 `fig_pbt` (`#fig-pbt`, PBT decision diagram from `reference/windowed_v2.py` and the configs), 3 `fig_baseline`
  (`#fig-baseline`, with the #16 inset), 4 `fig_lr` (`#fig-2`), 5 `fig_changes` (`#fig-changes`), 6 `fig4_regimes`
  (`#fig-3`, supporting), 7 `fig_ensemble_gain` (`#fig-ensemble`), 8 `fig6_ensemble` (`#fig-4`, supporting).
  The old anchors `fig-1`…`fig-4` keep their meaning (research overview → timeline, PBT learning rate, loss curves,
  model pairs) although the displayed numbers changed; `results.rst` links `#fig-4`. Never renumber anchors.
- `fig1_overview.svg` (five-stage overview) and `fig5_mechanism.svg` were removed on the user's request.
- Encoding (`MARK` in `publication_figures.py`): standard ParT blue filled circle, constant LR grey open circle, PBT
  orange square, ensemble green diamond. Never join values of different seeds, samples or comparators by a line.
- Main-text figures use the `@large` style (13 pt) for the 780 px column; SVGs are deterministic (`svg.hashsalt`).
- Anchors: `summary`, `timeline`, `setup` (inside the July–August note), `setup-definitions`, `conclusions`,
  `phase-0` (July–August note) … `phase-5`, `t1`…`t33` (primary rows, supporting list items, `t18` on the #18 table,
  `t27`/`t28` on its rows of the #28 table), figure anchors above. Check with
  `grep -o 'id="[^"]*"' docs/_extra/overview.html` before/after; both pages link to each other's `t<n>`.
- `pub/anchors.js` opens the enclosing `<details>` of a linked target; keep targets at most one `<details>` deep.
- Browser check: snap Firefox cannot read `/tmp`; copy the built HTML under `$HOME` and run
  `firefox --headless --profile <dir> --no-remote --window-size=1440,14000 --screenshot <png> file://…`.
- HTML reading experience is the priority; PDF/print layout is not.
- Report header has two dates (`report_text.META`): "Research status: 26 November 2026" (last run finished; change
  only if new results arrive) and "Report revised: <date>" (update on every content revision).
- After deploying, check that Pages serves the new text and figures: `curl` the live `overview.html` (add a
  `?nc=<random>` query) and compare the `fig*.svg?v=<hash>` and the SVG bytes with the local files.

## Outstanding

- None. P4 production version approved and deployed 2026-11-28 (layout final: refine text or figures only on request;
  no further redesign cycle). The setup link sits in the note under the four findings.
- Optional, needs approval: GPU equivalence test of full Lookahead state / sub-epoch (0.2-epoch) adaptation
  (`docs/ongoing.rst`, `sub-epoch-adaptation`).
