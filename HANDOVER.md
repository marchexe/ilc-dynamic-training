# Handover for new sessions

Read this first. The repository and the original records (`runs/eval/*/report.md`, `analysis.json`,
`configs/prepared/*`) are the source of truth; this file only points to them. Last updated 2026-11-28.

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
| The initial PBT advantage was reduced after tuning the constant-LR baseline. Subsequent experiments did not establish a consistent advantage over tuned constant-LR training. | Sep: +1.36% (#1 vs #3). Nov, seed 12345: +0.51% (#13), +1.00% (#14, preregistered) vs fast PBT #15. Nov 9, seed 22345 **and** new grid 6×10⁻⁵–1.6×10⁻⁴ (both changed together): −0.47% (#16). #18 PBT vs constant-LR grid: 0.65%, 0.03%, 0.94% better (3 seeds). |
| Training (800k): PBT better than standard ParT by 0.81% (0.67, 0.85, 0.91% per seed; 95% Test-sample interval 0.34–1.34%). | #18, 3 seeds |
| Training (8M): PBT 12.4% worse than standard ParT; probes + final decay + higher LR range (changed together) reduce it to 2.7%. | #22–#26, **one seed** |
| Training (8M): ensemble of two standard ParT models 5.01% better than the better single model (4.86–5.14% per seed; interval 4.17–5.84%). | #27 (models of #22, #24), #28 (two new seeds); 3 seeds |

Rules for reader-facing text:
- Keep observation, hypothesis and established result separate. "800k overfits, 8M does not" is an
  observation; that this *explains* why PBT helps only at 800k is a **hypothesis, not tested**.
- Say Training (800k / 8M), Validation (150k / 1M), Test (2.8M / 9.0M). Never "holdout".
- Plain terms for the supervisor; define every term; no internal run names in prose; no AI attribution lines.
- Improvement percentages are relative changes of the composite mistag; compute them from full-precision values
  in `analysis.json` (6-decimal values in `report.md` can round differently, e.g. #16 gives −0.46% vs exact −0.465%).

## Report architecture (do not redesign)

| Output | Generator | Inputs |
|---|---|---|
| `docs/_extra/overview.html` | `scripts/reports/build_report.py` + `report_template.html` | `PHASES` (trial rows), `PHASE_TEXT` (interpretation, figures, captions), `report_cards.py` (`CARD`, `PHASE_SUMMARY` Why/Changed/Found/Next, `KEY`/`BASE`) |
| `docs/_extra/log.html` | `scripts/reports/build_log.py` | same trial texts as the report, plus run tables |
| `docs/_extra/pub/fig*.svg` | `scripts/reports/publication_figures.py` | tracked data only (`published/results/report/figure_data.json`, `published/results/sel1m/*`); `--refresh` re-extracts from `runs/` |
| `docs/report_trials.inc`, `report_runs.inc` | `scripts/reports/export_report_trials.py` | Sphinx tables of all trials |

Rebuild after any change:

```bash
.venv/bin/python scripts/reports/publication_figures.py   # FIG_PREVIEW=<dir> also writes PNG previews
.venv/bin/python scripts/reports/build_report.py
.venv/bin/python scripts/reports/build_log.py
.venv/bin/sphinx-build -W -q -b html -D html_theme=alabaster docs <scratch>/html   # furo is not installed locally
```

- Figure numbering in the report: Figure 1 = `fig3_learning_rate.svg` (PBT vs best constant LR, 4 rows incl. #16),
  2 = `fig5_mechanism.svg`, 3 = `fig4_regimes.svg`, 4 = `fig6_ensemble.svg`. SVGs are deterministic (`svg.hashsalt`).
- Asset URLs carry content hashes (`asset()` in `build_report.py`); rebuild the report after regenerating figures.
- Anchors are explicit IDs: `summary`, `phase-0`…`phase-5`, `t1`…`t33` (both pages link to each other's `t<n>`).
  Changing headings never changes anchors; check with `grep -o 'id="[^"]*"' docs/_extra/*.html` before/after.
- Primary trials visible; supporting trials (`SUPPORTING` in `build_report.py`) in a collapsed `<details>`.
- Browser check: snap Firefox cannot read `/tmp`; copy the built HTML under `$HOME` and run
  `firefox --headless --profile <dir> --no-remote --window-size=1280,14000 --screenshot <png> file://…`.
- HTML reading experience is the priority; PDF/print layout is not.

## Outstanding

- README and `docs/results.rst` still say the 800k/8M overfitting difference "explains" the PBT reversal
  (`README.md`, `results.rst` "the reason is that the model overfits…"); the report correctly labels it a
  hypothesis. Align when the user agrees.
- Report header says "Status 26 November 2026".
- Optional, needs approval: GPU equivalence test of full Lookahead state / sub-epoch (0.2-epoch) adaptation
  (`docs/ongoing.rst`, `sub-epoch-adaptation`).
