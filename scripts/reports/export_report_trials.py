#!/usr/bin/env python3
"""Trial tables of docs/report.rst, with exact dates and links to the run configurations.

    .venv/bin/python scripts/reports/export_report_trials.py

Reads runs/pbt/<run>/manifest.json (start, end, GPU time, selected member and epoch, best validation value)
and writes docs/report_trials.inc (included by docs/report.rst) and published/results/report/runs.csv.
Test values are the published values of docs/results.rst. Times are JST (UTC+9).
"""
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'runs/pbt'
JST = timezone(timedelta(hours=9))
PRIMARY = 'validation_total_reference_mistag_geomean_percent'
FIRST_EPOCH = 17  # weaver epoch of the pretrained start; training epoch = weaver epoch - 17

CONFIG = {  # runs whose configuration is not in configs/experiments under the run's own name
    'foundation_fixed_lr_50epochs_20260916': 'configs/experiments/foundation_fixed_lr_50epochs.yaml',
    'fixed_lr_continuation_20260917_lr_14e-6': 'configs/prepared/fixed_lr_continuation_20260917/lr_14e-6.yaml',
    'fixed_lr_continuation_20260917_lr_8_5e-6': 'configs/prepared/fixed_lr_continuation_20260917/lr_8_5e-6.yaml',
    'windowed_pbt_v2_100epochs': 'configs/prepared/windowed_pbt_v2_100epochs',
    'cadenced_pbt_v1_50epochs': 'configs/experiments/cadenced_pbt_v1.yaml',
    'cadenced_pbt_v1_cadence5_50epochs': 'configs/experiments/cadenced_pbt_v1_cadence5.yaml',
    'cadenced_pbt_v1_seed22345_50epochs': 'configs/experiments/cadenced_pbt_v1_seed22345.yaml',
    'cadenced_pbt_v1_cadence5_seed22345_50epochs': 'configs/experiments/cadenced_pbt_v1_cadence5_seed22345.yaml',
    'stage3_train8m_fixed_lr_v1': 'configs/experiments/stage3_train8m_fixed_lr.yaml',
    'stage3_train8m_windowed_pbt_v2_v1': 'configs/experiments/stage3_train8m_windowed_pbt_v2.yaml',
}
TEST = {  # Test (2.8M) value of the run's selected checkpoint (docs/results.rst)
    'foundation_fixed_lr_50epochs_20260916': 0.3269, 'fixed_lr_continuation_20260917_lr_14e-6': 0.3156,
    'windowed_pbt_v2': 0.3225, 'windowed_pbt_v2_100epochs': 0.3111,
    'cadenced_pbt_v1_50epochs': 0.3177, 'cadenced_pbt_v1_cadence5_50epochs': 0.3162,
    'cadenced_pbt_v1_seed22345_50epochs': 0.3167, 'cadenced_pbt_v1_cadence5_seed22345_50epochs': 0.3163,
    'stage3_train8m_fixed_lr_v1': 0.2530, 'stage3_train8m_windowed_pbt_v2_v1': 0.2359,
    'fixed_lr_high_grid_50epochs': 0.3146, 'fixed_lr_ceiling_grid_50epochs': 0.3018,
    'fixed_lr_top_grid_50epochs': 0.3033, 'windowed_pbt_v2_fast': 0.3003,
    'windowed_pbt_v2_fast_seed22345': 0.3041, 'fixed_lr_best_grid_50epochs_seed22345': 0.3027,
    'windowed_pbt_v2_fast_fallback': 0.3026,
    'sel1m_standard_part': 0.3029, 'sel1m_fixed_lr_grid': 0.3028, 'sel1m_fast_pbt': 0.3008,
    'sel1m_standard_part_seed22345': 0.3038, 'sel1m_fixed_lr_grid_seed22345': 0.3013,
    'sel1m_fast_pbt_seed22345': 0.3012, 'sel1m_standard_part_seed32345': 0.3024,
    'sel1m_fixed_lr_grid_seed32345': 0.3024, 'sel1m_fast_pbt_seed32345': 0.2996,
    'sel1m_fast_pbt_decay': 0.3008, 'sel1m_fast_pbt_decay_seed22345': 0.3012, 'sel1m_fast_pbt_probe': 0.3024,
    'sel1m_train8m_standard_part': 0.1635, 'sel1m_train8m_fast_pbt': 0.1812,
    'sel1m_train8m_standard_part_3e-3': 0.1611, 'sel1m_train8m_fast_pbt_probe': 0.1699,
    'sel1m_train8m_fast_pbt_probe_decay': 0.1655, 'sel1m_train8m_standard_part_seed22345': 0.1597,
    'sel1m_train8m_standard_part_seed32345': 0.1623, 'sel1m_train8m_standard_part_1e-2': 0.1599,
}

# Phases: (title, data, validation set, trials). Trial: (number, what, runs, records, dates if no run, result,
# outcome). Records are repository paths (test specifications, analyses) or :ref: targets.
PHASES = [
    ('Phase 1: PBT against fixed LRs', 'Training (800k)', 'Validation (150k)', [
        (1, 'Fixed LRs 3×10⁻⁶ – 1.4×10⁻⁵', ['foundation_fixed_lr_50epochs_20260916'], [':doc:`run page <experiments/fixed-lr-50>`'], None, '0.3269', '🔧'),
        (2, 'Two fixed-LR members continued to 100 epochs', ['fixed_lr_continuation_20260917_lr_14e-6', 'fixed_lr_continuation_20260917_lr_8_5e-6'], [':doc:`run page <experiments/fixed-lr-100>`'], None, '0.3156', '🔧'),
        (3, 'Windowed PBT: decision every 5 epochs, ×0.8 or ×1.2', ['windowed_pbt_v2'], [':doc:`run page <experiments/pbt-v2-50>`'], None, '0.3225, −1.4% vs 1', '⚠️'),
        (4, 'Windowed PBT, 100 epochs', ['windowed_pbt_v2_100epochs'], [':doc:`run page <experiments/pbt-v2-100>`'], None, '0.3111, −1.4% vs 2', '⚠️'),
        (5, 'LR controller on a 60k Validation subset, observe only', ['pretrained_fixed_lr_8gpu_representative60k_shadow'], [':repo:`audit <published/studies/controller-shadow-audit/report.md>`'], None, 'controller did nothing', '➖'),
        (6, 'PBT every epoch vs every 5 epochs, 2 seeds', ['cadenced_pbt_v1_50epochs', 'cadenced_pbt_v1_cadence5_50epochs', 'cadenced_pbt_v1_seed22345_50epochs', 'cadenced_pbt_v1_cadence5_seed22345_50epochs'], [':doc:`run page <experiments/cadenced-pbt-v1-50>`'], None, '+0.5%, +0.1% (Test reverses Validation)', '➖'),
        (7, 'PBT decisions inside an epoch: restart test', [], [':ref:`result <sub-epoch-restart-test>`'], 'Sep 28', 'not possible: optimizer state lost', '🔧'),
    ]),
    ('Phase 2: first large-data comparison', 'Training (8M)', 'Validation (150k)', [
        (8, 'First Test (2.8M) evaluation of trials 1–6', [], [':repo:`protocol <configs/prepared/stage2a_holdout_v1>`'], 'Oct 12', 'agrees with Validation for large differences', '🔧'),
        (9, 'Prepare Training (8M) and Test (9.0M); one-epoch GPU check', [], [':repo:`records <configs/prepared/stage2b_train8m_v1>`'], 'Oct 13', 'ready', '🔧'),
        (10, 'PBT vs fixed LRs as in trial 1, preregistered', ['stage3_train8m_fixed_lr_v1', 'stage3_train8m_windowed_pbt_v2_v1'], [':repo:`protocol <configs/prepared/stage3_train8m_v1>`', ':ref:`result <large-dataset-results>`'], None, '−7.7% on Test (9.0M): 0.2375 vs 0.2572', '⚠️'),
        (11, 'Save the full optimizer state; GPU check', ['full_state_gpu_one_epoch', 'full_state_gpu_fifths'], [':ref:`details <sub-epoch-adaptation>`'], None, 'bit-identical; not used in a PBT run', '🔧'),
    ]),
    ('Phase 3: higher fixed LRs', 'Training (800k)', 'Validation (150k)', [
        (12, 'Fixed LRs 1.4×10⁻⁵ – 3×10⁻⁵', ['fixed_lr_high_grid_50epochs'], [':repo:`test spec <configs/prepared/holdout_high_lr_grid_v1>`'], None, '0.3146', '🔧'),
        (13, 'Fixed LRs 3×10⁻⁵ – 8×10⁻⁵', ['fixed_lr_ceiling_grid_50epochs'], [':repo:`test spec <configs/prepared/holdout_fixed_lr_ceiling_v1>`'], None, '**0.3018**, better than all PBT so far', '🔧'),
        (14, 'Fixed LRs 8×10⁻⁵ – 3.2×10⁻⁴', ['fixed_lr_top_grid_50epochs'], [':repo:`test spec <configs/prepared/holdout_fast_pbt_v1>`'], None, '0.3033', '🔧'),
        (15, 'Fast PBT: decision every 2 epochs, ×0.5 or ×2', ['windowed_pbt_v2_fast'], [':repo:`test spec <configs/prepared/holdout_fast_pbt_v1>`'], None, '0.3003, −1.0% vs 14', '✅'),
        (16, 'Trials 14 and 15 with seed 22345', ['windowed_pbt_v2_fast_seed22345', 'fixed_lr_best_grid_50epochs_seed22345'], [':repo:`test spec <configs/prepared/holdout_fast_pbt_seed22345_v1>`'], None, 'PBT 0.3041 vs 0.3027: +0.5%', '❌'),
        (17, 'Fast PBT with extra copies', ['windowed_pbt_v2_fast_fallback'], [':repo:`test spec <configs/prepared/holdout_fast_pbt_seed22345_v1>`'], None, '0.3026, +0.8% vs 15', '❌'),
    ]),
    ('Phase 4: PBT against standard ParT', 'Training (800k)', 'Validation (1M)', [
        (18, 'Standard ParT, fixed LRs and fast PBT; 3 seeds', ['sel1m_standard_part', 'sel1m_fixed_lr_grid', 'sel1m_fast_pbt', 'sel1m_standard_part_seed22345', 'sel1m_fixed_lr_grid_seed22345', 'sel1m_fast_pbt_seed22345', 'sel1m_standard_part_seed32345', 'sel1m_fixed_lr_grid_seed32345', 'sel1m_fast_pbt_seed32345'], [':repo:`plan <configs/prepared/sel1m_plan.md>`', ':repo:`test spec <configs/prepared/holdout_sel1m_v1>`', ':repo:`seed 32345 <configs/prepared/holdout_sel1m_seed32345_v1>`', ':ref:`result <pbt-vs-standard-part>`'], None, '**PBT −0.8%** (−0.7, −0.9, −0.9)', '✅'),
        (19, 'PBT with a final decay, 2 seeds', ['sel1m_fast_pbt_decay', 'sel1m_fast_pbt_decay_seed22345'], [], None, 'same models as 18', '➖'),
        (20, 'Standard ParT without pretraining', ['sel1m_scratch_standard_part'], [], None, '0.378 (Validation), far worse', '❌'),
        (21, 'Weight averages of checkpoints', [], [':repo:`spec <configs/prepared/weight_averaging_sel1m_v1/spec.json>`'], 'Nov 12', 'no gain (Validation)', '➖'),
    ]),
    ('Phase 5: large data against standard ParT', 'Training (8M)', 'Validation (1M)', [
        (22, 'Standard ParT, start LRs 1×10⁻⁵ – 1×10⁻³', ['sel1m_train8m_standard_part'], [':repo:`test spec <configs/prepared/holdout_sel1m_train8m_v1>`'], None, '0.1635', '🔧'),
        (23, 'Fast PBT with the LR range of standard ParT', ['sel1m_train8m_fast_pbt'], [':repo:`test spec <configs/prepared/holdout_sel1m_train8m_v1>`', ':ref:`result <pbt-vs-standard-part-8m>`'], None, '0.1812, **+12.4%** vs 24', '❌'),
        (24, 'Standard ParT, start LR 3×10⁻³', ['sel1m_train8m_standard_part_3e-3'], [':repo:`test spec <configs/prepared/holdout_sel1m_train8m_lr3e-3_v1>`'], None, '**0.1611**, the reference', '🔧'),
        (25, 'PBT with cool-down probes (8M and 800k)', ['sel1m_train8m_fast_pbt_probe', 'sel1m_fast_pbt_probe'], [':repo:`test spec <configs/prepared/holdout_sel1m_probe_v1>`', ':ref:`result <probe-scored-pbt>`'], None, '0.1699, +5.4% vs 24', '❌'),
        (26, 'PBT with probes and a final decay', ['sel1m_train8m_fast_pbt_probe_decay'], [':repo:`test spec <configs/prepared/holdout_sel1m_probe_decay_v1>`'], None, '0.1655, +2.7% vs 24', '❌'),
        (27, 'Ensemble of trials 22 and 24', [], [':repo:`spec <configs/prepared/ensemble_8m_dev_v1/spec.json>`', ':ref:`result <prediction-ensemble>`'], 'Nov 23', '0.1530, **−5.1%** vs 24', '✅'),
        (28, 'Trial 27 with seeds 22345 and 32345', ['sel1m_train8m_standard_part_seed22345', 'sel1m_train8m_standard_part_seed32345'], [':repo:`spec 22345 <configs/prepared/ensemble_seed22345_v1/spec.json>`', ':repo:`spec 32345 <configs/prepared/ensemble_seed32345_v1/spec.json>`'], None, '−4.9%, −5.1%; **mean −5.0%**', '✅'),
        (29, 'Ensembles at 800k, 15 models', [], [':ref:`result <ensemble-800k>`'], 'Nov 23', 'no gain (Validation)', '➖'),
        (30, 'Where the ensemble gain comes from', [], [':ref:`result <ensemble-diversity>`'], 'Nov 23–25', 'mostly from separate training', '🔧'),
        (31, 'Final decay of 24 without dropout', [], [':ref:`result <dropout-branch>`'], 'Nov 23–24', 'much worse (Validation)', '❌'),
        (32, 'Distillation of the ensemble into one model', [], [':repo:`network <networks/distill_particle_transformer.py>`', ':ref:`result <distillation>`'], 'Nov 24', '+1.0% (Validation)', '❌'),
        (33, 'Standard ParT, start LR 1×10⁻²', ['sel1m_train8m_standard_part_1e-2'], [':repo:`test spec <configs/prepared/holdout_sel1m_train8m_lr1e-2_v1>`'], None, '0.1599, as 24', '➖'),
    ]),
]


def ts(text):
    return datetime.fromisoformat(text).astimezone(JST)


def run_record(name):
    m = json.loads((RUNS / name / 'manifest.json').read_text())
    gens = m['generations']
    seconds = sum((ts(w['finished_at']) - ts(w['started_at'])).total_seconds()
                  for g in gens for w in g['workers'].values() if w.get('started_at') and w.get('finished_at'))
    values = [(w['metrics'][PRIMARY], g['epoch'], member, w.get('lr')) for g in gens
              for member, w in g['workers'].items() if (w.get('metrics') or {}).get(PRIMARY) is not None]
    value, epoch, member, lr = min(values)
    config = CONFIG.get(name) or f'configs/experiments/{name}.yaml'
    if not (ROOT / config).exists():
        raise SystemExit(f'{name}: {config} missing')
    return dict(run=name, config=config, start=ts(m['started_at']), end=ts(m['finished_at']),
                epochs=len(gens), members=len(m['config']['population']['members'])
                if isinstance(m['config'].get('population'), dict) and 'members' in m['config']['population']
                else len(gens[0]['workers']),
                gpu_hours=seconds / 3600, member=member, lr=lr, epoch=epoch - FIRST_EPOCH, validation=value,
                test=TEST.get(name))


def sci(x):
    m, e = f'{x:.2e}'.split('e')
    sup = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')
    return f'{float(m):g}×10{str(int(e)).translate(sup)}'


def span(records):
    a, b = min(r['start'] for r in records), max(r['end'] for r in records)
    return f'{a:%b %d %H:%M} → {b:%b %d %H:%M}'


def cell(lines):
    return ('\n       '.join(lines)) if lines else ''


def main():
    out = ['.. This file is generated by scripts/reports/export_report_trials.py; do not edit.', '']
    rows = []
    for title, data, validation, trials in PHASES:
        out += [f'.. _report-{title.split(":")[0].lower().replace(" ", "-")}:', '',
                f'**{title}** · {data} · {validation}', '',
                '.. list-table::', '   :header-rows: 1', '   :widths: 4 15 47 24 6', '',
                '   * - #', '     - Dates (JST)', '     - What, runs and records', '     - Result', '     -']
        for number, what, runs, records, dates, result, outcome in trials:
            recs = [run_record(r) for r in runs]
            rows += [dict(trial=number, **r) for r in recs]
            links = [f':repo:`{r["run"]} <{r["config"]}>`' for r in recs] + records
            when = span(recs) if recs else dates
            gpu = f' · {sum(r["gpu_hours"] for r in recs):.0f} GPU-h' if recs else ''
            out += [f'   * - {number}', f'     - {when}', f'     - {what}{gpu}', '',
                    f'       {" · ".join(links)}' if links else '', f'     - {result}', f'     - {outcome}']
        out.append('')
    (ROOT / 'docs/report_trials.inc').write_text('\n'.join(out) + '\n')

    run_table = ['.. This file is generated by scripts/reports/export_report_trials.py; do not edit.', '',
                 '.. list-table::', '   :header-rows: 1', '   :widths: 4 32 20 8 16 10 10', '',
                 '   * - #', '     - Run (configuration)', '     - Start → end (JST)', '     - GPU-h',
                 '     - Selected model: name, LR, epoch', '     - Validation', '     - Test (2.8M)']
    for r in rows:
        run_table += [f'   * - {r["trial"]}', f'     - :repo:`{r["run"]} <{r["config"]}>`',
                      f'     - {r["start"]:%b %d %H:%M} → {r["end"]:%b %d %H:%M}', f'     - {r["gpu_hours"]:.0f}',
                      f'     - {r["member"]}, {sci(r["lr"])}, {r["epoch"]}', f'     - {r["validation"]:.4f}',
                      f'     - {r["test"]:.4f}' if r['test'] else '     - —']
    (ROOT / 'docs/report_runs.inc').write_text('\n'.join(run_table) + '\n')

    path = ROOT / 'published/results/report/runs.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as handle:
        w = csv.writer(handle)
        w.writerow(['trial', 'run', 'config', 'start_jst', 'end_jst', 'epochs', 'gpu_hours', 'selected_member',
                    'selected_lr', 'selected_epoch', 'validation', 'test_2p8m'])
        for r in rows:
            w.writerow([r['trial'], r['run'], r['config'], f'{r["start"]:%Y-%m-%d %H:%M}', f'{r["end"]:%Y-%m-%d %H:%M}',
                        r['epochs'], f'{r["gpu_hours"]:.1f}', r['member'], f'{r["lr"]:.3g}', r['epoch'], f'{r["validation"]:.6f}',
                        '' if r['test'] is None else f'{r["test"]:.4f}'])
    print(f'{len(rows)} runs')


if __name__ == '__main__':
    main()
