"""Stage 3 analysis and report (preregistered endpoints, validity first).

The report is built from recorded run evidence only. Validity is checked
before any comparison is stated: an invalid or incomplete campaign is labelled
so and never presented as a successful comparison. Validation-selected values
are separated from the sealed independent test.
"""
import csv
import math
from pathlib import Path
from statistics import mean, pstdev

from experiments.stage3.common import ARMS, PRIMARY, file_record, load_json, sha256, utc_now, write_json
from training.pbt.reporting.endpoints import current_best_trajectory, final_window_mean
from training.pbt.state.checkpointing import checkpoint_paths
from training.runtime import WORKING_POINT_DEFINITION, mistag_percent_key

WORKING_POINTS = [mistag_percent_key(pair, eff) for pair, eff in WORKING_POINT_DEFINITION]
ARM_LABEL = dict(fixed_lr='A fixed LR', windowed_pbt_v2='B windowed PBT v2')
# Reference categorical slots 1-2 (validated pair in the dataviz reference palette).
ARM_COLOR = dict(fixed_lr='#2a78d6', windowed_pbt_v2='#eb6834')
MEMBER_COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4')


def completed(manifest):
    return [g for g in sorted(manifest.get('generations', []), key=lambda g: g['index']) if g.get('status') == 'completed']


def members(manifest):
    return [m['name'] for m in sorted(manifest['config']['population'], key=lambda m: m['start_lr'])]


def trajectories(manifest, key=PRIMARY, end=None):
    rows = completed(manifest)[:end]
    return {name: [g['workers'][name]['metrics'][key] for g in rows] for name in members(manifest)}


def seconds(start, end):
    from datetime import datetime
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()


def arm_summary(manifest, end, mode='min'):
    rows = completed(manifest)[:end]
    values = trajectories(manifest, end=end)
    current = current_best_trajectory(values, mode)
    final = rows[-1]
    final_values = {n: final['workers'][n]['metrics'][PRIMARY] for n in members(manifest)}
    events = [dict(generation=g['index'], full_epoch=g['index'] + 1, **{k: e[k] for k in ('donor', 'recipient', 'new_lr', 'mutation_factor')})
              for g in rows for e in (g.get('exploit') or [])]
    workers = [w for g in rows for w in g['workers'].values()]
    audits = [w['metrics']['train_data_audit'] for w in workers]
    gpu_seconds = sum(seconds(w['started_at'], w['finished_at']) for w in workers)
    best = manifest.get('best') or {}
    spreads = [dict(full_epoch=i + 1, std=pstdev(v), range=max(v) - min(v))
               for i, v in enumerate(zip(*values.values()))]
    lrs = {n: [g['workers'][n]['lr'] for g in rows] for n in members(manifest)}
    return dict(
        generations=len(rows), complete=len(rows) == manifest['config']['shared']['generations'],
        final10_current_best=final_window_mean(current, 10) if len(current) >= 10 else None,
        final10_epochs=[len(current) - 9, len(current)] if len(current) >= 10 else None,
        current_best=current, global_best=dict(value=best.get('metric_value'), member=best.get('member'),
                                               full_epoch=(best.get('generation', -2) + 1), epoch=best.get('epoch')),
        final_members=final_values, final_population_mean=mean(final_values.values()),
        final_current_best_member=min(sorted(final_values), key=final_values.get),
        final_working_points={n: {k: final['workers'][n]['metrics'][k] for k in WORKING_POINTS} for n in members(manifest)},
        train_loss=trajectories(manifest, 'train_loss', end), validation_loss=trajectories(manifest, 'validation_loss', end),
        trajectories=values, learning_rates=lrs, exploit_events=events, spreads=spreads,
        optimizer_steps=sum(a['optimizer_steps'] for a in audits), accepted_rows=sum(a['consumed']['count'] for a in audits),
        amp_skipped=sum(w['metrics'].get('train_amp_skipped_optimizer_steps', 0) or 0 for w in workers),
        max_cuda_mb=max(w['metrics'].get('train_max_cuda_memory_mb') or 0 for w in workers),
        worker_gpu_hours=gpu_seconds / 3600,
        wall_hours=seconds(min(w['started_at'] for w in workers), max(w['finished_at'] for w in workers)) / 3600)


def lineage_identity(runs, manifests, end):
    """Bit-identity of same-named members until each windowed member's first copy.

    A windowed member that has not yet been copied into has the same weights,
    LR, seed and data as its fixed-LR namesake, so deterministic training must
    reproduce it exactly. At the copy generation the recorded pre-copy archive
    must still match. Any mismatch means nondeterminism confounds the comparison.
    """
    fixed, windowed = manifests['fixed_lr'], manifests['windowed_pbt_v2']
    first_copy = {}
    for generation in completed(windowed)[:end]:
        for event in generation.get('exploit') or []:
            first_copy.setdefault(event['recipient'], (generation['index'], event))
    checked, mismatches = 0, []
    for name in members(fixed):
        stop, event = first_copy.get(name, (end, None))
        for generation in completed(fixed)[:stop]:
            paths = [checkpoint_paths(Path(runs[a]) / name, generation['epoch'])[0] for a in ARMS]
            checked += 1
            if not all(p.is_file() for p in paths) or sha256(paths[0]) != sha256(paths[1]):
                mismatches.append(dict(member=name, epoch=generation['epoch']))
        if event:
            epoch = completed(windowed)[stop]['epoch']
            reference = checkpoint_paths(Path(runs['fixed_lr']) / name, epoch)[0]
            checked += 1
            if sha256(reference) != event['pre_copy']['state']['sha256']:
                mismatches.append(dict(member=name, epoch=epoch, pre_copy=True))
    return dict(checked_states=checked, mismatches=mismatches, identical=not mismatches,
                first_copy={n: v[0] + 1 for n, v in first_copy.items()})


def validity(protocol, runs, manifests, *, interim=False, gate=None, verifier=None, identity=None):
    """Preregistered validity checks; an empty failure list is required to compare."""
    from training.pbt.config import contract_fingerprint
    from validation import verify_fixed_lr
    verifier = verifier or verify_fixed_lr
    checks = []

    def check(name, condition, detail=None, kind='required'):
        checks.append(dict(check=name, passed=bool(condition), detail=detail, kind=kind))

    end = min(len(completed(m)) for m in manifests.values())
    horizon = protocol['design']['horizon_epochs']
    if end == 0:
        check('at least one completed generation in both arms', False)
        return checks, 0, dict(checked_states=0, mismatches=[], identical=True, first_copy={})
    for arm, manifest in manifests.items():
        frozen = protocol['identity']['configs'][arm]
        check(f'{arm}: configuration equals the frozen protocol', contract_fingerprint(manifest['config']) == frozen['fingerprint'])
        if not interim:
            check(f'{arm}: completed {horizon} generations', manifest.get('status') == 'completed' and end == horizon)
        evidence = verifier.verify(Path(runs[arm]), None if not interim else max(end, 1))
        check(f'{arm}: verify_fixed_lr (traversal, continuation, decisions, final evaluations)', evidence['passed'],
              evidence['failures'][:20])
    schedules = []
    for generation_index in range(end):
        consumed = {json_key(manifests[a], generation_index, n) for a in ARMS for n in members(manifests[a])}
        schedules.append(len(consumed) == 1)
    check('identical accepted training rows for all ten workers in every generation', end > 0 and all(schedules),
          [i for i, ok in enumerate(schedules) if not ok][:10])
    initial = [((m.get('initial_evaluation') or {}).get('metrics') or {}).get(PRIMARY) for m in manifests.values()]
    check('identical initial evaluation', initial[0] is not None and initial[0] == initial[1], initial)
    if gate and gate.get('measurements', {}).get('reference_state_sha256'):
        epoch = protocol['design']['initial_epoch'] + 1
        hashes = [sha256(checkpoint_paths(Path(runs[a]) / protocol['gate']['member'], epoch)[0]) for a in ARMS]
        check('gate epoch-18 state reproduced by both arms', hashes == [gate['measurements']['reference_state_sha256']] * 2,
              kind='determinism')
    identity = identity or lineage_identity(runs, manifests, end)
    check('same-lineage members bit-identical until their first copy', identity['identical'], identity['mismatches'][:10],
          kind='determinism')
    return checks, end, identity


def json_key(manifest, index, name):
    import json
    audit = completed(manifest)[index]['workers'][name]['metrics']['train_data_audit']
    return json.dumps(audit['consumed'], sort_keys=True)


def comparison(summaries, margin):
    a, b = summaries['fixed_lr'], summaries['windowed_pbt_v2']

    def delta(left, right):
        if left is None or right is None:
            return None
        value = right - left
        return dict(fixed_lr=left, windowed_pbt_v2=right, delta_pp=value, relative_percent=100 * value / left,
                    within_selection_noise_margin=abs(value) < margin)

    return dict(primary_final10_current_best=delta(a['final10_current_best'], b['final10_current_best']),
                global_best_validation=delta(a['global_best']['value'], b['global_best']['value']),
                final_population_mean=delta(a['final_population_mean'], b['final_population_mean']),
                final_members={n: delta(a['final_members'][n], b['final_members'][n]) for n in a['final_members']},
                margin_pp=margin)


def figures(directory, summaries, *, test=None):
    from training.pbt.reporting.style import plot_setup
    plt = plot_setup()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []

    def save(fig, name):
        path = directory / name
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        written.append(path)

    epochs = range(1, max(s['generations'] for s in summaries.values()) + 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    end = len(epochs)
    if end >= 10:
        ax.axvspan(end - 9.5, end + 0.5, color='#d9d8d4', alpha=0.5, lw=0, label='final-10 window')
    for arm, summary in summaries.items():
        values = summary['current_best']
        ax.plot(range(1, len(values) + 1), values, color=ARM_COLOR[arm], lw=2, label=ARM_LABEL[arm])
        ax.annotate(ARM_LABEL[arm].split(' ', 1)[0], (len(values), values[-1]), xytext=(4, 0),
                    textcoords='offset points', va='center', fontsize=8, color='#52514e')
    ax.set(xlabel='full train8m epoch', ylabel='current-best primary mistag (%)',
           title='Population current-best on val50k_tail (selection split)')
    ax.grid(axis='y', color='#ebeae6', lw=0.8)
    ax.legend(frameon=False)
    save(fig, '01_current_best_primary.png')

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (arm, summary) in zip(axes, summaries.items()):
        for color, (name, values) in zip(MEMBER_COLORS, summary['trajectories'].items()):
            ax.plot(range(1, len(values) + 1), values, color=color, lw=1.4, label=name)
        for event in summary['exploit_events']:
            ax.axvline(event['full_epoch'], color='#a3a29d', lw=0.8, ls=':')
        ax.set(title=ARM_LABEL[arm], xlabel='full epoch')
        ax.grid(axis='y', color='#ebeae6', lw=0.8)
    axes[0].set_ylabel('member primary mistag (%)')
    axes[1].legend(frameon=False, fontsize=7, title='member (start LR)', title_fontsize=7)
    save(fig, '02_member_trajectories.png')

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    windowed = summaries['windowed_pbt_v2']
    for color, (name, values) in zip(MEMBER_COLORS, windowed['learning_rates'].items()):
        ax.step(range(1, len(values) + 1), values, where='post', color=color, lw=1.6, label=name)
    for event in windowed['exploit_events']:
        ax.scatter([event['full_epoch'] + 1], [event['new_lr']], s=36, marker='o', facecolor='white', edgecolor='#0b0b0b', zorder=3)
    ax.set_yscale('log')
    ax.set(xlabel='full epoch', ylabel='learning rate', title='Arm B learning rates (circles: copied members; arm A is constant)')
    ax.legend(frameon=False, fontsize=7)
    save(fig, '03_windowed_learning_rates.png')

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for arm, summary in summaries.items():
        losses = [mean(v) for v in zip(*summary['train_loss'].values())]
        ax.plot(range(1, len(losses) + 1), losses, color=ARM_COLOR[arm], lw=2, label=ARM_LABEL[arm])
    ax.set(xlabel='full epoch', ylabel='population mean training loss', title='Training stability')
    ax.grid(axis='y', color='#ebeae6', lw=0.8)
    ax.legend(frameon=False)
    save(fig, '04_training_loss.png')

    if test and test.get('comparisons', {}).get('selected_best'):
        working = test['comparisons']['selected_best']['working_points']
        left = test['metrics'][test['comparisons']['selected_best']['left']]
        names = list(working)
        relative = [100 * working[k]['delta_pp'] / left[k] for k in names]
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        ax.bar(range(len(names)), relative, color=ARM_COLOR['windowed_pbt_v2'], width=0.6)
        ax.axhline(0, color='#52514e', lw=1)
        ax.set_xticks(range(len(names)), [n.replace('validation_', '').replace('_mistag_eff_', '@').replace('_percent', '') for n in names],
                      rotation=30, ha='right')
        for i, value in enumerate(relative):
            ax.annotate(f'{value:+.1f}%', (i, value), xytext=(0, 3 if value >= 0 else -10), textcoords='offset points',
                        ha='center', fontsize=7, color='#52514e')
        ax.set(ylabel='B vs A relative mistag change (%)', title='Sealed test3000k: selected-best checkpoints (negative = B better)')
        save(fig, '05_test_working_points.png')
    return written


def build(output, protocol, runs, *, interim=False, gate=None, test=None, holdout=None, operations=None, verifier=None):
    """Write analysis.json, comparison.csv, figures and report.md (+ integrity record)."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifests = {arm: load_json(Path(runs[arm]) / 'manifest.json') for arm in ARMS}
    checks, end, identity = validity(protocol, runs, manifests, interim=interim, gate=gate, verifier=verifier)
    try:
        summaries = {arm: arm_summary(manifests[arm], end) for arm in ARMS} if end else {}
    except (KeyError, TypeError, ValueError) as error:
        checks.append(dict(check='finite, complete recorded trajectories', passed=False, detail=str(error)))
        summaries = {}
    valid = all(c['passed'] for c in checks if c.get('kind', 'required') == 'required')
    deterministic = all(c['passed'] for c in checks if c.get('kind') == 'determinism')
    status = ('INTERIM' if interim else 'COMPLETE') if valid else ('INVALID' if end else 'INCOMPLETE')
    compared = comparison(summaries, 0.002) if valid and end >= 10 else None
    data = dict(schema_version=1, generated_at=utc_now(), protocol=protocol['name'], status=status, interim=interim,
                determinism_confirmed=deterministic, completed_generations=end, validity=checks, lineage_identity=identity, summaries=summaries,
                comparison=compared, gate=gate, sealed_test=test, holdout=holdout, operations=operations)
    figure_paths = figures(output / 'figures', summaries, test=test if valid else None) if summaries else []
    write_json(output / 'analysis.json', data)
    with (output / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['quantity', 'fixed_lr', 'windowed_pbt_v2', 'delta_pp', 'relative_percent', 'within_margin'])
        for key, value in (compared or {}).items():
            rows = value.items() if key == 'final_members' else [(key, value)]
            for name, row in rows:
                if isinstance(row, dict):
                    writer.writerow([name if key != 'final_members' else f'final_member_{name}', row['fixed_lr'],
                                     row['windowed_pbt_v2'], row['delta_pp'], row['relative_percent'],
                                     row['within_selection_noise_margin']])
    (output / 'report.md').write_text(markdown(data, protocol, figure_paths, output))
    write_json(output / 'report_integrity.json', dict(generated_at=utc_now(), artifacts=[
        file_record(p) for p in [output / 'analysis.json', output / 'comparison.csv', output / 'report.md', *figure_paths]]))
    return data


def _f(value, digits=6):
    return '—' if value is None or (isinstance(value, float) and not math.isfinite(value)) else (
        f'{value:.{digits}f}' if isinstance(value, float) else str(value))


def markdown(data, protocol, figure_paths, output):
    status = data['status']
    banner = dict(COMPLETE='Полный сравнимый эксперимент: все предрегистрированные проверки валидности пройдены.',
                  INTERIM='ПРОМЕЖУТОЧНЫЙ описательный срез; не конечная точка, не основание для остановки или продления.',
                  INVALID='НЕДЕЙСТВИТЕЛЬНО: проверки валидности не пройдены; сравнение не делается.',
                  INCOMPLETE='НЕПОЛНЫЙ: нет завершённых поколений для сравнения.')[status]
    lines = [f'# Stage 3: fixed LR vs windowed PBT v2 на train8m — {status}', '', f'**{banner}**', '']
    if not data.get('determinism_confirmed', True):
        lines += ['**Детерминизм не подтверждён:** члены одной линии или эталон gate не совпали побитово; часть '
                  'различия может быть численным шумом, а не эффектом политики.', '']
    lines += [
             f'Протокол `{protocol["name"]}`; горизонт {protocol["design"]["horizon_epochs"]} эпох; '
             f'завершено поколений: {data["completed_generations"]}. Сформировано {data["generated_at"]}.', '',
             '## Проверки валидности', '', '| Проверка | Тип | Результат |', '|---|---|---|']
    for check in data['validity']:
        lines.append(f'| {check["check"]} | {check.get("kind", "required")} | '
                     f'{"OK" if check["passed"] else "FAIL " + str(check.get("detail"))[:160]} |')
    identity = data['lineage_identity']
    lines += ['', f'Проверено состояний одной линии: {identity["checked_states"]}; первые копирования в B (полная эпоха): '
              f'{identity["first_copy"] or "нет"}.']
    compared = data['comparison']
    if compared:
        p = compared['primary_final10_current_best']
        lines += ['', '## Первичная конечная точка (validation, отбор)', '',
                  'Final-10 current-best population mean (Stage 1), `val50k_tail`, меньше — лучше.', '',
                  '| | A fixed LR | B windowed PBT | Δ (B−A), п.п. | Относительно, % |', '|---|---:|---:|---:|---:|']
        for key, title in (('primary_final10_current_best', 'Final-10 current-best'),
                           ('global_best_validation', 'Global-best validation'),
                           ('final_population_mean', 'Среднее финальной популяции')):
            row = compared[key]
            lines.append(f'| {title} | {_f(row["fixed_lr"])} | {_f(row["windowed_pbt_v2"])} | {_f(row["delta_pp"])} | '
                         f'{_f(row["relative_percent"], 3)} |')
        lines += ['', f'|Δ| первичной конечной точки {"меньше" if p["within_selection_noise_margin"] else "не меньше"} '
                  f'запаса {compared["margin_pp"]} п.п. (порог решения windowed на той же выборке). Один согласованный '
                  'seed: результат описательный, статистическая значимость не утверждается. Validation-выборка '
                  'используется и для решений PBT, поэтому validation-значения B оптимистично смещены.']
    summaries = data['summaries']
    if summaries:
        lines += ['', '## Финальные члены (validation)', '', '| Член | A | B | Скорость обучения B в конце |', '|---|---:|---:|---:|']
        for name in summaries['fixed_lr']['final_members']:
            lines.append(f'| {name} | {_f(summaries["fixed_lr"]["final_members"][name])} | '
                         f'{_f(summaries["windowed_pbt_v2"]["final_members"][name])} | '
                         f'{summaries["windowed_pbt_v2"]["learning_rates"][name][-1]:.3g} |')
        lines += ['', '## Решения PBT, вычисления, стабильность', '']
        events = summaries['windowed_pbt_v2']['exploit_events']
        lines.append(f'- Копирований в B: {len(events)}; ' + ', '.join(
            f'эпоха {e["full_epoch"]}: {e["donor"]}→{e["recipient"]} (×{e["mutation_factor"]}, LR {e["new_lr"]:.3g})' for e in events))
        for arm, s in summaries.items():
            lines.append(f'- {ARM_LABEL[arm]}: optimizer steps {s["optimizer_steps"]:,}; принятых строк {s["accepted_rows"]:,}; '
                         f'AMP skipped {s["amp_skipped"]}; CUDA max {s["max_cuda_mb"]:.0f} MB; GPU-часы воркеров '
                         f'{s["worker_gpu_hours"]:.1f}; стена {s["wall_hours"]:.1f} ч; разброс членов в конце '
                         f'(std) {s["spreads"][-1]["std"]:.5f}.')
    test = data.get('sealed_test')
    lines += ['', '## Независимый тест test3000k (запечатан до завершения обеих рук)', '']
    if test and test.get('comparisons'):
        lines += ['| Сравнение (B−A) | A | B | Δ, п.п. | Снижение, % |', '|---|---:|---:|---:|---:|']
        for name, row in test['comparisons'].items():
            lines.append(f'| {name} | {_f(test["metrics"][row["left"]][PRIMARY])} | {_f(test["metrics"][row["right"]][PRIMARY])} | '
                         f'{_f(row["delta_pp"])} | {_f(row["reduction_percent"], 3)} |')
        if test.get('population_mean'):
            pm = test['population_mean']
            lines.append(f'| final population mean | {_f(pm["fixed_lr"])} | {_f(pm["windowed_pbt_v2"])} | {_f(pm["delta_pp"])} | '
                         f'{_f(pm["reduction_percent"], 3)} |')
        lines += ['', 'Парные счётчики фона по восьми рабочим точкам — в analysis.json. Без event ID нет '
                  'событийного доверительного интервала; без нескольких seed нет интервала по обучению.']
    else:
        lines.append('Не оценивался (кампания не завершена/не валидна) — запечатанная выборка не открывалась.')
    holdout = data.get('holdout')
    if holdout and holdout.get('metrics'):
        lines += ['', '## Описательный holdout Stage 2A (ранее использовался диагностически)', '']
        for key, metrics in holdout['metrics'].items():
            lines.append(f'- {key}: {_f(metrics[PRIMARY])} %')
    if figure_paths:
        lines += ['', '## Рисунки', ''] + [f'- `{Path(p).relative_to(output)}`' for p in figure_paths]
    lines += ['', '## Ограничения', '',
              '- Один согласованный график seed: различие описывает эту реализацию, а не распределение по seed.',
              '- Без физических event ID: только счётчики на уровне jet; доверительные интервалы по событиям недоступны.',
              '- Validation-метрики (включая первичную) смещены отбором, особенно для B; независимый тест — test3000k.',
              '- Holdout Stage 2A уже просматривался и используется только описательно.',
              '- Прерывание внутри эпохи теряет незавершённую работу этой эпохи (повтор с последнего checkpoint).']
    return '\n'.join(lines) + '\n'
