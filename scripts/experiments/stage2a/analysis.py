"""Paired comparisons of a frozen inventory; no checkpoint reselection."""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

from experiments.stage2a.common import PRIMARY, file_record, write_json
from experiments.stage2a.execution import prediction_arrays, validate_result
from training.runtime import BKG_REJECTION_EFFICIENCY_POINTS, WORKING_POINT_DEFINITION, _working_point_metrics, mistag_percent_key
from weaver.utils.nn.metrics import BKG_REJECTION_PAIRS, bkg_rejection_at_eff


def score_metrics(labels, scores):
    curves = dict(efficiencies=list(BKG_REJECTION_EFFICIENCY_POINTS),
                  pairs=bkg_rejection_at_eff(labels,scores))
    return _working_point_metrics(curves)


def paired_bootstrap(labels, left, right, *, physical_event_ids=None, replicates=200, seed=20261012):
    """Resample whole physical events with shared draws for both checkpoints.

    Row-position IDs are deliberately NOT accepted as substitute event IDs.
    Confidence intervals condition on these fixed models and this event sample;
    they never measure variability of a training policy across seeds.
    """
    if physical_event_ids is None:
        return dict(status='unavailable',unit='physical_event',interval=None,
                    reason='No physical event ID links jets from one event. Weaver row IDs are positions, not independent physical events.')
    events = np.asarray(physical_event_ids)
    if len(events) != len(labels):
        raise ValueError('Physical event IDs are not aligned')
    unique,inverse = np.unique(events,return_inverse=True)
    if len(unique)<10 or replicates<20:
        return dict(status='unavailable',interval=None,reason='Too few physical events/bootstrap draws')
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(replicates):
        counts = np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))
        rows = np.repeat(np.arange(len(labels)),counts[inverse])
        if set(np.unique(labels[rows])) != {0,1,2}:
            continue
        a,b = score_metrics(labels[rows],left[rows]),score_metrics(labels[rows],right[rows])
        keys = [PRIMARY,*[mistag_percent_key(pair,eff) for pair,eff in WORKING_POINT_DEFINITION]]
        values = [b[k]-a[k] for k in keys]
        if np.isfinite(values).all():
            differences.append(values)
    if len(differences)<.95*replicates:
        return dict(status='unavailable',interval=None,reason='Too many non-finite/missing-class bootstrap replicates')
    intervals = np.quantile(differences,[.025,.975],axis=0)
    return dict(status='available',unit='physical_event',method='paired physical-event cluster percentile bootstrap',
                n_events=len(unique),replicates=replicates,valid_replicates=len(differences),seed=seed,
                confidence_level=.95,intervals={k:[float(intervals[0,i]),float(intervals[1,i])] for i,k in enumerate(keys)})


def passing_background(labels, scores, pair, eff):
    tag,bkg = BKG_REJECTION_PAIRS[pair]
    selected = (labels==tag)|(labels==bkg)
    ratio = scores[selected,tag]/np.maximum(scores[selected,tag]+scores[selected,bkg],1e-6)
    truth = labels[selected]==tag
    _,tpr,threshold = roc_curve(truth,ratio)
    index = np.flatnonzero(tpr>=np.float32(eff))[0]
    return ratio[~truth]>=threshold[index]


def paired_comparison(labels, left, right, left_metrics, right_metrics, analysis):
    working_points = {}
    for pair,eff in WORKING_POINT_DEFINITION:
        key = mistag_percent_key(pair,eff)
        a,b = passing_background(labels,left,pair,eff),passing_background(labels,right,pair,eff)
        working_points[key] = dict(delta_pp=right_metrics[key]-left_metrics[key],n_paired_background_rows=len(a),
            both_passed=int(np.count_nonzero(a&b)),left_only=int(np.count_nonzero(a&~b)),
            right_only=int(np.count_nonzero(~a&b)),neither_passed=int(np.count_nonzero(~a&~b)))
    delta = right_metrics[PRIMARY]-left_metrics[PRIMARY]
    return dict(delta_pp=delta,reduction_percent=-100*delta/left_metrics[PRIMARY],
                n_paired_rows=len(labels),working_points=working_points,
                confidence=paired_bootstrap(labels,left,right,replicates=analysis['bootstrap_replicates'],seed=analysis['bootstrap_seed']))


def summarize(protocol, results, *, arrays=None, protocol_sha=None):
    rows = []; loaded = {}
    for checkpoint in protocol['checkpoints']:
        ident = checkpoint['id']; record = results[ident]
        if arrays is None:
            validate_result(record,checkpoint,protocol,protocol_sha)
            ids,labels,scores = prediction_arrays(record,protocol)
        else:
            ids,labels,scores = arrays[ident]
        recomputed = score_metrics(labels,scores)
        for key in [PRIMARY,*[mistag_percent_key(pair,eff) for pair,eff in WORKING_POINT_DEFINITION]]:
            if not np.isclose(recomputed[key],record['metrics'][key],rtol=1e-10,atol=1e-12):
                raise ValueError(f'Prediction metrics disagree with evaluation log: {ident}/{key}')
        # Pair by actual audited IDs, independent of file-chunk delivery order.
        order = np.argsort(ids)
        ids,labels,scores = ids[order],labels[order],scores[order]
        if loaded:
            old_ids,old_labels,_ = next(iter(loaded.values()))
            if not np.array_equal(ids,old_ids) or not np.array_equal(labels,old_labels):
                raise ValueError('Checkpoint predictions do not describe the same ordered holdout records')
        loaded[ident] = ids,labels,scores
        validation = checkpoint['validation_metrics'][PRIMARY]
        holdout = record['metrics'][PRIMARY]
        rows.append(dict(checkpoint_id=ident,selection=checkpoint['selection'],
            training_seed=checkpoint['training_seed'],horizon=checkpoint['horizon'],
            selection_generation=checkpoint['generation'],weaver_epoch=checkpoint['weaver_epoch'],
            checkpoint_sha256=checkpoint['checkpoint']['sha256'],
            original_validation_primary=validation,global_best_validation_primary=checkpoint['global_best_validation_metric'],
            final10_current_best=checkpoint['final10_current_best'],independent_holdout_primary=holdout,
            holdout_minus_validation_pp=holdout-validation,
            holdout_relative_difference_percent=100*(holdout-validation)/validation,
            working_points={mistag_percent_key(p,e):record['metrics'][mistag_percent_key(p,e)] for p,e in WORKING_POINT_DEFINITION}))
    comparisons = []
    for left,right in protocol['matched_pairs']:
        comparisons.append(dict(left=left,right=right,**paired_comparison(loaded[left][1],loaded[left][2],loaded[right][2],
                               results[left]['metrics'],results[right]['metrics'],protocol['analysis'])))
    from scipy.stats import rankdata
    validation_ranks = rankdata([r['original_validation_primary'] for r in rows])
    holdout_ranks = rankdata([r['independent_holdout_primary'] for r in rows])
    correlation = (float(np.corrcoef(validation_ranks,holdout_ranks)[0,1])
                   if np.std(validation_ranks)>0 and np.std(holdout_ranks)>0 else None)
    return dict(primary_metric=PRIMARY,protocol_sha256=protocol_sha,rows=rows,matched_comparisons=comparisons,
                validation_order=[r['checkpoint_id'] for r in sorted(rows,key=lambda r:r['original_validation_primary'])],
                diagnostic_holdout_order=[r['checkpoint_id'] for r in sorted(rows,key=lambda r:r['independent_holdout_primary'])],
                spearman_rank_correlation=correlation,no_holdout_reselection=True,
                inference_scope='Fixed selected checkpoints on identical holdout rows; no policy significance inference from two training seeds.')


def report_markdown(data, protocol):
    lines = ['# Stage 2A: независимая holdout-оценка','',
        'Checkpoints зафиксированы по исходной validation до запуска. Holdout не используется для повторного выбора модели.',
        '',f'Protocol SHA256: `{data["protocol_sha256"]}`. Holdout: {protocol["dataset"]["rows"]:,} строк, классы b/c/d.',
        '', '## Сравнение выбранных checkpoints','',
        '| Checkpoint | Validation выбранного checkpoint, % | Global-best validation, % | Final-10 current-best, % | Holdout, % | Holdout − validation, п.п. | Относительная разница, % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    fmt = lambda value:'—' if value is None else f'{value:.6f}'
    for row in data['rows']:
        lines.append('| '+row['checkpoint_id']+' | '+' | '.join(fmt(row[k]) for k in [
            'original_validation_primary','global_best_validation_primary','final10_current_best',
            'independent_holdout_primary','holdout_minus_validation_pp','holdout_relative_difference_percent'])+' |')
    lines += ['', 'Final-10 — среднее минимума популяции отдельно в каждой из последних десяти training epochs. Это не метрика одного выбранного checkpoint и не holdout.',
              '', '## Восемь physics working points','',
              '| Checkpoint | bc@80 | bd@80 | bc@90 | bd@90 | cb@50 | cd@50 | cb@80 | cd@80 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in data['rows']:
        lines.append('| '+row['checkpoint_id']+' | '+' | '.join(fmt(row['working_points'][mistag_percent_key(p,e)])
                                                               for p,e in WORKING_POINT_DEFINITION)+' |')
    lines += ['', '## Matched paired comparisons','',
              '| Baseline | Сравниваемый checkpoint | Δ mistag, п.п. | Снижение, % | Event-level CI |',
              '|---|---|---:|---:|---|']
    for pair in data['matched_comparisons']:
        ci = pair['confidence']
        lines.append(f'| {pair["left"]} | {pair["right"]} | {pair["delta_pp"]:.6f} | {pair["reduction_percent"]:.3f} | {ci["status"]} |')
    lines += ['', 'Отрицательная Δ означает меньший mistag у правого checkpoint. Paired counts совпадений/расхождений для каждого working point доступны в analysis.json.',
        '', '## Ranking и ограничения','',
        f'Spearman validation/holdout ranking: {fmt(data["spearman_rank_correlation"])}. Ranking является диагностикой, не процедурой выбора.',
        '', 'Физический event-level confidence interval недоступен: в исходных данных отсутствует event ID, связывающий jets одного события. Row IDs удостоверяют порядок записей, но не независимость физических событий.',
        'Bootstrap поддерживает общие физические event clusters, когда такие идентификаторы достоверно доступны; в этой кампании они не подменяются номерами строк.',
        'Два training seeds не дают основания утверждать статистическую значимость преимущества политики. Checkpoints разных горизонтов сравниваются описательно; matched comparisons выделены отдельно.',
        'Данные проходили заранее заданные cuts и reweighting при обучении. Evaluation — полный конечный проход без reweighting. Изоляция подтверждена source windows и отсутствием совпадающих scalar records с обоими training inputs; event-level lineage остаётся ограничением.',
        '', '## Артефакты','', '`comparison.csv`, `analysis.json`, `figures/holdout_comparison.png`, `figures/paired_differences.png`; логи, predictions и hashes — в `evaluations/<checkpoint>/`.']
    return '\n'.join(lines)+'\n'


def build_report(protocol, results, output, protocol_sha, *, arrays=None):
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    data = summarize(protocol,results,arrays=arrays,protocol_sha=protocol_sha)
    write_json(output/'analysis.json',data)
    csv_rows = [{**{k:v for k,v in r.items() if k!='working_points'},**r['working_points']} for r in data['rows']]
    with (output/'comparison.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = output/'figures'; figures.mkdir(exist_ok=True)
    x = np.arange(len(data['rows']))
    fig,ax = plt.subplots(figsize=(14,6),layout='constrained')
    ax.bar(x-.18,[r['original_validation_primary'] for r in data['rows']],width=.36,label='Original validation checkpoint')
    ax.bar(x+.18,[r['independent_holdout_primary'] for r in data['rows']],width=.36,label='Independent holdout')
    ax.set_xticks(x,[r['checkpoint_id'] for r in data['rows']],rotation=35,ha='right')
    ax.set_ylabel('Composite mistag (%)');ax.legend();ax.set_title('Frozen validation-selected checkpoints')
    fig.savefig(figures/'holdout_comparison.png',dpi=180);plt.close(fig)
    fig,ax = plt.subplots(figsize=(11,5),layout='constrained')
    pairs = data['matched_comparisons']
    ax.scatter([p['delta_pp'] for p in pairs],range(len(pairs)))
    ax.axvline(0,color='grey',lw=1)
    ax.set_yticks(range(len(pairs)),[p['right']+' vs '+p['left'] for p in pairs])
    ax.set_xlabel('Right − left composite mistag (percentage points)')
    ax.set_title('Paired point differences; physical-event CIs unavailable')
    fig.savefig(figures/'paired_differences.png',dpi=180);plt.close(fig)
    report = output/'report.md'
    temporary = report.with_suffix('.md.tmp');temporary.write_text(report_markdown(data,protocol));temporary.replace(report)
    artifacts = [file_record(p) for p in [report,output/'analysis.json',output/'comparison.csv',*sorted(figures.glob('*.png'))]]
    write_json(output/'report_integrity.json',dict(protocol_sha256=protocol_sha,artifacts=artifacts))
    return dict(path=str(report),artifacts=artifacts)
