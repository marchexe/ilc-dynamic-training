"""Sealed independent test (test3000k) and the descriptive Stage 2A holdout tier.

The test split is converted by the verified Stage 2B converter and evaluated
only after both arms are complete and the checkpoint selection is frozen from
validation evidence. Evaluation reuses the Stage 2A inference campaign
(leases, full-pass audit, row-ID coverage, finite eight working points) via a
Stage-2A-shaped evaluation contract; nothing here can select checkpoints.
"""
from pathlib import Path

import numpy as np

from experiments.stage2a.dataset import byte_rows, scalar_rows
from experiments.stage2b import dataset as data
from experiments.stage2b.provenance import root_scalars
from experiments.stage3.common import PRIMARY, PROJECT_DIR, digest_json, load_json, sha256, utc_now

PRETRAINING = PROJECT_DIR / 'datasets/20250218_ilc_nnqq_sgvnew_parquet'
VALIDATION = PROJECT_DIR / 'datasets/20250711_ilc_nnqq_sgv_10m_3cat_parquet'


def verify_test_isolation(test_spec, train_spec, *, root_reader=root_scalars, parquet_reader=scalar_rows):
    """Read-only: labels and zero scalar overlap with every training/selection input."""
    flavors = []
    for flavor in data.FLAVORS:
        source = next(s for s in test_spec['sources'] if s['flavor'] == flavor)
        test = root_reader(data.source_path(source))
        label = test_spec['flavors'][flavor]['label']
        labels = sorted(np.unique(test[:, 0]).astype(int).tolist())
        if len(test) != source['entries'] or labels != [label]:
            raise ValueError(f'{flavor}: test3000k rows/labels disagree with the specification')
        unique = np.unique(byte_rows(test))
        references = {
            'train8m': root_reader(data.source_path(next(s for s in train_spec['sources'] if s['flavor'] == flavor))),
            'val1000k': parquet_reader(VALIDATION / f'ml_flavtag_6cat_{flavor}_val1000k.parquet'),
            'pretraining_20250218_train800k': parquet_reader(PRETRAINING / f'ml_flavtag_6cat_{flavor}_train800k.parquet'),
            'pretraining_20250218_val50k': parquet_reader(PRETRAINING / f'ml_flavtag_6cat_{flavor}_val50k.parquet')}
        overlap = {name: int(len(np.intersect1d(unique, byte_rows(values)))) for name, values in references.items()}
        if any(overlap.values()):
            raise ValueError(f'{flavor}: test3000k shares scalar records with {overlap}')
        flavors.append(dict(flavor=flavor, label=label, rows=len(test), unique_scalar_records=int(len(unique)),
                            scalar_overlap=overlap, reference_rows={k: len(v) for k, v in references.items()}))
    return dict(schema_version=1, verified_at=utc_now(), spec_name=test_spec['name'], flavors=flavors,
                rows_total=sum(f['rows'] for f in flavors), entry_window=test_spec['source_selection']['entry_window'],
                physical_event_ids_available=False,
                project_usage='No configuration, script, manifest or run of this repository references test3000k '
                              'before Stage 3 (checked by search). Use by other researchers cannot be verified.')


def isolation_markdown(evidence):
    lines = ['# Sealed test split (test3000k) provenance', '',
             f'Verified {evidence["verified_at"]}; read-only. Window `{evidence["entry_window"]}` of each flavour file '
             '(train8m: Entry$ < 8M; val1000k and the Stage 2A holdout: 16M-17M).', '',
             '| Flavour | Label | Rows | Unique scalar records | train8m | val1000k | pretraining train800k | pretraining val50k |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for f in evidence['flavors']:
        o = f['scalar_overlap']
        lines.append(f'| {f["flavor"]} | {f["label"]} | {f["rows"]:,} | {f["unique_scalar_records"]:,} | {o["train8m"]} | '
                     f'{o["val1000k"]} | {o["pretraining_20250218_train800k"]} | {o["pretraining_20250218_val50k"]} |')
    lines += ['', f'Total {evidence["rows_total"]:,} rows; zero identical scalar records with any training, selection '
              'or pretraining input. ' + evidence['project_usage'],
              'No physical event ID exists, so event-level independence is not provable.']
    return '\n'.join(lines) + '\n'


def evaluation_contract(dataset_dir, suffix, *, arm_shared, seed, name):
    """Stage-2A-shaped contract: same evaluation settings as the arms' validation."""
    from weaver.utils.data_audit import parquet_manifest
    paths = []
    for flavor in data.FLAVORS:
        matches = sorted(Path(dataset_dir).resolve().glob(f'*_{flavor}_{suffix}.parquet'))
        if len(matches) != 1:
            raise ValueError(f'{suffix}: expected exactly one {flavor} file, found {matches}')
        paths.append(str(matches[0]))
    manifest = parquet_manifest(sorted(paths))
    evaluation = {k: arm_shared[k] for k in ('amp_dtype', 'batch_size', 'data_audit', 'data_config', 'data_extension',
                                              'deterministic', 'fetch_step', 'network_config', 'num_workers',
                                              'prefetch_factor', 'use_amp')}
    evaluation.update(dataset=str(Path(dataset_dir).resolve()), validation_suffix=suffix, seed=seed,
                      save_predictions=True, save_prediction_row_ids=True)
    return dict(name=name, evaluation=evaluation,
                dataset=dict(rows=manifest['total_rows'], audit_fingerprint=manifest['fingerprint'],
                             holdout_files=[dict(path=r['path'], rows=r['rows'], sha256=r['sha256']) for r in manifest['files']]))


def sealed_protocol(contract, checkpoints, allocation, policy_budgets):
    """Inputs for experiments.stage2a.execution.run_campaign; its digest binds results."""
    protocol = dict(schema_version=1, purpose=contract['name'], evaluation=contract['evaluation'],
                    dataset=contract['dataset'], checkpoints=checkpoints,
                    allocation=dict(allocation, evaluation_timeout_seconds=int(policy_budgets['test_eval_timeout_hours'] * 3600)))
    return protocol, digest_json(protocol)


def paired(results, protocol, pairs, analysis=None):
    """B-minus-A comparisons on identical rows (jet level; no event-level CI)."""
    from experiments.stage2a.analysis import paired_comparison, score_metrics
    from experiments.stage2a.execution import prediction_arrays
    analysis = analysis or dict(bootstrap_replicates=200, bootstrap_seed=20261012)
    arrays, metrics = {}, {}
    for key, record in results.items():
        ids, labels, scores = prediction_arrays(record, protocol)
        order = np.argsort(ids)
        arrays[key] = (labels[order], scores[order])
        metrics[key] = score_metrics(*arrays[key])
        if abs(metrics[key][PRIMARY] - record['metrics'][PRIMARY]) > 1e-9:
            raise ValueError(f'{key}: recomputed primary metric disagrees with the evaluation log')
    comparisons = {}
    for name, (left, right) in pairs.items():
        labels = arrays[left][0]
        if not np.array_equal(labels, arrays[right][0]):
            raise ValueError('Paired checkpoints were evaluated on different rows')
        comparisons[name] = dict(left=left, right=right, **paired_comparison(
            labels, arrays[left][1], arrays[right][1], metrics[left], metrics[right], analysis))
    return metrics, comparisons


def selection_freeze(arms, *, mode='min'):
    """Preregistered, validation-only checkpoint choice; written before any test inference."""
    from training.checkpoints import selected_best_bundle
    from training.pbt.state.checkpointing import checkpoint_paths
    checkpoints = []
    for arm, manifest in arms.items():
        run = Path(manifest['run_dir'])
        best = manifest['best']
        selected_best_bundle(best)
        checkpoints.append(dict(id=f'{arm}__selected_best', arm=arm, role='validation_global_best',
                                checkpoint=dict(path=best['state_path'], sha256=best['checkpoint']['state']['sha256'],
                                                size_bytes=Path(best['state_path']).stat().st_size),
                                validation_metric=best['metric_value'], member=best['member'], epoch=best['epoch']))
        final = manifest['generations'][-1]
        values = {name: worker['metrics'][PRIMARY] for name, worker in final['workers'].items()}
        current_best = (min if mode == 'min' else max)(sorted(values), key=values.get)
        for name in sorted(final['workers']):
            path = checkpoint_paths(run / name, final['epoch'])[0]
            checkpoints.append(dict(id=f'{arm}__final__{name}', arm=arm, role='final_member', member=name,
                                    final_current_best=name == current_best, epoch=final['epoch'],
                                    validation_metric=values[name],
                                    checkpoint=dict(path=str(path), sha256=sha256(path), size_bytes=path.stat().st_size)))
    return checkpoints


def preregistered_pairs(checkpoints):
    """Exactly the comparisons fixed in the protocol (arm B minus arm A)."""
    by = {c['id']: c for c in checkpoints}
    pairs = {'selected_best': ('fixed_lr__selected_best', 'windowed_pbt_v2__selected_best')}
    current = {c['arm']: c['id'] for c in checkpoints if c.get('final_current_best')}
    pairs['final_current_best_member'] = (current['fixed_lr'], current['windowed_pbt_v2'])
    for name in sorted({c['member'] for c in checkpoints if c['role'] == 'final_member'}):
        left, right = f'fixed_lr__final__{name}', f'windowed_pbt_v2__final__{name}'
        if left in by and right in by:
            pairs[f'final_member_{name}'] = (left, right)
    return pairs


def load_holdout_contract():
    """Stage 2A frozen holdout contract (descriptive tier, never used for choices)."""
    protocol = load_json(PROJECT_DIR / 'configs/prepared/stage2a_holdout_v1/protocol.json')
    return dict(name='stage2a_holdout_descriptive', evaluation=protocol['evaluation'],
                dataset={k: protocol['dataset'][k] for k in ('rows', 'audit_fingerprint', 'holdout_files')})


def verify_converted_matches_root(spec, *, root_reader=root_scalars, parquet_reader=scalar_rows):
    """After conversion: Parquet identity columns equal the ROOT source, row for row."""
    checks = []
    for source in spec['sources']:
        converted = parquet_reader(data.output_dir(spec) / data.output_name(source['flavor'], spec['suffix']))
        original = root_reader(data.source_path(source))
        if not np.array_equal(converted, original):
            raise ValueError(f'{source["flavor"]}: converted {spec["suffix"]} differs from its ROOT source')
        checks.append(dict(flavor=source['flavor'], rows=len(converted), identical_to_root=True))
    return dict(verified_at=utc_now(), checks=checks)
