"""Freeze a predetermined validation-selected inventory before any inference."""
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import torch
import yaml

from experiments.stage2a.common import PREPARED, PRIMARY, environment, file_record, verify_file, write_json
from experiments.stage2a.dataset import FLAVORS, resolve_splits, verify_training_isolation, verify_evaluation_files
from training.pbt.reporting.endpoints import current_best_trajectory, final_window_mean
from training.runtime import PROJECT_DIR, WORKING_POINT_DEFINITION, sha256, utc_now

SELECTIONS = (
    ('fixed50_s12345', 'foundation_fixed_lr_50epochs_20260916'),
    ('windowed50_s12345', 'windowed_pbt_v2'),
    ('windowed100_s12345', 'windowed_pbt_v2_100epochs'),
    ('fixed14_100_s12345', 'fixed_lr_continuation_20260917_lr_14e-6'),
    ('cadence1_s12345', 'cadenced_pbt_v1_50epochs'),
    ('cadence5_s12345', 'cadenced_pbt_v1_cadence5_50epochs'),
    ('cadence1_s22345', 'cadenced_pbt_v1_seed22345_50epochs'),
    ('cadence5_s22345', 'cadenced_pbt_v1_cadence5_seed22345_50epochs'),
)
EXPECTED_IDS = ['pretrained17', *[item[0] for item in SELECTIONS]]
MATCHED_PAIRS = [
    ['fixed50_s12345', 'windowed50_s12345'],
    ['fixed14_100_s12345', 'windowed100_s12345'],
    ['cadence5_s12345', 'cadence1_s12345'],
    ['cadence5_s22345', 'cadence1_s22345'],
    ['windowed50_s12345', 'cadence1_s12345'],
]


def cpu_model(data_config, network):
    raw = yaml.safe_load(Path(data_config).read_text())
    if raw.get('selection') is not None or raw.get('test_time_selection') is not None:
        raise ValueError('Stage2A must retain the existing unfiltered finite evaluation contract')
    if raw['labels']['value'] != ['mc_b', 'mc_c', 'mc_d']:
        raise ValueError('Wrong b/c/d label order')
    inputs = raw['inputs']
    config = SimpleNamespace(input_names=list(inputs), label_value=raw['labels']['value'],
        input_dicts={n:[v[0] if isinstance(v,list) else v for v in c['vars']] for n,c in inputs.items()},
        input_shapes={n:(1,len(c['vars']),c['length']) for n,c in inputs.items()})
    spec = importlib.util.spec_from_file_location('stage2a_frozen_network', network)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_model(config)[0].cpu()


def inventory(*, root=PROJECT_DIR, check_models=True):
    base = root / 'checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut'
    config_path = base / 'data_config.auto.yaml'
    network = root / 'networks/pretrained_sgv_particle_transformer.py'
    reference = json.loads((root/'runs/pbt/foundation_fixed_lr_50epochs_20260916/manifest.json').read_text())
    initial = reference['initial_evaluation']['metrics']
    records = [dict(id='pretrained17', selection='original_pretrained_epoch17', member='initial_resume',
                    generation=-1, weaver_epoch=17, completed_training_epochs=0, training_seed=None,
                    selection_worker_seed=None, pbt_seed=None,
                    effective_seed_base=None, horizon=0, validation_metrics=initial,
                    global_best_validation_metric=None, final10_current_best=None,
                    checkpoint=file_record(base/'net_epoch-17_state.pt'),
                    provenance=[file_record(base/'metadata.json'),file_record(base/'source.txt')])]
    model = cpu_model(config_path, network) if check_models else None
    for ident, run in SELECTIONS:
        path = root/'runs/pbt'/run/'manifest.json'
        manifest = json.loads(path.read_text())
        config = manifest['config']; shared = config['shared']; best = manifest['best']
        if manifest['status'] != 'completed' or config['pbt']['metric'] != PRIMARY or config['pbt']['mode'] != 'min':
            raise ValueError(f'Invalid completed primary-metric source: {run}')
        for key, expected in [('data_config',config_path),('network_config',network)]:
            if sha256(Path(shared[key])) != sha256(expected):
                raise ValueError(f'Evaluation contract differs for {run}: {key}')
        generations = manifest['generations']
        selected_generation = next(g for g in generations if g['index']==best['generation'])
        values = {n:[g['workers'][n]['metrics'][PRIMARY] for g in generations] for n in manifest['members']}
        horizon = len(generations)
        inherited = shared['initial_epoch'] - 17
        historical = []
        if manifest.get('continuation'):
            historical.append(file_record(Path(manifest['continuation']['source_run'])/'manifest.json'))
        if inherited:
            ancestor = root/'runs/pbt/foundation_fixed_lr_50epochs_20260916/manifest.json'
            if manifest.get('continuation'):
                ancestor = Path(manifest['continuation']['source_run'])/'manifest.json'
                inherited = 0  # Windowed history already includes its ancestor.
            if not historical:
                historical.append(file_record(ancestor))
        records.append(dict(id=ident, selection='validation_primary_global_best',
            member=best['member'], generation=best['generation'], weaver_epoch=best['epoch'],
            completed_training_epochs=best['epoch']-17,
            training_seed=shared['seed'], effective_seed_base=12345 if ident.startswith('fixed14_100') else shared['seed'],
            selection_worker_seed=selected_generation.get('seed'),pbt_seed=config['pbt'].get('seed'),
            horizon=horizon+inherited, validation_metrics=best['metrics'],
            global_best_validation_metric=best['metric_value'],
            final10_current_best=final_window_mean(current_best_trajectory(values)),
            checkpoint=file_record(best['state_path']),
            provenance=[file_record(path),file_record(path.parent/'resolved_config.yaml'),*historical],
            source_git=manifest.get('git'), run_fingerprint=manifest.get('fingerprint')))
    for record in records:
        if model is not None:
            state = torch.load(record['checkpoint']['path'], map_location='cpu', weights_only=False)
            if not state or any(not torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v)):
                raise ValueError(f'Unreadable/non-finite model: {record["id"]}')
            model.load_state_dict(state, strict=True)
        record['cpu_strict_load_verified'] = check_models
    return records


def freeze(destination=PREPARED):
    destination = Path(destination)
    if (destination/'protocol.json').exists():
        raise FileExistsError('Protocol already frozen; never replace it after observing holdout')
    manifest_path = PROJECT_DIR/'datasets/manifests/20250711_ilc_nnqq_sgv_10m_3cat_tail_proxy_v1.json'
    manifest = json.loads(manifest_path.read_text())
    data, values = resolve_splits(manifest)
    train_files = [PROJECT_DIR/f'datasets/{dataset}/ml_flavtag_6cat_{flavor}_train800k.parquet'
                   for dataset in ('20250711_ilc_nnqq_sgv_10m_3cat_parquet','20250218_ilc_nnqq_sgvnew_parquet')
                   for flavor in FLAVORS]
    data['training_isolation'] = verify_training_isolation(values, train_files)
    data['physical_event_ids_available'] = False
    data['isolation_limit'] = 'Exact scalar record overlap and source-window isolation checked; no physical event ID links different jets.'
    data['manifest'] = file_record(manifest_path)
    records = inventory()
    sources = [*sorted((PROJECT_DIR/'scripts/experiments').rglob('*.py')),
               PROJECT_DIR/'scripts/experiments/run_stage2a_holdout.sh',
               PROJECT_DIR/'scripts/cluster/check.sh', PROJECT_DIR/'scripts/training/runtime.py',
               PROJECT_DIR/'scripts/validation/results.py', PROJECT_DIR/'scripts/training/pbt/execution/weaver_command.py',
               PROJECT_DIR/'weaver-core/weaver/train.py', PROJECT_DIR/'weaver-core/weaver/utils/dataset.py',
               PROJECT_DIR/'weaver-core/weaver/utils/data_audit.py', PROJECT_DIR/'weaver-core/weaver/utils/nn/metrics.py',
               PROJECT_DIR/'weaver-core/weaver/utils/nn/tools.py',
               PROJECT_DIR/'networks/pretrained_sgv_particle_transformer.py',
               PROJECT_DIR/'checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/data_config.auto.yaml']
    # Include preprocessing/config/file-I/O dependencies, not just entry points.
    sources.extend((PROJECT_DIR/'weaver-core/weaver').rglob('*.py'))
    protocol = dict(schema_version=1, campaign='stage2a_holdout_v1', frozen_at=utc_now(),
        stage1_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT_DIR,text=True).strip(),
        primary_metric=PRIMARY, working_points=[list(p) for p in WORKING_POINT_DEFINITION],
        class_order=['b','c','d'], checkpoints=records, dataset=data, matched_pairs=MATCHED_PAIRS,
        environment=environment(),
        no_holdout_reselection=True, analysis=dict(bootstrap_replicates=200, bootstrap_seed=20261012,
            physical_event_ci='unavailable_without_physical_event_ids', training_seed_inference='descriptive_only'),
        allocation=dict(mechanism='direct_ssh_tmux', policy_source='explicit user confirmation 2026-10-12',
            authorized_nodes=[f'iutgpu0{i}' for i in range(1,8)], preferred_node='iutgpu05',
            max_parallelism=3, min_gpu_memory_mib=40000, max_used_memory_mib=500, max_util_percent=5,
            max_resource_wait_seconds=120, evaluation_timeout_seconds=14400),
        evaluation=dict(data_config=str(PROJECT_DIR/'checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/data_config.auto.yaml'),
            network_config=str(PROJECT_DIR/'networks/pretrained_sgv_particle_transformer.py'),
            dataset=str(PROJECT_DIR/'datasets/20250711_ilc_nnqq_sgv_10m_3cat_parquet'),
            validation_suffix='val_holdout', data_extension='parquet', batch_size=1024,
            num_workers=1, fetch_step=.01, prefetch_factor=2, seed=12345,
            use_amp=True, amp_dtype='fp16', deterministic=True, data_audit=True, save_predictions=True,
            save_prediction_row_ids=True),
        execution_files=[file_record(p) for p in sorted(set(sources))])
    write_json(destination/'protocol.json',protocol)
    (destination/'protocol.sha256').write_text(sha256(destination/'protocol.json')+'\n')
    return protocol


def load_protocol(directory=PREPARED, *, verify_inputs=True):
    directory = Path(directory)
    path = directory/'protocol.json'
    digest = sha256(path)
    if digest != (directory/'protocol.sha256').read_text().strip():
        raise ValueError('Frozen protocol hash changed')
    protocol = json.loads(path.read_text())
    if ([r['id'] for r in protocol['checkpoints']] != EXPECTED_IDS or protocol['primary_metric'] != PRIMARY
            or protocol['working_points'] != [list(p) for p in WORKING_POINT_DEFINITION]
            or protocol['dataset']['rows'] != 2825125 or not protocol['no_holdout_reselection']):
        raise ValueError('Invalid frozen scientific protocol')
    if verify_inputs:
        if protocol['environment'] != environment():
            raise ValueError('Frozen Python/package versions changed; inference is blocked')
        verify_evaluation_files(protocol)
        for record in [*protocol['execution_files'],protocol['dataset']['manifest'],
                       *protocol['dataset']['files'],*protocol['dataset']['training_isolation']]:
            verify_file(record)
        for checkpoint in protocol['checkpoints']:
            verify_file(checkpoint['checkpoint'])
            for source in checkpoint['provenance']:
                verify_file(source)
    if sha256(path) != digest or (directory/'protocol.sha256').read_text().strip() != digest:
        raise ValueError('Frozen protocol changed during preflight')
    return protocol, digest
