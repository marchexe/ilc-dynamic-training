"""Frozen Stage 3 protocol: preregistration, identities and horizon evidence.

`build()` assembles the protocol from the committed configs and inputs;
`--freeze` writes it once. Every launch/resume recomputes the identity block
and refuses to start on any drift (configs, data, checkpoints, policy, frozen
policy and execution code).
"""
import copy
import math
from pathlib import Path

import yaml

from experiments.stage2b import dataset as data
from experiments.stage3.common import (ARM_CONFIGS, GATE_CONFIG, POLICY, PRIMARY, PROJECT_DIR, PROTOCOL, PROTOCOL_V1,
                                       TEST_SPEC, digest_json, load_json, load_policy, relative, resolve, sha256,
                                       utc_now, write_json)
from training.pbt.config import contract_fingerprint

HORIZON = 50
WINDOW_MARGIN_PP = 0.002
EXECUTION_FILES = (
    'scripts/training/run_pbt.py', 'scripts/training/pbt/runner.py', 'scripts/training/pbt/config.py',
    'scripts/training/pbt/models/config.py', 'scripts/training/pbt/execution/backend.py',
    'scripts/training/pbt/execution/weaver_command.py', 'scripts/training/weaver.py', 'scripts/training/runtime.py',
    'scripts/training/checkpoints.py', 'scripts/training/pbt/state/checkpointing.py',
    'scripts/training/pbt/state/optimizer_state.py', 'scripts/training/pbt/metrics.py',
    'scripts/training/pbt/reference/windowed_v2.py', 'scripts/training/pbt/planning/windowed_pbt_v2.py',
    'scripts/training/pbt/planning/fixed_lr_grid.py', 'scripts/training/pbt/planning/dispatch.py',
    'scripts/training/pbt/reporting/endpoints.py', 'scripts/validation/results.py', 'scripts/validation/verify_fixed_lr.py',
    'networks/pretrained_sgv_particle_transformer.py', 'weaver-core/weaver/train.py',
    'weaver-core/weaver/utils/dataset.py', 'weaver-core/weaver/utils/data/fileio.py',
    'weaver-core/weaver/utils/data/config.py', 'weaver-core/weaver/utils/data_audit.py',
    'weaver-core/weaver/utils/nn/tools.py', 'weaver-core/weaver/utils/nn/optimizer/ranger.py',
    'weaver-core/weaver/utils/nn/optimizer/radam.py', 'weaver-core/weaver/utils/nn/optimizer/lookahead.py')


def _synthetic_generation(config, members, index, values):
    from training.pbt.reference.windowed_v2 import WORKING_POINTS
    keys = [config['pbt']['metric'], 'validation_loss', *WORKING_POINTS]
    return dict(index=index, epoch=config['shared']['initial_epoch'] + index + 1, status='running', exploit=None,
                workers={name: dict(lr=member['lr'], status='completed', returncode=0,
                                    metrics={key: values[name] for key in keys}) for name, member in members.items()})


def horizon_evidence(config=None, epochs=40, outcomes=None, continuation_horizon=30):
    """Replay the frozen planner on identical outcomes: native 50 vs 30->50 continuation.

    The continuation path follows state/continuation.py: the epoch-30 boundary
    is decided under the 30-epoch horizon (terminal, no copies) and is replayed
    as recorded; later boundaries use the extended horizon.
    """
    from training.pbt.reference.windowed_v2 import windowed_pbt_v2_plan
    config = copy.deepcopy(config or resolve(ARM_CONFIGS['windowed_pbt_v2']))
    names = [m['name'] for m in config['population']]
    # Fixed synthetic outcomes: the lowest-LR member persistently loses by more than the margin.
    outcomes = outcomes or dict(zip(names, (0.50, 0.42, 0.41, 0.405, 0.40)))

    def run(horizons):
        members = {m['name']: dict(name=m['name'], lr=m['start_lr']) for m in config['population']}
        history, decisions = [], []
        for index in range(epochs):
            local = copy.deepcopy(config)
            local['shared']['generations'] = horizons(index)
            generation = _synthetic_generation(local, members, index, outcomes)
            history.append(generation)
            _, plan = windowed_pbt_v2_plan(local, generation, members, dict(generations=history))
            generation['exploit'] = plan
            for event in plan:
                members[event['recipient']]['lr'] = event['new_lr']
            if 'windowed_pbt_v2' in generation:
                decision = generation['windowed_pbt_v2']
                decisions.append(dict(epoch=index + 1, terminal=decision['terminal'],
                                      copies=[(e['recipient'], e['new_lr']) for e in plan],
                                      loss_counters={n: m['next_loss_count'] for n, m in decision['members'].items()}))
        return decisions, {n: m['lr'] for n, m in members.items()}

    native, native_lr = run(lambda index: HORIZON)
    continued, continued_lr = run(lambda index: continuation_horizon if index < continuation_horizon else HORIZON)
    differing = [n['epoch'] for n, c in zip(native, continued) if n['copies'] != c['copies'] or n['loss_counters'] != c['loss_counters']]
    return dict(equivalent=not differing and native_lr == continued_lr, first_divergence_epoch=differing[0] if differing else None,
                differing_boundaries=differing, native_50=native, continuation_30_to_50=continued,
                live_lr_after=dict(native_50=native_lr, continuation_30_to_50=continued_lr),
                outcomes=outcomes, epochs_simulated=epochs)


def expected_accepted_rows(train_spec, data_config):
    """Binomial expectation of the existing flat reweighting (no_remake_weights)."""
    hists = yaml.safe_load(Path(data_config).read_text())['weights']['reweight_hists']
    classes = dict(bb='mc_b', cc='mc_c', dd='mc_d')
    mean = variance = 0.0
    for source in train_spec['sources']:
        p = float(hists[classes[source['flavor']]][0][0])
        mean += source['entries'] * p
        variance += source['entries'] * p * (1 - p)
    return mean, math.sqrt(variance)


def validation_identity(shared):
    from weaver.utils.data_audit import parquet_manifest
    paths = sorted(str(p) for f in data.FLAVORS
                   for p in Path(shared['validation_dataset']).glob(f'*_{f}_{shared["validation_suffix"]}.parquet'))
    manifest = parquet_manifest(paths)
    return dict(fingerprint=manifest['fingerprint'], total_rows=manifest['total_rows'])


def identity(*, policy_path=POLICY):
    """Everything a launch must find unchanged."""
    configs = {arm: resolve(path) for arm, path in ARM_CONFIGS.items()}
    gate = resolve(GATE_CONFIG)
    train_spec, train_spec_sha = data.load_spec()
    test_spec = data.validate_spec(load_json(TEST_SPEC))
    realized = PROJECT_DIR / 'datasets/manifests' / f'{train_spec["name"]}.json'
    shared = configs['fixed_lr']['shared']
    _, policy_sha = load_policy(policy_path)
    return dict(
        configs={arm: dict(path=relative(ARM_CONFIGS[arm]), experiment=config['experiment_name'],
                           output_root=relative(config['output_root']), fingerprint=contract_fingerprint(config))
                 for arm, config in configs.items()},
        gate=dict(path=relative(GATE_CONFIG), experiment=gate['experiment_name'],
                  output_root=relative(gate['output_root']), fingerprint=contract_fingerprint(gate)),
        train_spec_sha256=train_spec_sha, train_manifest=dict(path=str(realized.relative_to(PROJECT_DIR)), sha256=sha256(realized)),
        test_spec_sha256=digest_json(test_spec), policy_sha256=policy_sha,
        initial_checkpoint={part: dict(path=relative(shared[f'initial_{part}']), sha256=sha256(Path(shared[f'initial_{part}'])))
                            for part in ('state', 'optimizer')},
        data_config_sha256=sha256(Path(shared['data_config'])), network_config_sha256=sha256(Path(shared['network_config'])),
        execution_files={path: sha256(PROJECT_DIR / path) for path in EXECUTION_FILES})


def build():
    from training.pbt.reference.windowed_v2 import REFERENCE_REVISION
    configs = {arm: resolve(path) for arm, path in ARM_CONFIGS.items()}
    train_spec, _ = data.load_spec()
    realized = load_json(PROJECT_DIR / 'datasets/manifests' / f'{train_spec["name"]}.json')
    test_spec = data.validate_spec(load_json(TEST_SPEC))
    shared = configs['fixed_lr']['shared']
    accepted, sigma = expected_accepted_rows(train_spec, shared['data_config'])
    horizon = horizon_evidence(configs['windowed_pbt_v2'])
    if horizon['equivalent']:
        raise ValueError('Horizon evidence changed: review the horizon decision before freezing')
    return dict(
        schema_version=1, name='stage3_train8m_fixed_vs_windowed_v1', frozen_at=utc_now(),
        references=dict(stage1_commit='efa003bde5d6fa8f1dfa83c11572c4fd0942804f',
                        stage2a_commit='953a170e2281faca37ee976b02ec2d0a1a208804',
                        stage2b_commits=['9edb46fdf8c7b7b457913855acfcddd2f91690eb', '7882df065702fe5728b0198a34706293375006b5'],
                        windowed_reference_revision=REFERENCE_REVISION),
        design=dict(
            arms=dict(fixed_lr='A: fixed_lr_grid (five fixed LRs, never copies)',
                      windowed_pbt_v2='B: frozen windowed_pbt_v2 (5-epoch windows, 3-epoch score, margin 0.002, '
                                      '2 losing windows, <=2 recipients, x0.8/x1.2)'),
            population=[dict(name=m['name'], start_lr=m['start_lr']) for m in configs['fixed_lr']['population']],
            horizon_epochs=HORIZON, seed_base=shared['seed'], seed_rule='Weaver seed = seed_base + generation (identical across arms)',
            initial_epoch=shared['initial_epoch'], batch_size=shared['batch_size'],
            data=dict(train=dict(name=train_spec['name'], rows=realized['rows_total'], size_bytes=realized['size_bytes_total'],
                                 weaver_fingerprint=realized['weaver_audit']['fingerprint']),
                      selection_validation=dict(suffix=shared['validation_suffix'], **validation_identity(shared)),
                      sealed_test=dict(name=test_spec['name'], suffix=test_spec['suffix'], rows=test_spec['expected_rows_total']),
                      descriptive_holdout='Stage 2A val_holdout (previously inspected; descriptive only)'),
            only_intended_difference='adaptive selection/copy/mutation of windowed_pbt_v2',
            seeds='one matched training seed schedule; no seed replication in this campaign'),
        horizon=dict(
            decision='fixed_50_epochs',
            rationale=('A 30-epoch run marks its epoch-30 window boundary terminal (copies suppressed, loss counters '
                       'not reset); state/continuation.py replays that terminal decision as recorded. A run '
                       'configured for 50 epochs treats epoch 30 as a normal boundary. Conditional on identical '
                       'outcomes the policy trajectories therefore diverge from epoch 30 on, so a 30->50 '
                       'continuation is not the intended 50-epoch treatment. Fixed-LR arm continuation would be '
                       'equivalent, but both arms must share one horizon contract.'),
            evidence=horizon,
            interim=('Optional descriptive interim report (--interim-report) at any completed epoch, e.g. 30; '
                     'it never stops, extends or changes either arm and is not an endpoint.'),
            resume='Resuming an interrupted run keeps the configured 50-epoch horizon and is equivalent.'),
        endpoints=dict(
            primary=dict(name='final-10 current-best population mean', metric=PRIMARY, split=shared['validation_suffix'],
                         epochs='full epochs 41-50', implementation='training.pbt.reporting.endpoints.current_best_trajectory + final_window_mean (Stage 1)',
                         comparison='delta = B - A in percentage points and relative = delta / A; lower is better',
                         interpretation=f'|delta| < {WINDOW_MARGIN_PP} pp (the frozen windowed decision margin on the same split) is reported as within the selection-noise margin',
                         selection_bias='The same split drives windowed copy decisions and all validation selection; B is adapted to it.'),
            independent=dict(split='test3000k (sealed until both arms complete and selection is frozen)',
                             checkpoints='per arm: validation global-best bundle and all five final members',
                             comparisons=['selected_best: B - A', 'final_current_best_member: B - A',
                                          'final_member_<name>: B - A (paired by original member identity)',
                                          'final population mean: B - A'],
                             metrics=[PRIMARY, 'eight reference working points'], unit='jet rows; paired background pass counts',
                             confidence='no physical event IDs: no event-level interval; no training-seed interval'),
            descriptive_holdout=dict(split='Stage 2A val_holdout', checkpoints='per arm validation global-best only',
                                     use='continuity with Stage 2A; never for selection or tuning'),
            secondary=['global-best validation value and epoch', 'final-member validation metrics and population mean',
                       'training/validation losses and AMP skipped steps', 'exploit/explore events and LR trajectories',
                       'population diversity (metric spread, LR spread, final parameter distances)',
                       'optimizer steps and accepted samples per member and arm', 'GPU-hours and wall time',
                       'checkpoint/resume reliability and determinism checks']),
        validity=dict(
            required=['both arms complete 50 generations', 'scripts/validation/verify_fixed_lr.py passes for both arms',
                      'identical accepted training rows per generation across all ten workers (compare())',
                      'identical initial evaluation and validation traversal', 'complete finite final evaluations',
                      'protocol identity unchanged at every launch'],
            determinism=['same-named members bit-identical across arms until that member\'s first windowed copy '
                         '(and its recorded pre-copy state)', 'gate epoch-18 state equals both arms lr_14e-6 epoch-18 state'],
            determinism_failure=('reported prominently as "determinism not confirmed"; it does not invalidate the '
                                 'comparison but qualifies attributing differences to the policy'),
            invalid_or_incomplete='reported as INVALID/INCOMPLETE; no comparison is declared successful and the sealed '
                                  'test is not opened'),
        gate=dict(member='lr_14e-6', rationale='highest LR: most likely to expose AMP/non-finite behaviour',
                  full_epoch=True, reuse='outputs never reused by the arms; determinism reference only',
                  expected_accepted_rows=accepted, accepted_rows_tolerance=int(math.ceil(6 * sigma)),
                  scanned_rows=realized['rows_total'], max_amp_skipped_fraction=0.01,
                  cuda_max_allocated_mb=dict(reference=21539.7, relative_tolerance=0.10),
                  max_tree_rss_bytes=34359738368, training_rows_per_second=[3000, 20000],
                  max_startup_seconds=900, max_run_disk_bytes=2147483648,
                  validation_rows=validation_identity(shared)['total_rows']),
        identity=identity())


SCIENTIFIC = ('schema_version', 'name', 'frozen_at', 'references', 'design', 'horizon', 'endpoints', 'validity', 'gate')
REVISION_REASON = ('Administrative revision before any Stage 3 training: cluster node names moved from the committed '
                   'policy to the untracked local site configuration, identity paths recorded relative to the '
                   'checkout, sealed-test source paths and the train8m manifest made portable, and a node name '
                   'removed from the runner --slots help text. Scientific sections are copied verbatim from v1.')


def freeze(path=PROTOCOL):
    protocol = build()
    _write_frozen(path, protocol)
    return protocol


def _write_frozen(path, protocol):
    write_json(path, protocol)
    Path(str(path).replace('.json', '.sha256')).write_text(sha256(path) + '\n')


def freeze_revision(original=PROTOCOL_V1, path=PROTOCOL, reason=REVISION_REASON):
    """Administrative revision: v1 scientific sections verbatim, identity recomputed."""
    if Path(path).exists():
        raise ValueError('The administrative revision is already frozen')
    v1, v1_sha = _load_file(original)
    revision = {key: v1[key] for key in SCIENTIFIC}
    revision.update(administrative_revision=dict(version=2, revised_at=utc_now(), reason=reason,
                                                 supersedes=dict(file=Path(original).name, sha256=v1_sha)),
                    identity=identity())
    _write_frozen(path, revision)
    return revision


def _load_file(path):
    protocol = load_json(path)
    recorded = Path(str(path).replace('.json', '.sha256')).read_text().strip()
    if sha256(path) != recorded:
        raise ValueError('Frozen Stage 3 protocol was edited after freezing')
    return protocol, recorded


def load(path=PROTOCOL, original=PROTOCOL_V1):
    """Current protocol; a revision must cite the unchanged v1 and keep its science verbatim."""
    protocol, recorded = _load_file(path)
    revision = protocol.get('administrative_revision')
    if revision:
        v1, v1_sha = _load_file(original)
        if revision['supersedes'] != dict(file=Path(original).name, sha256=v1_sha):
            raise ValueError('Administrative revision does not cite the preserved original protocol')
        if any(protocol.get(key) != v1.get(key) for key in SCIENTIFIC):
            raise ValueError('Administrative revision changed scientific protocol content')
    return protocol, recorded


def drift(protocol, current=None):
    """List of identity keys that differ from the frozen protocol (empty = unchanged)."""
    current = current or identity()
    frozen = protocol['identity']
    differences = []
    for key in sorted(set(frozen) | set(current)):
        if key == 'execution_files':
            changed = sorted(p for p in set(frozen[key]) | set(current.get(key, {}))
                             if frozen[key].get(p) != current.get(key, {}).get(p))
            differences += [f'execution_files:{p}' for p in changed]
        elif frozen.get(key) != current.get(key):
            differences.append(key)
    return differences
