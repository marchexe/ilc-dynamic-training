"""Programmatic parity between the inactive Stage 3 train8m drafts.

Arm A (fixed_lr_grid) and arm B (windowed_pbt_v2) must differ only in the
adaptation policy. Everything that defines data, initialization, optimizer,
restart contract, seeds, evaluation and compute must be identical.
"""
import argparse
from pathlib import Path

from experiments.stage2b.common import DRAFT_CONFIGS, PROJECT_DIR, utc_now
from experiments.stage2b.dataset import check_validation_suffix, output_dir
from training.pbt.config import load_config
from training.pbt.planning.windowed_pbt_v2 import STRATEGY as WINDOWED
from training.pbt.reference.windowed_v2 import REFERENCE_REVISION

SHARED_PARITY = ('dataset', 'validation_dataset', 'data_extension', 'train_suffix', 'validation_suffix',
                 'checkpoint', 'initial_state', 'initial_optimizer', 'initial_epoch', 'initial_optimizer_mode',
                 'initial_controller', 'data_config', 'network_config', 'seed', 'generations',
                 'weaver_epochs_per_generation', 'samples_per_epoch', 'samples_per_epoch_val', 'batch_size',
                 'optimizer', 'lr_scheduler', 'use_amp', 'amp_dtype', 'freeze_batch_norm', 'deterministic',
                 'data_audit', 'no_remake_weights', 'num_workers', 'fetch_step', 'prefetch_factor',
                 'training_controller', 'proxy_validation')
PBT_PARITY = ('metric', 'mode', 'backend', 'evaluate_initial_checkpoint', 'evaluate_final_checkpoints',
              'rollback_fraction', 'baseline_guard_action', 'early_stop_degraded_generations')
FROZEN_WINDOW = PROJECT_DIR / 'configs/experiments/windowed_pbt_v2.yaml'


def resolve(path):
    return load_config(argparse.Namespace(config=Path(path), experiment_name=None, gpus=None, slots=None, smoke=False))


def cadence(config):
    """Decision schedule implied by the frozen windowed policy at this horizon."""
    window = config['pbt'][WINDOWED]
    generations = config['shared']['generations']
    boundaries = list(range(window['window_epochs'], generations + 1, window['window_epochs']))
    # A member needs `losing_windows` consecutive losing windows; the terminal boundary never copies.
    eligible = [b for b in boundaries if b // window['window_epochs'] >= window['losing_windows'] and b != generations]
    return dict(window_epochs=window['window_epochs'], boundaries_epochs=boundaries,
                earliest_copy_boundary_epoch=eligible[0] if eligible else None,
                non_terminal_copy_boundaries=eligible, max_copies=len(eligible) * window['max_recipients'])


def check(spec, configs=None, *, frozen=FROZEN_WINDOW, resolver=resolve):
    configs = configs or DRAFT_CONFIGS
    a, b = resolver(configs['fixed_lr']), resolver(configs['windowed_pbt_v2'])
    reference = resolver(frozen)
    results = []

    def require(name, condition, detail=None):
        results.append(dict(check=name, passed=bool(condition), detail=detail))

    for key in SHARED_PARITY:
        require(f'shared.{key} identical', a['shared'].get(key) == b['shared'].get(key),
                dict(fixed_lr=a['shared'].get(key), windowed_pbt_v2=b['shared'].get(key)))
    for key in PBT_PARITY:
        require(f'pbt.{key} identical', a['pbt'].get(key) == b['pbt'].get(key))
    require('population names and start LRs identical (matched seeds and equal population compute)',
            a['population'] == b['population'], [m['start_lr'] for m in a['population']])
    for name, config in (('fixed_lr', a), ('windowed_pbt_v2', b)):
        shared = config['shared']
        require(f'{name}: trains on the Stage 2B train8m directory',
                Path(shared['dataset']) == output_dir(spec).resolve() and shared['train_suffix'] == 'train8m')
        try:
            check_validation_suffix(spec, shared['validation_suffix'])
            approved = Path(shared['validation_dataset']) == (PROJECT_DIR / spec['validation']['dataset']).resolve()
        except ValueError:
            approved = False
        require(f'{name}: isolated approved selection split', approved, shared['validation_suffix'])
        require(f'{name}: full natural epochs (no sample caps)',
                shared.get('samples_per_epoch') is None and shared.get('samples_per_epoch_val') is None)
        require(f'{name}: epoch-restarted raw Ranger/FP16 contract',
                shared['optimizer'] == 'ranger' and shared['lr_scheduler'] == 'none' and shared['use_amp']
                and shared['amp_dtype'] == 'fp16' and shared['initial_optimizer_mode'] == 'raw'
                and shared['weaver_epochs_per_generation'] == 1 and shared['freeze_batch_norm'])
    require('arm A never exploits', a['pbt']['strategy'] == 'fixed_lr_grid'
            and a['pbt']['exploit_interval_generations'] > a['shared']['generations'])
    require('arm B uses the frozen windowed_pbt_v2 policy', b['pbt']['strategy'] == WINDOWED
            and all(b['pbt'].get(k) == reference['pbt'].get(k) for k in (WINDOWED, 'mutation_factors', 'min_lr', 'max_lr',
                                                                       'exploit_interval_generations')),
            dict(reference_revision=REFERENCE_REVISION, windowed=b['pbt'].get(WINDOWED)))
    members, generations = len(a['population']), a['shared']['generations']
    budget = dict(members=members, generations=generations, member_epochs=members * generations,
                  per_member_samples='generations x one full train8m epoch (identical seeds per generation)')
    require('matched optimizer-step/sample budgets', len(b['population']) == members
            and b['shared']['generations'] == generations, budget)
    return dict(checked_at=utc_now(), passed=all(r['passed'] for r in results), checks=results, budget=budget,
                cadence=cadence(b), configs={k: str(v) for k, v in configs.items()})
