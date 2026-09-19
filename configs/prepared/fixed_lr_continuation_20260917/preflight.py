"""CPU-only preparation checks. Never launches a worker or touches source artifacts."""
import contextlib, copy, io, json, shutil, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from launch.experiment import configuration
from training.pbt.config import contract_fingerprint, validate_inputs
from training.pbt.execution.weaver_command import make_command
from training.pbt.state.checkpointing import bootstrap_initial_checkpoint
from training.pbt.runner import _plan_generation_exploit, _run_generation_workers, _load_or_create_manifest
from training.pbt.execution.backend import LocalWeaverBackend
from training.pbt.planning import should_apply_exploit_for_strategy, strategy_uses_population_rollbacks
from training.runtime import sha256

PREP = Path(__file__).resolve().parent
OLD = ROOT / 'runs/pbt/foundation_fixed_lr_50epochs_20260916'
old = json.loads((OLD / 'manifest.json').read_text())
checks, arms, commands = [], {}, {}

def check(label, condition):
    assert condition, label
    checks.append(label)

source_hashes = {}
for name, gpu in [('lr_14e-6', 1), ('lr_8_5e-6', 3)]:
    config = configuration(PREP / (name + '.yaml'))
    validate_inputs(config)
    shared, pbt = config['shared'], config['pbt']
    lr = old['members'][name]['lr']
    check(name + ': own source', Path(shared['initial_state']) == OLD / name / 'net_epoch-67_state.pt'
          and Path(shared['initial_optimizer']) == OLD / name / 'net_epoch-67_optimizer.pt')
    check(name + ': unchanged training settings', shared == dict(old['config']['shared'], checkpoint=shared['initial_state'],
          initial_state=shared['initial_state'], initial_optimizer=shared['initial_optimizer'], initial_epoch=67, seed=12395))
    check(name + ': exact LR', config['population'] == [dict(name=name, start_lr=lr)])
    check(name + ': separate output', Path(config['output_root']) / config['experiment_name'] != OLD)
    check(name + ': raw optimizer', shared['initial_optimizer_mode'] == 'raw')
    check(name + ': adaptive paths disabled', pbt['strategy'] == 'fixed_lr_grid' and pbt['rollback_fraction'] == 0
          and pbt['early_stop_degraded_generations'] == 0 and pbt['baseline_guard_action'] == 'observe'
          and pbt['dynamic_controller']['mode'] == 'disabled' and not strategy_uses_population_rollbacks(config)
          and all(not pbt.get(k) for k in ['anchor_copy_lr_recenter','lr_controller','lr_radius','population_lr_policy','tiered_validation'])
          and not shared.get('training_controller') and shared['lr_scheduler'] == 'none')
    hashes = {}
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / name
        dest.mkdir()
        bootstrap_initial_checkpoint(config, dest)
        for part in ['state', 'optimizer', 'scaler']:
            source = OLD / name / f'net_epoch-67_{part}.pt'
            target = dest / source.name
            hashes[part] = sha256(source)
            check(name + ': exact copied ' + part, hashes[part] == sha256(target))
            torch.testing.assert_close(torch.load(source, map_location='cpu', weights_only=False),
                                       torch.load(target, map_location='cpu', weights_only=False), atol=0, rtol=0)
        check(name + ': verified source model', hashes['state'] == old['final_evaluations']['control'][name]['checkpoint_sha256'])
        optimizer = torch.load(OLD / name / 'net_epoch-67_optimizer.pt', map_location='cpu', weights_only=False)
        scaler = torch.load(OLD / name / 'net_epoch-67_scaler.pt', map_location='cpu', weights_only=False)
        check(name + ': optimizer LR already correct', all(g['lr'] == lr for g in optimizer['param_groups']))
        check(name + ': scaler state complete', set(scaler) == {'scale','growth_factor','backoff_factor','growth_interval','_growth_tracker'})
        members = {name: dict(name=name, lr=lr)}
        manifest = dict(members=copy.deepcopy(members), best={}, generations=[])
        commands[name] = []
        for g in range(50):
            command, _, epoch = make_command(config, members[name], '0', dest, g)
            get = lambda flag: command[command.index(flag) + 1]
            check(f'{name}/{g}: numbering and schedule', epoch == 68 + g and get('--load-epoch') == str(67 + g)
                  and get('--num-epochs') == str(69 + g) and get('--seed') == str(12395 + g))
            check(f'{name}/{g}: full dataset and retention', '--samples-per-epoch' not in command
                  and '--samples-per-epoch-val' not in command and '--auto-clean' not in command
                  and '--data-audit' in command and '--deterministic' in command and '--freeze-batch-norm' in command)
            check(f'{name}/{g}: exploit gate', not should_apply_exploit_for_strategy(config,g,g==49,False))
            record = dict(index=g, epoch=epoch, workers={name:dict(metrics={pbt['metric']:1.0},lr=lr)}, exploit=None)
            health = dict(consecutive_degraded_generations=1000,current_best_member=name,current_best_metric=1.0,
                          relative_to_global_best=10.0,status='degraded',member_lrs={name:lr})
            with contextlib.redirect_stdout(io.StringIO()), patch('training.pbt.runner.update_global_best',return_value=False), \
                 patch('training.pbt.runner.update_generation_health',return_value=health), \
                 patch('training.pbt.runner.add_baseline_guard_rollbacks',side_effect=AssertionError('rollback reached')), \
                 patch('training.pbt.runner.add_global_best_rollbacks',side_effect=AssertionError('rewind reached')), \
                 patch('training.pbt.controller.apply.build_observation',side_effect=AssertionError('controller reached')):
                _plan_generation_exploit(config,manifest,record,g,g==49,Path(tmp),Path(tmp)/'manifest.json',Path(tmp)/'pbt.log')
            check(f'{name}/{g}: no adaptive action', record['exploit']==[] and not record['early_stop_triggered'] and manifest['members']==members)
            run = Path(config['output_root']) / config['experiment_name']
            commands[name].append([arg.replace(str(Path(tmp)), str(run)) for arg in command])
        # Exercise the actual resume loader on an isolated copy of a real manifest.
        restored = copy.deepcopy(old)
        restored.update(status='failed', next_generation=9, fingerprint=contract_fingerprint(config), config=config)
        restored['members'] = {name: old['members'][name]}
        path = Path(tmp)/'resume/manifest.json';path.parent.mkdir();path.write_text(json.dumps(restored))
        resumed = _load_or_create_manifest(SimpleNamespace(resume=True),config,LocalWeaverBackend(),path.parent,path,
                                          contract_fingerprint(config),[])
        check(name + ': resume cursor retained', resumed['next_generation'] == 9)
        record = dict(index=9, workers={name:dict(status='completed')})
        backend = LocalWeaverBackend()
        with patch.object(backend,'run_generation') as execute:
            _run_generation_workers(config,backend,path.parent,resumed,record,path)
            check(name + ': completed worker not repeated', execute.call_args.args[4] == [])
            record['workers'][name]['status'] = 'failed'
            _run_generation_workers(config,backend,path.parent,resumed,record,path)
            check(name + ': failed worker retried alone', execute.call_args.args[4] == [name])
    source_hashes[name] = hashes
    arms[name] = dict(config=str(PREP/(name+'.yaml')),run=str(Path(config['output_root'])/config['experiment_name']),
                     gpu=gpu,lr=lr,source=str(OLD/name/'net_epoch-67_state.pt'),hashes=hashes,
                     scaler=scaler,optimizer_steps=sorted({int(v['step']) for v in optimizer['state'].values() if 'step' in v}),
                     fingerprint=contract_fingerprint(config))

files = old['datasets']['resolved_files']
rows = {s:sum(pq.ParquetFile(f).metadata.num_rows for group in files[s] for f in group['files']) for s in ['train','val']}
check('source dataset row counts',rows == dict(train=2392232,val=150000))
check('source artifacts unchanged',json.loads((PREP/'source_artifacts_snapshot.json').read_text()) == {
      str(p.relative_to(OLD)):[p.stat().st_size,p.stat().st_mtime_ns] for p in OLD.rglob('*') if p.is_file()})
bytes_per_set = sum((OLD/'lr_14e-6'/f'net_epoch-67_{part}.pt').stat().st_size for part in ['state','optimizer','scaler'])
check('disk reserve >= 3 GiB',shutil.disk_usage(ROOT).free > 3*1024**3)
report=dict(passed=True,checks_passed=len(checks),checks=checks,arms=arms,rows=rows,
            checkpoint_bytes_100_sets=100*bytes_per_set,disk_free_bytes=shutil.disk_usage(ROOT).free)
(PREP/'preflight.json').write_text(json.dumps(report,indent=2))
(PREP/'commands.json').write_text(json.dumps(commands,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))
