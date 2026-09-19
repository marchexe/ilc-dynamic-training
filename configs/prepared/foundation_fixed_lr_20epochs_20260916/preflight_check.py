import copy, contextlib, io, json, shlex, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pyarrow.parquet as pq
import torch
import yaml
sys.path.insert(0, str(Path('scripts').resolve()))
from training.pbt.config import load_config, validate_inputs
from training.pbt.runner import _plan_generation_exploit
from training.pbt.planning import should_apply_exploit_for_strategy, strategy_uses_population_rollbacks
from training.pbt.controller.decision import dynamic_controller_config
from training.pbt.state.checkpointing import bootstrap_initial_checkpoint
from training.pbt.execution.weaver_command import make_command, wrap_remote_command
from training.runtime import sha256
root=Path.cwd()
review=root/'configs/prepared/foundation_fixed_lr_20epochs_20260916'
smoke = len(sys.argv) > 1 and sys.argv[1] == 'smoke'
if smoke:
 review=review/'smoke'
review.mkdir(parents=True, exist_ok=True)
args=SimpleNamespace(config=root/('configs/experiments/foundation_fixed_lr_smoke.yaml' if smoke else 'configs/experiments/foundation_fixed_lr_20epochs.yaml'), experiment_name=None,
                     gpus=None,slots='iutgpu01:0,iutgpu01:1,iutgpu01:2,iutgpu01:3,iutgpu01:4',smoke=False)
c=load_config(args)
validate_inputs(c)
shared=c['shared']; pbt=c['pbt']
expected=[3e-6,5.75e-6,8.5e-6,11.25e-6,1.4e-5]
assert [m['start_lr'] for m in c['population']]==expected and len(set(expected))==5
assert shared['generations']==(1 if smoke else 20) and shared['weaver_epochs_per_generation']==1
assert shared.get('samples_per_epoch') is None and shared.get('samples_per_epoch_val') is None
assert shared['initial_optimizer_mode']=='raw'
assert shared['optimizer']=='ranger' and shared['batch_size']==1024 and shared['use_amp'] and shared['amp_dtype']=='fp16'
assert pbt['evaluate_initial_checkpoint'] and pbt['evaluate_final_checkpoints']
assert shared['freeze_batch_norm'] and shared['deterministic'] and shared['data_audit']
assert shared['lr_scheduler']=='none' and not shared.get('training_controller')
assert pbt['strategy']=='fixed_lr_grid' and pbt['rollback_fraction']==0
assert pbt['baseline_guard_action']=='observe' and pbt['early_stop_degraded_generations']==0
assert dynamic_controller_config(c) is None and not strategy_uses_population_rollbacks(c)
assert all(not pbt.get(k) for k in ['anchor_copy_lr_recenter','lr_controller','lr_radius','population_lr_policy','tiered_validation'])
pilot=root/'runs/pbt/foundation_correctness_20260916'
old=json.loads((pilot/'manifest.json').read_text())
for key in ['initial_state','initial_optimizer','seed','batch_size','num_workers','fetch_step','prefetch_factor','data_config','network_config']:
 assert shared[key]==old['config']['shared'][key],key
files={split:sorted(Path(shared['dataset']).glob('*_'+shared[split+'_suffix']+'.parquet')) for split in ['train','validation']}
rows={split:sum(pq.ParquetFile(f).metadata.num_rows for f in paths) for split,paths in files.items()}
assert rows=={'train':2392232,'validation':150000}
commands=[]
with tempfile.TemporaryDirectory(prefix='fixed_lr_preflight_') as tmp:
 tmp=Path(tmp)
 for m in c['population']:
  dest=tmp/m['name'];dest.mkdir()
  bootstrap_initial_checkpoint(c,dest)
  for component in ['state','optimizer']:
   a=torch.load(dest/f'net_epoch-17_{component}.pt',map_location='cpu',weights_only=False)
   source=Path(shared['initial_state' if component=='state' else 'initial_optimizer'])
   assert sha256(dest/f'net_epoch-17_{component}.pt') == sha256(source)
   b=torch.load(source,map_location='cpu',weights_only=False)
   torch.testing.assert_close(a, b, atol=0, rtol=0)
 members={m['name']:dict(name=m['name'],lr=m['start_lr']) for m in c['population']}
 manifest=dict(members=copy.deepcopy(members),best={},generations=[])
 for g in range(shared['generations']):
  assert not should_apply_exploit_for_strategy(c,g,g==shared['generations']-1,False)
  metrics={n:dict(metrics={pbt['metric']:float(i+1)},lr=m['lr']) for i,(n,m) in enumerate(members.items())}
  record=dict(index=g,epoch=18+g,workers=metrics,exploit=None)
  health=dict(consecutive_degraded_generations=1000,current_best_member=next(iter(members)),
              current_best_metric=1.,relative_to_global_best=10.,status='degraded',member_lrs={n:m['lr'] for n,m in members.items()})
  with contextlib.redirect_stdout(io.StringIO()), patch('training.pbt.runner.update_global_best',return_value=False), patch('training.pbt.runner.update_generation_health',return_value=health), patch('training.pbt.runner.add_baseline_guard_rollbacks',side_effect=AssertionError('rollback reached')), patch('training.pbt.runner.add_global_best_rollbacks',side_effect=AssertionError('rewind reached')), patch('training.pbt.controller.apply.build_observation',side_effect=AssertionError('controller reached')):
   _plan_generation_exploit(c,manifest,record,g,g==shared['generations']-1,tmp,tmp/'manifest.json',tmp/'pbt.log')
  assert record['exploit']==[] and not record['early_stop_triggered'] and manifest['members']==members
  assert not record.get('controller_actions') and not record.get('anchor_copy_lr_recenter')
  for m,slot in zip(c['population'],c['slots']):
   cmd,_,epoch=make_command(c,dict(name=m['name'],lr=m['start_lr']),slot['gpu'],tmp/m['name'],g)
   get=lambda flag: cmd[cmd.index(flag)+1]
   assert epoch==18+g and int(get('--num-epochs'))==19+g and int(get('--load-epoch'))==17+g
   assert int(get('--seed'))==12345+g and float(get('--start-lr'))==m['start_lr']
   assert get('--lr-scheduler')=='none'
   assert all(flag in cmd for flag in ['--freeze-batch-norm','--deterministic','--data-audit','--override-load-lr'])
   assert all(flag not in cmd for flag in ['--samples-per-epoch','--samples-per-epoch-val','--training-controller','--auto-clean'])
   actual=[x.replace(str(tmp),str(Path(c['output_root'])/c['experiment_name'])) for x in cmd]
   commands.append(dict(generation=g,epoch=epoch,member=m['name'],slot=slot['label'],command=wrap_remote_command(actual,slot)))
# The current dry-run builder creates empty log folders. Remove only those
# empty preparation folders, so the real fresh launch's existence guard passes.
run=Path(c['output_root'])/c['experiment_name']
if run.exists():
 assert all(p.is_dir() for p in run.rglob('*')), 'Refuse to remove any run containing files'
 for p in sorted(run.rglob('*'),key=lambda p:len(p.parts),reverse=True):p.rmdir()
 run.rmdir()
assert not run.exists()
state_size=(pilot/'identical_a/net_epoch-19_state.pt').stat().st_size
opt_size=(pilot/'identical_a/net_epoch-19_optimizer.pt').stat().st_size
scaler_size=(pilot/'identical_a/net_epoch-19_scaler.pt').stat().st_size
result=dict(status='PREPARED_NOT_LAUNCHED',commands_verified=len(commands),generations_with_no_adaptive_actions=shared['generations'], initial_optimizer_mode='raw',
            identical_initial_bootstraps=5,rows=rows,output_root=str(run),
            checkpoint_bytes_per_epoch=state_size+opt_size+scaler_size,
            checkpoint_bytes_all_arms=5*((shared['generations']+1)*(state_size+opt_size+scaler_size)+state_size+opt_size)+state_size+opt_size+scaler_size,
            fingerprints={k:sha256(Path(shared[k])) for k in ['initial_state','initial_optimizer','data_config','network_config']},
            config_sha256=sha256(args.config))
(review/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
(review/'resolved_config.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
(review/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')

(review/'preflight_check.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(result,indent=2))
