"""Resume only stopped prepared arms, using the runner's existing --resume path."""
import argparse, fcntl, json, os, socket, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from launch.experiment import configuration, locations, metadata, live_pids, available_gpus, launch_command, TOKEN_KEY
from training.pbt.config import contract_fingerprint
from training.runtime import atomic_json
PREP=Path(__file__).resolve().parent

def resume(name, gpu=None):
    config=configuration(PREP/(name+'.yaml'))
    run,logs=locations(config)
    with (logs/'resume.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        record=metadata(logs)
        if live_pids(record):raise RuntimeError(f'{name} still has live processes; refusing duplicate resume')
        manifest=json.loads((run/'manifest.json').read_text())
        if manifest['fingerprint']!=contract_fingerprint(config):raise RuntimeError('Resume fingerprint mismatch')
        if manifest['status']=='completed':
            print(f'{name}: already completed');return
        gpu_ids=[str(gpu)] if gpu is not None else record.get('physical_gpus',config['gpus'])
        uuids=available_gpus(gpu_ids)
        if gpu is None and uuids!=record['gpu_uuids']:
            raise RuntimeError('GPU mapping changed; choose a device explicitly with --gpu')
        command=launch_command(config,gpu_ids)+['--resume']
        with (logs/'main.log').open('a') as stream:
            process=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,**{TOKEN_KEY:record['token'],
                    'CUDA_VISIBLE_DEVICES':','.join(uuids),'PYTHONUNBUFFERED':'1'}),stdin=subprocess.DEVNULL,
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        record.update(pid=process.pid,command=command,gpu_uuids=uuids,physical_gpus=gpu_ids)
        atomic_json(logs/'launcher.json',record)
        (logs/'launcher.pid').write_text(f'{process.pid}\n')
        print(f'{name}: resumed at generation {manifest["next_generation"]}, PID {process.pid}')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm',choices=['lr_14e-6','lr_8_5e-6','both'])
    parser.add_argument('--gpu',type=int,help='Explicit physical GPU override for one stopped arm; config stays unchanged')
    args=parser.parse_args()
    if args.gpu is not None and (args.gpu<0 or args.arm=='both'):
        parser.error('--gpu requires a nonnegative index and one explicit arm')
    for name in (['lr_14e-6','lr_8_5e-6'] if args.arm=='both' else [args.arm]):resume(name,args.gpu)
