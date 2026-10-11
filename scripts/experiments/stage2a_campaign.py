#!/usr/bin/env python3
"""Prepare/probe/dry-run or launch the frozen Stage 2A campaign."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from experiments.stage2a.common import OUTPUT, PREPARED, PROJECT_DIR, lock_file, sha256, utc_now, verify_file, write_json
from experiments.stage2a.scheduling import choose_cohort, discover, probe_local, ssh_args


def output_preflight(output):
    output = Path(output).resolve()
    if output != OUTPUT:
        # Tests use temporary destinations. Never let an override target history.
        if (PROJECT_DIR/'runs/pbt') == output or (PROJECT_DIR/'runs/pbt') in output.parents:
            raise ValueError('Output must not be inside historical training runs')
    output.mkdir(parents=True,exist_ok=True)
    probe = output/f'.write-check-{os.getpid()}'
    probe.write_text('write-check');probe.unlink()
    free = shutil.disk_usage(output).free
    if free < 10*1024**3:
        raise RuntimeError('Less than 10 GiB free for predictions/logs/analysis')
    return dict(path=str(output),free_bytes=free,minimum_free_bytes=10*1024**3)


def completed_evaluations(protocol, protocol_sha, output):
    from experiments.stage2a.execution import verified_completed
    return all(verified_completed(Path(output)/'evaluations'/c['id']/'result.json',c,protocol,protocol_sha)
               for c in protocol['checkpoints'])


def completed_output(protocol, protocol_sha, output):
    try:
        status=json.loads((Path(output)/'status.json').read_text())
        integrity=json.loads((Path(output)/'report_integrity.json').read_text())
        if (status['status']!='completed' or status['protocol_sha256']!=protocol_sha
                or integrity['protocol_sha256']!=protocol_sha or not integrity['artifacts']):
            return False
        for artifact in integrity['artifacts']:
            verify_file(artifact)
        return completed_evaluations(protocol,protocol_sha,output)
    except (OSError,KeyError,ValueError,TypeError,RuntimeError):
        return False


def tmux_alive(host, session):
    result = subprocess.run(ssh_args(host,shlex.join(['tmux','has-session','-t','='+session])),
                            capture_output=True,text=True,timeout=15)
    if result.returncode not in (0,1):
        raise RuntimeError(f'Cannot establish persistent-session status on {host}: {result.stderr.strip()}')
    return result.returncode == 0


def launch_tmux(host, session, output, prepared, cohort):
    remote_probe = 'command -v tmux >/dev/null && test -x '+shlex.quote(str(PROJECT_DIR/'.venv/bin/python'))
    subprocess.run(ssh_args(host,remote_probe),check=True,capture_output=True,text=True,timeout=15)
    script = shlex.join([str(PROJECT_DIR/'.venv/bin/python'),str(Path(__file__).resolve()),
                        '--on-node','--prepared',str(prepared),'--output',str(output),'--cohort',str(output/'launch_cohort.json')])
    inner = 'cd '+shlex.quote(str(PROJECT_DIR))+' && exec '+script+' >> '+shlex.quote(str(output/'supervisor.log'))+' 2>&1'
    command = shlex.join(['tmux','new-session','-d','-s',session,'bash','-lc',inner])
    subprocess.run(ssh_args(host,command),check=True,capture_output=True,text=True,timeout=15)


def plan(protocol, protocol_sha, nodes, output, max_parallelism):
    from experiments.stage2a.execution import inference_command
    cohort = choose_cohort(nodes,protocol['allocation'],max_parallelism)
    return dict(status='ready',checked_at=utc_now(),protocol_sha256=protocol_sha,output=str(output),
                gpu_observation=nodes,cohort=cohort,inference_launched=False,gpus_reserved=False,
                jobs=[dict(checkpoint_id=c['id'],checkpoint_sha256=c['checkpoint']['sha256'],
                           command=inference_command(protocol,c,Path(output)/'evaluations'/c['id']/'evaluation.log'),
                           visible_device='one launch-time verified GPU UUID') for c in protocol['checkpoints']])


def run_on_node(args):
    from experiments.stage2a.protocol import load_protocol
    from experiments.stage2a.execution import run_campaign
    from experiments.stage2a.analysis import build_report
    cohort = json.loads(args.cohort.read_text())
    if socket.gethostname().split('.')[0] != cohort['host']:
        raise ValueError('Persistent worker started on a different node')
    stop = threading.Event()
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda *_:stop.set())
    output_preflight(args.output)
    protocol,protocol_sha = load_protocol(args.prepared)
    if cohort['host'] not in protocol['allocation']['authorized_nodes']:
        raise ValueError('Persistent worker host is not authorized')
    if not 1 <= cohort['max_parallelism'] <= 9 or len({g['uuid'] for g in cohort['gpus']}) != len(cohort['gpus']):
        raise ValueError('Invalid GPU cohort')
    def finalize(results):
        # Rehash immutable dependencies after inference, before successful publication.
        if load_protocol(args.prepared)[1] != protocol_sha:
            raise ValueError('Frozen protocol changed during inference; report publication blocked')
        return build_report(protocol,results,args.output,protocol_sha)
    campaign = run_campaign(protocol,protocol_sha,args.output,cohort,stop=stop,finalize=finalize)
    return 0 if campaign['status']=='completed' else 130 if campaign['status']=='interrupted' else 1


def start(args):
    from experiments.stage2a.protocol import load_protocol
    output = args.output.resolve()
    output_preflight(output)
    with lock_file(output/'.launch.lock'):
        protocol,protocol_sha = load_protocol(args.prepared)
        if not args.dry_run:
            if completed_output(protocol,protocol_sha,output):
                print(json.dumps(dict(status='already_completed_and_verified',output=str(output)),indent=2));return 0
            old = output/'launch.json'
            if old.exists():
                launch = json.loads(old.read_text())
                if launch['protocol_sha256'] != protocol_sha:
                    raise ValueError('Output directory belongs to a different protocol')
                if tmux_alive(launch['host'],launch['session']):
                    print(json.dumps(dict(status='already_running',**launch),indent=2));return 0
        analysis_only = not args.dry_run and completed_evaluations(protocol,protocol_sha,output)
        nodes = [] if analysis_only else discover(protocol['allocation'])
        maximum = (args.max_parallelism if args.max_parallelism is not None else
                   int(os.environ.get('STAGE2A_MAX_PARALLEL',protocol['allocation']['max_parallelism'])))
        try:
            prepared_plan = (dict(status='ready_for_analysis_only',protocol_sha256=protocol_sha,
                inference_launched=False,gpus_reserved=False,
                cohort=dict(host=protocol['allocation']['preferred_node'],name='analysis_only',total_mib=0,
                            max_parallelism=1,gpus=[])) if analysis_only else plan(protocol,protocol_sha,nodes,output,maximum))
        except RuntimeError as error:
            write_json(output/'resource_failure.json',dict(checked_at=utc_now(),error=str(error),nodes=nodes))
            raise
        if args.dry_run:
            write_json(args.prepared/'dry_run.json',prepared_plan)
            print(json.dumps(prepared_plan,indent=2));return 0
        # check.sh is retained as an independent operational snapshot, not allocation authority.
        snapshot = subprocess.run(['bash',str(PROJECT_DIR/'scripts/cluster/check.sh')],
                                  capture_output=True,text=True,timeout=120)
        (output/'cluster_snapshot.txt').write_text(snapshot.stdout+snapshot.stderr)
        cohort = prepared_plan['cohort']
        session = 'stage2a-'+protocol_sha[:12]
        write_json(output/'launch_cohort.json',cohort)
        write_json(output/'runtime_preflight.json',prepared_plan)
        launch_tmux(cohort['host'],session,output,args.prepared.resolve(),cohort)
        launch = dict(host=cohort['host'],session=session,protocol_sha256=protocol_sha,
                      output=str(output),started_at=utc_now(),max_parallelism=cohort['max_parallelism'])
        write_json(output/'launch.json',launch)
        print(json.dumps(dict(status='submitted_to_persistent_tmux',**launch),indent=2))
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--probe',action='store_true')
    modes.add_argument('--freeze',action='store_true')
    modes.add_argument('--dry-run',action='store_true')
    modes.add_argument('--on-node',action='store_true')
    parser.add_argument('--prepared',type=Path,default=PREPARED)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--cohort',type=Path)
    parser.add_argument('--max-parallelism',type=int)
    args = parser.parse_args(argv)
    try:
        if args.probe:
            print(json.dumps(probe_local()));return 0
        if args.freeze:
            from experiments.stage2a.protocol import freeze
            protocol = freeze(args.prepared)
            print(json.dumps(dict(protocol=str(args.prepared/'protocol.json'),checkpoint_count=len(protocol['checkpoints']),
                                  holdout_rows=protocol['dataset']['rows'],inference_launched=False)));return 0
        if args.on_node:
            if not args.cohort:
                parser.error('--on-node requires the internal launch cohort')
            code = run_on_node(args)
            write_json(args.output/'supervisor_exit.json',dict(time=utc_now(),exit_code=code))
            return code
        return start(args)
    except Exception as error:
        print(f'{utc_now()} Stage2A failed: {error}',file=sys.stderr,flush=True)
        if not args.probe:
            try:write_json(args.output/'last_failure.json',dict(time=utc_now(),error=str(error),inference_launched=False if args.dry_run else None))
            except OSError:pass
        code = 69 if isinstance(error,RuntimeError) else 75 if isinstance(error,BlockingIOError) else 2
        if args.on_node:
            try:
                write_json(args.output/'supervisor_exit.json',dict(time=utc_now(),exit_code=code,error=str(error)))
                status = args.output/'status.json'
                old = json.loads(status.read_text()) if status.exists() else {}
                write_json(status,dict(old,status='failed',supervisor_error=str(error)))
            except OSError:pass
        return code


if __name__=='__main__':
    raise SystemExit(main())
