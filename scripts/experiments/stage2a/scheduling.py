"""Direct-SSH allocation under the explicitly confirmed local cluster policy."""
import csv
from concurrent.futures import ThreadPoolExecutor
import os
import shlex
import socket
import subprocess

from experiments.stage2a.common import PROJECT_DIR


def ssh_args(host, command):
    args = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','-o','ConnectionAttempts=1']
    if os.environ.get('ML_PFA_SSH_CONFIG'):
        args += ['-F',os.environ['ML_PFA_SSH_CONFIG']]
    return [*args,host,command]


def probe_local():
    gpu = subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu,uuid',
                          '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15,check=True)
    apps = subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid',
                           '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15,check=True)
    return parse_probe(gpu.stdout, apps.stdout, socket.gethostname().split('.')[0])


def parse_probe(gpu_text, app_text, host):
    processes = {}
    for row in csv.reader(app_text.splitlines()):
        if not row or not any(x.strip() for x in row):
            continue
        if len(row) != 2:
            raise ValueError('Cannot establish GPU process ownership')
        pid, uuid = (x.strip() for x in row)
        processes.setdefault(uuid,[]).append(int(pid))
    result = []
    for row in csv.reader(gpu_text.splitlines()):
        if len(row) != 6:
            raise ValueError('Malformed GPU availability response')
        index,name,total,used,util,uuid = [x.strip() for x in row]
        result.append(dict(host=host,index=int(index),name=name,total_mib=int(total),used_mib=int(used),
                           utilization=int(util),uuid=uuid,pids=processes.get(uuid,[])))
    return result


def free_gpu(gpu, policy):
    return (not gpu['pids'] and gpu['used_mib'] < policy['max_used_memory_mib']
            and gpu['utilization'] < policy['max_util_percent']
            and gpu['total_mib'] >= policy['min_gpu_memory_mib'])


def probe_node(host):
    import json
    command = shlex.join([str(PROJECT_DIR/'.venv/bin/python'),
                          str(PROJECT_DIR/'scripts/experiments/stage2a_campaign.py'),'--probe'])
    try:
        result = subprocess.run(ssh_args(host,command),capture_output=True,text=True,timeout=30,check=True)
        return dict(host=host,gpus=json.loads(result.stdout),error=None)
    except Exception as error:
        return dict(host=host,gpus=[],error=str(error))


def discover(policy):
    requested = os.environ.get('ML_PFA_CLUSTER_NODES',' '.join(policy['authorized_nodes'])).split()
    if not requested or any(host not in policy['authorized_nodes'] for host in requested):
        raise ValueError('Requested GPU node is not authorized by the frozen allocation policy')
    with ThreadPoolExecutor(max_workers=len(requested)) as pool:
        return list(pool.map(probe_node,requested))


def choose_cohort(nodes, policy, max_parallelism=None):
    limit = policy['max_parallelism'] if max_parallelism is None else max_parallelism
    if not 1 <= limit <= 9:
        raise ValueError('Parallelism must be between one and nine checkpoints')
    groups = {}
    for node in nodes:
        if node['host'] not in policy['authorized_nodes']:
            continue
        for gpu in node['gpus']:
            if free_gpu(gpu,policy):
                key = (node['host'],gpu['name'],gpu['total_mib'])
                groups.setdefault(key,[]).append(gpu)
    if not groups:
        raise RuntimeError('No authorized, idle GPU with sufficient memory; CPU fallback is forbidden')
    def rank(item):
        (host,name,memory),gpus = item
        return (host == policy['preferred_node'], 'A100' in name, memory, min(len(gpus),limit),host)
    key,gpus = max(groups.items(),key=rank)
    return dict(host=key[0],name=key[1],total_mib=key[2],
                max_parallelism=min(len(gpus),limit),gpus=sorted(gpus,key=lambda x:x['index'])[:limit])
