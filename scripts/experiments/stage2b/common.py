"""Stage 2B locations and versioned policy; generic helpers come from Stage 2A."""
import json
import os
from pathlib import Path
import shutil

from experiments.stage2a.common import digest_json, file_record, lock_file, log, sha256, utc_now, verify_file, write_json
from training.runtime import PROJECT_DIR

PREPARED = PROJECT_DIR / 'configs/prepared/stage2b_train8m_v1'
SPEC = PREPARED / 'dataset_spec.json'
POLICY = PREPARED / 'smoke_policy.json'
OUTPUT = PROJECT_DIR / 'runs/eval/stage2b_train8m_readiness_v1'
SMOKE_CONFIG = PROJECT_DIR / 'configs/experiments/stage2b_train8m_smoke.yaml'
DRAFTS = PROJECT_DIR / 'configs/drafts/stage3_train8m_v1'
DRAFT_CONFIGS = dict(fixed_lr=DRAFTS / 'fixed_lr.yaml', windowed_pbt_v2=DRAFTS / 'windowed_pbt_v2.yaml')
PRIMARY = 'validation_total_reference_mistag_geomean_percent'

# Exit codes shared with the Stage 2A launcher convention.
EXIT_OK, EXIT_FAILED, EXIT_INVALID, EXIT_RESOURCES, EXIT_LOCKED, EXIT_INTERRUPTED = 0, 1, 2, 69, 75, 130

__all__ = ['DRAFTS', 'DRAFT_CONFIGS', 'OUTPUT', 'POLICY', 'PREPARED', 'PRIMARY', 'PROJECT_DIR', 'SMOKE_CONFIG',
           'SPEC', 'digest_json', 'disk_free', 'file_record', 'load_json', 'lock_file', 'log', 'project_relative',
           'sha256', 'utc_now', 'verify_file', 'write_json']


def load_json(path):
    return json.loads(Path(path).read_text())


def project_relative(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_DIR / path


def disk_free(path):
    """Free bytes on the filesystem that will hold `path` (its nearest existing ancestor)."""
    path = Path(path)
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def load_policy(path=POLICY):
    policy = load_json(path)
    if policy.get('schema_version') != 1:
        raise ValueError('Unsupported Stage 2B smoke policy')
    allocation, budgets, smoke = policy['allocation'], policy['budgets'], policy['smoke']
    if not 1 <= allocation['gpus_required'] <= 2:
        raise ValueError('The bounded smoke uses one or two GPUs')
    if allocation['preferred_node'] not in allocation['authorized_nodes']:
        raise ValueError('Preferred node must be authorized')
    for key in ('max_wall_seconds', 'max_gpu_hours', 'max_cuda_memory_mib', 'max_run_disk_bytes',
                'min_free_bytes_after_dataset', 'conversion_workers'):
        if not budgets[key] > 0:
            raise ValueError(f'Budget {key} must be positive')
    if smoke['interrupt_generation'] < 1:
        raise ValueError('Interruption must hit a generation that resumes from a smoke checkpoint')
    return policy, digest_json(policy)


def own_processes_with(token, *, proc=Path('/proc')):
    """PIDs of this user's processes whose environment carries `token`."""
    found = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            if f'STAGE2B_SMOKE_TOKEN={token}'.encode() in (entry / 'environ').read_bytes().split(b'\0'):
                found.append(int(entry.name))
        except OSError:
            continue
    return sorted(found)
