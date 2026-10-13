"""Stage 3 locations, policy and exit codes; generic helpers come from Stages 2A/2B."""
from pathlib import Path

from experiments.stage2a.common import digest_json, file_record, lock_file, log, sha256, utc_now, verify_file, write_json
from experiments.stage2b.common import disk_free, load_json, own_processes_with
from training.runtime import PROJECT_DIR

PREPARED = PROJECT_DIR / 'configs/prepared/stage3_train8m_v1'
# protocol.json is the original v1 freeze (kept verbatim); v2 is its administrative revision.
PROTOCOL_V1 = PREPARED / 'protocol.json'
PROTOCOL = PREPARED / 'protocol_v2.json'
POLICY = PREPARED / 'policy.json'
TEST_SPEC = PREPARED / 'test3000k_spec.json'
OUTPUT = PROJECT_DIR / 'runs/eval/stage3_train8m_v1'
ARMS = ('fixed_lr', 'windowed_pbt_v2')
ARM_CONFIGS = dict(fixed_lr=PROJECT_DIR / 'configs/experiments/stage3_train8m_fixed_lr.yaml',
                   windowed_pbt_v2=PROJECT_DIR / 'configs/experiments/stage3_train8m_windowed_pbt_v2.yaml')
GATE_CONFIG = PROJECT_DIR / 'configs/experiments/stage3_train8m_gate.yaml'
PRIMARY = 'validation_total_reference_mistag_geomean_percent'

EXIT_OK, EXIT_FAILED, EXIT_INVALID, EXIT_RESOURCES, EXIT_LOCKED, EXIT_INTERRUPTED = 0, 1, 2, 69, 75, 130

__all__ = ['ARMS', 'ARM_CONFIGS', 'GATE_CONFIG', 'OUTPUT', 'POLICY', 'PREPARED', 'PRIMARY', 'PROJECT_DIR', 'PROTOCOL',
           'PROTOCOL_V1', 'project_file',
           'TEST_SPEC', 'digest_json', 'disk_free', 'file_record', 'load_json', 'load_policy', 'lock_file', 'log',
           'own_processes_with', 'resolve', 'sha256', 'utc_now', 'verify_file', 'write_json']


def project_file(path):
    """Project-relative paths (portable records) resolved against the checkout."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_DIR / path


def relative(path):
    """Record a path inside the checkout without the machine-specific prefix."""
    path = Path(path)
    return str(path.relative_to(PROJECT_DIR)) if path.is_absolute() and path.is_relative_to(PROJECT_DIR) else str(path)


def resolve(path, **overrides):
    """Production config resolution (schema-validated), as the runner sees it."""
    import argparse
    from training.pbt.config import load_config
    options = dict(experiment_name=None, gpus=None, slots=None, smoke=False)
    options.update(overrides)
    return load_config(argparse.Namespace(config=Path(path), **options))


def load_policy(path=POLICY):
    """Committed policy plus local cluster nodes; the digest covers the committed file only."""
    from experiments.site_config import with_cluster
    committed = load_json(path)
    policy = dict(committed, allocation=with_cluster(committed['allocation']))
    if policy.get('schema_version') != 1:
        raise ValueError('Unsupported Stage 3 policy')
    allocation, budgets = policy['allocation'], policy['budgets']
    if allocation['preferred_node'] not in allocation['authorized_nodes']:
        raise ValueError('Preferred node must be authorized')
    if allocation['gpus_per_arm'] != 5 or allocation['max_concurrent_gpus'] not in (5, 10):
        raise ValueError('Each arm uses five GPUs; at most both arms run concurrently')
    for key, value in budgets.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value < 0:
            raise ValueError(f'Budget {key} must be non-negative')
    return policy, digest_json(committed)
