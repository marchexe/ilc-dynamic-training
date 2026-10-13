#!/usr/bin/env python3
"""Verify the completed Stage 2A campaign's frozen inputs and published results.

Read-only. Performs every check of the Stage 2A launcher's input verification
(protocol digest, scientific invariants, package versions, evaluation globs,
dataset, isolation and checkpoint hashes) and every pinned execution file.
A pinned file listed in a committed administrative revision must still be
recoverable byte for byte from the recorded Git commit with exactly the pinned
SHA256 and size, and its current content must match the revision's recorded
hash; any other difference fails. The Stage 2A protocol itself is never edited.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.stage2a.common import PREPARED, environment, sha256, verify_file  # noqa: E402
from experiments.stage2a.dataset import verify_evaluation_files  # noqa: E402
from experiments.stage2a.protocol import load_protocol  # noqa: E402
from training.runtime import PROJECT_DIR  # noqa: E402

REVISION = PREPARED / 'administrative_revision_1.json'
OUTPUT = PROJECT_DIR / 'runs/eval/stage2a_holdout_v1'


def relative(path):
    path = Path(path)
    return str(path.relative_to(PROJECT_DIR)) if path.is_absolute() and path.is_relative_to(PROJECT_DIR) else str(path)


def git_blob(commit, path):
    result = subprocess.run(['git', 'show', f'{commit}:{path}'], cwd=PROJECT_DIR, capture_output=True)
    if result.returncode:
        raise ValueError(f'{path} is not present in commit {commit}')
    return result.stdout


def load_revision(path=REVISION):
    revision = json.loads(Path(path).read_text())
    recorded = Path(str(path).replace('.json', '.sha256')).read_text().strip()
    if sha256(Path(path)) != recorded:
        raise ValueError('Stage 2A administrative revision record was edited')
    return revision


def verify_execution_files(records, revision, *, blob=git_blob):
    """Pinned files must match; revised files must match both their frozen and recorded new state."""
    changed = {item['path']: item for item in revision.get('changed_execution_files', [])}
    unknown = set(changed) - {relative(r['path']) for r in records}
    if unknown:
        raise ValueError(f'Administrative revision names files that are not pinned: {sorted(unknown)}')
    checked = []
    for record in records:
        path = relative(record['path'])
        if path not in changed:
            verify_file(record)
            checked.append(dict(path=path, state='unchanged'))
            continue
        item = changed[path]
        original = blob(item['original_commit'], path)
        if (item['frozen_sha256'] != record['sha256'] or hashlib.sha256(original).hexdigest() != record['sha256']
                or len(original) != record['size_bytes']):
            raise ValueError(f'{path}: frozen content is not recoverable from {item["original_commit"]}')
        if sha256(PROJECT_DIR / path) != item['current_sha256']:
            raise ValueError(f'{path}: current content differs from the administrative revision')
        checked.append(dict(path=path, state='administratively_revised', original_commit=item['original_commit']))
    return checked


def verify(prepared=PREPARED, revision_path=REVISION, output=OUTPUT):
    protocol, digest = load_protocol(prepared, verify_inputs=False)
    revision = load_revision(revision_path)
    if revision['protocol_sha256'] != digest:
        raise ValueError('Administrative revision belongs to a different Stage 2A protocol')
    if protocol['environment'] != environment():
        raise ValueError('Frozen Python/package versions changed')
    verify_evaluation_files(protocol)
    files = verify_execution_files(protocol['execution_files'], revision)
    for record in [protocol['dataset']['manifest'], *protocol['dataset']['files'], *protocol['dataset']['training_isolation']]:
        verify_file(record)
    for checkpoint in protocol['checkpoints']:
        verify_file(checkpoint['checkpoint'])
        for source in checkpoint['provenance']:
            verify_file(source)
    results = None
    integrity = Path(output) / 'report_integrity.json'
    if integrity.is_file():
        artifacts = json.loads(integrity.read_text())['artifacts']
        for artifact in artifacts:
            verify_file(artifact)
        results = len(artifacts)
    return dict(passed=True, protocol_sha256=digest, execution_files=len(files),
                administratively_revised=[f['path'] for f in files if f['state'] != 'unchanged'],
                verified_result_artifacts=results)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        result = verify()
    except (OSError, ValueError, KeyError) as error:
        result = dict(passed=False, error=f'{type(error).__name__}: {error}')
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
