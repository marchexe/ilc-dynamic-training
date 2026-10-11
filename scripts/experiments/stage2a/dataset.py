"""Read-only row-level split verification; never creates or converts data."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from experiments.stage2a.common import file_record
from training.runtime import PROJECT_DIR

SCALARS = ('label', 'jet_e', 'jet_px', 'jet_py', 'jet_pz', 'jet_phi', 'jet_theta')
FLAVORS = ('bb', 'cc', 'dd')


def scalar_rows(path):
    chunks = []
    for batch in pq.ParquetFile(path).iter_batches(columns=list(SCALARS), batch_size=65536):
        values = np.column_stack([batch.column(i).to_numpy() for i in range(len(SCALARS))]).astype('<f8')
        if not np.isfinite(values).all():
            raise ValueError(f'Non-finite split identity columns: {path}')
        chunks.append(values)
    return np.concatenate(chunks)


def byte_rows(values):
    values = np.ascontiguousarray(values)
    return values.view(np.dtype((np.void, values.dtype.itemsize * values.shape[1]))).reshape(-1)


def resolve_splits(manifest, *, root=PROJECT_DIR, inspect_content=True):
    """Check declared source ranges and actual scalar record identity/order.

    Identical scalar records are conservatively treated as overlap. This cannot
    prove physical-event independence when separate jets lack an event ID.
    """
    levels = manifest['levels']
    files, evidence = [], []
    holdout_values = []
    for flavor in FLAVORS:
        records = {}
        for tier in ('full', 'full_holdout', 'monitor', 'control'):
            matches = [r for r in levels[tier]['files'] if r['flavor'] == flavor]
            if len(matches) != 1:
                raise ValueError(f'Expected exactly one {flavor}/{tier} input')
            record = dict(matches[0])
            path = root / record['path']
            rows = pq.ParquetFile(path).metadata.num_rows
            if rows != record['rows']:
                raise ValueError(f'Dataset row count changed: {path}')
            record['path'] = str(path.resolve())
            records[tier] = record
            files.append(dict(tier=tier, flavor=flavor, rows=rows, **file_record(path)))
        source = records['full']
        windows = [records[t] for t in ('full_holdout', 'monitor', 'control')]
        cursor = 0
        for record in windows:
            if ((root / record['source']).resolve() != Path(source['path']) or
                    record['source_start_row'] != cursor or
                    record['source_stop_row'] - cursor != record['rows']):
                raise ValueError(f'Overlapping/gapped or inconsistent split: {flavor}')
            cursor = record['source_stop_row']
        if cursor != source['rows']:
            raise ValueError(f'Tail windows do not partition full source: {flavor}')
        if inspect_content:
            full = scalar_rows(source['path'])
            expected_label = {'bb':5,'cc':4,'dd':2}[flavor]
            if not np.all(full[:,0]==expected_label):
                raise ValueError(f'Wrong raw class labels in {flavor} validation source')
            for record in windows:
                actual = scalar_rows(record['path'])
                expected = full[record['source_start_row']:record['source_stop_row']]
                if not np.array_equal(actual, expected):
                    raise ValueError(f'Split content/order differs from declared source window: {record["path"]}')
                if record is windows[0]:
                    holdout_values.append(actual)
        evidence.append(dict(flavor=flavor, source=source['path'],
                             windows=[dict(start=r['source_start_row'], stop=r['source_stop_row']) for r in windows],
                             scalar_window_identity_verified=inspect_content))
    holdout_files = sorted((r for r in files if r['tier'] == 'full_holdout'), key=lambda r:r['path'])
    offset = 0
    for record in holdout_files:
        record['offset'] = offset
        offset += record['rows']
    row_manifest = [dict(path=r['path'], rows=r['rows'], offset=r['offset'], sha256=r['sha256']) for r in holdout_files]
    fingerprint = hashlib.sha256(json.dumps(row_manifest, sort_keys=True).encode()).hexdigest()
    return dict(files=files, holdout_files=holdout_files, rows=offset,
                audit_fingerprint=fingerprint, isolation=evidence), holdout_values


def verify_training_isolation(holdout_values, train_files):
    holdout = np.concatenate(holdout_values)
    held = np.unique(byte_rows(holdout))
    checks = []
    for path in train_files:
        overlap = np.intersect1d(held, byte_rows(scalar_rows(path)), assume_unique=False)
        if len(overlap):
            raise ValueError(f'Training records overlap holdout: {path}; count={len(overlap)}')
        checks.append(dict(**file_record(path), scalar_overlap_rows=0))
    return checks


def verify_evaluation_files(protocol):
    shared=protocol['evaluation']
    actual={str(p.resolve()) for flavor in FLAVORS for p in
            Path(shared['dataset']).glob(f'*_{flavor}_{shared["validation_suffix"]}.parquet')}
    expected={r['path'] for r in protocol['dataset']['holdout_files']}
    if actual!=expected:
        raise ValueError('Evaluation glob no longer resolves to exactly the frozen holdout files')
