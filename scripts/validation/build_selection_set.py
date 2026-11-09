#!/usr/bin/env python3
"""Build the 1M-jet selection set (sel1m_tail) and the matching 8M training input (train8m_head).

    .venv/bin/python scripts/validation/build_selection_set.py [--rows-per-flavor 333333] [--no-head] [--force]

The validation set that ranks members and picks checkpoints (val50k_tail, 150k jets) is too small: over two
seeds, validation and holdout disagreed on whether the fast PBT or the fixed-LR grid is better. Every row of
val1000k already has a role (Stage 2A holdout, val50k_tail, val5k_tail), and test3000k is the sealed Stage 3
test, so the selection set comes from train8m:

* sel1m_tail: the last N rows of each train8m flavor file.
* train8m_head: the other rows, for 8M runs that select on sel1m_tail.

train800k is the first rows of each train8m file (checked here on the scalar columns of every train800k row),
so train800k runs never train on sel1m_tail. train8m and val1000k come from different source files, so
sel1m_tail does not overlap the holdout, val50k_tail or val5k_tail either.
"""
import argparse
import hashlib
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.runtime import FLAVOR_SAMPLE_GROUPS, PROJECT_DIR, atomic_json, utc_now  # noqa: E402
from validation.tail_proxy_subsets import display_path  # noqa: E402

TRAIN8M = PROJECT_DIR / 'datasets/20250711_ilc_nnqq_sgv_10m_3cat_train8m_parquet'
TRAIN800K = PROJECT_DIR / 'datasets/20250711_ilc_nnqq_sgv_10m_3cat_parquet'
OUTPUT = PROJECT_DIR / 'datasets/20250711_ilc_nnqq_sgv_10m_3cat_sel1m_parquet'
MANIFEST = PROJECT_DIR / 'datasets/manifests/20250711_ilc_nnqq_sgv_10m_3cat_sel1m_v1.json'
NAME = '20250711_ilc_nnqq_sgv_10m_3cat_sel1m_v1'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def scalar_columns(schema):
    return [f.name for f in schema if not (pa.types.is_list(f.type) or pa.types.is_large_list(f.type))]


def copy_rows(source, output, start, stop, *, force):
    """Stream rows [start, stop) of a parquet file into a new lz4 parquet file."""
    if output.exists() and not force:
        raise FileExistsError(f'{output} exists; use --force')
    reader = pq.ParquetFile(source)
    position, written = 0, 0
    with pq.ParquetWriter(output, reader.schema_arrow, compression='lz4') as writer:
        for index in range(reader.metadata.num_row_groups):
            rows = reader.metadata.row_group(index).num_rows
            lo, hi = max(start, position), min(stop, position + rows)
            if lo < hi:
                table = reader.read_row_group(index).slice(lo - position, hi - lo)
                writer.write_table(table, row_group_size=10000)
                written += table.num_rows
            position += rows
            if position >= stop:
                break
    if written != stop - start:
        raise RuntimeError(f'{output}: wrote {written} rows, expected {stop - start}')
    return dict(path=display_path(output), source=display_path(source), source_start_row=start,
                source_stop_row=stop, rows=written, row_groups=pq.ParquetFile(output).metadata.num_row_groups,
                size_bytes=output.stat().st_size, sha256=sha256(output))


def train800k_is_prefix(flavor):
    """Every train800k row equals the train8m row with the same index on all scalar columns."""
    small = pq.ParquetFile(next(TRAIN800K.glob(f'*_{flavor}_train800k.parquet')))
    large = pq.ParquetFile(next(TRAIN8M.glob(f'*_{flavor}_train8m.parquet')))
    columns = scalar_columns(small.schema_arrow)
    rows = small.metadata.num_rows
    left = small.read(columns=columns)
    right = large.read_row_groups(range(-(-rows // large.metadata.row_group(0).num_rows)), columns=columns).slice(0, rows)
    return dict(rows=rows, columns=columns, equal=left.equals(right))


def build(rows_per_flavor, *, head=True, force=False):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    levels = dict(selection=dict(role='checkpoint and member selection', suffix='sel1m_tail', files=[]),
                  training=dict(role='train8m without the selection rows', suffix='train8m_head', files=[]))
    checks = {}
    for _, flavor in FLAVOR_SAMPLE_GROUPS:
        source = next(TRAIN8M.glob(f'*_{flavor}_train8m.parquet'))
        total = pq.ParquetFile(source).metadata.num_rows
        prefix = train800k_is_prefix(flavor)
        if not prefix['equal']:
            raise RuntimeError(f'train800k {flavor} is not the prefix of train8m')
        start = total - rows_per_flavor
        if start < prefix['rows']:
            raise RuntimeError(f'{flavor}: the selection rows would overlap train800k')
        stem = source.name.removesuffix('_train8m.parquet')
        record = copy_rows(source, OUTPUT / f'{stem}_sel1m_tail.parquet', start, total, force=force)
        levels['selection']['files'].append(dict(flavor=flavor, **record))
        if head:
            record = copy_rows(source, OUTPUT / f'{stem}_train8m_head.parquet', 0, start, force=force)
            levels['training']['files'].append(dict(flavor=flavor, **record))
        checks[flavor] = dict(train800k_rows=prefix['rows'], train800k_prefix_of_train8m=True,
                              compared_columns=prefix['columns'], selection_start_row=start, train8m_rows=total)
        print(flavor, 'done', flush=True)
    for level in levels.values():
        level['rows_total'] = sum(f['rows'] for f in level['files'])
    manifest = dict(schema_version=1, name=NAME, created_at=utc_now(), dataset=display_path(OUTPUT),
                    source=display_path(TRAIN8M), rows_per_flavor=rows_per_flavor, levels=levels, checks=checks,
                    disjoint_from=dict(train800k='prefix of train8m, checked row by row on scalar columns',
                                       val1000k='different source files (holdout, val50k_tail, val5k_tail)',
                                       test3000k='different source files (sealed Stage 3 test)'),
                    tracked_in_git=False, reason_not_tracked='Parquet files; keep only this manifest in Git.')
    atomic_json(MANIFEST, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rows-per-flavor', type=int, default=333333)
    parser.add_argument('--no-head', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    manifest = build(args.rows_per_flavor, head=not args.no_head, force=args.force)
    print(MANIFEST, {k: v['rows_total'] for k, v in manifest['levels'].items()})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
