"""Versioned train8m Parquet preparation: preflight, resumable conversion, verification.

One source ROOT tree becomes one Parquet file per flavour (the Weaver glob
`*_<flavour>_train8m.parquet` must resolve to exactly that file). Entries are
streamed in order into a hidden partial file, verified, hashed and only then
atomically renamed. A rerun skips flavours whose verified record still matches
and discards partial files from interrupted attempts; a per-flavour flock
prevents concurrent writers. Physics selection is never re-applied or changed:
the source ROOT trees already carry the frozen split/common cuts.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import time

import numpy as np

from experiments.site_config import expand
from experiments.stage2b.common import (PROJECT_DIR, SPEC, digest_json, disk_free, load_json, lock_file,
                                        project_relative, sha256, utc_now, write_json)

FLAVORS = ('bb', 'cc', 'dd')
# Frozen split windows of the 20250711 source (raw Entry$ ranges per flavour file).
SPLITS = {'train8m': dict(role='training', raw_entries=8_000_000, window='Entry$<8000000'),
          'test3000k': dict(role='sealed_test', raw_entries=3_000_000, window='Entry$>=17000000&&Entry$<20000000')}
FROZEN_CLASSES = {'bb': dict(label=5, class_index=0), 'cc': dict(label=4, class_index=1), 'dd': dict(label=2, class_index=2)}
SCALAR_FLOATS = ('jet_e', 'jet_px', 'jet_py', 'jet_pz', 'jet_phi', 'jet_theta')
STATE_DIR = '.stage2b'
# Existing datasets are read-only inputs for Stage 2B.
PROTECTED_DATASETS = ('datasets/20250711_ilc_nnqq_sgv_10m_3cat_parquet', 'datasets/20250218_ilc_nnqq_sgvnew_parquet',
                      'datasets/20250711_ilc_nnqq_sgv_10m_3cat_parquet_smoke_tail', 'datasets/20250711_ilc_nnqq_sgv_10m_3cat_root')


class DatasetError(ValueError):
    """Invalid source, specification or converted output (fail closed)."""


def output_name(flavor, suffix='train8m'):
    return f'ml_flavtag_6cat_{flavor}_{suffix}.parquet'


def source_path(source):
    """Source ROOT file; `${NAME}` roots resolve from the environment or the local site config."""
    return Path(expand(source['path']))


def output_dir(spec):
    return project_relative(spec['output_dir'])


def state_dir(spec):
    return output_dir(spec) / STATE_DIR


def schema_digest(parquet_file):
    return hashlib.sha256(parquet_file.schema_arrow.serialize().to_pybytes()).hexdigest()


def branch_schema(tree):
    return [[key, tree[key].typename] for key in tree.keys()]


def root_metadata(path, tree_name):
    import uproot
    with uproot.open(path) as handle:
        names = sorted({k.split(';')[0] for k, v in handle.items() if getattr(v, 'classname', '') == 'TTree'})
        if names != [tree_name]:
            raise DatasetError(f'Expected exactly one TTree {tree_name!r} in {path}, found {names}')
        tree = handle[tree_name]
        schema = branch_schema(tree)
        return dict(entries=int(tree.num_entries), branches=len(schema),
                    schema_sha256=hashlib.sha256(json.dumps(schema).encode()).hexdigest())


def validate_spec(spec):
    required = ('schema_version', 'name', 'suffix', 'source_dir', 'output_dir', 'tree', 'flavors', 'sources',
                'expected_rows_total', 'source_selection', 'conversion', 'reference', 'disk')
    split = SPLITS.get(spec.get('suffix'), {})
    if split.get('role') == 'training':
        required += ('validation',)
    missing = [key for key in required if key not in spec]
    if missing:
        raise DatasetError(f'Dataset specification is missing: {", ".join(missing)}')
    if spec['schema_version'] != 1 or not split or spec.get('role', 'training') != split['role']:
        raise DatasetError('Unsupported dataset specification (split/role)')
    if spec['source_selection'].get('entry_window') != split['window']:
        raise DatasetError(f'{spec["suffix"]} must keep its frozen Entry$ window {split["window"]}')
    if spec['flavors'] != FROZEN_CLASSES:
        raise DatasetError(f'{spec["suffix"]} must keep the frozen b/c/d classes (labels 5/4/2, class order b,c,d)')
    out = output_dir(spec).resolve()
    for protected in PROTECTED_DATASETS:
        protected = (PROJECT_DIR / protected).resolve()
        if out == protected or protected in out.parents or out in protected.parents:
            raise DatasetError(f'Output must not overlap an existing dataset: {protected}')
    if [s['flavor'] for s in spec['sources']] != list(FLAVORS):
        raise DatasetError('Exactly one bb, cc and dd source is required, in class order')
    for source in spec['sources']:
        if Path(source['path']).name != f'ml_flavtag_6cat_{source["flavor"]}_{spec["suffix"]}.root':
            raise DatasetError(f'Source is not the {source["flavor"]} {spec["suffix"]} split: {source["path"]}')
        if not (isinstance(source['entries'], int) and source['entries'] > 0):
            raise DatasetError('Source entry counts must be positive integers')
        if source['entries'] > spec['source_selection']['raw_entries_per_flavor']:
            raise DatasetError('A selected split cannot exceed its raw Entry$ window')
    if sum(s['entries'] for s in spec['sources']) != spec['expected_rows_total']:
        raise DatasetError('Per-flavour rows do not add up to the declared total')
    conversion = spec['conversion']
    if (conversion['row_group_size'] <= 0 or conversion['chunk_entries'] <= 0
            or conversion['chunk_entries'] % conversion['row_group_size']):
        raise DatasetError('Chunks must be whole multiples of the row-group size')
    if conversion['compression'] not in ('lz4', 'zstd', 'snappy', 'gzip'):
        raise DatasetError('Unsupported Parquet compression')
    if split['role'] == 'training':
        validation = spec['validation']
        if validation['selection_suffix'] not in validation['approved_suffixes']:
            raise DatasetError('Population-selection validation must use an approved isolated split')
        if set(validation['approved_suffixes']) & set(validation['forbidden_suffixes']):
            raise DatasetError('A validation split cannot be both approved and forbidden')
    return spec


def load_spec(path=SPEC):
    spec = validate_spec(load_json(path))
    return spec, digest_json(spec)


def check_validation_suffix(spec, suffix, *, purpose='selection'):
    """Reject validation splits that overlap train8m or are reserved holdout."""
    validation = spec['validation']
    if suffix in validation['forbidden_suffixes']:
        raise DatasetError(f'{suffix} cannot be used with train8m: {validation["forbidden_suffixes"][suffix]}')
    if purpose == 'selection' and suffix not in validation['approved_suffixes']:
        raise DatasetError(f'{suffix} is not an approved population-selection split for train8m')
    return suffix


def verify_sources(spec, *, metadata=root_metadata):
    """Cheap pinned source identity (size, mtime, entries, branch schema)."""
    records = []
    for source in spec['sources']:
        path = source_path(source)
        if not path.is_file():
            raise DatasetError(f'Missing source ROOT file: {path}')
        stat = path.stat()
        if stat.st_size != source['size_bytes'] or stat.st_mtime_ns != source['mtime_ns']:
            raise DatasetError(f'Source ROOT file changed since the specification was frozen: {path}')
        actual = metadata(path, spec['tree'])
        for key in ('entries', 'branches', 'schema_sha256'):
            if actual[key] != source[key]:
                raise DatasetError(f'Source {key} disagrees with the specification: {path}')
        records.append(dict(source, verified_at=utc_now()))
    return records


def required_free_bytes(spec, completed):
    remaining = sum(s['entries'] for s in spec['sources'] if s['flavor'] not in completed)
    return int(remaining * spec['disk']['estimated_bytes_per_row'] * spec['disk']['safety_factor']
               + spec['disk']['minimum_free_bytes_after'])


def preflight(spec, *, metadata=root_metadata, free=disk_free):
    """Read-only checks before conversion; never creates data."""
    sources = verify_sources(spec, metadata=metadata)
    status = dataset_status(spec)
    needed = required_free_bytes(spec, status['completed'])
    available = free(output_dir(spec))
    if available < needed:
        raise OSError(28, f'Insufficient disk for {spec["suffix"]}: need {needed} bytes, {available} free')
    reference = project_relative(spec['reference']['parquet'])
    if not reference.is_file():
        raise DatasetError(f'Missing reference schema file: {reference}')
    return dict(sources=sources, status=status, free_bytes=available, required_free_bytes=needed)


def _record_path(spec, flavor):
    return state_dir(spec) / f'{flavor}.json'


def _quick_valid(spec, record, flavor):
    final = output_dir(spec) / output_name(flavor, spec['suffix'])
    return (record.get('status') == 'verified' and record.get('spec_name') == spec['name']
            and record.get('flavor') == flavor and final.is_file()
            and final.stat().st_size == record['size_bytes'] and final.stat().st_mtime_ns == record['mtime_ns'])


def read_record(spec, flavor):
    try:
        return load_json(_record_path(spec, flavor))
    except (OSError, ValueError):
        return None


def dataset_status(spec):
    completed, partial = [], []
    for flavor in FLAVORS:
        record = read_record(spec, flavor)
        if record and _quick_valid(spec, record, flavor):
            completed.append(flavor)
        elif (output_dir(spec) / output_name(flavor, spec['suffix'])).exists() or any(
                output_dir(spec).glob(f'.{output_name(flavor, spec["suffix"])}.partial-*')):
            partial.append(flavor)
    state = 'complete' if len(completed) == len(FLAVORS) else 'partial' if completed or partial else 'absent'
    return dict(state=state, completed=completed, partial=partial)


def verify_parquet(path, spec, flavor, expected_rows):
    """Row count, row-group layout, schema identity and label content."""
    import pyarrow as pa
    try:
        return _verify_parquet(path, spec, flavor, expected_rows)
    except (pa.ArrowException, OSError) as error:
        raise DatasetError(f'{path}: unreadable Parquet ({error})') from error


def _verify_parquet(path, spec, flavor, expected_rows):
    import pyarrow.parquet as pq
    handle = pq.ParquetFile(path)
    meta, size = handle.metadata, spec['conversion']['row_group_size']
    if meta.num_rows != expected_rows:
        raise DatasetError(f'{path}: {meta.num_rows} rows, expected {expected_rows}')
    groups = [meta.row_group(i).num_rows for i in range(meta.num_row_groups)]
    if any(rows != size for rows in groups[:-1]) or not 0 < groups[-1] <= size:
        raise DatasetError(f'{path}: unexpected row-group layout')
    if schema_digest(handle) != spec['reference']['schema_sha256']:
        raise DatasetError(f'{path}: schema differs from the Weaver-compatible reference conversion')
    label = spec['flavors'][flavor]['label']
    for batch in handle.iter_batches(columns=['label'], batch_size=1_000_000):
        if not np.all(batch.column(0).to_numpy() == label):
            raise DatasetError(f'{path}: wrong class label (expected {label} only)')
    return dict(rows=meta.num_rows, row_groups=meta.num_row_groups)


def _write_progress(spec, flavor, **values):
    write_json(state_dir(spec) / 'progress' / f'{flavor}.json', dict(flavor=flavor, updated_at=utc_now(), **values))


def _chunks(spec, source, flavor, started, *, stop=None):
    """Yield entry-ordered chunks with fail-fast integrity checks."""
    import awkward as ak
    import uproot
    chunk, total = spec['conversion']['chunk_entries'], source['entries']
    label = spec['flavors'][flavor]['label']
    with uproot.open(source_path(source)) as handle:
        tree = handle[spec['tree']]
        if tree.num_entries != total:
            raise DatasetError(f'Source entries changed during conversion: {source["path"]}')
        for start in range(0, total, chunk):
            if stop is not None and stop.is_set():
                raise InterruptedError('Conversion interrupted')
            stop_entry = min(total, start + chunk)
            arrays = tree.arrays(entry_start=start, entry_stop=stop_entry)
            if len(arrays) != stop_entry - start:
                raise DatasetError(f'Short read at entries {start}-{stop_entry}: {source["path"]}')
            if not ak.all(arrays['label'] == label):
                raise DatasetError(f'Wrong class label in {source["path"]} entries {start}-{stop_entry}')
            for name in SCALAR_FLOATS:
                if not np.isfinite(ak.to_numpy(arrays[name])).all():
                    raise DatasetError(f'Non-finite {name} in {source["path"]} entries {start}-{stop_entry}')
            yield arrays
            elapsed = time.monotonic() - started
            rate = stop_entry / elapsed if elapsed > 0 else 0.0
            _write_progress(spec, flavor, phase='converting', rows_done=stop_entry, rows_total=total,
                            rate_rows_per_second=rate, eta_seconds=(total - stop_entry) / rate if rate else None)


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def convert_flavor(spec, flavor, *, stop=None, write=None, hash_file=sha256):
    """Convert, verify and publish one flavour. Idempotent; resumable per flavour."""
    import awkward as ak
    write = write or ak.to_parquet_row_groups
    source = next(s for s in spec['sources'] if s['flavor'] == flavor)
    out = output_dir(spec)
    final = out / output_name(flavor, spec['suffix'])
    state_dir(spec).mkdir(parents=True, exist_ok=True)
    with lock_file(state_dir(spec) / f'{flavor}.lock'):
        record = read_record(spec, flavor)
        if record and _quick_valid(spec, record, flavor):
            return dict(record, action='skipped_verified')
        for stale in out.glob(f'.{output_name(flavor, spec["suffix"])}.partial-*'):
            stale.unlink()
        if final.exists():
            # Published without a valid record: re-verify fully, otherwise keep as evidence.
            try:
                return _publish_record(spec, flavor, source, final, hash_file, action='reverified_existing')
            except DatasetError:
                final.rename(out / f'.{output_name(flavor, spec["suffix"])}.invalid-{int(time.time())}')
        started = time.monotonic()
        _write_progress(spec, flavor, phase='hashing_source', rows_done=0, rows_total=source['entries'])
        source_sha256 = hash_file(source_path(source))
        partial = out / f'.{output_name(flavor, spec["suffix"])}.partial-{socket.gethostname().split(".")[0]}-{os.getpid()}'
        conversion = spec['conversion']
        try:
            write(_chunks(spec, source, flavor, time.monotonic(), stop=stop), str(partial),
                  compression=conversion['compression'], row_group_size=conversion['row_group_size'])
            with partial.open('rb') as stream:
                os.fsync(stream.fileno())
            _write_progress(spec, flavor, phase='verifying', rows_done=source['entries'], rows_total=source['entries'])
            verify_parquet(partial, spec, flavor, source['entries'])
            stat = source_path(source).stat()
            if stat.st_size != source['size_bytes'] or stat.st_mtime_ns != source['mtime_ns']:
                raise DatasetError(f'Source changed during conversion: {source["path"]}')
            os.replace(partial, final)
            _fsync_dir(out)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        record = _publish_record(spec, flavor, source, final, hash_file, action='converted',
                                 source_sha256=source_sha256, elapsed=time.monotonic() - started)
        _write_progress(spec, flavor, phase='done', rows_done=source['entries'], rows_total=source['entries'])
        return record


def _publish_record(spec, flavor, source, final, hash_file, *, action, source_sha256=None, elapsed=None):
    layout = verify_parquet(final, spec, flavor, source['entries'])
    stat = final.stat()
    record = dict(status='verified', spec_name=spec['name'], flavor=flavor, path=str(final.resolve()),
                  size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=hash_file(final),
                  schema_sha256=spec['reference']['schema_sha256'], **layout,
                  source=dict(path=source['path'], size_bytes=source['size_bytes'], mtime_ns=source['mtime_ns'],
                              entries=source['entries'], sha256=source_sha256),
                  conversion=spec['conversion'], verified_at=utc_now(), elapsed_seconds=elapsed, action=action)
    write_json(_record_path(spec, flavor), record)
    return record


def audit_fingerprint(files):
    """Weaver's data_audit.parquet_manifest fingerprint from recorded hashes."""
    records, offset = [], 0
    for item in sorted(files, key=lambda f: f['path']):
        records.append(dict(path=item['path'], rows=item['rows'], offset=offset, sha256=item['sha256']))
        offset += item['rows']
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(), records, offset


def verify_dataset(spec, *, rehash=True, hash_file=sha256):
    """Full verification of the published dataset; returns the realized manifest."""
    out = output_dir(spec)
    files = []
    for flavor in FLAVORS:
        record = read_record(spec, flavor)
        if not record or not _quick_valid(spec, record, flavor):
            raise DatasetError(f'{spec["suffix"]} {flavor} file is missing, incomplete or changed')
        matches = sorted(p for p in out.glob(f'*_{flavor}_{spec["suffix"]}.parquet'))
        if [p.resolve() for p in matches] != [Path(record['path'])]:
            raise DatasetError(f'Weaver glob for {flavor} must resolve to exactly one verified file: {matches}')
        if rehash and hash_file(Path(record['path'])) != record['sha256']:
            raise DatasetError(f'Checksum mismatch: {record["path"]}')
        files.append(record)
    unexpected = sorted(p.name for p in out.glob('*.parquet') if p.resolve() not in {Path(r['path']) for r in files})
    if unexpected:
        raise DatasetError(f'Unexpected Parquet files in the {spec["suffix"]} directory: {unexpected}')
    rows = {r['flavor']: r['rows'] for r in files}
    if sum(rows.values()) != spec['expected_rows_total']:
        raise DatasetError('Converted rows disagree with the frozen expectation')
    fingerprint, audit_files, total = audit_fingerprint(files)
    return dict(schema_version=1, name=spec['name'], created_at=utc_now(), data_extension='parquet',
                spec_sha256=digest_json(spec), local_path=spec['output_dir'], role=spec.get('role', 'training'),
                suffix=spec['suffix'],
                flavors=list(FLAVORS), rows_by_flavor=rows, rows_total=total,
                size_bytes_total=sum(r['size_bytes'] for r in files),
                files=[dict(flavor=r['flavor'], path=str(Path(r['path']).relative_to(PROJECT_DIR)) if Path(r['path']).is_relative_to(PROJECT_DIR) else r['path'],
                            resolved_path=r['path'], rows=r['rows'], row_groups=r['row_groups'], size_bytes=r['size_bytes'],
                            sha256=r['sha256'], source_root_path=r['source']['path'],
                            source_root_size_bytes=r['source']['size_bytes'], source_root_sha256=r['source']['sha256'],
                            source_entries=r['source']['entries']) for r in files],
                weaver_audit=dict(fingerprint=fingerprint, files=audit_files, total_rows=total),
                validation=spec.get('validation'), tracked_in_git=False,
                reason_not_tracked='Large Parquet data; keep only this manifest in Git.')


def portable_manifest(manifest):
    """Publishable manifest: hashes, rows and relative paths, no machine-specific locations."""
    files = [{k: v for k, v in f.items() if k not in ('resolved_path', 'source_root_path')}
             | dict(source_root_file=Path(f['source_root_path']).name) for f in manifest['files']]
    audit = {k: v for k, v in manifest['weaver_audit'].items() if k != 'files'}
    return dict(manifest, files=files, weaver_audit=audit, portable=True,
                local_full_manifest=f'{manifest["local_path"]}/MANIFEST.json')


def publish_manifest(spec, manifest):
    """Full manifest beside the data (local); portable manifest in datasets/manifests/."""
    write_json(output_dir(spec) / 'MANIFEST.json', manifest)
    path = PROJECT_DIR / 'datasets/manifests' / f'{spec["name"]}.json'
    write_json(path, portable_manifest(manifest))
    return path


def build_spec(*, name, source_dir, output, macro, reference, metadata=root_metadata, suffix='train8m'):
    """Freeze source identities and split facts (read-only metadata, no conversion)."""
    import pyarrow.parquet as pq
    split = SPLITS[suffix]
    source_dir = Path(source_dir)
    sources = []
    for flavor in FLAVORS:
        path = source_dir / f'ml_flavtag_6cat_{flavor}_{suffix}.root'
        stat = path.stat()
        sources.append(dict(flavor=flavor, path=str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                            **metadata(path, 'ntp')))
    macro = Path(macro)
    text = macro.read_text()
    cuts = dict(zip(re.findall(r'"(_[a-z0-9]+)"', text), re.findall(r'"(Entry\$[^"]*)"', text)))
    common = re.search(r'TCut commoncut\("([^"]+)"\)', text)
    window = cuts.get('_' + suffix)
    if window != split['window'] or not common:
        raise DatasetError(f'Unexpected {suffix} split macro')
    reference = project_relative(reference)
    spec = dict(
        schema_version=1, name=name, suffix=suffix, role=split['role'], source_dir=str(source_dir), output_dir=output,
        tree='ntp', flavors=FROZEN_CLASSES, sources=sources,
        expected_rows_total=sum(s['entries'] for s in sources),
        source_selection=dict(macro=str(macro), macro_sha256=sha256(macro), entry_window=window,
                              raw_entries_per_flavor=split['raw_entries'], common_cut=common.group(1), split_cuts=cuts,
                              note='The ROOT split files already contain these cuts; conversion copies rows 1:1.'),
        conversion=dict(engine='awkward.to_parquet_row_groups', branches='all', compression='lz4',
                        row_group_size=10_000, chunk_entries=100_000,
                        row_group_size_rationale=('train800k used 1000-row groups (799 per file, 12 MB footer). '
                                                  'At 8M rows that is 8000 groups and a ~113 MB footer that Weaver '
                                                  'reparses on every fetch (~1.2 s each). 10000 keeps ~800 groups per '
                                                  'file; content, order and Weaver row IDs are layout-independent.')),
        reference=dict(parquet=str(reference.relative_to(PROJECT_DIR)),
                       schema_sha256=schema_digest(pq.ParquetFile(reference))),
        disk=dict(estimated_bytes_per_row=2700, safety_factor=1.10, minimum_free_bytes_after=20 * 1024**3,
                  estimate_basis='measured 2675 bytes/row for an lz4 test conversion of train8m entries'),
    )
    if split['role'] != 'training':
        return spec
    return dict(spec,
        validation=dict(dataset='datasets/20250711_ilc_nnqq_sgv_10m_3cat_parquet', selection_suffix='val50k_tail',
                        approved_suffixes=['val50k_tail', 'val5k_tail'],
                        forbidden_suffixes={
                            'val5k': 'original 100k-ladder split (Entry$ 80000-85000) lies inside the train8m window',
                            'val50k': 'original 1m-ladder split lies inside the train8m window',
                            'val_holdout': 'Stage 2A holdout; never used for training-time selection',
                            'val1000k': 'contains the Stage 2A holdout rows',
                            'train800k': 'training data (prefix of train8m)'}),
    )
