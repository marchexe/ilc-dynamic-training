"""Read-only provenance and split-isolation evidence for train8m.

Records are compared by their scalar identity (label and six jet kinematics),
exactly as Stage 2A did for the holdout. Identical scalar records are
conservatively treated as overlap. Without a stored physical event ID this
cannot prove event-level independence between different jets of one event.
"""
from pathlib import Path
import re

import numpy as np

from experiments.stage2a.dataset import SCALARS, byte_rows, scalar_rows
from experiments.stage2b.common import PROJECT_DIR, load_json, sha256, utc_now
from experiments.stage2b.dataset import FLAVORS, check_validation_suffix, output_dir, output_name, root_metadata

def root_scalars(path, tree='ntp', entry_stop=None):
    import uproot
    with uproot.open(path) as handle:
        arrays = handle[tree].arrays(list(SCALARS), entry_stop=entry_stop, library='np')
    values = np.column_stack([arrays[name] for name in SCALARS]).astype('<f8')
    if not np.isfinite(values).all():
        raise ValueError(f'Non-finite split identity columns: {path}')
    return values


def overlap_count(left_unique, values):
    return int(len(np.intersect1d(left_unique, byte_rows(values), assume_unique=False)))


def macro_windows(source_dir):
    windows = {}
    for macro in sorted(Path(source_dir).glob('divide_energy_*.C')):
        text = macro.read_text(errors='replace')
        suffixes = re.findall(r'"(_[a-z0-9]+)"', text)
        cuts = re.findall(r'"(Entry\$[^"]*)"', text)
        for suffix, cut in zip(suffixes, cuts):
            windows.setdefault(suffix[1:], []).append(dict(macro=macro.name, cut=cut, sha256=sha256(macro)))
    return windows


def verify_root_provenance(spec, *, readers=None):
    """Counts, schemas, labels, prefix relation and isolation from the ROOT sources."""
    readers = readers or {}
    scalars = readers.get('scalars', root_scalars)
    metadata = readers.get('metadata', root_metadata)
    parquet_scalars = readers.get('parquet_scalars', scalar_rows)
    source_dir = Path(spec['source_dir'])
    train_parquet = PROJECT_DIR / spec['validation']['dataset']
    tail = load_json(PROJECT_DIR / 'datasets/manifests/20250711_ilc_nnqq_sgv_10m_3cat_tail_proxy_v1.json')
    flavors, files = [], []
    for flavor in FLAVORS:
        label = spec['flavors'][flavor]['label']
        paths = {s: source_dir / f'ml_flavtag_6cat_{flavor}_{s}.root'
                 for s in ('train8m', 'train800k', 'val1000k', 'val50k', 'val5k')}
        meta = {s: metadata(p, spec['tree']) for s, p in paths.items()}
        for split, item in meta.items():
            files.append(dict(flavor=flavor, split=split, path=str(paths[split]),
                              size_bytes=paths[split].stat().st_size, **item))
        if len({m['schema_sha256'] for m in meta.values()}) != 1:
            raise ValueError(f'{flavor}: branch schemas differ across splits')
        train = scalars(paths['train8m'])
        if len(train) != meta['train8m']['entries']:
            raise ValueError(f'{flavor}: train8m scalar read is incomplete')
        labels = sorted(np.unique(train[:, 0]).astype(int).tolist())
        if labels != [label]:
            raise ValueError(f'{flavor}: train8m labels {labels}, expected [{label}]')
        train_bytes = byte_rows(train)
        train_unique = np.unique(train_bytes)
        # train800k Parquet (used by every production run) is the exact prefix of train8m.
        previous = parquet_scalars(train_parquet / f'ml_flavtag_6cat_{flavor}_train800k.parquet')
        prefix = bool(np.array_equal(previous, train[:len(previous)]))
        if not prefix:
            raise ValueError(f'{flavor}: train800k is not the ordered prefix of train8m')
        val = scalars(paths['val1000k'])
        isolation = overlap_count(train_unique, val)
        if isolation:
            raise ValueError(f'{flavor}: train8m shares {isolation} scalar records with val1000k')
        # Tail windows used for selection/holdout must remain exact val1000k windows.
        windows = {}
        for tier in ('full_holdout', 'monitor', 'control'):
            record = next(r for r in tail['levels'][tier]['files'] if r['flavor'] == flavor)
            windows[tail['levels'][tier]['suffix']] = dict(start=record['source_start_row'], stop=record['source_stop_row'],
                                                         rows=record['rows'])
        legacy_values = {split: scalars(paths[split]) for split in ('val50k', 'val5k')}
        legacy = {split: overlap_count(train_unique, values) for split, values in legacy_values.items()}
        previous_unique = np.unique(byte_rows(previous))
        legacy_previous = {split: overlap_count(previous_unique, values) for split, values in legacy_values.items()}
        flavors.append(dict(
            flavor=flavor, label=label, train8m_rows=len(train), train8m_unique_scalar_records=int(len(train_unique)),
            train8m_duplicate_scalar_records=int(len(train) - len(train_unique)), labels=labels,
            train800k_rows=len(previous), train800k_is_ordered_prefix=prefix,
            val1000k_rows=len(val), train8m_val1000k_scalar_overlap=isolation, tail_windows_in_val1000k=windows,
            legacy_overlap_with_train8m=dict((k, dict(rows=meta[k]['entries'], overlapping=v)) for k, v in legacy.items()),
            legacy_overlap_with_train800k=dict((k, dict(rows=meta[k]['entries'], overlapping=v))
                                               for k, v in legacy_previous.items())))
    for suffix in spec['validation']['forbidden_suffixes']:
        try:
            check_validation_suffix(spec, suffix)
        except ValueError:
            continue
        raise ValueError(f'Forbidden suffix {suffix} was accepted')
    return dict(schema_version=1, verified_at=utc_now(), spec_name=spec['name'], files=files, flavors=flavors,
                macro_windows=macro_windows(source_dir), totals=dict(
                    train8m_rows=sum(f['train8m_rows'] for f in flavors),
                    train800k_rows=sum(f['train800k_rows'] for f in flavors),
                    val1000k_rows=sum(f['val1000k_rows'] for f in flavors),
                    train8m_val1000k_scalar_overlap=sum(f['train8m_val1000k_scalar_overlap'] for f in flavors)),
                identity_columns=list(SCALARS), physical_event_ids_available=False)


def verify_realized_isolation(spec, *, parquet_scalars=scalar_rows):
    """After conversion: Parquet content equals the ROOT-verified facts."""
    validation_dir = PROJECT_DIR / spec['validation']['dataset']
    checks = []
    for flavor in FLAVORS:
        converted = parquet_scalars(output_dir(spec) / output_name(flavor))
        previous = parquet_scalars(validation_dir / f'ml_flavtag_6cat_{flavor}_train800k.parquet')
        unique = np.unique(byte_rows(converted))
        overlaps = {suffix: overlap_count(unique, parquet_scalars(validation_dir / f'ml_flavtag_6cat_{flavor}_{suffix}.parquet'))
                    for suffix in ('val50k_tail', 'val5k_tail', 'val_holdout')}
        check = dict(flavor=flavor, rows=len(converted),
                     labels=sorted(np.unique(converted[:, 0]).astype(int).tolist()),
                     train800k_is_ordered_prefix=bool(np.array_equal(previous, converted[:len(previous)])),
                     scalar_overlap=overlaps)
        if (check['labels'] != [spec['flavors'][flavor]['label']] or not check['train800k_is_ordered_prefix']
                or any(overlaps.values())):
            raise ValueError(f'Converted {flavor} train8m failed isolation/provenance: {check}')
        checks.append(check)
    return dict(verified_at=utc_now(), checks=checks)


def markdown(evidence, spec):
    lines = ['# train8m provenance and split verification', '',
             f'Specification: `{spec["name"]}`; verified {evidence["verified_at"]}. Read-only; no conversion.', '',
             '| Flavour | Label | train8m rows | Unique scalar records | train800k prefix | val1000k overlap | '
             'Legacy val50k overlap | Legacy val5k overlap |', '|---|---:|---:|---:|---|---:|---:|---:|']
    for f in evidence['flavors']:
        legacy = f['legacy_overlap_with_train8m']
        lines.append(f'| {f["flavor"]} | {f["label"]} | {f["train8m_rows"]:,} | {f["train8m_unique_scalar_records"]:,} | '
                     f'{"exact" if f["train800k_is_ordered_prefix"] else "NO"} ({f["train800k_rows"]:,}) | '
                     f'{f["train8m_val1000k_scalar_overlap"]} | {legacy["val50k"]["overlapping"]:,}/{legacy["val50k"]["rows"]:,} | '
                     f'{legacy["val5k"]["overlapping"]:,}/{legacy["val5k"]["rows"]:,} |')
    totals = evidence['totals']
    lines += ['', f'Totals: train8m {totals["train8m_rows"]:,}; train800k {totals["train800k_rows"]:,}; '
              f'val1000k {totals["val1000k_rows"]:,}; train8m/val1000k scalar overlap {totals["train8m_val1000k_scalar_overlap"]}.', '',
              '## Entry windows from the source split macros', '']
    for suffix in ('train8m', 'val500k', 'test1500k', 'train16m', 'val1000k', 'test3000k', 'train80k', 'val5k', 'test15k'):
        for item in evidence['macro_windows'].get(suffix, []):
            lines.append(f'- `{suffix}`: `{item["cut"]}` ({item["macro"]})')
    lines += ['', 'No macro for the original `train800k`/`val50k`/`test150k` ladder is readable in the source '
              'directory; their placement is established from content: train800k is the ordered prefix of train8m, '
              'and every legacy val50k/val5k record also appears in train8m.', '',
              '## Consequences', '',
              '- train8m (Entry$ < 8M of each flavour file) and val1000k (16M-17M) are disjoint windows; '
              'no identical scalar record is shared. val50k_tail, val5k_tail and val_holdout are exact windows '
              'of val1000k (Stage 2A verified their content and order).',
              '- Legacy `val5k`/`val50k` lie inside train8m and are rejected for train8m selection. '
              '`val_holdout`/`val1000k` are reserved for independent evaluation.',
              '- All production train800k rows are contained in train8m (as its first rows per flavour).',
              '- Historical note (outside Stage 2B): legacy `val5k` is also inside train800k '
              f'({sum(f["legacy_overlap_with_train800k"]["val5k"]["overlapping"] for f in evidence["flavors"]):,} shared records), '
              'legacy `val50k` is disjoint from train800k '
              f'({sum(f["legacy_overlap_with_train800k"]["val50k"]["overlapping"] for f in evidence["flavors"]):,} shared). '
              'Current production configurations select on `val50k_tail`.',
              '- ROOT entries are jets, not physical events. No event identifier is stored (`event_nbh` is 0), so '
              'event-level lineage cannot be verified; contiguous Entry$ windows keep jets of one event together '
              'only if the source stores them in adjacent entries (unverifiable assumption).',
              f'- Identity columns: {", ".join(evidence["identity_columns"])}.']
    return '\n'.join(lines) + '\n'
