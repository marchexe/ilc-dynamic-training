"""Shared checkpoint bundles, preserving the historical copy semantics.

No serialization or optimizer transformation occurs in copy_bundle. The scaler
companion follows the optimizer exactly as in the reference implementation.
"""
import os
import json
import re
import shutil
import tempfile
from pathlib import Path

from training.runtime import atomic_json, sha256


def copy_optimizer_companion(source, destination):
    """Keep the optional AMP scaler with copied optimizer state, including
    removal of a stale recipient scaler when restoring a legacy checkpoint."""
    source, destination = Path(source), Path(destination)
    if not source.name.endswith("_optimizer.pt"):
        return
    src = source.with_name(source.name.removesuffix("_optimizer.pt") + "_scaler.pt")
    dst = destination.with_name(destination.name.removesuffix("_optimizer.pt") + "_scaler.pt")
    if src.resolve() == dst.resolve():
        return
    if src.is_file():
        atomic_copy(src, dst)
    else:
        dst.unlink(missing_ok=True)


def atomic_copy(source, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".pbt-tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    copy_optimizer_companion(source, destination)


def atomic_copy_pair(pairs):
    """Stage all sources before replacing destinations in the original order.

    Every source is staged to a temporary file first; only once every
    staging copy has succeeded are the temp files committed in place via
    os.replace. Staging failures leave destinations untouched; replacements
    are sequential, not a filesystem transaction. The optional scaler follows
    afterward. Preserve this ordering for historical resume/replay behavior.
    """
    staged = []
    try:
        for source, destination in pairs:
            destination = Path(destination)
            temporary = destination.with_suffix(destination.suffix + ".pbt-tmp")
            shutil.copy2(source, temporary)
            staged.append((temporary, destination))
    except BaseException:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        raise
    for temporary, destination in staged:
        os.replace(temporary, destination)
    for source, destination in pairs:
        copy_optimizer_companion(source, destination)


def checkpoint_paths(member_dir, epoch):
    prefix = member_dir / f"net_epoch-{epoch}"
    return Path(f"{prefix}_state.pt"), Path(f"{prefix}_optimizer.pt")


def bundle_paths(directory, epoch):
    state, optimizer = checkpoint_paths(Path(directory), epoch)
    return dict(state=state, optimizer=optimizer, scaler=optimizer.with_name(optimizer.name.replace('_optimizer.pt', '_scaler.pt')))


def bundle_identity(paths):
    return {part: dict(path=str(path), sha256=sha256(path)) for part, path in paths.items()}


def check_bundle(identity):
    for item in identity.values():
        if sha256(Path(item['path'])) != item['sha256']:
            raise ValueError(f"Checkpoint identity changed: {item['path']}")


def copy_bundle(source, destination):
    # atomic_copy_pair stages model/optimizer and propagates the scaler companion.
    destination['state'].parent.mkdir(parents=True, exist_ok=True)
    atomic_copy_pair([(source[k], destination[k]) for k in ('state', 'optimizer')])
    copied = bundle_identity(destination)
    if any(copied[k]['sha256'] != sha256(source[k]) for k in source):
        raise ValueError('Checkpoint bundle copy failed identity verification')
    return copied


def selected_best_bundle(record, *, require_scaler=False, resolve=Path):
    """Validate a selected archive, including legacy records without identities.

    Never repair an archive using mutable member checkpoints: exploit may have
    overwritten that member's files at the very same epoch. Legacy missing AMP
    state remains explicitly missing; callers needing a resume bundle reject it.
    """
    paths = {part: resolve(record[part + '_path']) for part in ('state', 'optimizer')}
    scaler = resolve(record['scaler_path']) if record.get('scaler_path') else paths['optimizer'].with_name(
        paths['optimizer'].name.removesuffix('_optimizer.pt') + '_scaler.pt')
    if scaler.is_file():
        paths['scaler'] = scaler
    elif require_scaler or record.get('scaler_path'):
        raise FileNotFoundError(f'Selected-best checkpoint is missing AMP scaler: {scaler}')
    identity = record.get('checkpoint')
    if record.get('bundle_schema_version') is not None:
        if record['bundle_schema_version'] != 1 or not identity:
            raise ValueError('Invalid selected-best bundle identity')
        if record.get('source_controller_path'):
            paths['controller'] = resolve(record['controller_path'])
        if set(paths) != set(identity):
            raise ValueError('Selected-best checkpoint components disagree with metadata')
        metadata = json.loads(resolve(record['metadata_path']).read_text())
        for key in ('bundle_schema_version', 'generation', 'epoch', 'member', 'checkpoint', 'scaler_status'):
            if metadata.get(key) != record.get(key):
                raise ValueError(f'Selected-best metadata disagrees: {key}')
    actual = bundle_identity(paths)
    if identity:
        if set(actual) != set(identity) or any(
                actual[k]['sha256'] != identity[k]['sha256'] or resolve(identity[k]['path']).resolve() != paths[k].resolve()
                for k in actual):
            raise ValueError('Selected-best checkpoint identity changed')
    return actual


def archive_selected_best(checkpoint_dir, sources, record, *, require_scaler,
                          optimizer_transform=None):
    """Publish an immutable, verified archive by one directory rename.

    The manifest is the selection pointer. An interrupted copy cannot alter the
    previous selection or expose a partial bundle as the new selection. Source
    and staged hashes must agree before any metadata or directory is published.
    """
    sources = {part: Path(path) for part, path in sources.items() if path is not None}
    if not {'state', 'optimizer'}.issubset(sources):
        raise ValueError('Selected-best bundle requires model and optimizer')
    for part, path in sources.items():
        epoch = re.fullmatch(r'net_epoch-(-?\d+)_' + part + r'\.pt', path.name)
        if epoch and int(epoch[1]) != int(record['epoch']):
            raise ValueError(f'Selected-best source belongs to a different epoch: {path}')
    if 'scaler' not in sources or not sources['scaler'].is_file():
        sources.pop('scaler', None)
        if require_scaler:
            raise FileNotFoundError('Best checkpoint is incomplete: missing AMP scaler')
    original = bundle_identity(sources)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.global-best-', dir=checkpoint_dir))
    destination = staging.with_name(staging.name[1:])
    try:
        for part, path in sources.items():
            shutil.copy2(path, staging / f'global_best_{part}.pt')
        staged = {part: staging / f'global_best_{part}.pt' for part in sources}
        copied = bundle_identity(staged)
        check_bundle(original)
        if any(copied[k]['sha256'] != original[k]['sha256'] for k in sources):
            raise ValueError('Selected-best copy failed identity verification')
        if optimizer_transform:
            record['optimizer_resume'] = optimizer_transform(staged['optimizer'])
            copied = bundle_identity(staged)
        record = dict(record, bundle_schema_version=1,
                      scaler_status='present' if 'scaler' in sources else 'legacy_missing' if record['generation'] == -1 else 'not_required',
                      source_checkpoint=original,
                      checkpoint={k: dict(path=str(destination / f'global_best_{k}.pt'), sha256=v['sha256'])
                                  for k, v in copied.items()},
                      metadata_path=str(destination / 'global_best_metadata.json'))
        record.update({part + '_path': str(destination / f'global_best_{part}.pt') if part in sources else None
                       for part in ('state', 'optimizer', 'scaler', 'controller')})
        atomic_json(staging / 'global_best_metadata.json', record)
        os.replace(staging, destination)
        return record
    finally:
        if staging.exists():
            shutil.rmtree(staging)
