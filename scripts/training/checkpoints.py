"""Shared checkpoint bundles, preserving the historical copy semantics.

No serialization or optimizer transformation occurs in copy_bundle. The scaler
companion follows the optimizer exactly as in the reference implementation.
"""
import os
import shutil
from pathlib import Path

from training.runtime import sha256


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
