"""Content identity of restart bundles for the Stage 3 determinism checks.

File hashes cannot show that two workers hold the same training state:
torch.save records in `data.pkl` the CUDA device each tensor lived on
(`cuda:4` vs `cuda:0`), and `.data/serialization_id` is a checksum over the
archive records, so it changes with it. A worker on another slot index writes
different bytes for bit-identical tensors; the device is not training state.

The determinism checks therefore compare the deserialized restart bundle that
Weaver writes after every epoch and loads on resume: model weights (`state`),
optimizer state (`optimizer`: RAdam moments, step counts, Ranger buffer and
param groups including LR), the AMP grad scaler (`scaler`) and, when present,
the training controller (`controller`). Every field is compared, none is
skipped: tensors must agree in dtype, shape and every byte; Python values must
be of the same type and equal (floats bitwise); mappings must have the same
keys. Tensors are loaded to CPU, so only the recorded device tag is ignored.
"""
import struct
from pathlib import Path

import numpy as np
import torch

from training.checkpoints import bundle_paths

REQUIRED = ('state', 'optimizer', 'scaler')
OPTIONAL = ('controller',)


def restart_bundle(directory, epoch):
    """Paths of the epoch-`epoch` restart bundle in a member directory."""
    paths = bundle_paths(Path(directory), epoch)
    paths['controller'] = paths['state'].with_name(f'net_epoch-{epoch}_controller.pt')
    return paths


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def _tensor_bytes(tensor):
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8)


def differences(left, right, where=''):
    """Locations where two deserialized objects differ; empty means bit-identical."""
    if type(left) is not type(right):
        return [f'{where}: type {type(left).__name__} != {type(right).__name__}']
    if torch.is_tensor(left):
        if left.dtype != right.dtype or left.shape != right.shape:
            return [f'{where}: {left.dtype}{tuple(left.shape)} != {right.dtype}{tuple(right.shape)}']
        return [] if torch.equal(_tensor_bytes(left), _tensor_bytes(right)) else [f'{where}: tensor values differ']
    if isinstance(left, np.ndarray):
        same = left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()
        return [] if same else [f'{where}: array differs']
    if isinstance(left, dict):
        if set(left) != set(right):
            return [f'{where}: keys differ {sorted(map(str, set(left) ^ set(right)))[:5]}']
        return [d for key in left for d in differences(left[key], right[key], f'{where}/{key}')]
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return [f'{where}: length {len(left)} != {len(right)}']
        return [d for i, (a, b) in enumerate(zip(left, right)) for d in differences(a, b, f'{where}[{i}]')]
    if isinstance(left, float):
        return [] if struct.pack('<d', left) == struct.pack('<d', right) else [f'{where}: {left!r} != {right!r}']
    return [] if left == right else [f'{where}: {left!r} != {right!r}']


def bundle_differences(left, right):
    """Compare two restart bundles ({part: path}); empty means identical training state.

    A required part missing on either side is a difference; an optional part
    must be present on both sides or on neither.
    """
    found = []
    for part in REQUIRED + OPTIONAL:
        present = [paths.get(part) is not None and Path(paths[part]).is_file() for paths in (left, right)]
        if not all(present):
            if part in REQUIRED or any(present):
                found.append(f'{part}: missing on the {" and ".join(s for s, p in zip(("left", "right"), present) if not p)}')
            continue
        found += [f'{part}{d}' for d in differences(load(left[part]), load(right[part]))]
    return found
