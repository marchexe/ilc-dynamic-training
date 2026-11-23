"""Distillation of the train8m standard ParT ensemble of seed 12345 (configs/prepared/ensemble_8m_dev_v1).

The teachers are the two models of that ensemble; the loss is 0.5 x cross-entropy on the true labels plus
0.5 x cross-entropy on the mean teacher probabilities (distill_particle_transformer.py). Paths are relative to the
repository root, and the checkpoint hashes are checked before training. A checkpoint of this network is a plain
pretrained_sgv_particle_transformer checkpoint (the student only).
"""
import hashlib
import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "distill_particle_transformer", Path(__file__).with_name("distill_particle_transformer.py"))
distill = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(distill)

TEACHERS = {
    "runs/pbt/sel1m_train8m_standard_part_3e-3/flat_decay_3000e-6/net_epoch-62_state.pt":
        "365d817623c64d25824d764e5f793a12bde2647ce4e10277b94d2b8cd5b86cbf",
    "runs/pbt/sel1m_train8m_standard_part/flat_decay_1000e-6/net_epoch-67_state.pt":
        "1af683e64c4504989cdda9852018dd739a9fdbc243eb54e1bb80925d814cf4c0",
}
HARD_WEIGHT = 0.5


def _teacher_paths():
    paths = []
    for relative, expected in TEACHERS.items():
        path = _ROOT / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"teacher checkpoint {relative} changed: {digest}")
        paths.append(str(path))
    return paths


def get_model(data_config, for_inference=False, **kwargs):
    teachers = [] if for_inference else _teacher_paths()
    return distill.get_model(data_config, teacher_paths=teachers, hard_weight=HARD_WEIGHT,
                             for_inference=for_inference, **kwargs)


def get_loss(data_config, **kwargs):
    return distill.get_loss(data_config, hard_weight=HARD_WEIGHT)
