import random

import numpy as np
import torch

from weaver.train import _set_random_seed


def _draw_values():
    return random.random(), np.random.random(), torch.rand(1).item()


def test_set_random_seed_repeats_python_numpy_and_torch_streams():
    assert _set_random_seed(12345) == 12345
    first = _draw_values()
    assert _set_random_seed(12345) == 12345
    second = _draw_values()

    np.testing.assert_allclose(first, second)


def test_set_random_seed_offsets_distributed_rank():
    assert _set_random_seed(12345, local_rank=2) == 12347
