"""Epoch cursor: consecutive processes reproduce one uninterrupted epoch of batches and randomness."""
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

import tests.helpers  # noqa: F401  (puts scripts/ on sys.path)
from training.data_cursor import EpochCursor

BATCHES, SEED = 23, 77


class Loader:
    """Stand-in for Weaver's loader: the batch order is fixed by the epoch, like SimpleIterDataset."""

    def __init__(self, epoch):
        self.dataset, self.num_workers, self.epoch = object(), 1, epoch

    def __iter__(self):
        order = torch.randperm(BATCHES, generator=torch.Generator().manual_seed(1000003 * self.epoch))
        return iter([(int(i), torch.full((4,), float(i))) for i in order])


class SequenceTrimmer(torch.nn.Module):
    """Like the network's trimmer: an unsaved warmup counter changes the first steps of every process."""

    def __init__(self):
        super().__init__()
        self._counter = 0

    def forward(self):
        if self._counter < 5:
            self._counter += 1
            return 0.0
        return random.uniform(0.9, 1.02) + float(np.random.rand()) + float(torch.rand(1))


def process(make_cursor):
    """One Weaver process: seeded at start, builds the model, then trains on the cursor."""
    random.seed(SEED), np.random.seed(SEED), torch.manual_seed(SEED)
    model = torch.nn.Sequential(SequenceTrimmer())
    torch.rand(5)  # model setup consumes randomness
    return [(index, model[0]()) for index, _ in make_cursor(lambda: model)]


class EpochCursorTest(unittest.TestCase):
    def chunked(self, bounds, restore=True):
        records = []
        with tempfile.TemporaryDirectory() as tmp:
            for chunk, (skip, take) in enumerate(bounds):
                state_in = Path(tmp) / f'{chunk - 1}.pt' if chunk and restore else None
                records += process(lambda model: EpochCursor(Loader(18), skip, take, state_in, Path(tmp) / f'{chunk}.pt', model))
        return records

    def test_chunks_reproduce_the_uninterrupted_epoch(self):
        uninterrupted = process(lambda model: EpochCursor(Loader(18), model=model))
        self.assertEqual(len(uninterrupted), BATCHES)
        self.assertEqual(self.chunked([(0, 3), (3, 5), (8, 5), (13, 5), (18, None)]), uninterrupted)
        self.assertEqual(self.chunked([(0, 22), (22, None)]), uninterrupted)

    def test_without_the_continuation_state_the_chunks_differ(self):
        uninterrupted = process(lambda model: EpochCursor(Loader(18), model=model))
        chunked = self.chunked([(0, 8), (8, None)], restore=False)
        self.assertEqual([i for i, _ in chunked], [i for i, _ in uninterrupted])
        self.assertNotEqual([r for _, r in chunked], [r for _, r in uninterrupted])
        self.assertEqual([r for _, r in chunked[8:13]], [0.0] * 5)  # the warmup repeats after a plain restart

    def test_epoch_fixes_the_stream_and_the_cursor_validates_bounds(self):
        self.assertNotEqual([i for i, _ in Loader(18)], [i for i, _ in Loader(19)])
        cursor = EpochCursor(Loader(18), 20, 10)
        self.assertEqual(len(list(cursor)), 3)
        self.assertEqual(cursor.yielded, 3)
        self.assertEqual(cursor.num_workers, 1)
        with self.assertRaisesRegex(ValueError, 'cannot skip 30'):
            list(EpochCursor(Loader(18), 30))
        for skip, take in ((-1, None), (0, 0)):
            with self.assertRaisesRegex(ValueError, 'Invalid cursor'):
                EpochCursor(Loader(18), skip, take)

    def test_module_state_must_match_the_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / 'state.pt'
            process(lambda model: EpochCursor(Loader(18), 0, 3, None, state, model))
            with self.assertRaisesRegex(ValueError, 'module state does not match'):
                list(EpochCursor(Loader(18), 3, 1, state, None, lambda: torch.nn.Linear(1, 1)))


if __name__ == '__main__':
    unittest.main()
