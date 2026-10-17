"""Continue one data epoch across Weaver processes, for generations shorter than an epoch.

Weaver's training stream for an epoch is fixed by the run seed and the epoch number, but every new process
starts it from the beginning. ``EpochCursor`` rebuilds the same stream, skips the batches earlier processes
trained on, and yields the next ``take`` batches (or the rest of the epoch).

Training also depends on process state that no checkpoint holds:
- the global generators of torch (dropout, trimming permutations), Python ``random`` (trimming quantile)
  and NumPy;
- ``SequenceTrimmer._counter``, which disables trimming for the first five training steps of a process.

The cursor saves this continuation state after the training step of its last batch and restores it just
before its first. Together with the full Lookahead checkpoint contract (``optimizer_contract.py``),
consecutive cursors reproduce one uninterrupted epoch.

Skipped batches are read but not trained or audited, so the training audits of the chunks of one epoch
partition its rows; only the last chunk exhausts the stream.
"""
from pathlib import Path
import random

import numpy as np
import torch

# Module attributes that change during training but are not in the state_dict, by module class name.
PROCESS_STATE = {'SequenceTrimmer': ('_counter',)}


def continuation_state(model=None):
    name, keys, position, has_gauss, gauss = np.random.get_state()
    state = dict(torch_cpu=torch.get_rng_state(), python=random.getstate(),
                 numpy=(name, keys.tolist(), position, has_gauss, gauss), modules={})
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    for path, module in (model.named_modules() if model is not None else ()):
        attributes = PROCESS_STATE.get(type(module).__name__, ())
        if attributes:
            state['modules'][path] = {attribute: getattr(module, attribute) for attribute in attributes}
    return state


def restore_continuation_state(state, model=None):
    torch.set_rng_state(state['torch_cpu'])
    if 'torch_cuda' in state:
        if not torch.cuda.is_available():
            raise ValueError('The saved continuation state includes CUDA, but CUDA is not available')
        torch.cuda.set_rng_state_all(state['torch_cuda'])
    random.setstate(state['python'])
    name, keys, position, has_gauss, gauss = state['numpy']
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, gauss))
    modules = dict(model.named_modules()) if model is not None else {}
    if set(state['modules']) != {p for p, m in modules.items() if type(m).__name__ in PROCESS_STATE}:
        raise ValueError('The saved module state does not match the model')
    for path, attributes in state['modules'].items():
        for attribute, value in attributes.items():
            setattr(modules[path], attribute, value)


class EpochCursor:
    """Batches ``[skip, skip + take)`` of a loader's epoch stream; ``take=None`` runs to the end of the epoch.

    ``model`` is a callable returning the model being trained (or None), read when iteration starts.
    """

    def __init__(self, loader, skip=0, take=None, state_in=None, state_out=None, model=lambda: None):
        if skip < 0 or (take is not None and take < 1):
            raise ValueError(f'Invalid cursor skip={skip} take={take}')
        self.loader, self.skip, self.take = loader, skip, take
        self.state_in, self.state_out, self.model = state_in, state_out, model
        self.yielded = None

    def __getattr__(self, name):
        return getattr(self.loader, name)  # dataset, num_workers and the rest of the DataLoader interface

    def __iter__(self):
        iterator = iter(self.loader)
        for index in range(self.skip):
            if next(iterator, None) is None:
                raise ValueError(f'The epoch ended after {index} batches; cannot skip {self.skip}')
        if self.state_in:
            restore_continuation_state(torch.load(self.state_in, weights_only=True), self.model())
        count = 0
        while self.take is None or count < self.take:
            batch = next(iterator, None)
            if batch is None:
                break
            yield batch
            count += 1
        self.yielded = count
        if self.state_out:
            Path(self.state_out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(continuation_state(self.model()), self.state_out)


def install(data_epoch, skip=0, take=None, state_in=None, state_out=None):
    """Make the next Weaver run train on a cursor into ``data_epoch``'s training stream."""
    import weaver.train as weaver_train
    from weaver.utils.dataset import SimpleIterDataset

    load, setup, built = weaver_train.train_load, weaver_train.model_setup, {}

    def model_setup(*args, **kwargs):
        result = setup(*args, **kwargs)
        built['model'] = result[0]
        return result

    def train_load(args):
        train_loader, *rest = load(args)
        dataset = train_loader.dataset
        # Weaver sets the epoch before each training pass; pin the stream to the data epoch being continued.
        dataset.set_epoch = lambda epoch: SimpleIterDataset.set_epoch(dataset, data_epoch)
        return (EpochCursor(train_loader, skip, take, state_in, state_out, lambda: built.get('model')), *rest)

    weaver_train.model_setup, weaver_train.train_load = model_setup, train_load
