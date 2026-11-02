"""Versioned Ranger checkpoint contract that preserves the Lookahead state across restarts.

Weaver's ``Lookahead.state_dict()`` stores only the inner RAdam state. On every restart Lookahead
re-caches its slow weights from the current model and restarts its step counter at zero, so each
process restart changes the optimizer trajectory. ``FullStateLookahead`` also saves the slow weights,
the step counter, ``alpha`` and ``k`` under one extra top-level key. Restarted training then continues
exactly as uninterrupted training. The inner keys stay at the top level, so tools that read or edit
``state`` and ``param_groups`` (learning-rate overrides, step accounting) work unchanged. A checkpoint
without the key, such as every historical one, loads with the historical restart behaviour.

Weaver and its optimizer files are not modified: ``install()`` makes Weaver's ``Ranger`` build a
``FullStateLookahead``, and ``scripts/training/weaver_full_state.py`` runs Weaver with it installed.
"""
from weaver.utils.nn.optimizer import ranger
from weaver.utils.nn.optimizer.lookahead import Lookahead

FORMAT = 'lookahead-full-v1'
KEY = 'lookahead'


def _parameters(optimizer):
    return [parameter for group in optimizer.param_groups for parameter in group['params']]


class FullStateLookahead(Lookahead):
    def __init__(self, optimizer, alpha=0.5, k=6, pullback_momentum='none'):
        if pullback_momentum != 'none':
            raise ValueError('The full Lookahead contract supports pullback_momentum="none" only, as used by Ranger')
        super().__init__(optimizer, alpha, k, pullback_momentum)

    def state_dict(self):
        state = self.optimizer.state_dict()
        state[KEY] = dict(format=FORMAT, step_counter=self.step_counter, alpha=self.alpha, k=self.k,
                          slow_weights=[self.state[p]['cached_params'].detach().clone() for p in _parameters(self.optimizer)])
        return state

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        saved = state_dict.pop(KEY, None)
        super().load_state_dict(state_dict)  # inner state, then the historical reset of the slow weights
        if saved is None:
            return
        parameters = _parameters(self.optimizer)
        slow = saved.get('slow_weights') or []
        if saved.get('format') != FORMAT:
            raise ValueError(f'Unknown Lookahead checkpoint format: {saved.get("format")!r}')
        if (saved.get('alpha'), saved.get('k')) != (self.alpha, self.k):
            raise ValueError(f'Lookahead alpha/k {saved.get("alpha")}/{saved.get("k")} differ from {self.alpha}/{self.k}')
        if not 0 <= saved.get('step_counter', -1) < self.k:
            raise ValueError(f'Invalid Lookahead step counter {saved.get("step_counter")} for k={self.k}')
        if len(slow) != len(parameters) or any(s.shape != p.shape for s, p in zip(slow, parameters)):
            raise ValueError('Lookahead slow weights do not match the model parameters')
        for parameter, weights in zip(parameters, slow):
            self.state[parameter]['cached_params'].copy_(weights)
        self.step_counter = int(saved['step_counter'])


def install():
    """Make Weaver's Ranger construct FullStateLookahead (idempotent)."""
    ranger.Lookahead = FullStateLookahead
