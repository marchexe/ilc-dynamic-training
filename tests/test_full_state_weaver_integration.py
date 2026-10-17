"""Opt-in: real Weaver on CPU continues exactly across restarts through weaver_full_state.py.

Same fixture and production command as tests/test_stage2b_cpu_integration.py (STAGE2B_CPU_INTEGRATION=1).
1. Each generation trains STEPS optimizer steps; with k=6 the saved Lookahead step counter must continue
   across the restart (4, then (4 + 4) % 6 = 2), which the historical contract, restarting at zero, cannot do.
2. Through the PBT command builder, two full natural epochs trained as four half-epoch generations must
   equal two one-epoch generations bit for bit (model, RAdam and Lookahead state), and each epoch's
   slices must partition its rows.
"""
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import torch

from tests.helpers import PROJECT_DIR
from tests.test_stage2b_cpu_integration import ENABLED, REASON, STEPS, build_fixture, run_generation
from training.runtime import read_metrics
from training.optimizer_contract import FORMAT, KEY

ENTRY = [sys.executable, str(PROJECT_DIR / 'scripts/training/weaver_full_state.py')]


@unittest.skipUnless(ENABLED, REASON)
class FullStateWeaverIntegrationTest(unittest.TestCase):
    def test_lookahead_state_survives_a_real_weaver_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, config, member, member_dir = build_fixture(Path(tmp))
            pretrained = torch.load(member_dir / 'net_epoch-17_optimizer.pt', map_location='cpu', weights_only=False)
            self.assertNotIn(KEY, pretrained)  # historical checkpoint: loads with the historical reset
            for generation, counter in ((0, STEPS % 6), (1, 2 * STEPS % 6)):
                log_path, epoch = run_generation(config, member, member_dir, generation, executable=ENTRY)
                saved = torch.load(member_dir / f'net_epoch-{epoch}_optimizer.pt', map_location='cpu', weights_only=True)
                lookahead = saved[KEY]
                self.assertEqual((lookahead['format'], lookahead['step_counter'], lookahead['k']), (FORMAT, counter, 6))
                self.assertEqual(len(lookahead['slow_weights']), sum(len(g['params']) for g in saved['param_groups']))
                self.assertIn('Overrode loaded optimizer learning rate', log_path.read_text())
                self.assertEqual({group['lr'] for group in saved['param_groups']}, {1.4e-5})

    def test_half_epoch_generations_equal_one_epoch_generations(self):
        """The engine's exact-restart contract: two epochs as 2 one-epoch or 4 half-epoch generations."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _, config, member, one_epoch = build_fixture(tmp)
            config['shared'].update(samples_per_epoch=None, optimizer_state_contract='full_lookahead_v1')
            half_epoch = tmp / 'half' / one_epoch.name
            shutil.copytree(one_epoch, half_epoch)
            audits = [read_metrics(run_generation(config, member, one_epoch, g)[0])['train_data_audit'] for g in (0, 1)]
            half = min(a['batches'] for a in audits) // 2
            sliced = dict(config, shared=dict(config['shared'], generations_per_epoch=2, batches_per_generation=half))
            slices = [read_metrics(run_generation(sliced, member, half_epoch, g)[0])['train_data_audit'] for g in range(4)]
            for epoch, audit in enumerate(audits):
                first, second = slices[2 * epoch:2 * epoch + 2]
                self.assertEqual((first['batches'], second['batches']), (half, audit['batches'] - half))
                self.assertEqual(first['consumed']['count'] + second['consumed']['count'], audit['consumed']['count'])
                self.assertEqual((first['exhausted'], second['exhausted']), (False, True))
            for part in ('state', 'optimizer'):
                expected = torch.load(one_epoch / f'net_epoch-19_{part}.pt', map_location='cpu', weights_only=True)
                actual = torch.load(half_epoch / f'net_epoch-21_{part}.pt', map_location='cpu', weights_only=True)
                self.assertTrue(same(expected, actual), part)

def same(expected, actual):
    """Exact equality of nested checkpoint content (tensors compared bit for bit)."""
    if isinstance(expected, torch.Tensor):
        return isinstance(actual, torch.Tensor) and torch.equal(expected, actual)
    if isinstance(expected, dict):
        return expected.keys() == actual.keys() and all(same(expected[k], actual[k]) for k in expected)
    if isinstance(expected, (list, tuple)):
        return len(expected) == len(actual) and all(same(e, a) for e, a in zip(expected, actual))
    return expected == actual


if __name__ == '__main__':
    unittest.main()
