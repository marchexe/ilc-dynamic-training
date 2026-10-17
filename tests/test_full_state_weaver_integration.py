"""Opt-in: real Weaver on CPU continues exactly across restarts through weaver_full_state.py.

Same fixture and production command as tests/test_stage2b_cpu_integration.py (STAGE2B_CPU_INTEGRATION=1).
1. Each generation trains STEPS optimizer steps; with k=6 the saved Lookahead step counter must continue
   across the restart (4, then (4 + 4) % 6 = 2), which the historical contract, restarting at zero, cannot do.
2. One full natural epoch split into three processes with an epoch cursor must equal the uninterrupted
   epoch bit for bit (model, RAdam and Lookahead state), and the chunks' audits must partition its rows.
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

    def test_three_restarted_chunks_equal_one_uninterrupted_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _, config, member, uninterrupted = build_fixture(tmp)
            config['shared']['samples_per_epoch'] = None  # full natural epoch, as in production
            chunked = tmp / 'chunked' / uninterrupted.name
            shutil.copytree(uninterrupted, chunked)
            seed = int(config['shared']['seed'])
            log, epoch = run_generation(config, member, uninterrupted, 0, executable=ENTRY)
            reference = read_metrics(log)['train_data_audit']
            third = reference['batches'] // 3
            audits = []
            for generation, (skip, take) in enumerate(((0, third), (third, third), (2 * third, None))):
                cursor = ['--cursor-data-epoch', str(epoch), '--cursor-skip', str(skip),
                          '--cursor-state-out', str(tmp / f'state-{generation}.pt')]
                cursor += ['--cursor-take', str(take)] if take else []
                cursor += ['--cursor-state-in', str(tmp / f'state-{generation - 1}.pt')] if generation else []
                log, last = run_generation(config, member, chunked, generation, executable=[*ENTRY, *cursor], seed=seed)
                audits.append(read_metrics(log)['train_data_audit'])
            self.assertEqual([a['batches'] for a in audits], [third, third, reference['batches'] - 2 * third])
            self.assertEqual(sum(a['consumed']['count'] for a in audits), reference['consumed']['count'])
            self.assertEqual([a['exhausted'] for a in audits], [False, False, True])
            for part in ('state', 'optimizer'):
                expected = torch.load(uninterrupted / f'net_epoch-{epoch}_{part}.pt', map_location='cpu', weights_only=True)
                actual = torch.load(chunked / f'net_epoch-{last}_{part}.pt', map_location='cpu', weights_only=True)
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
