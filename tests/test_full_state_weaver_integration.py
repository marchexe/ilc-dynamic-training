"""Opt-in: real Weaver on CPU saves and restores the full Lookahead state through weaver_full_state.py.

Same fixture and production command as tests/test_stage2b_cpu_integration.py (STAGE2B_CPU_INTEGRATION=1).
Each generation trains STEPS optimizer steps; with k=6 the saved step counter must continue across the
restart (4 after the first generation, (4 + 4) % 6 = 2 after the second), which the historical contract,
restarting at zero, cannot produce.
"""
from pathlib import Path
import sys
import tempfile
import unittest

import torch

from tests.helpers import PROJECT_DIR
from tests.test_stage2b_cpu_integration import ENABLED, REASON, STEPS, build_fixture, run_generation
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


if __name__ == '__main__':
    unittest.main()
