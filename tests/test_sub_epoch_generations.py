"""Engine commands for the exact-restart contract and generations shorter than an epoch."""
from pathlib import Path
import tempfile
import unittest

from tests.helpers import PROJECT_DIR, namespace, synthetic_config
from training.members import MemberState
from training.pbt import config as config_module
from training.pbt.execution.weaver_command import make_command

BASE = 'configs/experiments/foundation_fixed_lr_smoke.yaml'
ENTRY = str(PROJECT_DIR / 'scripts/training/weaver_full_state.py')
SLOT = dict(host=None, gpu='0', label='0')
MEMBER = MemberState.from_legacy(dict(name='lr_3e-6', lr=3e-6))


def option(command, name):
    return command[command.index(name) + 1] if name in command else None


class SubEpochGenerationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.member_dir = Path(self.tmp.name) / 'run' / 'lr_3e-6'

    def tearDown(self):
        self.tmp.cleanup()

    def config(self, **shared):
        with tempfile.TemporaryDirectory() as directory:
            return synthetic_config(directory, {'shared': shared}, base=BASE)

    def command(self, config, generation):
        return make_command(config, MEMBER, SLOT, self.member_dir, generation)[0]

    def test_existing_configurations_are_unchanged(self):
        config = self.config()
        command = self.command(config, 3)
        self.assertTrue(command[0].endswith('/weaver'))
        self.assertEqual(option(command, '--seed'), str(int(config['shared']['seed']) + 3))
        for path in ('configs/experiments/foundation_fixed_lr_50epochs.yaml', 'configs/experiments/windowed_pbt_v2.yaml',
                     'configs/experiments/cadenced_pbt_v1.yaml', 'configs/experiments/cadenced_pbt_v1_cadence5.yaml',
                     'configs/experiments/stage3_train8m_fixed_lr.yaml', 'configs/experiments/stage3_train8m_windowed_pbt_v2.yaml'):
            resolved = config_module.load_config(namespace(config=PROJECT_DIR / path, experiment_name='unit', gpus='0,1,2,3,4',
                                                           slots=None, smoke=False))
            shared = resolved['shared']
            self.assertNotIn('lr_mutation_interval_generations', resolved['pbt'].get('cadenced_pbt_v1') or {}, path)
            self.assertFalse({'optimizer_state_contract', 'generations_per_epoch', 'batches_per_generation'} & set(shared), path)

    def test_full_contract_runs_the_same_weaver_arguments_through_the_entry_point(self):
        historical = self.command(self.config(), 2)
        full = self.command(self.config(optimizer_state_contract='full_lookahead_v1'), 2)
        self.assertEqual(full[1], ENTRY)
        self.assertTrue(full[0].endswith('/python'))
        self.assertEqual(full[2:], historical[1:])

    def test_slices_share_their_epoch_seed_and_stream_and_chain_the_process_state(self):
        config = self.config(optimizer_state_contract='full_lookahead_v1', generations_per_epoch=5, batches_per_generation=100)
        seed = int(config['shared']['seed'])
        for generation in range(11):
            command, _, epoch = make_command(config, MEMBER, SLOT, self.member_dir, generation)
            data_epoch, part = divmod(generation, 5)
            self.assertEqual(epoch, 18 + generation)
            self.assertEqual(option(command, '--load-epoch'), str(epoch - 1))
            self.assertEqual(option(command, '--seed'), str(seed + data_epoch))
            self.assertEqual(option(command, '--cursor-data-epoch'), str(18 + data_epoch))
            self.assertEqual(option(command, '--cursor-skip'), str(100 * part))
            last = part == 4
            self.assertEqual(option(command, '--cursor-take'), None if last else '100')
            self.assertEqual(option(command, '--cursor-state-out'),
                             None if last else str(self.member_dir / f'net_epoch-{epoch}_continuation.pt'))
            self.assertEqual(option(command, '--cursor-state-in'),
                             str(self.member_dir / f'net_epoch-{epoch - 1}_continuation.pt') if part else None)
            self.assertLess(command.index('--cursor-data-epoch'), command.index('--seed'))  # entry-point options first

    def test_inconsistent_settings_are_rejected(self):
        full = dict(optimizer_state_contract='full_lookahead_v1')
        cases = [(dict(generations_per_epoch=5), 'configured together'),
                 (dict(batches_per_generation=100), 'configured together'),
                 (dict(generations_per_epoch=5, batches_per_generation=100), 'require optimizer_state_contract'),
                 (dict(full, generations_per_epoch=5, batches_per_generation=100, samples_per_epoch=1024), 'full natural epochs'),
                 (dict(full, generations_per_epoch=5, batches_per_generation=100, weaver_epochs_per_generation=2),
                  'full natural epochs'),
                 (dict(full, generations_per_epoch=1, batches_per_generation=100), 'greater than or equal to 2'),
                 (dict(full, optimizer='adamw'), 'ranger optimizer only')]
        for shared, message in cases:
            with self.assertRaisesRegex(ValueError, message):
                self.config(**shared)


if __name__ == '__main__':
    unittest.main()
