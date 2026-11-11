"""The flat+decay ending of windowed_pbt_v2 (final_decay), and the step 2 follow-up configurations."""
import copy
import unittest

from tests.helpers import PROJECT_DIR
from launch.experiment import configuration
from training.pbt.models.config import ResolvedPBTConfig
from training.pbt.planning.lr_schedule import apply_lr_schedule, final_decay_start, schedule_factor
from training.pbt.planning.windowed_pbt_v2 import windowed_pbt_v2_plan

EXPERIMENTS = PROJECT_DIR / 'configs/experiments'
METRIC = 'validation_total_reference_mistag_geomean_percent'


def load(name):
    return configuration(EXPERIMENTS / f'{name}.yaml')


def differences(new, base):
    return {section: {k for k in set(new[section]) | set(base[section]) if new[section].get(k) != base[section].get(k)}
            for section in ('shared', 'pbt')}


class FinalDecayTest(unittest.TestCase):
    def setUp(self):
        self.config = load('sel1m_fast_pbt_decay')

    def test_decay_starts_at_generation_35_of_50(self):
        self.assertEqual(final_decay_start(self.config), 35)
        self.assertIsNone(final_decay_start(load('sel1m_fast_pbt')))

    def test_lrs_decay_from_their_value_at_the_start_like_flat_decay(self):
        manifest = dict(members=dict(a=dict(lr=8e-5), b=dict(lr=3e-5)))
        self.assertEqual(apply_lr_schedule(self.config, manifest, 34), {})
        for generation in range(35, 50):
            apply_lr_schedule(self.config, manifest, generation)
            factor = schedule_factor(dict(kind='flat_decay'), generation, 50)
            self.assertAlmostEqual(manifest['members']['a']['lr'], 8e-5 * factor, delta=1e-18)
            self.assertAlmostEqual(manifest['members']['b']['lr'], 3e-5 * factor, delta=1e-18)
        self.assertAlmostEqual(manifest['members']['a']['lr'], 8e-7, delta=1e-18)
        self.assertEqual(manifest['members']['a']['final_decay_base_lr'], 8e-5)
        before = copy.deepcopy(manifest)
        apply_lr_schedule(self.config, manifest, 49)
        self.assertEqual(manifest, before)

    def test_no_decision_from_the_generation_before_the_decay(self):
        members = {n: dict(lr=lr) for n, lr in dict(a=1e-5, b=2e-5, c=4e-5, d=8e-5, e=1.6e-4).items()}
        for index in (34, 35, 41, 49):
            generation = dict(index=index, workers={n: dict(metrics={METRIC: 0.3 + i / 100})
                                                    for i, n in enumerate(members)})
            ranking, plan = windowed_pbt_v2_plan(self.config, generation, members, dict(generations=[]))
            self.assertEqual((ranking, plan), (list(members), []))
            self.assertNotIn('windowed_pbt_v2', generation)

    def test_decay_must_leave_a_decision_window(self):
        payload = copy.deepcopy(self.config)
        payload['pbt']['windowed_pbt_v2']['final_decay']['decay_fraction'] = 0.96
        with self.assertRaisesRegex(ValueError, 'no complete decision window'):
            ResolvedPBTConfig.model_validate(payload)
        ResolvedPBTConfig.model_validate(self.config)


class FollowUpConfigurationTest(unittest.TestCase):
    def test_pbt_decay_changes_only_the_final_decay(self):
        for name, base in (('sel1m_fast_pbt_decay', 'sel1m_fast_pbt'),
                           ('sel1m_fast_pbt_decay_seed22345', 'sel1m_fast_pbt_seed22345')):
            config, reference = load(name), load(base)
            self.assertEqual(differences(config, reference), dict(shared=set(), pbt={'windowed_pbt_v2'}))
            options = dict(config['pbt']['windowed_pbt_v2'])
            self.assertEqual(options.pop('final_decay'), dict(decay_fraction=0.3, final_factor=0.01))
            self.assertEqual({k: v for k, v in options.items() if v is not None},
                             {k: v for k, v in reference['pbt']['windowed_pbt_v2'].items() if v is not None})
            self.assertEqual(config['population'], reference['population'])

    def test_scratch_standard_part_loads_nothing_and_keeps_the_rest(self):
        config, reference = load('sel1m_scratch_standard_part'), load('sel1m_standard_part')
        self.assertEqual(config['shared']['initialization_mode'], 'scratch')
        self.assertFalse(config['shared']['freeze_batch_norm'])
        for key in ('checkpoint', 'initial_state', 'initial_optimizer', 'initial_epoch'):
            self.assertIsNone(config['shared'].get(key))
        for key in ('dataset', 'train_suffix', 'validation_dataset', 'validation_suffix', 'seed', 'generations',
                    'optimizer', 'batch_size', 'data_config', 'network_config', 'amp_dtype'):
            self.assertEqual(config['shared'][key], reference['shared'][key], key)
        self.assertEqual([m['start_lr'] for m in config['population']], [2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3])


if __name__ == '__main__':
    unittest.main()
