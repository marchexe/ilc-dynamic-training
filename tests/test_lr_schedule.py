"""Per-member LR schedules for fixed_lr_grid runs, and the sel1m step-2 configurations."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from tests.helpers import PROJECT_DIR
from launch.experiment import configuration
from training.pbt import runner
from training.pbt.models.config import ResolvedPBTConfig
from training.pbt.planning.lr_schedule import apply_lr_schedule, schedule_factor, scheduled_lr

EXPERIMENTS = PROJECT_DIR / 'configs/experiments'
VALIDATION = dict(validation_dataset='datasets/20250711_ilc_nnqq_sgv_10m_3cat_sel1m_parquet', validation_suffix='sel1m_tail')


def load(name):
    return configuration(EXPERIMENTS / f'{name}.yaml')


def weaver_flat_decay(base, epochs):
    """LR per epoch of Weaver's flat+decay (train.py, MultiStepLR branch)."""
    decay = max(1, int(epochs * 0.3))
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=base)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(range(epochs - decay, epochs)),
                                                     gamma=0.01 ** (1.0 / decay))
    rates = []
    for _ in range(epochs):
        rates.append(opt.param_groups[0]['lr'])
        opt.step(); scheduler.step()
    return rates


class ScheduleTest(unittest.TestCase):
    def test_flat_decay_equals_weaver(self):
        for epochs in (20, 50):
            ours = [1e-3 * schedule_factor(dict(kind='flat_decay'), g, epochs) for g in range(epochs)]
            for a, b in zip(ours, weaver_flat_decay(1e-3, epochs)):
                self.assertTrue(math.isclose(a, b, rel_tol=1e-12))

    def test_flat_decay_reproduces_the_pretrained_epoch17_lr(self):
        # The pretrained optimizer after epoch 17 holds the LR of epoch 18 of a 20-epoch schedule from 1e-3.
        self.assertTrue(math.isclose(1e-3 * schedule_factor(dict(kind='flat_decay'), 18, 20), 2.1544346900318847e-05,
                                     rel_tol=1e-12))

    def test_cosine_end_points(self):
        schedule = dict(kind='cosine', final_factor=0.1)
        self.assertEqual(schedule_factor(schedule, 0, 50), 1.0)
        self.assertTrue(math.isclose(schedule_factor(schedule, 49, 50), 0.1))
        with self.assertRaises(ValueError):
            schedule_factor(schedule, 50, 50)

    def test_apply_sets_only_scheduled_members(self):
        config = dict(shared=dict(generations=50), population=[
            dict(name='a', start_lr=1e-3, lr_schedule=dict(kind='flat_decay')), dict(name='b', start_lr=1e-4)])
        manifest = dict(members=dict(a=dict(lr=1e-3), b=dict(lr=1e-4)))
        self.assertEqual(apply_lr_schedule(config, manifest, 0), {})
        changes = apply_lr_schedule(config, manifest, 49)
        self.assertEqual(set(changes), {'a'})
        self.assertTrue(math.isclose(manifest['members']['a']['lr'], 1e-5))
        self.assertEqual(manifest['members']['b']['lr'], 1e-4)
        self.assertEqual(apply_lr_schedule(config, manifest, 49), {})

    def test_new_generation_record_uses_the_scheduled_lr(self):
        config = load('sel1m_standard_part')
        manifest = dict(members={m['name']: dict(name=m['name'], lr=m['start_lr']) for m in config['population']},
                        generations=[])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'manifest.json'
            record = runner._generation_record_for(config, manifest, path, 40)
            saved = json.loads(path.read_text())
        self.assertEqual(record['index'], 40)
        for member in config['population']:
            expected = scheduled_lr(member, 40, 50)
            self.assertEqual(manifest['members'][member['name']]['lr'], expected)
            self.assertEqual(saved['members'][member['name']]['lr'], expected)
        self.assertLess(manifest['members']['flat_decay_1000e-6']['lr'], 1e-3)

    def test_schedule_only_with_fixed_lr_grid(self):
        config = load('sel1m_fast_pbt')
        payload = copy.deepcopy(config)
        payload['population'][0]['lr_schedule'] = dict(kind='flat_decay')
        with self.assertRaisesRegex(ValueError, 'lr_schedule requires fixed_lr_grid'):
            ResolvedPBTConfig.model_validate({k: v for k, v in payload.items()})


class Sel1mConfigurationTest(unittest.TestCase):
    def differences(self, new, base):
        return {section: {k for k in set(new[section]) | set(base[section]) if new[section].get(k) != base[section].get(k)}
                for section in ('shared', 'pbt')}

    def test_runs_change_only_the_selection_set_and_what_they_test(self):
        for new, base, pbt in (('sel1m_fixed_lr_grid', 'fixed_lr_best_grid_50epochs_seed22345', set()),
                               ('sel1m_fast_pbt', 'windowed_pbt_v2_fast_fallback', set()),
                               ('sel1m_standard_part', 'foundation_fixed_lr_50epochs', {'min_lr', 'max_lr'})):
            config, reference = load(new), load(base)
            changed = self.differences(config, reference)
            expected_shared = set(VALIDATION) | ({'seed'} if reference['shared']['seed'] != 12345 else set())
            self.assertEqual(changed, dict(shared=expected_shared, pbt=pbt), new)
            self.assertEqual({k: config['shared'][k] for k in VALIDATION},
                             {k: str(PROJECT_DIR / v) if k == 'validation_dataset' else v for k, v in VALIDATION.items()})
            self.assertEqual(config['shared']['seed'], 12345)
            second = load(f'{new}_seed22345')
            self.assertEqual(self.differences(second, config), dict(shared={'seed'}, pbt=set()))
            self.assertEqual(second['shared']['seed'], 22345)
            self.assertEqual(second['population'], config['population'])

    def test_standard_part_is_flat_decay_from_the_standard_lr_down(self):
        config = load('sel1m_standard_part')
        self.assertEqual([m['start_lr'] for m in config['population']], [1e-5, 3e-5, 1e-4, 3e-4, 1e-3])
        self.assertTrue(all(m['lr_schedule'] == dict(kind='flat_decay', decay_fraction=0.3, final_factor=0.01)
                            or m['lr_schedule'] == dict(kind='flat_decay') for m in config['population']))
        grid = load('sel1m_fixed_lr_grid')
        self.assertEqual([m['start_lr'] for m in grid['population']],
                         [m['start_lr'] for m in load('fixed_lr_best_grid_50epochs_seed22345')['population']])


if __name__ == '__main__':
    unittest.main()
