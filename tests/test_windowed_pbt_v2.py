import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests.helpers import PROJECT_DIR
from scripts.launch.experiment import configuration
from scripts.validation import verify_fixed_lr as verifier
from training.pbt.models.config import ResolvedPBTConfig
from training.pbt.models.manifest import PBTManifest
from training.pbt.planning import windowed_pbt_v2 as strategy
from training.pbt.planning.dispatch import plan_for_strategy
from training.pbt.runner import _plan_generation_exploit, _finalize_generation, run_final_checkpoint_evaluations
from training.pbt.reporting import ensure_run_layout
from training.runtime import atomic_json, sha256


def configuration_fixture():
    config = configuration(PROJECT_DIR / 'configs/experiments/windowed_pbt_v2.yaml')
    config['population'] = [dict(name=chr(97 + i), start_lr=m['start_lr']) for i, m in enumerate(config['population'])]
    config['shared']['initial_epoch'] = 2
    return config


def manifest_fixture(config):
    return dict(schema_version=1, experiment='fixture', fingerprint='fixture', status='running', next_generation=0,
                config=config, members={m['name']: dict(name=m['name'], lr=m['start_lr']) for m in config['population']},
                generations=[], best=None)


def generation_fixture(manifest, index, values=None):
    config = manifest['config']
    values = values or [0.45, 0.44, 0.43, 0.42, 0.40]
    keys = [config['pbt']['metric'], 'validation_loss', *strategy.WORKING_POINTS]
    return dict(index=index, epoch=config['shared']['initial_epoch'] + index + 1, status='running', exploit=None,
                workers={n: dict(lr=m['lr'], status='completed', returncode=0,
                                 metrics={k: values[i] for k in keys}) for i, (n, m) in enumerate(manifest['members'].items())})


def plan_until(manifest, end, values=None):
    for i in range(len(manifest['generations']), end):
        g = generation_fixture(manifest, i, values)
        manifest['generations'].append(g)
        g['ranking'], g['exploit'] = plan_for_strategy(manifest['config'], g, manifest['members'], manifest)
        g['status'] = 'completed'
    return g


def save_bundle(root, name, epoch, lr):
    paths = strategy.bundle_paths(root / name, epoch)
    paths['state'].parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(weight=torch.tensor([epoch, ord(name)])), paths['state'])
    torch.save(dict(state={0: dict(step=epoch, exp_avg=torch.tensor([0.3]), exp_avg_sq=torch.tensor([0.7]),
                                   slow_buffer=torch.tensor([0.9]))},
                    param_groups=[dict(lr=lr, params=[0], step_counter=epoch)]), paths['optimizer'])
    torch.save(dict(scale=4096., _growth_tracker=epoch), paths['scaler'])
    return paths


class WindowedPBTTest(unittest.TestCase):
    def setUp(self):
        self.config = configuration_fixture()
        self.manifest = manifest_fixture(self.config)

    def test_no_actions_inside_window_and_two_window_delay(self):
        for end in range(1, 11):
            g = plan_until(self.manifest, end)
            if end < 10:
                self.assertEqual(g['exploit'], [])
            self.assertEqual(strategy.STRATEGY in g, end % 5 == 0)
        self.assertEqual(len(g['exploit']), 2)
        self.assertEqual({e['mutation_factor'] for e in g['exploit']}, {0.8, 1.2})
        self.assertIn(14e-6 * 1.2, [e['new_lr'] for e in g['exploit']])
        self.assertTrue(all(e['donor'] == 'e' and e['recipient'] != 'e' for e in g['exploit']))

    def test_score_uses_last_three_not_lucky_final_epoch(self):
        plan_until(self.manifest, 4)
        g = generation_fixture(self.manifest, 4, [0.34, 0.44, 0.43, 0.42, 0.40])
        rank, _ = plan_for_strategy(self.config, g, self.manifest['members'], self.manifest)
        self.assertEqual(rank[0], 'e')  # 'a' won the last epoch but loses the window.
        self.assertAlmostEqual(g[strategy.STRATEGY]['members']['a']['window_score'], (0.45 * 2 + 0.34) / 3)
        self.assertAlmostEqual(g[strategy.STRATEGY]['members']['a']['mean_window'], (0.45 * 4 + 0.34) / 5)
        self.assertLess(g[strategy.STRATEGY]['members']['a']['slope'], 0)

    def test_margin_ties_reset_loss_counter_including_exact_margin(self):
        plan_until(self.manifest, 5)
        g = plan_until(self.manifest, 10, [0.402, 0.401, 0.40, 0.40, 0.40])
        self.assertEqual(g['exploit'], [])
        self.assertTrue(all(m['consecutive_losses'] == 0 for m in g[strategy.STRATEGY]['members'].values()))

    def test_mutations_deterministic_bounded_and_distinct(self):
        for donor_lr in (2e-6, 14e-6, 3e-5):
            members = copy.deepcopy(self.manifest['members'])
            members['e']['lr'] = donor_lr
            for recipients in (['a'], ['a', 'b']):
                events = strategy._mutations(self.config, members, 'e', recipients)
                self.assertEqual(events, strategy._mutations(self.config, dict(reversed(list(members.items()))), 'e', recipients))
                self.assertLessEqual(len(events), 2)
                for event in events:
                    self.assertTrue(2e-6 <= event['new_lr'] <= 3e-5)
                    self.assertNotEqual(event['new_lr'], donor_lr)
                resulting = {n: m['lr'] for n, m in members.items()}
                resulting.update({e['recipient']: e['new_lr'] for e in events})
                self.assertEqual(len(resulting), len(set(resulting.values())))

    def test_replay_after_manifest_roundtrip_and_terminal_no_copy(self):
        plan_until(self.manifest, 9)
        reloaded = PBTManifest.parse_payload(json.loads(json.dumps(self.manifest))).to_runtime_dict()
        a = plan_until(self.manifest, 10)
        b = plan_until(reloaded, 10)
        self.assertEqual(a[strategy.STRATEGY], b[strategy.STRATEGY])
        self.assertEqual(a['exploit'], b['exploit'])
        self.config['shared']['generations'] = 15
        terminal = plan_until(self.manifest, 15)
        self.assertEqual(terminal['exploit'], [])
        self.assertTrue(terminal[strategy.STRATEGY]['terminal'])

    def test_incomplete_nonfinite_and_midwindow_lr_change_fail_closed(self):
        plan_until(self.manifest, 4)
        for change in (lambda g: g['workers']['a'].update(returncode=1),
                       lambda g: g['workers']['a'].update(lr=1e-5),
                       lambda g: g['workers']['a']['metrics'].update(validation_loss=float('nan'))):
            g = generation_fixture(self.manifest, 4)
            change(g)
            with self.assertRaises(ValueError):
                plan_for_strategy(self.config, g, self.manifest['members'], self.manifest)

    def test_schema_rejects_legacy_adaptation_and_broken_foundation(self):
        for section, key, value in [('pbt', 'rollback_fraction', .1), ('pbt', 'early_stop_degraded_generations', 3),
                                    ('shared', 'lr_scheduler', 'flat'), ('shared', 'initial_optimizer_mode', 'damped'),
                                    ('shared', 'samples_per_epoch', 100), ('shared', 'freeze_batch_norm', False),
                                    ('shared', 'generations', 51)]:
            bad = copy.deepcopy(self.config)
            bad[section][key] = value
            with self.assertRaises(ValueError, msg=key):
                ResolvedPBTConfig.model_validate(bad)

    def test_copy_raw_then_only_lr_donor_unchanged_and_interrupted_copy_replays(self):
        g = plan_until(self.manifest, 10)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for n, m in self.manifest['members'].items():
                save_bundle(root, n, g['epoch'], m['lr'])
            strategy.prepare_boundary(root, self.manifest, g)
            pending = copy.deepcopy(self.manifest)
            donor = strategy.bundle_identity(strategy.bundle_paths(root / 'e', g['epoch']))
            with patch.object(strategy, 'atomic_set_optimizer_lr', side_effect=RuntimeError('interrupted after raw copy')):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    strategy.apply_window_exploits(root, self.manifest, g, root / 'manifest.json')
            self.manifest = pending
            g = self.manifest['generations'][-1]
            strategy.apply_window_exploits(root, self.manifest, g, root / 'manifest.json')
            self.assertEqual(donor, strategy.bundle_identity(strategy.bundle_paths(root / 'e', g['epoch'])))
            for event in g['exploit']:
                self.assertTrue(event['applied'])
                for part in ('state', 'optimizer', 'scaler'):
                    self.assertEqual(event['copied_checkpoint'][part]['sha256'], donor[part]['sha256'])
                    self.assertEqual(event['pre_copy'][part]['sha256'], event['pre_copy_archive'][part]['sha256'])
                source = torch.load(donor['optimizer']['path'], weights_only=False)
                target = torch.load(event['post_copy']['optimizer']['path'], weights_only=False)
                self.assertEqual(target['param_groups'][0]['lr'], event['new_lr'])
                target['param_groups'][0]['lr'] = source['param_groups'][0]['lr']
                self.assertEqual(target['param_groups'], source['param_groups'])
                for key in source['state'][0]:
                    torch.testing.assert_close(target['state'][0][key], source['state'][0][key], rtol=0, atol=0)
            before = (root / 'manifest.json').read_bytes()
            strategy.apply_window_exploits(root, self.manifest, g, root / 'manifest.json')
            self.assertEqual(before, (root / 'manifest.json').read_bytes())

    def test_protected_best_never_regresses_or_modifies_population(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protected = None
            for end, score in [(5, .40), (10, .41), (15, .40), (20, .39)]:
                g = plan_until(self.manifest, end, [score] * 5)
                for n, m in self.manifest['members'].items():
                    save_bundle(root, n, g['epoch'], m['lr'])
                before = copy.deepcopy(self.manifest['members'])
                strategy.prepare_boundary(root, self.manifest, g)
                self.assertEqual(before, self.manifest['members'])
                if end in (10, 15):
                    self.assertEqual(protected, self.manifest['protected_best'])
                    self.assertIsNone(g[strategy.STRATEGY]['protected_best_update'])
                protected = copy.deepcopy(self.manifest['protected_best'])
            self.assertEqual(protected['window_score'], .39)

    def test_runner_bypasses_legacy_actions_and_preserves_decision_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_until(self.manifest, 9)
            g = generation_fixture(self.manifest, 9)
            self.manifest['generations'].append(g)
            for n, m in self.manifest['members'].items():
                save_bundle(root, n, g['epoch'], m['lr'])
            with patch('training.pbt.runner.update_global_best'), \
                 patch('training.pbt.runner.run_generation_controller') as controller, \
                 patch('training.pbt.runner.update_generation_health') as health, \
                 patch('training.pbt.runner.apply_exploit') as legacy:
                _plan_generation_exploit(self.config, self.manifest, g, 9, False, root, root / 'manifest.json', root / 'log')
                _finalize_generation(self.config, self.manifest, g, root, root / 'manifest.json', root / 'log', 9)
                controller.assert_not_called(); health.assert_not_called(); legacy.assert_not_called()
            self.assertEqual(self.manifest['next_generation'], 10)
            self.assertLessEqual(len(g['exploit']), 2)
            self.assertEqual(g[strategy.STRATEGY]['full_epochs'], [6, 7, 8, 9, 10])
            for event in g['exploit']:
                self.assertTrue({'pre_copy', 'pre_copy_archive', 'donor_checkpoint', 'copied_checkpoint', 'post_copy', 'mutation_factor'} <= event.keys())

    def test_interrupted_preparation_is_replanned(self):
        g = plan_until(self.manifest, 10)
        g['exploit'] = None
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for n, m in self.manifest['members'].items():
                save_bundle(root, n, g['epoch'], m['lr'])
            original = strategy._copy_bundle
            calls = []
            def interrupted(source, destination):
                calls.append(destination)
                if len(calls) == 2:
                    raise OSError('archive interrupted')
                return original(source, destination)
            with patch('training.pbt.runner.update_global_best'), patch.object(strategy, '_copy_bundle', side_effect=interrupted):
                with self.assertRaisesRegex(OSError, 'archive interrupted'):
                    _plan_generation_exploit(self.config, self.manifest, g, 9, False, root, root / 'manifest.json', root / 'log')
            self.assertIsNone(g['exploit'])
            self.assertNotIn('protected_best', self.manifest)
            with patch('training.pbt.runner.update_global_best'):
                _plan_generation_exploit(self.config, self.manifest, g, 9, False, root, root / 'manifest.json', root / 'log')
            self.assertIsNotNone(g[strategy.STRATEGY]['protected_best_update'])
            self.assertEqual(len(g['exploit']), 2)

    def test_fifty_epoch_fixture_runner_verifier_comparison_and_failure_detection(self):
        # CPU checkpoint fixtures exercise orchestration and auditing, not training.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.manifest
            ensure_run_layout(root)
            epoch = self.config['shared']['initial_epoch']
            resume = dict(epoch=epoch)
            for name, member in manifest['members'].items():
                paths = save_bundle(root, name, epoch, member['lr'])
                # The initial model and raw optimizer are identical before load-LR override.
                for part in ('state', 'optimizer'):
                    paths[part].write_bytes(strategy.bundle_paths(root / 'a', epoch)[part].read_bytes())
                    resume[part + '_sha256'] = sha256(paths[part])
            self.config['shared']['initial_optimizer'] = str(root / 'legacy_optimizer.pt')
            manifest['initial_resume'] = resume
            def audit(split):
                rows = dict(count=3, unique_ids=3, repeated_ids=0, sequence_sha256=split, id_set_sha256=split)
                return dict(dataset=dict(files=[dict(path=split, rows=3)], total_rows=3, fingerprint=split),
                            consumed=rows, exhausted=True, optimizer_steps=1, batches=1,
                            traversal=[dict(scanned=rows, accepted=rows, exhausted=True, wraps=0)], prediction_sha256='prediction')
            manifest['datasets'] = dict(resolved_files={s: [dict(files=[s])] for s in ('train', 'val')})
            initial = generation_fixture(manifest, 0)['workers']['a']['metrics']
            initial.update(validation_data_audit=audit('val'))
            manifest['initial_evaluation'] = dict(status='completed', checkpoint_sha256=resume['state_sha256'], metrics=initial)
            for i in range(50):
                g = generation_fixture(manifest, i, [0.6 - m['lr'] * 10000 - i * .0001 for m in manifest['members'].values()])
                manifest['generations'].append(g)
                for name, worker in g['workers'].items():
                    save_bundle(root, name, g['epoch'], worker['lr'])
                    worker['metrics'].update(train_data_audit=audit('train'), validation_data_audit=audit('val'),
                                             train_loaded_optimizer_lr=worker['lr'])
                _plan_generation_exploit(self.config, manifest, g, i, i == 49, root, root / 'manifest.json', root / 'log')
                _finalize_generation(self.config, manifest, g, root, root / 'manifest.json', root / 'log', i)
                # Serialization must retain counter/copy/protected state used next epoch.
                manifest = PBTManifest.parse_payload(json.loads((root / 'manifest.json').read_text())).to_runtime_dict()
            def evaluate(*args):
                result = {}
                for name in args[6]:
                    metrics = manifest['best' if name == 'selected_best' else 'protected_best']['metrics'] if name in ('selected_best', 'protected_best') else manifest['generations'][-1]['workers'][name]['metrics']
                    result[name] = dict(status='completed', metrics=copy.deepcopy(metrics))
                return result
            with patch('training.pbt.runner.run_tiered_evaluation', side_effect=evaluate):
                run_final_checkpoint_evaluations(self.config, manifest, root, root / 'manifest.json', root / 'log')
            manifest['status'] = 'completed'
            atomic_json(root / 'manifest.json', manifest)
            verified = verifier.verify(root)
            self.assertTrue(verified['passed'], verified['failures'])
            comparison = verifier.compare(root, root)
            self.assertTrue(comparison['above_initial_max'])
            self.assertLessEqual(comparison['donor_copies'], 16)
            self.assertEqual(len(comparison['lr_evolution']), 10)
            self.assertEqual(set(comparison['live']['a']['metrics']), {self.config['pbt']['metric'], 'validation_loss', *strategy.WORKING_POINTS})
            for change in (lambda m: m['generations'][9]['exploit'][0].update(new_lr=1e-6),
                           lambda m: m['generations'][4][strategy.STRATEGY]['members']['a'].update(window_score=0),
                           lambda m: m['generations'][9]['exploit'][0]['copied_checkpoint']['optimizer'].update(sha256='bad'),
                           lambda m: m['generations'][9][strategy.STRATEGY].update(protected_best_score=9),
                           lambda m: m['final_evaluations']['control']['protected_best']['metrics'].update(validation_loss=99)):
                broken = copy.deepcopy(manifest)
                change(broken)
                atomic_json(root / 'manifest.json', broken)
                self.assertFalse(verifier.verify(root)['passed'])
            atomic_json(root / 'manifest.json', manifest)
            self.assertTrue(verifier.verify(root, through=10)['passed'])


if __name__ == '__main__':
    unittest.main()
