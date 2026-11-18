import copy
import json
from pathlib import Path
import tempfile
import unittest

from tests.test_windowed_pbt_v2 import configuration_fixture, generation_fixture, manifest_fixture, save_bundle
from training.pbt.models.config import ResolvedPBTConfig
from training.pbt.planning import window_probe as probe
from training.pbt.planning import windowed_pbt_v2 as strategy
from training.pbt.planning.dispatch import plan_for_strategy


def probe_configuration(**options):
    config = configuration_fixture()
    config['pbt'][strategy.STRATEGY].update(window_epochs=2, score_epochs=1)
    config['pbt']['exploit_interval_generations'] = 2
    config['shared']['generations'] = 10
    config['pbt'][strategy.STRATEGY]['probe'] = dict(lr_factor=0.1, samples=4096, **options)
    return config


class FakeBackend:
    """Writes one metric per probe: lower for a lower base LR unless `values` says otherwise."""

    def __init__(self, values):
        self.values, self.calls = values, []

    def run_generation(self, config, root, manifest, record, names, manifest_path):
        self.calls.append(dict(config=copy.deepcopy(config), root=Path(root), manifest=copy.deepcopy(manifest),
                               record=copy.deepcopy(record), names=list(names)))
        keys = [config['pbt']['metric'], 'validation_loss', *strategy.WORKING_POINTS]
        for name in names:
            record['workers'][name].update(status='completed', returncode=0, lr=manifest['members'][name]['lr'],
                                           metrics={k: self.values[name] for k in keys}, log=f'{name}.log')


class WindowProbeTest(unittest.TestCase):
    def setUp(self):
        self.config = probe_configuration()
        self.manifest = manifest_fixture(self.config)

    def test_due_only_at_boundaries_that_can_copy(self):
        due = [i for i in range(10) if probe.probe_due(self.config, i)]
        self.assertEqual(due, [1, 3, 5, 7])  # generation 9 ends the run, so its decision copies nothing
        plain = configuration_fixture()
        self.assertFalse(any(probe.probe_due(plain, i) for i in range(50)))
        self.config['pbt'][strategy.STRATEGY]['final_decay'] = dict(decay_fraction=0.3, final_factor=0.01)
        self.assertEqual([i for i in range(10) if probe.probe_due(self.config, i)], [1, 3, 5])

    def test_schema_requires_one_score_epoch(self):
        ResolvedPBTConfig.model_validate(self.config)
        broken = copy.deepcopy(self.config)
        broken['pbt'][strategy.STRATEGY]['score_epochs'] = 2
        with self.assertRaisesRegex(ValueError, 'score_epochs: 1'):
            ResolvedPBTConfig.model_validate(broken)

    def test_probes_copy_checkpoints_lower_the_lr_and_cap_the_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            g = generation_fixture(self.manifest, 1)
            self.manifest['generations'] = [generation_fixture(self.manifest, 0), g]
            for name, member in self.manifest['members'].items():
                save_bundle(run, name, g['epoch'], member['lr'])
            backend = FakeBackend({n: 0.3 + 0.01 * i for i, n in enumerate(sorted(self.manifest['members']))})
            result = probe.run_window_probes(self.config, backend, run, self.manifest, g, run / 'manifest.json')
            call = backend.calls[0]
            self.assertEqual(call['config']['shared']['samples_per_epoch'], 4096)
            self.assertIsNone(self.config['shared'].get('samples_per_epoch'))  # only the probe is capped
            self.assertEqual(call['record']['index'], 2)
            for name, member in self.manifest['members'].items():
                self.assertAlmostEqual(call['manifest']['members'][name]['lr'], 0.1 * member['lr'])
                self.assertTrue(strategy.bundle_paths(call['root'] / name, g['epoch'])['state'].exists())
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(json.loads((run / 'manifest.json').read_text())['generations'][1]['probes']['status'], 'completed')
            # A completed probe is not repeated.
            probe.run_window_probes(self.config, backend, run, self.manifest, g, run / 'manifest.json')
            self.assertEqual(len(backend.calls), 1)

    def test_decision_ranks_on_probes_and_keeps_population_metrics(self):
        names = sorted(self.manifest['members'])
        # Raw window values favour the last member; the probes favour the first one.
        raw = [0.45, 0.44, 0.43, 0.42, 0.40]
        probes = {n: 0.30 + 0.01 * i for i, n in enumerate(names)}
        for index in range(4):
            g = generation_fixture(self.manifest, index, raw)
            self.manifest['generations'].append(g)
            if probe.probe_due(self.config, index):
                g['probes'] = dict(status='completed', members={n: dict(metrics={k: probes[n] for k in
                                   g['workers'][n]['metrics']}) for n in names})
            g['ranking'], g['exploit'] = plan_for_strategy(self.config, g, self.manifest['members'], self.manifest)
            g['status'] = 'completed'
        decision = self.manifest['generations'][3][strategy.STRATEGY]
        self.assertEqual(decision['scored_on'], 'probe')
        self.assertEqual(decision['donor'], names[0])
        self.assertTrue(all(e['donor'] == names[0] for e in self.manifest['generations'][3]['exploit']))
        self.assertEqual(self.manifest['generations'][3]['workers'][names[-1]]['metrics'][self.config['pbt']['metric']], 0.40)

    def test_decision_without_completed_probes_fails_closed(self):
        g0 = generation_fixture(self.manifest, 0)
        g1 = generation_fixture(self.manifest, 1)
        self.manifest['generations'] = [g0, g1]
        with self.assertRaisesRegex(ValueError, 'completed probes'):
            plan_for_strategy(self.config, g1, self.manifest['members'], self.manifest)


if __name__ == '__main__':
    unittest.main()
