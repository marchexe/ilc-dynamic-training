import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.test_windowed_pbt_v2 import configuration_fixture, manifest_fixture, generation_fixture, save_bundle
from training.pbt import runner
from training.pbt.config import contract_fingerprint
from training.pbt.execution.backend import LocalWeaverBackend
from training.pbt.state import continuation
from training.pbt.planning import windowed_pbt_v2 as strategy
from training.pbt.reporting import ensure_run_layout
from training.runtime import atomic_json, sha256
from scripts.validation import verify_fixed_lr as verifier


def audit(split):
    rows = dict(count=3, unique_ids=3, repeated_ids=0, sequence_sha256=split, id_set_sha256=split)
    return dict(dataset=dict(files=[dict(path=split, rows=3)], total_rows=3, fingerprint=split),
                consumed=rows, exhausted=True, optimizer_steps=1, batches=1,
                traversal=[dict(scanned=rows, accepted=rows, exhausted=True, wraps=0)], prediction_sha256='prediction')


def workers(config, backend, run, manifest, g, path):
    g.update(generation_fixture(manifest, g['index']))
    for name, worker in g['workers'].items():
        save_bundle(run, name, g['epoch'], worker['lr'])
        worker['metrics'].update(train_data_audit=audit('train'), validation_data_audit=audit('val'),
                                 train_loaded_optimizer_lr=float(format(worker['lr'], '.6g')))


def final_evaluations(config, manifest, run, path, log):
    final = {}
    for name in [*manifest['members'], 'selected_best', 'protected_best']:
        if name in ('selected_best', 'protected_best'):
            record = manifest['best' if name == 'selected_best' else 'protected_best']
            state, metrics = Path(record['state_path']), record['metrics']
        else:
            state = strategy.bundle_paths(run / name, manifest['generations'][-1]['epoch'])['state']
            metrics = manifest['generations'][-1]['workers'][name]['metrics']
        final[name] = dict(status='completed', checkpoint_sha256=sha256(state), metrics=copy.deepcopy(metrics))
    manifest['final_evaluations'] = dict(control=final)


def make_source(run, horizon=15):
    config = configuration_fixture()
    config['shared']['generations'] = horizon
    config['shared']['initial_optimizer'] = str(run / 'legacy_optimizer.pt')
    manifest = manifest_fixture(config)
    manifest['fingerprint'] = contract_fingerprint(config)
    ensure_run_layout(run)
    epoch = config['shared']['initial_epoch']
    resume = dict(epoch=epoch)
    for name, member in manifest['members'].items():
        member['parent'] = None
        paths = save_bundle(run, name, epoch, member['lr'])
        for part in ('state', 'optimizer'):
            paths[part].write_bytes(strategy.bundle_paths(run / 'a', epoch)[part].read_bytes())
            resume[part + '_sha256'] = sha256(paths[part])
    manifest['initial_resume'] = resume
    metrics = generation_fixture(manifest, 0)['workers']['a']['metrics']
    metrics['validation_data_audit'] = audit('val')
    manifest.update(initial_evaluation=dict(status='completed', checkpoint_sha256=resume['state_sha256'], metrics=metrics),
                    datasets=dict(resolved_files={s: [dict(files=[s])] for s in ('train', 'val')}),
                    baseline_evaluation={}, metric_definition={})
    for i in range(horizon):
        g = runner._generation_record_for(config, manifest, run / 'manifest.json', i)
        workers(config, None, run, manifest, g, run / 'manifest.json')
        runner._plan_generation_exploit(config, manifest, g, i, i == horizon - 1, run, run / 'manifest.json', run / 'log')
        runner._finalize_generation(config, manifest, g, run, run / 'manifest.json', run / 'log', i)
    final_evaluations(config, manifest, run, None, None)
    manifest['status'] = 'completed'
    atomic_json(run / 'manifest.json', manifest)
    return manifest


def continuation_config(run, source, destination, end):
    config = copy.deepcopy(source['config'])
    config.update(experiment_name=destination.name, output_root=str(destination.parent))
    config['shared']['generations'] = end
    _, digest = continuation.source_state(run, source)
    config['continuation'] = dict(source_run=str(run), source_manifest_sha256=sha256(run / 'manifest.json'),
                                source_state_sha256=digest)
    return config


class ContinuationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source_run = self.root / 'source'
        self.source = make_source(self.source_run)
        self.destination = self.root / 'continued'
        self.config = continuation_config(self.source_run, self.source, self.destination, 25)

    def test_no_repeated_boundary_then_second_extension_and_verifier(self):
        original = {p: p.read_bytes() for p in self.source_run.rglob('*') if p.is_file()}
        self.assertTrue(verifier.verify(self.source_run)['passed'])
        args = SimpleNamespace(resume=False, dry_run=False)
        for source_run, destination, end in [(self.source_run, self.destination, 25),
                                             (self.destination, self.root / 'again', 35)]:
            source = json.loads((source_run / 'manifest.json').read_text())
            config = continuation_config(source_run, source, destination, end)
            context = (config, LocalWeaverBackend(), destination, destination / 'manifest.json',
                       destination / 'log', ['fixture'], contract_fingerprint(config))
            with patch.object(runner, '_resolve_run_context', return_value=context), \
                 patch.object(runner, 'initial_manifest', side_effect=lambda c, *a: dict(manifest_fixture(c), fingerprint=contract_fingerprint(c))), \
                 patch.object(runner, '_run_generation_workers', side_effect=workers) as work, \
                 patch.object(runner, 'run_final_checkpoint_evaluations', side_effect=final_evaluations), \
                 patch.object(runner, 'write_canonical_outputs', return_value={'report': 'fixture'}), \
                 patch.object(runner, 'run_initial_evaluation') as initial:
                runner.run(args)
                initial.assert_not_called()
                indices = [call.args[4]['index'] for call in work.call_args_list]
                self.assertEqual(indices, list(range(source['next_generation'], end)))
            manifest = json.loads((destination / 'manifest.json').read_text())
            start = source['next_generation']
            self.assertEqual(manifest['generations'][:start], source['generations'])
            boundaries = [g['index'] + 1 for g in manifest['generations'][start:] if strategy.STRATEGY in g]
            self.assertEqual(boundaries, [start + 5, start + 10])
            first = manifest['generations'][start + 4][strategy.STRATEGY]
            previous = source['generations'][-1][strategy.STRATEGY]
            for name, row in first['members'].items():
                if row['donor_gap'] > config['pbt'][strategy.STRATEGY]['decision_margin']:
                    self.assertEqual(row['consecutive_losses'], previous['members'][name]['next_loss_count'] + 1)
            verified = verifier.verify(destination)
            self.assertTrue(verified['passed'], verified['failures'])
            broken = copy.deepcopy(manifest)
            broken['generations'][start - 1][strategy.STRATEGY]['terminal'] = False
            atomic_json(destination / 'manifest.json', broken)
            with self.assertRaisesRegex(ValueError, 'history'):
                verifier.verify(destination)
            atomic_json(destination / 'manifest.json', manifest)
        self.assertEqual(original, {p: p.read_bytes() for p in original})

    def test_interrupted_bootstrap_retries_exact_bundles_and_never_recopies(self):
        source, evidence = continuation.plan_continuation(self.config)
        manifest = continuation.inherit_history(manifest_fixture(self.config), source, evidence)
        ensure_run_layout(self.destination)
        path = self.destination / 'manifest.json'
        atomic_json(path, manifest)
        real_copy = continuation._copy_bundle
        calls = []
        def interrupted(*args):
            calls.append(None)
            if len(calls) == 2:
                raise RuntimeError('interrupted')
            return real_copy(*args)
        with patch.object(continuation, '_copy_bundle', side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                continuation.bootstrap_continuation(self.destination, manifest, path)
        self.assertFalse(json.loads(path.read_text())['continuation']['initialized'])
        continuation.bootstrap_continuation(self.destination, manifest, path)
        for name, bundle in evidence['members'].items():
            for part, item in bundle.items():
                self.assertEqual(sha256(self.destination / name / Path(item['path']).name), item['sha256'])
        self.assertEqual(manifest['protected_best'], source['protected_best'])
        manifest['next_generation'] += 1
        with patch.object(continuation, '_copy_bundle') as copier:
            continuation.bootstrap_continuation(self.destination, manifest, path)
            copier.assert_not_called()

    def test_dry_run_is_read_only_and_uses_live_lr_epoch_and_seed(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            runner._print_dry_run_commands(self.config, LocalWeaverBackend(), self.destination)
        self.assertFalse(self.destination.exists())
        text = output.getvalue()
        start = self.source['next_generation']
        self.assertNotIn('[initial_evaluation]', text)
        self.assertIn('--load-epoch ' + str(self.source['generations'][-1]['epoch']), text)
        self.assertIn('--seed ' + str(self.config['shared']['seed'] + start), text)
        for member in self.source['members'].values():
            self.assertIn('--start-lr ' + str(member['lr']), text)

    def test_changed_contract_state_source_or_output_fail_closed(self):
        for edit in (lambda c: c['shared'].update(seed=999),
                     lambda c: c['pbt']['windowed_pbt_v2'].update(decision_margin=.003),
                     lambda c: c['shared'].update(generations=15),
                     lambda c: c.update(experiment_name=self.source_run.name),
                     lambda c: c['continuation'].update(source_manifest_sha256='0' * 64),
                     lambda c: c['continuation'].update(source_state_sha256='0' * 64)):
            bad = copy.deepcopy(self.config); edit(bad)
            with self.assertRaises(ValueError):
                continuation.plan_continuation(bad)
        path = strategy.bundle_paths(self.source_run / 'a', self.source['generations'][-1]['epoch'])['scaler']
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'state changed'):
            continuation.plan_continuation(self.config)


if __name__ == '__main__':
    unittest.main()
