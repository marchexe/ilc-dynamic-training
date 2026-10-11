import copy
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

from tests.helpers import PROJECT_DIR
from training import checkpoints
from training.pbt import metrics
from training.pbt.models.events import GlobalBestRollbackEvent
from training.pbt.state.checkpointing import seed_initial_global_best
from training.runtime import atomic_json, sha256


class SelectedBestBundleTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.member = self.root / 'arm'
        self.member.mkdir()
        self.config = dict(shared=dict(use_amp=True, amp_dtype='fp16', training_controller=None),
                           pbt=dict(metric='primary', mode='min'))
        self.manifest = dict(config=self.config, members=dict(arm=dict(lr=1e-5)), best=None)
        self.path = self.root / 'manifest.json'

    def save(self, epoch, score):
        for part in ('state', 'optimizer', 'scaler'):
            (self.member / f'net_epoch-{epoch}_{part}.pt').write_bytes(f'{epoch}/{part}'.encode())
        return dict(index=epoch, epoch=epoch, ranking=['arm'], workers=dict(arm=dict(metrics=dict(primary=score))))

    def update(self, generation):
        return metrics.update_global_best(self.root, self.manifest, generation, self.path)

    def test_creation_binds_all_components_and_metadata_to_generation(self):
        self.assertTrue(self.update(self.save(1, .5)))
        best = self.manifest['best']
        self.assertEqual(best['scaler_status'], 'present')
        self.assertEqual(set(best['checkpoint']), {'state', 'optimizer', 'scaler'})
        self.assertEqual(json.loads(Path(best['metadata_path']).read_text()), best)
        self.assertEqual(json.loads(self.path.read_text())['best'], best)
        for part in best['checkpoint']:
            self.assertEqual(best['checkpoint'][part]['sha256'], best['source_checkpoint'][part]['sha256'])
            self.assertEqual(Path(best[part + '_path']).read_bytes(), f'1/{part}'.encode())
        self.assertEqual(checkpoints.selected_best_bundle(best, require_scaler=True), best['checkpoint'])

    def test_later_selection_and_member_overwrite_preserve_previous_archive(self):
        self.update(self.save(1, .5))
        previous = copy.deepcopy(self.manifest['best'])
        self.update(self.save(2, .4))
        self.save(1, .8)
        (self.member / 'net_epoch-1_scaler.pt').write_bytes(b'exploited scaler')
        self.assertNotEqual(previous['state_path'], self.manifest['best']['state_path'])
        self.assertEqual(checkpoints.selected_best_bundle(previous), previous['checkpoint'])

    def test_missing_required_component_does_not_select_or_synthesize(self):
        for part in ('state', 'optimizer', 'scaler'):
            with self.subTest(part=part):
                generation = self.save(1, .5)
                (self.member / f'net_epoch-1_{part}.pt').unlink()
                with self.assertRaises(FileNotFoundError):
                    self.update(generation)
                self.assertIsNone(self.manifest['best'])
                self.assertFalse(self.path.exists())
                self.assertEqual(list((self.root / 'checkpoints').glob('*/global_best_metadata.json')), [])

    def test_required_controller_is_archived_or_rejected(self):
        generation = self.save(1, .5)
        self.config['shared']['training_controller'] = 'controller.yaml'
        with self.assertRaisesRegex(FileNotFoundError, 'controller'):
            self.update(generation)
        (self.member / 'net_epoch-1_controller.pt').write_bytes(b'controller')
        self.update(generation)
        self.assertEqual(Path(self.manifest['best']['controller_path']).read_bytes(), b'controller')
        checkpoints.selected_best_bundle(self.manifest['best'])

    def test_copy_failure_preserves_selected_bundle_and_retry_recovers(self):
        self.update(self.save(1, .5))
        previous = copy.deepcopy(self.manifest['best'])
        before = self.path.read_bytes()
        generation = self.save(2, .4)
        copy_file = checkpoints.shutil.copy2
        def fail_scaler(source, destination):
            copy_file(source, destination)
            if str(source).endswith('_scaler.pt'):
                raise OSError('interrupted scaler copy')
        with patch.object(checkpoints.shutil, 'copy2', side_effect=fail_scaler):
            with self.assertRaisesRegex(OSError, 'interrupted'):
                self.update(generation)
        self.assertEqual(self.manifest['best'], previous)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list((self.root / 'checkpoints').glob('.global-best-*')), [])
        self.assertTrue(self.update(generation))
        self.assertEqual(checkpoints.selected_best_bundle(previous), previous['checkpoint'])

    def test_failed_directory_commit_leaves_no_visible_new_bundle(self):
        generation = self.save(1, .5)
        replace = checkpoints.os.replace
        def fail_directory(source, destination):
            if Path(source).is_dir():
                raise OSError('interrupted directory commit')
            return replace(source, destination)
        with patch.object(checkpoints.os, 'replace', side_effect=fail_directory):
            with self.assertRaisesRegex(OSError, 'directory commit'):
                self.update(generation)
        self.assertEqual(list((self.root / 'checkpoints').iterdir()), [])
        self.assertIsNone(self.manifest['best'])
        self.assertTrue(self.update(generation))

    def test_abrupt_process_exit_during_staging_keeps_old_selection_recoverable(self):
        self.update(self.save(1, .5))
        previous = copy.deepcopy(self.manifest['best'])
        generation = self.save(2, .4)
        script = '''
import json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from training import checkpoints
from training.pbt.metrics import update_global_best
root = Path(sys.argv[1])
manifest = json.loads((root / 'manifest.json').read_text())
copy_file = checkpoints.shutil.copy2
def interrupt(source, destination):
    copy_file(source, destination)
    if str(source).endswith('_optimizer.pt'):
        os._exit(17)
checkpoints.shutil.copy2 = interrupt
generation = dict(index=2, epoch=2, ranking=['arm'], workers=dict(arm=dict(metrics=dict(primary=.4))))
update_global_best(root, manifest, generation, root / 'manifest.json')
'''
        result = subprocess.run([sys.executable, '-c', script, str(self.root), str(PROJECT_DIR / 'scripts')],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertEqual(json.loads(self.path.read_text())['best'], previous)
        self.assertEqual(checkpoints.selected_best_bundle(previous), previous['checkpoint'])
        self.assertTrue(list((self.root / 'checkpoints').glob('.global-best-*')))
        self.assertEqual(len(list((self.root / 'checkpoints').glob('global-best-*'))), 1)
        self.assertTrue(self.update(generation))

    def test_exported_bundle_matches_selection_and_copy_corruption_is_rejected(self):
        from scripts.publish import export_run
        self.update(self.save(1, .5))
        self.manifest.update(method='cadenced_pbt_v1', generations=[dict(epoch=1, ranking=['arm'],
                            workers=dict(arm=dict(metrics=dict(primary=.5))))])
        models, resume = export_run.export_models(self.root, self.manifest, self.root / 'export', resume_bundle=None)
        for record in resume['assets']:
            self.assertEqual(record['sha256'], self.manifest['best']['checkpoint'][record['kind']]['sha256'])
        copy_file = export_run.shutil.copy2
        def corrupt(source, destination):
            result = copy_file(source, destination)
            if str(destination).endswith('global_best_scaler.pt'):
                Path(destination).write_bytes(b'damaged scaler')
            return result
        with patch.object(export_run.shutil, 'copy2', side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                export_run.export_models(self.root, self.manifest, self.root / 'broken_export', resume_bundle=None)
        checkpoints.selected_best_bundle(self.manifest['best'])

    def test_manifest_commit_failure_keeps_previous_selection_and_can_retry(self):
        self.update(self.save(1, .5))
        previous = copy.deepcopy(self.manifest['best'])
        before = self.path.read_bytes()
        generation = self.save(2, .4)
        with patch.object(metrics, 'atomic_json', side_effect=OSError('manifest commit failed')):
            with self.assertRaises(OSError):
                self.update(generation)
        self.assertEqual(self.manifest['best'], previous)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertTrue(self.update(generation))

    def test_changing_source_during_copy_is_rejected(self):
        generation = self.save(1, .5)
        copy_file = checkpoints.shutil.copy2
        def mutate(source, destination):
            result = copy_file(source, destination)
            if str(source).endswith('_optimizer.pt'):
                Path(source).write_bytes(b'next generation optimizer')
            return result
        with patch.object(checkpoints.shutil, 'copy2', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                self.update(generation)
        self.assertIsNone(self.manifest['best'])

    def test_sources_from_different_epochs_are_rejected(self):
        self.save(1, .5)
        self.save(2, .4)
        with self.assertRaisesRegex(ValueError, 'different epoch'):
            checkpoints.archive_selected_best(self.root / 'checkpoints',
                dict(state=self.member / 'net_epoch-1_state.pt', optimizer=self.member / 'net_epoch-2_optimizer.pt',
                     scaler=self.member / 'net_epoch-2_scaler.pt'),
                dict(generation=2, epoch=2, member='arm'), require_scaler=True)

    def test_corrupt_bundle_is_rejected_on_recovery_and_rollback(self):
        generation = self.save(1, .5)
        self.update(generation)
        Path(self.manifest['best']['scaler_path']).write_bytes(b'wrong generation')
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            self.update(generation)
        event = GlobalBestRollbackEvent(source='global_best', recipient='arm', donor='arm', new_lr=1e-5, global_best_generation=1)
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            event.donor_paths(self.root, self.member, 1, self.manifest)

    def test_missing_archive_metadata_or_scaler_is_rejected(self):
        self.update(self.save(1, .5))
        best = self.manifest['best']
        metadata = Path(best['metadata_path'])
        original = metadata.read_bytes()
        metadata.unlink()
        with self.assertRaises(FileNotFoundError):
            checkpoints.selected_best_bundle(best)
        metadata.write_bytes(original)
        Path(best['scaler_path']).unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'scaler'):
            checkpoints.selected_best_bundle(best)

    def test_metadata_generation_mismatch_is_rejected(self):
        self.update(self.save(1, .5))
        best = self.manifest['best']
        other = dict(best, generation=2)
        atomic_json(Path(best['metadata_path']), other)
        with self.assertRaisesRegex(ValueError, 'generation'):
            checkpoints.selected_best_bundle(best)

    def test_legacy_missing_scaler_is_explicit_and_full_resume_rejected(self):
        generation = self.save(17, .5)
        (self.member / 'net_epoch-17_scaler.pt').unlink()
        self.config['shared'].update(initial_state=str(self.member / 'net_epoch-17_state.pt'),
                                     initial_optimizer=str(self.member / 'net_epoch-17_optimizer.pt'), initial_epoch=17)
        self.config['population'] = [dict(start_lr=1e-5)]
        self.config['pbt'].update(baseline_guard_seed_initial_best=True, baseline_metric_value=.5)
        self.assertTrue(seed_initial_global_best(self.config, self.root, self.manifest))
        best = self.manifest['best']
        self.assertEqual(best['scaler_status'], 'legacy_missing')
        self.assertIsNone(best['scaler_path'])
        self.assertEqual(set(checkpoints.selected_best_bundle(best)), {'state', 'optimizer'})
        with self.assertRaisesRegex(FileNotFoundError, 'AMP scaler'):
            checkpoints.selected_best_bundle(best, require_scaler=True)
        self.assertFalse(seed_initial_global_best(self.config, self.root, self.manifest))

    def test_legacy_complete_archive_keeps_original_bytes(self):
        self.save(1, .5)
        legacy = dict(state_path=str(self.member / 'net_epoch-1_state.pt'),
                      optimizer_path=str(self.member / 'net_epoch-1_optimizer.pt'))
        before = {p: sha256(p) for p in self.member.iterdir()}
        identity = checkpoints.selected_best_bundle(legacy, require_scaler=True)
        self.assertEqual(set(identity), {'state', 'optimizer', 'scaler'})
        self.assertEqual(before, {p: sha256(p) for p in self.member.iterdir()})

    def test_initial_optimizer_transform_preserves_source_and_real_scaler(self):
        import torch
        from training.pbt.state.optimizer_state import prepare_initial_optimizer
        source = self.member / 'net_epoch-17_optimizer.pt'
        torch.save(dict(state={0: dict(exp_avg=torch.tensor([2.]), exp_avg_sq=torch.tensor([4.]), step=3)},
                        param_groups=[dict(lr=1e-5)]), source)
        state = self.member / 'net_epoch-17_state.pt'
        state.write_bytes(b'reference model')
        scaler = self.member / 'net_epoch-17_scaler.pt'
        torch.save(dict(scale=1024., _growth_tracker=7), scaler)
        digest = sha256(source)
        record = checkpoints.archive_selected_best(self.root / 'checkpoints', dict(state=state, optimizer=source, scaler=scaler),
            dict(generation=-1, epoch=17, member='initial_resume'), require_scaler=True,
            optimizer_transform=lambda p: prepare_initial_optimizer(p, p, mode='damped', damping_factor=.25))
        self.assertEqual(sha256(source), digest)
        self.assertEqual(record['source_checkpoint']['optimizer']['sha256'], digest)
        self.assertNotEqual(record['checkpoint']['optimizer']['sha256'], digest)
        self.assertEqual(torch.load(record['scaler_path'], weights_only=False), dict(scale=1024., _growth_tracker=7))
        checkpoints.selected_best_bundle(record, require_scaler=True)
