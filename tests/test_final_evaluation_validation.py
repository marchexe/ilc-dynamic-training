import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from tests.helpers import PROJECT_DIR
from training.pbt import runner
from training.pbt.execution import backend
from training.runtime import sha256
from validation.results import require_checkpoint_result


class FinalEvaluationValidationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.state = self.root / 'checkpoint.pt'
        self.state.write_bytes(b'model')
        self.digest = sha256(self.state)
        self.metric = 'configured_composite'

    def record(self, value=.5):
        return dict(status='completed', metrics={self.metric: value}, checkpoint=str(self.state),
                    checkpoint_sha256=self.digest, returncode=0)

    def require(self, record):
        return require_checkpoint_result(record, self.state, self.digest, 'control/selected_best', self.metric)

    def test_accepts_valid_current_and_legacy_formats(self):
        for value in [0, .5, '0.5']:
            with self.subTest(value=value):
                record = dict(status='completed', metrics={self.metric: value})
                self.assertIs(self.require(record), record)
                self.assertEqual(record['checkpoint_sha256'], self.digest)
        self.require(self.record())

    def test_rejects_missing_nan_infinities_and_invalid_primary(self):
        for value in [None, float('nan'), float('inf'), float('-inf'), 'bad', [], True, 10**400]:
            with self.subTest(value=str(value)):
                record = self.record(value)
                with self.assertRaisesRegex(RuntimeError, 'configured_composite'):
                    self.require(record)
                self.assertEqual(record['status'], 'failed')
        for metrics in [None, {}, [], {'another_metric': .1}]:
            with self.subTest(metrics=metrics):
                with self.assertRaises(RuntimeError):
                    self.require(dict(status='completed', metrics=metrics))

    def test_rejects_inconsistent_process_checkpoint_and_hash(self):
        for changes in [dict(status='failed'), dict(returncode=1), dict(checkpoint=str(self.root / 'other.pt')),
                        dict(checkpoint_sha256='stale hash')]:
            with self.subTest(changes=changes):
                record = self.record()
                record.update(changes)
                with self.assertRaises(RuntimeError):
                    self.require(record)
                self.assertEqual(record['status'], 'failed')
        self.state.write_bytes(b'changed model')
        with self.assertRaisesRegex(RuntimeError, 'checkpoint changed'):
            self.require(self.record())

    def test_log_must_contain_same_finite_configured_metric(self):
        log = self.root / 'evaluation.log'
        log.write_text('Eval AvgLoss: 0.5, AvgAcc: 0.9\n')
        record = dict(status='completed', metrics={'validation_loss': .5}, log=str(log))
        require_checkpoint_result(record, self.state, self.digest, 'control/member', 'validation_loss')
        for content in ['Eval AvgLoss: 0.7, AvgAcc: 0.9\n', 'successful exit without evaluation\n']:
            log.write_text(content)
            record['status'] = 'completed'
            with self.assertRaisesRegex(RuntimeError, 'log and recorded'):
                require_checkpoint_result(record, self.state, self.digest, 'control/member', 'validation_loss')
        log.unlink()
        record['status'] = 'completed'
        with self.assertRaises(RuntimeError):
            require_checkpoint_result(record, self.state, self.digest, 'control/member', 'validation_loss')

    def test_prediction_and_traversal_artifacts_must_agree(self):
        import awkward as ak
        import numpy as np
        predictions = self.root / 'predictions.parquet'
        scores = np.array([[.1, .2, .7], [.3, .2, .5], [.4, .1, .5]], dtype='<f4')
        ak.to_parquet(ak.Array(dict(scores=scores)), predictions)
        audit = dict(exhausted=True, consumed=dict(count=3, unique_ids=3, repeated_ids=0),
                     prediction_sha256=hashlib.sha256(scores.tobytes()).hexdigest())
        record = self.record()
        record.update(prediction_path=str(predictions))
        record['metrics']['validation_data_audit'] = audit
        self.require(record)
        for changes in [dict(exhausted=False), dict(consumed=dict(count=3, unique_ids=2, repeated_ids=1)),
                        dict(consumed=dict(count=0, unique_ids=0, repeated_ids=0)), dict(prediction_sha256='wrong'),
                        dict(traversal=[dict(exhausted=True, wraps=0, accepted=dict(count=2))])]:
            with self.subTest(changes=changes):
                invalid = copy.deepcopy(record)
                invalid['metrics']['validation_data_audit'].update(changes)
                with self.assertRaises(RuntimeError):
                    self.require(invalid)
        ak.to_parquet(ak.Array(dict(other=[1, 2, 3])), predictions)
        with self.assertRaises(RuntimeError):
            self.require(copy.deepcopy(record))
        predictions.unlink()
        with self.assertRaises(RuntimeError):
            self.require(record)

    def test_final_backend_uses_primary_instead_of_curve_presence(self):
        config = dict(slots=['0'], pbt=dict(metric=self.metric), shared={})
        for value, expected in [(.5, 'completed'), (float('nan'), 'failed'), (float('inf'), 'failed'),
                                (float('-inf'), 'failed'), (None, 'failed')]:
            with self.subTest(value=value):
                metrics = {self.metric: value, 'validation_bkg_rejection_at_eff': {'pairs': {}}}
                with patch.object(backend, 'make_tiered_evaluation_command', return_value=([sys.executable, '-c', 'pass'], None)), \
                     patch.object(backend, 'read_metrics', return_value=metrics):
                    results = backend.run_tiered_evaluation(config, self.root, 0, 'final_control', 'data', 'val',
                                                           dict(arm=self.state), self.root / 'pbt.log')
                self.assertEqual(results['arm']['status'], expected)

    def test_real_cpu_evaluation_cannot_reuse_stale_log_after_successful_exit(self):
        config = dict(slots=['0'], pbt=dict(metric='validation_loss'), shared={})
        log = self.root / 'logs/tiered_evaluation/final_control/arm/generation-000.log'
        log.parent.mkdir(parents=True)
        log.write_text('Eval AvgLoss: 0.5, AvgAcc: 0.9\n')
        with patch.object(backend, 'make_tiered_evaluation_command', return_value=([sys.executable, '-c', 'pass'], None)):
            results = backend.run_tiered_evaluation(config, self.root, 0, 'final_control', 'data', 'val',
                                                   dict(arm=self.state), self.root / 'pbt.log')
        self.assertEqual(results['arm']['returncode'], 0)
        self.assertEqual(results['arm']['status'], 'failed')
        self.assertIsNone(results['arm']['metrics'])

    def test_failure_is_persisted_and_blocks_finalization_and_publication(self):
        arm = self.root / 'arm'
        arm.mkdir()
        (arm / 'net_epoch-1_state.pt').write_bytes(b'arm')
        config = dict(shared=dict(dataset='data', validation_suffix='val'),
                      pbt=dict(evaluate_final_checkpoints=True, metric=self.metric))
        manifest = dict(status='running', config=config, members=dict(arm={}),
                        generations=[dict(index=0, epoch=1)])
        path = self.root / 'manifest.json'
        invalid = dict(arm=self.record(float('nan')))
        with patch.object(runner, 'run_tiered_evaluation', return_value=invalid), \
             patch.object(runner, 'write_canonical_outputs') as publish:
            with self.assertRaisesRegex(RuntimeError, 'Required final evaluation failed'):
                runner._finalize_run(config, manifest, self.root, path, self.root / 'pbt.log', time.monotonic())
            publish.assert_not_called()
        saved = json.loads(path.read_text())
        self.assertNotEqual(saved['status'], 'completed')
        self.assertEqual(saved['final_evaluations']['control']['arm']['status'], 'failed')
        from scripts.publish.export_run import export
        from types import SimpleNamespace
        with self.assertRaisesRegex(RuntimeError, 'Required final evaluation failed'):
            export(SimpleNamespace(run=self.root, related_run=[], output_root=self.root / 'export',
                                   docs_dir=self.root / 'docs', slug='invalid'))
        self.assertFalse((self.root / 'export/invalid').exists())
