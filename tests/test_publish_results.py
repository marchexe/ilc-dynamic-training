"""Public results export and path-free bundle copies."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.helpers import PROJECT_DIR
from scripts.publish import export_results_summary as results
from scripts.publish import public_copies

SOURCE = Path(os.environ.get('ILC_RESULTS_SOURCE_ROOT', PROJECT_DIR))


class PublicCopiesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.bundle = Path(self.tmp.name) / 'bundle'
        self.bundle.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_checkout_prefix_becomes_repository_relative(self):
        (self.bundle / 'metrics.csv').write_text('path,value\n/ckout/x/runs/pbt/a/net.pt,0.5\n')
        (self.bundle / 'lr_history.csv').write_text('epoch,lr\n1,3e-06\n')
        with patch.object(public_copies, 'PROJECT_DIR', Path(self.tmp.name)):
            records = public_copies.export(self.bundle, '/ckout/x/')
        self.assertEqual((self.bundle / 'public/metrics.csv').read_text(), 'path,value\nruns/pbt/a/net.pt,0.5\n')
        self.assertEqual((self.bundle / 'public/lr_history.csv').read_text(), 'epoch,lr\n1,3e-06\n')
        manifest = json.loads((self.bundle / 'public/MANIFEST.json').read_text())
        self.assertEqual([r['redacted_occurrences'] for r in manifest['files']], [1, 0])
        self.assertEqual(records, manifest['files'])
        self.assertNotIn('/ckout/', json.dumps(manifest))

    def test_refuses_when_an_absolute_path_remains(self):
        (self.bundle / 'config.yaml').write_text('dataset: /data/elsewhere/file.parquet\n')
        with patch.object(public_copies, 'PROJECT_DIR', Path(self.tmp.name)), \
                self.assertRaisesRegex(ValueError, 'absolute path remains'):
            public_copies.export(self.bundle, '/ckout/x/')
        self.assertFalse((self.bundle / 'public/config.yaml').exists())

    def test_committed_public_copies_are_path_free(self):
        for manifest in (PROJECT_DIR / 'published/experiments').glob('*/public/MANIFEST.json'):
            for record in json.loads(manifest.read_text())['files']:
                text = (manifest.parent / record['file']).read_text()
                self.assertIsNone(public_copies.ABSOLUTE.search(text), (manifest.parent, record['file']))


class ResultsSummaryTest(unittest.TestCase):
    def test_committed_summary_has_canonical_endpoints(self):
        summary = json.loads((PROJECT_DIR / 'published/results/summary.json').read_text())
        endpoints = {r['experiment']: r for r in summary['validation_endpoints']}
        self.assertAlmostEqual(endpoints['windowed-pbt-100']['final10_current_best'], 0.32694362048311043)
        self.assertAlmostEqual(endpoints['windowed-pbt-100']['member_final10']['lr_3e-6'], 0.327181, places=6)
        comparisons = {c['comparison']: c for c in summary['independent_holdout']['comparisons']}
        self.assertLess(comparisons['Windowed PBT vs fixed-LR population, 50 ep.']['holdout_selected_delta_pp'], 0)
        for seed in (12345, 22345):
            cadence = comparisons[f'Every-epoch vs every-5-epochs PBT, seed {seed}']
            self.assertLess(cadence['validation_final10_delta_pp'], 0)
            self.assertGreater(cadence['holdout_selected_delta_pp'], 0)

    @unittest.skipUnless((SOURCE / results.HOLDOUT / 'analysis.json').is_file(), 'completed run artifacts not available')
    def test_export_reproduces_committed_tables(self):
        with tempfile.TemporaryDirectory() as tmp, patch('builtins.print'):
            results.main(['--source-root', str(SOURCE), '--output', tmp])
            for name in ('validation_endpoints.csv', 'holdout_checkpoints.csv', 'holdout_comparisons.csv',
                         'holdout_paired_counts.csv', 'current_best_trajectories.csv'):
                self.assertEqual((Path(tmp) / name).read_text(), (PROJECT_DIR / 'published/results' / name).read_text(), name)
            fresh = json.loads((Path(tmp) / 'summary.json').read_text())
            committed = json.loads((PROJECT_DIR / 'published/results/summary.json').read_text())
            for key in ('validation_endpoints', 'independent_holdout', 'sources'):
                self.assertEqual(fresh[key], committed[key], key)


if __name__ == '__main__':
    unittest.main()
