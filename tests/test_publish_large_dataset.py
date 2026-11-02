"""Large-dataset results export: only a COMPLETE, integrity-checked report is published, without private data."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_stage3_campaign import NAMES, PRIMARY, WORKING_POINTS, arm_fixture
from experiments.stage3 import analysis, common, testset
from scripts.publish import export_large_dataset_results as exporter

PRIVATE = ('/data/private', 'node-x', 'GPU-1234')


def pair(left, right, delta):
    counts = dict(n_paired_background_rows=100, both_passed=1, left_only=2, right_only=1, neither_passed=96, delta_pp=-0.01)
    return dict(left=left, right=right, delta_pp=delta, reduction_percent=-100 * delta / 0.3, n_paired_rows=1000,
                working_points={k: dict(counts) for k in WORKING_POINTS}, confidence=dict(status='unavailable'))


def write_report(root, **changes):
    """A final report with the campaign's real summary structures (analysis.arm_summary / comparison)."""
    runs = {arm: arm_fixture(root, arm, copy_at=9) for arm in common.ARMS}
    summaries = {arm: analysis.arm_summary(common.load_json(Path(run) / 'manifest.json'), 12) for arm, run in runs.items()}
    checkpoints = []
    for arm in common.ARMS:
        checkpoints.append(dict(id=f'{arm}__selected_best', arm=arm, role='validation_global_best',
                                checkpoint=dict(path=f'{PRIVATE[0]}/{arm}.pt')))
        checkpoints += [dict(id=f'{arm}__final__{name}', arm=arm, role='final_member', member=name,
                             final_current_best=name == NAMES[0], checkpoint=dict(path=f'{PRIVATE[0]}/{name}.pt'))
                        for name in NAMES]
    metrics = {c['id']: {k: 0.3 - 0.001 * (c['arm'] == 'windowed_pbt_v2') for k in (PRIMARY, *WORKING_POINTS)}
               for c in checkpoints}
    comparisons = {name: pair(*ids, -0.001) for name, ids in testset.preregistered_pairs(checkpoints).items()}
    selected = [f'{arm}__selected_best' for arm in common.ARMS]
    data = dict(schema_version=1, status='COMPLETE', interim=False, determinism_confirmed=True, completed_generations=12,
                summaries=summaries, comparison=analysis.comparison(summaries, 0.002),
                sealed_test=dict(metrics=metrics, comparisons=comparisons, checkpoints=checkpoints,
                                 population_mean=dict(fixed_lr=0.3, windowed_pbt_v2=0.299, delta_pp=-0.001, reduction_percent=0.33)),
                holdout=dict(metrics={k: metrics[k] for k in selected}, comparisons={'selected_best': pair(*selected, -0.002)}),
                operations=dict(gpu_hours_consumed=12.5, allocations=[dict(host=PRIVATE[1], gpus=[dict(uuid=PRIVATE[2])])]))
    data.update(changes)
    report = Path(root) / exporter.REPORT
    report.mkdir(parents=True)
    (report / 'analysis.json').write_text(json.dumps(data))
    (report / 'report_integrity.json').write_text(json.dumps(dict(artifacts=[
        dict(path=str(report / 'analysis.json'), sha256=exporter.sha256(report / 'analysis.json'))])))
    return report


class LargeDatasetExportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def export(self):
        with patch('builtins.print'):
            exporter.main(['--source-root', str(self.root), '--output', str(self.root / 'published')])
        return self.root / 'published'

    def test_complete_report_is_published_without_private_data(self):
        write_report(self.root)
        output = self.export()
        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual(summary['status'], 'COMPLETE')
        self.assertEqual(summary['validation_comparison']['primary_final10_current_best']['delta_pp'], 0.0)
        self.assertEqual(summary['arms']['windowed_pbt_v2']['copies'], 1)
        self.assertEqual(len(summary['sealed_test']['checkpoints']), 12)
        self.assertEqual(len(summary['sealed_test']['comparisons']), 2 + len(NAMES))
        with open(output / 'sealed_test_paired_counts.csv') as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), (2 + len(NAMES)) * len(WORKING_POINTS))
        self.assertEqual(sorted(p.name for p in (output / 'figures').iterdir()),
                         ['current_best_trajectories.png', 'sealed_test_differences.png'])
        for path in output.rglob('*'):
            if path.suffix in ('.json', '.csv'):
                text = path.read_text()
                self.assertFalse([p for p in PRIVATE if p in text], path.name)

    def test_incomplete_interim_or_modified_reports_are_refused(self):
        for changes, message in ((dict(status='INVALID'), 'INVALID'), (dict(status='INCOMPLETE'), 'INCOMPLETE'),
                                 (dict(interim=True), 'COMPLETE')):
            with tempfile.TemporaryDirectory() as other:
                self.root = Path(other)
                write_report(self.root, **changes)
                with self.assertRaisesRegex(ValueError, message):
                    self.export()
        with tempfile.TemporaryDirectory() as other:
            self.root = Path(other)
            report = write_report(self.root)
            data = json.loads((report / 'analysis.json').read_text())
            data['comparison']['primary_final10_current_best']['delta_pp'] = -1.0
            (report / 'analysis.json').write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'integrity hash'):
                self.export()


if __name__ == '__main__':
    unittest.main()
