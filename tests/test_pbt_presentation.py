from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from tests.helpers import PROJECT_DIR  # Sets up script imports.
from scripts.reports import plot_pbt_presentation as presentation


class PresentationOutputTest(unittest.TestCase):
    @staticmethod
    def figure_data(epochs=10):
        xs = list(range(1, epochs + 1))
        return {
            'epochs': xs,
            'identities': {'donor': 1, 'recipient': 2},
            'values': {
                'donor': [0.40 - i * .005 for i in range(epochs)],
                'recipient': [0.42 - i * .004 for i in range(epochs)],
            },
            'lrs': {'donor': [14e-6] * epochs, 'recipient': [3e-6] * 5 + [12e-6] * (epochs - 5)},
            'fixed': [0.41 - i * .003 for i in range(epochs)],
            'fixed_lr': 14e-6,
            'events': [{
                'after_epoch': 5, 'active_from_epoch': 6, 'donor': 'donor', 'recipient': 'recipient',
                'old_lr': 3e-6, 'new_lr': 12e-6, 'factor': 0.8,
                'pbt_generation': 1, 'copy_state_verified': True,
                'post_copy_evaluation_recorded': False,
            }],
            'winner': 'recipient',
            'final_member': 'recipient',
            'best_recorded_member': 'recipient',
            'best_recorded_epoch': epochs,
            'best_recorded_value': 0.42 - (epochs - 1) * .004,
            'protected_member': 'recipient',
            'comparisons': {'final10': {'pbt': .35, 'fixed': .36, 'reduction_percent': 2.78}},
            'initial_max_lr': 14e-6,
            'window_epochs': 5,
            'pbt_generations': epochs // 5,
            'generation_ranges': [
                {'generation': i + 1, 'start_epoch': i * 5 + 1, 'end_epoch': (i + 1) * 5}
                for i in range(epochs // 5)
            ],
        }

    def test_performance_plot_branches_recipient_from_donor_checkpoint(self):
        plt = presentation.plot_setup()
        data = self.figure_data()
        fig = presentation.progression(plt, data)
        self.addCleanup(plt.close, fig)
        performance = fig.axes[0]
        self.assertEqual(performance.get_xlabel(), 'Training epoch')
        self.assertEqual(performance.get_xlim(), (0, 10))
        self.assertEqual(list(performance.get_xticks()), [0, 10])
        recipient_color = presentation.colors(self.figure_data())['recipient']
        solid_segments = [list(line.get_xdata()) for line in performance.lines
                          if line.get_color() == recipient_color and line.get_linestyle() == '-']
        self.assertIn([1, 2, 3, 4, 5], solid_segments)
        self.assertIn([6, 7, 8, 9, 10], solid_segments)
        self.assertIn([5, 6], solid_segments)
        branch = next(line for line in performance.lines
                      if line.get_color() == recipient_color and list(line.get_xdata()) == [5, 6])
        self.assertEqual(list(branch.get_ydata()), [data['values']['donor'][4],
                                                   data['values']['recipient'][5]])
        self.assertNotEqual(data['values']['recipient'][4], branch.get_ydata()[0])
        self.assertEqual([text.get_text() for text in performance.texts if '→' in text.get_text()], [])
        self.assertEqual(len(performance.collections), 2)  # copied checkpoint + final-best star
        self.assertEqual(list(performance.collections[0].get_offsets()[0]),
                         [5, data['values']['donor'][4]])
        self.assertTrue(any('Global best: Member 2' in text.get_text() for text in fig.texts))
        self.assertTrue(any('final-10 comparison window' in text.get_text() for text in fig.texts))
        self.assertTrue(any('M2 late-window mean' in text.get_text() for text in performance.texts))
        legend_labels = [text.get_text() for text in fig.legends[0].get_texts()]
        self.assertEqual(legend_labels,
                         ['Member 1', 'Member 2', 'Fixed 14e-6 baseline', 'Copied checkpoint'])

    def test_linked_figures_share_epoch_scale_colors_and_event_positions(self):
        from matplotlib.colors import to_rgba

        plt = presentation.plot_setup()
        data = self.figure_data()
        performance = presentation.progression(plt, data)
        rates = presentation.learning_rates(plt, data)
        self.addCleanup(plt.close, performance)
        self.addCleanup(plt.close, rates)
        performance_ax, rates_ax = performance.axes[0], rates.axes[0]
        self.assertEqual(performance_ax.get_xlabel(), rates_ax.get_xlabel())
        self.assertEqual(performance_ax.get_xlim(), rates_ax.get_xlim())
        self.assertEqual(list(performance_ax.get_xticks()), list(rates_ax.get_xticks()))
        self.assertEqual([text.get_text() for text in rates_ax.texts if '→' in text.get_text()],
                         ['M1→M2  ×0.8'])
        self.assertEqual([text.get_text() for text in rates.legends[0].get_texts()],
                         ['Member 1', 'Member 2', 'Copied checkpoint', 'Mutated LR'])
        recipient_color = presentation.colors(data)['recipient']
        lr_segments = [line for line in rates_ax.lines if line.get_color() == recipient_color]
        self.assertTrue(any(list(line.get_xdata()) == [0, 5] and
                            list(line.get_ydata()) == [3, 3] for line in lr_segments))
        self.assertTrue(any(list(line.get_xdata()) == [5, 10] and
                            list(line.get_ydata()) == [12, 12] for line in lr_segments))
        self.assertTrue(any(list(line.get_xdata()) == [5, 5] and
                            list(line.get_ydata()) == [14, 12] for line in lr_segments))
        performance_colors = {line.get_label(): to_rgba(line.get_color())
                              for line in performance_ax.lines if line.get_label().startswith('Member')}
        rate_colors = {line.get_label(): to_rgba(line.get_color())
                       for line in rates_ax.lines if line.get_label().startswith('Member')}
        self.assertEqual(performance_colors, rate_colors)

    def test_pngs_belong_to_candidate_run_and_existing_plots_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / 'baseline'
            baseline.mkdir()
            for name in ('run_a', 'continuation_b'):
                run = root / name
                plots = run / 'plots'
                plots.mkdir(parents=True)
                diagnostic = plots / 'diagnostic.png'
                diagnostic.write_bytes(b'unchanged')
                figure = MagicMock()
                figure.savefig.side_effect = lambda path, **kwargs: path.write_bytes(b'png')
                with patch.object(presentation, 'presentation_data', return_value={'verified': True}), \
                     patch.object(presentation, 'plot_setup'), \
                     patch.object(presentation, 'progression', return_value=figure), \
                     patch.object(presentation, 'learning_rates', return_value=figure):
                    output = presentation.export_figures(run / 'manifest.json', baseline, 'control')
                    self.assertEqual(output, plots)
                    self.assertEqual(len(list(plots.glob('*.png'))), 3)
                    self.assertEqual(list(plots.glob('*.pdf')), [])
                    self.assertTrue((plots / 'presentation_source_values.json').is_file())
                    self.assertTrue(all(c.kwargs['dpi'] == 300 for c in figure.savefig.call_args_list))
                    with self.assertRaises(FileExistsError):
                        presentation.export_figures(run, baseline, 'control')
                self.assertEqual(diagnostic.read_bytes(), b'unchanged')
            self.assertEqual(list(baseline.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
