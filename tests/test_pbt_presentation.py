from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from tests.helpers import PROJECT_DIR  # Sets up script imports.
from scripts.reports import plot_pbt_presentation as presentation


class PresentationOutputTest(unittest.TestCase):
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
                     patch.object(presentation, 'learning_rates', return_value=figure), \
                     patch.object(presentation, 'final_comparison', return_value=figure):
                    output = presentation.export_figures(run / 'manifest.json', baseline, 'control')
                    self.assertEqual(output, plots)
                    self.assertEqual(len(list(plots.glob('*.png'))), 4)
                    self.assertEqual(list(plots.glob('*.pdf')), [])
                    self.assertTrue((plots / 'presentation_source_values.json').is_file())
                    self.assertTrue(all(c.kwargs['dpi'] == 300 for c in figure.savefig.call_args_list))
                    with self.assertRaises(FileExistsError):
                        presentation.export_figures(run, baseline, 'control')
                self.assertEqual(diagnostic.read_bytes(), b'unchanged')
            self.assertEqual(list(baseline.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
