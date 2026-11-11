import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from experiments import weight_averaging as wa  # noqa: E402
from experiments.stage2a.common import PRIMARY, sha256  # noqa: E402


def manifest(copies=(), best=('a', 21), metrics=None):
    """Epochs 18-23, members a and b; copies are (epoch, donor, recipient)."""
    generations = []
    for epoch in range(18, 24):
        exploit = [dict(applied=True, donor=d, recipient=r) for t, d, r in copies if t == epoch]
        values = (metrics or {}).get(epoch, dict(a=0.30, b=0.31))
        generations.append(dict(epoch=epoch, exploit=exploit,
                                workers={n: dict(metrics={PRIMARY: v}) for n, v in values.items()}))
    return dict(generations=generations, best=dict(member=best[0], epoch=best[1]))


class LineageTest(unittest.TestCase):
    def test_ancestry_switches_to_the_donor_before_a_copy(self):
        m = manifest(copies=[(20, 'b', 'a')])
        paths = wa.lineage(m, 'a', 22, Path('/r'))
        self.assertEqual({t: p.parent.name for t, p in paths.items()},
                         {22: 'a', 21: 'a', 20: 'a', 19: 'b', 18: 'b'})

    def test_window_stops_after_the_selected_epoch_at_a_later_copy(self):
        m = manifest(copies=[(23, 'b', 'a')], best=('a', 21))
        paths = wa.window_paths(m, Path('/r'), dict(before=1, after=2))
        self.assertEqual([(p.parent.name, p.name) for p in paths],
                         [('a', 'net_epoch-20_state.pt'), ('a', 'net_epoch-21_state.pt'), ('a', 'net_epoch-22_state.pt')])

    def test_window_is_cut_at_the_first_epoch(self):
        paths = wa.window_paths(manifest(best=('a', 19)), Path('/r'), dict(before=4, after=0))
        self.assertEqual([p.name for p in paths], ['net_epoch-18_state.pt', 'net_epoch-19_state.pt'])

    def test_soup_order_drops_recipients_and_ranks_by_metric(self):
        m = manifest(copies=[(21, 'a', 'c')], metrics={21: dict(a=0.30, b=0.32, c=0.29)})
        self.assertEqual([p.parent.name for p in wa.soup_order(m, Path('/r'))], ['a', 'b'])


class AverageTest(unittest.TestCase):
    def test_float_tensors_are_averaged_and_counters_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, w in enumerate((1.0, 2.0, 6.0)):
                path = Path(tmp) / f'{i}.pt'
                torch.save(dict(w=torch.full((2,), w), n=torch.tensor(5)), path)
                paths.append(path)
            out = wa.average_states(paths)
            self.assertTrue(torch.equal(out['w'], torch.full((2,), 3.0)))
            self.assertEqual(int(out['n']), 5)
            torch.save(dict(w=torch.ones(2), n=torch.tensor(6)), paths[2])
            with self.assertRaises(ValueError):
                wa.average_states(paths)


class ScoreRunTest(unittest.TestCase):
    def test_greedy_soup_keeps_only_improving_members_and_the_best_candidate_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / 'runs/pbt/r'
            for member, value in (('a', 1.0), ('b', 2.0), ('c', 3.0)):
                for epoch in range(18, 24):
                    path = run_dir / member / f'net_epoch-{epoch}_state.pt'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(dict(w=torch.tensor([value + epoch])), path)
            m = manifest(best=('a', 21), metrics={21: dict(a=0.30, b=0.31, c=0.32)})
            m['best']['source_checkpoint'] = dict(state=dict(sha256=sha256(run_dir / 'a/net_epoch-21_state.pt')))
            m['best']['metric_value'] = 0.30

            def evaluate(checkpoint, directory):
                sources = [s['path'] for s in wa.load_json(directory / 'sources.json')['sources']]
                members = sorted({Path(s).parent.name for s in sources})
                return {('a',): 0.300, ('a', 'b'): 0.298, ('a', 'b', 'c'): 0.299}.get(tuple(members), 0.301) \
                    - 0.0001 * (len(sources) == 5 and members == ['a'])

            recipes = [dict(name='plain', kind='plain'), dict(name='trail5', kind='lineage_window', before=4, after=0),
                       dict(name='soup', kind='greedy_population_soup')]
            with mock.patch.object(wa, 'PROJECT_DIR', root), mock.patch.object(wa, 'run_manifest', return_value=m):
                result = wa.score_run(dict(id='x', run='r'), recipes, root / 'out', evaluate)
            soup = next(c for c in result['candidates'] if c['recipe'] == 'soup')
            self.assertEqual(soup['members'], ['a', 'b'])
            self.assertEqual([t['accepted'] for t in soup['trials']], [True, False])
            self.assertEqual(result['selection'], 'soup')
            self.assertAlmostEqual(result['selection_metric'], 0.298)
            self.assertEqual(sorted(c['recipe'] for c in result['candidates']), ['plain', 'soup', 'trail5'])


if __name__ == '__main__':
    unittest.main()
