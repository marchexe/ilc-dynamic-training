"""Stage 3 determinism checks compare restart-bundle contents, not torch.save file bytes."""
import copy
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile

import torch

from tests.test_stage3_campaign import NAMES, arm_fixture, load_protocol
from tests.helpers import PROJECT_DIR
from experiments.stage3 import analysis, common, protocol as frozen
from experiments.stage3.checkpoint_contents import bundle_differences, differences, restart_bundle

GATE_MEMBER = 'lr_14e-6'


def as_saved_by_another_worker(path):
    """Rewrite an archive the way a worker on another slot saves identical tensors:
    another CUDA device tag in data.pkl and hence another `.data/serialization_id`."""
    with zipfile.ZipFile(path) as archive:
        items = [(info, archive.read(info.filename)) for info in archive.infolist()]
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_STORED) as archive:
        for info, data in items:
            if info.filename.endswith('.data/serialization_id'):
                data = b'%040d' % (int(data) + 1)
            elif info.filename.endswith('data.pkl') and b'X\x03\x00\x00\x00cpu' in data:
                data = data.replace(b'X\x03\x00\x00\x00cpu', b'X\x06\x00\x00\x00cuda:4')
            archive.writestr(info, data)


def realistic_bundle(directory, epoch, *, controller=False):
    """A small bundle with the structure Weaver writes (model, RAdam/Ranger optimizer, scaler)."""
    torch.manual_seed(epoch)
    directory.mkdir(parents=True, exist_ok=True)
    model = {'embed.weight': torch.randn(4, 3), 'embed.bias': torch.randn(4), 'bn.num_batches_tracked': torch.tensor(7)}
    optimizer = dict(state={i: dict(step=1000 + epoch, exp_avg=torch.randn(4), exp_avg_sq=torch.rand(4)) for i in range(2)},
                     param_groups=[dict(lr=1.4e-05, betas=(0.95, 0.999), eps=1e-05, weight_decay=0,
                                        buffer=[[1000, 1998.9999999999982, 1.5e-3]] * 2, initial_lr=0.001, params=[0, 1])])
    paths = restart_bundle(directory, epoch)
    torch.save(model, paths['state'])
    torch.save(optimizer, paths['optimizer'])
    torch.save({'scale': 65536.0, 'growth_factor': 2.0, 'backoff_factor': 0.5, 'growth_interval': 2000,
                '_growth_tracker': 345}, paths['scaler'])
    if controller:
        torch.save(dict(phase='a'), paths['controller'])
    return paths


def rewrite(path, change):
    value = torch.load(path, map_location='cpu', weights_only=False)
    change(value)
    torch.save(value, path)


class ContentComparisonTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.left = realistic_bundle(self.root / 'left', 18)
        self.right = realistic_bundle(self.root / 'right', 18)

    def tearDown(self):
        self.tmp.cleanup()

    def test_identical_state_saved_by_different_workers_is_identical(self):
        for path in self.right.values():
            if path.is_file():
                as_saved_by_another_worker(path)
        self.assertNotEqual(common.sha256(self.left['state']), common.sha256(self.right['state']))
        with zipfile.ZipFile(self.right['state']) as archive:
            self.assertIn(b'cuda:4', archive.read(next(n for n in archive.namelist() if n.endswith('data.pkl'))))
        self.assertEqual(bundle_differences(self.left, self.right), [])

    def test_every_part_of_the_restart_bundle_is_compared(self):
        def bump(tensor):
            tensor.view(torch.int32)[0] += 1  # one unit in the last place

        cases = {
            'state/embed.weight: tensor values differ': ('state', lambda v: bump(v['embed.weight'])),
            'state/bn.num_batches_tracked: tensor values differ': ('state', lambda v: v['bn.num_batches_tracked'].add_(1)),
            'state/embed.bias: torch.float32(4,) != torch.float64(4,)':
                ('state', lambda v: v.update({'embed.bias': v['embed.bias'].double()})),
            "state: keys differ ['extra']": ('state', lambda v: v.update(extra=torch.zeros(1))),
            'optimizer/state/1/exp_avg_sq: tensor values differ': ('optimizer', lambda v: bump(v['state'][1]['exp_avg_sq'])),
            'optimizer/state/0/step: 1018 != 1019': ('optimizer', lambda v: v['state'][0].update(step=1019)),
            'optimizer/param_groups[0]/lr: 1.4e-05 != 1.12e-05':
                ('optimizer', lambda v: v['param_groups'][0].update(lr=1.12e-05)),
            'optimizer/param_groups[0]/buffer[1][1]: 1998.9999999999982 != 1999.0':
                ('optimizer', lambda v: v['param_groups'][0].update(
                    buffer=[[1000, 1998.9999999999982, 1.5e-3], [1000, 1999.0, 1.5e-3]])),
            'scaler/scale: 65536.0 != 32768.0': ('scaler', lambda v: v.update(scale=32768.0)),
            'scaler/_growth_tracker: 345 != 346': ('scaler', lambda v: v.update(_growth_tracker=346)),
        }
        for expected, (part, change) in cases.items():
            with self.subTest(expected):
                other = realistic_bundle(self.root / 'changed', 18)
                rewrite(other[part], change)
                self.assertEqual(bundle_differences(self.left, other), [expected])

    def test_missing_parts_and_one_sided_controller_are_differences(self):
        self.right['scaler'].unlink()
        self.assertEqual(bundle_differences(self.left, self.right), ['scaler: missing on the right'])
        self.assertEqual(bundle_differences(self.right, self.right), ['scaler: missing on the left and right'])
        with_controller = realistic_bundle(self.root / 'controller', 18, controller=True)
        self.assertEqual(bundle_differences(self.left, with_controller), ['controller: missing on the left'])
        self.assertEqual(bundle_differences(with_controller, with_controller), [])

    def test_python_values_are_compared_exactly(self):
        self.assertEqual(differences(float('nan'), float('nan')), [])
        self.assertTrue(differences(0.0, -0.0))
        self.assertTrue(differences(1, 1.0))
        self.assertTrue(differences((1, 2), [1, 2]))
        self.assertEqual(differences(torch.tensor([float('nan')]), torch.tensor([float('nan')])), [])


class AnalysisDeterminismTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.protocol = load_protocol()
        self.protocol['design']['horizon_epochs'] = 12
        self.runs = {arm: arm_fixture(self.root, arm, copy_at=9) for arm in common.ARMS}
        self.manifests = {arm: common.load_json(Path(run) / 'manifest.json') for arm, run in self.runs.items()}
        gate_run = self.root / 'gate' / 'attempt'
        epoch = self.protocol['design']['initial_epoch'] + 1
        reference = restart_bundle(gate_run / GATE_MEMBER, epoch)
        for part in ('state', 'optimizer', 'scaler'):
            reference[part].parent.mkdir(parents=True, exist_ok=True)
            reference[part].write_bytes(restart_bundle(Path(self.runs['fixed_lr']) / GATE_MEMBER, epoch)[part].read_bytes())
            as_saved_by_another_worker(reference[part])
        self.gate = dict(run=str(gate_run), measurements=dict(reference_state_sha256=common.sha256(reference['state'])))

    def tearDown(self):
        self.tmp.cleanup()

    def checks(self):
        checks, _, identity = analysis.validity(self.protocol, self.runs, self.manifests, gate=self.gate,
                                                verifier=type('V', (), dict(verify=staticmethod(
                                                    lambda run, through=None: dict(passed=True, failures=[])))))
        return {c['check']: c for c in checks if c['kind'] == 'determinism'}, identity

    def resave_windowed(self):
        for path in Path(self.runs['windowed_pbt_v2']).glob('lr_*/net_epoch-*.pt'):
            as_saved_by_another_worker(path)

    def test_independently_saved_identical_states_confirm_determinism(self):
        self.resave_windowed()
        fixed = restart_bundle(Path(self.runs['fixed_lr']) / NAMES[0], 18)['state']
        windowed = restart_bundle(Path(self.runs['windowed_pbt_v2']) / NAMES[0], 18)['state']
        self.assertNotEqual(common.sha256(fixed), common.sha256(windowed))  # the file-hash check failed here
        checks, identity = self.checks()
        self.assertEqual([c['passed'] for c in checks.values()], [True, True], checks)
        self.assertEqual(identity['checked_states'], 4 * 12 + 9 + 1)  # lr_3e-6: nine generations before its copy + pre-copy
        self.assertEqual(identity['first_copy'], {'lr_3e-6': 10})

    def test_optimizer_or_scaler_divergence_is_reported(self):
        self.resave_windowed()
        windowed = Path(self.runs['windowed_pbt_v2'])
        rewrite(restart_bundle(windowed / 'lr_5_75e-6', 21)['optimizer'], lambda v: v['state'][0].update(step=v['state'][0]['step'] + 1))
        rewrite(restart_bundle(windowed / GATE_MEMBER, 18)['scaler'], lambda v: v.update(scale=512.0))
        checks, identity = self.checks()
        self.assertFalse(checks['gate epoch-18 state reproduced by both arms']['passed'])
        self.assertEqual(checks['gate epoch-18 state reproduced by both arms']['detail']['differences'],
                         dict(fixed_lr=[], windowed_pbt_v2=['scaler/scale: 512.0 != 1024.0']))
        self.assertEqual([(m['member'], m['epoch']) for m in identity['mismatches']], [('lr_5_75e-6', 21), ('lr_14e-6', 18)])

    def test_gate_reference_and_pre_copy_archive_are_anchored_to_their_recorded_hashes(self):
        reference = restart_bundle(Path(self.gate['run']) / GATE_MEMBER, 18)['state']
        rewrite(reference, lambda v: None)  # same tensors, new bytes: no longer the recorded reference file
        event = self.manifests['windowed_pbt_v2']['generations'][9]['exploit'][0]
        as_saved_by_another_worker(Path(event['pre_copy_archive']['optimizer']['path']))
        checks, identity = self.checks()
        gate = checks['gate epoch-18 state reproduced by both arms']
        self.assertEqual((gate['passed'], gate['detail']['reference_intact']), (False, False))
        self.assertEqual(identity['mismatches'], [dict(member='lr_3e-6', epoch=27, pre_copy=True, differences=[
            'pre-copy archive optimizer differs from the recorded pre-copy state'])])
        broken = copy.deepcopy(event)
        del broken['pre_copy_archive']
        self.assertEqual(analysis.pre_copy_bundle(broken), (None, 'pre-copy archive missing or incomplete'))


class AnalysisRevisionRecordTest(unittest.TestCase):
    def test_records_are_pinned_and_chain_to_the_current_files(self):
        protocol_sha = (common.PREPARED / 'protocol_v2.sha256').read_text().strip()
        latest = {}
        for number in (1, 2):
            path = common.PREPARED / f'analysis_revision_{number}.json'
            record = common.load_json(path)
            self.assertEqual(common.sha256(path), path.with_suffix('.sha256').read_text().strip())
            self.assertEqual(record['revision'], number)
            self.assertEqual((record['protocol']['sha256'], record['protocol']['changed']), (protocol_sha, False))
            for item in record['correction']['files']:
                self.assertNotIn(item['file'], frozen.EXECUTION_FILES)
                original = subprocess.run(['git', 'show', f'{record["correction"]["original_commit"]}:{item["file"]}'],
                                          cwd=PROJECT_DIR, capture_output=True)
                self.assertEqual(hashlib.sha256(original.stdout).hexdigest() if original.returncode == 0 else None,
                                 item['original_sha256'])
                if item['file'] in latest:  # each revision starts from the previous revision's file
                    self.assertEqual(item['original_sha256'], latest[item['file']])
                latest[item['file']] = item['corrected_sha256']
        self.assertEqual(common.load_json(common.PREPARED / 'analysis_revision_1.json')['protocol']['determinism_criteria'],
                         common.load_json(common.PREPARED / 'protocol_v2.json')['validity']['determinism'])
        for file, digest in latest.items():
            self.assertEqual(common.sha256(PROJECT_DIR / file), digest, file)

if __name__ == '__main__':
    unittest.main()
