"""Follow-up train800k experiments: configurations, placement, holdout addendum and reproduction checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from tests.helpers import PROJECT_DIR
from experiments import holdout_addendum, reproduction_checks, train800k
from launch.experiment import configuration
from training.pbt.execution.weaver_command import make_command
from training.members import MemberState

EXPERIMENTS = PROJECT_DIR / 'configs/experiments'
CONTROL = PROJECT_DIR / 'runs/pbt/foundation_fixed_lr_50epochs_20260916'
PRIMARY = 'validation_total_reference_mistag_geomean_percent'


def load(name, gpus=None):
    return configuration(EXPERIMENTS / f'{name}.yaml', gpus)


class ConfigurationTest(unittest.TestCase):
    def test_high_grid_changes_only_the_learning_rates(self):
        high, control = load('fixed_lr_high_grid_50epochs'), load('foundation_fixed_lr_50epochs')
        self.assertEqual(high['shared'], control['shared'])
        self.assertEqual([m['start_lr'] for m in high['population']], [1.4e-5, 1.8e-5, 2.2e-5, 2.6e-5, 3.0e-5])
        self.assertEqual((high['pbt']['strategy'], high['pbt']['min_lr'], high['pbt']['max_lr']), ('fixed_lr_grid', 1.4e-5, 3.0e-5))
        drop = lambda pbt: {k: v for k, v in pbt.items() if k not in ('min_lr', 'max_lr')}  # noqa: E731
        self.assertEqual(drop(high['pbt']), drop(control['pbt']))

    @unittest.skipUnless((CONTROL / 'manifest.json').is_file(), 'original control run not present')
    def test_anchor_member_repeats_the_recorded_weaver_commands(self):
        high = load('fixed_lr_high_grid_50epochs')
        recorded = json.loads((CONTROL / 'manifest.json').read_text())
        run = PROJECT_DIR / 'runs/pbt/fixed_lr_high_grid_50epochs'
        member = MemberState.from_legacy(dict(name='lr_14e-6', lr=1.4e-5, parent=None))
        for generation in (0, 1, 49):
            worker = recorded['generations'][generation]['workers']['lr_14e-6']
            old = worker['command']
            new = make_command(high, member, old[old.index('--gpus') + 1], run.resolve() / 'lr_14e-6', generation)
            new = new[0] if isinstance(new, tuple) else new
            plain = lambda command, root: [str(x).replace(str(root.resolve()), 'RUN') for x in command]  # noqa: E731
            self.assertEqual(plain(new, run), plain(old, CONTROL))

    def test_equivalence_runs_differ_only_in_slicing(self):
        one, fifths = load('full_state_gpu_one_epoch', '0'), load('full_state_gpu_fifths', '0')
        self.assertEqual([m['name'] for m in one['population']], ['lr_14e-6'])
        self.assertEqual(one['shared']['optimizer_state_contract'], 'full_lookahead_v1')
        self.assertEqual((one['shared']['generations'], fifths['shared']['generations']), (1, 5))
        sliced = {k: fifths['shared'][k] for k in ('generations_per_epoch', 'batches_per_generation')}
        self.assertEqual(sliced, dict(generations_per_epoch=5, batches_per_generation=420))
        rest = {k: v for k, v in fifths['shared'].items() if k not in ('generations', *sliced)}
        self.assertEqual(rest, {k: v for k, v in one['shared'].items() if k != 'generations'})
        self.assertGreater(2102 - 4 * 420, 0)  # the last slice of the 2,102-batch epoch is not empty


def gpu(host, index, name='NVIDIA A100 80GB PCIe', used=0, pids=()):
    return dict(host=host, index=index, name=name, total_mib=81920, used_mib=used, utilization=0,
                uuid=f'{host}-{index}', pids=list(pids))


def node(host, gpus):
    return dict(host=host, gpus=gpus, error=None)


class PlacementTest(unittest.TestCase):
    cluster = dict(authorized_nodes=['n1', 'n2'], preferred_node='n2')

    def setUp(self):
        patcher = mock.patch.object(train800k.site_config, 'cluster', return_value=self.cluster)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.config = load('fixed_lr_high_grid_50epochs')

    def test_needs_one_idle_gpu_per_member_on_one_node(self):
        nodes = [node('n1', [gpu('n1', i) for i in range(5)]),
                 node('n2', [gpu('n2', i) for i in range(4)] + [gpu('n2', 4, pids=[7])])]
        cohort = train800k.placement(self.config, nodes)
        self.assertEqual((cohort['host'], [g['index'] for g in cohort['gpus']]), ('n1', [0, 1, 2, 3, 4]))
        self.assertIsNone(train800k.placement(self.config, nodes[1:]))  # four idle GPUs: no partial set
        self.assertIsNone(train800k.placement(self.config, nodes, host='n2'))
        mixed = [node('n1', [gpu('n1', i) for i in range(3)] + [gpu('n1', i, name='other') for i in (3, 4)])]
        self.assertIsNone(train800k.placement(self.config, mixed))

    def test_prefers_the_preferred_node(self):
        nodes = [node(h, [gpu(h, i) for i in range(6)]) for h in ('n1', 'n2')]
        self.assertEqual(train800k.placement(self.config, nodes)['host'], 'n2')

    def launch(self, cgroup, *, dry_run=False):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp) / 'logs'

            def run(host, command, check=True):
                calls.append((host, command))
                if ' start ' in command:
                    logs.mkdir()
                    (logs / 'launcher.json').write_text(json.dumps(dict(pid=4242)))
                if command.startswith('cat /proc/'):
                    return mock.Mock(returncode=0, stdout=cgroup, stderr='')
                if command.startswith('cat '):
                    return mock.Mock(returncode=0, stdout=(logs / 'launcher.json').read_text(), stderr='')
                return mock.Mock(returncode=0, stdout='ok', stderr='')
            with mock.patch.object(train800k, 'locations', return_value=(Path(tmp) / 'run', logs)):
                nodes = [node('n2', [gpu('n2', i) for i in range(5)])]
                try:
                    return train800k.launch('start', self.config, 'c.yaml', dry_run=dry_run,
                                            discover_nodes=lambda _: nodes, run=run), calls, None
                except RuntimeError as error:
                    return None, calls, error
                finally:
                    self.history = train800k.placements(self.config) if logs.exists() else None

    def test_dry_run_starts_nothing(self):
        plan, calls, _ = self.launch('0::/user.slice/user-1.slice/session-2.scope', dry_run=True)
        self.assertEqual((calls, plan['cohort']['host']), ([], 'n2'))

    def test_start_records_placement_outside_oomd(self):
        record, calls, _ = self.launch('0::/user.slice/user-1.slice/session-2.scope')
        self.assertIn('experiment.py start', calls[0][1])
        self.assertIn('--gpus 0,1,2,3,4', calls[0][1])
        self.assertIn('env -u DBUS_SESSION_BUS_ADDRESS', calls[0][1])
        self.assertEqual((record['host'], record['runner_pid'], len(self.history)), ('n2', 4242, 1))

    def test_runner_under_oomd_is_stopped(self):
        _, calls, error = self.launch('0::/user.slice/user-1.slice/user@1.service/app.slice')
        self.assertIn('oomd-managed', str(error))
        self.assertIn('experiment.py stop', calls[-1][1])


class HoldoutAddendumTest(unittest.TestCase):
    def test_spec_validation(self):
        spec = json.loads((PROJECT_DIR / 'configs/prepared/holdout_high_lr_grid_v1/spec.json').read_text())
        self.assertIs(holdout_addendum.validate_spec(spec), spec)
        for bad in (dict(spec, pairs=[['fixed50_s12345', 'unknown']]), dict(spec, pairs=[['a', 'a']]),
                    dict(spec, references=['fixed_high50_s12345']), dict(spec, runs=[])):
            with self.assertRaises(ValueError):
                holdout_addendum.validate_spec(bad)

    def test_parallelism_is_bounded_by_idle_gpus_and_checkpoints(self):
        protocol = dict(allocation=dict(authorized_nodes=['n1'], preferred_node='n1', max_used_memory_mib=500,
                                        max_util_percent=5, min_gpu_memory_mib=40000),
                        checkpoints=[{}] * 3)
        two_idle = [node('n1', [gpu('n1', 0), gpu('n1', 1), gpu('n1', 2, pids=[9])])]
        self.assertEqual(holdout_addendum.plan(protocol, two_idle)['max_parallelism'], 2)
        many = [node('n1', [gpu('n1', i) for i in range(8)])]
        self.assertEqual(holdout_addendum.plan(protocol, many)['max_parallelism'], 3)

    @unittest.skipUnless((CONTROL / 'manifest.json').is_file(), 'original control run not present')
    def test_selection_rule_reproduces_the_stage2a_choice(self):
        stage2a = json.loads((holdout_addendum.STAGE2A / 'protocol.json').read_text())
        frozen = next(c for c in stage2a['checkpoints'] if c['id'] == 'fixed50_s12345')
        mine = holdout_addendum.selected_checkpoint('x', CONTROL.name, stage2a['evaluation'])
        self.assertEqual(mine['checkpoint'], frozen['checkpoint'])
        self.assertEqual((mine['member'], mine['weaver_epoch']), (frozen['member'], frozen['weaver_epoch']))

    def test_report_states_reference_reproduction(self):
        protocol = dict(campaign='c', purpose='p', pairs=[['ref', 'new']], analysis={},
                        checkpoints=[dict(id='new', selection='validation_primary_global_best'),
                                     dict(id='ref', selection='stage2a_reference', stage2a_holdout_metric=0.32)])
        from training.runtime import WORKING_POINT_DEFINITION, mistag_percent_key
        points = {mistag_percent_key(*p): dict(delta_pp=-0.01, left_only=5, right_only=3) for p in WORKING_POINT_DEFINITION}
        comparison = dict(delta_pp=-0.003, reduction_percent=1.0, working_points=points)
        for measured, identical in ((0.32, True), (0.3200001, False)):
            with tempfile.TemporaryDirectory() as tmp, mock.patch('experiments.stage3.testset.paired', return_value=(
                    dict(new={PRIMARY: 0.317}, ref={PRIMARY: measured}), dict(new_minus_ref=comparison))):
                result = holdout_addendum.report(protocol, 'sha', {}, Path(tmp))
                self.assertIs(result['references_reproduced'], identical)
                text = (Path(tmp) / 'report.md').read_text()
                self.assertIn('bc80 5/3', text)
                self.assertIn('identical' if identical else 'DIFFERENT', text)
                integrity = json.loads((Path(tmp) / 'report_integrity.json').read_text())
                self.assertEqual((integrity['protocol_sha256'], len(integrity['artifacts'])), ('sha', 2))


class HoldoutRelaunchTest(unittest.TestCase):
    """The first holdout_high_lr_grid_v1 launch stopped after 26 minutes: Ctrl-C in an attached tmux pane."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.output = Path(tmp.name)
        self.protocol = dict(allocation=dict(authorized_nodes=['n1'], preferred_node='n1', max_used_memory_mib=500,
                                             max_util_percent=5, min_gpu_memory_mib=40000),
                             checkpoints=[dict(id='a'), dict(id='b')])
        for target, value in (('load_protocol', (self.protocol, 'f' * 64)), ('output_dir', self.output),
                              ('launch_tmux', '0::/user.slice/user-1.slice/session-2.scope')):
            patcher = mock.patch.object(holdout_addendum, target, return_value=value)
            self.addCleanup(patcher.stop)
            setattr(self, target, patcher.start())

    def start(self, retry=False, alive=False):
        with mock.patch.object(holdout_addendum, 'tmux_alive', return_value=alive):
            return holdout_addendum.start('prepared', retry=retry,
                                          discover_nodes=lambda _: [node('n1', [gpu('n1', 0), gpu('n1', 1)])])

    def fail_first_launch(self):
        first = self.start()
        attempt = self.output / 'evaluations/a/attempt-001'
        attempt.mkdir(parents=True)
        (attempt / 'evaluation.log').write_text('412it [24:40]')
        (attempt.parent / 'result.json').write_text(json.dumps(dict(status='failed', attempt=1)))
        (self.output / 'status.json').write_text(json.dumps(dict(status='interrupted')))
        return first

    def test_sigint_is_ignored_and_sigterm_or_sighup_stop(self):
        import os
        import signal
        import threading
        saved = {n: signal.getsignal(n) for n in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
        self.addCleanup(lambda: [signal.signal(n, h) for n, h in saved.items()])
        for stopping in (signal.SIGTERM, signal.SIGHUP):
            stop, received = threading.Event(), []
            with mock.patch('sys.stderr'):
                holdout_addendum.supervise_signals(stop, received)
                os.kill(os.getpid(), signal.SIGINT)
                self.assertFalse(stop.is_set())
                os.kill(os.getpid(), stopping)
            self.assertTrue(stop.is_set())
            self.assertEqual([(r['signal'], r['stopped']) for r in received],
                             [('SIGINT', False), (stopping.name, True)])

    def test_attach_hint_is_read_only(self):
        self.assertIn(' attach -r -t holdout-', self.start()['attach'])

    def test_failed_launch_needs_retry_and_keeps_its_records(self):
        first = self.fail_first_launch()
        with self.assertRaisesRegex(RuntimeError, '--retry'):
            self.start()
        second = self.start(retry=True)
        self.assertEqual(second['status'], 'submitted_to_persistent_tmux')
        archive = self.output / 'failed_launches/001'
        self.assertEqual(second['previous_launch_records'], str(archive))
        self.assertEqual(json.loads((archive / 'launch.json').read_text())['started_at'], first['started_at'])
        self.assertEqual(json.loads((archive / 'status.json').read_text())['status'], 'interrupted')
        self.assertEqual(json.loads((archive / 'a.result.json').read_text())['status'], 'failed')
        self.assertEqual((self.output / 'evaluations/a/attempt-001/evaluation.log').read_text(), '412it [24:40]')
        self.assertEqual(self.launch_tmux.call_count, 2)
        self.start(retry=True)  # the second launch also ended without a report
        self.assertEqual(json.loads((self.output / 'failed_launches/002/launch.json').read_text())['started_at'],
                         second['started_at'])

    def test_running_and_completed_launches_are_not_relaunched(self):
        self.start()
        self.assertEqual(self.start(retry=True, alive=True)['status'], 'already_running')
        with mock.patch.object(holdout_addendum, 'completed', return_value=True):
            self.assertEqual(self.start(retry=True)['status'], 'already_completed')
        self.assertEqual(self.launch_tmux.call_count, 1)

    def test_retry_needs_an_earlier_launch(self):
        with self.assertRaisesRegex(RuntimeError, 'Nothing to retry'):
            self.start(retry=True)
        self.launch_tmux.assert_not_called()

    def test_progress_reads_the_last_batch_count(self):
        console = self.output / 'console.log'
        console.write_bytes(b'0it [00:00, ?it/s]\r438it [24:20,  5.14it/s]\r440it [24:30,  6.54it/s]')
        self.assertEqual(holdout_addendum.last_batch(console), 440)
        self.assertIsNone(holdout_addendum.last_batch(self.output / 'missing.log'))


def save_bundle(directory, epoch, value, counter=0):
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(dict(w=torch.tensor([float(value)])), directory / f'net_epoch-{epoch}_state.pt')
    torch.save(dict(state={}, param_groups=[dict(lr=1.4e-5, params=[0])],
                    lookahead=dict(step_counter=counter, slow_weights=[torch.tensor([float(value)])])),
               directory / f'net_epoch-{epoch}_optimizer.pt')
    torch.save(dict(scale=1024.0, _growth_tracker=counter), directory / f'net_epoch-{epoch}_scaler.pt')


def worker(batches, rows, exhausted, metric, seconds):
    return dict(started_at='2026-11-03T00:00:00+00:00', finished_at=f'2026-11-03T00:{seconds // 60:02d}:{seconds % 60:02d}+00:00',
                metrics={PRIMARY: metric, 'train_data_audit': dict(batches=batches, consumed=dict(count=rows), exhausted=exhausted)})


def write_run(root, name, generations, extra=None):
    (root / name).mkdir(parents=True, exist_ok=True)
    manifest = dict(status='completed', generations=generations, **(extra or {}))
    (root / name / 'manifest.json').write_text(json.dumps(manifest))


class ReproductionChecksTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        patcher = mock.patch.object(reproduction_checks, 'RUNS', self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        save_bundle(self.root / reproduction_checks.CONTROL / 'lr_14e-6', 18, 1.5)

    def fixture(self, *, final=1.5, rows=(1024,) * 5, metric=0.33):
        member = 'lr_14e-6'
        save_bundle(self.root / 'one' / member, 18, 1.5, counter=4)
        write_run(self.root, 'one', [dict(epoch=18, workers={member: worker(2102, 5120, True, 0.33, 240)})])
        for epoch in range(18, 23):
            save_bundle(self.root / 'fifths' / member, epoch, final if epoch == 22 else 0.0, counter=4)
        batches = [420, 420, 420, 420, 422]
        write_run(self.root, 'fifths', [dict(epoch=18 + i, workers={member: worker(
            b, r, i == 4, metric if i == 4 else 0.5, 120)}) for i, (b, r) in enumerate(zip(batches, rows))],
            dict(config=dict(shared=dict(batches_per_generation=420))))
        return reproduction_checks.full_state('one', 'fifths')

    def test_equivalent_restarted_epoch(self):
        result = self.fixture()
        self.assertTrue(result['equivalent'])
        self.assertEqual(result['cost']['restarted_over_uninterrupted'], 2.5)
        self.assertTrue(result['original_control_epoch18_model_identical'])

    def test_any_difference_fails(self):
        self.assertIn('state/w: tensor values differ', self.fixture(final=1.25)['bundles']['differences'])
        self.assertFalse(self.fixture(final=1.25)['equivalent'])
        self.assertFalse(self.fixture(rows=(1024, 1024, 1024, 1024, 1000))['equivalent'])
        self.assertFalse(self.fixture(metric=0.34)['equivalent'])

    def test_anchor_reports_identical_and_differing_epochs(self):
        member = 'lr_14e-6'
        for run, values in ((reproduction_checks.CONTROL, (1.5, 2.0)), ('high', (1.5, 2.5))):
            for epoch, value in zip((18, 19), values):
                save_bundle(self.root / run / member, epoch, value)
            write_run(self.root, run, [dict(epoch=e, workers={member: dict(metrics={PRIMARY: m})})
                                       for e, m in zip((18, 19), (0.33, 0.32 if run == 'high' else 0.31))])
        result = reproduction_checks.anchor('high')
        self.assertEqual((result['identical_epochs'], result['first_differing_epoch']), (1, 19))
        self.assertAlmostEqual(result['max_abs_metric_delta'], 0.01)


if __name__ == '__main__':
    unittest.main()
