"""Stage 3 preparation: protocol, gate, orchestration, recovery and analysis (CPU only)."""
import copy
import hashlib
import io
import json
import math
import os
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tests.helpers import PROJECT_DIR
from tests.test_stage2a_holdout import synthetic_result
from experiments.stage2a.common import lock_file
from experiments.stage2b.smoke import Status
from experiments.stage3 import analysis, campaign, common, gate as gate_module, protocol as frozen, testset
from scripts.experiments import stage3_campaign as cli
from training.checkpoints import archive_selected_best
from training.pbt.config import contract_fingerprint
from training.runtime import WORKING_POINT_DEFINITION, mistag_percent_key

PRIMARY = common.PRIMARY
WORKING_POINTS = [mistag_percent_key(p, e) for p, e in WORKING_POINT_DEFINITION]
NAMES = ('lr_3e-6', 'lr_5_75e-6', 'lr_8_5e-6', 'lr_11_25e-6', 'lr_14e-6')
LRS = dict(zip(NAMES, (3e-6, 5.75e-6, 8.5e-6, 1.125e-5, 1.4e-5)))


def load_protocol():
    return frozen.load()[0]


def fake_protocol(root, protocol=None):
    """Frozen protocol with run locations redirected into a temporary directory."""
    protocol = copy.deepcopy(protocol or load_protocol())
    for arm in common.ARMS:
        protocol['identity']['configs'][arm]['output_root'] = str(Path(root) / 'pbt')
    protocol['identity']['gate']['output_root'] = str(Path(root) / 'gate')
    return protocol


class ProtocolTest(unittest.TestCase):
    def test_frozen_protocol_is_unchanged_and_matches_current_inputs(self):
        protocol, digest = frozen.load()
        self.assertEqual(digest, (common.PREPARED / 'protocol_v2.sha256').read_text().strip())
        self.assertEqual(frozen.drift(protocol), [])
        original = common.load_json(common.PROTOCOL_V1)
        self.assertEqual(protocol['administrative_revision']['supersedes'],
                         dict(file='protocol.json', sha256=(common.PREPARED / 'protocol.sha256').read_text().strip()))
        self.assertTrue(all(protocol[key] == original[key] for key in frozen.SCIENTIFIC))
        for arm in common.ARMS:
            self.assertEqual(protocol['identity']['configs'][arm]['fingerprint'], original['identity']['configs'][arm]['fingerprint'])
        self.assertEqual(protocol['horizon']['decision'], 'fixed_50_epochs')
        self.assertEqual(protocol['design']['horizon_epochs'], 50)
        self.assertEqual(protocol['design']['data']['train']['rows'], 23_921_449)
        self.assertEqual(protocol['design']['data']['sealed_test']['rows'], 8_970_477)

    def test_drift_reports_changed_configs_inputs_and_code(self):
        protocol = load_protocol()
        current = copy.deepcopy(protocol['identity'])
        current['configs']['windowed_pbt_v2']['fingerprint'] = 'horizon-30'
        current['initial_checkpoint']['state']['sha256'] = 'corrupt'
        current['policy_sha256'] = 'changed'
        current['execution_files']['scripts/training/pbt/reference/windowed_v2.py'] = 'edited'
        current['train_manifest']['sha256'] = 'missing-shard'
        self.assertEqual(frozen.drift(protocol, current), [
            'configs', 'execution_files:scripts/training/pbt/reference/windowed_v2.py', 'initial_checkpoint',
            'policy_sha256', 'train_manifest'])

    def test_edited_protocol_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'protocol_v2.json'
            path.write_text(common.PROTOCOL.read_text().replace('fixed_50_epochs', 'continuation_30_50'))
            (Path(tmp) / 'protocol_v2.sha256').write_text((common.PREPARED / 'protocol_v2.sha256').read_text())
            with self.assertRaisesRegex(ValueError, 'edited'):
                frozen.load(path)

    def test_administrative_revision_cannot_change_science_or_lose_the_original(self):
        from experiments.stage2a.common import sha256 as digest
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            def write(name, payload):
                (tmp / f'{name}.json').write_text(json.dumps(payload))
                (tmp / f'{name}.sha256').write_text(digest(tmp / f'{name}.json'))
            original, revision = common.load_json(common.PROTOCOL_V1), common.load_json(common.PROTOCOL)
            for name in ('protocol.json', 'protocol.sha256'):
                (tmp / name).write_bytes((common.PREPARED / name).read_bytes())
            write('protocol_v2', revision)
            self.assertEqual(frozen.load(tmp / 'protocol_v2.json', tmp / 'protocol.json')[0]['name'], original['name'])
            changed = copy.deepcopy(revision)
            changed['design']['horizon_epochs'] = 30
            write('protocol_v2', changed)
            with self.assertRaisesRegex(ValueError, 'scientific'):
                frozen.load(tmp / 'protocol_v2.json', tmp / 'protocol.json')
            write('protocol_v2', revision)
            write('protocol', dict(original, frozen_at='rewritten'))
            with self.assertRaisesRegex(ValueError, 'original'):
                frozen.load(tmp / 'protocol_v2.json', tmp / 'protocol.json')
            with self.assertRaisesRegex(ValueError, 'already frozen'):
                frozen.freeze_revision(original=tmp / 'protocol.json', path=tmp / 'protocol_v2.json')

    def test_horizon_extension_is_not_equivalent_but_native_control_is(self):
        evidence = frozen.horizon_evidence()
        self.assertFalse(evidence['equivalent'])
        self.assertEqual(evidence['first_divergence_epoch'], 30)
        boundary = {d['epoch']: d for d in evidence['continuation_30_to_50']}
        native = {d['epoch']: d for d in evidence['native_50']}
        self.assertTrue(boundary[30]['terminal'] and not boundary[30]['copies'])
        self.assertTrue(native[30]['copies'] and not native[30]['terminal'])
        self.assertTrue(frozen.horizon_evidence(continuation_horizon=50)['equivalent'])

    def test_expected_accepted_rows_from_reweighting(self):
        from experiments.stage2b import dataset as data
        spec, _ = data.load_spec()
        mean, sigma = frozen.expected_accepted_rows(spec, PROJECT_DIR / 'checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/data_config.auto.yaml')
        self.assertAlmostEqual(mean, 21_512_820.98, places=1)
        self.assertGreater(sigma, 1000)

    def test_committed_sealed_test_evidence(self):
        evidence = common.load_json(common.PREPARED / 'test3000k_provenance.json')
        self.assertEqual(evidence['rows_total'], 8_970_477)
        for flavor in evidence['flavors']:
            self.assertEqual(set(flavor['scalar_overlap'].values()), {0})
            self.assertEqual(flavor['unique_scalar_records'], flavor['rows'])

    def test_parity_holds_and_detects_horizon_or_test_leak(self):
        self.assertTrue(cli.parity_checks()['passed'])
        original = cli.resolve
        for change in (lambda c: c['shared'].update(generations=30), lambda c: c['shared'].update(validation_suffix='test3000k'),
                       lambda c: c.update(continuation=dict(source_run='x'))):
            def resolve(path, change=change):
                config = original(path)
                if 'windowed' in str(path):
                    change(config)
                return config
            with patch.object(cli, 'resolve', resolve):
                self.assertFalse(cli.parity_checks()['passed'])


def gate_fixture(root, *, accepted=21_512_000, primary=0.41, skipped=3, cuda=21539.7, caps=False, scaler=True,
                 grad_norm='2.00000', log_skipped=None, audit_skipped=None, log_lines=None, metric_changes=None,
                 extra_batches=0):
    """Synthetic single-member gate run in the real checkpoint and Weaver log formats.

    The manifest gradient norm mirrors the shared log parser, which records only decimal values.
    """
    run = Path(root) / 'gate_a001'
    member = run / 'lr_14e-6'
    member.mkdir(parents=True)
    torch.save(dict(w=torch.tensor([1.0, 2.0])), member / 'net_epoch-17_state.pt')
    torch.save(dict(state={0: dict(step=torch.tensor(100.0))}, param_groups=[dict(lr=1e-4, params=[0])]), member / 'net_epoch-17_optimizer.pt')
    batches = math.ceil(accepted / 1024) + extra_batches
    torch.save(dict(w=torch.tensor([1.1, 2.1])), member / 'net_epoch-18_state.pt')
    torch.save(dict(state={0: dict(step=torch.tensor(100.0 + batches - skipped))}, param_groups=[dict(lr=1.4e-5, params=[0])]),
               member / 'net_epoch-18_optimizer.pt')
    if scaler:
        torch.save(dict(scale=65536.0, _growth_tracker=3), member / 'net_epoch-18_scaler.pt')
    metrics = {k: primary for k in (PRIMARY, *WORKING_POINTS)}
    audit_skipped = skipped if audit_skipped is None else audit_skipped
    metrics.update(train_loss=0.3, validation_loss=0.29, train_amp_skipped_optimizer_steps=skipped,
                   train_max_cuda_memory_mb=cuda,
                   train_data_audit=dict(exhausted=True, batches=batches, optimizer_steps=batches - audit_skipped,
                                         consumed=dict(count=accepted, unique_ids=accepted, repeated_ids=0),
                                         traversal=[dict(scanned=dict(count=23_921_449), accepted=dict(count=accepted),
                                                         exhausted=True, wraps=0)]),
                   validation_data_audit=dict(exhausted=True, consumed=dict(count=150000)))
    if grad_norm is not None and gate_module.DECIMAL.fullmatch(grad_norm):
        metrics['train_max_grad_norm'] = float(grad_norm)
    for key, value in (metric_changes or {}).items():
        if value is None:
            metrics.pop(key, None)
        else:
            metrics[key] = value
    if log_lines is None:
        log_lines = ([] if grad_norm is None else [f'Max Grad Norm: {grad_norm}']) + \
                    [f'AMP skipped optimizer steps: {skipped if log_skipped is None else log_skipped}']
    log = run / 'logs' / 'lr_14e-6' / 'generation-000.log'
    log.parent.mkdir(parents=True)
    log.write_text(''.join(f'[2026-10-20 00:44:00,000] INFO: {line}\n' for line in
                           ['Epoch #18 training', 'Train AvgLoss: 0.30000, AvgAcc: 0.88000', *log_lines]))
    command = ['weaver', '--data-train', 'x'] + (['--samples-per-epoch', '1'] if caps else [])
    manifest = dict(status='completed', config=dict(shared=dict(samples_per_epoch=None)),
                    initial_evaluation=dict(metrics={PRIMARY: 0.45}),
                    generations=[dict(index=0, epoch=18, status='completed', workers={'lr_14e-6': dict(
                        lr=1.4e-5, status='completed', returncode=0, command=command, metrics=metrics, log=str(log),
                        started_at='2026-10-20T00:00:00+00:00', finished_at='2026-10-20T00:45:00+00:00')})])
    (run / 'manifest.json').write_text(json.dumps(manifest))
    observations = dict(exit_code=0, leftover_processes=[], foreign_gpu_events=[], peak_gpu_memory_mib=24894,
                        peak_rss_bytes=19 * 1024**3, timing=dict(train_rate=8500.0, startup_seconds=70.0))
    return run, observations


class GateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.protocol = load_protocol()
        self.policy, _ = common.load_policy()
        self.passing = patch('validation.verify_fixed_lr.verify', return_value=dict(passed=True, failures=[]))
        self.passing.start()

    def tearDown(self):
        self.passing.stop()
        self.tmp.cleanup()

    def verdict(self, **fixture):
        observations = fixture.pop('observations', None)
        run, default = gate_fixture(Path(self.tmp.name) / str(len(list(Path(self.tmp.name).iterdir()))), **fixture)
        verdict, checks, measurements = gate_module.verify_gate(run, self.protocol, self.policy, observations or default)
        return verdict, [c['check'] for c in checks if not c['passed']], measurements

    def test_real_full_epoch_evidence_passes_and_projects_budget(self):
        verdict, failed, measurements = self.verdict()
        self.assertEqual((verdict, failed), ('passed', []))
        projection = measurements['projection_concurrent']
        self.assertEqual(projection['member_epochs'], 500)
        self.assertAlmostEqual(projection['training_gpu_hours'], 500 * 2700 * 1.3 / 3600)
        self.assertLess(projection['wall_hours'], measurements['projection_sequential']['wall_hours'])

    def test_criteria_failures_block(self):
        cases = [(dict(primary=float('nan')), 'finite loss'), (dict(accepted=21_000_000), 'accepted rows'),
                 (dict(caps=True), 'no sample caps'), (dict(scaler=False), 'bundle'),
                 (dict(cuda=30000.0), 'CUDA memory'), (dict(skipped=500), 'AMP skipped')]
        for fixture, expected in cases:
            verdict, failed, _ = self.verdict(**fixture)
            self.assertEqual(verdict, gate_module.CRITERIA)
            self.assertTrue(any(expected in f for f in failed), (expected, failed))
        _, default = gate_fixture(Path(self.tmp.name) / 'obs')
        for change, expected in ((dict(peak_rss_bytes=40 * 1024**3), 'host RAM'), (dict(timing=dict(train_rate=500.0, startup_seconds=60)), 'throughput'),
                                 (dict(leftover_processes=[9]), 'clean termination'), (dict(peak_gpu_memory_mib=50000), 'GPU memory')):
            _, failed, _ = self.verdict(observations=dict(default, **change))
            self.assertTrue(any(expected in f for f in failed), expected)

    def test_verifier_failure_and_budget_overrun_block(self):
        with patch('validation.verify_fixed_lr.verify', return_value=dict(passed=False, failures=['epoch 18: traversal'])):
            self.assertEqual(self.verdict()[0], gate_module.CRITERIA)
        self.policy['budgets']['max_campaign_gpu_hours'] = 10
        _, failed, _ = self.verdict()
        self.assertIn('projected campaign GPU-hours within budget', failed)

    GRADIENT = 'maximum gradient norm finite, or +inf only with confirmed AMP-skipped steps'
    METRICS = 'finite loss, primary metric and eight working points'

    def checks(self, **fixture):
        run, observations = gate_fixture(Path(self.tmp.name) / str(len(list(Path(self.tmp.name).iterdir()))), **fixture)
        verdict, checks, _ = gate_module.verify_gate(run, self.protocol, self.policy, observations)
        return verdict, {c['check']: c for c in checks}

    def test_gradient_norm_finite_or_inf_only_from_amp_skips(self):
        for fixture, reason in ((dict(), 'finite'), (dict(grad_norm='0.00000', skipped=0), 'finite'),
                                (dict(grad_norm='inf', skipped=7), 'inf only from AMP-skipped steps')):
            verdict, checks = self.checks(**fixture)
            self.assertEqual(verdict, 'passed', (fixture, [c for c in checks.values() if not c['passed']]))
            self.assertEqual(checks[self.GRADIENT]['detail']['reason'], reason)
        _, checks = self.checks(grad_norm='inf', skipped=7)
        self.assertEqual(checks[self.GRADIENT]['detail'], dict(
            logged='inf', amp_skipped_log='7', amp_skipped_manifest=7, amp_skipped_audit=7, manifest_value=None,
            reason='inf only from AMP-skipped steps'))

    def test_gradient_norm_rejections(self):
        cases = [(dict(grad_norm='inf', skipped=0), 'without AMP-skipped steps'),
                 (dict(grad_norm='nan'), 'NaN'), (dict(grad_norm='-inf'), 'NaN'), (dict(grad_norm='2.0x'), 'malformed'),
                 (dict(grad_norm='1e999'), 'overflows'), (dict(grad_norm=None), 'found 0 and 1'),
                 (dict(log_lines=['Max Grad Norm: 2.00000', 'Max Grad Norm: 2.00000', 'AMP skipped optimizer steps: 3']),
                  'found 2 and 1'),
                 (dict(log_lines=['Max Grad Norm: inf']), 'found 1 and 0'),
                 (dict(grad_norm='inf', metric_changes=dict(train_amp_skipped_optimizer_steps=None)), 'AMP skip counter'),
                 (dict(grad_norm='inf', metric_changes=dict(train_amp_skipped_optimizer_steps=True)), 'AMP skip counter'),
                 (dict(grad_norm='inf', skipped=7, log_skipped=6), 'between log and manifest'),
                 (dict(grad_norm='inf', skipped=7, log_skipped='x'), 'malformed'),
                 (dict(grad_norm='inf', skipped=7, audit_skipped=6), 'data-audit step accounting'),
                 (dict(grad_norm='inf', skipped=7, metric_changes=dict(train_max_grad_norm=3.0)), 'manifest records'),
                 (dict(metric_changes=dict(train_max_grad_norm=None)), 'disagrees with the log'),
                 (dict(metric_changes=dict(train_max_grad_norm=2.5)), 'disagrees with the log'),
                 (dict(grad_norm='inf', skipped=500), 'AMP fraction'),
                 (dict(grad_norm='inf', skipped=7, extra_batches=1), 'step accounting or AMP fraction')]
        for fixture, reason in cases:
            verdict, checks = self.checks(**fixture)
            self.assertEqual(verdict, gate_module.CRITERIA, fixture)
            self.assertFalse(checks[self.GRADIENT]['passed'], fixture)
            self.assertIn(reason, checks[self.GRADIENT]['detail']['reason'], fixture)
            json.dumps(checks[self.GRADIENT]['detail'], allow_nan=False)

    def test_missing_worker_log_fails(self):
        run, observations = gate_fixture(Path(self.tmp.name) / 'nolog')
        (run / 'logs' / 'lr_14e-6' / 'generation-000.log').unlink()
        verdict, checks, _ = gate_module.verify_gate(run, self.protocol, self.policy, observations)
        failed = {c['check']: c for c in checks if not c['passed']}
        self.assertEqual(verdict, gate_module.CRITERIA)
        self.assertEqual(failed[self.GRADIENT]['detail']['reason'], 'worker log missing')

    def test_nonfinite_loss_and_working_points_report_observed_values(self):
        working_point = WORKING_POINTS[3]
        for changes, expected in ((dict(train_loss=float('nan')), {'train_loss': 'nan'}),
                                  (dict(validation_loss=float('inf')), {'validation_loss': 'inf'}),
                                  (dict(train_loss=None), {'train_loss': 'missing'}),
                                  ({PRIMARY: float('-inf'), working_point: float('nan')}, {PRIMARY: '-inf', working_point: 'nan'}),
                                  ({working_point: None}, {working_point: 'missing'}),
                                  ({working_point: '0.1'}, {working_point: "'0.1'"})):
            for grad in ('2.00000', 'inf'):
                verdict, checks = self.checks(metric_changes=changes, grad_norm=grad, skipped=7)
                self.assertEqual(verdict, gate_module.CRITERIA, changes)
                self.assertFalse(checks[self.METRICS]['passed'])
                self.assertEqual(checks[self.METRICS]['detail'], expected)
                self.assertTrue(checks[self.GRADIENT]['passed'], changes)

    def test_gate_revision_record_is_pinned_and_recoverable(self):
        path = common.PREPARED / 'gate_revision_1.json'
        record = common.load_json(path)
        self.assertEqual(common.sha256(path), (common.PREPARED / 'gate_revision_1.sha256').read_text().strip())
        self.assertEqual(record['protocol']['sha256'], (common.PREPARED / 'protocol_v2.sha256').read_text().strip())
        self.assertFalse(record['protocol']['changed'])
        correction = record['correction']
        self.assertEqual(correction['corrected_sha256'], common.sha256(PROJECT_DIR / correction['file']))
        original = subprocess.run(['git', 'show', f'{correction["original_commit"]}:{correction["file"]}'], cwd=PROJECT_DIR,
                                  capture_output=True, check=True).stdout
        self.assertEqual(hashlib.sha256(original).hexdigest(), correction['original_sha256'])
        self.assertNotIn(correction['file'], frozen.EXECUTION_FILES)
        self.assertEqual(record['failed_attempt']['verdict_sha256'],
                         'b5c217ee971cdccac58fda810ba08a7b6bf9d33c85f9a9a1d33ed3c077e8d3f5')


def bundle(directory, epoch, value, lr=1e-5):
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(dict(w=torch.tensor([float(value)])), directory / f'net_epoch-{epoch}_state.pt')
    torch.save(dict(state={0: dict(step=torch.tensor(float(epoch)))}, param_groups=[dict(lr=lr, params=[0])]),
               directory / f'net_epoch-{epoch}_optimizer.pt')
    torch.save(dict(scale=1024.0, _growth_tracker=epoch), directory / f'net_epoch-{epoch}_scaler.pt')


class LedgerAndStateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.protocol = fake_protocol(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def manifest(self, completed=2, status='running'):
        return dict(status=status, generations=[dict(index=i, epoch=18 + i, status='completed', workers={'a': {}}) for i in range(completed)])

    def test_ledger_detects_corruption_rollback_and_duplicates(self):
        run = self.root / 'run'
        for epoch in (18, 19):
            bundle(run / 'a', epoch, epoch)
        manifest = self.manifest()
        ledger = campaign.update_ledger(self.root, 'fixed_lr', run, manifest)
        self.assertEqual(sorted(ledger['generations']), ['0', '1'])
        self.assertEqual(campaign.verify_ledger(self.root, 'fixed_lr', run, manifest, full=True)['verified'], 2)
        (run / 'a' / 'net_epoch-19_optimizer.pt').write_bytes(b'corrupt')
        with self.assertRaisesRegex(campaign.CampaignFailure, 'corruption'):
            campaign.verify_ledger(self.root, 'fixed_lr', run, manifest)
        with self.assertRaisesRegex(campaign.CampaignFailure, 'rolled back'):
            campaign.verify_ledger(self.root, 'fixed_lr', run, self.manifest(completed=1))
        duplicate = self.manifest()
        duplicate['generations'].append(dict(duplicate['generations'][1]))
        with self.assertRaisesRegex(campaign.CampaignFailure, 'duplicate'):
            campaign.completed_generations(duplicate)

    def test_arm_state_never_restarts_silently(self):
        self.assertEqual(campaign.arm_state(self.protocol, 'fixed_lr')[0], 'absent')
        run = campaign.arm_run(self.protocol, 'fixed_lr')
        run.mkdir(parents=True)
        with self.assertRaisesRegex(campaign.CampaignFailure, 'without a manifest'):
            campaign.arm_state(self.protocol, 'fixed_lr')
        fingerprint = contract_fingerprint(common.resolve(common.ARM_CONFIGS['fixed_lr']))
        (run / 'manifest.json').write_text(json.dumps(dict(fingerprint='other-config', status='failed')))
        with self.assertRaisesRegex(campaign.CampaignFailure, 'different configuration'):
            campaign.arm_state(self.protocol, 'fixed_lr')
        (run / 'manifest.json').write_text(json.dumps(dict(fingerprint=fingerprint, status='failed')))
        self.assertEqual(campaign.arm_state(self.protocol, 'fixed_lr')[0], 'resumable')
        (run / 'manifest.json').write_text(json.dumps(dict(fingerprint=fingerprint, status='completed')))
        self.assertEqual(campaign.arm_state(self.protocol, 'fixed_lr')[0], 'completed')
        command = campaign.runner_command(common.ARM_CONFIGS['fixed_lr'], 5, resume=True)
        self.assertEqual(command[-3:], ['--gpus', '0,1,2,3,4', '--resume'])


class FakeLeases:
    """Lease provider with a fixed number of idle GPUs (counts concurrent holders)."""

    def __init__(self, root, idle):
        self.root, self.idle, self.held, self.peak = Path(root), idle, set(), 0

    def count(self, cohort, allocation):
        return self.idle - len(self.held)

    def acquire(self, output, cohort, allocation, stop):
        free = [i for i in range(self.idle) if i not in self.held]
        if not free:
            raise RuntimeError('busy')
        index = free[0]
        lease = lock_file(self.root / f'gpu{index}.lock')
        self.held.add(index)
        self.peak = max(self.peak, len(self.held))
        original = lease.close
        def close(index=index, original=original):
            self.held.discard(index)
            original()
        lease.close = close
        return dict(host='node', index=index, name='A100', uuid=f'GPU-{index}'), lease


class FakeObserver:
    def __init__(self, uuids, policy):
        self.uuids, self.peak_gpu_memory_mib, self.peak_rss_bytes, self.foreign_events = set(uuids), 1, 1, []

    def sample(self, roots):
        return {}


FAKE_ARM = r'''
import json, os, signal, sys, time
from pathlib import Path
arm, run = sys.argv[1], Path(sys.argv[2])
mode = os.environ.get('FAKE_' + arm.upper(), 'complete')
run.mkdir(parents=True, exist_ok=True)
log = run.parent / f'{arm}.calls'
with log.open('a') as stream:
    stream.write(' '.join(sys.argv[3:]) + '\n')
manifest = dict(status='running', fingerprint=os.environ['FAKE_FINGERPRINT_' + arm.upper()], generations=[])
def save():
    (run / 'manifest.json').write_text(json.dumps(manifest))
def interrupted(*_):
    manifest['status'] = 'interrupted'; save(); sys.exit(130)
signal.signal(signal.SIGINT, interrupted)
save()
if mode == 'fail':
    manifest['status'] = 'failed'; save(); sys.exit(2)
if mode == 'hang':
    while True:
        time.sleep(0.05)
time.sleep(0.3)
manifest['generations'] = [dict(index=0, epoch=18, status='completed', workers={'lr_3e-6': {}})]
manifest['status'] = 'completed'
save()
'''


class OrchestrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.protocol = fake_protocol(self.root)
        self.policy, _ = common.load_policy()
        self.policy['allocation']['arm_wait_poll_seconds'] = 0
        self.fake = self.root / 'fake_arm.py'
        self.fake.write_text(FAKE_ARM)
        self.status = Status(self.root / 'out', {})
        self.gate = dict(measurements=dict(peak_rss_bytes=1))
        env = {f'FAKE_FINGERPRINT_{a.upper()}': contract_fingerprint(common.resolve(common.ARM_CONFIGS[a])) for a in common.ARMS}
        self.patches = [patch.dict(os.environ, env), patch.object(campaign, 'check_memory', lambda *a: dict(ok=True)),
                        patch.object(campaign, 'runner_command', self.command)]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.tmp.cleanup()

    def command(self, config, gpus, *, resume=False, name=None):
        arm = next(a for a, path in common.ARM_CONFIGS.items() if path == config)
        return [sys.executable, str(self.fake), arm, str(campaign.arm_run(self.protocol, arm)), f'--gpus={gpus}'] + (
            ['--resume'] if resume else [])

    def run_arms(self, leases, stop=None, **kwargs):
        return campaign.run_arms(self.protocol, self.policy, dict(host='node'), self.root / 'out', self.status,
                                 stop or threading.Event(), self.gate, acquire=leases.acquire, idle=leases.count,
                                 observer_factory=FakeObserver, poll_seconds=0.05, **kwargs)

    def test_concurrent_arms_with_ten_gpus(self):
        leases = FakeLeases(self.root, 10)
        self.assertEqual(self.run_arms(leases), dict(fixed_lr='completed', windowed_pbt_v2='completed'))
        self.assertEqual(leases.peak, 10)
        self.assertEqual(leases.held, set())
        allocations = common.load_json(self.root / 'out' / 'allocations.json')
        self.assertEqual(sorted(a['arm'] for a in allocations), ['fixed_lr', 'windowed_pbt_v2'])
        self.assertTrue((self.root / 'out' / 'ledger' / 'fixed_lr.json').is_file())

    def test_five_gpus_run_arms_sequentially_without_hoarding(self):
        leases = FakeLeases(self.root, 7)
        self.assertEqual(self.run_arms(leases), dict(fixed_lr='completed', windowed_pbt_v2='completed'))
        self.assertEqual(leases.peak, 5)
        self.assertEqual(leases.held, set())

    def test_failed_worker_marks_only_that_arm_and_rerun_resumes(self):
        with patch.dict(os.environ, FAKE_WINDOWED_PBT_V2='fail'):
            outcome = self.run_arms(FakeLeases(self.root, 10))
        self.assertEqual(outcome, dict(fixed_lr='completed', windowed_pbt_v2='failed'))
        outcome = self.run_arms(FakeLeases(self.root, 10))
        self.assertEqual(outcome['windowed_pbt_v2'], 'completed')
        calls = (campaign.arm_run(self.protocol, 'windowed_pbt_v2').parent / 'windowed_pbt_v2.calls').read_text().splitlines()
        self.assertTrue(calls[-1].endswith('--resume'))
        self.assertEqual(len((campaign.arm_run(self.protocol, 'fixed_lr').parent / 'fixed_lr.calls').read_text().splitlines()), 1)

    def test_interruption_and_budget_stop_runners_resumably(self):
        stop = threading.Event()
        timer = threading.Timer(0.5, stop.set)
        timer.start()
        with patch.dict(os.environ, FAKE_FIXED_LR='hang', FAKE_WINDOWED_PBT_V2='hang'):
            with self.assertRaises(InterruptedError):
                self.run_arms(FakeLeases(self.root, 10), stop=stop)
        for arm in common.ARMS:
            self.assertEqual(common.load_json(campaign.arm_run(self.protocol, arm) / 'manifest.json')['status'], 'interrupted')
        with patch.dict(os.environ, FAKE_FIXED_LR='hang', FAKE_WINDOWED_PBT_V2='hang'), \
             patch.object(campaign, 'consumed_gpu_hours', return_value=10_000):
            with self.assertRaisesRegex(campaign.CampaignFailure, 'budget'):
                self.run_arms(FakeLeases(self.root, 10))

    def test_busy_gpus_wait_then_time_out_without_launching(self):
        self.policy['allocation']['max_arm_wait_hours'] = 0
        outcome = self.run_arms(FakeLeases(self.root, 3))
        self.assertEqual(outcome, dict(fixed_lr='waiting_timeout', windowed_pbt_v2='waiting_timeout'))
        self.assertFalse((self.root / 'out' / 'allocations.json').exists())

    def test_acquire_group_is_all_or_nothing(self):
        leases = FakeLeases(self.root, 3)
        self.assertIsNone(campaign.acquire_group(self.root, {}, self.policy['allocation'], threading.Event(), 5,
                                                 acquire=leases.acquire))
        self.assertEqual(leases.held, set())

    def test_gate_attempts_reuse_block_and_retry(self):
        run_gate = lambda **kw: campaign.run_gate(self.protocol, 'protocol-sha', self.policy, dict(host='node'), self.root / 'out',
                                                  self.status, threading.Event(), acquire=FakeLeases(self.root, 1).acquire,
                                                  observer_factory=FakeObserver, poll_seconds=0.05, **kw)
        gate_cmd = lambda config, gpus, name=None, resume=False: [sys.executable, '-c', 'raise SystemExit(0)']
        with patch.object(campaign, 'runner_command', gate_cmd):
            failing = lambda run, p, policy, obs: ('failed_criteria', [dict(check='finite loss', passed=False)], {})
            with self.assertRaisesRegex(campaign.GateBlocked, 'finite loss'):
                run_gate(verify=failing)
            with self.assertRaisesRegex(campaign.GateBlocked, 'blocked'):
                run_gate(verify=failing)
            passing = lambda run, p, policy, obs: ('passed', [], dict(peak_rss_bytes=1))
            record = run_gate(verify=passing, retry=True)
            self.assertEqual((record['verdict'], record['attempt']), ('passed', 2))
            self.assertEqual(run_gate(verify=failing)['attempt'], 2)  # passed gate is reused, never rerun
        with patch.object(campaign, 'runner_command', lambda *a, **k: [sys.executable, '-c', 'raise SystemExit(1)']):
            out = self.root / 'other'
            (self.root / 'x').mkdir()
            with self.assertRaisesRegex(campaign.GateBlocked, 'exited 1'):
                campaign.run_gate(self.protocol, 'p2', self.policy, dict(host='node'), out, self.status, threading.Event(),
                                  acquire=FakeLeases(self.root / 'x', 1).acquire, observer_factory=FakeObserver, poll_seconds=0.05)
            self.assertEqual(gate_module.gate_record(out)['verdict'], gate_module.OPERATIONAL)


def arm_fixture(root, arm, *, divergence=False, generations=12, steps=21_000, copy_at=None):
    """Two-arm manifests with real member bundles; windowed copies optional."""
    run = Path(root) / 'pbt' / arm
    config = common.resolve(common.ARM_CONFIGS[arm])
    rows = []
    for g in range(generations):
        epoch = 18 + g
        workers = {}
        for i, name in enumerate(NAMES):
            value = 0.40 - 0.001 * g + 0.002 * i
            if arm == 'windowed_pbt_v2' and divergence and name == 'lr_3e-6':
                value += 1e-4
            bundle(run / name, epoch, value + (0.5 if divergence and arm == 'windowed_pbt_v2' and name == 'lr_8_5e-6' else 0))
            metrics = {k: value for k in (PRIMARY, *WORKING_POINTS)}
            metrics.update(train_loss=0.3 - 0.001 * g, validation_loss=0.29,
                           train_data_audit=dict(optimizer_steps=steps, consumed=dict(count=steps * 1024, sequence_sha256=str(g))))
            workers[name] = dict(lr=LRS[name], status='completed', returncode=0, metrics=metrics,
                                 started_at='2026-10-20T00:00:00+00:00', finished_at='2026-10-20T00:40:00+00:00')
        exploit = []
        if copy_at is not None and g == copy_at and arm == 'windowed_pbt_v2':
            pre = dict(state=dict(sha256=common.sha256(run / 'lr_3e-6' / f'net_epoch-{epoch}_state.pt')))
            exploit = [dict(donor='lr_14e-6', recipient='lr_3e-6', new_lr=1.12e-5, mutation_factor=0.8, pre_copy=pre)]
        rows.append(dict(index=g, epoch=epoch, status='completed', workers=workers, exploit=exploit))
    best_member = NAMES[0]
    best = archive_selected_best(run / 'checkpoints', {p: run / best_member / f'net_epoch-{18 + generations - 1}_{p}.pt'
                                                       for p in ('state', 'optimizer', 'scaler')},
                                 dict(generation=generations - 1, epoch=18 + generations - 1, member=best_member,
                                      metric_value=0.39), require_scaler=True)
    manifest = dict(status='completed', config=config, members={n: dict(name=n, lr=LRS[n]) for n in NAMES},
                    generations=rows, best=best, initial_evaluation=dict(metrics={PRIMARY: 0.45}))
    (run / 'manifest.json').write_text(json.dumps(manifest))
    return str(run)


class AnalysisTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.protocol = load_protocol()
        self.verifier = type('V', (), dict(verify=staticmethod(lambda run, through=None: dict(passed=True, failures=[]))))

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, runs, **kwargs):
        self.protocol['design']['horizon_epochs'] = 12
        return analysis.build(self.root / 'report', self.protocol, runs, verifier=self.verifier, **kwargs)

    def test_complete_matched_campaign_reports_preregistered_endpoints(self):
        runs = {arm: arm_fixture(self.root, arm, copy_at=9) for arm in common.ARMS}
        data = self.build(runs)
        self.assertEqual(data['status'], 'COMPLETE', [c for c in data['validity'] if not c['passed']])
        values = {n: [0.40 - 0.001 * g + 0.002 * i for g in range(12)] for i, n in enumerate(NAMES)}
        expected = sum(min(v) for v in list(zip(*values.values()))[-10:]) / 10
        self.assertAlmostEqual(data['summaries']['fixed_lr']['final10_current_best'], expected)
        primary = data['comparison']['primary_final10_current_best']
        self.assertEqual(primary['delta_pp'], 0.0)
        self.assertTrue(primary['within_selection_noise_margin'])
        self.assertEqual(data['lineage_identity']['first_copy'], {'lr_3e-6': 10})
        report = (self.root / 'report' / 'report.md').read_text()
        self.assertIn('COMPLETE', report)
        self.assertIn('статистическая значимость не утверждается', report)
        self.assertEqual(len(list((self.root / 'report' / 'figures').glob('*.png'))), 4)
        for artifact in common.load_json(self.root / 'report' / 'report_integrity.json')['artifacts']:
            common.verify_file(artifact)

    def test_unmatched_budget_nondeterminism_incomplete_and_nonfinite_are_invalid(self):
        runs = {arm: arm_fixture(self.root, arm) for arm in common.ARMS}
        manifest = common.load_json(Path(runs['windowed_pbt_v2']) / 'manifest.json')
        manifest['generations'][3]['workers']['lr_5_75e-6']['metrics']['train_data_audit']['consumed']['count'] -= 1024
        (Path(runs['windowed_pbt_v2']) / 'manifest.json').write_text(json.dumps(manifest))
        data = self.build(runs)
        self.assertEqual(data['status'], 'INVALID')
        self.assertIsNone(data['comparison'])
        self.assertIn('НЕДЕЙСТВИТЕЛЬНО', (self.root / 'report' / 'report.md').read_text())
        other = tempfile.TemporaryDirectory()
        self.addCleanup(other.cleanup)
        runs = {arm: arm_fixture(other.name, arm, divergence=True) for arm in common.ARMS}
        data = self.build(runs)
        failed = [c['check'] for c in data['validity'] if not c['passed']]
        self.assertEqual(failed, ['same-lineage members bit-identical until their first copy'])
        self.assertEqual((data['status'], data['determinism_confirmed']), ('COMPLETE', False))
        self.assertIn('Детерминизм не подтверждён', (self.root / 'report' / 'report.md').read_text())
        third = tempfile.TemporaryDirectory()
        self.addCleanup(third.cleanup)
        runs = {arm: arm_fixture(third.name, arm, generations=12 if arm == 'fixed_lr' else 11) for arm in common.ARMS}
        self.protocol['design']['horizon_epochs'] = 12
        data = analysis.build(Path(third.name) / 'r', self.protocol, runs, verifier=self.verifier)
        self.assertIn('windowed_pbt_v2: completed 12 generations', [c['check'] for c in data['validity'] if not c['passed']])
        manifest = common.load_json(Path(runs['fixed_lr']) / 'manifest.json')
        manifest['generations'][5]['workers']['lr_3e-6']['metrics'][PRIMARY] = float('nan')
        (Path(runs['fixed_lr']) / 'manifest.json').write_text(json.dumps(manifest))
        self.assertEqual(analysis.build(Path(third.name) / 'r2', self.protocol, runs, verifier=self.verifier,
                                        interim=True)['status'], 'INVALID')

    def test_interim_is_labelled_and_never_compares_the_test(self):
        runs = {arm: arm_fixture(self.root, arm) for arm in common.ARMS}
        data = self.build(runs, interim=True)
        self.assertEqual(data['status'], 'INTERIM')
        self.assertIsNone(data['sealed_test'])
        self.assertIn('ПРОМЕЖУТОЧНЫЙ', (self.root / 'report' / 'report.md').read_text())


class SealedTestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_selection_freeze_pairs_and_paired_metrics(self):
        runs = {arm: arm_fixture(self.root, arm) for arm in common.ARMS}
        manifests = {a: dict(common.load_json(Path(r) / 'manifest.json'), run_dir=r) for a, r in runs.items()}
        checkpoints = testset.selection_freeze(manifests)
        self.assertEqual(len(checkpoints), 12)
        self.assertTrue(all({'path', 'sha256', 'size_bytes'} <= set(c['checkpoint']) for c in checkpoints))
        current = [c for c in checkpoints if c.get('final_current_best')]
        self.assertEqual({c['member'] for c in current}, {'lr_3e-6'})
        pairs = testset.preregistered_pairs(checkpoints)
        self.assertEqual(len(pairs), 7)
        frozen_protocol = dict(dataset=dict(rows=180, audit_fingerprint='data', holdout_files=[dict(rows=60)] * 3))
        results = {}
        for seed, key in enumerate(['fixed_lr__selected_best', 'windowed_pbt_v2__selected_best']):
            item = next(c for c in checkpoints if c['id'] == key)
            results[key], _ = synthetic_result(self.root / 'pred', item, frozen_protocol, seed=seed + 3)
        metrics, comparisons = testset.paired(results, frozen_protocol, {'selected_best': ('fixed_lr__selected_best',
                                                                                          'windowed_pbt_v2__selected_best')})
        row = comparisons['selected_best']
        self.assertAlmostEqual(row['delta_pp'], metrics['windowed_pbt_v2__selected_best'][PRIMARY] - metrics['fixed_lr__selected_best'][PRIMARY])
        self.assertEqual(row['confidence']['status'], 'unavailable')
        self.assertEqual(set(row['working_points']), set(WORKING_POINTS))

    def test_isolation_and_conversion_identity_fail_closed(self):
        from experiments.stage2b import dataset as data
        spec = data.validate_spec(common.load_json(common.TEST_SPEC))
        train, _ = data.load_spec()
        rows = lambda n, offset=0: np.column_stack([np.full(n, 5.0), np.arange(n * 6, dtype=float).reshape(n, 6) + offset])
        def root_reader(path):
            label = {'bb': 5.0, 'cc': 4.0, 'dd': 2.0}[Path(path).name.split('_')[3]]
            values = rows(4, 0 if 'test3000k' in str(path) else 1000)
            values[:, 0] = label
            return values
        spec_small = copy.deepcopy(spec)
        for source in spec_small['sources']:
            source['entries'] = 4
        evidence = testset.verify_test_isolation(spec_small, train, root_reader=root_reader, parquet_reader=lambda p: rows(3, 5000))
        self.assertEqual(evidence['rows_total'], 12)
        leaking = lambda path: root_reader(str(path).replace('val1000k', 'test3000k')) if 'val1000k' in str(path) else rows(3, 5000)
        with self.assertRaisesRegex(ValueError, 'shares scalar records'):
            testset.verify_test_isolation(spec_small, train, root_reader=root_reader, parquet_reader=leaking)
        with self.assertRaisesRegex(ValueError, 'differs from its ROOT'):
            testset.verify_converted_matches_root(spec_small, root_reader=root_reader, parquet_reader=lambda p: rows(4, 7))

    def test_evaluation_contract_requires_one_file_per_flavour(self):
        import awkward as ak
        shared = common.resolve(common.ARM_CONFIGS['fixed_lr'])['shared']
        for flavor in ('bb', 'cc', 'dd'):
            ak.to_parquet(ak.Array(dict(label=np.arange(3))), self.root / f'ml_flavtag_6cat_{flavor}_test3000k.parquet')
        contract = testset.evaluation_contract(self.root, 'test3000k', arm_shared=shared, seed=12345, name='t')
        self.assertEqual((contract['dataset']['rows'], contract['evaluation']['validation_suffix']), (9, 'test3000k'))
        self.assertTrue(contract['evaluation']['save_prediction_row_ids'])
        (self.root / 'x_ml_flavtag_6cat_bb_test3000k.parquet').write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'exactly one bb'):
            testset.evaluation_contract(self.root, 'test3000k', arm_shared=shared, seed=12345, name='t')

    def test_finalize_keeps_test_sealed_when_validation_is_not_complete(self):
        protocol = fake_protocol(self.root)
        for arm in common.ARMS:
            campaign.arm_run(protocol, arm).mkdir(parents=True)
            (campaign.arm_run(protocol, arm) / 'manifest.json').write_text(json.dumps(dict(status='completed', generations=[])))
        status = Status(self.root / 'out', {})
        evaluate = unittest.mock.Mock()
        with patch.object(campaign, 'verify_ledger'), \
             patch.object(analysis, 'build', return_value=dict(status='INVALID')):
            with self.assertRaisesRegex(campaign.CampaignFailure, 'stays closed'):
                campaign.finalize(protocol, 'sha', common.load_policy()[0], dict(host='node'), self.root / 'out', status,
                                  threading.Event(), {}, evaluate=evaluate)
        evaluate.assert_not_called()
        self.assertFalse((self.root / 'out' / 'sealed_test').exists())


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.output = Path(self.tmp.name)
        self.checks = (dict(git=dict(commit='abc'), protocol_sha256='p'), load_protocol(), 'p', common.load_policy()[0])

    def tearDown(self):
        self.tmp.cleanup()

    def remote(self, cgroup):
        """Fake ssh: record remote commands; the pane cgroup query returns ``cgroup``."""
        calls = []

        def run(args, **kwargs):
            calls.append(args[-1])
            stdout = cgroup + '\n' if '/cgroup' in args[-1] else ''
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr='')
        return calls, run

    def test_supervisor_starts_outside_the_oomd_managed_user_service(self):
        calls, run = self.remote('0::/user.slice/user-1082.slice/session-7.scope')
        with patch.object(cli.subprocess, 'run', run):
            cgroup = cli.launch_tmux('gpu-node-02', 'stage3-x', self.output, self.output / 'cohort.json')
        self.assertEqual(cgroup, '0::/user.slice/user-1082.slice/session-7.scope')
        new_session = next(c for c in calls if 'new-session' in c)
        self.assertTrue(new_session.startswith('env -u DBUS_SESSION_BUS_ADDRESS -u XDG_RUNTIME_DIR tmux -L ilc-campaign '))
        self.assertFalse(any('kill-session' in c for c in calls))

    def test_supervisor_in_an_oomd_managed_scope_is_stopped(self):
        calls, run = self.remote('0::/user.slice/user-1082.slice/user@1082.service/tmux-spawn-1.scope')
        with patch.object(cli.subprocess, 'run', run), self.assertRaisesRegex(RuntimeError, 'oomd-managed'):
            cli.launch_tmux('gpu-node-02', 'stage3-x', self.output, self.output / 'cohort.json')
        self.assertIn('tmux -L ilc-campaign kill-session -t =stage3-x', calls)

    def test_liveness_checks_the_campaign_socket_then_the_default_socket(self):
        for code, legacy, alive in ((0, False, True), (1, True, True), (1, False, False)):
            result = subprocess.CompletedProcess([], code, stdout='', stderr='')
            with patch.object(cli.subprocess, 'run', return_value=result), \
                    patch.object(cli, 'default_socket_alive', return_value=legacy):
                self.assertEqual(cli.tmux_alive('gpu-node-02', 'stage3-x'), alive)
        with patch.object(cli.subprocess, 'run', return_value=subprocess.CompletedProcess([], 255, stdout='', stderr='x')), \
                self.assertRaisesRegex(RuntimeError, 'persistent-session'):
            cli.tmux_alive('gpu-node-02', 'stage3-x')

    def nodes(self, busy=()):
        gpus = [dict(host='gpu-node-02', index=i, name='NVIDIA A100 80GB PCIe', total_mib=81920, used_mib=9000 if i in busy else 0,
                     utilization=0, uuid=f'GPU-{i}', pids=[1] if i in busy else []) for i in range(10)]
        return [dict(host='gpu-node-02', gpus=gpus, error=None)]

    def invoke(self, *argv):
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = cli.main(['--output', str(self.output), *argv])
        return code, stream.getvalue()

    def test_dry_run_plans_without_launching(self):
        with patch.object(cli, 'preflight', lambda output: self.checks), patch.object(cli, 'discover', lambda a: self.nodes()), \
             patch.object(cli, 'launch_tmux', side_effect=AssertionError('launched')):
            code, text = self.invoke('--dry-run')
        plan = json.loads(text)
        self.assertEqual((code, plan['training_launched'], plan['idle_campaign_gpus_observed']), (0, False, 10))
        self.assertEqual(plan['expected_schedule'], 'concurrent arms (10 GPUs)')

    def test_busy_cluster_failed_submission_lock_and_disk(self):
        with patch.object(cli, 'preflight', lambda output: self.checks), patch.object(cli, 'discover', lambda a: self.nodes(range(6))):
            self.assertEqual(self.invoke()[0], common.EXIT_RESOURCES)
        with patch.object(cli, 'preflight', lambda output: self.checks), patch.object(cli, 'discover', lambda a: self.nodes()), \
             patch.object(cli, 'cluster_snapshot', lambda: ''), patch.object(cli, 'ssh_args', lambda h, c: ['false']):
            self.assertEqual(self.invoke()[0], common.EXIT_RESOURCES)
        self.assertIn('Remote tmux submission', common.load_json(self.output / 'last_failure.json')['error'])
        with lock_file(self.output / '.launch.lock'):
            self.assertEqual(self.invoke()[0], common.EXIT_LOCKED)
        def no_disk(output):
            raise OSError(28, 'Insufficient disk')
        with patch.object(cli, 'preflight', no_disk):
            self.assertEqual(self.invoke()[0], common.EXIT_RESOURCES)

    def test_live_session_is_reused_stale_session_is_replaced(self):
        common.write_json(self.output / 'launch.json', dict(host='gpu-node-02', session='stage3-old'))
        with patch.object(cli, 'tmux_alive', return_value=True):
            code, text = self.invoke()
        self.assertEqual((code, json.loads(text)['status']), (0, 'already_running'))
        launched = []
        with patch.object(cli, 'tmux_alive', return_value=False), patch.object(cli, 'preflight', lambda output: self.checks), \
             patch.object(cli, 'discover', lambda a: self.nodes()), patch.object(cli, 'cluster_snapshot', lambda: ''), \
             patch.object(cli, 'launch_tmux', lambda *a, **k: launched.append(a)):
            code, text = self.invoke()
        self.assertEqual((code, json.loads(text)['status'], len(launched)), (0, 'submitted_to_persistent_tmux', 1))

    def test_failed_gate_blocks_submission_until_retry(self):
        common.write_json(self.output / 'gate' / 'verdict.json', dict(protocol_sha256='p', verdict=gate_module.CRITERIA, attempt=1,
                                                                      checks=[dict(check='finite loss', passed=False)]))
        with patch.object(cli, 'preflight', lambda output: self.checks):
            code, text = self.invoke()
        self.assertEqual((code, json.loads(text)['status']), (common.EXIT_FAILED, 'blocked_by_failed_gate'))
        with patch.object(cli, 'preflight', lambda output: self.checks), patch.object(cli, 'discover', lambda a: self.nodes()):
            self.assertEqual(json.loads(self.invoke('--dry-run', '--retry-gate')[1])['gate_state'], gate_module.CRITERIA)

    def on_node(self, **phases):
        import socket
        protocol, digest = frozen.load()
        cohort = self.output / 'cohort.json'
        common.write_json(cohort, dict(host=socket.gethostname().split('.')[0], git=dict(commit='abc'), protocol_sha256=digest))
        patches = [patch.object(cli, 'repository_revision', lambda: dict(commit='abc')), patch.object(campaign, 'install_stop')]
        patches += [patch.object(campaign, name, value) for name, value in phases.items()]
        for item in patches:
            item.start()
        try:
            return self.invoke('--on-node', '--cohort', str(cohort))[0], common.load_json(self.output / 'status.json')
        finally:
            for item in reversed(patches):
                item.stop()

    def test_on_node_phase_order_and_failure_handling(self):
        calls = []
        record = lambda name, value=None: (lambda *a, **k: calls.append(name) or value)
        code, status = self.on_node(prepare_data=record('prep'), run_gate=record('gate', dict(verdict='passed')),
                                    run_arms=record('arms', dict(fixed_lr='completed', windowed_pbt_v2='completed')),
                                    finalize=record('final', dict(status='COMPLETE')))
        self.assertEqual((code, status['status'], calls), (0, 'completed', ['prep', 'gate', 'arms', 'final']))
        def blocked(*a, **k):
            raise campaign.GateBlocked('gate failed_criteria: finite loss')
        code, status = self.on_node(prepare_data=record('prep'), run_gate=blocked, run_arms=record('never'))
        self.assertEqual((code, status['status']), (common.EXIT_FAILED, 'gate_blocked'))
        self.assertNotIn('never', calls)
        code, status = self.on_node(prepare_data=record('prep'), run_gate=record('gate', {}),
                                    run_arms=record('arms', dict(fixed_lr='completed', windowed_pbt_v2='failed')))
        self.assertEqual((code, status['status']), (common.EXIT_FAILED, 'failed'))
        code, status = self.on_node(prepare_data=record('prep'), run_gate=record('gate', {}),
                                    run_arms=record('arms', dict(fixed_lr='completed', windowed_pbt_v2='completed')),
                                    finalize=record('final', dict(status='INVALID')))
        self.assertEqual((code, status['report_status']), (common.EXIT_FAILED, 'INVALID'))
        def interrupted(*a, **k):
            raise InterruptedError('Supervisor stopped')
        self.assertEqual(self.on_node(prepare_data=interrupted)[0], common.EXIT_INTERRUPTED)
        with patch.object(frozen, 'drift', return_value=['execution_files:scripts/training/pbt/runner.py']):
            self.assertEqual(self.on_node(prepare_data=record('prep'))[0], common.EXIT_INVALID)

    def test_status_dirty_tree_and_refreeze(self):
        self.assertEqual(json.loads(self.invoke('--status')[1])['status'], 'not_started')
        dirty = subprocess.CompletedProcess([], 0, stdout=' M scripts/x.py\n', stderr='')
        with patch.object(cli, 'git', lambda *a: dirty):
            with self.assertRaisesRegex(ValueError, 'uncommitted'):
                cli.repository_revision()
        self.assertEqual(self.invoke('--freeze')[0], common.EXIT_INVALID)
        self.assertEqual(subprocess.run(['bash', '-n', str(PROJECT_DIR / 'scripts/experiments/run_stage3_train8m.sh')]).returncode, 0)


if __name__ == '__main__':
    unittest.main()
