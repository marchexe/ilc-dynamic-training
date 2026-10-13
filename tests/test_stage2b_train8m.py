"""Stage 2B: train8m preparation, smoke verification and orchestration (CPU only)."""
import argparse
import copy
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import awkward as ak
import numpy as np
import pyarrow.parquet as pq
import torch

from tests.helpers import PROJECT_DIR
from experiments.stage2a.common import lock_file
from experiments.stage2b import common, dataset as data, parity, report, smoke, verify
from experiments.stage2b.provenance import verify_realized_isolation
from scripts.experiments import stage2b_campaign as campaign
from training.checkpoints import archive_selected_best
from training.pbt.config import load_config, validate_inputs
from training.pbt.execution.weaver_command import make_command
from training.runtime import WORKING_POINT_DEFINITION, mistag_percent_key, sha256

# The frozen train8m specification records where its ROOT sources were read from.
SOURCE_DIR = Path(data.load_spec()[0]['source_dir'])
LABELS = dict(bb=5, cc=4, dd=2)
ENTRIES = dict(bb=2500, cc=2300, dd=2100)


def write_root(path, flavor, entries, *, bad_label_at=None, offset=0.0):
    import uproot
    rng = np.random.default_rng(entries + LABELS[flavor])
    labels = np.full(entries, LABELS[flavor], dtype=np.int32)
    if bad_label_at is not None:
        labels[bad_label_at] = 4 if flavor != 'cc' else 5
    counts = rng.integers(1, 6, entries)
    branches = dict(label=labels, pfcand_e=ak.unflatten(rng.random(counts.sum()).astype(np.float32), counts))
    for name in data.SCALAR_FLOATS:
        branches[name] = (rng.random(entries) + offset).astype(np.float32)
    table = ak.Array(branches)
    with uproot.recreate(path) as handle:
        # mktree writes a classic TTree (dict assignment would write an RNTuple).
        tree = handle.mktree('ntp', {k: table[k].type for k in table.fields})
        tree.extend({k: table[k] for k in table.fields})


def load(path, **overrides):
    return load_config(argparse.Namespace(config=Path(path), experiment_name=None, gpus=None, slots=None,
                                          smoke=False, **overrides))


class SyntheticDataset:
    """Three tiny flavour ROOT files and a matching Stage 2B specification."""

    def __init__(self, root, *, row_group_size=500, chunk_entries=1000, bad_label=None):
        self.root = Path(root)
        source = self.root / 'source'
        source.mkdir()
        for flavor, entries in ENTRIES.items():
            write_root(source / f'ml_flavtag_6cat_{flavor}_train8m.root', flavor, entries,
                       bad_label_at=7 if flavor == bad_label else None)
        import uproot
        with uproot.open(source / 'ml_flavtag_6cat_bb_train8m.root') as handle:
            reference = self.root / 'reference.parquet'
            ak.to_parquet(handle['ntp'].arrays(), reference, compression='lz4', row_group_size=row_group_size)
        spec = json.loads(common.SPEC.read_text())
        spec['source_dir'] = str(source)
        spec['output_dir'] = str(self.root / 'train8m')
        spec['sources'] = []
        for flavor in data.FLAVORS:
            path = source / f'ml_flavtag_6cat_{flavor}_train8m.root'
            stat = path.stat()
            spec['sources'].append(dict(flavor=flavor, path=str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                                        **data.root_metadata(path, 'ntp')))
        spec['expected_rows_total'] = sum(ENTRIES.values())
        spec['conversion'] = dict(spec['conversion'], row_group_size=row_group_size, chunk_entries=chunk_entries)
        spec['reference'] = dict(parquet=str(reference), schema_sha256=data.schema_digest(pq.ParquetFile(reference)))
        spec['disk'] = dict(spec['disk'], minimum_free_bytes_after=0)
        self.spec = data.validate_spec(spec)

    def final(self, flavor):
        return data.output_dir(self.spec) / data.output_name(flavor)


class DatasetSpecificationTest(unittest.TestCase):
    def setUp(self):
        self.spec, _ = data.load_spec()

    def test_frozen_spec_contract(self):
        self.assertEqual(self.spec['expected_rows_total'], 23_921_449)
        self.assertEqual([s['entries'] for s in self.spec['sources']], [7_989_060, 7_976_224, 7_956_165])
        self.assertEqual(self.spec['source_selection']['entry_window'], 'Entry$<8000000')
        self.assertEqual(self.spec['flavors'], data.FROZEN_CLASSES)
        self.assertEqual(self.spec['validation']['selection_suffix'], 'val50k_tail')
        self.assertEqual(self.spec['conversion']['chunk_entries'] % self.spec['conversion']['row_group_size'], 0)

    @unittest.skipUnless(SOURCE_DIR.is_dir(), 'train8m ROOT source is not mounted')
    def test_real_sources_match_frozen_identity(self):
        records = data.verify_sources(self.spec)
        self.assertEqual(sum(r['entries'] for r in records), 23_921_449)

    def test_rejects_wrong_classes_and_overlapping_output(self):
        for change in (lambda s: s['flavors']['bb'].update(label=4),
                       lambda s: s['flavors']['cc'].update(class_index=2),
                       lambda s: s.update(output_dir='datasets/20250711_ilc_nnqq_sgv_10m_3cat_parquet'),
                       lambda s: s.update(output_dir='datasets/20250711_ilc_nnqq_sgv_10m_3cat_parquet/train8m'),
                       lambda s: s.update(expected_rows_total=1),
                       lambda s: s['conversion'].update(chunk_entries=12345),
                       lambda s: s['sources'][0].update(path='/x/ml_flavtag_6cat_bb_train800k.root'),
                       lambda s: s['sources'].reverse(),
                       lambda s: s.pop('validation'),
                       lambda s: s['validation'].update(selection_suffix='val50k')):
            broken = copy.deepcopy(self.spec)
            change(broken)
            with self.assertRaises((ValueError, KeyError)):
                data.validate_spec(broken)

    def test_validation_split_guard(self):
        for suffix in ('val5k', 'val50k', 'val_holdout', 'val1000k', 'train800k'):
            with self.assertRaises(data.DatasetError):
                data.check_validation_suffix(self.spec, suffix)
        self.assertEqual(data.check_validation_suffix(self.spec, 'val50k_tail'), 'val50k_tail')

    def test_changed_or_missing_sources_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = SyntheticDataset(tmp)
            data.verify_sources(fixture.spec)
            for change in (lambda s: s['sources'][0].update(size_bytes=1), lambda s: s['sources'][1].update(entries=7),
                           lambda s: s['sources'][2].update(schema_sha256='0' * 64),
                           lambda s: s['sources'][0].update(path=str(Path(tmp) / 'missing' / 'ml_flavtag_6cat_bb_train8m.root'))):
                broken = copy.deepcopy(fixture.spec)
                change(broken)
                with self.assertRaises(data.DatasetError):
                    data.verify_sources(broken)


class ConversionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fixture = SyntheticDataset(self.tmp.name)
        self.spec = self.fixture.spec

    def tearDown(self):
        self.tmp.cleanup()

    def convert_all(self):
        return [data.convert_flavor(self.spec, f) for f in data.FLAVORS]

    def test_conversion_is_exact_layout_controlled_and_idempotent(self):
        import uproot
        self.assertEqual(data.dataset_status(self.spec)['state'], 'absent')
        records = self.convert_all()
        self.assertEqual({r['action'] for r in records}, {'converted'})
        for flavor in data.FLAVORS:
            final = self.fixture.final(flavor)
            meta = pq.ParquetFile(final).metadata
            self.assertEqual([meta.row_group(i).num_rows for i in range(meta.num_row_groups)][:-1],
                             [500] * (meta.num_row_groups - 1))
            with uproot.open(self.spec['sources'][data.FLAVORS.index(flavor)]['path']) as handle:
                expected = handle['ntp'].arrays()
            actual = ak.from_parquet(final)
            self.assertEqual(ak.to_list(actual), ak.to_list(expected))
        mtimes = [self.fixture.final(f).stat().st_mtime_ns for f in data.FLAVORS]
        self.assertEqual({r['action'] for r in self.convert_all()}, {'skipped_verified'})
        self.assertEqual(mtimes, [self.fixture.final(f).stat().st_mtime_ns for f in data.FLAVORS])
        self.assertEqual(data.dataset_status(self.spec)['state'], 'complete')

    def test_manifest_fingerprint_matches_weaver_audit_and_glob_is_unique(self):
        from weaver.utils.data_audit import parquet_manifest
        self.convert_all()
        manifest = data.verify_dataset(self.spec)
        weaver = parquet_manifest(str(self.fixture.final(f).resolve()) for f in data.FLAVORS)
        self.assertEqual(manifest['weaver_audit']['fingerprint'], weaver['fingerprint'])
        self.assertEqual(manifest['rows_total'], sum(ENTRIES.values()))
        extra = self.fixture.final('bb').with_name('copy_ml_flavtag_6cat_bb_train8m.parquet')
        extra.write_bytes(self.fixture.final('bb').read_bytes())
        with self.assertRaisesRegex(data.DatasetError, 'exactly one'):
            data.verify_dataset(self.spec, rehash=False)

    def test_weaver_reads_identical_rows_independent_of_row_group_size(self):
        from weaver.utils.data.fileio import _read_parquet
        self.convert_all()
        other = tempfile.TemporaryDirectory()
        self.addCleanup(other.cleanup)
        coarse = SyntheticDataset(other.name, row_group_size=1000, chunk_entries=2000)
        data.convert_flavor(coarse.spec, 'bb')
        for load_range in ((0.0, 0.01), (0.33, 0.34), (0.99, 1.0)):
            fine_rows = _read_parquet(str(self.fixture.final('bb')), ['label', 'jet_e', 'pfcand_e'], load_range)
            coarse_rows = _read_parquet(str(coarse.final('bb')), ['label', 'jet_e', 'pfcand_e'], load_range)
            self.assertEqual(ak.to_list(fine_rows), ak.to_list(coarse_rows))

    def test_interrupted_conversion_leaves_no_partial_and_resumes(self):
        def interrupted(iterator, destination, **options):
            Path(destination).write_bytes(b'partial')
            next(iterator)
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            data.convert_flavor(self.spec, 'bb', write=interrupted)
        out = data.output_dir(self.spec)
        self.assertFalse(self.fixture.final('bb').exists())
        self.assertEqual(list(out.glob('.*.partial-*')), [])
        stop = threading.Event()
        stop.set()
        with self.assertRaises(InterruptedError):
            data.convert_flavor(self.spec, 'cc', stop=stop)
        self.assertEqual(data.convert_flavor(self.spec, 'bb')['action'], 'converted')

    def test_stale_partial_from_killed_writer_is_discarded(self):
        out = data.output_dir(self.spec)
        out.mkdir(parents=True)
        stale = out / f'.{data.output_name("dd")}.partial-deadhost-999'
        stale.write_bytes(b'killed mid-write')
        self.assertEqual(data.dataset_status(self.spec)['partial'], ['dd'])
        data.convert_flavor(self.spec, 'dd')
        self.assertFalse(stale.exists())

    def test_concurrent_writer_is_blocked_but_stale_lock_file_is_harmless(self):
        data.state_dir(self.spec).mkdir(parents=True)
        lock = data.state_dir(self.spec) / 'bb.lock'
        with lock_file(lock):
            with self.assertRaises(BlockingIOError):
                data.convert_flavor(self.spec, 'bb')
        self.assertTrue(lock.exists())  # left behind by a dead holder: flock is released
        self.assertEqual(data.convert_flavor(self.spec, 'bb')['action'], 'converted')

    def test_wrong_class_label_fails_without_output(self):
        broken = tempfile.TemporaryDirectory()
        self.addCleanup(broken.cleanup)
        fixture = SyntheticDataset(broken.name, bad_label='bb')
        with self.assertRaisesRegex(data.DatasetError, 'Wrong class label'):
            data.convert_flavor(fixture.spec, 'bb')
        self.assertFalse(fixture.final('bb').exists())
        self.assertEqual(list(data.output_dir(fixture.spec).glob('.*.partial-*')), [])

    def test_source_entry_mismatch_and_corrupt_output_are_detected(self):
        broken = copy.deepcopy(self.spec)
        broken['sources'][0]['entries'] -= 1
        broken['expected_rows_total'] -= 1
        with self.assertRaisesRegex(data.DatasetError, 'entries changed'):
            data.convert_flavor(broken, 'bb')
        self.convert_all()
        final = self.fixture.final('cc')
        stat = final.stat()
        raw = bytearray(final.read_bytes())
        raw[len(raw) // 2] ^= 0xFF
        final.write_bytes(bytes(raw))
        os.utime(final, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with self.assertRaisesRegex(data.DatasetError, 'Checksum mismatch'):
            data.verify_dataset(self.spec)
        final.write_bytes(b'truncated')
        self.assertEqual(data.dataset_status(self.spec)['state'], 'partial')
        record = data.convert_flavor(self.spec, 'cc')
        self.assertEqual(record['action'], 'converted')
        self.assertTrue(list(data.output_dir(self.spec).glob(f'.{data.output_name("cc")}.invalid-*')))
        data.verify_dataset(self.spec)

    def test_preflight_requires_disk_and_reference(self):
        with self.assertRaises(OSError) as raised:
            data.preflight(self.spec, free=lambda path: 0)
        self.assertEqual(raised.exception.errno, 28)
        self.assertGreater(data.preflight(self.spec, free=lambda path: 10**15)['required_free_bytes'], 0)
        broken = copy.deepcopy(self.spec)
        broken['reference']['parquet'] = str(Path(self.tmp.name) / 'missing.parquet')
        with self.assertRaisesRegex(data.DatasetError, 'reference'):
            data.preflight(broken, free=lambda path: 10**15)


class ProvenanceTest(unittest.TestCase):
    def test_committed_root_evidence(self):
        evidence = json.loads((common.PREPARED / 'provenance.json').read_text())
        self.assertEqual(evidence['totals']['train8m_rows'], 23_921_449)
        self.assertEqual(evidence['totals']['train8m_val1000k_scalar_overlap'], 0)
        for flavor in evidence['flavors']:
            self.assertTrue(flavor['train800k_is_ordered_prefix'])
            self.assertEqual(flavor['labels'], [LABELS[flavor['flavor']]])
            self.assertEqual(flavor['train8m_duplicate_scalar_records'], 0)
            for legacy in flavor['legacy_overlap_with_train8m'].values():
                self.assertEqual(legacy['overlapping'], legacy['rows'])

    def test_realized_isolation_fails_on_overlap_prefix_or_label(self):
        spec, _ = data.load_spec()
        base = np.column_stack([np.full(6, 5.0), np.arange(6 * 6, dtype=float).reshape(6, 6)])
        def reader(tables):
            def read(path):
                name = Path(path).name
                return tables.get(next((k for k in tables if k in name), 'default'), tables['default'])
            return read
        good = dict(train8m=base, train800k=base[:2], default=base[:0] + 999)
        self.assertEqual(len(verify_realized_isolation(dict(spec, flavors=dict(bb=dict(label=5), cc=dict(label=5),
                                                                               dd=dict(label=5))),
                                                       parquet_scalars=reader(good))['checks']), 3)
        for tables in (dict(good, val50k_tail=base[3:4]), dict(good, train800k=base[1:3]),
                       dict(good, train8m=np.vstack([base, [[4.0, *base[0, 1:]]]]))):
            with self.assertRaises(ValueError):
                verify_realized_isolation(dict(spec, flavors=dict(bb=dict(label=5), cc=dict(label=5), dd=dict(label=5))),
                                          parquet_scalars=reader(tables))


class ParityAndConfigTest(unittest.TestCase):
    def setUp(self):
        self.spec, _ = data.load_spec()

    def test_drafts_are_matched_and_policy_is_frozen(self):
        result = parity.check(self.spec)
        self.assertTrue(result['passed'], [c for c in result['checks'] if not c['passed']])
        self.assertEqual(result['cadence']['non_terminal_copy_boundaries'], [10, 15])
        self.assertEqual(result['budget']['member_epochs'], 100)

    def test_parity_detects_any_unmatched_condition(self):
        perturbations = (('shared', 'seed', 22345), ('shared', 'batch_size', 512), ('shared', 'generations', 25),
                         ('shared', 'validation_suffix', 'val5k'), ('shared', 'train_suffix', 'train800k'),
                         ('shared', 'samples_per_epoch', 1024), ('shared', 'fetch_step', 0.001),
                         ('shared', 'initial_optimizer_mode', 'damped'), ('pbt', 'metric', 'validation_loss'))
        for section, key, value in perturbations:
            def resolver(path, section=section, key=key, value=value):
                config = load(path)
                if 'windowed' in str(path) and 'drafts' in str(path):
                    config[section][key] = value
                return config
            self.assertFalse(parity.check(self.spec, resolver=resolver)['passed'], (section, key))
        def population(path):
            config = load(path)
            if 'drafts' in str(path) and 'windowed' in str(path):
                config['population'][0]['start_lr'] = 4e-6
            return config
        self.assertFalse(parity.check(self.spec, resolver=population)['passed'])
        def policy(path):
            config = load(path)
            if 'drafts' in str(path) and 'windowed' in str(path):
                config['pbt']['windowed_pbt_v2']['decision_margin'] = 0.001
            return config
        self.assertFalse(parity.check(self.spec, resolver=policy)['passed'])

    def test_drafts_never_fall_back_to_other_data(self):
        for path in common.DRAFT_CONFIGS.values():
            config = load(path)
            with tempfile.TemporaryDirectory() as empty:
                config['shared']['dataset'] = empty
                with self.assertRaisesRegex(FileNotFoundError, 'train8m'):
                    validate_inputs(config)

    def test_smoke_config_contract(self):
        policy, _ = common.load_policy()
        config = load(common.SMOKE_CONFIG)
        baseline = load(PROJECT_DIR / 'configs/experiments/foundation_fixed_lr_20epochs.yaml')
        shared, smoke_policy = config['shared'], policy['smoke']
        self.assertEqual(Path(shared['dataset']), data.output_dir(self.spec).resolve())
        self.assertEqual(shared['train_suffix'], 'train8m')
        self.assertEqual(shared['samples_per_epoch'], smoke_policy['steps_per_generation'] * smoke_policy['batch_size'])
        self.assertEqual(shared['generations'], smoke_policy['generations'])
        self.assertEqual(len(config['population']), smoke_policy['members'])
        self.assertGreater(config['pbt']['exploit_interval_generations'], shared['generations'])
        self.assertEqual(data.check_validation_suffix(self.spec, shared['validation_suffix']), smoke_policy['validation_suffix'])
        differing = {k for k in set(shared) | set(baseline['shared']) if shared.get(k) != baseline['shared'].get(k)}
        self.assertEqual(differing, {'dataset', 'train_suffix', 'generations', 'samples_per_epoch'})
        self.assertGreater(smoke_policy['steps_per_generation'] * smoke_policy['batch_size'],
                           0.01 * self.spec['expected_rows_total'] * 0.9)  # crosses a 1% fetch boundary


def fixture_metrics(epoch, lr, train_audit, steps=256, skipped=1):
    metrics = {key: 0.4 for key in (common.PRIMARY, *[mistag_percent_key(p, e) for p, e in WORKING_POINT_DEFINITION])}
    metrics.update(train_loss=0.3, validation_loss=0.29, train_max_grad_norm=1.5, train_max_cuda_memory_mb=21000.0,
                   train_amp_skipped_optimizer_steps=skipped, train_loaded_optimizer_lr=lr,
                   train_data_audit=dict(train_audit, epoch=epoch, batches=steps, optimizer_steps=steps - skipped,
                                         loss=0.3, consumed=dict(count=steps * 1024, unique_ids=steps * 1024, repeated_ids=0)),
                   validation_data_audit=dict(exhausted=True, optimizer_steps=0, dataset=dict(fingerprint='val-fp'),
                                              consumed=dict(count=150000)))
    return metrics


def build_smoke_run(root):
    """A complete synthetic run with real command construction and checkpoint formats."""
    root = Path(root)
    policy, _ = common.load_policy()
    config = load(common.SMOKE_CONFIG)
    train_dir = root / 'train8m'
    config['shared']['dataset'] = str(train_dir)
    run = root / 'pbt' / 'smoke_a001'
    files = [dict(path=str(train_dir / data.output_name(f)), rows=n, sha256=f'{i}' * 64)
             for i, (f, n) in enumerate(dict(bb=10, cc=11, dd=12).items())]
    fingerprint, audit_files, total = data.audit_fingerprint(files)
    dataset_manifest = dict(rows_total=total, weaver_audit=dict(fingerprint=fingerprint),
                            files=[dict(f, resolved_path=f['path']) for f in files])
    train_audit = dict(dataset=dict(fingerprint=fingerprint, total_rows=total, files=audit_files))
    members = {m['name']: dict(name=m['name'], lr=m['start_lr']) for m in config['population']}
    slot = dict(host=None, gpu='0', label='0')
    for index, name in enumerate(members):
        member = run / name
        member.mkdir(parents=True)
        weights = torch.tensor([1.0, 2.0])
        torch.save(dict(w=weights, bn=torch.tensor([1.0])), member / 'net_epoch-17_state.pt')
        torch.save(dict(state={0: dict(step=torch.tensor(1000.0), exp_avg=torch.zeros(2)), 1: dict(step=torch.tensor(5.0))},
                        param_groups=[dict(lr=1e-4, params=[0, 1])]), member / 'net_epoch-17_optimizer.pt')
    generations = []
    for g in range(3):
        epoch = 18 + g
        workers = {}
        for index, (name, member) in enumerate(members.items()):
            command, log_path, target = make_command(config, member, slot, run / name, g)
            self_check = target == epoch
            assert self_check
            log_path.parent.mkdir(parents=True, exist_ok=True)
            amp = ('Initial checkpoint has no AMP scaler; using a fresh common scaler' if g == 0
                   else f'Restored AMP scaler from {run / name}/net_epoch-{epoch - 1}_scaler.pt')
            log_path.write_text(
                f'[2026-10-13 01:00:00,000] INFO: args:\n[2026-10-13 01:00:00,100] INFO: Resume training from epoch {epoch - 1}\n'
                f'[2026-10-13 01:00:00,200] INFO: Overrode loaded optimizer learning rate: 0.0001 -> {member["lr"]:.6g}\n'
                f'[2026-10-13 01:00:00,300] INFO: {amp}\n[2026-10-13 01:01:40,000] INFO: Epoch #{epoch} training\n'
                '[2026-10-13 01:02:30,000] INFO: Processed 262144 entries in total (avg. speed 5242.9 entries/s)\n'
                f'[2026-10-13 01:02:31,000] INFO: Epoch #{epoch} validating\n'
                '[2026-10-13 01:03:07,000] INFO: Processed 150000 entries in total (avg. speed 4166.7 entries/s)\n')
            previous_state = torch.load(run / name / f'net_epoch-{epoch - 1}_state.pt', weights_only=True)
            previous_optimizer = torch.load(run / name / f'net_epoch-{epoch - 1}_optimizer.pt', weights_only=False)
            torch.save(dict(previous_state, w=previous_state['w'] + 0.01 * (index + 1)), run / name / f'net_epoch-{epoch}_state.pt')
            optimizer = copy.deepcopy(previous_optimizer)
            optimizer['state'][0]['step'] = previous_optimizer['state'][0]['step'] + 255
            optimizer['param_groups'][0]['lr'] = member['lr']
            torch.save(optimizer, run / name / f'net_epoch-{epoch}_optimizer.pt')
            torch.save(dict(scale=32768.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, _growth_tracker=3),
                       run / name / f'net_epoch-{epoch}_scaler.pt')
            workers[name] = dict(status='completed', returncode=0, lr=member['lr'], command=command, log=str(log_path),
                                 metrics=fixture_metrics(epoch, member['lr'], train_audit))
        generations.append(dict(index=g, epoch=epoch, status='completed', workers=workers))
    best_member = 'lr_14e-6'
    sources = {part: run / best_member / f'net_epoch-20_{part}.pt' for part in ('state', 'optimizer', 'scaler')}
    best = archive_selected_best(run / 'checkpoints', sources, dict(generation=2, epoch=20, member=best_member,
                                                                     metric_value=0.4), require_scaler=True)
    control = {name: dict(status='completed', returncode=0, metrics={common.PRIMARY: 0.4},
                          checkpoint=str(run / name / 'net_epoch-20_state.pt'),
                          checkpoint_sha256=sha256(run / name / 'net_epoch-20_state.pt')) for name in members}
    control['selected_best'] = dict(status='completed', returncode=0, metrics={common.PRIMARY: 0.4},
                                    checkpoint=best['state_path'], checkpoint_sha256=best['checkpoint']['state']['sha256'])
    manifest = dict(status='completed', next_generation=3, config=config, members=members, generations=generations,
                    initial_evaluation=dict(status='completed', metrics={common.PRIMARY: 0.45}), best=best,
                    final_evaluations=dict(control=control))
    (run / 'manifest.json').write_text(json.dumps(manifest))
    observations = dict(interruption=dict(triggered=True, generation=1, runner_exit_code=130, manifest_status='interrupted',
                                          terminated_workers=list(members), checkpoints_present_at_interrupt=[], stragglers=[]),
                        resume_exit_code=0, leftover_processes=[], foreign_gpu_events=[], peak_gpu_memory_mib=22000,
                        peak_rss_bytes=10**9)
    return run, dict(dataset_manifest=dataset_manifest, validation_audit=dict(fingerprint='val-fp'), policy=policy,
                     observations=observations)


class SmokeVerificationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run, self.inputs = build_smoke_run(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def failed(self, **inputs):
        result = verify.verify_run(self.run, **dict(self.inputs, **inputs))
        return [c['check'] for c in result['checks'] if not c['passed']]

    def edit(self, change):
        manifest = json.loads((self.run / 'manifest.json').read_text())
        change(manifest)
        (self.run / 'manifest.json').write_text(json.dumps(manifest))

    def test_complete_evidence_passes(self):
        result = verify.verify_run(self.run, **self.inputs)
        self.assertTrue(result['passed'], [c for c in result['checks'] if not c['passed']])
        self.assertGreater(len(result['checks']), 60)

    def test_train800k_substitution_fails(self):
        def substitute(manifest):
            worker = manifest['generations'][1]['workers']['lr_3e-6']
            worker['metrics']['train_data_audit']['dataset']['fingerprint'] = 'train800k-fingerprint'
            command = worker['command']
            command[command.index('--data-train') + 1] = command[command.index('--data-train') + 1].replace('train8m', 'train800k')
        self.edit(substitute)
        failures = self.failed()
        self.assertIn('generation 1 lr_3e-6: training audit is the real train8m dataset', failures)
        self.assertIn('generation 1 lr_3e-6: command reads only train8m', failures)

    def test_nonfinite_or_unbounded_or_wrong_steps_fail(self):
        for change, expected in (
                (lambda m: m['generations'][0]['workers']['lr_3e-6']['metrics'].update(train_loss=float('nan')), 'finite losses'),
                (lambda m: m['generations'][2]['workers']['lr_14e-6']['metrics'].update(train_max_cuda_memory_mb=50000.0), 'bounded CUDA'),
                (lambda m: m['generations'][1]['workers']['lr_14e-6']['metrics']['train_data_audit'].update(optimizer_steps=200), 'Ranger state advanced'),
                (lambda m: m['generations'][1]['workers']['lr_14e-6']['metrics']['validation_data_audit'].update(exhausted=False), 'complete validation'),
                (lambda m: m.update(status='interrupted'), 'run completed')):
            backup = (self.run / 'manifest.json').read_text()
            self.edit(change)
            self.assertTrue(any(expected in failure for failure in self.failed()), expected)
            (self.run / 'manifest.json').write_text(backup)

    def test_missing_scaler_restore_and_corrupt_checkpoints_fail(self):
        log = Path(json.loads((self.run / 'manifest.json').read_text())['generations'][2]['workers']['lr_3e-6']['log'])
        log.write_text(log.read_text().replace('Restored AMP scaler', 'Ignored AMP scaler'))
        self.assertIn('generation 2 lr_3e-6: new process restored model, Ranger state, LR and AMP scaler', self.failed())
        best = Path(json.loads((self.run / 'manifest.json').read_text())['best']['state_path'])
        best.write_bytes(best.read_bytes() + b'corrupt')
        failures = self.failed()
        self.assertIn('selected-best bundle integrity (Stage 1)', failures)
        self.assertIn('final evaluation selected_best', failures)
        state = self.run / 'lr_14e-6' / 'net_epoch-19_state.pt'
        torch.save(dict(w=torch.tensor([float('inf'), 1.0]), bn=torch.tensor([1.0])), state)
        self.assertIn('generation 1 lr_14e-6: finite updated weights', self.failed())

    def test_process_observations_are_required(self):
        observations = self.inputs['observations']
        for change, expected in ((dict(leftover_processes=[123]), 'clean termination'),
                                 (dict(resume_exit_code=1), 'resume completed'),
                                 (dict(foreign_gpu_events=[dict(pid=1)]), 'foreign GPU'),
                                 (dict(interruption=dict(observations['interruption'], triggered=False)), 'interrupted'),
                                 (dict(interruption=dict(observations['interruption'],
                                                         checkpoints_present_at_interrupt=['x'])), 'interrupted')):
            self.assertTrue(any(expected in f for f in self.failed(observations=dict(observations, **change))), expected)

    def test_report_separates_measurements_and_estimates(self):
        result = verify.verify_run(self.run, **self.inputs)
        rows = report.smoke_measurements(self.run)
        self.assertEqual(rows[0]['startup_seconds'], 100.0)
        self.assertEqual(rows[0]['train_seconds'], 50.0)
        self.assertEqual(rows[0]['validation_seconds'], 36.0)
        status = dict(git=dict(commit='abc'), dataset=dict(state='complete_verified', rows_total=23_921_449))
        dry = dict(fixed_lr=dict(returncode=0, uses_train8m=True, weaver_commands=6),
                   windowed_pbt_v2=dict(returncode=0, uses_train8m=True, weaver_commands=6))
        summary = report.build(self.tmp.name, status, smoke=dict(run=str(self.run), attempt=1, verification=result,
                                                                 observations=self.inputs['observations']),
                               parity=dict(passed=True, cadence={}), dry_runs=dry)
        self.assertTrue(summary['ready'])
        self.assertIn('ГОТОВО', (Path(self.tmp.name) / 'report.md').read_text())
        low, high = summary['estimates']['member_epoch_minutes']
        self.assertLess(low, high)
        self.assertTrue(campaign.completed(self.tmp.name) is False)  # no completed status.json yet
        summary = report.build(self.tmp.name, status, smoke=dict(run=str(self.run), attempt=1, verification=result,
                                                                 observations={}), parity=dict(passed=False), dry_runs=dry)
        self.assertFalse(summary['ready'])


FAKE_RUNNER = r'''
import json, os, signal, sys, time
from pathlib import Path
run = Path(os.environ['FAKE_RUN'])
mode = os.environ.get('FAKE_MODE', 'interruptible')
def save(manifest):
    (run / 'manifest.json').write_text(json.dumps(manifest))
if '--resume' in sys.argv:
    manifest = json.loads((run / 'manifest.json').read_text())
    manifest.update(status='completed', next_generation=3)
    save(manifest)
    sys.exit(0)
run.mkdir(parents=True)
logs = {}
for name in ('a', 'b'):
    path = run / 'logs' / name / 'generation-001.log'
    path.parent.mkdir(parents=True)
    path.write_text('Epoch #19 training\n')
    logs[name] = str(path)
manifest = dict(status='running', next_generation=1, generations=[
    dict(index=0, epoch=18, workers=dict(a=dict(status='completed'), b=dict(status='completed'))),
    dict(index=1, epoch=19, workers={n: dict(status='running', log=p) for n, p in logs.items()})])
if mode == 'complete':
    manifest.update(status='completed', next_generation=3)
    for worker in manifest['generations'][1]['workers'].values():
        worker['status'] = 'completed'
    save(manifest)
    sys.exit(0)
save(manifest)
def interrupted(*_):
    manifest['status'] = 'interrupted'
    for worker in manifest['generations'][1]['workers'].values():
        worker['status'] = 'terminated'
    save(manifest)
    sys.exit(130)
signal.signal(signal.SIGINT, interrupted)
while True:
    time.sleep(0.05)
'''


class FakeObserver:
    def __init__(self, uuids, policy):
        self.peak_gpu_memory_mib, self.peak_rss_bytes, self.foreign_events = 1000, 10**8, []

    def sample(self, root):
        return {}


class SupervisorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.policy, _ = common.load_policy()
        self.policy['smoke']['interrupt_delay_seconds'] = 0
        self.fake = self.root / 'fake_runner.py'
        self.fake.write_text(FAKE_RUNNER)
        self.status = smoke.Status(self.root / 'out', {})
        self.leases = []

    def tearDown(self):
        for lease in self.leases:
            lease.close()
        self.tmp.cleanup()

    def acquire(self, output, cohort, allocation, stop):
        lease = lock_file(self.root / f'gpu{len(self.leases)}.lock')
        self.leases.append(lease)
        return dict(host='node', index=len(self.leases) - 1, name='A100', uuid=f'GPU-{len(self.leases)}'), lease

    def run_smoke(self, mode, **kwargs):
        name = f'{self.policy["smoke"]["experiment_prefix"]}_a001'
        environment = dict(FAKE_RUN=str(self.root / 'pbt' / name), FAKE_MODE=mode)
        command = lambda name, gpus, resume=False: [sys.executable, str(self.fake)] + (['--resume'] if resume else [])
        with patch.dict(os.environ, environment), patch.object(smoke, 'runner_command', command):
            return smoke.run_smoke(None, self.policy, {}, dict(host='node'), self.status, threading.Event(),
                                   validation_audit={}, run_root=self.root / 'pbt', acquire=self.acquire,
                                   observer_factory=FakeObserver, poll_seconds=0.05,
                                   verify=lambda run, **_: dict(passed=True, checks=[]), **kwargs)

    def test_interrupt_then_resume_in_new_runner_process(self):
        result = self.run_smoke('interruptible')
        interruption = result['observations']['interruption']
        self.assertTrue(interruption['triggered'])
        self.assertEqual((interruption['runner_exit_code'], interruption['manifest_status']), (130, 'interrupted'))
        self.assertEqual(interruption['terminated_workers'], ['a', 'b'])
        self.assertEqual(result['observations']['resume_exit_code'], 0)
        self.assertEqual(result['observations']['leftover_processes'], [])
        phases = [p['phase'] for p in json.loads(self.status.path.read_text())['phases']]
        self.assertEqual(phases, ['gpu_acquisition', 'smoke_training', 'smoke_resume', 'verification'])
        self.assertIn('[GPU]', (self.root / 'out' / 'campaign.log').read_text())
        self.assertEqual(all(lease.closed for lease in self.leases), True)

    def test_missed_interruption_fails(self):
        with self.assertRaisesRegex(smoke.SmokeFailure, 'missed'):
            self.run_smoke('complete')

    def test_budget_and_foreign_process_abort_only_the_smoke(self):
        ticks = iter([0.0, *([10**9] * 100)])
        with self.assertRaisesRegex(smoke.SmokeFailure, 'budget'):
            self.run_smoke('interruptible', clock=lambda: next(ticks))
        apps = lambda: [dict(pid=1, uuid='GPU-1', used_mib=10)]
        observer = smoke.Observer(['GPU-1'], self.policy, apps=apps, table=lambda: {})
        observer.sample(99999)
        with self.assertRaisesRegex(smoke.SmokeFailure, 'Foreign GPU process'):
            observer.sample(99999)
        greedy = smoke.Observer(['GPU-1'], self.policy, apps=lambda: [dict(pid=7, uuid='GPU-1', used_mib=10**6)],
                                table=lambda: {7: dict(ppid=1, rss=1)})
        with self.assertRaisesRegex(smoke.SmokeFailure, 'memory budget'):
            greedy.sample(7)

    def test_descendants_follow_new_sessions(self):
        table = {1: dict(ppid=0, rss=1), 2: dict(ppid=1, rss=2), 3: dict(ppid=2, rss=3), 4: dict(ppid=9, rss=4)}
        self.assertEqual(smoke.descendants(1, table), {1, 2, 3})

    def test_dataset_phase_fails_fast_and_distinguishes_progress(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        fixture = SyntheticDataset(tmp.name)
        class Failed:
            pid = 0
            def poll(self):
                return 3
        with patch.object(data, 'preflight', lambda spec: dict(status=dict(state='absent', completed=[]), free_bytes=1,
                                                                 required_free_bytes=1)):
            with self.assertRaisesRegex(smoke.SmokeFailure, 'bb conversion failed'):
                smoke.run_dataset_phase(fixture.spec, self.policy, self.status, threading.Event(),
                                        launch=lambda *a, **k: Failed(), poll_seconds=0)
        self.assertIn('[DATASET]', (self.root / 'out' / 'campaign.log').read_text())

    def test_dataset_phase_converts_with_real_workers_and_verifies(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        fixture = SyntheticDataset(tmp.name)
        def launch(command, **kwargs):
            # In-process conversion through the same per-flavour entry point.
            data.convert_flavor(fixture.spec, command[-1])
            return subprocess.Popen([sys.executable, '-c', 'pass'])
        manifest = smoke.run_dataset_phase(fixture.spec, self.policy, self.status, threading.Event(), launch=launch,
                                           poll_seconds=0, isolation=lambda spec: dict(checks=[]),
                                           publish=lambda spec, manifest: Path(tmp.name) / 'manifest.json')
        self.assertEqual(manifest['rows_total'], sum(ENTRIES.values()))
        status = json.loads(self.status.path.read_text())
        self.assertEqual(status['dataset']['state'], 'complete_verified')
        self.assertEqual([p['phase'] for p in status['phases']],
                         ['dataset_preflight', 'dataset_conversion', 'dataset_verification', 'split_isolation'])


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.output = Path(self.tmp.name)
        self.policy, _ = common.load_policy()

    def tearDown(self):
        self.tmp.cleanup()

    def nodes(self, busy=()):
        gpus = [dict(host='gpu-node-02', index=i, name='NVIDIA A100 80GB PCIe', total_mib=81920,
                     used_mib=9000 if i in busy else 0, utilization=0, uuid=f'GPU-{i}', pids=[1] if i in busy else [])
                for i in range(3)]
        other = [dict(host='gpu-node-01', index=0, name='Tesla V100', total_mib=32768, used_mib=0, utilization=0,
                      uuid='V', pids=[])]
        return [dict(host='gpu-node-02', gpus=gpus, error=None), dict(host='gpu-node-01', gpus=other, error=None),
                dict(host='elsewhere', gpus=gpus, error=None)]

    def test_choose_host_requires_idle_homogeneous_pair(self):
        cohort = campaign.choose_host(self.nodes(busy=(0,)), self.policy['allocation'])
        self.assertEqual((cohort['host'], [g['index'] for g in cohort['gpus']]), ('gpu-node-02', [1, 2]))
        with self.assertRaisesRegex(RuntimeError, 'idle GPUs'):
            campaign.choose_host(self.nodes(busy=(0, 1)), self.policy['allocation'])

    def test_busy_gpu_acquisition_is_bounded(self):
        from experiments.stage2a.execution import acquire_gpu
        allocation = dict(self.policy['allocation'], max_resource_wait_seconds=0)
        busy = lambda: [dict(host='node', index=0, name='A100', total_mib=81920, used_mib=9000, utilization=90,
                             uuid='GPU-0', pids=[5])]
        cohort = dict(host='node', name='A100', total_mib=81920, gpus=[])
        with self.assertRaisesRegex(RuntimeError, 'busy'):
            acquire_gpu(self.output, cohort, allocation, threading.Event(), probe=busy)

    def invoke(self, *argv):
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = campaign.main(['--output', str(self.output), *argv])
        return code, stream.getvalue()

    def test_failed_remote_submission_is_reported(self):
        checks = dict(git=dict(commit='abc'), dataset=dict(state='absent'), free_bytes=1, required_free_bytes=1,
                      sources_verified=3)
        with patch.object(campaign, 'preflight', lambda *a: checks), \
             patch.object(campaign, 'discover', lambda allocation: self.nodes()), \
             patch.object(campaign, 'cluster_snapshot', lambda: 'snapshot'), \
             patch.object(campaign, 'ssh_args', lambda host, command: ['false']):
            code, _ = self.invoke()
        self.assertEqual(code, common.EXIT_RESOURCES)
        self.assertIn('Remote tmux submission', json.loads((self.output / 'last_failure.json').read_text())['error'])
        self.assertFalse((self.output / 'launch.json').exists())

    def test_unavailable_gpus_and_dirty_tree_and_lock(self):
        checks = dict(git=dict(commit='abc'), dataset=dict(state='absent'), free_bytes=1, required_free_bytes=1,
                      sources_verified=3)
        with patch.object(campaign, 'preflight', lambda *a: checks), \
             patch.object(campaign, 'discover', lambda allocation: self.nodes(busy=(0, 1, 2))):
            self.assertEqual(self.invoke()[0], common.EXIT_RESOURCES)
        self.assertTrue((self.output / 'resource_failure.json').exists())
        dirty = subprocess.CompletedProcess([], 0, stdout=' M scripts/x.py\n', stderr='')
        with patch.object(campaign, 'git', lambda *args: dirty):
            with self.assertRaisesRegex(ValueError, 'uncommitted'):
                campaign.repository_revision()
        with lock_file(self.output / '.launch.lock'):
            self.assertEqual(self.invoke()[0], common.EXIT_LOCKED)

    def test_insufficient_disk_is_a_resource_failure(self):
        def no_disk(*args):
            raise OSError(28, 'Insufficient disk for train8m')
        with patch.object(campaign, 'preflight', no_disk):
            self.assertEqual(self.invoke()[0], common.EXIT_RESOURCES)

    def test_status_and_completion_detection(self):
        code, text = self.invoke('--status')
        self.assertEqual((code, json.loads(text)['status']), (0, 'not_started'))
        smoke.Status(self.output, dict(host='gpu-node-02')).phase('dataset_conversion', 'DATASET')
        self.assertEqual(json.loads(self.invoke('--status')[1])['phase'], 'dataset_conversion')
        self.assertFalse(campaign.completed(self.output))

    def test_on_node_failure_publishes_not_ready_report_and_detects_drift(self):
        import socket
        cohort = dict(host=socket.gethostname().split('.')[0], git=dict(commit='abc'),
                      spec_sha256=data.load_spec()[1], policy_sha256=common.load_policy()[1])
        path = self.output / 'cohort.json'
        path.write_text(json.dumps(cohort))
        argv = ['--on-node', '--cohort', str(path)]
        with patch.object(campaign.signal, 'signal'), patch.object(campaign, 'repository_revision', lambda: dict(commit='abc')), \
             patch.object(smoke, 'run_dataset_phase', side_effect=data.DatasetError('Wrong class label in bb')):
            self.assertEqual(self.invoke(*argv)[0], common.EXIT_FAILED)
        status = json.loads((self.output / 'status.json').read_text())
        self.assertEqual(status['status'], 'failed')
        text = (self.output / 'report.md').read_text()
        self.assertIn('НЕ ГОТОВО', text)
        self.assertIn('Wrong class label', text)
        self.assertEqual(json.loads((self.output / 'supervisor_exit.json').read_text())['exit_code'], common.EXIT_FAILED)
        self.assertFalse(campaign.completed(self.output))
        with patch.object(campaign.signal, 'signal'), patch.object(campaign, 'repository_revision', lambda: dict(commit='new')):
            self.assertEqual(self.invoke(*argv)[0], common.EXIT_INVALID)

    def test_shell_launcher_and_dry_run_do_not_launch(self):
        self.assertEqual(subprocess.run(['bash', '-n', str(PROJECT_DIR / 'scripts/experiments/run_stage2b_smoke.sh')]).returncode, 0)
        checks = dict(git=dict(commit='abc'), dataset=dict(state='absent'), free_bytes=1, required_free_bytes=1,
                      sources_verified=3)
        with patch.object(campaign, 'preflight', lambda *a: checks), \
             patch.object(campaign, 'discover', lambda allocation: self.nodes()), \
             patch.object(campaign, 'launch_tmux', side_effect=AssertionError('dry run submitted')):
            code, text = self.invoke('--dry-run')
        plan = json.loads(text)
        self.assertEqual((code, plan['training_launched'], plan['conversion_launched']), (0, False, False))
        self.assertTrue((self.output / 'dry_run.json').exists())


if __name__ == '__main__':
    unittest.main()
