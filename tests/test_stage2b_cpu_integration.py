"""Opt-in CPU integration: real train8m rows -> Stage 2B converter -> real Weaver.

Slices of the actual train8m ROOT trees are streamed with the converter's
engine and options (schema digest must equal the pinned production digest) and
trained/resumed with the production PBT command builder, minus GPU/AMP flags
and with a CPU-sized batch. Run with STAGE2B_CPU_INTEGRATION=1 (about four
minutes on the login node; reads 36k source rows, writes ~100 MB to a temp dir).
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.helpers import PROJECT_DIR
from tests.test_stage2b_train8m import load
from experiments.stage2b import common, dataset as data
from training.members import MemberState
from training.pbt.execution.weaver_command import make_command
from training.runtime import read_metrics

ROWS = 12_000
STEPS = 4
# FP32 ParT at the production batch (1024) needs >30 GB on CPU; the GPU smoke keeps 1024.
BATCH = 128


@unittest.skipUnless(os.environ.get('STAGE2B_CPU_INTEGRATION') == '1' and Path(data.load_spec()[0]['source_dir']).is_dir(),
                     'set STAGE2B_CPU_INTEGRATION=1 on a host that mounts the train8m source')
class RealWeaverCpuIntegrationTest(unittest.TestCase):
    def test_converted_train8m_rows_train_checkpoint_resume_and_validate(self):
        import awkward as ak
        import uproot
        from weaver.utils.data_audit import parquet_manifest
        spec, _ = data.load_spec()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            train_dir = tmp / 'train8m'
            train_dir.mkdir()
            files = []
            for record in spec['sources']:
                # Same engine/options as convert_flavor, on the first real train8m entries.
                out = train_dir / data.output_name(record['flavor'])
                with uproot.open(record['path']) as handle:
                    tree = handle[spec['tree']]
                    ak.to_parquet_row_groups((tree.arrays(entry_start=s, entry_stop=min(ROWS, s + 5000)) for s in range(0, ROWS, 5000)),
                                             str(out), compression=spec['conversion']['compression'], row_group_size=1000)
                verified = data.verify_parquet(out, dict(spec, conversion=dict(spec['conversion'], row_group_size=1000)),
                                               record['flavor'], ROWS)
                self.assertEqual(verified['rows'], ROWS)  # includes the pinned production schema digest
                files.append(str(out.resolve()))
            manifest = dict(weaver_audit=parquet_manifest(files), rows_total=3 * ROWS)
            config = load(common.SMOKE_CONFIG)
            config['shared'].update(dataset=str(train_dir.resolve()), validation_suffix='val5k_tail',
                                    samples_per_epoch=STEPS * BATCH, batch_size=BATCH, num_workers=1)
            run = tmp / 'run'
            member = MemberState.from_legacy(dict(name='lr_14e-6', lr=1.4e-5))
            member_dir = run / member.member_id
            member_dir.mkdir(parents=True)
            for part in ('state', 'optimizer'):
                shutil.copy(config['shared'][f'initial_{part}'], member_dir / f'net_epoch-17_{part}.pt')
            for generation in (0, 1):
                command, log_path, epoch = make_command(config, member, dict(host=None, gpu='0', label='0'),
                                                        member_dir, generation)
                amp = command.index('--use-amp')
                command = command[:amp] + command[amp + 3:]
                command[command.index('--gpus') + 1] = ''
                log_path.parent.mkdir(parents=True, exist_ok=True)
                result = subprocess.run(command, cwd=PROJECT_DIR, capture_output=True, text=True, timeout=3600,
                                        env=dict(os.environ, OMP_NUM_THREADS='8'))
                self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
                metrics = read_metrics(log_path)
                audit = metrics['train_data_audit']
                self.assertEqual(audit['dataset']['fingerprint'], manifest['weaver_audit']['fingerprint'])
                self.assertEqual(audit['dataset']['total_rows'], 3 * ROWS)
                self.assertEqual(audit['consumed']['count'], STEPS * BATCH)
                self.assertEqual(audit['consumed']['unique_ids'], STEPS * BATCH)
                self.assertEqual(audit['optimizer_steps'], STEPS)
                self.assertTrue(metrics['validation_data_audit']['exhausted'])
                self.assertEqual(metrics['validation_data_audit']['consumed']['count'], 15000)
                for part in ('state', 'optimizer'):
                    self.assertTrue((member_dir / f'net_epoch-{epoch}_{part}.pt').is_file())
                text = log_path.read_text()
                self.assertIn(f'Resume training from epoch {epoch - 1}', text)
                self.assertIn('Overrode loaded optimizer learning rate', text)
            print(json.dumps(dict(rows=manifest['rows_total'], fingerprint=manifest['weaver_audit']['fingerprint'],
                                  final_primary=metrics.get(common.PRIMARY), train_loss=metrics['train_loss'])))
