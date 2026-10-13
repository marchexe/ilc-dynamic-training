import copy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
import io
from contextlib import redirect_stdout
from unittest.mock import patch

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tests.helpers import PROJECT_DIR
from experiments.stage2a import analysis, common, dataset, execution, protocol, scheduling
from scripts.experiments import stage2a_campaign as launcher
from training.runtime import read_metrics, sha256, WORKING_POINT_DEFINITION
from weaver.utils.nn.metrics import bkg_rejection_at_eff


def policy():
    return dict(authorized_nodes=[f'gpu-node-0{i}' for i in range(1,8)],preferred_node='gpu-node-05',
                max_parallelism=3,min_gpu_memory_mib=40000,max_used_memory_mib=500,max_util_percent=5,
                max_resource_wait_seconds=0,evaluation_timeout_seconds=5)


def gpu(index=0, *, host='gpu-node-05', name='NVIDIA A100 80GB PCIe', memory=81920, used=0, util=0, pids=None):
    return dict(host=host,index=index,name=name,total_mib=memory,used_mib=used,utilization=util,
                uuid=f'GPU-{host}-{index}',pids=pids or [])


def synthetic_protocol(root):
    raw = dict(schema_version=1,primary_metric=common.PRIMARY,working_points=[list(p) for p in WORKING_POINT_DEFINITION],
               dataset=dict(rows=180,audit_fingerprint='data',holdout_files=[dict(rows=60)]*3),
               analysis=dict(bootstrap_replicates=20,bootstrap_seed=7),allocation=policy(),
               evaluation=dict(dataset=str(root),validation_suffix='val_holdout',data_extension='parquet',
                               network_config=str(PROJECT_DIR/'networks/pretrained_sgv_particle_transformer.py'),
                               data_config=str(PROJECT_DIR/'checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/data_config.auto.yaml'),
                               batch_size=1024,num_workers=1,fetch_step=.01,use_amp=True,amp_dtype='fp16',
                               deterministic=True,data_audit=True,save_predictions=True,seed=12345),
               no_holdout_reselection=True,matched_pairs=protocol.MATCHED_PAIRS,checkpoints=[])
    for ident in protocol.EXPECTED_IDS:
        state = root/(ident+'.pt');state.write_bytes(ident.encode())
        raw['checkpoints'].append(dict(id=ident,selection='validation_primary_global_best',generation=2,weaver_epoch=20,
            training_seed=12345,horizon=50,checkpoint=common.file_record(state),
            validation_metrics={common.PRIMARY:1.},global_best_validation_metric=1.,final10_current_best=1.1))
    return raw


def synthetic_result(root, checkpoint, frozen, *, seed=7):
    directory = root/checkpoint['id'];directory.mkdir(parents=True,exist_ok=True)
    n = frozen['dataset']['rows'];labels = np.repeat(np.arange(3),n//3)
    rng = np.random.default_rng(seed)
    scores = rng.uniform(.05,1,(n,3)).astype('<f4');scores /= scores.sum(axis=1,keepdims=True)
    predictions = directory/'evaluation.predictions.parquet'
    ids = np.arange(n,dtype='<i8'); seq=hashlib.sha256(ids.tobytes()).hexdigest()
    ak.to_parquet(ak.Array(dict(scores=scores,_label_=labels,__weaver_row_id__=ids)),predictions)
    consumed = dict(count=n,unique_ids=n,repeated_ids=0,sequence_sha256=seq,id_set_sha256=seq)
    audit = dict(dataset=dict(fingerprint='data',total_rows=n),consumed=consumed,exhausted=True,
                 loss=.5,accuracy=.9,optimizer_steps=0,prediction_sha256=hashlib.sha256(scores.tobytes()).hexdigest(),
                 traversal=[dict(exhausted=True,wraps=0,accepted=dict(count=n))])
    log = directory/'evaluation.log'
    curves = bkg_rejection_at_eff(labels,scores)
    log.write_text('Eval AvgLoss: 0.5, AvgAcc: 0.9\n    - bkg_rejection_at_eff:\n'+json.dumps(curves)+
                   '\n[fixture] Data audit validation: '+json.dumps(audit)+'\n')
    record = dict(status='completed',checkpoint_id=checkpoint['id'],protocol_sha256='protocol',returncode=0,
                  checkpoint=checkpoint['checkpoint']['path'],checkpoint_sha256=checkpoint['checkpoint']['sha256'],
                  log=str(log),prediction_path=str(predictions),metrics=read_metrics(log),
                  artifacts=[common.file_record(log),common.file_record(predictions)])
    return record,(ids,labels,scores)


class GPUAllocationTest(unittest.TestCase):
    def test_prefers_authorized_homogeneous_a100_80_on_gpu05(self):
        nodes=[dict(host='gpu-node-02',gpus=[gpu(i,host='gpu-node-02',memory=40960) for i in range(4)]),
               dict(host='gpu-node-05',gpus=[gpu(i) for i in range(10)])]
        choice=scheduling.choose_cohort(nodes,policy())
        self.assertEqual(choice['host'],'gpu-node-05');self.assertEqual(choice['max_parallelism'],3)
        self.assertEqual(len(choice['gpus']),3)

    def test_rejects_empty_busy_foreign_or_insufficient_memory(self):
        for gpus in [[],[gpu(pids=[123])],[gpu(used=500)],[gpu(util=5)],[gpu(memory=24576)]]:
            with self.subTest(gpus=gpus):
                with self.assertRaisesRegex(RuntimeError,'CPU fallback'):
                    scheduling.choose_cohort([dict(host='gpu-node-05',gpus=gpus)],policy())
        with self.assertRaises(RuntimeError):
            scheduling.choose_cohort([dict(host='unauthorized',gpus=[gpu()])],policy())

    def test_uses_free_fallback_and_does_not_mix_models(self):
        nodes=[dict(host='gpu-node-05',gpus=[gpu(pids=[99])]),dict(host='gpu-node-02',gpus=[
               gpu(0,host='gpu-node-02',memory=40960),gpu(1,host='gpu-node-02',name='A6000',memory=49152)])]
        choice=scheduling.choose_cohort(nodes,policy())
        self.assertEqual(choice['host'],'gpu-node-02');self.assertEqual(choice['max_parallelism'],1)

    def test_unknown_process_or_gpu_data_fails_closed(self):
        with self.assertRaises(ValueError):scheduling.parse_probe('bad','', 'gpu-node-05')
        with self.assertRaises(ValueError):scheduling.parse_probe('0,A100,81920,0,0,GPU-1','unknown,GPU-1','gpu-node-05')
        parsed=scheduling.parse_probe('0,A100,81920,0,0,GPU-1','123,GPU-1','gpu-node-05')
        self.assertEqual(parsed[0]['pids'],[123]);self.assertFalse(scheduling.free_gpu(parsed[0],policy()))

    def test_bounded_parallelism(self):
        nodes=[dict(host='gpu-node-05',gpus=[gpu(i) for i in range(10)])]
        for limit in [0,-1,10]:
            with self.assertRaises(ValueError):scheduling.choose_cohort(nodes,policy(),limit)
        self.assertEqual(scheduling.choose_cohort(nodes,policy(),2)['max_parallelism'],2)

    def test_gpu_is_rechecked_after_lease_acquisition(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary);cohort=dict(host='gpu-node-05',gpus=[gpu()])
            with patch.object(execution,'probe_local'):
                with self.assertRaisesRegex(RuntimeError,'became busy'):
                    execution.acquire_gpu(output,cohort,policy(),threading.Event(),
                        probe=unittest.mock.Mock(side_effect=[[gpu()],[gpu(pids=[999])]]))
            lease=common.lock_file(output/'.gpu-locks/gpu-node-05_GPU-gpu-node-05-0.lock');lease.close()

    def test_gpu_cooperative_lease_prevents_duplicate_allocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);cohort=dict(host='gpu-node-05',gpus=[gpu()])
            _,lease=execution.acquire_gpu(root,cohort,policy(),threading.Event(),probe=lambda:[gpu()])
            with self.assertRaises(RuntimeError):
                execution.acquire_gpu(root,cohort,policy(),threading.Event(),probe=lambda:[gpu()])
            lease.close()

    def test_newly_available_homogeneous_gpu_replaces_busy_preflight_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            cohort=dict(host='gpu-node-05',name=gpu()['name'],total_mib=81920,gpus=[gpu()])
            allocated,lease=execution.acquire_gpu(Path(temporary),cohort,policy(),threading.Event(),
                probe=lambda:[gpu(pids=[999]),gpu(3)])
            self.assertEqual(allocated['index'],3);lease.close()


class FrozenInputTest(unittest.TestCase):
    def test_actual_inventory_is_predetermined_readable_and_validation_selected(self):
        if not (PROJECT_DIR/'runs/pbt/windowed_pbt_v2/checkpoints/global_best_state.pt').exists():
            self.skipTest('Historical checkpoint fixtures are not installed')
        records=protocol.inventory()
        self.assertEqual([r['id'] for r in records],protocol.EXPECTED_IDS)
        self.assertTrue(all(r['cpu_strict_load_verified'] for r in records))
        self.assertEqual(records[3]['horizon'],100);self.assertEqual(records[4]['horizon'],100)
        self.assertEqual(records[4]['training_seed'],12395);self.assertEqual(records[4]['effective_seed_base'],12345)
        self.assertTrue(all('holdout' not in r['selection'] for r in records))

    def test_protocol_change_during_preflight_is_rejected(self):
        digest=sha256(common.PREPARED/'protocol.json')
        with patch.object(protocol,'sha256',side_effect=[digest,'changed']):
            with self.assertRaisesRegex(ValueError,'changed during preflight'):
                protocol.load_protocol(verify_inputs=False)

    def test_hash_verification_rejects_missing_or_modified_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'state.pt';path.write_bytes(b'model')
            record=common.file_record(path);common.verify_file(record)
            path.write_bytes(b'other')
            with self.assertRaises(ValueError):common.verify_file(record)
            path.unlink()
            with self.assertRaises(ValueError):common.verify_file(record)

    def test_new_unfrozen_holdout_file_is_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);files=[]
            for flavor in dataset.FLAVORS:
                path=root/f'ml_{flavor}_val_holdout.parquet';path.write_bytes(b'data');files.append(dict(path=str(path)))
            frozen=dict(evaluation=dict(dataset=str(root),validation_suffix='val_holdout'),dataset=dict(holdout_files=files))
            dataset.verify_evaluation_files(frozen)
            (root/'extra_bb_val_holdout.parquet').write_bytes(b'extra')
            with self.assertRaisesRegex(ValueError,'exactly the frozen'):dataset.verify_evaluation_files(frozen)

    def dataset_fixture(self, root):
        levels={k:dict(files=[]) for k in ['full','full_holdout','monitor','control']}
        for flavor in dataset.FLAVORS:
            values={k:np.arange(12,dtype=np.float32)+i for i,k in enumerate(dataset.SCALARS)}
            values['label']=np.full(12,{'bb':5,'cc':4,'dd':2}[flavor])
            source=root/(flavor+'_full.parquet');pq.write_table(pa.table(values),source)
            levels['full']['files'].append(dict(flavor=flavor,path=source.name,rows=12))
            for tier,start,stop in [('full_holdout',0,6),('monitor',6,10),('control',10,12)]:
                path=root/(flavor+'_'+tier+'.parquet');pq.write_table(pa.table({k:v[start:stop] for k,v in values.items()}),path)
                levels[tier]['files'].append(dict(flavor=flavor,path=path.name,rows=stop-start,source=source.name,
                                                  source_start_row=start,source_stop_row=stop))
        return dict(levels=levels)

    def test_split_resolution_verifies_ranges_and_actual_contents(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);manifest=self.dataset_fixture(root)
            record,values=dataset.resolve_splits(manifest,root=root)
            self.assertEqual(record['rows'],18);self.assertTrue(all(r['scalar_window_identity_verified'] for r in record['isolation']))
            manifest['levels']['monitor']['files'][0]['source_start_row']=5
            with self.assertRaisesRegex(ValueError,'Overlapping'):dataset.resolve_splits(manifest,root=root)

    def test_wrong_source_window_or_training_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);manifest=self.dataset_fixture(root)
            record,values=dataset.resolve_splits(manifest,root=root)
            source=root/'bb_full.parquet';table=pq.read_table(source).slice(1,6)
            pq.write_table(table,root/'bb_full_holdout.parquet')
            with self.assertRaisesRegex(ValueError,'content/order'):dataset.resolve_splits(manifest,root=root)
            train=root/'train.parquet';pq.write_table(table,train)
            with self.assertRaisesRegex(ValueError,'overlap'):dataset.verify_training_isolation(values,[train])

    def test_command_is_inference_only_and_has_all_required_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);frozen=synthetic_protocol(root)
            command=execution.inference_command(frozen,frozen['checkpoints'][0],root/'eval.log')
            self.assertEqual(command[command.index('--run-mode')+1],'test')
            for flag in ['--predict-output','--data-audit','--deterministic','--use-amp','--gpus','--save-prediction-row-ids']:
                self.assertIn(flag,command)
            for flag in ['--data-train','--load-epoch','--optimizer','--samples-per-epoch','--steps-per-epoch']:
                self.assertNotIn(flag,command)


class EvaluationRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.frozen=synthetic_protocol(self.root)

    def test_verified_completed_result_is_skipped_but_corrupt_incomplete_or_wrong_protocol_is_not(self):
        checkpoint=self.frozen['checkpoints'][0]
        record,_=synthetic_result(self.root,checkpoint,self.frozen)
        path=self.root/'result.json';common.write_json(path,record)
        self.assertIsNotNone(execution.verified_completed(path,checkpoint,self.frozen,'protocol'))
        for changes in [dict(status='running'),dict(protocol_sha256='other'),dict(checkpoint_id='other'),dict(artifacts=[])]:
            common.write_json(path,dict(record,**changes))
            self.assertIsNone(execution.verified_completed(path,checkpoint,self.frozen,'protocol'))
        common.write_json(path,record)
        Path(record['prediction_path']).write_bytes(b'corrupted')
        self.assertIsNone(execution.verified_completed(path,checkpoint,self.frozen,'protocol'))

    def test_invalid_optimizer_steps_or_row_order_is_rejected(self):
        checkpoint=self.frozen['checkpoints'][0];record,_=synthetic_result(self.root,checkpoint,self.frozen)
        execution.validate_result(record,checkpoint,self.frozen,'protocol')
        for changes in [dict(optimizer_steps=1),dict(consumed=dict(count=180,sequence_sha256='bad'))]:
            invalid=copy.deepcopy(record);invalid['metrics']['validation_data_audit'].update(changes)
            with self.assertRaises((ValueError,RuntimeError,KeyError)):
                execution.validate_result(invalid,checkpoint,self.frozen,'protocol')

    def test_mock_inference_success_writes_verified_atomic_result_and_inherits_locks(self):
        import shutil
        checkpoint=self.frozen['checkpoints'][0]
        template,_=synthetic_result(self.root/'template',checkpoint,self.frozen)
        process=unittest.mock.Mock(pid=999,returncode=0)
        process.poll.return_value=0
        def start(command,**kwargs):
            log=Path(command[command.index('--log-file')+1])
            log.write_text(Path(template['log']).read_text())
            shutil.copy2(template['prediction_path'],log.with_suffix('.predictions.parquet'))
            self.assertEqual(len(kwargs['pass_fds']),2)
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],gpu()['uuid'])
            return process
        cohort=dict(host='gpu-node-05',gpus=[gpu()],max_parallelism=1)
        with patch.object(execution.subprocess,'Popen',side_effect=start), \
             patch.object(execution,'git_metadata',return_value=dict(commit='synthetic')):
            record=execution.evaluate(checkpoint,self.frozen,'protocol',self.root/'run',cohort,threading.Event(),probe=lambda:[gpu()])
        self.assertEqual(record['status'],'completed',record.get('error'))
        path=self.root/'run/evaluations'/checkpoint['id']/'result.json'
        self.assertIsNotNone(execution.verified_completed(path,checkpoint,self.frozen,'protocol'))
        with patch.object(execution.subprocess,'Popen') as popen:
            execution.evaluate(checkpoint,self.frozen,'protocol',self.root/'run',cohort,threading.Event(),probe=lambda:[gpu()])
            popen.assert_not_called()

    def test_success_exit_without_predictions_is_explicitly_failed(self):
        checkpoint=self.frozen['checkpoints'][0]
        process=unittest.mock.Mock(pid=999,returncode=0);process.poll.return_value=0
        with patch.object(execution.subprocess,'Popen',return_value=process), \
             patch.object(execution,'git_metadata',return_value=dict(commit='synthetic')):
            record=execution.evaluate(checkpoint,self.frozen,'protocol',self.root/'no_artifact',
                dict(host='gpu-node-05',gpus=[gpu()]),threading.Event(),probe=lambda:[gpu()])
        self.assertEqual(record['status'],'failed');self.assertIn('Required final evaluation',record['error'])

    def test_foreign_gpu_collision_stops_only_own_worker(self):
        checkpoint=self.frozen['checkpoints'][0]
        process=unittest.mock.Mock(pid=999,returncode=None);process.poll.return_value=None
        stop=unittest.mock.Mock();stop.is_set.return_value=False;stop.wait.return_value=False
        probe=unittest.mock.Mock(side_effect=[[gpu()],[gpu()],[gpu(pids=[999,888])]])
        with patch.object(execution.subprocess,'Popen',return_value=process), \
             patch.object(execution,'git_metadata',return_value=dict(commit='synthetic')), \
             patch.object(execution,'terminate_own') as terminate:
            record=execution.evaluate(checkpoint,self.frozen,'protocol',self.root/'foreign',dict(host='gpu-node-05',gpus=[gpu()]),stop,probe=probe)
        self.assertEqual(record['status'],'failed');self.assertIn('Foreign GPU',record['error'])
        terminate.assert_called_once_with(process)

    def test_successful_mock_campaign_and_restart_validate_before_skipping(self):
        counts={};records={}
        def mock(checkpoint,frozen,digest,output,cohort,stop):
            path=Path(output)/'evaluations'/checkpoint['id']/'result.json'
            old=execution.verified_completed(path,checkpoint,frozen,digest)
            if old:return old
            counts[checkpoint['id']]=counts.get(checkpoint['id'],0)+1
            record,_=synthetic_result(Path(output)/'evaluations',checkpoint,frozen,seed=7)
            common.write_json(path,record);records[checkpoint['id']]=record
            return record
        cohort=dict(host='gpu-node-05',gpus=[gpu()],max_parallelism=2)
        output=self.root/'campaign'
        first=execution.run_campaign(self.frozen,'protocol',output,cohort,evaluator=mock)
        self.assertEqual(first['status'],'evaluated')
        second=execution.run_campaign(self.frozen,'protocol',output,cohort,evaluator=mock)
        self.assertEqual(second['status'],'evaluated');self.assertEqual(set(counts.values()),{1})
        bad=next(iter(records.values()));Path(bad['prediction_path']).write_bytes(b'bad')
        execution.run_campaign(self.frozen,'protocol',output,cohort,evaluator=mock)
        self.assertEqual(counts[bad['checkpoint_id']],2)

    def test_failed_checkpoint_does_not_block_others_or_generate_successful_report(self):
        called=[]
        def mock(checkpoint,*args):
            called.append(checkpoint['id'])
            return dict(status='failed' if checkpoint['id']=='pretrained17' else 'completed')
        finalize=unittest.mock.Mock()
        result=execution.run_campaign(self.frozen,'protocol',self.root/'failed',dict(max_parallelism=3),
                                      evaluator=mock,finalize=finalize)
        self.assertEqual(result['status'],'failed');self.assertEqual(len(called),9);finalize.assert_not_called()

    def test_interrupted_campaign_preserves_per_checkpoint_status(self):
        stop=threading.Event();stop.set()
        result=execution.run_campaign(self.frozen,'protocol',self.root/'interrupted',dict(max_parallelism=2),
            evaluator=lambda *args:dict(status='failed',error='interrupted'),stop=stop)
        self.assertEqual(result['status'],'interrupted');self.assertEqual(len(result['checkpoints']),9)

    def test_analysis_failure_cannot_mark_campaign_completed(self):
        result=execution.run_campaign(self.frozen,'protocol',self.root/'analysis_failed',dict(max_parallelism=2),
            evaluator=lambda *args:dict(status='completed'),finalize=unittest.mock.Mock(side_effect=ValueError('bad report')))
        self.assertEqual(result['status'],'failed');self.assertEqual(result['analysis_error'],'bad report')


class PairedAnalysisTest(unittest.TestCase):
    def test_report_uses_verified_synthetic_predictions_and_distinguishes_endpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);frozen=synthetic_protocol(root);results={};arrays={}
            for i,checkpoint in enumerate(frozen['checkpoints']):
                results[checkpoint['id']],arrays[checkpoint['id']]=synthetic_result(root,checkpoint,frozen,seed=i+1)
            report=analysis.build_report(frozen,results,root/'report','protocol',arrays=arrays)
            text=Path(report['path']).read_text()
            for word in ['Final-10','Global-best','Holdout','event-level','Два training seeds']:
                self.assertIn(word,text)
            data=json.loads((root/'report/analysis.json').read_text())
            self.assertEqual(len(data['rows']),9);self.assertEqual(len(data['matched_comparisons']),5)
            self.assertTrue(all(c['confidence']['status']=='unavailable' for c in data['matched_comparisons']))
            self.assertTrue((root/'report/comparison.csv').exists());self.assertEqual(len(report['artifacts']),5)

    def test_paired_event_bootstrap_uses_same_draws_and_preserves_grouped_events(self):
        labels=np.repeat(np.arange(3),60);rng=np.random.default_rng(5)
        scores=rng.uniform(.01,1,(180,3)).astype('<f4');scores/=scores.sum(axis=1,keepdims=True)
        result=analysis.paired_bootstrap(labels,scores,scores,physical_event_ids=np.repeat(np.arange(90),2),replicates=20,seed=7)
        self.assertEqual(result['status'],'available');self.assertEqual(result['n_events'],90)
        self.assertTrue(all(bounds==[0.,0.] for bounds in result['intervals'].values()))
        self.assertEqual(analysis.paired_bootstrap(labels,scores,scores)['status'],'unavailable')

    def test_prediction_reordering_or_metric_mismatch_blocks_analysis(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);frozen=synthetic_protocol(root);records={};arrays={}
            for checkpoint in frozen['checkpoints']:
                records[checkpoint['id']],arrays[checkpoint['id']]=synthetic_result(root,checkpoint,frozen)
            ident=protocol.EXPECTED_IDS[-1];ids,labels,scores=arrays[ident];arrays[ident]=(ids[::-1],labels,scores)
            with self.assertRaisesRegex(ValueError,'same ordered'):analysis.summarize(frozen,records,arrays=arrays)
            arrays[ident]=(ids,labels,scores);records[ident]['metrics'][common.PRIMARY]+=1
            with self.assertRaisesRegex(ValueError,'disagree'):analysis.summarize(frozen,records,arrays=arrays)

    def test_valid_different_delivery_orders_are_paired_by_actual_row_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);frozen=synthetic_protocol(root);records={};arrays={}
            for checkpoint in frozen['checkpoints']:
                records[checkpoint['id']],arrays[checkpoint['id']]=synthetic_result(root,checkpoint,frozen)
            ident=protocol.EXPECTED_IDS[-1];ids,labels,scores=arrays[ident]
            arrays[ident]=(ids[::-1],labels[::-1],scores[::-1])
            data=analysis.summarize(frozen,records,arrays=arrays)
            self.assertTrue(all(c['delta_pp']==0 for c in data['matched_comparisons']))


class LaunchSafetyTest(unittest.TestCase):
    def test_complete_dry_run_never_starts_tmux_or_subprocess_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);frozen=synthetic_protocol(root);prepared=root/'prepared';prepared.mkdir()
            nodes=[dict(host='gpu-node-05',gpus=[gpu(i) for i in range(5)])]
            with patch('experiments.stage2a.protocol.load_protocol',return_value=(frozen,'protocol')), \
                 patch.object(launcher,'discover',return_value=nodes),patch.object(launcher,'launch_tmux') as tmux:
                with redirect_stdout(io.StringIO()):
                    code=launcher.main(['--dry-run','--prepared',str(prepared),'--output',str(root/'output')])
            self.assertEqual(code,0);tmux.assert_not_called()
            plan=json.loads((prepared/'dry_run.json').read_text())
            self.assertFalse(plan['inference_launched']);self.assertFalse(plan['gpus_reserved']);self.assertEqual(len(plan['jobs']),9)

    def test_busy_cluster_has_clear_exit_and_no_cpu_or_tmux_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);frozen=synthetic_protocol(root)
            with patch('experiments.stage2a.protocol.load_protocol',return_value=(frozen,'protocol')), \
                 patch.object(launcher,'discover',return_value=[dict(host='gpu-node-05',gpus=[gpu(pids=[999])])]), \
                 patch.object(launcher,'launch_tmux') as tmux:
                code=launcher.main(['--dry-run','--prepared',str(root),'--output',str(root/'output')])
            self.assertEqual(code,69);tmux.assert_not_called()
            self.assertTrue((root/'output/resource_failure.json').exists())

    def test_remote_command_quotes_paths_and_uses_persistent_tmux(self):
        with patch.object(launcher.subprocess,'run') as run:
            launcher.launch_tmux('gpu-node-05','stage2a-test',Path('/tmp/space path'),Path('/tmp/prepared'),{})
        command=run.call_args_list[-1].args[0][-1]
        self.assertIn('tmux new-session -d',command);self.assertIn('--on-node',command)
        self.assertIn('supervisor.log',command);self.assertNotIn('nohup',command)

    def test_verified_completed_campaign_does_not_require_free_gpus(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);frozen=synthetic_protocol(root)
            with patch('experiments.stage2a.protocol.load_protocol',return_value=(frozen,'protocol')), \
                 patch.object(launcher,'completed_output',return_value=True),patch.object(launcher,'discover') as discover, \
                 patch.object(launcher,'launch_tmux') as tmux,redirect_stdout(io.StringIO()):
                code=launcher.main(['--prepared',str(root),'--output',str(root/'output')])
            self.assertEqual(code,0);discover.assert_not_called();tmux.assert_not_called()


if __name__=='__main__':
    unittest.main()
