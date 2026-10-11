import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from tests.helpers import PROJECT_DIR
from experiments.stage2a.execution import prediction_arrays
from weaver.utils.dataset import SimpleIterDataset
from weaver.utils.nn.tools import evaluate_classification as evaluate
from weaver.train import save_parquet
from weaver.utils.data_audit import ID_KEY


class PredictionRowIDsTest(unittest.TestCase):
    def test_opt_in_preserves_actual_interleaved_row_ids_without_changing_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);files=[]
            for label in range(3):
                path=root/f'class{label}.parquet'
                pq.write_table(pa.table(dict(x=np.arange(17,dtype=np.float32),label=[label]*17)),path)
                files.append(str(path))
            config=root/'config.yaml'
            config.write_text(yaml.safe_dump(dict(preprocess=dict(method='manual'),
                inputs=dict(features=dict(vars=[['x',0,1]],length=None)),
                labels=dict(type='simple',value=['label == 0','label == 1','label == 2']))))
            class Tiny(torch.nn.Module):
                def forward(self,x):
                    v=x.reshape(-1)*.01
                    return torch.stack([-v,torch.zeros_like(v),v],dim=1)
            outputs=[]
            for enabled in (False,True):
                prefix=str(root/f'eval-{enabled}')
                ds=SimpleIterDataset({'_':files},str(config),for_training=False,fetch_step=.1,
                                     audit_prefix=prefix,infinity_mode=False)
                loader=torch.utils.data.DataLoader(ds,batch_size=8,num_workers=0,drop_last=False)
                args=SimpleNamespace(log=prefix,use_amp=False,amp_dtype='fp16',save_prediction_row_ids=enabled)
                _,scores,labels,observers=evaluate(Tiny(),loader,torch.device('cpu'),epoch=None,for_training=False,
                    loss_func=torch.nn.CrossEntropyLoss(),extra_args={'args':args})
                outputs.append((scores,labels))
                if not enabled:
                    self.assertNotIn(ID_KEY,observers)
                    continue
                ids=observers[ID_KEY]
                self.assertEqual(sorted(ids.tolist()),list(range(51)))
                self.assertFalse(np.array_equal(ids,np.arange(51)))
                audit=json.loads(Path(prefix+'.validation.None.audit.json').read_text())
                self.assertEqual(hashlib.sha256(np.asarray(ids,dtype='<i8').tobytes()).hexdigest(),
                                 audit['consumed']['sequence_sha256'])
                prediction=root/'predictions.parquet'
                save_parquet(args,str(prediction),scores,labels,observers)
                self.assertTrue(np.array_equal(ak.to_numpy(ak.from_parquet(prediction)[ID_KEY]),ids))
                verified_ids,verified_labels,verified_scores=prediction_arrays(
                    dict(prediction_path=str(prediction),metrics=dict(validation_data_audit=audit)),
                    dict(dataset=dict(rows=51,audit_fingerprint=audit['dataset']['fingerprint'],
                                      holdout_files=[dict(rows=17)]*3)))
                self.assertTrue(np.array_equal(verified_ids,ids))
                self.assertTrue(np.array_equal(verified_scores,scores))
            self.assertTrue(np.array_equal(outputs[0][0],outputs[1][0]))
            self.assertTrue(np.array_equal(outputs[0][1]['_label_'],outputs[1][1]['_label_']))
