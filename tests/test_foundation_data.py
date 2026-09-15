import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from weaver.utils.dataset import SimpleIterDataset
from weaver.utils.data_audit import ID_KEY, ConsumptionAudit
from weaver.train import evaluation_dataset


class FoundationDataTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.files = []
        for c in range(3):
            path = self.root / f"class{c}.parquet"
            pq.write_table(pa.table(dict(x=np.arange(103, dtype=np.float32), label=[c]*103,
                                         weight=[0.9]*103)), path, row_group_size=17)
            self.files.append(str(path))
        config = dict(preprocess=dict(method="manual"), inputs=dict(features=dict(vars=[["x", 0, 1]], length=None)),
                      labels=dict(type="simple", value=["label == 0", "label == 1", "label == 2"]),
                      weights=dict(use_precomputed_weights=True, weight_branches=["weight"]))
        self.config = self.root / "data.yaml"
        self.config.write_text(yaml.safe_dump(config))

    def consume(self, training, seed, epoch, groups=None):
        prefix = str(self.root / f"{training}_{seed}_{epoch}_{len(list(self.root.glob('*.json')))}")
        ds = SimpleIterDataset(groups or {"_": self.files}, str(self.config), batch_size=32,
                               for_training=training, seed=seed, fetch_step=.01,
                               audit_prefix=prefix, infinity_mode=False)
        ds.set_epoch(epoch)
        loader = torch.utils.data.DataLoader(ds, batch_size=32, num_workers=0, drop_last=False)
        audit = ConsumptionAudit(loader, SimpleNamespace(log=prefix), "train" if training else "validation", epoch)
        ids = []
        for _, _, z in loader:
            audit.add(z)
            ids.extend(z[ID_KEY].tolist())
        record = audit.finish(batches=(len(ids)+31)//32)
        return ids, record

    def test_validation_once_even_with_weights_seed_and_group_changes(self):
        first, a = self.consume(False, 1, 0)
        second, b = self.consume(False, 999, 1, {"z": self.files[2:], "a": self.files[:2]})
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), list(range(309)))
        self.assertEqual(a["dataset"]["fingerprint"], b["dataset"]["fingerprint"])
        self.assertEqual(a["consumed"]["repeated_ids"], 0)
        self.assertTrue(a["exhausted"])
        self.assertEqual(a["traversal"][0]["scanned"]["count"], 309)
        self.assertEqual(a["traversal"][0]["wraps"], 0)

    def test_training_full_traversal_sampling_reproducible_with_owned_rng(self):
        first, a = self.consume(True, 7, 1)
        np.random.seed(999)
        second, b = self.consume(True, 7, 1)
        third, c = self.consume(True, 7, 2)
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)
        for record in (a, b, c):
            self.assertEqual(record["traversal"][0]["scanned"]["unique_ids"], 309)
            self.assertEqual(record["traversal"][0]["scanned"]["repeated_ids"], 0)
            self.assertEqual(record["traversal"][0]["wraps"], 0)
            self.assertEqual(record["consumed"]["count"], record["traversal"][0]["accepted"]["count"])

    def test_shared_evaluation_factory_ignores_training_sampling(self):
        args = SimpleNamespace(fetch_step_val=.1, log=str(self.root / "eval"), data_audit=True)
        ds = evaluation_dataset(args, self.files, str(self.config), None, "val")
        self.assertFalse(ds._infinity_mode)
        self.assertFalse(ds._sampler_options["shuffle"])
        self.assertFalse(ds._sampler_options["reweight"])
