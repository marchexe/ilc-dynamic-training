"""Content-addressed Parquet row identities and consumed-data accounting.

IDs are offsets in the sorted, content-fingerprinted file manifest, not physics
event numbers. They identify input rows even when labels/observables coincide.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ID_KEY = "__weaver_row_id__"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parquet_manifest(files):
    import pyarrow.parquet as pq
    records, offset = [], 0
    for path in sorted(set(map(str, files))):
        if Path(path).suffix != ".parquet":
            raise ValueError("Audited row identities currently require Parquet inputs")
        rows = pq.ParquetFile(path).metadata.num_rows
        records.append(dict(path=path, rows=rows, offset=offset, sha256=file_sha256(path)))
        offset += rows
    # Paths/order are part of identity; same content in a different manifest
    # must not silently masquerade as the same data sequence.
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return dict(files=records, total_rows=offset, fingerprint=fingerprint)


class RowAccumulator:
    def __init__(self, total_rows):
        self.seen = np.zeros(total_rows, dtype=np.bool_)
        self.digest = hashlib.sha256()
        self.count = 0

    def add(self, ids):
        ids = np.asarray(ids, dtype="<i8").reshape(-1)
        if len(ids) and (ids.min() < 0 or ids.max() >= len(self.seen)):
            raise ValueError("Row ID outside declared dataset")
        self.digest.update(ids.tobytes())
        self.seen[ids] = True
        self.count += len(ids)

    def summary(self):
        unique = int(self.seen.sum())
        return dict(count=self.count, unique_ids=unique, repeated_ids=self.count - unique,
                    sequence_sha256=self.digest.hexdigest(),
                    id_set_sha256=hashlib.sha256(np.flatnonzero(self.seen).astype("<i8").tobytes()).hexdigest())


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class ConsumptionAudit:
    def __init__(self, loader, args, kind, epoch):
        self.manifest = getattr(loader.dataset, "audit_manifest", None)
        self.kind, self.epoch = kind, epoch
        self.prefix = getattr(args, "log", "")
        self.loader = loader
        self.rows = RowAccumulator(self.manifest["total_rows"]) if self.manifest else None
        if self.rows is not None:
            prefix = loader.dataset._audit_prefix
            for worker in range(max(1, loader.num_workers)):
                Path(f"{prefix}.{loader.dataset._epoch}.worker{worker}.json").unlink(missing_ok=True)

    def add(self, observers):
        if self.rows is not None:
            ids = observers[ID_KEY]
            self.rows.add(ids.numpy() if hasattr(ids, "numpy") else ids)

    def finish(self, *, batches, optimizer_steps=0, loss=None, accuracy=None, scores=None):
        if self.rows is None:
            return None
        payload = dict(schema_version=1, kind=self.kind, epoch=self.epoch,
                       dataset=self.manifest, consumed=self.rows.summary(), batches=batches,
                       optimizer_steps=optimizer_steps, loss=loss, accuracy=accuracy)
        if scores is not None:
            payload["prediction_sha256"] = hashlib.sha256(np.asarray(scores, dtype="<f4").tobytes()).hexdigest()
        # These worker reports are emitted only after natural exhaustion.
        prefix = getattr(self.loader.dataset, "_audit_prefix", None)
        loader_epoch = getattr(self.loader.dataset, "_epoch", 0)
        workers = max(1, self.loader.num_workers)
        reports = []
        for worker in range(workers):
            path = Path(f"{prefix}.{loader_epoch}.worker{worker}.json")
            if prefix and path.exists():
                reports.append(json.loads(path.read_text()))
        payload["traversal"] = reports
        payload["exhausted"] = len(reports) == workers
        from .logger import _logger
        _logger.info("Data audit %s: %s", self.kind, json.dumps(payload, sort_keys=True))
        if self.prefix:
            atomic_json(f"{self.prefix}.{self.kind}.{self.epoch}.audit.json", payload)
        return payload
