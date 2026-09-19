import csv
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from training.runtime import _count_mistag_uncertainty, sha256
from validation.qualify_proxy import checkpoint_shard
from validation.proxy_qualification import (
    build_proxy_sets,
    deduplicate_checkpoint_entries,
    summarize_candidate,
    write_results_csv,
)


class ProxyQualificationTest(unittest.TestCase):
    def make_fixture(self, root):
        dataset = root / "dataset"
        dataset.mkdir()
        scores, labels = [], []
        for class_index, flavor in enumerate(("bb", "cc", "dd")):
            table = pa.table(
                {
                    "event_id": [1000 * class_index + index for index in range(12)],
                    "feature": [float(index) for index in range(12)],
                }
            )
            pq.write_table(table, dataset / f"toy_{flavor}_val50k_tail.parquet")
            for index in range(12):
                # Entropy increases with index, making hard membership easy
                # to distinguish from the deterministic random sample.
                confidence = 0.98 - index * 0.04
                row = [(1.0 - confidence) / 2] * 3
                row[class_index] = confidence
                scores.append(row)
                labels.append(class_index)
        predictions = root / "anchor_predictions.parquet"
        pq.write_table(pa.table({"scores": scores, "_label_": labels}), predictions)
        checkpoint = root / "anchor.pt"
        checkpoint.write_bytes(b"fixed anchor state")
        return dataset, predictions, checkpoint

    def build(self, root, output_name):
        dataset, predictions, checkpoint = self.make_fixture(root)
        manifest = build_proxy_sets(
            dataset=dataset,
            source_suffix="val50k_tail",
            output_root=root / output_name,
            manifest_output=root / f"{output_name}.json",
            anchor_checkpoint=checkpoint,
            anchor_predictions=predictions,
            sizes=(9, 18),
            seed=17,
            compression="snappy",
        )
        return dataset, manifest

    def test_deterministic_balanced_frozen_membership(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            first_root = Path(temporary) / "first"
            second_root = Path(temporary) / "second"
            first_root.mkdir()
            second_root.mkdir()
            _, first = self.build(first_root, "proxy")
            _, second = self.build(second_root, "proxy")
            self.assertEqual(len(first["candidates"]), 6)
            for left, right in zip(first["candidates"], second["candidates"]):
                self.assertEqual(left["class_counts"], {"bb": left["size"] // 3, "cc": left["size"] // 3, "dd": left["size"] // 3})
                left_membership = json.loads((first_root / left["membership_manifest"]).read_text())
                right_membership = json.loads((second_root / right["membership_manifest"]).read_text())
                self.assertEqual(left_membership["events_by_class"], right_membership["events_by_class"])
                self.assertEqual(sha256(first_root / left["membership_manifest"]), left["proxy_manifest_sha256"])

    def test_existing_proxy_is_frozen_without_force(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            dataset, predictions, checkpoint = self.make_fixture(root)
            kwargs = dict(
                dataset=dataset,
                source_suffix="val50k_tail",
                output_root=root / "proxy",
                manifest_output=root / "proxy_sets.json",
                anchor_checkpoint=checkpoint,
                anchor_predictions=predictions,
                sizes=(9,),
            )
            build_proxy_sets(**kwargs)
            with self.assertRaises(FileExistsError):
                build_proxy_sets(**kwargs)

    def test_source_dataset_is_not_mutated(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            dataset, predictions, checkpoint = self.make_fixture(root)
            before = {path: sha256(path) for path in dataset.glob("*.parquet")}
            build_proxy_sets(
                dataset=dataset,
                source_suffix="val50k_tail",
                output_root=root / "proxy",
                manifest_output=root / "proxy_sets.json",
                anchor_checkpoint=checkpoint,
                anchor_predictions=predictions,
                sizes=(9,),
            )
            self.assertEqual(before, {path: sha256(path) for path in dataset.glob("*.parquet")})

    def test_checkpoint_hash_deduplication(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            one, two, three = root / "one.pt", root / "two.pt", root / "three.pt"
            one.write_bytes(b"same")
            two.write_bytes(b"same")
            three.write_bytes(b"different")
            distinct, duplicates = deduplicate_checkpoint_entries(
                [{"id": "one", "canonical_source_path": str(one)}, {"id": "two", "canonical_source_path": str(two)}, {"id": "three", "canonical_source_path": str(three)}]
            )
            self.assertEqual([item["id"] for item in distinct], ["one", "three"])
            self.assertEqual(duplicates[0]["duplicate_of"], "one")

    def test_checkpoint_shards_are_disjoint_and_complete(self):
        checkpoints = [{"id": str(index)} for index in range(11)]
        shards = [checkpoint_shard(checkpoints, 4, index) for index in range(4)]
        self.assertEqual(
            [[item["id"] for item in shard] for shard in shards],
            [["0", "4", "8"], ["1", "5", "9"], ["2", "6", "10"], ["3", "7"]],
        )
        self.assertEqual(
            sorted(item["id"] for shard in shards for item in shard),
            sorted(item["id"] for item in checkpoints),
        )

    def test_result_serialization_and_statistics(self):
        rows = [
            {"checkpoint_id": "a", "reference_metric": 1.0, "proxy_metric": 1.1, "runtime_seconds": 2.0},
            {"checkpoint_id": "b", "reference_metric": 2.0, "proxy_metric": 1.9, "runtime_seconds": 2.5},
            {"checkpoint_id": "c", "reference_metric": 3.0, "proxy_metric": 3.2, "runtime_seconds": 3.0},
        ]
        summary = summarize_candidate(rows)
        self.assertAlmostEqual(summary["spearman"], 1.0)
        self.assertTrue(summary["best_checkpoint"]["agrees"])
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            output = Path(temporary) / "results.csv"
            write_results_csv(output, rows)
            with output.open(newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 3)

    def test_zero_count_uncertainty_is_not_zero(self):
        counts = {"bc": [{"signal_efficiency": 0.8, "background_passed": 0, "background_total": 5}]}
        uncertainty = _count_mistag_uncertainty(counts, "bc", 0.8)
        self.assertIsNotNone(uncertainty)
        self.assertGreater(uncertainty, 0.0)


if __name__ == "__main__":
    unittest.main()
