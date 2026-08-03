import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from validation.proxy_subsets import build_proxy_subsets
from validation.tail_proxy_subsets import build_tail_proxy_subsets


def write_val_file(path, label, rows):
    table = pa.table({
        "label": [label] * rows,
        "event_id": list(range(rows)),
        "jet_e": [float(index) for index in range(rows)],
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


class ProxySubsetBuilderTest(unittest.TestCase):
    def test_build_proxy_subsets_writes_disjoint_control_and_monitor_files(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            for flavor, label in (("bb", 5), ("cc", 4), ("dd", 1)):
                write_val_file(dataset / f"toy_{flavor}_val50k.parquet", label, 12)

            manifest = build_proxy_subsets(
                dataset=dataset,
                output_root=root / "proxy",
                manifest_output=root / "manifest.json",
                name="toy_proxy",
                seed=7,
                control_rows_per_class=3,
                monitor_rows_per_class=4,
                compression="snappy",
            )

            self.assertEqual(manifest["levels"]["control"]["rows_total"], 9)
            self.assertEqual(manifest["levels"]["monitor"]["rows_total"], 12)
            self.assertEqual(manifest["levels"]["full"]["rows_total"], 36)

            for flavor in ("bb", "cc", "dd"):
                control_path = root / "proxy/toy_proxy/control" / f"toy_{flavor}_val50k.parquet"
                monitor_path = root / "proxy/toy_proxy/monitor" / f"toy_{flavor}_val50k.parquet"
                control_ids = set(pq.read_table(control_path, columns=["event_id"]).column("event_id").to_pylist())
                monitor_ids = set(pq.read_table(monitor_path, columns=["event_id"]).column("event_id").to_pylist())
                self.assertEqual(len(control_ids), 3)
                self.assertEqual(len(monitor_ids), 4)
                self.assertTrue(control_ids.isdisjoint(monitor_ids))

    def test_build_proxy_subsets_rejects_oversized_proxy_request(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            for flavor, label in (("bb", 5), ("cc", 4), ("dd", 1)):
                write_val_file(dataset / f"toy_{flavor}_val50k.parquet", label, 5)

            with self.assertRaisesRegex(ValueError, "exceed available rows"):
                build_proxy_subsets(
                    dataset=dataset,
                    output_root=root / "proxy",
                    manifest_output=root / "manifest.json",
                    name="toy_proxy",
                    control_rows_per_class=3,
                    monitor_rows_per_class=3,
                )


class TailProxySubsetBuilderTest(unittest.TestCase):
    def test_build_tail_proxy_subsets_uses_disjoint_tail_windows(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            for flavor, label in (("bb", 5), ("cc", 4), ("dd", 1)):
                write_val_file(dataset / f"toy_{flavor}_val1000k.parquet", label, 20)

            manifest = build_tail_proxy_subsets(
                dataset=dataset,
                manifest_output=root / "tail_manifest.json",
                name="toy_tail_proxy",
                control_rows_per_class=3,
                monitor_rows_per_class=5,
                compression="snappy",
            )

            self.assertEqual(manifest["levels"]["control"]["rows_total"], 9)
            self.assertEqual(manifest["levels"]["monitor"]["rows_total"], 15)
            for flavor in ("bb", "cc", "dd"):
                control_path = dataset / f"toy_{flavor}_val5k_tail.parquet"
                monitor_path = dataset / f"toy_{flavor}_val50k_tail.parquet"
                control_ids = pq.read_table(control_path, columns=["event_id"]).column("event_id").to_pylist()
                monitor_ids = pq.read_table(monitor_path, columns=["event_id"]).column("event_id").to_pylist()
                self.assertEqual(control_ids, [17, 18, 19])
                self.assertEqual(monitor_ids, [12, 13, 14, 15, 16])
                self.assertTrue(set(control_ids).isdisjoint(monitor_ids))

    def test_build_tail_proxy_subsets_builds_disjoint_full_holdout_by_default(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            for flavor, label in (("bb", 5), ("cc", 4), ("dd", 1)):
                write_val_file(dataset / f"toy_{flavor}_val1000k.parquet", label, 20)

            manifest = build_tail_proxy_subsets(
                dataset=dataset,
                manifest_output=root / "tail_manifest.json",
                name="toy_tail_proxy",
                control_rows_per_class=3,
                monitor_rows_per_class=5,
                compression="snappy",
            )

            self.assertIn("full_holdout", manifest["levels"])
            self.assertEqual(manifest["levels"]["full_holdout"]["rows_total"], 36)  # (20-3-5)*3 flavors
            for flavor in ("bb", "cc", "dd"):
                control_path = dataset / f"toy_{flavor}_val5k_tail.parquet"
                monitor_path = dataset / f"toy_{flavor}_val50k_tail.parquet"
                holdout_path = dataset / f"toy_{flavor}_val_holdout.parquet"
                control_ids = set(pq.read_table(control_path, columns=["event_id"]).column("event_id").to_pylist())
                monitor_ids = set(pq.read_table(monitor_path, columns=["event_id"]).column("event_id").to_pylist())
                holdout_ids = set(pq.read_table(holdout_path, columns=["event_id"]).column("event_id").to_pylist())
                self.assertEqual(holdout_ids, set(range(12)))  # rows [0, 20-3-5) = [0, 12)
                self.assertTrue(holdout_ids.isdisjoint(control_ids))
                self.assertTrue(holdout_ids.isdisjoint(monitor_ids))
                self.assertEqual(holdout_ids | monitor_ids | control_ids, set(range(20)))

    def test_build_tail_proxy_subsets_can_skip_full_holdout(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            for flavor, label in (("bb", 5), ("cc", 4), ("dd", 1)):
                write_val_file(dataset / f"toy_{flavor}_val1000k.parquet", label, 20)

            manifest = build_tail_proxy_subsets(
                dataset=dataset,
                control_rows_per_class=3,
                monitor_rows_per_class=5,
                compression="snappy",
                build_full_holdout=False,
            )

            self.assertNotIn("full_holdout", manifest["levels"])

    def test_build_tail_proxy_subsets_rejects_oversized_tail_request(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            for flavor, label in (("bb", 5), ("cc", 4), ("dd", 1)):
                write_val_file(dataset / f"toy_{flavor}_val1000k.parquet", label, 5)

            with self.assertRaisesRegex(ValueError, "exceed source rows"):
                build_tail_proxy_subsets(
                    dataset=dataset,
                    manifest_output=root / "tail_manifest.json",
                    control_rows_per_class=3,
                    monitor_rows_per_class=3,
                )


if __name__ == "__main__":
    unittest.main()
