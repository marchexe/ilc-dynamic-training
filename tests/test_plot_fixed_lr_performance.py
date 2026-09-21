import json
import struct
import tempfile
import unittest
from pathlib import Path

from tests.helpers import SCRIPTS_DIR  # noqa: F401
from reports.plot_fixed_lr_performance import fixed_lr_data, plot_runs


def manifest(member_values, *, initial_epoch=17):
    members = {
        member: {"name": member, "lr": lr, "parent": None}
        for member, (lr, _values) in member_values.items()
    }
    generations = []
    for index in range(len(next(iter(member_values.values()))[1])):
        generations.append({
            "index": index,
            "epoch": initial_epoch + index + 1,
            "status": "completed",
            "workers": {
                member: {"lr": lr, "metrics": {"metric": values[index]}}
                for member, (lr, values) in member_values.items()
            },
        })
    return {
        "status": "completed",
        "method": "fixed_lr_grid",
        "members": members,
        "config": {"pbt": {"metric": "metric"}, "shared": {"initial_epoch": initial_epoch}},
        "generations": generations,
    }


class FixedLrPerformanceTest(unittest.TestCase):
    def write_manifest(self, directory, data):
        directory.mkdir()
        (directory / "manifest.json").write_text(json.dumps(data), encoding="utf-8")

    def test_stitches_matching_prefix_members_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefix = root / "prefix"
            run_a = root / "run_a"
            run_b = root / "run_b"
            self.write_manifest(prefix, manifest({
                "lr_a": (1e-5, [0.5] * 10),
                "lr_b": (2e-5, [0.6] * 10),
                "not_continued": (3e-5, [0.7] * 10),
            }))
            self.write_manifest(run_a, manifest({"lr_a": (1e-5, [0.4] * 10)}, initial_epoch=27))
            self.write_manifest(run_b, manifest({"lr_b": (2e-5, [0.45] * 10)}, initial_epoch=27))

            data = fixed_lr_data([run_a, run_b], [prefix])

            self.assertEqual(set(data["curves"]), {"lr_a", "lr_b"})
            self.assertEqual(data["epochs"], list(range(1, 21)))
            self.assertEqual(data["boundaries"], [10])
            self.assertEqual(data["curves"]["lr_a"], [0.5] * 10 + [0.4] * 10)

    def test_writes_fixed_size_png(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            self.write_manifest(run, manifest({
                "lr_a": (1e-5, [0.5 - index * 0.01 for index in range(10)]),
                "lr_b": (2e-5, [0.6 - index * 0.01 for index in range(10)]),
            }))

            output = plot_runs([run])

            self.assertEqual(output, run / "plots" / "fixed_lr_performance_comparison.png")
            with output.open("rb") as stream:
                stream.seek(16)
                self.assertEqual(struct.unpack(">II", stream.read(8)), (3960, 2250))


if __name__ == "__main__":
    unittest.main()
