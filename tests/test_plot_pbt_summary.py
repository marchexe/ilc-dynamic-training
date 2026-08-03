import unittest

from tests.helpers import SCRIPTS_DIR  # noqa: F401

from reports import plot_pbt_summary


class PlotPBTSummaryTest(unittest.TestCase):
    def test_worker_lr_prefers_explicit_manifest_lr(self):
        self.assertEqual(
            plot_pbt_summary.worker_lr(
                {"lr": 0.03, "command": ["weaver", "--start-lr", "0.1"]}
            ),
            0.03,
        )

    def test_worker_lr_reads_legacy_local_command_list(self):
        self.assertEqual(
            plot_pbt_summary.worker_lr({"command": ["weaver", "--start-lr", "0.1"]}),
            0.1,
        )

    def test_worker_lr_reads_legacy_ssh_wrapped_command_string(self):
        worker = {
            "command": [
                "ssh",
                "iutgpu01",
                "cd /work && exec /work/.venv/bin/python /work/.venv/bin/weaver "
                "--run-mode train,val --start-lr 6.571428571428571e-06 --gpus 6",
            ]
        }

        self.assertEqual(plot_pbt_summary.worker_lr(worker), 6.571428571428571e-06)


if __name__ == "__main__":
    unittest.main()
