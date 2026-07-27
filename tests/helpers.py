import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from training.pbt import config  # noqa: E402


def pbt_smoke_config():
    return config.load_config(
        SimpleNamespace(
            config=PROJECT_DIR / "configs/experiments/pbt_smoke.yaml",
            experiment_name="unit_test",
            gpus="0,2",
            slots=None,
            smoke=True,
        )
    )


def namespace(**kwargs):
    return SimpleNamespace(**kwargs)
