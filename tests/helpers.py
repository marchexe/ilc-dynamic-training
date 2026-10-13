import os
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
# Tests are hermetic: never read a developer's private site configuration.
os.environ["ILC_SITE_CONFIG"] = str(PROJECT_DIR / "configs/site.example.json")

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


def synthetic_config(directory, overrides, *, base="configs/experiments/pbt_smoke.yaml", **load_args):
    """Load a temporary configuration that inherits ``base`` and applies ``overrides``.

    Mappings merge into the base; lists such as ``population`` replace it.
    """
    path = Path(directory) / "synthetic.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 1, "presets": [str(PROJECT_DIR / base)], **overrides}))
    arguments = {"experiment_name": "unit_synthetic", "gpus": "0,1,2,3", "slots": None, "smoke": False, **load_args}
    return config.load_config(SimpleNamespace(config=path, **arguments))
