from pathlib import Path

import yaml

from .linucb import LinUCBLearningRateController
from weaver.utils.logger import _logger


def build_training_controller(config_path, optimizer, *, default_log_path=None, steps_per_epoch=None):
    """Build a controller from a small versionable YAML configuration."""

    if config_path is None:
        return None

    path = Path(config_path)
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError("training controller config must be a YAML mapping")

    controller_type = config.pop("type", None)
    if controller_type != "linucb_lr":
        raise ValueError(f"unsupported training controller type: {controller_type!r}")
    for fraction_key, steps_key in (
        ("interval_fraction", "interval_steps"),
        ("warmup_fraction", "warmup_steps"),
    ):
        fraction = config.pop(fraction_key, None)
        if fraction is None:
            continue
        if steps_key in config:
            raise ValueError(f"use either {fraction_key} or {steps_key}, not both")
        if steps_per_epoch is None:
            raise ValueError(f"{fraction_key} requires a finite steps_per_epoch")
        if fraction_key == "interval_fraction" and not 0 < fraction <= 1:
            raise ValueError("interval_fraction must be in (0, 1]")
        if fraction_key == "warmup_fraction" and not 0 <= fraction <= 1:
            raise ValueError("warmup_fraction must be in [0, 1]")
        config[steps_key] = max(1 if steps_key == "interval_steps" else 0, round(fraction * steps_per_epoch))
    if config.get("log_path") is None:
        config["log_path"] = default_log_path
    _logger.info("[training-control] loading configuration: %s", path)
    return LinUCBLearningRateController(optimizer, **config)
