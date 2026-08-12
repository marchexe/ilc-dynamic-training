#!/usr/bin/env python3
"""PBT configuration loading and validation."""

import hashlib
import json
from pathlib import Path

import yaml

from training.pbt.models.config import PBTYamlConfig, ResolvedPBTConfig
from training.runtime import (
    PROJECT_DIR,
    normalize_data_extension,
    project_path,
    validate_dataset,
)


def deep_merge(base, override):
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_preset_path(raw_path, parent):
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidate = parent / path
    if candidate.exists():
        return candidate.resolve()
    return project_path(path).resolve()


def load_yaml_with_presets(config_path, seen=None):
    config_path = config_path.resolve()
    seen = set() if seen is None else set(seen)
    if config_path in seen:
        chain = " -> ".join(str(path) for path in [*seen, config_path])
        raise ValueError(f"Preset cycle detected: {chain}")
    seen.add(config_path)

    payload = yaml.safe_load(config_path.read_text())
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {config_path}")

    preset_paths = payload.get("presets") or []
    if isinstance(preset_paths, (str, Path)):
        preset_paths = [str(preset_paths)]
    if not isinstance(preset_paths, list) or any(
        not isinstance(item, str) for item in preset_paths
    ):
        raise ValueError("presets must be a string or a list of strings")

    merged = {}
    for raw_preset in preset_paths:
        preset_path = resolve_preset_path(raw_preset, config_path.parent)
        merged = deep_merge(merged, load_yaml_with_presets(preset_path, seen))

    local = dict(payload)
    local.pop("presets", None)
    merged = deep_merge(merged, local)
    seen.remove(config_path)
    return merged


def parse_slots(args, resources):
    if args.gpus and args.slots:
        raise ValueError("Use either --gpus or --slots, not both")

    if args.slots:
        raw_slots = [slot.strip() for slot in args.slots.split(",")]
        slots = []
        for raw in raw_slots:
            if not raw:
                raise ValueError("--slots contains an empty slot")
            if "@" in raw:
                raise ValueError("Per-host virtualenv overrides are no longer supported; use the project .venv")
            if ":" not in raw:
                raise ValueError(f"Expected host:gpu slot, got: {raw}")
            host, gpu = raw.rsplit(":", 1)
            host = host.strip()
            gpu = gpu.strip()
            if not host or not gpu:
                raise ValueError(f"Expected host:gpu slot, got: {raw}")
            slots.append({"host": host, "gpu": gpu, "label": f"{host}:{gpu}"})
    else:
        raw_gpus = args.gpus.split(",") if args.gpus else [str(gpu) for gpu in resources.get("gpus", [])]
        slots = []
        for raw in raw_gpus:
            gpu = raw.strip()
            if not gpu:
                raise ValueError("GPU slots must be non-empty")
            slots.append({"host": None, "gpu": gpu, "label": gpu})

    return slots


def absolute_project_path(value, *, resolve=True):
    path = project_path(value)
    return str(path.resolve() if resolve else path.absolute())


def resolve_proxy_validation_paths(proxy_validation):
    if not proxy_validation:
        return None
    resolved = dict(proxy_validation)
    for key in ("manifest", "control_dataset", "monitor_dataset", "full_dataset", "full_holdout_dataset"):
        if resolved.get(key):
            resolved[key] = absolute_project_path(resolved[key], resolve=False)
    return resolved


def resolve_shared_paths(shared):
    resolved = dict(shared)
    for key in ("dataset", "data_config", "network_config"):
        resolved[key] = absolute_project_path(resolved[key])
    proxy_validation = resolve_proxy_validation_paths(resolved.get("proxy_validation"))
    if proxy_validation:
        resolved["proxy_validation"] = proxy_validation
        active_subset = proxy_validation.get("active_subset", "control")
        dataset_key = f"{active_subset}_dataset"
        suffix_key = f"{active_subset}_suffix"
        if proxy_validation.get("train_suffix") and not resolved.get("train_suffix"):
            resolved["train_suffix"] = proxy_validation["train_suffix"]
        if proxy_validation.get(dataset_key) and not resolved.get("validation_dataset"):
            resolved["validation_dataset"] = proxy_validation[dataset_key]
        if proxy_validation.get(suffix_key) and not resolved.get("validation_suffix"):
            resolved["validation_suffix"] = proxy_validation[suffix_key]
    if resolved.get("validation_dataset"):
        resolved["validation_dataset"] = absolute_project_path(resolved["validation_dataset"])
    if resolved.get("training_controller"):
        resolved["training_controller"] = absolute_project_path(resolved["training_controller"])
    resolved["checkpoint"] = absolute_project_path(resolved["checkpoint"], resolve=False)
    for key in ("initial_state", "initial_optimizer", "initial_controller"):
        if resolved.get(key):
            resolved[key] = absolute_project_path(resolved[key], resolve=False)
    resolved["data_extension"] = normalize_data_extension(resolved.get("data_extension", "root"))
    return resolved


def smoke_overrides(shared, population):
    shared = dict(shared)
    population = [dict(member) for member in population]
    shared.update(
        generations=2,
        weaver_epochs_per_generation=1,
        samples_per_epoch=7680,
        samples_per_epoch_val=3000,
    )
    return shared, population[:2]


def load_config(args):
    config_path = args.config.resolve()
    payload = load_yaml_with_presets(config_path)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Expected a schema_version: 1 PBT configuration")

    mapping_sections = ("experiment", "shared", "resources", "pbt")
    if any(not isinstance(payload.get(key), dict) for key in mapping_sections):
        raise ValueError(f"PBT configuration requires mappings: {', '.join(mapping_sections)}")
    if not isinstance(payload.get("population"), list):
        raise ValueError("PBT configuration requires a population list")

    sections = PBTYamlConfig.parse_payload(payload).runtime_sections()
    experiment = dict(sections["experiment"])
    shared = resolve_shared_paths(sections["shared"])
    resources = dict(sections["resources"])
    population = [dict(member) for member in sections["population"]]
    pbt = dict(sections["pbt"])

    slots = parse_slots(args, resources)
    if args.smoke:
        shared, population = smoke_overrides(shared, population)

    name = args.experiment_name or experiment.get("name")
    if args.smoke and not args.experiment_name and not str(name).endswith("_smoke"):
        name = f"{name}_smoke"

    resolved = ResolvedPBTConfig.from_sections(
        config_path=config_path,
        experiment_name=name,
        output_root=absolute_project_path(experiment["output_root"]),
        shared=shared,
        slots=slots,
        population=population,
        pbt=pbt,
        smoke=args.smoke,
    )
    return resolved.to_runtime_dict()


def validate_inputs(config):
    shared = config["shared"]
    files = ("checkpoint", "data_config", "network_config")
    for key in files:
        if not Path(shared[key]).is_file():
            raise FileNotFoundError(f"{key} not found: {shared[key]}")
    for key in ("initial_state", "initial_optimizer", "initial_controller"):
        if shared.get(key) and not Path(shared[key]).is_file():
            raise FileNotFoundError(f"{key} not found: {shared[key]}")
    if shared.get("training_controller") and not Path(shared["training_controller"]).is_file():
        raise FileNotFoundError(
            f"training_controller not found: {shared['training_controller']}"
        )
    validate_dataset(
        shared["dataset"],
        shared.get("data_extension", "root"),
        shared.get("validation_dataset"),
        shared.get("train_suffix"),
        shared.get("validation_suffix"),
    )
    if not (PROJECT_DIR / ".venv/bin/weaver").is_file():
        raise FileNotFoundError("Project Weaver executable is missing")


def contract_fingerprint(config):
    contract = {
        "schema_version": config["schema_version"],
        "shared": config["shared"],
        "population": config["population"],
        "pbt": config["pbt"],
        "smoke": config["smoke"],
    }
    encoded = json.dumps(contract, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()
