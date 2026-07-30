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


def resolve_shared_paths(shared):
    resolved = dict(shared)
    for key in ("dataset", "data_config", "network_config"):
        resolved[key] = absolute_project_path(resolved[key])
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
        epochs_per_generation=1,
        samples_per_epoch=7680,
        samples_per_epoch_val=3000,
    )
    return shared, population[:2]


def load_config(args):
    config_path = args.config.resolve()
    payload = yaml.safe_load(config_path.read_text())
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
    validate_dataset(shared["dataset"], shared.get("data_extension", "root"))
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
