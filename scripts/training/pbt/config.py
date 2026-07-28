#!/usr/bin/env python3
"""PBT configuration loading and validation."""

import hashlib
import json
import re
from pathlib import Path

import yaml

from training.runtime import (
    PROJECT_DIR,
    normalize_data_extension,
    project_path,
    validate_dataset,
)


MEMBER_NAME_RE = re.compile(r"[A-Za-z0-9_.-]+")


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
            slot_part = raw
            if ":" not in slot_part:
                raise ValueError(f"Expected host:gpu slot, got: {raw}")
            host, gpu = slot_part.rsplit(":", 1)
            host = host.strip()
            gpu = gpu.strip()
            if not host or not gpu:
                raise ValueError(f"Expected host:gpu slot, got: {raw}")
            label = f"{host}:{gpu}"
            slots.append({"host": host, "gpu": gpu, "label": label})
    else:
        raw_gpus = args.gpus.split(",") if args.gpus else [str(gpu) for gpu in resources.get("gpus", [])]
        slots = []
        for raw in raw_gpus:
            gpu = raw.strip()
            if not gpu:
                raise ValueError("GPU slots must be non-empty")
            slots.append({"host": None, "gpu": gpu, "label": gpu})

    if not slots:
        raise ValueError("At least one GPU slot is required")
    labels = [slot["label"] for slot in slots]
    if len(set(labels)) != len(labels):
        raise ValueError("GPU slots must be unique")
    return slots


def absolute_project_path(value, *, resolve=True):
    path = project_path(value)
    return str(path.resolve() if resolve else path.absolute())


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

    experiment = dict(payload["experiment"])
    shared = dict(payload["shared"])
    resources = dict(payload["resources"])
    population = [dict(member) for member in payload["population"]]
    pbt = dict(payload["pbt"])

    required_shared = {
        "dataset",
        "checkpoint",
        "data_config",
        "network_config",
        "seed",
        "generations",
        "epochs_per_generation",
        "samples_per_epoch",
        "samples_per_epoch_val",
        "batch_size",
        "optimizer",
        "lr_scheduler",
        "num_workers",
        "fetch_step",
        "use_amp",
        "amp_dtype",
        "no_remake_weights",
    }
    missing = sorted(required_shared - shared.keys())
    if missing:
        raise ValueError(f"Missing shared options: {', '.join(missing)}")

    for key in ("dataset", "data_config", "network_config"):
        shared[key] = absolute_project_path(shared[key])
    if shared.get("training_controller"):
        shared["training_controller"] = absolute_project_path(shared["training_controller"])
    shared["checkpoint"] = absolute_project_path(shared["checkpoint"], resolve=False)
    shared["data_extension"] = normalize_data_extension(shared.get("data_extension", "root"))

    slots = parse_slots(args, resources)

    if args.smoke:
        shared.update(
            generations=2,
            epochs_per_generation=1,
            samples_per_epoch=7680,
            samples_per_epoch_val=3000,
        )
        population = population[:2]

    if len(population) < 2:
        raise ValueError("PBT requires at least two population members")
    names = [member.get("name") for member in population]
    if any(not isinstance(name, str) or not MEMBER_NAME_RE.fullmatch(name) for name in names):
        raise ValueError("Every population member requires a filesystem-safe name")
    if len(set(names)) != len(names):
        raise ValueError("Population member names must be unique")

    required_pbt = {
        "metric",
        "mode",
        "exploit_fraction",
        "mutation_factors",
        "min_lr",
        "max_lr",
        "seed",
    }
    missing = sorted(required_pbt - pbt.keys())
    if missing:
        raise ValueError(f"Missing PBT options: {', '.join(missing)}")
    if pbt["metric"] not in {
        "validation_accuracy",
        "validation_auc",
        "validation_loss",
        "validation_bkg_rejection_bc_score",
        "validation_bkg_rejection_bd_score",
        "validation_bkg_rejection_cb_score",
        "validation_bkg_rejection_cd_score",
        "validation_b_tag_rejection_score",
        "validation_c_tag_rejection_score",
        "validation_bkg_rejection_score",
        "validation_working_point_mistag_percent",
        "validation_ctag_reference_mistag_percent",
    }:
        raise ValueError("Unsupported PBT metric")
    if pbt["mode"] not in {"max", "min"}:
        raise ValueError("PBT mode must be 'max' or 'min'")
    fraction = float(pbt["exploit_fraction"])
    if not 0 < fraction <= 0.5:
        raise ValueError("exploit_fraction must be in (0, 0.5]")
    factors = [float(value) for value in pbt["mutation_factors"]]
    if not factors or any(value <= 0 for value in factors):
        raise ValueError("mutation_factors must contain positive values")
    pbt["mutation_factors"] = factors
    pbt["exploit_fraction"] = fraction
    pbt["min_lr"] = float(pbt["min_lr"])
    pbt["max_lr"] = float(pbt["max_lr"])
    pbt["degradation_tolerance"] = float(pbt.get("degradation_tolerance", 0.02))
    pbt["degradation_window"] = int(pbt.get("degradation_window", 3))
    pbt["early_stop_degraded_generations"] = int(pbt.get("early_stop_degraded_generations", 0))
    pbt["rollback_fraction"] = float(pbt.get("rollback_fraction", 0.0))
    pbt["controller_state_on_exploit"] = pbt.get("controller_state_on_exploit", "copy")
    pbt["backend"] = pbt.get("backend", "local_weaver")
    pbt["strategy"] = pbt.get("strategy", "exploit_mutate")
    if pbt["strategy"] == "anchored_lr_sweep":
        pbt["base_start_lr"] = float(pbt["base_start_lr"])
        if pbt.get("lr_radius"):
            radius = dict(pbt["lr_radius"])
            radius["initial"] = float(radius["initial"])
            radius["minimum"] = float(radius["minimum"])
            radius["shrink_factor"] = float(radius["shrink_factor"])
            radius["shrink_after_inner_wins"] = int(radius["shrink_after_inner_wins"])
            radius["keep_if_edge_wins"] = bool(radius.get("keep_if_edge_wins", True))
            pbt["lr_radius"] = radius
        else:
            pbt["lr_factors"] = [float(value) for value in pbt["lr_factors"]]
    if not 0 < pbt["min_lr"] < pbt["max_lr"]:
        raise ValueError("Expected 0 < min_lr < max_lr")
    if not 0 <= pbt["degradation_tolerance"] < 1:
        raise ValueError("degradation_tolerance must be in [0, 1)")
    if pbt["degradation_window"] < 1:
        raise ValueError("degradation_window must be positive")
    if pbt["early_stop_degraded_generations"] < 0:
        raise ValueError("early_stop_degraded_generations must be non-negative")
    if not 0 <= pbt["rollback_fraction"] <= 0.5:
        raise ValueError("rollback_fraction must be in [0, 0.5]")
    if pbt["controller_state_on_exploit"] not in {"copy", "reset"}:
        raise ValueError("controller_state_on_exploit must be 'copy' or 'reset'")
    if pbt["backend"] not in {"local_weaver", "ray_weaver", "ray_tune"}:
        raise ValueError("pbt.backend must be 'local_weaver', 'ray_weaver', or legacy 'ray_tune'")
    if pbt["strategy"] not in {"exploit_mutate", "anchored_lr_sweep", "fixed_lr_grid"}:
        raise ValueError("pbt.strategy must be 'exploit_mutate', 'anchored_lr_sweep' or 'fixed_lr_grid'")
    if pbt["strategy"] == "anchored_lr_sweep":
        if pbt.get("lr_radius"):
            radius = pbt["lr_radius"]
            if radius["initial"] < 0 or radius["minimum"] < 0:
                raise ValueError("anchored_lr_sweep lr_radius values must be non-negative")
            if radius["minimum"] > radius["initial"]:
                raise ValueError("anchored_lr_sweep lr_radius.minimum must be <= initial")
            if not 0 < radius["shrink_factor"] <= 1:
                raise ValueError("anchored_lr_sweep lr_radius.shrink_factor must be in (0, 1]")
            if radius["shrink_after_inner_wins"] < 1:
                raise ValueError("anchored_lr_sweep lr_radius.shrink_after_inner_wins must be positive")
            if len(population) == 2:
                offsets = (radius["initial"], -radius["initial"])
            elif len(population) == 4:
                offsets = (radius["initial"], radius["initial"] / 2, -radius["initial"] / 2, -radius["initial"])
            else:
                step = 2 * radius["initial"] / (len(population) - 1)
                offsets = [radius["initial"] - index * step for index in range(len(population))]
            active_factors = [1.0 + offset for offset in offsets]
        else:
            factors = pbt["lr_factors"]
            if len(factors) < len(population):
                raise ValueError("anchored_lr_sweep requires at least one lr_factor per member")
            if any(value <= 0 for value in factors):
                raise ValueError("anchored_lr_sweep lr_factors must be positive")
            if len(population) == 2 and len(factors) >= 2:
                active_factors = [factors[0], factors[-1]]
            else:
                active_factors = factors[:len(population)]
        for member, factor in zip(population, active_factors):
            member["start_lr"] = pbt["base_start_lr"] * factor
    if any(float(member.get("start_lr", 0)) <= 0 for member in population):
        raise ValueError("Every population member requires a positive start_lr")
    if any(
        not pbt["min_lr"] <= float(member["start_lr"]) <= pbt["max_lr"]
        for member in population
    ):
        raise ValueError("Population start_lr values must lie within PBT LR bounds")

    integer_options = ("generations", "epochs_per_generation", "samples_per_epoch",
                       "samples_per_epoch_val", "batch_size", "num_workers")
    if any(int(shared[key]) < 1 for key in integer_options):
        raise ValueError("Generation, epoch, sample, batch and worker counts must be positive")
    if shared["lr_scheduler"] != "none":
        raise ValueError("PBT learning-rate mutation requires lr_scheduler: none")

    name = args.experiment_name or experiment.get("name")
    if args.smoke and not args.experiment_name and not str(name).endswith("_smoke"):
        name = f"{name}_smoke"
    if not isinstance(name, str) or not MEMBER_NAME_RE.fullmatch(name):
        raise ValueError("Experiment requires a filesystem-safe name")

    return {
        "schema_version": 1,
        "config_path": str(config_path),
        "experiment_name": name,
        "output_root": absolute_project_path(experiment["output_root"]),
        "shared": shared,
        "gpus": [slot["gpu"] for slot in slots],
        "slots": slots,
        "population": population,
        "pbt": pbt,
        "smoke": args.smoke,
    }


def validate_inputs(config):
    shared = config["shared"]
    files = ("checkpoint", "data_config", "network_config")
    for key in files:
        if not Path(shared[key]).is_file():
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
    import hashlib

    return hashlib.sha256(encoded).hexdigest()
