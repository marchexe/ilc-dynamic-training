#!/usr/bin/env python3
"""Build deterministic proxy-validation parquet subsets for physics-aware control."""

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from training.runtime import FLAVOR_SAMPLE_GROUPS, PROJECT_DIR, atomic_json, project_path, sha256, utc_now


VAL_FILE_RE = re.compile(r"_(?P<flavor>bb|cc|dd)_val50k\.parquet$")
DEFAULT_PROXY_NAME = "parquet_proxy_v1"


@dataclass(frozen=True)
class ProxySubsetSpec:
    role: str
    rows_per_class: int | None
    copy_rows: bool


def stable_seed(seed, flavor):
    payload = f"{int(seed)}:{flavor}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def flavor_from_path(path):
    match = VAL_FILE_RE.search(path.name)
    if not match:
        raise ValueError(f"cannot infer validation flavor from file name: {path}")
    return match.group("flavor")


def display_path(path):
    path = Path(path)
    return str(path.relative_to(PROJECT_DIR) if path.is_relative_to(PROJECT_DIR) else path)


def validation_files(dataset, data_extension="parquet"):
    dataset = project_path(dataset)
    files = {}
    for _, flavor in FLAVOR_SAMPLE_GROUPS:
        matches = sorted(dataset.glob(f"*_{flavor}_val50k.{data_extension}"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected exactly one validation file for flavor {flavor} in {dataset}, found {len(matches)}"
            )
        files[flavor] = matches[0]
    return files


def disjoint_indices(total_rows, control_rows, monitor_rows, *, seed, flavor):
    if control_rows < 0 or monitor_rows < 0:
        raise ValueError("proxy subset sizes must be non-negative")
    if control_rows + monitor_rows > total_rows:
        raise ValueError(
            f"control+monitor rows exceed available rows for {flavor}: "
            f"{control_rows}+{monitor_rows}>{total_rows}"
        )
    order = list(range(total_rows))
    random.Random(stable_seed(seed, flavor)).shuffle(order)
    control = sorted(order[:control_rows])
    monitor = sorted(order[control_rows : control_rows + monitor_rows])
    return {"control": control, "monitor": monitor}


def take_rows(table, indices):
    return table.take(pa.array(indices, type=pa.int64()))


def write_subset_file(source_path, output_path, indices, *, compression="lz4", force=False):
    if output_path.exists() and not force:
        raise FileExistsError(f"proxy subset file exists, use --force: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(source_path)
    subset = take_rows(table, indices)
    pq.write_table(subset, output_path, compression=compression)
    return {
        "path": display_path(output_path),
        "source": display_path(source_path),
        "rows": subset.num_rows,
        "sha256": sha256(output_path),
    }


def source_file_record(source_path):
    parquet_file = pq.ParquetFile(source_path)
    return {
        "path": display_path(source_path),
        "rows": parquet_file.metadata.num_rows,
        "row_groups": parquet_file.metadata.num_row_groups,
    }


def build_proxy_subsets(
    *,
    dataset,
    output_root,
    manifest_output,
    name=DEFAULT_PROXY_NAME,
    data_extension="parquet",
    seed=2026,
    control_rows_per_class=5000,
    monitor_rows_per_class=10000,
    compression="lz4",
    force=False,
):
    if data_extension != "parquet":
        raise ValueError("proxy subset builder currently supports parquet datasets only")
    dataset = project_path(dataset)
    output_root = project_path(output_root) / name
    manifest_output = project_path(manifest_output)
    files = validation_files(dataset, data_extension)

    levels = {
        "control": {
            "role": "high_frequency_control_proxy",
            "dataset": display_path(output_root / "control"),
            "rows_per_class_requested": int(control_rows_per_class),
            "files": {},
        },
        "monitor": {
            "role": "independent_lower_frequency_monitor_proxy",
            "dataset": display_path(output_root / "monitor"),
            "rows_per_class_requested": int(monitor_rows_per_class),
            "files": {},
        },
        "full": {
            "role": "coarse_full_validation_reference",
            "dataset": display_path(dataset),
            "files": {},
        },
    }

    for flavor, source_path in files.items():
        parquet_file = pq.ParquetFile(source_path)
        row_sets = disjoint_indices(
            parquet_file.metadata.num_rows,
            int(control_rows_per_class),
            int(monitor_rows_per_class),
            seed=seed,
            flavor=flavor,
        )
        for level in ("control", "monitor"):
            output_path = output_root / level / source_path.name
            record = write_subset_file(
                source_path,
                output_path,
                row_sets[level],
                compression=compression,
                force=force,
            )
            record["flavor"] = flavor
            levels[level]["files"][flavor] = record
        full_record = source_file_record(source_path)
        full_record["flavor"] = flavor
        levels["full"]["files"][flavor] = full_record

    for level in levels.values():
        level["rows_total"] = sum(file_record["rows"] for file_record in level["files"].values())
        level["rows_by_flavor"] = {
            flavor: file_record["rows"] for flavor, file_record in sorted(level["files"].items())
        }

    manifest = {
        "schema_version": 1,
        "name": name,
        "created_at": utc_now(),
        "source_dataset": display_path(dataset),
        "data_extension": data_extension,
        "strategy": {
            "name": "class_balanced_disjoint_random",
            "seed": int(seed),
            "control_rows_per_class": int(control_rows_per_class),
            "monitor_rows_per_class": int(monitor_rows_per_class),
            "notes": "Truth-class balance is inherited from the bb/cc/dd validation files. Score/hardness stratification is intentionally left to the next proxy version once reference predictions are exported.",
        },
        "levels": levels,
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(manifest_output, manifest)
    return manifest
