#!/usr/bin/env python3
"""Build disjoint tail proxy-validation parquet files from larger validation parquet files."""

from pathlib import Path

import pyarrow.parquet as pq

from training.runtime import FLAVOR_SAMPLE_GROUPS, PROJECT_DIR, atomic_json, utc_now


DEFAULT_SOURCE_SUFFIX = "val1000k"
DEFAULT_CONTROL_SUFFIX = "val5k_tail"
DEFAULT_MONITOR_SUFFIX = "val50k_tail"


def display_path(path):
    path = Path(path)
    return str(path.relative_to(PROJECT_DIR) if path.is_absolute() and path.is_relative_to(PROJECT_DIR) else path)


def source_file(dataset, flavor, source_suffix=DEFAULT_SOURCE_SUFFIX):
    dataset = Path(dataset)
    matches = sorted(dataset.glob(f"*_{flavor}_{source_suffix}.parquet"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one parquet file for flavor={flavor} suffix={source_suffix} in {dataset}, "
            f"found {len(matches)}"
        )
    return matches[0]


def row_group_window(parquet_file, start, stop):
    if start < 0 or stop < start or stop > parquet_file.metadata.num_rows:
        raise ValueError(
            f"invalid row window [{start}, {stop}) for {parquet_file.metadata.num_rows} rows"
        )
    groups = []
    cursor = 0
    local_start = None
    for index in range(parquet_file.metadata.num_row_groups):
        rows = parquet_file.metadata.row_group(index).num_rows
        group_start = cursor
        group_stop = cursor + rows
        if group_stop > start and group_start < stop:
            if local_start is None:
                local_start = start - group_start
            groups.append(index)
        cursor = group_stop
    if local_start is None:
        local_start = 0
    return groups, local_start


def read_row_window(path, start, length):
    parquet_file = pq.ParquetFile(path)
    stop = start + length
    groups, local_start = row_group_window(parquet_file, start, stop)
    table = parquet_file.read_row_groups(groups)
    return table.slice(local_start, length)


def write_tail_subset(source_path, output_path, start, rows, *, compression="lz4", force=False):
    output_path = Path(output_path)
    if output_path.exists() and not force:
        raise FileExistsError(f"tail proxy file exists, use --force: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = read_row_window(source_path, start, rows)
    pq.write_table(table, output_path, compression=compression)
    return {
        "path": display_path(output_path),
        "source": display_path(source_path),
        "source_start_row": int(start),
        "source_stop_row": int(start + rows),
        "rows": int(table.num_rows),
        "row_groups": pq.ParquetFile(output_path).metadata.num_row_groups,
        "size_bytes": output_path.stat().st_size,
    }


def build_tail_proxy_subsets(
    *,
    dataset,
    output_dataset=None,
    manifest_output=None,
    name="20250711_ilc_nnqq_sgv_10m_3cat_tail_proxy_v1",
    source_suffix=DEFAULT_SOURCE_SUFFIX,
    control_suffix=DEFAULT_CONTROL_SUFFIX,
    monitor_suffix=DEFAULT_MONITOR_SUFFIX,
    control_rows_per_class=5000,
    monitor_rows_per_class=50000,
    compression="lz4",
    force=False,
):
    dataset = Path(dataset)
    output_dataset = Path(output_dataset) if output_dataset else dataset
    manifest_output = Path(manifest_output) if manifest_output else None
    control_rows_per_class = int(control_rows_per_class)
    monitor_rows_per_class = int(monitor_rows_per_class)
    if control_rows_per_class <= 0 or monitor_rows_per_class <= 0:
        raise ValueError("tail proxy row counts must be positive")

    levels = {
        "control": {
            "role": "high_frequency_control_proxy_tail",
            "dataset": display_path(output_dataset),
            "suffix": control_suffix,
            "selection": "last_rows_of_full_validation",
            "files": [],
        },
        "monitor": {
            "role": "independent_lower_frequency_monitor_proxy_tail",
            "dataset": display_path(output_dataset),
            "suffix": monitor_suffix,
            "selection": "rows_immediately_before_control_tail_window",
            "files": [],
        },
        "full": {
            "role": "coarse_full_validation_reference",
            "dataset": display_path(dataset),
            "suffix": source_suffix,
            "files": [],
        },
    }

    for _, flavor in FLAVOR_SAMPLE_GROUPS:
        source_path = source_file(dataset, flavor, source_suffix)
        parquet_file = pq.ParquetFile(source_path)
        total_rows = parquet_file.metadata.num_rows
        needed = control_rows_per_class + monitor_rows_per_class
        if needed > total_rows:
            raise ValueError(
                f"control+monitor rows exceed source rows for {flavor}: {needed}>{total_rows}"
            )

        control_start = total_rows - control_rows_per_class
        monitor_start = control_start - monitor_rows_per_class
        source_stem = source_path.name.removesuffix(f"_{source_suffix}.parquet")
        control_path = output_dataset / f"{source_stem}_{control_suffix}.parquet"
        monitor_path = output_dataset / f"{source_stem}_{monitor_suffix}.parquet"

        control_record = write_tail_subset(
            source_path,
            control_path,
            control_start,
            control_rows_per_class,
            compression=compression,
            force=force,
        )
        control_record["flavor"] = flavor
        monitor_record = write_tail_subset(
            source_path,
            monitor_path,
            monitor_start,
            monitor_rows_per_class,
            compression=compression,
            force=force,
        )
        monitor_record["flavor"] = flavor
        full_record = {
            "flavor": flavor,
            "path": display_path(source_path),
            "rows": total_rows,
            "row_groups": parquet_file.metadata.num_row_groups,
            "size_bytes": source_path.stat().st_size,
        }
        levels["control"]["files"].append(control_record)
        levels["monitor"]["files"].append(monitor_record)
        levels["full"]["files"].append(full_record)

    for level in levels.values():
        level["rows_total"] = sum(item["rows"] for item in level["files"])
        level["rows_by_flavor"] = {item["flavor"]: item["rows"] for item in level["files"]}
        level["size_bytes_total"] = sum(item["size_bytes"] for item in level["files"])

    manifest = {
        "schema_version": 1,
        "name": name,
        "created_at": utc_now(),
        "source_dataset": display_path(dataset),
        "output_dataset": display_path(output_dataset),
        "data_extension": "parquet",
        "flavors": [flavor for _, flavor in FLAVOR_SAMPLE_GROUPS],
        "strategy": {
            "name": "disjoint_tail_windows_from_full_validation",
            "source_suffix": source_suffix,
            "control_suffix": control_suffix,
            "monitor_suffix": monitor_suffix,
            "control_rows_per_class": control_rows_per_class,
            "monitor_rows_per_class": monitor_rows_per_class,
            "notes": "Control uses the final rows of each full-validation parquet file. Monitor uses the immediately preceding rows, so control and monitor are disjoint while both come from the dataset tail.",
        },
        "levels": levels,
    }
    if manifest_output:
        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(manifest_output, manifest)
    return manifest
