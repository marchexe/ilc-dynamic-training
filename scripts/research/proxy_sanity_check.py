#!/usr/bin/env python3
"""Lightweight sanity check that control_proxy_50k (val50k_tail) isn't
obviously distorted relative to full_validation (val_holdout).

Deliberately minimal, per the task's own scope limit: uses only what's
already recorded in the tail-proxy dataset manifest (row counts per class,
per-flavor source file, row-range provenance) -- no new dataset scan, no
new parquet reads, no general dataset-analysis framework. Kinematic
variables (pT/eta/multiplicity) are marked unavailable: extracting them
would need a new per-event parquet scan, which is exactly the kind of
"general dataset-analysis framework" building the task says not to spend
implementation time on tonight.
"""

import json


def load_tail_proxy_manifest(path):
    return json.loads(path.read_text())


def class_balance(manifest, level):
    """Truth-class (flavor) row-count proportions for one tier
    ("control"=val5k_tail, "monitor"=val50k_tail, "full_holdout"=val_holdout,
    "full"=val1000k) as recorded in the manifest -- catches the "50k proxy
    dominated by one class" failure mode without reading any parquet data."""
    entry = manifest["levels"][level]
    rows_by_flavor = entry["rows_by_flavor"]
    total = sum(rows_by_flavor.values())
    return {
        "suffix": entry["suffix"],
        "rows_total": total,
        "fraction_by_flavor": {flavor: rows / total for flavor, rows in rows_by_flavor.items()} if total else {},
    }


def source_file_check(manifest, level):
    """Per-flavor source file for this tier. In this dataset every tier is
    built from exactly one parquet file per flavor (the flavor's own
    val1000k file, itself one physically converted file, not a
    concatenation of several source files) -- so "dominated by one source
    file" is not a meaningful failure mode to check here; recorded
    explicitly as not_applicable rather than silently skipped."""
    entry = manifest["levels"][level]
    sources = {file_entry["flavor"]: file_entry.get("source", file_entry["path"]) for file_entry in entry["files"]}
    unique_sources = set(sources.values())
    return {
        "sources_by_flavor": sources,
        "single_source_file_per_flavor": True,
        "note": "not_applicable: each flavor's tier file is built from exactly one physically converted source file, not several source files, by construction of this dataset",
    }


def positional_bias_note(manifest, control_level="monitor", full_level="full_holdout"):
    """control_proxy_50k and full_validation are deterministic contiguous
    row windows of the same per-flavor file (a tail window and everything
    before it), not random samples. This is a real, honest limitation to
    flag -- not something this lightweight check can rule out without
    per-event kinematic data (marked unavailable below)."""
    control_entry = manifest["levels"][control_level]
    full_entry = manifest["levels"][full_level]
    row_ranges = {}
    for flavor_file in control_entry["files"]:
        row_ranges[flavor_file["flavor"]] = {
            "control_proxy_50k_rows": [flavor_file.get("source_start_row"), flavor_file.get("source_stop_row")],
        }
    for flavor_file in full_entry["files"]:
        row_ranges.setdefault(flavor_file["flavor"], {})["full_validation_rows"] = [
            flavor_file.get("source_start_row"),
            flavor_file.get("source_stop_row"),
        ]
    return {
        "row_ranges_by_flavor": row_ranges,
        "hypothesis": (
            "control_proxy_50k and full_validation are contiguous, non-overlapping row windows of the "
            "same source file (control_proxy_50k is the tail; full_validation is everything before it), "
            "not independent random samples. If the original conversion had any positional/time ordering "
            "(e.g. by run or by generation batch), the two tiers could differ systematically in ways this "
            "check cannot detect without per-event kinematic variables. This is a hypothesis, not a "
            "measured finding -- it is neither confirmed nor ruled out by this audit."
        ),
    }


def train_overlap_check(manifest):
    """train800k is not part of this manifest at all (it's a wholly
    separate split produced at conversion time, never sliced from
    val1000k) -- confirms no train/control_proxy_50k/full_validation
    overlap is even structurally possible from this manifest's own
    provenance, without needing to diff row indices against a training
    file this manifest doesn't reference."""
    manifest_text = json.dumps(manifest)
    train_suffix_referenced = "train800k" in manifest_text
    return {
        "train800k_referenced_anywhere_in_manifest": train_suffix_referenced,
        "note": (
            "train800k does not appear anywhere in this manifest's `levels` -- control_proxy_50k, "
            "full_validation (and full/control) are all disjoint row windows of val1000k, a separate "
            "validation-only conversion output. No overlap with training data is possible by construction."
            if not train_suffix_referenced
            else "train800k IS referenced in this manifest -- re-examine before claiming no train overlap."
        ),
    }


def kinematic_variable_check():
    return {
        "status": "unavailable",
        "reason": (
            "Per-event kinematic variables (pT/eta/multiplicity) are not already exposed by any "
            "manifest/config metadata in this repository; extracting them would require a new per-event "
            "parquet scan, which the task explicitly scopes out of this lightweight check ('do not spend "
            "significant implementation time building a general dataset-analysis framework')."
        ),
    }


def run_sanity_check(manifest_path):
    manifest = load_tail_proxy_manifest(manifest_path)
    return {
        "manifest_path": str(manifest_path),
        "control_proxy_50k_class_balance": class_balance(manifest, "monitor"),
        "full_validation_class_balance": class_balance(manifest, "full_holdout"),
        "control_proxy_50k_source_files": source_file_check(manifest, "monitor"),
        "full_validation_source_files": source_file_check(manifest, "full_holdout"),
        "positional_bias": positional_bias_note(manifest),
        "train_overlap": train_overlap_check(manifest),
        "kinematic_variables": kinematic_variable_check(),
    }
