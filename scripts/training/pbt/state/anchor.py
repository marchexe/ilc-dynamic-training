#!/usr/bin/env python3
"""Single persisted population anchor for the anchor_copy_lr_recenter
strategy.

Deliberately not a reuse of global_best_paths/update_global_best
(checkpointing.py, metrics.py): that mechanism is monotone (no rewind
concept), copies state and optimizer via two *separate* atomic_copy calls
(a real gap -- a kill between them can pair a new state with a stale
optimizer), carries no checkpoint hash, and has no lr_center or
validation-tier identity. anchor_copy_lr_recenter needs a genuine
accept/reuse/rewind cycle, so it gets its own bundle here, atomic across
every file in the bundle at once via checkpointing.atomic_copy_pair
(imported, not modified), plus a sha256 recorded per file so resume can
detect a corrupted/incomplete bundle instead of silently trusting it.
"""

from pathlib import Path

from training.pbt.state.checkpointing import atomic_copy_pair, checkpoint_paths, controller_checkpoint_path
from training.runtime import atomic_json, sha256, utc_now


def anchor_paths(experiment_dir):
    checkpoint_dir = Path(experiment_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return {
        "state_path": str(checkpoint_dir / "anchor_state.pt"),
        "optimizer_path": str(checkpoint_dir / "anchor_optimizer.pt"),
        "controller_path": str(checkpoint_dir / "anchor_controller.pt"),
        "metadata_path": str(checkpoint_dir / "anchor_metadata.json"),
    }


def update_anchor(experiment_dir, manifest, generation_record, winner_name, metric_value, lr, lr_center, eval_tier):
    """accepted_new_anchor: persist `winner_name`'s current checkpoint as
    the new anchor -- state + optimizer + optional controller copied as one
    atomic unit -- and update manifest["anchor"] in place. Raises if the
    winner's own checkpoint is incomplete (mirrors update_global_best's same
    guard). Deliberately does NOT persist manifest.json itself: this is
    called from inside apply_exploit's per-event loop (state/transitions.py),
    which already does exactly one atomic_json(manifest_path, manifest) per
    event uniformly for every event type -- a second write here would just
    be redundant, not incorrect, so it's left to the caller for consistency
    with how every other event in that loop is handled."""
    config = manifest["config"]
    experiment_dir = Path(experiment_dir)
    member_dir = experiment_dir / winner_name
    state_path, optimizer_path = checkpoint_paths(member_dir, generation_record["epoch"])
    if not state_path.is_file() or not optimizer_path.is_file():
        raise FileNotFoundError(f"Anchor source checkpoint is incomplete: {winner_name}")

    paths = anchor_paths(experiment_dir)
    controller_source = controller_checkpoint_path(member_dir, generation_record["epoch"])
    has_controller = bool(config["shared"].get("training_controller")) and controller_source.is_file()

    pairs = [(state_path, Path(paths["state_path"])), (optimizer_path, Path(paths["optimizer_path"]))]
    if has_controller:
        pairs.append((controller_source, Path(paths["controller_path"])))
    atomic_copy_pair(pairs)
    controller_path = Path(paths["controller_path"])
    if not has_controller and controller_path.exists():
        controller_path.unlink()

    anchor_record = {
        "generation": generation_record["index"],
        "epoch": generation_record["epoch"],
        "member": winner_name,
        "metric": config["pbt"]["metric"],
        "mode": config["pbt"]["mode"],
        "metric_value": float(metric_value),
        "lr": float(lr),
        "lr_center": float(lr_center),
        "eval_tier": eval_tier,
        "updated_at": utc_now(),
        "source_state_path": str(state_path),
        "source_optimizer_path": str(optimizer_path),
        "source_controller_path": str(controller_source) if has_controller else None,
        "sha256_state": sha256(Path(paths["state_path"])),
        "sha256_optimizer": sha256(Path(paths["optimizer_path"])),
        "sha256_controller": sha256(controller_path) if has_controller else None,
        **paths,
    }
    manifest["anchor"] = anchor_record
    atomic_json(Path(paths["metadata_path"]), anchor_record)
    return anchor_record


def update_anchor_lr_center(manifest, lr_center):
    """reused_previous_anchor: the anchor's weights/optimizer/metric stay
    exactly as they were (no bundle copy -- this is what distinguishes
    "reused" from "accepted"), but lr_center still moves toward this
    generation's winner. Updates just that one field in manifest["anchor"]
    and re-persists anchor_metadata.json; manifest.json persistence is the
    caller's responsibility, same as update_anchor above."""
    anchor = manifest.get("anchor")
    if anchor is None:
        raise ValueError("update_anchor_lr_center requires an existing anchor")
    anchor = dict(anchor)
    anchor["lr_center"] = float(lr_center)
    anchor["updated_at"] = utc_now()
    manifest["anchor"] = anchor
    metadata_path = anchor.get("metadata_path")
    if metadata_path:
        atomic_json(Path(metadata_path), anchor)
    return anchor


def verify_anchor_on_disk(manifest):
    """Best-effort resume-time check: does the anchor's recorded sha256
    still match what's actually on disk? update_global_best's on-disk
    bundle is never re-verified against manifest["best"] on resume at all
    -- this closes that gap for the anchor instead of inheriting it.
    Returns (ok: bool, reason: str | None); never raises."""
    anchor = manifest.get("anchor")
    if anchor is None:
        return True, None
    for path_key, hash_key in (("state_path", "sha256_state"), ("optimizer_path", "sha256_optimizer")):
        path_value = anchor.get(path_key)
        expected = anchor.get(hash_key)
        if not path_value or not expected:
            continue
        path = Path(path_value)
        if not path.is_file():
            return False, f"anchor {path_key} missing on disk: {path}"
        if sha256(path) != expected:
            return False, f"anchor {path_key} sha256 mismatch: {path}"
    return True, None
