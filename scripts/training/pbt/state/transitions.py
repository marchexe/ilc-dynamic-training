#!/usr/bin/env python3
"""Apply typed PBT exploit events to checkpoints and member state."""

from training.pbt.reporting import record_exploit_application
from training.pbt.state.anchor import update_anchor, update_anchor_lr_center
from training.pbt.state.checkpointing import (
    atomic_copy,
    atomic_copy_pair,
    checkpoint_paths,
    controller_checkpoint_path,
    population_lr_policy_snapshot_paths,
)
from training.pbt.state.optimizer_state import atomic_set_optimizer_lr
from training.pbt.models.events import dump_exploit_event, parse_exploit_event
from training.runtime import atomic_json, utc_now

ANCHOR_PSEUDO_RECIPIENT = "__anchor__"


def _apply_anchor_bundle_update(experiment_dir, manifest, generation_record, event_model):
    """anchor_copy_lr_recenter's once-per-generation anchor bookkeeping.
    Intercepted before the generic per-member copy loop below -- there is
    no real "__anchor__" member directory to resolve donor/recipient paths
    against, so this never reaches that code. accepted_new_anchor and
    plateau_escape_accepted both copy the winner's checkpoint into
    dedicated anchor storage (state/anchor.py, atomic across the whole
    bundle) -- a plateau escape is a forced accept, not a different kind of
    write; reused_previous_anchor only moves lr_center (weights/optimizer/
    metric stay exactly as they were); rewound_to_previous_anchor is a pure
    no-op here, since the anchor is by definition already the state every
    member is about to be restored to.
    """
    if event_model.decision in ("accepted_new_anchor", "plateau_escape_accepted"):
        winner_lr = float(manifest["members"][event_model.winner]["lr"])
        update_anchor(
            experiment_dir, manifest, generation_record,
            event_model.winner, event_model.winner_metric_value, winner_lr,
            event_model.lr_center, "control",
        )
    elif event_model.decision == "reused_previous_anchor":
        update_anchor_lr_center(manifest, event_model.lr_center)
    # rewound_to_previous_anchor: nothing to write -- the anchor already is
    # the state every member's event below will restore.


def apply_exploit(experiment_dir, manifest, generation_record, manifest_path):
    epoch = generation_record["epoch"]
    for event in generation_record["exploit"]:
        event_model = parse_exploit_event(event)
        event.clear()
        event.update(dump_exploit_event(event_model))
        if event_model.applied:
            continue

        if event_model.source == "anchor_copy_lr_recenter" and event_model.recipient == ANCHOR_PSEUDO_RECIPIENT:
            _apply_anchor_bundle_update(experiment_dir, manifest, generation_record, event_model)
            event["applied"] = True
            manifest["updated_at"] = utc_now()
            atomic_json(manifest_path, manifest)
            continue

        recipient_dir = experiment_dir / event_model.recipient
        donor_state, donor_optimizer, donor_controller = event_model.donor_paths(
            experiment_dir, recipient_dir, epoch, manifest
        )
        recipient_state, recipient_optimizer = checkpoint_paths(recipient_dir, epoch)
        if not donor_state.is_file() or not donor_optimizer.is_file():
            raise FileNotFoundError(f"Donor checkpoint is incomplete: {event_model.error_donor_name()}")

        if event_model.source == "population_lr_policy":
            # net_epoch-{epoch}_* is both the recipient's own just-trained
            # checkpoint and the donor-copy destination -- snapshot the
            # recipient's pre-copy state to a side path first, so a later
            # population_lr_policy_resolution rollback has something to
            # restore that isn't just the donor's state again.
            snapshot_state, snapshot_optimizer = population_lr_policy_snapshot_paths(recipient_dir, epoch)
            if recipient_state.is_file() and recipient_optimizer.is_file() and not snapshot_state.is_file():
                atomic_copy_pair([(recipient_state, snapshot_state), (recipient_optimizer, snapshot_optimizer)])

        weight_copied = donor_state.resolve() != recipient_state.resolve()
        optimizer_copied = donor_optimizer.resolve() != recipient_optimizer.resolve()
        # Weight and optimizer copy must land together or not at all: a
        # recipient must never continue training with donor weights paired
        # with its own unrelated, pre-copy optimizer state.
        pairs = []
        if weight_copied:
            pairs.append((donor_state, recipient_state))
        if optimizer_copied:
            pairs.append((donor_optimizer, recipient_optimizer))
        if pairs:
            atomic_copy_pair(pairs)

        if event_model.source == "anchor_copy_lr_recenter":
            # The copied optimizer.pt is a raw byte copy of the anchor's own
            # file, so its param_groups still hold whichever LR the anchor
            # donor was training at -- never this recipient's newly
            # assigned spread LR. Weaver's own --override-load-lr fixes
            # this in memory at the *next* training resume regardless (see
            # weaver-core/weaver/train.py:806-831), but the file would
            # briefly hold a stale LR at rest between this copy and that
            # resume. Patch it now so the persisted invariant ("a copied
            # optimizer must not keep the donor LR while the manifest
            # reports a different member LR") holds without depending on
            # that override flag always being correctly wired.
            atomic_set_optimizer_lr(recipient_optimizer, event_model.new_lr)

        shared_config = manifest.get("config", {}).get("shared", {})
        config_payload = manifest.get("config", {}).get("pbt", {})
        if shared_config.get("training_controller"):
            recipient_controller = controller_checkpoint_path(recipient_dir, epoch)
            if config_payload.get("controller_state_on_exploit", "copy") == "reset":
                if recipient_controller.exists():
                    recipient_controller.unlink()
            else:
                if not donor_controller.is_file():
                    raise FileNotFoundError(
                        f"Donor controller checkpoint is incomplete: {event_model.error_donor_name()}"
                    )
                if donor_controller.resolve() != recipient_controller.resolve():
                    atomic_copy(donor_controller, recipient_controller)

        member = manifest["members"][event_model.recipient]
        member["lr"] = event_model.new_lr
        member.update(event_model.parent_update())
        member["last_exploit_generation"] = generation_record["index"]
        event["applied"] = True
        record_exploit_application(
            experiment_dir,
            manifest.get("config", {}),
            generation_record,
            event,
            donor_state,
            donor_optimizer,
            recipient_state,
            recipient_optimizer,
            weight_copied=weight_copied,
            optimizer_copied=optimizer_copied,
        )
        manifest["updated_at"] = utc_now()
        atomic_json(manifest_path, manifest)

        if event_model.source == "population_lr_policy_resolution":
            # Only after "applied" is durably persisted above: this
            # snapshot's one and only consumer (this resolution event) has
            # now definitely used it, whether the outcome was accepted or
            # rolled_back, so it can never be needed again. Deleting before
            # the persist would risk a resume finding applied=False with no
            # snapshot left to restore from.
            snapshot_state, snapshot_optimizer = population_lr_policy_snapshot_paths(
                recipient_dir, event_model.rollback_epoch
            )
            snapshot_state.unlink(missing_ok=True)
            snapshot_optimizer.unlink(missing_ok=True)
