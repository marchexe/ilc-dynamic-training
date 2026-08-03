#!/usr/bin/env python3
"""Apply typed PBT exploit events to checkpoints and member state."""

from training.pbt.artifacts import record_exploit_application
from training.pbt.checkpointing import atomic_copy, atomic_copy_pair, checkpoint_paths, controller_checkpoint_path
from training.pbt.models.events import dump_exploit_event, parse_exploit_event
from training.runtime import atomic_json, utc_now


def apply_exploit(experiment_dir, manifest, generation_record, manifest_path):
    epoch = generation_record["epoch"]
    for event in generation_record["exploit"]:
        event_model = parse_exploit_event(event)
        event.clear()
        event.update(dump_exploit_event(event_model))
        if event_model.applied:
            continue

        recipient_dir = experiment_dir / event_model.recipient
        donor_state, donor_optimizer, donor_controller = event_model.donor_paths(
            experiment_dir, recipient_dir, epoch, manifest
        )
        recipient_state, recipient_optimizer = checkpoint_paths(recipient_dir, epoch)
        if not donor_state.is_file() or not donor_optimizer.is_file():
            raise FileNotFoundError(f"Donor checkpoint is incomplete: {event_model.error_donor_name()}")
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
