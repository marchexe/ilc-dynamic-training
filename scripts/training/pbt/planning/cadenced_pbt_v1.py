"""One-epoch best/worst PBT, isolated from frozen windowed_pbt_v2.

Selection has no window or persistence rule.  A qualifying metric gap always
copies one complete checkpoint bundle; LR collision can suppress only explore.
The generic exploit interval is a global completed-epoch cadence: interval N
is due after epochs N, 2N, 3N, ...; warm-up is an independent eligibility gate.
"""

import math
from decimal import Decimal
from pathlib import Path

from training.checkpoints import bundle_identity, bundle_paths, check_bundle, copy_bundle
from training.pbt.reporting.events import record_exploit_application
from training.pbt.state.optimizer_state import atomic_set_optimizer_lr
from training.runtime import atomic_json, utc_now


STRATEGY = "cadenced_pbt_v1"
_COLLISION_LOG_DISTANCE = 1.0e-12


def select_mutation(config, members, donor, recipient):
    """Use v2's bounded separation preference among distinct LR changes only."""
    pbt = config["pbt"]
    donor_lr = float(members[donor]["lr"])
    recipient_lr = float(members[recipient]["lr"])
    candidates = []
    rejected = []
    for factor in pbt["mutation_factors"]:
        new_lr = min(float(pbt["max_lr"]), max(float(pbt["min_lr"]), donor_lr * float(factor)))
        if abs(math.log(new_lr / recipient_lr)) < _COLLISION_LOG_DISTANCE:
            rejected.append({"factor": float(factor), "candidate_lr": new_lr, "reason": "unchanged_recipient_lr"})
            continue
        distances = [
            abs(math.log(new_lr / float(member["lr"])))
            for name, member in members.items() if name != recipient
        ]
        if any(distance < _COLLISION_LOG_DISTANCE for distance in distances):
            rejected.append({"factor": float(factor), "candidate_lr": new_lr, "reason": "lr_collision"})
            continue
        # For a single recipient this is v2's distance key: maximize the
        # nearest log-LR separation, then total separation, then numeric LR.
        candidates.append(((min(distances), sum(distances), new_lr), float(factor), new_lr))
    if not candidates:
        reason = "lr_collision" if any(item["reason"] == "lr_collision" for item in rejected) else "no_distinct_candidate"
        return {
            "new_lr": recipient_lr,
            "mutation_factor": None,
            "mutation_applied": False,
            "mutation_reason": reason,
            "rejected_mutations": rejected,
        }
    _, factor, new_lr = max(candidates)
    return {
        "new_lr": new_lr,
        "mutation_factor": factor,
        "mutation_applied": True,
        "mutation_reason": "mutated",
        "rejected_mutations": rejected,
    }


def cadenced_pbt_v1_plan(config, generation, members, manifest=None):
    pbt = config["pbt"]
    options = pbt[STRATEGY]
    metric = pbt["metric"]
    index = int(generation["index"])
    completed_epochs = index + 1
    terminal = completed_epochs == int(config["shared"]["generations"])
    warmup_epochs = int(options["warmup_epochs"])
    cadence_interval_epochs = int(pbt["exploit_interval_generations"])
    cadence_boundary = completed_epochs % cadence_interval_epochs == 0
    values = {}
    for name in members:
        worker = generation["workers"][name]
        if worker.get("status") != "completed" or worker.get("returncode") != 0:
            raise ValueError(f"cadenced_pbt_v1 requires a completed validation: {name}")
        value = (worker.get("metrics") or {}).get(metric)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"cadenced_pbt_v1 requires a finite reference metric: {name}")
        values[name] = float(value)
    ranking = sorted(members, key=lambda name: (values[name], name))
    donor, recipient = ranking[0], ranking[-1]
    # Compare the decimal values written to JSON/logs, avoiding a binary
    # roundoff artefact that could turn an exact 0.002 tie into an action.
    decimal_gap = Decimal(str(values[recipient])) - Decimal(str(values[donor]))
    gap = float(decimal_gap)
    margin = float(options["decision_margin"])
    opportunity = completed_epochs >= warmup_epochs and cadence_boundary and not terminal
    if terminal:
        reason = "terminal_generation"
    elif completed_epochs < warmup_epochs:
        reason = "warmup"
    elif not cadence_boundary:
        reason = "off_cadence"
    elif decimal_gap <= Decimal(str(margin)):
        reason = "within_margin"
    else:
        reason = "metric_gap_exceeds_margin"
    decision = {
        "generation": index,
        "completed_epoch": completed_epochs,
        "weaver_epoch": generation["epoch"],
        "validation_complete": True,
        "decision_metric_backend": "full_reference",
        "metric": metric,
        "ranking": ranking,
        "scores": values,
        "donor": donor,
        "recipient": recipient,
        "donor_lr": float(members[donor]["lr"]),
        "old_lr": float(members[recipient]["lr"]),
        "new_lr": float(members[recipient]["lr"]),
        "mutation_applied": False,
        "mutation_reason": "not_attempted",
        "metric_gap": gap,
        "decision_margin": margin,
        "cadence_interval_epochs": cadence_interval_epochs,
        "cadence_boundary": cadence_boundary,
        "warmup_active_during_training": completed_epochs <= warmup_epochs,
        "warmup_complete_at_boundary": completed_epochs >= warmup_epochs,
        "copy_opportunity": opportunity,
        "lr_mutation_opportunity": opportunity,
        "terminal": terminal,
        "reason": reason,
        "copy_planned": reason == "metric_gap_exceeds_margin",
        "copy_applied": False,
    }
    generation[STRATEGY] = decision
    if not decision["copy_planned"]:
        return ranking, []
    mutation = select_mutation(config, members, donor, recipient)
    event = {
        "event_id": f"{STRATEGY}:g{index:03d}:{donor}:{recipient}",
        "source": STRATEGY,
        "donor": donor,
        "recipient": recipient,
        "donor_lr": float(members[donor]["lr"]),
        "recipient_lr": float(members[recipient]["lr"]),
        "new_lr": mutation["new_lr"],
        "mutation_factor": mutation["mutation_factor"],
        "mutation_applied": mutation["mutation_applied"],
        "mutation_reason": mutation["mutation_reason"],
        "rejected_mutations": mutation["rejected_mutations"],
        "metric_gap": gap,
        "decision_margin": margin,
        "reason": "single_epoch_best_worst",
        "applied": False,
    }
    decision.update(
        old_lr=event["recipient_lr"], new_lr=event["new_lr"],
        mutation_factor=event["mutation_factor"],
        mutation_applied=event["mutation_applied"],
        mutation_reason=event["mutation_reason"],
    )
    return ranking, [event]


def _archive_paths(run, generation, name, epoch):
    return bundle_paths(Path(run) / "checkpoints" / STRATEGY / f"generation-{generation:03d}" / name, epoch)


def prepare_boundary(run, generation):
    """Snapshot validated bundles before the plan is committed or live state is touched."""
    if not generation["exploit"]:
        return
    event = generation["exploit"][0]
    epoch = generation["epoch"]
    donor = event["donor"]
    recipient = event["recipient"]
    donor_live = bundle_paths(Path(run) / donor, epoch)
    recipient_live = bundle_paths(Path(run) / recipient, epoch)
    donor_before = bundle_identity(donor_live)
    recipient_before = bundle_identity(recipient_live)
    donor_archive = copy_bundle(donor_live, _archive_paths(run, generation["index"], donor, epoch))
    recipient_archive = copy_bundle(recipient_live, _archive_paths(run, generation["index"], recipient, epoch))
    event.update(
        donor_checkpoint=donor_before,
        donor_archive=donor_archive,
        pre_copy=recipient_before,
        pre_copy_archive=recipient_archive,
    )
    generation["workers"][donor]["evaluated_checkpoint"] = donor_archive
    generation["workers"][recipient]["evaluated_checkpoint"] = recipient_archive


def apply_cadenced_exploits(run, manifest, generation, manifest_path):
    """Replay safely from immutable evidence if copying was interrupted."""
    for event in generation["exploit"]:
        if event["applied"]:
            post_copy = event.get("post_copy")
            if not post_copy:
                raise ValueError("Applied cadenced exploit is missing post-copy checkpoint identity")
            check_bundle(post_copy)
            for key in ("donor_archive", "pre_copy_archive"):
                if not event.get(key):
                    raise ValueError(f"Applied cadenced exploit is missing {key}")
                check_bundle(event[key])
            continue
        for key in ("donor_checkpoint", "donor_archive", "pre_copy_archive"):
            check_bundle(event[key])
        source = {part: Path(item["path"]) for part, item in event["donor_archive"].items()}
        destination = bundle_paths(Path(run) / event["recipient"], generation["epoch"])
        copied = copy_bundle(source, destination)
        atomic_set_optimizer_lr(destination["optimizer"], event["new_lr"])
        post_copy = bundle_identity(destination)
        check_bundle(event["donor_archive"])
        check_bundle(event["pre_copy_archive"])
        event.update(copied_checkpoint=copied, post_copy=post_copy, applied=True)
        generation["workers"][event["recipient"]]["resume_checkpoint"] = post_copy
        member = manifest["members"][event["recipient"]]
        member.update(
            lr=event["new_lr"], parent=event["donor"], parent_source=STRATEGY,
            last_exploit_generation=generation["index"],
        )
        generation[STRATEGY]["copy_applied"] = True
        record_exploit_application(
            run, manifest["config"], generation, event,
            source["state"], source["optimizer"],
            destination["state"], destination["optimizer"],
            weight_copied=True, optimizer_copied=True,
        )
        manifest["updated_at"] = utc_now()
        atomic_json(manifest_path, manifest)
