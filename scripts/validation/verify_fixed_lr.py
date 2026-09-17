#!/usr/bin/env python3
"""Read-only evidence verification for full-epoch fixed-LR and windowed PBT studies.

Arms, horizons, datasets and initialization hashes come from the run manifest.
Requires raw optimizer continuation and one epoch per generation: the manifest
only retains the last epoch's audit for multi-epoch generations. Never launches
training/inference or writes into the run. --through checks an ongoing prefix;
final standalone parity is required when checking the whole completed run.
"""
import argparse
import copy
import json
import math
from pathlib import Path
from statistics import mean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.pbt.state.checkpointing import checkpoint_paths, epoch_for_generation
from training.runtime import sha256
from training.pbt.planning.windowed_pbt_v2 import STRATEGY, WORKING_POINTS, windowed_pbt_v2_plan


def verify_window(run, config, generation, members, history, check, saved):
    """Replay decisions from epoch evidence, independent of recorded decisions."""
    replay = copy.deepcopy(generation)
    replay.pop(STRATEGY, None)
    ranking, expected = windowed_pbt_v2_plan(config, replay, members, dict(generations=history))
    actual = generation.get('exploit') or []
    label = f"generation {generation['index']}: window"
    check(label + ' ranking', ranking == generation.get('ranking'))
    check(label + ' action count', len(expected) == len(actual))
    decision = replay.get(STRATEGY)
    recorded = generation.get(STRATEGY)
    check(label + ' boundary', bool(decision) == bool(recorded))

    def bundle(identity):
        check(label + ' bundle components', set(identity) == {'state', 'optimizer', 'scaler'})
        for item in identity.values():
            saved(label + ' bundle hash', Path(item['path']), item['sha256'])

    if decision and recorded:
        for name, evidence in decision['members'].items():
            observed = recorded['members'][name]
            check(label + ' member evidence ' + name,
                  all(observed.get(k) == v for k, v in evidence.items() if k != 'copied'))
            check(label + ' copied flag ' + name, observed.get('copied') == any(e['recipient'] == name for e in expected))
        for key in ('window_number', 'generation_indices', 'epochs', 'full_epochs', 'decision_margin', 'score_epochs', 'donor', 'terminal'):
            check(label + ' ' + key, decision[key] == recorded.get(key))
        donor = decision['donor']
        bundle(recorded.get('donor_checkpoint', {}))
        for part, item in recorded['donor_checkpoint'].items():
            check(label + ' synchronized donor path', Path(item['path']) == run / donor / f"net_epoch-{generation['epoch']}_{part}.pt")
        score = decision['members'][donor]['window_score']
        previous = [g[STRATEGY]['protected_best_score'] for g in history if STRATEGY in g]
        improved = not previous or score < previous[-1]
        update = recorded.get('protected_best_update')
        check(label + ' protected improvement', bool(update) == improved)
        protected_score = min(score, previous[-1]) if previous else score
        check(label + ' protected monotonicity', recorded.get('protected_best_score') == protected_score)
        decision['protected_best_score'] = protected_score
        if update:
            bundle(update['checkpoint'])
            check(label + ' protected source', update['member'] == donor and update['window_score'] == score
                  and update['epoch'] == generation['epoch']
                  and update['metrics'] == generation['workers'][donor]['metrics']
                  and update['window_metrics'] == decision['members'][donor]['metrics'])
            check(label + ' protected raw copy', all(update['checkpoint'][k]['sha256'] == recorded['donor_checkpoint'][k]['sha256']
                                                    for k in ('state', 'optimizer', 'scaler')))
    for wanted, event in zip(expected, actual):
        check(label + ' replayed action', all(event.get(k) == v for k, v in wanted.items() if k != 'applied'))
        check(label + ' applied', event.get('applied') is True)
        check(label + ' synchronized event donor', event['donor_checkpoint'] == recorded['donor_checkpoint'])
        for key in ('donor_checkpoint', 'pre_copy_archive', 'post_copy'):
            bundle(event.get(key, {}))
        for part in ('state', 'optimizer', 'scaler'):
            check(label + ' recipient destination', Path(event['post_copy'][part]['path']) == run / event['recipient'] / f"net_epoch-{generation['epoch']}_{part}.pt")
            check(label + ' raw copy ' + part, event['copied_checkpoint'][part]['sha256'] == event['donor_checkpoint'][part]['sha256'])
            check(label + ' preserved pre-copy ' + part, event['pre_copy'][part]['sha256'] == event['pre_copy_archive'][part]['sha256'])
        for part in ('state', 'scaler'):
            check(label + ' unchanged copied ' + part, event['post_copy'][part]['sha256'] == event['donor_checkpoint'][part]['sha256'])
        # Compare every optimizer field recursively; serialization hashes may differ.
        import torch
        from training.pbt.state.optimizer_state import load_optimizer_state, set_optimizer_state_lr
        def equal(a, b):
            if torch.is_tensor(a):
                return torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a, b)
            if isinstance(a, dict):
                return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
            if isinstance(a, (list, tuple)):
                return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
            return a == b
        source = load_optimizer_state(event['donor_checkpoint']['optimizer']['path'])
        target = load_optimizer_state(event['post_copy']['optimizer']['path'])
        check(label + ' optimizer only LR changed', equal(set_optimizer_state_lr(source, event['new_lr']), target))
    for event in expected:
        members[event['recipient']]['lr'] = event['new_lr']
    history.append(replay)


def verify(run, through=None):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    config = manifest["config"]
    shared, pbt = config["shared"], config["pbt"]
    windowed = pbt.get('strategy') == STRATEGY
    arms = {m["name"]: m["start_lr"] for m in config["population"]}
    limit = shared["generations"] if through is None else through
    if not 1 <= limit <= shared["generations"]:
        raise ValueError("--through must be within the configured generation range")
    failures, results = [], []

    def check(label, condition):
        if not condition:
            failures.append(label)

    def finite(value):
        return isinstance(value, (int, float)) and math.isfinite(value)

    def saved(label, path, digest=None):
        check(label, path.is_file() and path.stat().st_size > 0
              and (digest is None or sha256(path) == digest))

    check("nonempty unique member names", bool(arms) and len(arms) == len(config["population"]))
    check("full-epoch audited deterministic configuration", shared.get("weaver_epochs_per_generation") == 1
          and shared.get("samples_per_epoch") is None and shared.get("samples_per_epoch_val") is None
          and shared.get("deterministic") and shared.get("data_audit"))
    check("raw optimizer continuation", shared.get("initial_optimizer_mode") == "raw")
    check("legacy adaptive mechanisms disabled", pbt.get("strategy") in ("fixed_lr_grid", STRATEGY)
          and pbt.get("rollback_fraction") == 0 and pbt.get("baseline_guard_action") == "observe"
          and pbt.get("early_stop_degraded_generations") == 0
          and (pbt.get("dynamic_controller") or {}).get("mode", "disabled") == "disabled"
          and all(not pbt.get(k) for k in ["anchor_copy_lr_recenter", "lr_radius", "lr_controller", "population_lr_policy"])
          and not shared.get("training_controller") and shared.get("lr_scheduler") == "none")
    generations = manifest.get("generations", [])
    if through is None:
        check("run completed", manifest.get("status") == "completed" and len(generations) == limit)
    initial = manifest.get("initial_evaluation") or {}
    resume = manifest.get("initial_resume") or {}
    reference = (initial.get("metrics") or {}).get("validation_data_audit", {})
    check("initial evaluation", initial.get("status") == "completed"
          and bool(resume.get("state_sha256")) and initial.get("checkpoint_sha256") == resume.get("state_sha256"))
    metric_keys = {"validation_loss", pbt["metric"]}
    metric_keys.update(k for k in initial.get("metrics", {}) if "_mistag_eff_" in k)

    def dataset(label, data, split):
        files = data.get("files", [])
        expected = {f for group in manifest["datasets"]["resolved_files"][split] for f in group["files"]}
        check(label + ": dataset evidence", bool(files) and bool(data.get("fingerprint"))
              and {f["path"] for f in files} == expected
              and data.get("total_rows", 0) > 0
              and sum(f["rows"] for f in files) == data.get("total_rows"))

    dataset("validation", reference.get("dataset", {}), "val")

    def traversal(label, audit):
        items, consumed = audit.get("traversal", []), audit.get("consumed", {})
        check(label + ": full finite traversal", audit.get("exhausted") and bool(items)
              and sum(t.get("scanned", {}).get("count", 0) for t in items) == audit.get("dataset", {}).get("total_rows")
              and all(t.get("exhausted") and t.get("wraps") == 0
                      and t.get("scanned", {}).get("repeated_ids") == 0 for t in items))
        accepted = sum(t.get("accepted", {}).get("count", 0) for t in items)
        check(label + ": all accepted consumed once", accepted > 0
              and accepted == consumed.get("count") == consumed.get("unique_ids")
              and consumed.get("repeated_ids") == 0
              and bool(consumed.get("sequence_sha256")) and bool(consumed.get("id_set_sha256")))

    def validation(label, audit):
        traversal(label, audit)
        check(label + ": matched validation", audit.get("dataset") == reference.get("dataset")
              and audit.get("consumed") == reference.get("consumed")
              and audit.get("consumed", {}).get("count") == reference.get("dataset", {}).get("total_rows"))

    validation("initial", reference)
    initial_epoch = shared["initial_epoch"]
    check("initial checkpoint epoch", resume.get("epoch") == initial_epoch)
    for name in arms:
        for component in ("state", "optimizer"):
            digest = resume.get(component + "_sha256")
            check(name + ": recorded initial " + component, bool(digest))
            saved(name + ": raw initial " + component, run / name / f"net_epoch-{initial_epoch}_{component}.pt", digest)
        # Legacy checkpoints have no scaler. When supplied, its state must copy too.
        source_scaler = Path(shared["initial_optimizer"].replace("_optimizer.pt", "_scaler.pt"))
        if source_scaler.is_file():
            saved(name + ": initial scaler", run / name / f"net_epoch-{initial_epoch}_scaler.pt", sha256(source_scaler))
    train_reference = None
    replay_members = {name: dict(lr=lr) for name, lr in arms.items()}
    replay_history = []
    for index in range(limit):
        matches = [g for g in generations if g.get("index") == index]
        check(f"generation {index}: unique completed record", len(matches) == 1 and matches[0].get("status") == "completed")
        if not matches:
            continue
        generation = matches[0]
        epoch = epoch_for_generation(config, index)
        check(f"generation {index}: checkpoint epoch", generation.get("epoch") == epoch)
        check(f"generation {index}: no legacy adaptive actions", (windowed or generation.get("exploit") == [])
              and not any(generation.get(k) for k in ["controller_actions", "controller_lr_changes",
                                                     "anchor_copy_lr_recenter", "early_stop_triggered"]))
        sequences = []
        check(f"generation {index}: configured workers", set(generation.get("workers", {})) == set(arms))
        for name, lr in arms.items():
            if windowed:
                lr = replay_members[name]['lr']
            worker = generation.get("workers", {}).get(name, {})
            metrics = worker.get("metrics") or {}
            train, val = metrics.get("train_data_audit", {}), metrics.get("validation_data_audit", {})
            label = f"epoch {epoch}/{name}"
            check(label + ": worker/LR", worker.get("status") == "completed" and worker.get("returncode") == 0
                  and worker.get("lr") == metrics.get("train_loaded_optimizer_lr") == lr)
            if train_reference is None:
                train_reference = train.get("dataset", {})
                dataset("training", train_reference, "train")
            check(label + ": matched training dataset", train.get("dataset") == train_reference)
            traversal(label, train)
            check(label + ": optimizer steps", 0 < train.get("optimizer_steps", 0) <= train.get("batches", 0))
            sequences.append(json.dumps([train.get("consumed"), train.get("traversal")], sort_keys=True))
            validation(label + "/validation", val)
            for key in metric_keys:
                check(label + ": finite " + key, finite(metrics.get(key)))
            state, optimizer = checkpoint_paths(run / name, epoch)
            for path in (state, optimizer):
                saved(label + ": retained " + path.name, path)
            if shared.get("use_amp") and shared.get("amp_dtype") == "fp16":
                saved(label + ": retained scaler", run / name / f"net_epoch-{epoch}_scaler.pt")
            if index == limit - 1:
                results.append(dict(name=name, lr=lr, full_epoch=index + 1, checkpoint_epoch=epoch,
                                    loss=metrics.get("validation_loss"), metric=metrics.get(pbt["metric"])))
        check(f"generation {index}: matched training sequences", len(set(sequences)) == 1)
        if windowed:
            verify_window(run.resolve(), config, generation, replay_members, replay_history, check, saved)
    if windowed:
        updates = [g[STRATEGY]['protected_best_update'] for g in generations[:limit]
                   if g.get(STRATEGY, {}).get('protected_best_update')]
        if through is None:
            check('protected best matches last strict improvement', bool(updates) and manifest.get('protected_best') == updates[-1])
            check('live LRs match replay', all(manifest['members'][n]['lr'] == m['lr'] for n, m in replay_members.items()))
    if through is None:
        final = manifest.get("final_evaluations", {}).get("control", {})
        for name in [*arms, "selected_best", *(['protected_best'] if windowed else [])]:
            record = final.get(name, {})
            archived = name in ('selected_best', 'protected_best')
            best = manifest.get('protected_best' if name == 'protected_best' else 'best') or {}
            path = Path(best.get("state_path", "")) if archived else checkpoint_paths(
                run / name, epoch_for_generation(config, limit - 1))[0]
            training = best.get("metrics", {}) if archived else next(
                (g.get("workers", {}).get(name, {}).get("metrics", {}) for g in generations if g.get("index") == limit - 1), {})
            check(name + ": final evaluation completed", record.get("status") == "completed" and bool(record.get("checkpoint_sha256")))
            saved(name + ": final checkpoint hash", path, record.get("checkpoint_sha256"))
            metrics = record.get("metrics") or {}
            audit = metrics.get("validation_data_audit", {})
            validation(name + "/standalone", audit)
            prediction = training.get("validation_data_audit", {}).get("prediction_sha256")
            check(name + ": checkpoint prediction parity", bool(prediction) and prediction == audit.get("prediction_sha256"))
            for key in metric_keys:
                check(name + ": standalone " + key, finite(metrics.get(key)) and metrics.get(key) == training.get(key))
    return dict(passed=not failures, failures=failures, through_full_epoch=limit,
                run_status=manifest.get("status"), results=results)


def compare(run, baseline):
    """Compare matched horizons using stable windows, retaining lineage changes."""
    runs = [json.loads((Path(p) / 'manifest.json').read_text()) for p in (run, baseline)]
    candidate, control = runs
    metric = candidate['config']['pbt']['metric']
    horizon = candidate['config']['shared']['generations']
    if any(m.get('status') != 'completed' for m in runs):
        raise ValueError('Comparison requires completed runs')
    if control['config']['shared']['generations'] < horizon:
        raise ValueError('Baseline is shorter than the candidate horizon')
    for key in ('state_sha256', 'optimizer_sha256'):
        if candidate['initial_resume'][key] != control['initial_resume'][key]:
            raise ValueError('Unmatched starting checkpoints')
    if candidate['initial_evaluation']['metrics']['validation_data_audit']['consumed'] != control['initial_evaluation']['metrics']['validation_data_audit']['consumed']:
        raise ValueError('Unmatched validation events')
    for g, baseline_generation in zip(candidate['generations'][:horizon], control['generations'][:horizon]):
        baseline_worker = next(iter(baseline_generation['workers'].values()))
        if any(w['metrics']['train_data_audit']['consumed'] != baseline_worker['metrics']['train_data_audit']['consumed']
               for w in g['workers'].values()):
            raise ValueError('Unmatched training schedules')

    def summaries(manifest, end):
        rows = manifest['generations'][:end]
        result = {}
        for member in manifest['config']['population']:
            name = member['name']
            workers = [g['workers'][name] for g in rows]
            values = [w['metrics'][metric] for w in workers]
            metrics = {key: dict(final=workers[-1]['metrics'][key],
                                mean3=mean(w['metrics'][key] for w in workers[-3:]),
                                mean5=mean(w['metrics'][key] for w in workers[-5:]),
                                mean10=mean(w['metrics'][key] for w in workers[-10:]))
                       for key in (metric, 'validation_loss', *WORKING_POINTS)}
            copies = [g['index'] + 1 for g in rows if any(e['recipient'] == name for e in g.get('exploit', []))]
            result[name] = dict(final_lr=workers[-1]['lr'], metrics=metrics,
                                best_single=min(values), best_full_epoch=values.index(min(values)) + 1,
                                copies=copies, final10_crosses_copy=any(end - 10 < e < end for e in copies))
        return result

    fixed = summaries(control, horizon)
    live = summaries(candidate, horizon)
    best_fixed = min(fixed, key=lambda n: fixed[n]['metrics'][metric]['mean5'])
    target = fixed[best_fixed]['metrics'][metric]['mean5']
    for manifest, result in ((candidate, live), (control, fixed)):
        for name, row in result.items():
            values = [g['workers'][name]['metrics'][metric] for g in manifest['generations'][:horizon]]
            row['first_full_epoch_mean5_at_fixed_best_final5'] = next(
                (i + 1 for i in range(4, horizon) if mean(values[i - 4:i + 1]) <= target), None)
    boundaries = [g for g in candidate['generations'] if STRATEGY in g]
    actions = [e for g in boundaries for e in g['exploit']]
    protected = candidate.get('protected_best')
    protected_summary = None
    if protected:
        protected_summary = dict(member=protected['member'], full_epoch=protected['generation'] + 1,
                                 window_score=protected['window_score'], checkpoint=protected['checkpoint'],
                                 trajectory=summaries(candidate, protected['generation'] + 1)[protected['member']])
    changes = {name: [e for e in actions if e['recipient'] == name] for name in live}
    reversals = {name: sum((a['new_lr'] - a['recipient_lr']) * (b['new_lr'] - b['recipient_lr']) < 0
                           for a, b in zip(events, events[1:])) for name, events in changes.items()}
    return dict(horizon=horizon, fixed=fixed, live=live, best_fixed_mean5=best_fixed,
                best_live_mean5=min(live, key=lambda n: live[n]['metrics'][metric]['mean5']),
                best_single_checkpoint={k: candidate['best'][k] for k in ('member', 'epoch', 'metric_value', 'state_path')},
                protected=protected_summary,
                exploit_boundaries=sum(bool(g['exploit']) for g in boundaries), donor_copies=len(actions),
                protected_updates=sum(bool(g[STRATEGY]['protected_best_update']) for g in boundaries),
                lr_direction_reversals=reversals,
                above_initial_max=any(e['new_lr'] > max(m['start_lr'] for m in candidate['config']['population']) for e in actions),
                lr_evolution=[dict(full_epoch=g['index'] + 1,
                                   entering={n: w['lr'] for n, w in g['workers'].items()},
                                   changes=[{k: e[k] for k in ('donor', 'recipient', 'new_lr')} for e in g['exploit']]) for g in boundaries],
                interpretation='Means describe recorded member trajectories; copy-crossing windows mix lineages. LR reversals are an oscillation diagnostic, not proof of degradation. Judge stable means and all working points, not the best single epoch.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--through", type=int, metavar="EPOCH")
    parser.add_argument("--baseline", type=Path, help="Also compare stable trajectories with a completed matched control")
    args = parser.parse_args()
    try:
        result = verify(args.run.resolve(), args.through)
        if result['passed'] and args.baseline:
            if args.through is not None:
                raise ValueError('--baseline requires final verification')
            result['comparison'] = compare(args.run, args.baseline)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = dict(passed=False, failures=[f"Missing/malformed run evidence: {error}"])
    print(json.dumps(result, indent=2, allow_nan=False))
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
