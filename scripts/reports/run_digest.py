#!/usr/bin/env python3
"""Short status of PBT and standard ParT runs: progress, best value so far, LRs, and the windowed decisions.

    .venv/bin/python scripts/reports/run_digest.py [RUN ...] [--windows N]

Without RUN names, every run under runs/pbt whose manifest says running. For each run: generations done, the
best selection-set value so far and the same for the reference run at the same epoch (the standard ParT run of
the same training set), the current LRs, and for windowed PBT the last N decisions, with the probe value after
the arrow when the decision ranked probes. Read-only.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.runtime import PROJECT_DIR  # noqa: E402

PRIMARY = 'validation_total_reference_mistag_geomean_percent'
RUNS = PROJECT_DIR / 'runs/pbt'
REFERENCES = {'train8m_head': 'sel1m_train8m_standard_part', 'train800k': 'sel1m_standard_part'}


def load(run):
    return json.loads((RUNS / run / 'manifest.json').read_text())


def done(manifest):
    return [g for g in manifest['generations'] if all(w.get('metrics') for w in g['workers'].values())]


def best_so_far(manifest):
    best, out = None, {}
    for g in done(manifest):
        value = min(w['metrics'][PRIMARY] for w in g['workers'].values())
        best = value if best is None else min(best, value)
        out[g['epoch']] = best
    return out


def digest(run, windows):
    manifest = load(run)
    generations = done(manifest)
    lines = [f"{run}: {manifest['status']}, {len(generations)}/{manifest['config']['shared']['generations']} generations"]
    if not generations:
        return lines
    curve = best_so_far(manifest)
    epoch = generations[-1]['epoch']
    reference = REFERENCES.get(manifest['config']['shared'].get('train_suffix'))
    line = f"  epoch {epoch}: best so far {curve[epoch]:.4f}"
    if reference and reference != run and (RUNS / reference / 'manifest.json').is_file():
        other = best_so_far(load(reference)).get(epoch)
        if other:
            line += f"; {reference} at this epoch {other:.4f} ({100 * (curve[epoch] / other - 1):+.1f}%)"
    lines.append(line)
    lrs = sorted(m['lr'] for m in manifest['members'].values())
    lines.append('  LRs now: ' + ' '.join(f'{lr:.1e}' for lr in lrs))
    decisions = [g for g in generations if g.get('windowed_pbt_v2')][-windows:]
    for g in decisions:
        d, probes = g['windowed_pbt_v2'], (g.get('probes') or {}).get('members', {})
        names = sorted(d['members'], key=lambda n: d['members'][n]['lr'])
        cells = []
        for n in names:
            cell = f"{d['members'][n]['lr']:.1e} {g['workers'][n]['metrics'][PRIMARY]:.4f}"
            if n in probes and 'value' in probes[n]:
                cell += f"->{probes[n]['value']:.4f}"
            cells.append(cell)
        copies = ', '.join(f"{e['recipient_lr']:.1e}->{e['new_lr']:.1e}" for e in g['exploit'] or []) or 'none'
        lines.append(f"  epochs {d['epochs'][0]}-{d['epochs'][-1]} best {d['members'][d['donor']]['lr']:.1e}"
                     f"{' (probe)' if d.get('scored_on') == 'probe' else ''}: {' | '.join(cells)}; copies: {copies}")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='*')
    parser.add_argument('--windows', type=int, default=3)
    args = parser.parse_args(argv)
    runs = args.runs or sorted(p.parent.name for p in RUNS.glob('*/manifest.json')
                               if json.loads(p.read_text()).get('status') == 'running')
    for run in runs:
        print('\n'.join(digest(run, args.windows)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
