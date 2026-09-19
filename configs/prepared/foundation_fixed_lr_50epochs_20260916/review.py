"""Read-only production status or coverage/retention verification; never runs training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from validation.verify_fixed_lr import verify
RUN = ROOT / 'runs/pbt/foundation_fixed_lr_50epochs_20260916'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('action', choices=['status', 'verify'])
parser.add_argument('--through', type=int, choices=range(1, 51), metavar='EPOCH')
args = parser.parse_args()
if not (RUN / 'manifest.json').is_file():
    print(json.dumps(dict(status='not_started', run=str(RUN))))
    raise SystemExit(1)
m = json.loads((RUN / 'manifest.json').read_text())
completed = [g for g in m['generations'] if g['status'] == 'completed']
latest = m['generations'][-1] if m['generations'] else {}
if args.action == 'status':
    print(json.dumps(dict(status=m['status'], completed_full_epochs=len(completed),
                         target_full_epochs=m['config']['shared']['generations'],
                         latest_checkpoint_epoch=latest.get('epoch'),
                         arms={n: {k: w.get(k) for k in ['status', 'pid', 'lr', 'started_at', 'finished_at']}
                               for n, w in latest.get('workers', {}).items()},
                         updated_at=m.get('updated_at'), failure=m.get('failure')), indent=2))
    raise SystemExit(0)
result = verify(RUN, args.through)
print(json.dumps(result, indent=2))
raise SystemExit(not result['passed'])
