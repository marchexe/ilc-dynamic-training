#!/usr/bin/env python3
"""Build the summary note of the project: docs/_extra/overview.html.

    .venv/bin/python scripts/reports/build_report_page.py

A static page (no scripts), copied to the site root by Sphinx (html_extra_path): the trial table and the
interpretation, written in scripts/reports/report_note.html.
"""
import html
import shutil
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_report_trials as trials_mod  # noqa: E402

ROOT = trials_mod.ROOT
RUNS = trials_mod.RUNS
PRIMARY = trials_mod.PRIMARY
REPO = 'https://github.com/marchexe/ilc-dynamic-training/blob/main/'
REF_PAGE = {'sub-epoch-restart-test': 'results', 'large-dataset-results': 'results', 'sub-epoch-adaptation': 'ongoing',
            'pbt-vs-standard-part': 'results', 'pbt-vs-standard-part-8m': 'results', 'probe-scored-pbt': 'results',
            'prediction-ensemble': 'results', 'ensemble-800k': 'results', 'ensemble-diversity': 'results',
            'dropout-branch': 'ongoing', 'distillation': 'results'}


def link(markup):
    """RST role markup of export_report_trials.py -> (label, url)."""
    role, label, target = re.fullmatch(r':(\w+):`([^<]+) <([^>]+)>`', markup).groups()
    label = label.strip()
    if role == 'repo':
        return label, REPO + target
    if role == 'doc':
        return label, f'{target}.html'
    return label, f'{REF_PAGE[target]}.html#{target}'


def plain(text):
    return re.sub(r'\*\*([^*]+)\*\*', r'\1', text)


def trials():
    phases = []
    for title, data, validation, items in trials_mod.PHASES:
        rows = []
        for number, what, runs, records, dates, result, outcome in items:
            recs = [trials_mod.run_record(r) for r in runs]
            rows.append(dict(
                n=number, what=what, result=plain(result), outcome=outcome,
                dates=trials_mod.span(recs) if recs else dates,
                gpu=round(sum(r['gpu_hours'] for r in recs), 1) if recs else None,
                records=[link(m) for m in records],
                runs=[dict(run=r['run'], url=REPO + r['config'], start=f'{r["start"]:%b %d %H:%M}',
                           end=f'{r["end"]:%b %d %H:%M}', gpu=round(r['gpu_hours'], 1), member=r['member'],
                           lr=trials_mod.shown_lr(r), epoch=r['epoch'], val=round(r['validation'], 4),
                           test=r['test']) for r in recs]))
        phases.append(dict(title=title, data=data, validation=validation, trials=rows))
    return phases




def appendix():
    out = []
    for phase in trials():
        out.append(f'<h3>{html.escape(phase["title"])}</h3>\n<p class="meta">{phase["data"]} &middot; {phase["validation"]}</p>')
        for t in phase['trials']:
            rows = []
            for r in t['runs']:
                gpu = '&lt;1' if r['gpu'] < 1 else f'{r["gpu"]:.0f}'
                test = '&ndash;' if r['test'] is None else f'{r["test"]:.4f}'
                rows.append(f'<tr><td><a href="{r["url"]}">{r["run"]}</a></td><td>{r["start"]} &rarr; {r["end"]}</td>'
                            f'<td class="n">{gpu}</td><td>{r["member"]}, {r["lr"]}, {r["epoch"]}</td>'
                            f'<td class="n">{r["val"]:.4f}</td><td class="n">{test}</td></tr>')
            table = ('<table class="runs"><tr><th>Run (configuration file)</th><th>Start &rarr; end (JST)</th>'
                     '<th class="n">GPU-h</th><th>Selected model, LR, epoch</th><th class="n">Valid.</th>'
                     '<th class="n">Test</th></tr>' + ''.join(rows) + '</table>') if rows else ''
            recs = ' &middot; '.join(f'<a href="{u}">{html.escape(l)}</a>' for l, u in t['records'])
            gpu = '' if t['gpu'] is None else ('&lt;1 GPU-h' if t['gpu'] < 1 else f'{t["gpu"]:.0f} GPU-h')
            extra = ' &middot; '.join(x for x in (gpu, f'records: {recs}' if recs else '') if x)
            out.append(f'<details><summary><span class="tn">{t["n"]}</span><span class="td">{t["dates"]}</span>'
                       f'<span class="tw">{html.escape(t["what"])}</span><span class="tr">{html.escape(t["result"])}</span>'
                       f'</summary><div class="body">{table}<p class="meta">{extra}</p></div></details>')
    return '\n'.join(out)


def main():
    out = ROOT / 'docs/_extra/overview.html'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(NOTE)
    print(f'{out.relative_to(ROOT)}: {len(NOTE) // 1024} kB')


NOTE = open(Path(__file__).with_name('report_note.html')).read()

if __name__ == '__main__':
    main()
