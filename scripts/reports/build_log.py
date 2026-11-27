#!/usr/bin/env python3
"""Build the experiment log docs/_extra/log.html: every trial with its runs, dates and configuration files.

    .venv/bin/python scripts/reports/build_log.py

The trials, results and links are those of export_report_trials.py; run dates (JST), GPU time and the selected
model come from the run manifests in runs/pbt. Static HTML; it uses the style of the research overview.
"""
import html
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_report_trials as T  # noqa: E402

ROOT = T.ROOT
REPO = 'https://github.com/marchexe/ilc-dynamic-training/blob/main/'
REF_PAGE = {'sub-epoch-restart-test': 'results', 'large-dataset-results': 'results', 'sub-epoch-adaptation': 'ongoing',
            'pbt-vs-standard-part': 'results', 'pbt-vs-standard-part-8m': 'results', 'probe-scored-pbt': 'results',
            'prediction-ensemble': 'results', 'ensemble-800k': 'results', 'ensemble-diversity': 'results',
            'dropout-branch': 'ongoing', 'distillation': 'results'}
SECTION = {  # step of export_report_trials -> (heading, section of the overview)
    1: ('PBT against constant learning rates', 'baseline'),
    2: ('Higher constant learning rates', 'baseline'),
    3: ('PBT against standard ParT, 800k training jets', 'small'),
    4: ('PBT against standard ParT, 8M training jets', 'large'),
    5: ('Ensembles and other ways to gain, 8M training jets', 'ensemble'),
}
OUTCOME = {'✅': 'better', '❌': 'worse', '➖': 'no clear difference', '⚠️': 'not valid', '🔧': 'set-up or check'}


def link(markup):
    role, label, target = re.fullmatch(r':(\w+):`([^<]+) <([^>]+)>`', markup).groups()
    label = html.escape(label.strip())
    if role == 'repo':
        return f'<a href="{REPO}{target}">{label}</a>'
    if role == 'doc':
        return f'<a href="{target}.html">{label}</a>'
    return f'<a href="{REF_PAGE[target]}.html#{target}">{label}</a>'


def text(s):
    s = html.escape(s)
    s = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', s)
    return s


def run_row(r):
    gpu = '&lt;1' if r['gpu_hours'] < 1 else f"{r['gpu_hours']:.0f}"
    test = '–' if r['test'] is None else f"{r['test']:.4f}"
    return (f'<tr><td><a href="{REPO}{r["config"]}">{r["run"]}</a></td>'
            f'<td class="nowrap">{r["start"]:%b %d %H:%M} → {r["end"]:%b %d %H:%M}</td>'
            f'<td class="num">{gpu}</td><td>{r["member"]}, {T.shown_lr(r)}, epoch {r["epoch"]}</td>'
            f'<td class="num">{r["validation"]:.4f}</td><td class="num">{test}</td></tr>')


def main():
    parts = []
    for k, (title, data, validation, items) in enumerate(T.PHASES, 1):
        heading, section = SECTION[k]
        parts.append(f'<h2 id="phase-{k}" data-anchor="phase-{k}">{heading}</h2>\n<p class="sub">{data} · {validation} · '
                     f'<a href="overview.html#phase-{k}">phase {k} in the report</a></p>')
        for number, what, runs, records, dates, result, outcome in items:
            recs = [T.run_record(r) for r in runs]
            when = T.span(recs) if recs else dates
            rows = ''.join(run_row(r) for r in recs)
            table = ('<div class="table-wrap"><table><thead><tr><th>Run (configuration file)</th><th>Start → end (JST)</th>'
                     '<th class="num">GPU-h</th><th>Selected model, LR, epoch</th><th class="num">Validation</th>'
                     f'<th class="num">Test</th></tr></thead><tbody>{rows}</tbody></table></div>') if rows else ''
            extra = ' · '.join(link(m) for m in records)
            gpu = sum(r['gpu_hours'] for r in recs)
            meta = ' · '.join(x for x in (f'{OUTCOME[outcome]}', (f'{gpu:.0f} GPU-h' if gpu >= 1 else '&lt;1 GPU-h') if recs else '',
                                          f'records: {extra}' if extra else '') if x)
            parts.append(f'<details class="trial" id="t{number}"><summary><span class="no" data-anchor="t{number}">{number}</span>'
                         f'<span class="when">{when}</span><span class="what">{text(what)}</span>'
                         f'<span class="res">{text(result)}</span></summary>'
                         f'<div class="body">{table}<p>{meta} · <a href="overview.html#t{number}">trial {number} in the report</a>'
                         '</p></div></details>')
    page = TEMPLATE.replace('<!--TRIALS-->', '\n'.join(parts))
    import hashlib
    for name in ('style.css', 'anchors.js'):  # content hash in the URL: no stale cached copies
        h = hashlib.sha256((ROOT / 'docs/_extra/pub' / name).read_bytes()).hexdigest()[:12]
        page = page.replace(f'pub/{name}"', f'pub/{name}?v={h}"')
    out = ROOT / 'docs/_extra/log.html'
    out.write_text(page)
    print(f'{out.relative_to(ROOT)}: {len(page) // 1024} kB')


TEMPLATE = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Experiment Log</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Lato:wght@400;700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">
<link rel="stylesheet" href="pub/style.css">
<script defer src="pub/anchors.js"></script>
</head>
<body>
<div class="page">
<div class="site wide"><a href="index.html">ILC dynamic training</a> &nbsp;/&nbsp; <a href="overview.html">Research report</a> &nbsp;/&nbsp; Experiment log</div>
<header class="title wide" style="padding-block:48px 8px">
  <h1 style="font-size:38px">Experiment log</h1>
  <p class="dek">Every trial from September to November 2026, with its runs, dates, GPU time, selected model and
  configuration files. The summary and interpretation are in the <a href="overview.html">report</a>.</p>
</header>
<main class="log wide">
<p style="font:15px/1.55 var(--sans);color:var(--muted);max-width:780px">Click a trial to show its runs. Dates are
JST. Results are the mistag on the Test sample (2.8M jets) unless marked Validation. The selected model is the one
with the best Validation mistag over the run; for standard ParT the LR shown is the start LR, otherwise the LR at the
selected epoch. July–August was set-up with an older training method and is not listed.</p>
<!--TRIALS-->
</main>
<footer class="end wide">Generated by scripts/reports/build_log.py from the run manifests. Detailed results:
<a href="results.html">results</a>; frozen test specifications:
<a href="https://github.com/marchexe/ilc-dynamic-training/tree/main/configs/prepared">configs/prepared</a>.</footer>
</div>
</body>
</html>
'''

if __name__ == '__main__':
    main()
