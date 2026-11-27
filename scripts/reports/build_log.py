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
import build_report as R  # noqa: E402  (trial descriptions shared with the report)
from report_cards import CARD  # noqa: E402,F401

# trial number -> (configuration, result, interpretation), exactly as in the research report
REPORT = {n: (d, c, r, i) for _, _, rows in R.PHASES for n, d, c, r, i in rows if n != '–'}

ROOT = T.ROOT
REPO = 'https://github.com/marchexe/ilc-dynamic-training/blob/main/'
REF_PAGE = {'sub-epoch-restart-test': 'results', 'large-dataset-results': 'results', 'sub-epoch-adaptation': 'ongoing',
            'pbt-vs-standard-part': 'results', 'pbt-vs-standard-part-8m': 'results', 'probe-scored-pbt': 'results',
            'prediction-ensemble': 'results', 'ensemble-800k': 'results', 'ensemble-diversity': 'results',
            'dropout-branch': 'ongoing', 'distillation': 'results'}
SECTION = {  # step of export_report_trials -> (heading, section of the overview)
    1: ('PBT against constant learning rates', 'baseline'),
    2: ('Higher constant learning rates', 'baseline'),
    3: ('PBT against standard ParT, Training (800k)', 'small'),
    4: ('PBT against standard ParT, Training (8M)', 'large'),
    5: ('Ensembles and other ways to gain, Training (8M)', 'ensemble'),
}


def link(markup):
    role, label, target = re.fullmatch(r':(\w+):`([^<]+) <([^>]+)>`', markup).groups()
    label = html.escape(label.strip())
    if role == 'repo':
        return f'<a href="{REPO}{target}">{label}</a>'
    if role == 'doc':
        return f'<a href="{target}.html">{label}</a>'
    return f'<a href="{REF_PAGE[target]}.html#{target}">{label}</a>'


def run_row(r):
    gpu = '&lt;1' if r['gpu_hours'] < 1 else f"{r['gpu_hours']:.0f}"
    test = '–' if r['test'] is None else f"{r['test']:.4f}"
    return (f'<tr><td><a href="{REPO}{r["config"]}">{r["run"]}</a></td>'
            f'<td class="nowrap">{r["start"]:%b %d %H:%M} → {r["end"]:%b %d %H:%M}</td>'
            f'<td class="num">{gpu}</td><td>{r["member"]}, {T.shown_lr(r)}, epoch {r["epoch"]}</td>'
            f'<td class="num">{r["validation"]:.4f}</td><td class="num">{test}</td></tr>')


def ensemble_models():
    """#27/#28: run, member, checkpoint and Test value of both models of each seed's ensemble (from figure_data.json)."""
    import json
    d = json.loads(R.FIGDATA.read_text())['overview']['ensemble']['seeds']
    rows = []
    for s in d:
        for m, v, lr in ((s['model_3e3'], s['better'], '3×10⁻³'), (s['model_1e3'], s['other'], '10⁻³')):
            cfg = T.CONFIG.get(m['run']) or f'configs/experiments/{m["run"]}.yaml'
            rows.append(f'<tr><td>{s["seed"]} (#{s["trial"]})</td><td>{lr}</td><td><a href="{REPO}{cfg}">{m["run"]}</a></td>'
                        f'<td>{m["member"]}</td><td class="num">{m["epoch"] - T.FIRST_EPOCH}</td>'
                        f'<td><code>net_epoch-{m["epoch"]}_state.pt</code></td><td class="num">{v:.4f}</td></tr>')
    return ('<div class="table-wrap"><table><thead><tr><th>Seed</th><th>Start LR</th><th>Run (configuration file)</th>'
            '<th>Member</th><th class="num">Epoch</th><th>Checkpoint file</th><th class="num">Test</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div><p>Ensemble models: the checkpoint of each model with the best '
            'Validation (1M) value. Epoch: training epoch (the file name counts from the pretrained start, epoch 17). '
            'Records: <code>runs/eval/holdout_ensemble_8m_dev_v1</code>, <code>runs/eval/ensemble_seed22345_v1</code>, '
            '<code>runs/eval/ensemble_seed32345_v1</code> (result.json, with SHA-256 of every checkpoint).</p>')


def main():
    import json
    figdata = json.loads(R.FIGDATA.read_text())
    self_ref = 'Run names, members and checkpoint files are in the <a href="log.html#t28">experiment log</a>.'
    extra_tables = {18: R.table_18(figdata),
                    28: R.table_28(figdata).replace(self_ref, 'Run names, members and checkpoint files: next table.')
                    + ensemble_models()}
    parts = []
    for k, (title, data, validation, items) in enumerate(T.PHASES, 1):
        heading, section = SECTION[k]
        parts.append(f'<h2 id="phase-{k}" data-anchor="phase-{k}">{heading}</h2>\n<p class="sub">{data} · {validation} · '
                     f'<a href="overview.html#phase-{k}">phase {k} in the report</a></p>')
        for number, what, runs, records, dates, result, outcome in items:
            recs = [T.run_record(r) for r in runs]
            rows = ''.join(run_row(r) for r in recs)
            table = ('<div class="table-wrap"><table><thead><tr><th>Run (configuration file)</th><th>Start → end (JST)</th>'
                     '<th class="num">GPU-h</th><th>Selected model, LR, epoch</th><th class="num">Validation</th>'
                     f'<th class="num">Test</th></tr></thead><tbody>{rows}</tbody></table></div>') if rows else ''
            extra = ' · '.join(link(m) for m in records)
            gpu = sum(r['gpu_hours'] for r in recs)
            report_date, what_html, result_html, interp_html = REPORT[str(number)]
            when = T.span(recs) if recs else report_date  # trials without runs: the date of the report
            meta = ' · '.join(x for x in ((f'{gpu:.0f} GPU-h' if gpu >= 1 else '&lt;1 GPU-h') if recs else '',
                                          f'records: {extra}' if extra else '') if x)
            name, change, value, sample, versus = CARD[str(number)]
            smp = f' <span class="smp">{sample}</span>' if sample else ''
            parts.append(f'<details class="trial" id="t{number}"><summary><span class="no" data-anchor="t{number}">{number}</span>'
                         f'<span class="when">{when}</span><span class="what"><b>{name}</b></span>'
                         f'<span class="res"><b>{value}</b>{smp} {versus}</span></summary>'
                         f'<div class="body"><p><b>Change.</b> {change}</p><p><b>Configuration.</b> {what_html}</p>'
                         f'<p><b>Result.</b> {result_html}</p><p class="interp"><b>Interpretation.</b> {interp_html}</p>'
                         f'{table}{extra_tables.get(number, "")}<p>{meta} · <a href="overview.html#t{number}">trial {number} in the report</a>'
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
<style>
nav.lognav { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 16px; font: 14px/1.4 var(--sans);
  padding: 10px 0; border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule); margin: 8px 0 4px; }
nav.lognav a { color: var(--link); text-decoration: none; }
nav.lognav a:hover { text-decoration: underline; }
nav.lognav input { margin-left: auto; font: 14px var(--sans); padding: 4px 8px; border: 1px solid var(--rule);
  border-radius: 4px; min-width: 15em; color: var(--ink); background: #fff; }
@media (max-width: 760px) { nav.lognav input { margin-left: 0; width: 100%; } }
table.cmp { font: 13.8px/1.4 var(--sans); border-collapse: collapse; margin: 10px 0 2px; min-width: 760px; }
table.cmp caption { text-align: left; font-weight: 700; color: var(--ink); padding-bottom: 6px; }
table.cmp .iv { display: block; font-size: 12.5px; color: var(--muted); }
table.cmp tr.mean td { border-top: 1.5px solid var(--ink); }
p.tnote { font: 13px/1.45 var(--sans); color: var(--muted); }
</style>
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
<nav class="lognav" aria-label="Phases and search">
  <a href="#phase-1">1 Constant LRs</a><a href="#phase-2">2 Higher constant LRs</a><a href="#phase-3">3 Training (800k)</a>
  <a href="#phase-4">4 Training (8M)</a><a href="#phase-5">5 Ensembles</a>
  <input type="search" id="logsearch" placeholder="Filter trials…" aria-label="Filter trials">
</nav>
<!--TRIALS-->
<p id="nomatch" hidden>No trial matches.</p>
</main>
<script>
(function () {  // filter trials by text; open the matching ones so the matching runs are visible
  var box = document.getElementById('logsearch'), none = document.getElementById('nomatch');
  if (!box) return;
  var trials = Array.prototype.slice.call(document.querySelectorAll('details.trial'));
  var heads = Array.prototype.slice.call(document.querySelectorAll('main.log h2, main.log p.sub'));
  box.addEventListener('input', function () {
    var q = box.value.trim().toLowerCase(), shown = 0;
    trials.forEach(function (t) {
      var hit = !q || t.textContent.toLowerCase().indexOf(q) >= 0;
      t.hidden = !hit; if (hit) shown++;
      if (q && hit) t.open = true;
    });
    heads.forEach(function (h) { h.hidden = !!q; });
    none.hidden = shown > 0;
  });
})();
</script>
<footer class="end wide">Generated by scripts/reports/build_log.py from the run manifests. Detailed results:
<a href="results.html">results</a>; frozen test specifications:
<a href="https://github.com/marchexe/ilc-dynamic-training/tree/main/configs/prepared">configs/prepared</a>.</footer>
</div>
</body>
</html>
'''

if __name__ == '__main__':
    main()
