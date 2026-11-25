#!/usr/bin/env python3
"""Build the interactive overview page of the project: docs/_extra/overview.html.

    .venv/bin/python scripts/reports/build_report_page.py

The page is self-contained (no external scripts) and is copied to the site root by Sphinx (html_extra_path).
Trials, runs, dates and selected models come from export_report_trials.py (run manifests); learning-rate and
Validation trajectories come from the run manifests; loss curves from published/results/sel1m/train8m_followups.json.
"""
import json
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


def trajectory(run, members=None):
    """Per training epoch: LR of each member, and the best Validation value so far over the members."""
    m = json.loads((RUNS / run / 'manifest.json').read_text())
    out, best = [], None
    for g in m['generations']:
        ws = {k: w for k, w in g['workers'].items() if members is None or k in members}
        vals = [w['metrics'][PRIMARY] for w in ws.values() if (w.get('metrics') or {}).get(PRIMARY) is not None]
        if vals:
            best = min(vals) if best is None else min(best, min(vals))
        out.append(dict(e=g['epoch'] - trials_mod.FIRST_EPOCH, lr=sorted(w['lr'] for w in ws.values()),
                        best=round(best, 5)))
    return out


def main():
    loss = json.loads((ROOT / 'published/results/sel1m/train8m_followups.json').read_text())['loss_curves']
    data = dict(
        phases=trials(),
        traj={
            '8M': dict(standard=trajectory('sel1m_train8m_standard_part_3e-3'),
                       pbt=trajectory('sel1m_train8m_fast_pbt'),
                       fixed=trajectory('sel1m_train8m_fast_pbt_probe_decay')),
            '800k': dict(standard=trajectory('sel1m_standard_part', ['flat_decay_100e-6']),
                         pbt=trajectory('sel1m_fast_pbt')),
        },
        loss={k: [dict(e=p['epoch'] - trials_mod.FIRST_EPOCH, t=p['train'], v=p['selection']) for p in v['epochs']]
              for k, v in loss.items()},
    )
    page = TEMPLATE.replace('/*DATA*/null', json.dumps(data, ensure_ascii=False, separators=(',', ':')))
    out = ROOT / 'docs/_extra/overview.html'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f'{out.relative_to(ROOT)}: {len(page) // 1024} kB')


TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PBT Study Overview</title>
<style>
:root{--bg:#f6f6f4;--card:#ffffff;--ink:#191919;--ink2:#5f5f5f;--ink3:#8a8a8a;--line:#e4e4e0;--grid:#ececea;
--fixed:#257de0;--pbt:#f47230;--std:#00a383;--ens:#6a4ce2;--good:#0a8a1f;--bad:#d03b3b;--neutral:#8a8a8a;--tint:#f0f4fb}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#111111;--card:#1b1b1b;--ink:#efefef;--ink2:#b0b0b0;
--ink3:#808080;--line:#2e2e2e;--grid:#262626;--fixed:#4b9bf0;--pbt:#f07a40;--std:#1fbf9b;--ens:#9887ef;--good:#2fbf45;--bad:#ef6363;
--neutral:#9a9a9a;--tint:#1a2230}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
a{color:var(--fixed)}
.wrap{max-width:1120px;margin:0 auto;padding:28px 20px 60px}
header h1{font-size:28px;line-height:1.2;margin:0 0 6px}
header p{margin:0;color:var(--ink2);max-width:780px}
.chip{display:inline-block;font-size:12px;padding:2px 9px;border-radius:99px;background:var(--card);border:1px solid var(--line);color:var(--ink2);margin:10px 6px 0 0}
h2{font-size:20px;margin:42px 0 4px}
.lead{color:var(--ink2);margin:0 0 16px;max-width:820px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px}
.heroes{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:22px}
.hero .num{font-size:38px;font-weight:700;line-height:1.1;letter-spacing:-.5px}
.hero .what{font-weight:600;margin-top:4px}
.hero .sub{color:var(--ink2);font-size:13px;margin-top:4px}
.hero.good .num{color:var(--good)} .hero.bad .num{color:var(--bad)}
.answer{margin-top:14px;font-size:16px}
.steps{position:relative;margin-top:8px}
.step{display:grid;grid-template-columns:150px 1fr;gap:20px;position:relative;padding-bottom:18px}
.step:before{content:"";position:absolute;left:150px;top:0;bottom:0;border-left:2px solid var(--line);transform:translateX(9px)}
.step:last-child:before{bottom:auto;height:20px}
.when{text-align:right;padding-top:16px;color:var(--ink2);font-size:13px}
.when b{display:block;color:var(--ink);font-size:15px}
.step .card{position:relative;margin-left:22px}
.dot{position:absolute;left:-31px;top:18px;width:18px;height:18px;border-radius:50%;border:3px solid var(--card);box-shadow:0 0 0 2px var(--line)}
.step h3{margin:0 0 8px;font-size:17px}
.cols{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}
.cols div h4{margin:0 0 3px;font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:var(--ink3)}
.cols div p{margin:0;font-size:14px}
.badge{display:inline-flex;align-items:center;gap:5px;font-size:12px;font-weight:600;padding:2px 9px;border-radius:99px;margin-left:8px;vertical-align:2px}
.badge.good{background:color-mix(in srgb,var(--good) 14%,transparent);color:var(--good)}
.badge.bad{background:color-mix(in srgb,var(--bad) 14%,transparent);color:var(--bad)}
.badge.warn{background:color-mix(in srgb,#d28e05 16%,transparent);color:#b07400}
.mini{margin-top:12px}
.tlinks{margin-top:10px;font-size:12px;color:var(--ink3)}
.tlinks a{margin-right:6px;text-decoration:none;border:1px solid var(--line);border-radius:6px;padding:1px 7px;color:var(--ink2)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.chart-title{font-weight:600;margin:0 0 2px;font-size:14px}
.chart-sub{color:var(--ink2);font-size:12px;margin:0 0 6px}
svg text{fill:var(--ink2);font-size:11px}
svg .lbl{fill:var(--ink);font-size:12px}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:var(--ink2);margin-top:4px}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:5px;vertical-align:-1px}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--card);font-size:12px;padding:7px 10px;border-radius:8px;max-width:320px;opacity:0;transition:opacity .1s;z-index:9}
.filters{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}
.filters button{font:inherit;font-size:13px;border:1px solid var(--line);background:var(--card);color:var(--ink2);border-radius:99px;padding:3px 12px;cursor:pointer}
.filters button.on{background:var(--ink);color:var(--card);border-color:var(--ink)}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:var(--ink3);font-weight:600;padding:8px;border-bottom:1px solid var(--line)}
td{padding:8px;}td:first-child{white-space:nowrap}td{border-bottom:1px solid var(--line);vertical-align:top}
tr.trial{cursor:pointer} tr.trial:hover td{background:var(--tint)}
tr.phase td{background:transparent;font-weight:600;padding-top:18px;border-bottom:2px solid var(--line)}
tr.phase span{font-weight:400;color:var(--ink2);margin-left:8px;font-size:12.5px}
.oc{font-weight:600;white-space:nowrap} .oc.better{color:var(--good)} .oc.worse{color:var(--bad)} .oc.none,.oc.setup{color:var(--ink3)} .oc.invalid{color:#b07400}
.runs td{background:var(--tint);font-size:12.5px}
.runs table td,.runs table th{background:transparent;border-bottom:1px solid var(--line);padding:5px 6px}
.caret{display:inline-block;width:12px;color:var(--ink3)}
.gloss{display:grid;grid-template-columns:repeat(2,1fr);gap:8px 24px;font-size:14px}
.gloss b{display:block}
.gloss span{color:var(--ink2)}
footer{margin-top:40px;color:var(--ink3);font-size:12.5px}
@media (max-width:820px){.heroes,.cols,.grid2,.gloss{grid-template-columns:1fr}.step{grid-template-columns:1fr}.step:before{display:none}
.when{text-align:left;padding:0}.step .card{margin-left:0}.dot{display:none}.hide-s{display:none}}
</style>
</head>
<body>
<div id="tip"></div>
<div class="wrap">
<header>
  <h1>Does PBT give a better jet tagger? What we did from July to November 2026</h1>
  <p>Population Based Training (PBT) changes the learning rate (LR) of a pretrained Particle Transformer while it
  trains. We compared it with normal training on ILC jets (b, c and light quarks). Lower mistag is better; all
  numbers below are on Test data that was used only after each model was selected.</p>
  <span class="chip">No run in progress</span><span class="chip">Updated 2026-11-26</span>
  <span class="chip"><a href="report.html">Text version</a></span><span class="chip"><a href="results.html">Full results</a></span>
</header>

<div class="heroes">
  <div class="card hero good"><div class="num">+0.81%</div><div class="what">PBT is better than normal training</div>
    <div class="sub">little data (800k training jets) · 3 seeds · 95% interval +0.34 to +1.34%</div></div>
  <div class="card hero bad"><div class="num">−2.7%</div><div class="what">PBT is worse than normal training</div>
    <div class="sub">much data (8M training jets) · 1 seed · after two fixes; −12.4% without them</div></div>
  <div class="card hero good"><div class="num">+5.0%</div><div class="what">An ensemble of two normal models is better</div>
    <div class="sub">much data (8M) · 3 seeds · 95% interval +4.2 to +5.8% · no PBT needed</div></div>
</div>
<p class="answer"><b>In one line:</b> PBT helps only a little and only with little data. With much data, the mean of two
normally trained models helps, and PBT does not.</p>

<h2>What happened, step by step</h2>
<p class="lead">Each step says what we did, what we saw, and why we did the next step. The numbers in grey boxes are
trial numbers; click one to open it in the list at the end.</p>
<div class="steps" id="steps"></div>

<h2>Why more data changes the answer</h2>
<p class="lead">With little data the model starts to memorise its training data after about 20 epochs, so an early
lower LR is right and PBT finds it. With much data the model keeps learning; the best plan is a high LR for a long
time and a decay at the end. PBT judges each LR change after two epochs, where a lower LR almost always looks
better, so it lowers the LR too early.</p>
<div class="grid2">
  <div class="card"><p class="chart-title">Little data (800k): the model memorises</p>
    <p class="chart-sub">Loss of the best normal model per epoch. The training loss falls below the Validation loss.</p>
    <div id="loss800"></div></div>
  <div class="card"><p class="chart-title">Much data (8M): it does not</p>
    <p class="chart-sub">Same plot. The two losses stay together to the end.</p><div id="loss8"></div></div>
  <div class="card"><p class="chart-title">8M: learning rate per epoch</p>
    <p class="chart-sub">Normal training keeps 3×10⁻³ for 35 epochs. PBT (band = its five models) is below 2.5×10⁻⁴ from epoch 11.</p>
    <div id="lr8"></div></div>
  <div class="card"><p class="chart-title">8M: best Validation mistag so far</p>
    <p class="chart-sub">PBT leads early, normal training wins in its final decay (epochs 36–50).</p><div id="val8"></div></div>
</div>

<h2>Every comparison</h2>
<p class="lead">Right of zero: the first named method is better. Hatched: not a valid comparison, because the LRs it was
compared with were too low. Hover a bar for details.</p>
<div class="card"><div id="allcmp"></div>
<div class="legend"><span><i style="background:var(--good)"></i>better</span><span><i style="background:var(--bad)"></i>worse</span>
<span><i style="background:var(--neutral)"></i>no clear difference</span><span><i style="background:repeating-linear-gradient(45deg,var(--neutral) 0 2px,transparent 2px 5px);border:1px solid var(--neutral)"></i>not valid</span>
<span>— 95% interval</span></div></div>

<h2 id="trials">All 33 trials and 42 runs, by step</h2>
<p class="lead">Click a trial to see its runs: exact start and end (JST), GPU hours, the selected model and links to the
configuration files. Validation is 150k jets until trial 17, then 1M jets.</p>
<div class="filters" id="filters"></div>
<div class="card" style="padding:6px 10px"><table id="tbl"><thead><tr><th>#</th><th class="hide-s">Dates (JST)</th><th>What</th><th>Result</th><th>Outcome</th></tr></thead><tbody></tbody></table></div>

<h2>Words used</h2>
<div class="card gloss">
  <div><b>Training (800k), (8M)</b><span>Training data: 2.4M and 24M jets (source entries below 800k and 8M of each file).</span></div>
  <div><b>Validation (150k), (1M)</b><span>Checked after each epoch; selects the model and drives PBT. 1M from November 10, because 150k was too noisy.</span></div>
  <div><b>Test (2.8M), (9.0M)</b><span>Used once per selected model, after selection. Earlier names: "holdout" and "sealed test".</span></div>
  <div><b>Mistag</b><span>Mean background mistag rate (%) at eight working points of b- and c-tagging. Lower is better.</span></div>
  <div><b>Normal training (standard ParT)</b><span>Constant LR for 35 epochs, then a decay by ×0.01 over the last 15 epochs; five start LRs, the best one is kept.</span></div>
  <div><b>PBT</b><span>Five models train in parallel. Every two epochs a weak one copies the weights of a strong one and changes its LR.</span></div>
  <div><b>Ensemble</b><span>The mean of the outputs of two models.</span></div>
  <div><b>Seed</b><span>The random start of a run. Three seeds = the same experiment three times.</span></div>
</div>
<footer>Limits: the 8M PBT results have one seed; all trials start from one pretrained model; Test (2.8M) was also used in
diagnostics before November, so it is not fully unseen. Generated by scripts/reports/build_report_page.py from the run records.</footer>
</div>

<script>
const D=/*DATA*/null;
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const NS='http://www.w3.org/2000/svg';
function el(tag,attrs,parent){const e=document.createElementNS(NS,tag);for(const k in attrs)e.setAttribute(k,attrs[k]);if(parent)parent.appendChild(e);return e}
const tip=document.getElementById('tip');
function hover(node,html){node.addEventListener('mousemove',ev=>{tip.innerHTML=html;tip.style.opacity=1;
  const x=Math.min(ev.clientX+14,innerWidth-330);tip.style.left=x+'px';tip.style.top=(ev.clientY+14)+'px'});
  node.addEventListener('mouseleave',()=>tip.style.opacity=0)}
const sup=s=>s.replace(/e-0?(\d+)/,(m,d)=>'×10⁻'+[...d].map(c=>'⁰¹²³⁴⁵⁶⁷⁸⁹'[c]).join(''));
const fmtLR=v=>{const e=Math.floor(Math.log10(v)),m=v/10**e;return (Math.abs(m-1)<1e-9?'':(+m.toFixed(2))+'×')+'10'+('⁻'+[...String(-e)].map(c=>'⁰¹²³⁴⁵⁶⁷⁸⁹'[c]).join(''))};

/* ---------- horizontal bar chart (improvement %, + = better) ---------- */
function bars(host,rows,opt={}){
  const W=host.clientWidth||900,lw=opt.lw??(W<600?150:300),rh=opt.rh??30,pad=10,H=rows.length*rh+34;
  const lo=opt.lo??Math.min(0,...rows.map(r=>r.ci?r.ci[0]:r.v))-1,hi=opt.hi??Math.max(0,...rows.map(r=>r.ci?r.ci[1]:r.v))+2.5;
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,width:'100%'},host);
  const defs=el('defs',{},svg),pat=el('pattern',{id:'h'+host.id,width:6,height:6,patternUnits:'userSpaceOnUse',patternTransform:'rotate(45)'},defs);
  el('rect',{width:6,height:6,fill:'transparent'},pat);el('line',{x1:0,y1:0,x2:0,y2:6,stroke:css('--neutral'),'stroke-width':2.5},pat);
  const x=v=>lw+(v-lo)/(hi-lo)*(W-lw-pad);
  for(let t=Math.ceil(lo/2)*2;t<=hi;t+=2){el('line',{x1:x(t),x2:x(t),y1:4,y2:H-24,stroke:css('--grid')},svg);
    el('text',{x:x(t),y:H-8,'text-anchor':'middle'},svg).textContent=(t>0?'+':'')+t+'%'}
  el('line',{x1:x(0),x2:x(0),y1:2,y2:H-24,stroke:css('--ink2'),'stroke-width':1.2},svg);
  rows.forEach((r,i)=>{const y=8+i*rh;
    if(r.sep)el('line',{x1:8,x2:W-pad,y1:y-4,y2:y-4,stroke:css('--line'),'stroke-dasharray':'3 3'},svg);
    const t=el('text',{x:lw-10,y:y+rh/2,'text-anchor':'end','dominant-baseline':'middle',class:'lbl'},svg);t.textContent=r.label;
    const c=r.kind==='invalid'?`url(#h${host.id})`:css(r.kind==='better'?'--good':r.kind==='worse'?'--bad':'--neutral');
    const x0=Math.min(x(0),x(r.v)),w=Math.max(Math.abs(x(r.v)-x(0)),2);
    const b=el('rect',{x:x0,y:y+5,width:w,height:rh-12,rx:4,fill:c,stroke:r.kind==='invalid'?css('--neutral'):'none'},svg);
    if(r.ci){const yy=y+rh/2-1;el('line',{x1:x(r.ci[0]),x2:x(r.ci[1]),y1:yy,y2:yy,stroke:css('--ink'),'stroke-width':1.8},svg);
      for(const v of r.ci)el('line',{x1:x(v),x2:x(v),y1:yy-5,y2:yy+5,stroke:css('--ink'),'stroke-width':1.8},svg)}
    const edge=r.ci?(r.v>=0?r.ci[1]:r.ci[0]):r.v;
    const vt=el('text',{x:x(edge)+(r.v>=0?6:-6),y:y+rh/2,'text-anchor':r.v>=0?'start':'end','dominant-baseline':'middle',class:'lbl'},svg);
    vt.textContent=(r.v>0?'+':'')+r.v.toFixed(2)+'%';
    const hit=el('rect',{x:0,y:y,width:W,height:rh,fill:'transparent'},svg);if(r.tip)hover(hit,r.tip);
  });
}

/* ---------- line chart over epochs ---------- */
function lines(host,series,opt){
  const W=host.clientWidth||500,H=opt.h||230,L=54,R=12,T=10,B=30;
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,width:'100%'},host);
  const xs=series.flatMap(s=>s.pts.map(p=>p[0])),x0=Math.min(...xs),x1=Math.max(...xs);
  const ys=series.flatMap(s=>s.pts.flatMap(p=>p.length>2?[p[1],p[2]]:[p[1]]));
  const lg=!!opt.log,f=v=>lg?Math.log10(v):v,y0=opt.y0??Math.min(...ys.map(f)),y1=opt.y1??Math.max(...ys.map(f));
  const X=v=>L+(v-x0)/(x1-x0)*(W-L-R),Y=v=>T+(1-(f(v)-y0)/(y1-y0))*(H-T-B);
  const ticks=opt.yt||[];for(const t of ticks){el('line',{x1:L,x2:W-R,y1:Y(t),y2:Y(t),stroke:css('--grid')},svg);
    el('text',{x:L-6,y:Y(t),'text-anchor':'end','dominant-baseline':'middle'},svg).textContent=opt.yf?opt.yf(t):t}
  for(const t of opt.xt||[10,20,30,40,50]){if(t<x0||t>x1)continue;el('text',{x:X(t),y:H-10,'text-anchor':'middle'},svg).textContent=t}
  el('text',{x:L,y:H-10,'text-anchor':'start'},svg).textContent='epoch';
  if(opt.shade)el('rect',{x:X(opt.shade[0]),y:T,width:X(opt.shade[1])-X(opt.shade[0]),height:H-T-B,fill:css('--tint')},svg);
  for(const s of series){
    if(s.band){const top=s.pts.map(p=>`${X(p[0])},${Y(p[2])}`),bot=s.pts.slice().reverse().map(p=>`${X(p[0])},${Y(p[1])}`);
      el('polygon',{points:top.concat(bot).join(' '),fill:css(s.color),opacity:.25},svg)}
    else el('polyline',{points:s.pts.map(p=>`${X(p[0])},${Y(p[1])}`).join(' '),fill:'none',stroke:css(s.color),'stroke-width':2.2,'stroke-dasharray':s.dash||''},svg)}
  const leg=document.createElement('div');leg.className='legend';
  leg.innerHTML=series.map(s=>`<span><i style="background:${css(s.color)};${s.band?'opacity:.45':''}"></i>${s.name}</span>`).join('');host.appendChild(leg);
  /* hover: nearest epoch */
  const hit=el('rect',{x:L,y:T,width:W-L-R,height:H-T-B,fill:'transparent'},svg),guide=el('line',{y1:T,y2:H-B,stroke:css('--ink3'),opacity:0},svg);
  hit.addEventListener('mousemove',ev=>{const r=svg.getBoundingClientRect(),px=(ev.clientX-r.left)*W/r.width;
    const e=Math.round(x0+(px-L)/(W-L-R)*(x1-x0));guide.setAttribute('x1',X(e));guide.setAttribute('x2',X(e));guide.setAttribute('opacity',.6);
    tip.innerHTML=`<b>epoch ${e}</b><br>`+series.map(s=>{const p=s.pts.find(q=>q[0]===e);if(!p)return'';
      const v=s.band?`${opt.yf(p[1])} – ${opt.yf(p[2])}`:(opt.vf||opt.yf||String)(p[1]);return `${s.name}: ${v}`}).filter(Boolean).join('<br>');
    tip.style.opacity=1;tip.style.left=Math.min(ev.clientX+14,innerWidth-330)+'px';tip.style.top=(ev.clientY+14)+'px'});
  hit.addEventListener('mouseleave',()=>{tip.style.opacity=0;guide.setAttribute('opacity',0)});
}

/* ---------- steps ---------- */
const STEPS=[
 {when:'Sep 15 – Nov 2',n:1,title:'PBT seems to win',badge:['warn','not valid, see step 2'],color:'--pbt',
  did:'Compared PBT with training at a constant LR (five LRs, 3×10⁻⁶ to 1.4×10⁻⁵), on 800k and then on 8M training jets.',
  saw:'PBT better by 1.4% (800k) and by 7.7% (8M, preregistered, Test 9.0M).',
  next:'In both runs the best constant LR was the highest one, and PBT had moved above the range. So: are the constant LRs simply too low?',
  trials:[1,2,3,4,6,10],chart:[{label:'800k',v:1.36,kind:'invalid'},{label:'8M',v:7.67,kind:'invalid'}]},
 {when:'Nov 2 – Nov 9',n:2,title:'The win was not real',badge:['bad','PBT = good constant LR'],color:'--fixed',
  did:'Trained constant LRs up to 3.2×10⁻⁴ (800k), and a faster PBT with a wider LR range; then a second seed.',
  saw:'A constant LR of 8×10⁻⁵ (6× higher) is as good as PBT. With seed 22345 the order reverses (PBT +0.5% worse).',
  next:'PBT won because it reached higher LRs. The 150k Validation set was too noisy to pick models. Next: 1M Validation, and compare with normal ParT training.',
  trials:[12,13,14,15,16,17],chart:[{label:'seed 12345',v:1.00,kind:'better'},{label:'seed 22345',v:-0.47,kind:'worse'}]},
 {when:'Nov 9 – Nov 13',n:3,title:'Fair test with little data: PBT wins a little',badge:['good','+0.81%, 3 seeds'],color:'--pbt',
  did:'PBT against normal training (standard ParT) and a constant-LR grid, 800k training jets, 1M Validation, three seeds.',
  saw:'PBT better on every seed: +0.67%, +0.85%, +0.91%. The constant LRs are not stable across seeds.',
  next:'A small but real gain. Does it hold with ten times more data?',
  trials:[18,19,20,21],chart:[{label:'seed 12345',v:0.67,kind:'better'},{label:'seed 22345',v:0.85,kind:'better'},{label:'seed 32345',v:0.91,kind:'better'},{label:'mean',v:0.81,ci:[0.34,1.34],kind:'better'}]},
 {when:'Nov 13 – Nov 23',n:4,title:'Much data: PBT loses',badge:['bad','−2.7% after two fixes'],color:'--bad',
  did:'Same comparison on 8M training jets. Then two fixes to PBT: judge LRs after a short cool-down, and add the final decay of normal training.',
  saw:'PBT 12.4% worse. The fixes reduce it to 5.4% and 2.7%, but PBT stays worse.',
  next:'The model does not memorise at 8M (see the figures below), so a long high LR wins. PBT\'s two-epoch view cannot see that. Stop PBT here; is there another way to gain?',
  trials:[22,23,24,25,26],chart:[{label:'PBT',v:-12.4,kind:'worse'},{label:'+ fix 1',v:-5.4,kind:'worse'},{label:'+ fix 1 + fix 2',v:-2.7,kind:'worse'}]},
 {when:'Nov 23 – Nov 25',n:5,title:'Much data: an ensemble wins',badge:['good','+5.0%, 3 seeds'],color:'--ens',
  did:'Took the mean of the outputs of two normally trained models (start LRs 3×10⁻³ and 1×10⁻³). Fixed the test in advance, then repeated on two new seeds.',
  saw:'+5.05%, +4.86%, +5.14% against the better single model. Two PBT models give no gain (PBT copies weights, so they make the same errors). Distillation into one model and weight averaging: no gain. At 800k: no gain.',
  next:'The usable result of the project. It needs two models at inference time.',
  trials:[27,28,29,30,31,32,33],chart:[{label:'seed 12345',v:5.05,kind:'better'},{label:'seed 22345',v:4.86,kind:'better'},{label:'seed 32345',v:5.14,kind:'better'},{label:'mean',v:5.01,ci:[4.17,5.84],kind:'better'}]},
];
const sh=document.getElementById('steps');
STEPS.forEach((s,i)=>{const d=document.createElement('div');d.className='step';
  d.innerHTML=`<div class="when"><b>Step ${s.n}</b>${s.when}</div><div class="card"><span class="dot" style="background:${css(s.color)}"></span>
  <h3>${s.title}<span class="badge ${s.badge[0]}">${s.badge[1]}</span></h3>
  <div class="cols"><div><h4>What we did</h4><p>${s.did}</p></div><div><h4>What we saw</h4><p>${s.saw}</p></div><div><h4>So next</h4><p>${s.next}</p></div></div>
  <div class="mini" id="mini${i}"></div>
  <div class="tlinks">Trials ${s.trials.map(t=>`<a href="#t${t}" data-t="${t}">${t}</a>`).join('')}</div></div>`;sh.appendChild(d)});
requestAnimationFrame(()=>STEPS.forEach((s,i)=>bars(document.getElementById('mini'+i),s.chart,{lw:120,rh:26,lo:s.n===4?-14:Math.min(0,...s.chart.map(r=>r.v))-1,hi:s.n===4?2:Math.max(...s.chart.map(r=>r.ci?r.ci[1]:r.v))+2})));

/* ---------- why-figures ---------- */
const fl=v=>v.toFixed(3);
function lossChart(id,pts,y0,y1,yt){lines(document.getElementById(id),[
  {name:'training data',color:'--ink2',pts:pts.map(p=>[p.e,p.t])},{name:'Validation (not trained on)',color:'--std',pts:pts.map(p=>[p.e,p.v])}],
  {y0,y1,yt,yf:fl,h:210})}
const T8=D.traj['8M'];
function draw2(){
  lossChart('loss800',D.loss['800k'],0.268,0.296,[0.27,0.28,0.29]);
  lossChart('loss8',D.loss['8M'],0.242,0.292,[0.25,0.26,0.27,0.28,0.29]);
  lines(document.getElementById('lr8'),[
    {name:'normal training (start 3×10⁻³)',color:'--std',pts:T8.standard.map(p=>[p.e,p.lr[0]])},
    {name:'PBT, range of 5 models',color:'--pbt',band:true,pts:T8.pbt.map(p=>[p.e,p.lr[0],p.lr[p.lr.length-1]])}],
    {log:true,y0:-5.2,y1:-2.3,yt:[1e-5,1e-4,1e-3],yf:fmtLR,vf:fmtLR,h:230,shade:[36,50]});
  lines(document.getElementById('val8'),[
    {name:'normal training',color:'--std',pts:T8.standard.map(p=>[p.e,p.best])},
    {name:'PBT',color:'--pbt',pts:T8.pbt.map(p=>[p.e,p.best])},
    {name:'PBT + both fixes',color:'--pbt',dash:'5 4',pts:T8.fixed.map(p=>[p.e,p.best])}],
    {y0:0.155,y1:0.215,yt:[0.16,0.18,0.20],yf:v=>v.toFixed(3),h:230,shade:[36,50]});
}
requestAnimationFrame(draw2);

/* ---------- all comparisons ---------- */
const CMP=[
 {label:'800k · PBT vs constant LR (too low)',v:1.36,kind:'invalid',tip:'Trials 1, 3 · Sep 2026 · Test (2.8M) 0.3225 vs 0.3269'},
 {label:'800k · PBT vs constant LR, 100 epochs',v:1.44,kind:'invalid',tip:'Trials 2, 4 · 0.3111 vs 0.3156'},
 {label:'800k · PBT every epoch vs every 5',v:-0.30,kind:'none',tip:'Trial 6 · two seeds: −0.47%, −0.13%; Validation said the opposite'},
 {label:'8M · PBT vs constant LR (too low)',v:7.67,kind:'invalid',tip:'Trial 10 · preregistered · Test (9.0M) 0.2375 vs 0.2572'},
 {label:'800k · PBT vs best constant LR, s12345',v:1.00,kind:'better',sep:1,tip:'Trials 14, 15 · 0.3003 vs 0.3033'},
 {label:'800k · PBT vs best constant LR, s22345',v:-0.47,kind:'worse',tip:'Trial 16 · 0.3041 vs 0.3027'},
 {label:'800k · PBT vs normal training, 3 seeds',v:0.81,ci:[0.34,1.34],kind:'better',tip:'Trial 18 · +0.67, +0.85, +0.91% · 1M Validation'},
 {label:'8M · PBT vs normal training',v:-12.4,kind:'worse',sep:1,tip:'Trials 23, 24 · 0.1812 vs 0.1611 · 1 seed'},
 {label:'8M · PBT + probes',v:-5.4,kind:'worse',tip:'Trial 25 · 0.1699'},
 {label:'8M · PBT + probes + final decay',v:-2.7,kind:'worse',tip:'Trial 26 · 0.1655'},
 {label:'8M · ensemble of 2 vs best single, 3 seeds',v:5.01,ci:[4.17,5.84],kind:'better',tip:'Trials 27, 28 · +5.05, +4.86, +5.14% · fixed before scoring'},
 {label:'800k · ensemble of 2 vs best single',v:0.0,kind:'none',tip:'Trial 29 · Validation only · 15 models, best pairs ≈ chance'},
 {label:'8M · distillation vs plain',v:-1.0,kind:'worse',tip:'Trial 32 · Validation only · 0.1600 vs 0.1584'},
];
requestAnimationFrame(()=>bars(document.getElementById('allcmp'),CMP,{lo:-16.5,hi:10.5,rh:30}));

/* ---------- trial table ---------- */
const OC={'✅':['better','better'],'❌':['worse','worse'],'➖':['none','no difference'],'⚠️':['invalid','not valid'],'🔧':['setup','set-up / check']};
const tb=document.querySelector('#tbl tbody'),filt={phase:'all',oc:'all'};
function render(){tb.innerHTML='';D.phases.forEach((ph,pi)=>{if(filt.phase!=='all'&&+filt.phase!==pi)return;
  const rows=ph.trials.filter(t=>filt.oc==='all'||OC[t.outcome][0]===filt.oc);if(!rows.length)return;
  const p=document.createElement('tr');p.className='phase';p.innerHTML=`<td colspan="5">${ph.title}<span>${ph.data} · ${ph.validation}</span></td>`;tb.appendChild(p);
  rows.forEach(t=>{const r=document.createElement('tr');r.className='trial';r.id='t'+t.n;const o=OC[t.outcome];
    r.innerHTML=`<td><span class="caret">${t.runs.length||t.records.length?'▸':''}</span>${t.n}</td><td class="hide-s">${t.dates}</td><td>${t.what}${t.gpu!=null?` <span style="color:var(--ink3)">· ${t.gpu<1?'<1':Math.round(t.gpu)} GPU-h</span>`:''}</td><td>${t.result}</td><td class="oc ${o[0]}">${o[1]}</td>`;
    const x=document.createElement('tr');x.className='runs';x.style.display='none';
    const recs=t.records.map(([l,u])=>`<a href="${u}">${l}</a>`).join(' · ');
    x.innerHTML=`<td></td><td colspan="4">${t.runs.length?`<table><tr><th>Run (configuration)</th><th>Start → end</th><th>GPU-h</th><th>Selected model, LR, epoch</th><th>Valid.</th><th>Test</th></tr>`+
      t.runs.map(u=>`<tr><td><a href="${u.url}">${u.run}</a></td><td>${u.start} → ${u.end}</td><td>${u.gpu<1?'<1':Math.round(u.gpu)}</td><td>${u.member}, ${u.lr}, ${u.epoch}</td><td>${u.val.toFixed(4)}</td><td>${u.test?u.test.toFixed(4):'—'}</td></tr>`).join('')+'</table>':''}
      ${recs?`<div style="margin-top:6px">Records: ${recs}</div>`:''}</td>`;
    r.onclick=()=>{const open=x.style.display==='none';x.style.display=open?'':'none';r.querySelector('.caret').textContent=open?'▾':'▸'};
    tb.appendChild(r);tb.appendChild(x)})})}
const fh=document.getElementById('filters');
function btn(label,key,val){const b=document.createElement('button');b.textContent=label;b.onclick=()=>{filt[key]=val;
  fh.querySelectorAll(`[data-k=${key}]`).forEach(z=>z.classList.toggle('on',z===b));render()};b.dataset.k=key;if(val==='all')b.classList.add('on');fh.appendChild(b)}
btn('All steps','phase','all');D.phases.forEach((p,i)=>btn('Step '+(i+1),'phase',String(i)));
const sp=document.createElement('span');sp.style.width='14px';fh.appendChild(sp);
btn('All outcomes','oc','all');for(const k of ['better','worse','none','invalid','setup'])btn(Object.values(OC).find(v=>v[0]===k)[1],'oc',k);
render();
document.addEventListener('click',ev=>{const a=ev.target.closest('a[data-t]');if(!a)return;ev.preventDefault();
  filt.phase='all';filt.oc='all';fh.querySelectorAll('button').forEach(b=>b.classList.toggle('on',b.textContent.startsWith('All')));render();
  const r=document.getElementById('t'+a.dataset.t);r.click();r.scrollIntoView({behavior:'smooth',block:'center'})});
</script>
</body>
</html>
'''

if __name__ == '__main__':
    main()
