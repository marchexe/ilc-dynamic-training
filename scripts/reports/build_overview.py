#!/usr/bin/env python3
"""Build the study report page docs/_extra/overview.html (served at the site root by Sphinx).

    .venv/bin/python scripts/reports/build_overview.py

Static HTML with inline SVG charts (no scripts). Test values are the published values of docs/results.rst;
learning-rate and Validation trajectories come from the run manifests (runs/pbt), loss curves from
published/results/sel1m/train8m_followups.json.
"""
import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'runs/pbt'
PRIMARY = 'validation_total_reference_mistag_geomean_percent'
W = 720  # SVG width in user units


def esc(s):
    return html.escape(str(s))


def sup(x):
    """3e-5 -> 3×10<sup>−5</sup> (HTML)."""
    m, e = f'{x:.1e}'.split('e')
    m = f'{float(m):g}'
    return ('' if m == '1' else f'{m}×') + f'10<sup>−{-int(e)}</sup>'


def svg_sup(x):
    m, e = f'{x:.1e}'.split('e')
    m = f'{float(m):g}'
    return ('' if m == '1' else f'{m}×') + f'10<tspan dy="-5" font-size="9">−{-int(e)}</tspan>'


# ---------------------------------------------------------------- charts

def delta_bars(rows, lo, hi, label_w=270, row_h=34, note=None):
    """Horizontal bars of a relative change in %, zero in the middle. rows: (label, value, kind, interval|None).
    kind: good | bad | invalid. Negative = lower mistag = better."""
    top, bottom = 8, 34
    h = top + len(rows) * row_h + bottom
    x0, x1 = label_w, W - 70
    X = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    out = [f'<svg viewBox="0 0 {W} {h}" role="img">',
           '<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
           '<rect width="6" height="6" class="f-bg"/><line x1="0" y1="0" x2="0" y2="6" class="s-ink2" stroke-width="2"/>'
           '</pattern></defs>']
    step = 2 if hi - lo <= 20 else 5
    t = math.ceil(lo / step) * step
    while t <= hi:
        out.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top}" y2="{h - bottom + 4}" class="s-grid"/>')
        out.append(f'<text x="{X(t):.1f}" y="{h - bottom + 18}" text-anchor="middle" class="t-axis">'
                   f'{"+" if t > 0 else ""}{t:g}%</text>')
        t += step
    out.append(f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="{top - 4}" y2="{h - bottom + 4}" class="s-ink" stroke-width="1.2"/>')
    out.append(f'<text x="{X(0) - 8:.1f}" y="{h - 1}" text-anchor="end" class="t-axis">← better</text>')
    out.append(f'<text x="{X(0) + 8:.1f}" y="{h - 1}" class="t-axis">worse →</text>')
    for i, (label, v, kind, ci) in enumerate(rows):
        y = top + i * row_h
        cls = {'good': 'f-good', 'bad': 'f-bad', 'invalid': 'f-hatch'}[kind]
        a, b = sorted((X(0), X(v)))
        out.append(f'<text x="{label_w - 12}" y="{y + row_h / 2 + 4:.1f}" text-anchor="end" class="t-label">{label}</text>')
        out.append(f'<rect x="{a:.1f}" y="{y + 8}" width="{max(b - a, 1.5):.1f}" height="{row_h - 16}" rx="2" class="{cls}"/>')
        edge = v
        if ci:
            yy = y + row_h / 2
            out.append(f'<line x1="{X(ci[0]):.1f}" x2="{X(ci[1]):.1f}" y1="{yy}" y2="{yy}" class="s-ink" stroke-width="1.5"/>')
            for c in ci:
                out.append(f'<line x1="{X(c):.1f}" x2="{X(c):.1f}" y1="{yy - 5}" y2="{yy + 5}" class="s-ink" stroke-width="1.5"/>')
            edge = ci[0] if v < 0 else ci[1]
        tx = X(edge) + (8 if v >= 0 else -8)
        out.append(f'<text x="{tx:.1f}" y="{y + row_h / 2 + 4:.1f}" text-anchor="{"start" if v >= 0 else "end"}" '
                   f'class="t-value">{"+" if v > 0 else "−" if v < 0 else ""}{abs(v):.2f}%</text>')
    out.append('</svg>')
    return '\n'.join(out)


def dot_rows(rows, lo, hi, ticks, label_w=230, row_h=30, xlabel='Test mistag (%)'):
    """rows: (label, [values], cls, shape) or None for a separator. shape: c (circle), s (square), d (diamond), o (ring)."""
    top, bottom = 6, 40
    h = top + len(rows) * row_h + bottom
    x0, x1 = label_w, W - 20
    X = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    out = [f'<svg viewBox="0 0 {W} {h}" role="img">']
    for t in ticks:
        out.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top}" y2="{h - bottom + 4}" class="s-grid"/>')
        out.append(f'<text x="{X(t):.1f}" y="{h - bottom + 18}" text-anchor="middle" class="t-axis">{t:.3f}</text>')
    out.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{h - 4}" text-anchor="middle" class="t-axis">{xlabel}, lower is better</text>')
    for i, row in enumerate(rows):
        y = top + i * row_h + row_h / 2
        if row is None:
            out.append(f'<line x1="12" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}" class="s-rule"/>')
            continue
        label, vals, cls, shape = row
        out.append(f'<text x="{label_w - 12}" y="{y + 4:.1f}" text-anchor="end" class="t-label">{label}</text>')
        for v in vals:
            out.append(marker(X(v), y, cls, shape))
    out.append('</svg>')
    return '\n'.join(out)


def marker(x, y, cls, shape, r=6):
    if shape == 's':
        return f'<rect x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r}" height="{2 * r}" class="{cls}"/>'
    if shape == 'd':
        return (f'<path d="M{x:.1f},{y - r - 1:.1f} L{x + r + 1:.1f},{y:.1f} L{x:.1f},{y + r + 1:.1f} '
                f'L{x - r - 1:.1f},{y:.1f} Z" class="{cls}"/>')
    if shape == 'o':
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r - 0.5}" class="f-bg {cls.replace("f-", "s-")}" stroke-width="2"/>'
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" class="{cls}"/>'


def lr_curve():
    """Test mistag against the LR of the selected model, 800k: constant LRs and PBT."""
    h, L, R, T, B = 300, 70, 20, 14, 44
    lx0, lx1, y0, y1 = math.log10(2.5e-6), math.log10(2.5e-4), 0.298, 0.330
    X = lambda v: L + (math.log10(v) - lx0) / (lx1 - lx0) * (W - L - R)
    Y = lambda v: T + (1 - (v - y0) / (y1 - y0)) * (h - T - B)
    out = [f'<svg viewBox="0 0 {W} {h}" role="img">',
           f'<rect x="{X(3e-6):.1f}" y="{T}" width="{X(1.4e-5) - X(3e-6):.1f}" height="{h - T - B}" class="f-band"/>',
           f'<text x="{(X(3e-6) + X(1.4e-5)) / 2:.1f}" y="{h - B - 10}" text-anchor="middle" class="t-axis">LRs used in September</text>']
    for v in (0.300, 0.305, 0.310, 0.315, 0.320, 0.325, 0.330):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="s-grid"/>')
        out.append(f'<text x="{L - 8}" y="{Y(v) + 4:.1f}" text-anchor="end" class="t-axis">{v:.3f}</text>')
    for v in (1e-5, 1e-4):
        out.append(f'<text x="{X(v):.1f}" y="{h - B + 18}" text-anchor="middle" class="t-axis">{svg_sup(v)}</text>')
    for v in (3e-6, 3e-5, 3e-4):
        if X(v) < W - R:
            out.append(f'<text x="{X(v):.1f}" y="{h - B + 18}" text-anchor="middle" class="t-axis">{svg_sup(v)}</text>')
    out.append(f'<text x="{(L + W - R) / 2:.1f}" y="{h - 4}" text-anchor="middle" class="t-axis">learning rate of the selected model (log scale)</text>')
    pts = [(1.4e-5, 0.3269, '1'), (3e-5, 0.3146, '12'), (8e-5, 0.3018, '13'), (1.2e-4, 0.3033, '14')]
    out.append('<polyline points="' + ' '.join(f'{X(a):.1f},{Y(b):.1f}' for a, b, _ in pts) + '" class="s-const" fill="none" stroke-width="2"/>')
    for a, b, n in pts:
        out.append(marker(X(a), Y(b), 'f-const', 'o'))
        out.append(f'<text x="{X(a) + 9:.1f}" y="{Y(b) - 8:.1f}" class="t-note">#{n}</text>')
    for a, b, n, lab in ((2.02e-5, 0.3225, '3', 'PBT, Sep'), (8.75e-5, 0.3003, '15', 'PBT, Nov')):
        out.append(marker(X(a), Y(b), 'f-pbt', 's'))
        out.append(f'<text x="{X(a) + 10:.1f}" y="{Y(b) + 16:.1f}" class="t-label t-pbt">{lab} (#{n})</text>')
    out.append(f'<text x="{X(1.4e-5) + 70:.1f}" y="{Y(0.3269) - 2:.1f}" class="t-label t-const">constant LR, best model of each range</text>')
    out.append('</svg>')
    return '\n'.join(out)


def line_chart(series, y0, y1, yticks, yfmt, log=False, h=250, shade=None, ylabel='', legend=None):
    """series: (points [(x, y)] or band [(x, lo, hi)], cls, kind) ; x = epoch 1..50."""
    L, R, T, B = 64, 14, 12, 40
    wd = W // 2 - 10
    f = (lambda v: math.log10(v)) if log else (lambda v: v)
    X = lambda e: L + (e - 1) / 49 * (wd - L - R)
    Y = lambda v: T + (1 - (f(v) - f(y0)) / (f(y1) - f(y0))) * (h - T - B)
    out = [f'<svg viewBox="0 0 {wd} {h}" role="img">']
    if shade:
        out.append(f'<rect x="{X(shade[0]):.1f}" y="{T}" width="{X(shade[1]) - X(shade[0]):.1f}" height="{h - T - B}" class="f-band"/>')
    for v in yticks:
        out.append(f'<line x1="{L}" x2="{wd - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="s-grid"/>')
        out.append(f'<text x="{L - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" class="t-axis">{yfmt(v)}</text>')
    for e in (1, 10, 20, 30, 40, 50):
        out.append(f'<text x="{X(e):.1f}" y="{h - B + 16}" text-anchor="middle" class="t-axis">{e}</text>')
    out.append(f'<text x="{(L + wd - R) / 2:.1f}" y="{h - 6}" text-anchor="middle" class="t-axis">epoch</text>')
    for pts, cls, kind in series:
        if kind == 'band':
            poly = [f'{X(e):.1f},{Y(min(max(hi, y0), y1)):.1f}' for e, lo_, hi in pts] + \
                   [f'{X(e):.1f},{Y(min(max(lo_, y0), y1)):.1f}' for e, lo_, hi in reversed(pts)]
            out.append(f'<polygon points="{" ".join(poly)}" class="{cls}" opacity="0.28"/>')
        else:
            p = [(e, min(max(v, y0), y1)) for e, v in pts]
            dash = ' stroke-dasharray="5 4"' if kind == 'dash' else ''
            out.append('<polyline points="' + ' '.join(f'{X(e):.1f},{Y(v):.1f}' for e, v in p) +
                       f'" fill="none" class="{cls}" stroke-width="2.2"{dash}/>')
    out.append('</svg>')
    leg = ''.join(f'<span><i class="{c}"></i>{t}</span>' for c, t in (legend or []))
    return f'<div class="mini"><div class="mini-y">{ylabel}</div>{chr(10).join(out)}<div class="legend">{leg}</div></div>'


def trajectory(run, members=None):
    m = json.loads((RUNS / run / 'manifest.json').read_text())
    out, best = [], None
    for g in m['generations']:
        ws = {k: w for k, w in g['workers'].items() if members is None or k in members}
        vals = [w['metrics'][PRIMARY] for w in ws.values() if (w.get('metrics') or {}).get(PRIMARY) is not None]
        if vals:
            best = min(vals) if best is None else min(best, min(vals))
        lrs = sorted(w['lr'] for w in ws.values())
        out.append((g['epoch'] - 17, lrs[0], lrs[-1], best))
    return out


# ---------------------------------------------------------------- trial tables

def table(rows):
    """rows: (#, dates, configuration, result, mark) ; mark: good | bad | none | invalid | ''."""
    body = ''.join(
        f'<tr><td class="n">{n}</td><td class="d">{d}</td><td>{c}</td>'
        f'<td class="r {m}">{r}</td></tr>' for n, d, c, r, m in rows)
    return (f'<div class="tw"><table><thead><tr><th>#</th><th>Dates</th><th>Configuration</th><th>Result on Test</th>'
            f'</tr></thead><tbody>{body}</tbody></table></div>')


def main():
    std = trajectory('sel1m_train8m_standard_part_3e-3')
    pbt = trajectory('sel1m_train8m_fast_pbt')
    fix = trajectory('sel1m_train8m_fast_pbt_probe_decay')
    loss = json.loads((ROOT / 'published/results/sel1m/train8m_followups.json').read_text())['loss_curves']
    lc = {k: [(p['epoch'] - 17, p['train'], p['selection']) for p in v['epochs']] for k, v in loss.items()}

    charts = dict(
        found=delta_bars([
            ('800k jets: PBT vs standard ParT (3 seeds)', -0.81, 'good', (-1.34, -0.34)),
            ('8M jets: PBT vs standard ParT (1 seed)', 12.4, 'bad', None),
            ('8M jets: PBT with two fixes (1 seed)', 2.7, 'bad', None),
            ('8M jets: ensemble of two vs best single (3 seeds)', -5.01, 'good', (-5.84, -4.17))],
            -11, 14, label_w=340),
        s1=delta_bars([('800k jets, Test (2.8M)', -1.36, 'invalid', None),
                       ('8M jets, Test (9.0M)', -7.67, 'invalid', None)], -11, 3, label_w=230),
        s2=lr_curve(),
        s3=dot_rows([
            ('standard ParT', [0.3029, 0.3038, 0.3024], 'f-std', 'c'),
            ('constant LR grid', [0.3028, 0.3013, 0.3024], 'f-const', 'o'),
            ('PBT', [0.3008, 0.3012, 0.2996], 'f-pbt', 's')],
            0.2985, 0.3045, [0.299, 0.300, 0.301, 0.302, 0.303, 0.304], label_w=170, row_h=36,
            xlabel='Test mistag (%) of the selected model; one point per seed'),
        s4=delta_bars([('PBT', 12.4, 'bad', None), ('PBT + cool-down probes', 5.4, 'bad', None),
                       ('PBT + probes + final LR decay', 2.7, 'bad', None)], -2, 14, label_w=250),
        s5=dot_rows([
            ('standard ParT, start 10⁻³', [0.1635, 0.1623, 0.1634], 'f-std', 'o'),
            ('standard ParT, start 3×10⁻³', [0.1611, 0.1597, 0.1623], 'f-std', 'c'),
            ('ensemble of the two', [0.1530, 0.1519, 0.1540], 'f-ens', 'd'),
            None,
            ('PBT (seed 12345)', [0.1812], 'f-pbt', 's'),
            ('PBT + probes + decay', [0.1655], 'f-pbt', 's')],
            0.148, 0.186, [0.150, 0.160, 0.170, 0.180], label_w=230, row_h=32,
            xlabel='Test mistag (%); one point per seed'),
        lr8=line_chart([([(e, a, b) for e, a, b, _ in pbt], 'f-pbt', 'band'), ([(e, a) for e, a, _, _ in std], 's-std', 'line')],
                       1e-5, 6e-3, [1e-5, 1e-4, 1e-3], lambda v: svg_sup(v), log=True, shade=(36, 50),
                       ylabel='learning rate', legend=[('k-std', 'standard ParT'), ('k-pbt-band', 'PBT, range of 5 models')]),
        val8=line_chart([([(e, v) for e, _, _, v in std], 's-std', 'line'), ([(e, v) for e, _, _, v in pbt], 's-pbt', 'line'),
                         ([(e, v) for e, _, _, v in fix], 's-pbt', 'dash')],
                        0.155, 0.215, [0.16, 0.18, 0.20], lambda v: f'{v:.2f}', shade=(36, 50),
                        ylabel='best Validation mistag so far (%)',
                        legend=[('k-std', 'standard ParT'), ('k-pbt', 'PBT'), ('k-pbt-dash', 'PBT + both fixes')]),
        loss800=line_chart([([(e, t) for e, t, _ in lc['800k']], 's-ink', 'line'), ([(e, v) for e, _, v in lc['800k']], 's-std', 'line')],
                           0.268, 0.296, [0.27, 0.28, 0.29], lambda v: f'{v:.2f}', ylabel='loss, 800k jets',
                           legend=[('k-ink', 'training data'), ('k-std', 'Validation')]),
        loss8=line_chart([([(e, t) for e, t, _ in lc['8M']], 's-ink', 'line'), ([(e, v) for e, _, v in lc['8M']], 's-std', 'line')],
                         0.242, 0.292, [0.25, 0.26, 0.27, 0.28, 0.29], lambda v: f'{v:.2f}', ylabel='loss, 8M jets',
                         legend=[('k-ink', 'training data'), ('k-std', 'Validation')]),
    )
    tables = dict(
        s1=table([
            ('1', 'Sep 16', f'800k · constant LRs {sup(3e-6)}–{sup(1.4e-5)}', '0.3269 · best = highest LR', ''),
            ('2', 'Sep 17–18', '800k · two LRs of #1, 100 epochs', '0.3156', ''),
            ('3', 'Sep 18', f'800k · PBT: every 5 epochs, LR ×0.8 or ×1.2, max {sup(3e-5)}', '0.3225 · −1.4% vs #1', 'invalid'),
            ('4', 'Sep 19', '800k · PBT as #3, 100 epochs', '0.3111 · −1.4% vs #2', 'invalid'),
            ('6', 'Sep 23–29', '800k · PBT every epoch vs every 5 epochs · 2 seeds', '+0.5%, +0.1% · inconclusive', 'none'),
            ('10', 'Oct 14–21', '8M · #1 vs #3, preregistered · Test (9.0M)', '−7.7% · best = highest LR again', 'invalid')]),
        s2=table([
            ('12', 'Nov 3', f'800k · constant LRs {sup(1.4e-5)}–{sup(3e-5)}', '0.3146 · best = highest LR', ''),
            ('13', 'Nov 3–4', f'800k · constant LRs {sup(3e-5)}–{sup(8e-5)}', '<b>0.3018</b> · better than all PBT so far', 'good'),
            ('14', 'Nov 4', f'800k · constant LRs {sup(8e-5)}–{sup(3.2e-4)}', '0.3033', ''),
            ('15', 'Nov 4', f'800k · fast PBT: every 2 epochs, LR ×0.5 or ×2, max {sup(5e-4)}', '0.3003 · −1.0% vs #14', 'good'),
            ('16', 'Nov 9', '800k · #14 and #15 with seed 22345', '+0.5% · sign reversed', 'bad'),
            ('17', 'Nov 9', '800k · fast PBT with extra copies', '0.3026 · +0.8% vs #15', 'bad')]),
        s3=table([
            ('18', 'Nov 10–13', f'800k · standard ParT (start {sup(1e-5)}–{sup(1e-3)}), constant LRs, fast PBT · 3 seeds',
             '<b>PBT −0.81%</b> vs standard ParT', 'good'),
            ('19', 'Nov 12', '800k · PBT with a final LR decay · 2 seeds', 'same models as #18', 'none'),
            ('20', 'Nov 12', '800k · standard ParT without pretraining', '0.378 (Validation) · far worse', 'bad'),
            ('21', 'Nov 12', '800k · weight averaging of checkpoints', 'no gain (Validation)', 'none')]),
        s4=table([
            ('22', 'Nov 14–15', f'8M · standard ParT, start {sup(1e-5)}–{sup(1e-3)}', '0.1635 · best = highest LR', ''),
            ('23', 'Nov 15–17', f'8M · fast PBT, max LR {sup(1e-3)}', '0.1812 · <b>+12.4%</b> vs #24', 'bad'),
            ('24', 'Nov 19–20', f'8M · standard ParT, start {sup(3e-3)}', '<b>0.1611</b> · 8M reference', ''),
            ('25', 'Nov 19–21', '8M · PBT with cool-down probes', '0.1699 · +5.4% vs #24', 'bad'),
            ('26', 'Nov 20–23', '8M · PBT with probes and a final LR decay', '0.1655 · +2.7% vs #24', 'bad')]),
        s5=table([
            ('27', 'Nov 23', '8M · ensemble: mean output of #22 and #24', '0.1530 · −5.1% vs #24', 'good'),
            ('28', 'Nov 23–25', '8M · #27 repeated with seeds 22345 and 32345', '−4.9%, −5.1% · <b>mean −5.0%</b>', 'good'),
            ('29', 'Nov 23', '800k · ensembles of the 800k models', 'no gain (Validation)', 'none'),
            ('31', 'Nov 23–24', '8M · final decay of #24 without dropout', 'worse (Validation)', 'bad'),
            ('32', 'Nov 24', '8M · distillation of the ensemble into one model', 'no gain (Validation)', 'none'),
            ('33', 'Nov 24–25', f'8M · standard ParT, start {sup(1e-2)}', '0.1599 · same as #24', 'none')]),
    )
    page = TEMPLATE
    for k, v in charts.items():
        page = page.replace('{{chart:%s}}' % k, v)
    for k, v in tables.items():
        page = page.replace('{{table:%s}}' % k, v)
    out = ROOT / 'docs/_extra/overview.html'
    out.write_text(page)
    print(f'{out.relative_to(ROOT)}: {len(page) // 1024} kB')


TEMPLATE = (Path(__file__).with_name('overview_template.html')).read_text()

if __name__ == '__main__':
    main()
