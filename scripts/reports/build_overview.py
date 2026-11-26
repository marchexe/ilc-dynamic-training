#!/usr/bin/env python3
"""Build the study report page docs/_extra/overview.html (served at the site root by Sphinx).

    .venv/bin/python scripts/reports/build_overview.py

Static HTML (no scripts): scripts/reports/overview_template.html with the trial tables filled in. The figures are
PNG files from scripts/reports/plot_overview_figs.py (run it first).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sup(x):
    """3e-5 -> 3×10<sup>−5</sup> (HTML)."""
    m, e = f'{x:.1e}'.split('e')
    m = f'{float(m):g}'
    return ('' if m == '1' else f'{m}×') + f'10<sup>−{-int(e)}</sup>'


# ---------------------------------------------------------------- trial tables

def table(rows):
    """rows: (#, dates, configuration, result, mark) ; mark: good | bad | none | invalid | ''."""
    body = ''.join(
        f'<tr><td class="n">{n}</td><td class="d">{d}</td><td>{c}</td>'
        f'<td class="r {m}">{r}</td></tr>' for n, d, c, r, m in rows)
    return (f'<div class="tw"><table><thead><tr><th>#</th><th>Dates</th><th>Configuration</th><th>Result</th>'
            f'</tr></thead><tbody>{body}</tbody></table></div>')


def main():
    tables = dict(
        s1=table([
            ('1', 'Sep 16', f'Constant LR, five values {sup(3e-6)} to {sup(1.4e-5)}', '0.3269; best value is the highest one', ''),
            ('2', 'Sep 17–18', 'Two of these LRs continued to 100 epochs', '0.3156', ''),
            ('3', 'Sep 18', f'PBT, LR changed every 5 epochs (×0.8 or ×1.2), at most {sup(3e-5)}', '0.3225: 1.4% better than #1', 'invalid'),
            ('4', 'Sep 19', 'PBT as #3, 100 epochs', '0.3111: 1.4% better than #2', 'invalid'),
            ('6', 'Sep 23–29', 'PBT, LR changed every epoch or every 5 epochs; 2 seeds', 'no clear difference', 'none'),
            ('10', 'Oct 14–21', '8M jets: constant LR as #1 against PBT as #3; fixed in advance', '7.7% better (Test 9.0M)', 'invalid')]),
        s2=table([
            ('12', 'Nov 3', f'Constant LR {sup(1.4e-5)} to {sup(3e-5)}', '0.3146; best is again the highest', ''),
            ('13', 'Nov 3–4', f'Constant LR {sup(3e-5)} to {sup(8e-5)}', '0.3018: better than every PBT run before', 'good'),
            ('14', 'Nov 4', f'Constant LR {sup(8e-5)} to {sup(3.2e-4)}', '0.3033: no further gain', ''),
            ('15', 'Nov 4', f'Faster PBT: LR changed every 2 epochs (×0.5 or ×2), at most {sup(5e-4)}', '0.3003: 1.0% better than #14', 'good'),
            ('16', 'Nov 9', '#14 and #15 again with another seed', 'PBT 0.5% worse: the sign reverses', 'bad'),
            ('17', 'Nov 9', 'Faster PBT with extra weight copies', '0.8% worse than #15', 'bad')]),
        s3=table([
            ('18', 'Nov 10–13', 'Standard ParT, constant LR and PBT; 3 seeds', 'PBT 0.81% better than standard ParT', 'good'),
            ('19', 'Nov 12', 'PBT with a final LR decay; 2 seeds', 'same result as #18', 'none'),
            ('20', 'Nov 12', 'Standard ParT from random weights (no pretraining)', 'far worse', 'bad'),
            ('21', 'Nov 12', 'Averaging the weights of several checkpoints', 'no gain', 'none')]),
        s4=table([
            ('22', 'Nov 14–15', f'Standard ParT, start LR {sup(1e-5)} to {sup(1e-3)}', '0.1635; best is the highest', ''),
            ('23', 'Nov 15–17', f'PBT, LR at most {sup(1e-3)}', '0.1812: 12.4% worse than #24', 'bad'),
            ('24', 'Nov 19–20', f'Standard ParT, start LR {sup(3e-3)}', '0.1611: the reference', ''),
            ('25', 'Nov 19–21', 'PBT with fix 1: judge each LR after a short cool-down', '0.1699: 5.4% worse than #24', 'bad'),
            ('26', 'Nov 20–23', 'PBT with fixes 1 and 2: plus the final LR decay', '0.1655: 2.7% worse than #24', 'bad')]),
        s5=table([
            ('27', 'Nov 23', 'Ensemble: average of the outputs of #22 and #24', '0.1530: 5.1% better than #24', 'good'),
            ('28', 'Nov 23–25', 'The same ensemble with two new seeds', '4.9% and 5.1% better; mean 5.0%', 'good'),
            ('29', 'Nov 23', 'Ensembles of the 800k models', 'no gain', 'none'),
            ('31', 'Nov 23–24', 'Final LR decay of #24 without dropout', 'worse', 'bad'),
            ('32', 'Nov 24', 'Distillation: train one model on the ensemble output', 'no gain', 'none'),
            ('33', 'Nov 24–25', f'Standard ParT, start LR {sup(1e-2)}', '0.1599: same as #24', 'none')]),
    )
    page = TEMPLATE
    for k, v in tables.items():
        page = page.replace('{{table:%s}}' % k, v)
    out = ROOT / 'docs/_extra/overview.html'
    out.write_text(page)
    print(f'{out.relative_to(ROOT)}: {len(page) // 1024} kB')


TEMPLATE = (Path(__file__).with_name('overview_template.html')).read_text()

if __name__ == '__main__':
    main()
