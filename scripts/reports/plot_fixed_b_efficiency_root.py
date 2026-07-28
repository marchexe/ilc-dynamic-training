#!/usr/bin/env python3
"""ROOT version of the fixed b-tag efficiency mistag plot."""

import argparse
from array import array
from pathlib import Path

from reports.plot_fixed_b_efficiency import (
    DEFAULT_B_EFFICIENCIES,
    BACKGROUND_LABELS,
    collect_series,
    event_label,
    load_manifest,
)


ROOT_STYLES = {
    ("c", 0.8): {"color": 632, "marker": 20},  # red
    ("d", 0.8): {"color": 600, "marker": 21},  # blue
    ("c", 0.9): {"color": 616, "marker": 24},  # magenta open circle
    ("d", 0.9): {"color": 432, "marker": 25},  # cyan open square
    ("c", 1.0): {"color": 800, "marker": 26},
    ("d", 1.0): {"color": 416, "marker": 32},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="ROOT plot of background efficiency at fixed b-tag efficiency."
    )
    parser.add_argument("manifest", type=Path, help="PBT manifest.json or run directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--member",
        default="best",
        help="Member to plot, or 'best' for the best member in each completed generation.",
    )
    parser.add_argument(
        "--b-eff",
        default=",".join(str(value) for value in DEFAULT_B_EFFICIENCIES),
        help="Comma-separated fixed b-tag efficiencies, e.g. 0.8,0.9,1.0.",
    )
    parser.add_argument("--title", default="Background efficiency at fixed b-tag efficiency")
    return parser.parse_args()


def default_output(manifest_path):
    return Path(manifest_path).parent / "plots" / "diagnostics" / "working_point_mistag_history_root.pdf"


def make_graph(root, xs, ys, background, b_eff):
    graph = root.TGraph(len(xs), array("d", xs), array("d", ys))
    style = ROOT_STYLES.get((background, b_eff), {"color": 1, "marker": 20})
    graph.SetLineColor(style["color"])
    graph.SetMarkerColor(style["color"])
    graph.SetMarkerStyle(style["marker"])
    graph.SetMarkerSize(1.0)
    graph.SetLineWidth(3)
    return graph


def plot_manifest(manifest_path, output=None, member="best", b_efficiencies=DEFAULT_B_EFFICIENCIES, title=None):
    try:
        import ROOT
    except ImportError as error:
        raise RuntimeError("PyROOT is not available. Load ROOT first, then rerun this script.") from error

    manifest, resolved_manifest_path = load_manifest(manifest_path)
    output = Path(output) if output is not None else default_output(resolved_manifest_path)
    series, point_labels = collect_series(manifest, b_efficiencies, member=member)

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetTitleFont(42, "")
    ROOT.gStyle.SetLabelFont(42, "XYZ")
    ROOT.gStyle.SetTitleFont(42, "XYZ")
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)

    canvas = ROOT.TCanvas("fixed_b_efficiency", "fixed_b_efficiency", 900, 650)
    canvas.SetLogx(True)
    canvas.SetLogy(True)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.04)
    canvas.SetBottomMargin(0.12)
    canvas.SetTopMargin(0.08)

    all_x = [x for values in series.values() for x in values["x"]]
    if not all_x:
        raise RuntimeError("manifest has no b-tag efficiency rejection curves to plot")
    min_x = min(all_x) * 0.8
    max_x = max(all_x) * 1.25

    frame = canvas.DrawFrame(min_x, 1e-4, max_x, 1.0)
    frame.SetTitle(title or "Background efficiency at fixed b-tag efficiency")
    frame.GetXaxis().SetTitle("Training size [events]")
    frame.GetYaxis().SetTitle("Background efficiency")
    frame.GetXaxis().SetTitleOffset(1.15)
    frame.GetYaxis().SetTitleOffset(1.35)
    frame.GetXaxis().SetMoreLogLabels(True)
    frame.GetXaxis().SetNoExponent(True)
    frame.GetYaxis().SetMoreLogLabels(True)
    frame.GetYaxis().SetNoExponent(True)

    legend = ROOT.TLegend(0.55, 0.66, 0.90, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.035)

    graphs = []
    for b_eff in b_efficiencies:
        for background in ("c", "d"):
            values = series[(background, b_eff)]
            if not values["x"]:
                continue
            graph = make_graph(ROOT, values["x"], values["y"], background, b_eff)
            graph.Draw("LP SAME")
            legend.AddEntry(graph, f"{BACKGROUND_LABELS[background]} / b-eff={b_eff:.2f}", "lp")
            graphs.append(graph)

    labels = []
    seen = set()
    for events, _, _ in point_labels:
        if events in seen:
            continue
        seen.add(events)
        y_values = [
            y
            for values in series.values()
            for x, y in zip(values["x"], values["y"])
            if x == events
        ]
        if not y_values:
            continue
        label = ROOT.TLatex(events * 1.02, max(y_values) * 1.06, event_label(events))
        label.SetTextFont(42)
        label.SetTextSize(0.032)
        label.Draw()
        labels.append(label)

    legend.Draw()
    canvas.Update()
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.SaveAs(str(output))
    return output


def main():
    args = parse_args()
    b_efficiencies = tuple(float(value.strip()) for value in args.b_eff.split(",") if value.strip())
    print(plot_manifest(args.manifest, args.output, args.member, b_efficiencies, args.title))


if __name__ == "__main__":
    main()
