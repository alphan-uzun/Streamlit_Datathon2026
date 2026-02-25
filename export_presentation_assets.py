from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _need(mod: str):
    try:
        return __import__(mod)
    except Exception:
        return None


alt = _need("altair")
graphviz = _need("graphviz")
dfi = _need("dataframe_image")

if alt is None:
    print("Missing dependency: altair")
    print("Install with: pip install altair")
    sys.exit(1)

try:
    import streamlit_cumulative_gains_dashboard as dash
except Exception as e:
    print(f"Failed to import streamlit_cumulative_gains_dashboard.py: {e}")
    sys.exit(1)


def require_vl_convert() -> None:
    try:
        import vl_convert  # noqa: F401
    except Exception:
        print("Missing dependency for PNG/SVG chart export: vl-convert-python")
        print("Install with: pip install vl-convert-python")
        sys.exit(1)


def save_altair(chart: Any, out_png: Path, out_svg: Path | None = None) -> None:
    require_vl_convert()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    chart.save(out_png.as_posix())  # infers PNG from extension
    print(f"Saved {out_png}")
    if out_svg is not None:
        chart.save(out_svg.as_posix())
        print(f"Saved {out_svg}")


def save_table(df: pd.DataFrame, out_png: Path, out_csv: Path | None = None) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if out_csv is not None:
        df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"Saved {out_csv}")
    if dfi is None:
        print(f"Skipping PNG table export for {out_png.name}: missing dataframe_image")
        print("Install with: pip install dataframe-image")
        return
    try:
        # dataframe_image renders styled dataframes to PNG (closest to table visuals without Streamlit).
        dfi.export(df, out_png.as_posix(), table_conversion="matplotlib")
        print(f"Saved {out_png}")
    except Exception as e:
        print(f"Failed table PNG export ({out_png.name}): {e}")
        print("Tip: pip install dataframe-image")


def save_graphviz_png(dot_src: str, out_png: Path) -> None:
    if graphviz is None:
        print(f"Skipping {out_png.name}: missing python graphviz package")
        print("Install with: pip install graphviz")
        return
    try:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        src = graphviz.Source(dot_src)
        # render() writes <filename>.png when format set.
        src.format = "png"
        # graphviz adds extension automatically; strip for path stem
        target = out_png.with_suffix("")
        src.render(filename=target.name, directory=target.parent.as_posix(), cleanup=True)
        print(f"Saved {out_png}")
    except Exception as e:
        print(f"Failed schema export ({out_png.name}): {e}")
        print("You may need Graphviz binary installed and `dot` on PATH.")
        print("Windows (winget): winget install Graphviz.Graphviz")


def build_step_chart(split: str, k: int):
    odf = dash.official_metric_table(split)
    step_order = [
        "AWS Baseline (1 Guess)",
        "Base Generator",
        "Fine-Tuned Generator",
        "Frequency-Aware Reranker",
        "Frequency-Aware Reranker + AWS First Guess",
    ]
    step_df = odf[(odf["k"] == k) & (odf["label"].isin(step_order))].copy()
    step_df["label"] = pd.Categorical(step_df["label"], categories=step_order, ordered=True)
    step_df = step_df.sort_values("label")
    step_colors = [dash.SYSTEM_COLORS.get(x, "#888888") for x in step_order]
    base = (
        alt.Chart(step_df)
        .mark_bar(size=38, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("label:N", title=None, sort=step_order, axis=alt.Axis(labelAngle=-20, labelLimit=220)),
            y=alt.Y("official_f1:Q", title="Official Weighted Macro F1"),
            color=alt.Color("label:N", title=None, sort=step_order, scale=alt.Scale(domain=step_order, range=step_colors)),
            tooltip=["label", "k", alt.Tooltip("official_f1:Q", format=".4f")],
        )
        .properties(height=300, width=720)
    )
    labels = (
        alt.Chart(step_df)
        .mark_text(dy=-8, fontSize=12, fontWeight="bold")
        .encode(
            x=alt.X("label:N", sort=step_order),
            y=alt.Y("official_f1:Q"),
            text=alt.Text("official_f1:Q", format=".3f"),
        )
    )
    step_tbl = step_df.copy().reset_index(drop=True)
    step_tbl["delta_vs_prev"] = step_tbl["official_f1"].diff()
    step_tbl["pct_change_vs_prev"] = step_tbl["official_f1"].pct_change()
    step_tbl["official_f1"] = step_tbl["official_f1"].map(lambda x: f"{x:.3f}")
    step_tbl["delta_vs_prev"] = step_tbl["delta_vs_prev"].map(lambda x: "" if pd.isna(x) else f"{100*x:+.2f} pts")
    step_tbl["pct_change_vs_prev"] = step_tbl["pct_change_vs_prev"].map(lambda x: "" if pd.isna(x) else f"{100*x:+.2f}%")
    step_tbl = step_tbl.rename(columns={"label": "Step", "official_f1": "Official Weighted Macro F1", "delta_vs_prev": "Absolute Change (pts)", "pct_change_vs_prev": "Relative Change (%)"})
    return base + labels, step_tbl


def build_topk_chart(split: str):
    odf = dash.official_metric_table(split)
    line_order = [
        "Base Generator",
        "Fine-Tuned Generator",
        "Frequency-Aware Reranker",
        "Frequency-Aware Reranker + AWS First Guess",
        "AWS Baseline (1 Guess)",
    ]
    selected = [x for x in line_order if x in set(odf["label"].tolist())]
    line_df = odf[odf["label"].isin(selected)].copy()
    if line_df.empty:
        return None
    y_min = float(line_df["official_f1"].min())
    y_max = float(line_df["official_f1"].max())
    pad = max((y_max - y_min) * 0.15, 0.01)
    chart = (
        alt.Chart(line_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("k:O", title="k"),
            y=alt.Y("official_f1:Q", title="Official Weighted Macro F1", scale=alt.Scale(domain=[max(0.0, y_min - pad), min(1.0, y_max + pad)])),
            color=alt.Color("label:N", sort=selected, title=None, scale=alt.Scale(domain=selected, range=[dash.SYSTEM_COLORS.get(x, "#888888") for x in selected])),
            tooltip=["label", "k", alt.Tooltip("official_f1:Q", format=".4f")],
        )
        .properties(height=320, width=720)
    )
    return chart


def build_coverage_chart_and_table(split: str, k: int):
    gold = dash.load_gold(dash.PATHS[f"{split}_gold"].as_posix())
    available = dash._available_systems(split)
    by_id = {x["id"]: x for x in available}
    selected_id = "freq_anchor" if "freq_anchor" in by_id else ("pairwise_anchor" if "pairwise_anchor" in by_id else "ft_raw")
    baseline_id = "base_raw" if "base_raw" in by_id else selected_id
    sel_preds = dash._pred_map(split, selected_id, k)
    base_preds = dash._pred_map(split, baseline_id, k)
    oracle_preds = {
        pid: [t for t, _w in sorted(zip(g["translations"], g["weights"]), key=lambda x: float(x[1]), reverse=True)[:k]]
        for pid, g in gold.items()
    }
    sel_cov = dash.coverage_stats(gold, sel_preds, k)
    base_cov = dash.coverage_stats(gold, base_preds, k)
    oracle_cov = dash.coverage_stats(gold, oracle_preds, k)
    cov_tbl = pd.DataFrame(
        [
            {"System": by_id[baseline_id]["label"], "Gold Variant Coverage %": 100*base_cov["macro_variant_coverage"], "Gold Frequency Covered %": 100*base_cov["macro_weighted_recall"], "Prompt Hit Rate %": 100*base_cov["prompt_hit_rate"], "Best Gold Found %": 100*base_cov["best_gold_hit_ratio"]},
            {"System": by_id[selected_id]["label"], "Gold Variant Coverage %": 100*sel_cov["macro_variant_coverage"], "Gold Frequency Covered %": 100*sel_cov["macro_weighted_recall"], "Prompt Hit Rate %": 100*sel_cov["prompt_hit_rate"], "Best Gold Found %": 100*sel_cov["best_gold_hit_ratio"]},
            {"System": f"Oracle @ {k}", "Gold Variant Coverage %": 100*oracle_cov["macro_variant_coverage"], "Gold Frequency Covered %": 100*oracle_cov["macro_weighted_recall"], "Prompt Hit Rate %": 100*oracle_cov["prompt_hit_rate"], "Best Gold Found %": 100*oracle_cov["best_gold_hit_ratio"]},
        ]
    )
    plot_df = cov_tbl.melt(id_vars=["System"], value_vars=["Gold Variant Coverage %", "Gold Frequency Covered %", "Prompt Hit Rate %", "Best Gold Found %"], var_name="Metric", value_name="Percent")
    order = [s for s in cov_tbl["System"].tolist() if not str(s).startswith("Oracle @")] + [s for s in cov_tbl["System"].tolist() if str(s).startswith("Oracle @")]
    color_domain = order
    color_range = [dash.SYSTEM_COLORS.get(x, dash.ORACLE_COLOR if str(x).startswith("Oracle @") else "#888888") for x in color_domain]
    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("Metric:N", title=None),
            xOffset=alt.XOffset("System:N", sort=order),
            y=alt.Y("Percent:Q", title="Percent"),
            color=alt.Color("System:N", sort=order, title=None, scale=alt.Scale(domain=color_domain, range=color_range)),
            tooltip=["System", "Metric", alt.Tooltip("Percent:Q", format=".2f")],
        )
        .properties(height=330, width=760)
    )
    return chart, cov_tbl.round(2)


BIG_DOT = r"""
digraph G {
  rankdir=LR;
  graph [pad="0.25", nodesep="0.55", ranksep="0.7", bgcolor="white"];
  node [shape=box, style="rounded,filled", fontsize=14];
  p [label="English Prompt", fillcolor="white"];
  g [label="Generator\n(Base NLLB + LoRA Fine-Tuning)", shape=component, fillcolor="#dff0d8", color="#59A14F", penwidth=2];
  c [label="Candidate Pool\nBeam + Sampling + Dedupe", shape=folder, fillcolor="#eef2f7", color="#9AA0A6", penwidth=2];
  f1 [label="Pairwise Neural Ranker", shape=component, fillcolor="#fde6ce", color="#F28E2B", penwidth=2];
  f2 [label="Frequency-Aware Reranker", shape=component, fillcolor="#efe3f5", color="#B07AA1", penwidth=2];
  pol [label="Anchoring / Final Policy\n(Optional AWS First Guess)", shape=diamond, fillcolor="#fbe0e3", color="#E15759", penwidth=2];
  o [label="Top-k Hungarian Guesses", fillcolor="white"];
  p -> g -> c;
  c -> f1; c -> f2; f1 -> pol; f2 -> pol; pol -> o;
}
"""

GEN_DOT = r"""
digraph GEN {
  rankdir=TB; node [shape=box, style=rounded];
  p [label="English Prompt"];
  d [label="STAPLE Training Data\n(weighted sampling)", shape=folder];
  base [label="Base NLLB"];
  lora [label="LoRA Adapters", style="rounded,filled", fillcolor="#fff2cc"];
  ft [label="Fine-Tuned Generator", style="rounded,filled", fillcolor="#e8f5e9"];
  beam [label="Beam Decode"]; samp [label="Sampling Decode"];
  dd [label="Merge + Dedupe"]; out [label="Candidate Pool"];
  d -> lora [label="finetunes"]; base -> ft; lora -> ft; p -> ft;
  ft -> beam -> dd; ft -> samp -> dd; dd -> out;
}
"""

PAIRWISE_DOT = r"""
digraph P {
  rankdir=TB; node [shape=box, style=rounded];
  p [label="Prompt"]; a [label="AWS Translation"]; c [label="Candidates"];
  e [label="Embedding Encoder"];
  g [label="Candidate-Pool Geometry\n(unsupervised centers)"];
  f [label="Feature Builder Neural Network"];
  r [label="Pairwise MLP Ranker"]; s [label="Scores -> Sort"];
  p -> e; a -> e; c -> e; c -> g; e -> f; g -> f; f -> r -> s;
}
"""

FREQ_DOT = r"""
digraph F {
  rankdir=TB; node [shape=box, style=rounded];
  p [label="Prompt"]; a [label="AWS Translation"]; c [label="Candidates"];
  e [label="Embedding Encoder"]; f [label="Feature Builder Neural Network"];
  r [label="MLP Regressor\n(predicted relevance)"]; s [label="Scores -> Sort"];
  t [label="Training target:\ngold weight or 0", shape=note];
  p -> e; a -> e; c -> e; e -> f; f -> r -> s; t -> r;
}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Export presentation assets using the same Altair/Graphviz visual style (no Streamlit UI).")
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    ap.add_argument("--k", type=int, default=5, choices=dash.KS)
    ap.add_argument("--out-dir", default="presentation_exports")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting presentation assets to: {out_dir}")
    print("If chart export fails, install: pip install vl-convert-python")
    print("If schema export fails, install: pip install graphviz  and install Graphviz binary (dot) on PATH")
    print("If table PNG export is skipped, install: pip install dataframe-image")

    # Charts + tables
    step_chart, step_tbl = build_step_chart(args.split, args.k)
    save_altair(step_chart, out_dir / f"stepwise_gains_{args.split}_k{args.k}.png", out_dir / f"stepwise_gains_{args.split}_k{args.k}.svg")
    save_table(step_tbl, out_dir / f"stepwise_gains_table_{args.split}_k{args.k}.png", out_dir / f"stepwise_gains_table_{args.split}_k{args.k}.csv")

    topk_chart = build_topk_chart(args.split)
    if topk_chart is not None:
        save_altair(topk_chart, out_dir / f"topk_behavior_{args.split}.png", out_dir / f"topk_behavior_{args.split}.svg")

    cov_chart, cov_tbl = build_coverage_chart_and_table(args.split, args.k)
    save_altair(cov_chart, out_dir / f"coverage_bars_{args.split}_k{args.k}.png", out_dir / f"coverage_bars_{args.split}_k{args.k}.svg")
    save_table(cov_tbl, out_dir / f"coverage_table_{args.split}_k{args.k}.png", out_dir / f"coverage_table_{args.split}_k{args.k}.csv")

    off = dash.official_metric_table(args.split)
    if not off.empty:
        off_tbl = off.pivot(index="label", columns="k", values="official_f1").reset_index().rename(columns={"label": "System"})
        save_table(off_tbl, out_dir / f"official_summary_{args.split}.png", out_dir / f"official_summary_{args.split}.csv")

    # Schemas
    save_graphviz_png(BIG_DOT, out_dir / "schema_big_picture.png")
    save_graphviz_png(GEN_DOT, out_dir / "schema_generator.png")
    save_graphviz_png(PAIRWISE_DOT, out_dir / "schema_pairwise_ranker.png")
    save_graphviz_png(FREQ_DOT, out_dir / "schema_frequency_reranker.png")

    print("Export complete.")


if __name__ == "__main__":
    main()

