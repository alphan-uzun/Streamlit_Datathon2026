from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import altair as alt
import pandas as pd
import streamlit as st

from compare_all_metrics import gold_oracle_topk_metrics
from data_io import parse_aws_baseline, parse_gold
from metrics import evaluate, score_prompt
from textnorm import normalize_text


ROOT = Path(".")
KS = [1, 2, 3, 5, 10, 20]

PATHS = {
    "dev_gold": ROOT / "staple-2020/en_hu/dev.en_hu.2020-02-20.gold.txt",
    "test_gold": ROOT / "staple-2020/en_hu/test.en_hu.2020-02-20.gold.txt",
    "dev_aws": ROOT / "staple-2020/en_hu/dev.en_hu.aws_baseline.pred.txt",
    "test_aws": ROOT / "staple-2020/en_hu/test.en_hu.aws_baseline.pred.txt",
    "dev_base": ROOT / "candidates_dev_base.json",
    "test_base": ROOT / "candidates_test_base.json",
    "dev_ft": ROOT / "candidates_dev_v34e3_big.json",
    "test_ft": ROOT / "candidates_test_v34e3_big.json",
    "dev_pairwise_soft": ROOT / "reranked_dev_v34e3_big_pairwise_noleak_v26a.json",
    "test_pairwise_soft": ROOT / "reranked_test_v34e3_big_pairwise_noleak_v26a.json",
    "dev_pairwise_anchor": ROOT / "reranked_dev_v34e3_big_pairwise_noleak_v26a_aws_anchor.json",
    "test_pairwise_anchor": ROOT / "reranked_test_v34e3_big_pairwise_noleak_v26a_aws_anchor.json",
    "dev_freq_soft": ROOT / "reranked_dev_v34e3_big_relevance_v27a.json",
    "test_freq_soft": ROOT / "reranked_test_v34e3_big_relevance_v27a.json",
    "dev_freq_anchor": ROOT / "reranked_dev_v34e3_big_relevance_v27a_aws_anchor.json",
    "test_freq_anchor": ROOT / "reranked_test_v34e3_big_relevance_v27a_aws_anchor.json",
    "official_dev_summary": ROOT / "official_scorer_dev_summary_all_systems.json",
    "official_test_summary": ROOT / "official_scorer_test_summary_all_systems.json",
    "deepl_worst5_test_freq_anchor_k5": ROOT / "deepl_backtranslation_worst5_test_v27a_anchor_k5.json",
    "deepl_worst15_test_freq_anchor_k5": ROOT / "deepl_backtranslation_worst15_test_v27a_anchor_k5.json",
    "deepl_worst15_dev_freq_anchor_k5": ROOT / "deepl_backtranslation_worst15_dev_v27a_anchor_k5.json",
}

SYSTEMS = [
    ("aws_top1", "AWS Baseline (1 Guess)", "single", None),
    ("base_raw", "Base Generator", "raw", "base"),
    ("ft_raw", "Fine-Tuned Generator", "raw", "ft"),
    ("pairwise_soft", "Pairwise Neural Ranker", "reranked", "pairwise_soft"),
    ("pairwise_anchor", "Pairwise Neural Ranker + AWS First Guess", "reranked", "pairwise_anchor"),
    ("freq_soft", "Frequency-Aware Reranker", "reranked", "freq_soft"),
    ("freq_anchor", "Frequency-Aware Reranker + AWS First Guess", "reranked", "freq_anchor"),
]

SYSTEM_COLORS = {
    "AWS Baseline (1 Guess)": "#4C78A8",
    "Base Generator": "#9AA0A6",
    "Fine-Tuned Generator": "#59A14F",
    "Pairwise Neural Ranker": "#F28E2B",
    "Pairwise Neural Ranker + AWS First Guess": "#E15759",
    "Frequency-Aware Reranker": "#B07AA1",
    "Frequency-Aware Reranker + AWS First Guess": "#D37295",
}
ORACLE_COLOR = "#2F2F2F"


@st.cache_data(show_spinner=False)
def load_json(path_str: str) -> Any:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_gold(path_str: str) -> Dict[str, Dict[str, Any]]:
    return parse_gold(path_str)


@st.cache_data(show_spinner=False)
def load_aws(path_str: str) -> Dict[str, str]:
    return parse_aws_baseline(path_str)


def _take_topk(cands: Dict[str, List[str]], k: int) -> Dict[str, List[str]]:
    return {pid: lst[:k] for pid, lst in cands.items()}


def _available_systems(split: str) -> List[dict]:
    rows = []
    for sid, label, kind, suffix in SYSTEMS:
        if kind == "single":
            rows.append({"id": sid, "label": label, "kind": kind})
            continue
        p = PATHS.get(f"{split}_{suffix}")
        if p and p.exists():
            rows.append({"id": sid, "label": label, "kind": kind, "path": p})
    return rows


def _preds_for_k(split: str, sys_row: dict, k: int, aws_map: Dict[str, str]) -> Dict[str, Sequence[str] | str]:
    if sys_row["kind"] == "single":
        return aws_map
    obj = load_json(sys_row["path"].as_posix())
    if sys_row["kind"] == "raw":
        return _take_topk(obj, k)
    return obj[str(k)]


@st.cache_data(show_spinner=False)
def local_metric_table(split: str) -> pd.DataFrame:
    gold = load_gold(PATHS[f"{split}_gold"].as_posix())
    aws = load_aws(PATHS[f"{split}_aws"].as_posix())
    rows: List[dict] = []
    for s in _available_systems(split):
        for k in KS:
            if s["id"] == "aws_top1" and k != 1:
                continue
            preds = _preds_for_k(split, s, k, aws)
            m = evaluate(gold, preds)
            oracle = gold_oracle_topk_metrics(gold, k)
            rows.append(
                {
                    "split": split,
                    "system_id": s["id"],
                    "label": s["label"],
                    "k": k,
                    "f1": m["macro_weighted_f1"],
                    "weighted_recall": m["macro_weighted_recall"],
                    "precision": m["macro_precision"],
                    "oracle_f1": oracle["macro_weighted_f1"],
                    "oracle_recall": oracle["macro_weighted_recall"],
                }
            )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def official_metric_table(split: str) -> pd.DataFrame:
    summary_path = PATHS[f"official_{split}_summary"]
    if not summary_path.exists():
        return pd.DataFrame()
    summary = load_json(summary_path.as_posix())
    by_system = summary.get("by_system") or {}

    desired: Dict[str, tuple[str, str]] = {"aws_top1": ("aws_top1", "AWS Baseline (1 Guess)")}
    for sid, label, kind, suffix in SYSTEMS:
        if kind == "single":
            continue
        p = PATHS.get(f"{split}_{suffix}")
        if p and p.exists():
            desired[p.stem] = (sid, label)

    rows = []
    for summary_key, (sid, label) in desired.items():
        block = by_system.get(summary_key)
        if not block:
            continue
        for k_str, m in (block.get("by_k") or {}).items():
            rows.append(
                {
                    "split": split,
                    "system_id": sid,
                    "label": label,
                    "k": int(k_str),
                    "official_f1": float(m["weighted_macro_f1"]),
                }
            )
    return pd.DataFrame(rows)


def coverage_stats(gold: Dict[str, Dict[str, Any]], preds: Dict[str, List[str]], k: int) -> Dict[str, float]:
    prompt_hit = 0
    best_gold_hit = 0
    weighted_recall_sum = 0.0
    macro_variant_cov_sum = 0.0
    total_gold_variants = 0
    total_gold_variants_hit = 0
    for pid, g in gold.items():
        gmap: Dict[str, float] = {}
        for t, w in zip(g["translations"], g["weights"]):
            nk = normalize_text(t)
            if nk:
                gmap[nk] = gmap.get(nk, 0.0) + float(w)
        pset = {normalize_text(x) for x in preds.get(pid, [])[:k] if normalize_text(x)}
        inter = pset.intersection(gmap.keys())
        if inter:
            prompt_hit += 1
        if gmap:
            max_w = max(gmap.values())
            best_keys = {nk for nk, w in gmap.items() if w == max_w}
            if pset.intersection(best_keys):
                best_gold_hit += 1
        weighted_recall_sum += sum(gmap[x] for x in inter)
        macro_variant_cov_sum += len(inter) / max(len(gmap), 1)
        total_gold_variants += len(gmap)
        total_gold_variants_hit += len(inter)
    n = max(len(gold), 1)
    return {
        "prompt_hit_rate": prompt_hit / n,
        "best_gold_hit_ratio": best_gold_hit / n,
        "macro_weighted_recall": weighted_recall_sum / n,
        "macro_variant_coverage": macro_variant_cov_sum / n,
        "micro_variant_coverage": total_gold_variants_hit / max(total_gold_variants, 1),
    }


def _pred_map(split: str, system_id: str, k: int) -> Dict[str, List[str]]:
    aws = load_aws(PATHS[f"{split}_aws"].as_posix())
    for s in _available_systems(split):
        if s["id"] == system_id:
            preds = _preds_for_k(split, s, k, aws)
            return {pid: ([x] if isinstance(x, str) else list(x)) for pid, x in preds.items()}
    return {}


@st.cache_data(show_spinner=False)
def worst_prompt_table(split: str, system_id: str, k: int) -> pd.DataFrame:
    gold = load_gold(PATHS[f"{split}_gold"].as_posix())
    preds = _pred_map(split, system_id, k)
    rows = []
    for pid, g in gold.items():
        s = score_prompt(g["translations"], g["weights"], preds.get(pid, []))
        rows.append(
            {
                "prompt_id": pid,
                "english": g["english"],
                "f1": s["weighted_f1"],
                "weighted_recall": s["weighted_recall"],
            }
        )
    return pd.DataFrame(rows).sort_values(["f1", "weighted_recall"], ascending=[True, True])


def render_story() -> None:
    st.header("1. What We Built")
    st.caption("Focus: improve the generator through fine-tuning, then improve final guesses through filtering.")
    st.graphviz_chart(
        """
        digraph G {
          rankdir=LR;
          graph [pad="0.25", nodesep="0.55", ranksep="0.7", bgcolor="white"];
          node [shape=box, style="rounded,filled", fontsize=14];
          p [label="English Prompt", fillcolor="white"];
          g [label="Generator\\n(Base NLLB + LoRA Fine-Tuning)", shape=component, fillcolor="#dff0d8", color="#59A14F", penwidth=2];
          f2 [label="Frequency-Aware Reranker", shape=component, fillcolor="#efe3f5", color="#B07AA1", penwidth=2];
          pol [label="Anchoring / Final Policy\\n(Optional AWS First Guess)", shape=diamond, fillcolor="#fbe0e3", color="#E15759", penwidth=2];
          o [label="Top-k Hungarian Guesses", fillcolor="white"];
          p -> g;
          g -> f2;
          f2 -> pol;
          pol -> o;
        }
        """
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Fine-Tuning (Generator)**")
        st.write(
            "- LoRA fine-tuning on STAPLE-style data\n"
            "- Goal: generate more valid Hungarian variants\n"
            "- Main effect: better coverage/diversity"
        )
    with c2:
        st.markdown("**Filtering (Reranking)**")
        st.write(
            "- Neural rerankers select better final guesses\n"
            "- Uses embeddings + structured features\n"
            "- Main effect: better top-k selection quality"
        )

    st.markdown("### Visual Legend")
    legend_items = [
        ("AWS Baseline (1 Guess)", SYSTEM_COLORS["AWS Baseline (1 Guess)"]),
        ("Base Generator", SYSTEM_COLORS["Base Generator"]),
        ("Fine-Tuned Generator", SYSTEM_COLORS["Fine-Tuned Generator"]),
        ("Frequency-Aware Reranker", SYSTEM_COLORS["Frequency-Aware Reranker"]),
        ("Anchored Final Step (+ AWS First Guess)", SYSTEM_COLORS["Frequency-Aware Reranker + AWS First Guess"]),
        ("Oracle", ORACLE_COLOR),
    ]
    lcols = st.columns(3)
    for i, (name, color) in enumerate(legend_items):
        lcols[i % 3].markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin:4px 0;'>"
            f"<span style='display:inline-block;width:12px;height:12px;border-radius:2px;background:{color};'></span>"
            f"<span>{name}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("### Click to Inspect Internals (Optional)")
    if "cg_arch_view" not in st.session_state:
        st.session_state["cg_arch_view"] = None
    b1, b2 = st.columns(2)
    if b1.button("Generator Internals", use_container_width=True):
        st.session_state["cg_arch_view"] = "generator"
    if b2.button("Frequency Reranker Internals", use_container_width=True):
        st.session_state["cg_arch_view"] = "freq"

    view = st.session_state.get("cg_arch_view")
    if view == "generator":
        st.graphviz_chart(
            """
            digraph GEN {
              rankdir=TB; node [shape=box, style=rounded];
              p [label="English Prompt"];
              d [label="STAPLE Training Data\\n(weighted sampling)", shape=folder];
              base [label="Base NLLB"];
              lora [label="LoRA Adapters", style="rounded,filled", fillcolor="#fff2cc"];
              ft [label="Fine-Tuned Generator", style="rounded,filled", fillcolor="#e8f5e9"];
              beam [label="Beam Decode"];
              samp [label="Sampling Decode"];
              dd [label="Merge + Dedupe"];
              out [label="Candidate Pool"];
              d -> lora [label="finetunes"];
              base -> ft; lora -> ft; p -> ft;
              ft -> beam -> dd; ft -> samp -> dd; dd -> out;
            }
            """
        )
    elif view == "pairwise":
        st.graphviz_chart(
            """
            digraph P {
              rankdir=TB; node [shape=box, style=rounded];
              p [label="Prompt"]; a [label="AWS Translation"]; c [label="Candidates"];
              e [label="Embedding Encoder"];
              g [label="Candidate-Pool Geometry\\n(unsupervised centers)"];
              f [label="Feature Builder Neural Network"];
              r [label="Pairwise MLP Ranker"];
              s [label="Scores -> Sort"];
              p -> e; a -> e; c -> e; c -> g; e -> f; g -> f; f -> r -> s;
            }
            """
        )
    elif view == "freq":
        st.graphviz_chart(
            """
            digraph F {
              rankdir=TB; node [shape=box, style=rounded];
              p [label="Prompt"]; a [label="AWS Translation"]; c [label="Candidates"];
              e [label="Embedding Encoder"];
              f [label="Feature Builder Neural Network"];
              r [label="MLP Regressor\\n(predicted relevance)"];
              s [label="Scores -> Sort"];
              t [label="Training target:\\ngold weight or 0", shape=note];
              p -> e; a -> e; c -> e; e -> f; f -> r -> s; t -> r;
            }
            """
        )


def render_cumulative_gains() -> None:
    st.header("2. Cumulative Gains (Metrics)")
    st.info(
        "Main metric: weighted macro F1. It balances precision (avoiding extra incorrect guesses) and weighted recall "
        "(covering high-frequency/important gold translations), averaged across prompts."
    )
    split = st.radio("Split", ["test", "dev"], horizontal=True, key="cg_split")
    k = st.selectbox("k (main presentation setting)", KS, index=3, key="cg_k")

    odf = official_metric_table(split)
    if odf.empty:
        st.warning("Official scorer summary not found for this split.")
        return

    step_order = [
        "AWS Baseline (1 Guess)",
        "Base Generator",
        "Fine-Tuned Generator",
        "Frequency-Aware Reranker",
        "Frequency-Aware Reranker + AWS First Guess",
    ]
    step_df = odf[odf["k"] == k].copy()
    step_df = step_df[step_df["label"].isin(step_order)]
    step_df["label"] = pd.Categorical(step_df["label"], categories=step_order, ordered=True)
    step_df = step_df.sort_values("label")

    st.markdown("**Step-by-step improvement view (including anchoring as a final step)**")
    if not step_df.empty:
        step_colors = [SYSTEM_COLORS.get(x, "#888888") for x in step_order]
        step_chart = (
            alt.Chart(step_df)
            .mark_bar(size=38, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("label:N", title=None, sort=step_order, axis=alt.Axis(labelAngle=-20, labelLimit=220)),
                y=alt.Y("official_f1:Q", title="Official Weighted Macro F1"),
                color=alt.Color(
                    "label:N",
                    title=None,
                    sort=step_order,
                    scale=alt.Scale(domain=step_order, range=step_colors),
                ),
                tooltip=["label", "k", alt.Tooltip("official_f1:Q", format=".4f")],
            )
            .properties(height=300)
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
        combined_step_chart = step_chart + labels
        st.altair_chart(combined_step_chart, use_container_width=True)

        # Show incremental gains explicitly with presentation-friendly indicators (relative percent change).
        step_tmp = step_df.copy().reset_index(drop=True)
        step_tmp["delta_vs_prev"] = step_tmp["official_f1"].diff()
        step_tmp["pct_change_vs_prev"] = step_tmp["official_f1"].pct_change()
        st.markdown("**Incremental gains (relative % change vs previous step)**")
        metric_cols = st.columns(min(len(step_tmp), 5))
        for i, row in step_tmp.iterrows():
            col = metric_cols[i] if i < len(metric_cols) else st.columns(1)[0]
            f1_val = float(row["official_f1"])
            if pd.isna(row["pct_change_vs_prev"]):
                delta_txt = None
            else:
                delta_txt = f"{100.0 * float(row['pct_change_vs_prev']):+.2f}%"
            col.metric(str(row["label"]), f"{f1_val:.3f}", delta_txt)

        show_tmp = step_tmp[["label", "official_f1", "delta_vs_prev", "pct_change_vs_prev"]].copy()
        show_tmp["official_f1"] = show_tmp["official_f1"].map(lambda x: f"{100*x:.2f}%")
        show_tmp["delta_vs_prev"] = show_tmp["delta_vs_prev"].map(lambda x: "" if pd.isna(x) else f"{100*x:+.2f} pts")
        show_tmp["pct_change_vs_prev"] = show_tmp["pct_change_vs_prev"].map(lambda x: "" if pd.isna(x) else f"{100*x:+.2f}%")
        show_tmp = show_tmp.rename(
            columns={
                "label": "Step",
                "official_f1": "Official Weighted Macro F1",
                "delta_vs_prev": "Absolute Change (pts)",
                "pct_change_vs_prev": "Relative Change (%)",
            }
        )
        with st.expander("Show step table"):
            st.dataframe(show_tmp, use_container_width=True, hide_index=True)

    pretty = odf[odf["k"] == k].copy().sort_values("official_f1", ascending=False)
    pretty = pretty.rename(columns={"label": "System", "official_f1": "Official Weighted Macro F1"})
    st.dataframe(pretty[["System", "Official Weighted Macro F1"]], use_container_width=True, hide_index=True)

    st.markdown("**Top-k behavior (official scorer)**")
    line_order = step_order + ["Pairwise Neural Ranker", "Pairwise Neural Ranker + AWS First Guess"]
    line_order = [x for x in line_order if x in set(odf["label"].tolist())]
    default_lines = [x for x in [
        "Base Generator",
        "Fine-Tuned Generator",
        "Frequency-Aware Reranker",
        "Frequency-Aware Reranker + AWS First Guess",
        "AWS Baseline (1 Guess)",
    ] if x in line_order]
    selected_lines = st.multiselect(
        "Lines to show",
        options=line_order,
        default=default_lines if default_lines else line_order,
        key="cg_topk_lines",
    )
    line_df = odf[odf["label"].isin(selected_lines)].copy()
    if not line_df.empty:
        y_min = float(line_df["official_f1"].min())
        y_max = float(line_df["official_f1"].max())
        pad = max((y_max - y_min) * 0.15, 0.01)
        line_chart = (
            alt.Chart(line_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("k:O", title="k"),
                y=alt.Y(
                    "official_f1:Q",
                    title="Official Weighted Macro F1",
                    scale=alt.Scale(domain=[max(0.0, y_min - pad), min(1.0, y_max + pad)]),
                ),
                color=alt.Color(
                    "label:N",
                    sort=selected_lines,
                    title=None,
                    scale=alt.Scale(
                        domain=selected_lines,
                        range=[SYSTEM_COLORS.get(x, "#888888") for x in selected_lines],
                    ),
                ),
                tooltip=["label", "k", alt.Tooltip("official_f1:Q", format=".4f")],
            )
            .properties(height=320)
        )
        st.altair_chart(line_chart, use_container_width=True)


def render_coverage() -> None:
    st.header("3. Coverage (Why It Improves)")
    st.caption("Coverage shows how many gold variants and how much gold frequency mass the guesses capture.")

    split = st.radio("Split", ["dev", "test"], horizontal=True, key="cov_split_pres")
    k = st.selectbox("k", KS, index=3, key="cov_k_pres")
    available = _available_systems(split)
    by_id = {x["id"]: x for x in available}
    opts = [x["id"] for x in available if x["id"] != "aws_top1"]
    if not opts:
        st.warning("No systems available.")
        return

    default_selected = "freq_anchor" if "freq_anchor" in opts else ("pairwise_anchor" if "pairwise_anchor" in opts else "ft_raw")
    default_baseline = "base_raw" if "base_raw" in opts else opts[0]
    c1, c2 = st.columns(2)
    with c1:
        selected_id = st.selectbox("Selected system", opts, index=opts.index(default_selected), format_func=lambda x: by_id[x]["label"])
    with c2:
        baseline_id = st.selectbox("Baseline system", opts, index=opts.index(default_baseline), format_func=lambda x: by_id[x]["label"])

    gold = load_gold(PATHS[f"{split}_gold"].as_posix())
    sel_preds = _pred_map(split, selected_id, k)
    base_preds = _pred_map(split, baseline_id, k)
    oracle_preds = {
        pid: [t for t, _w in sorted(zip(g["translations"], g["weights"]), key=lambda x: float(x[1]), reverse=True)[:k]]
        for pid, g in gold.items()
    }
    sel_cov = coverage_stats(gold, sel_preds, k)
    base_cov = coverage_stats(gold, base_preds, k)
    oracle_cov = coverage_stats(gold, oracle_preds, k)

    def _rel_delta_pct(sel: float, base_val: float) -> str:
        if abs(base_val) < 1e-12:
            return "n/a"
        return f"{((sel - base_val) / base_val) * 100:+.1f}%"

    st.markdown("**Coverage change vs baseline (relative %)**")
    dcols = st.columns(4)
    dcols[0].metric(
        "Gold Variant Coverage",
        f"{100*sel_cov['macro_variant_coverage']:.1f}%",
        _rel_delta_pct(sel_cov["macro_variant_coverage"], base_cov["macro_variant_coverage"]),
    )
    dcols[1].metric(
        "Gold Frequency Covered",
        f"{100*sel_cov['macro_weighted_recall']:.1f}%",
        _rel_delta_pct(sel_cov["macro_weighted_recall"], base_cov["macro_weighted_recall"]),
    )
    dcols[2].metric(
        "Best Gold Found",
        f"{100*sel_cov['best_gold_hit_ratio']:.1f}%",
        _rel_delta_pct(sel_cov["best_gold_hit_ratio"], base_cov["best_gold_hit_ratio"]),
    )
    dcols[3].metric(
        "Prompt Hit Rate",
        f"{100*sel_cov['prompt_hit_rate']:.1f}%",
        _rel_delta_pct(sel_cov["prompt_hit_rate"], base_cov["prompt_hit_rate"]),
    )

    cov_tbl = pd.DataFrame(
        [
            {
                "System": by_id[baseline_id]["label"],
                "Gold Variant Coverage %": 100 * base_cov["macro_variant_coverage"],
                "Gold Frequency Covered %": 100 * base_cov["macro_weighted_recall"],
                "Prompt Hit Rate %": 100 * base_cov["prompt_hit_rate"],
                "Best Gold Found %": 100 * base_cov["best_gold_hit_ratio"],
            },
            {
                "System": by_id[selected_id]["label"],
                "Gold Variant Coverage %": 100 * sel_cov["macro_variant_coverage"],
                "Gold Frequency Covered %": 100 * sel_cov["macro_weighted_recall"],
                "Prompt Hit Rate %": 100 * sel_cov["prompt_hit_rate"],
                "Best Gold Found %": 100 * sel_cov["best_gold_hit_ratio"],
            },
            {
                "System": f"Oracle @ {k}",
                "Gold Variant Coverage %": 100 * oracle_cov["macro_variant_coverage"],
                "Gold Frequency Covered %": 100 * oracle_cov["macro_weighted_recall"],
                "Prompt Hit Rate %": 100 * oracle_cov["prompt_hit_rate"],
                "Best Gold Found %": 100 * oracle_cov["best_gold_hit_ratio"],
            },
        ]
    )
    st.dataframe(cov_tbl, use_container_width=True, hide_index=True)

    plot_df = cov_tbl.melt(
        id_vars=["System"],
        value_vars=["Gold Variant Coverage %", "Gold Frequency Covered %", "Prompt Hit Rate %", "Best Gold Found %"],
        var_name="Metric",
        value_name="Percent",
    )
    order = [r for r in cov_tbl["System"].tolist() if not str(r).startswith("Oracle @")] + [r for r in cov_tbl["System"].tolist() if str(r).startswith("Oracle @")]
    cov_color_domain = order
    cov_color_range = [SYSTEM_COLORS.get(x, ORACLE_COLOR if str(x).startswith("Oracle @") else "#888888") for x in cov_color_domain]
    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("Metric:N", title=None),
            xOffset=alt.XOffset("System:N", sort=order),
            y=alt.Y("Percent:Q", title="Percent"),
            color=alt.Color(
                "System:N",
                sort=order,
                title=None,
                scale=alt.Scale(domain=cov_color_domain, range=cov_color_range),
            ),
            tooltip=["System", "Metric", alt.Tooltip("Percent:Q", format=".2f")],
        )
        .properties(height=330)
    )
    st.altair_chart(chart, use_container_width=True)


def render_error_examples() -> None:
    st.header("4. Error Examples (Outputs Still Make Sense)")
    st.caption("Worst prompts for Frequency-Aware Reranker + AWS First Guess with model guesses (HU) and DeepL back-translations (EN).")

    split = st.radio("Split", ["test", "dev"], horizontal=True, key="err_split_pres")
    available = _available_systems(split)
    by_id = {x["id"]: x for x in available}
    opts = [x["id"] for x in available if x["id"] != "aws_top1"]
    if not opts:
        st.warning("No systems available.")
        return
    system_id = "freq_anchor" if "freq_anchor" in opts else None
    if system_id is None:
        st.warning("Frequency-Aware Reranker + AWS First Guess results are not available for this split.")
        return
    k = 5
    st.caption(f"Model: {by_id[system_id]['label']} | k={k}")
    n_show = st.slider("Worst prompts", 3, 15, 5, 1, key="err_n_pres")

    gold = load_gold(PATHS[f"{split}_gold"].as_posix())
    preds = _pred_map(split, system_id, k)
    worst_df = worst_prompt_table(split, system_id, k)

    deepl_index: Dict[str, Any] = {}
    deepl_path = None
    if system_id == "freq_anchor" and k == 5:
        if split == "test":
            if PATHS["deepl_worst15_test_freq_anchor_k5"].exists():
                deepl_path = PATHS["deepl_worst15_test_freq_anchor_k5"]
            elif PATHS["deepl_worst5_test_freq_anchor_k5"].exists():
                deepl_path = PATHS["deepl_worst5_test_freq_anchor_k5"]
        elif split == "dev" and PATHS["deepl_worst15_dev_freq_anchor_k5"].exists():
            deepl_path = PATHS["deepl_worst15_dev_freq_anchor_k5"]
    if deepl_path is not None:
        rows = load_json(deepl_path.as_posix())
        if isinstance(rows, list):
            deepl_index = {str(r.get("prompt_id")): r for r in rows if isinstance(r, dict)}

    st.dataframe(worst_df.head(n_show)[["english", "f1", "weighted_recall"]].rename(columns={"english": "Prompt", "f1": "Prompt F1", "weighted_recall": "Weighted Recall"}), use_container_width=True, hide_index=True)

    for pid in worst_df.head(n_show)["prompt_id"].tolist():
        g = gold[str(pid)]
        pred_list = list(preds.get(str(pid), []))
        s = score_prompt(g["translations"], g["weights"], pred_list)
        st.markdown("---")
        st.markdown(f"### {g['english']}")
        m1, m2 = st.columns(2)
        m1.metric("Prompt F1", f"{s['weighted_f1']:.3f}")
        m2.metric("Weighted Recall", f"{s['weighted_recall']:.3f}")

        bt_map = {}
        bt_item = deepl_index.get(str(pid))
        if bt_item:
            for r in bt_item.get("predictions", []) or []:
                bt_map[str(r.get("hu", ""))] = str(r.get("back_en", ""))

        for idx, hu in enumerate(pred_list[:k], start=1):
            left, right = st.columns(2)
            with left:
                st.markdown(f"**Guess {idx} (HU)**")
                st.write(hu)
            with right:
                st.markdown(f"**DeepL Back-Translation {idx} (EN)**")
                st.write(bt_map.get(hu, "(not available)"))

    if system_id == "freq_anchor" and k == 5 and not deepl_index:
        if split == "test":
            st.info(
                "To show DeepL back-translations here, save "
                "`deepl_backtranslation_worst15_test_v27a_anchor_k5.json` "
                "(or fallback `deepl_backtranslation_worst5_test_v27a_anchor_k5.json`) in the project root."
            )
        else:
            st.info(
                "To show DeepL back-translations here, save "
                "`deepl_backtranslation_worst15_dev_v27a_anchor_k5.json` in the project root."
            )


def render_conclusion() -> None:
    st.header("5. Conclusion")
    st.caption("Presentation-ready summary of the main findings and next steps.")

    split = st.radio("Split", ["test", "dev"], horizontal=True, key="conc_split")
    k = st.selectbox("k (summary view)", KS, index=3, key="conc_k")

    odf = official_metric_table(split)
    ldf = local_metric_table(split)
    if odf.empty or ldf.empty:
        st.warning("Missing official or local summary data for this split.")
        return

    def get_official(label: str, k_: int) -> float | None:
        m = odf[(odf["label"] == label) & (odf["k"] == k_)]
        return None if m.empty else float(m.iloc[0]["official_f1"])

    def get_local(label: str, k_: int, col: str) -> float | None:
        m = ldf[(ldf["label"] == label) & (ldf["k"] == k_)]
        if m.empty or col not in m.columns:
            return None
        return float(m.iloc[0][col])

    base_label = "Base Generator"
    ft_label = "Fine-Tuned Generator"
    final_label = "Frequency-Aware Reranker + AWS First Guess"

    base_f1 = get_official(base_label, k)
    ft_f1 = get_official(ft_label, k)
    final_f1 = get_official(final_label, k)

    c1, c2, c3 = st.columns(3)
    if base_f1 is not None and ft_f1 is not None:
        c1.metric(
            "Fine-Tuning Gain vs Base",
            f"{ft_f1:.3f}",
            f"{100.0 * ((ft_f1 - base_f1) / base_f1):+.2f}%" if base_f1 > 0 else None,
        )
    if ft_f1 is not None and final_f1 is not None:
        c2.metric(
            "Filtering + Policy Gain vs Fine-Tuned",
            f"{final_f1:.3f}",
            f"{100.0 * ((final_f1 - ft_f1) / ft_f1):+.2f}%" if ft_f1 > 0 else None,
        )
    if base_f1 is not None and final_f1 is not None:
        c3.metric(
            "Total Gain vs Base",
            f"{final_f1:.3f}",
            f"{100.0 * ((final_f1 - base_f1) / base_f1):+.2f}%" if base_f1 > 0 else None,
        )

    st.markdown("### Key Takeaways")
    takeaways = [
        "Fine-tuning improves the generator by increasing useful translation coverage (more valid Hungarian variants appear in the candidate pool).",
        "Filtering/reranking improves final selection quality by choosing better guesses from the generated candidates.",
        "Coverage analysis shows gains are not only metric artifacts: the system covers more gold variants and more gold frequency mass.",
        "Error analysis shows some low-scoring outputs are still semantically reasonable (supported by DeepL back-translation), suggesting gold-set coverage limits in some cases.",
    ]
    for t in takeaways:
        st.write(f"- {t}")

    st.markdown("### Evidence Snapshot (selected k)")
    cov_base = get_local(base_label, k, "weighted_recall")
    cov_final = get_local(final_label, k, "weighted_recall")
    var_base = get_local(base_label, k, "oracle_recall")  # not exact variant coverage, but useful context
    if cov_base is not None and cov_final is not None:
        st.write(
            f"- Gold frequency covered (weighted recall): `{cov_base:.3f}` -> `{cov_final:.3f}` "
            f"({100.0 * ((cov_final - cov_base) / cov_base):+.2f}% relative)"
        )
    if base_f1 is not None and final_f1 is not None:
        st.write(
            f"- Official weighted macro F1: `{base_f1:.3f}` -> `{final_f1:.3f}` "
            f"({100.0 * ((final_f1 - base_f1) / base_f1):+.2f}% relative)"
        )
    if var_base is not None:
        st.caption("Oracle values remain above current results, which indicates further headroom for better candidate coverage and/or reranking.")

    st.markdown("### Next Steps")
    next_steps = [
        "Increase candidate diversity further (e.g., larger candidate pools) while controlling noise.",
        "Improve filtering/reranking calibration and confidence-based policies.",
        "Continue qualitative error analysis to separate true translation errors from gold-set omissions.",
    ]
    for n in next_steps:
        st.write(f"- {n}")

    st.markdown("### Limitations (Important Context)")
    limitations = [
        "Some low-scoring predictions may still be semantically valid translations but are not matched in the gold set.",
        "DeepL back-translation checks are useful qualitative evidence, but they are heuristic and not a replacement for the official scorer.",
        "Current results still show oracle headroom, so both candidate diversity and final selection can be improved further.",
    ]
    for item in limitations:
        st.write(f"- {item}")

    st.markdown("### Future Work: Paraphrase / Agent Expansion")
    future = [
        "Add a paraphrasing model (or LLM-based module) to generate additional Hungarian variants and increase candidate diversity.",
        "Use an agent-style loop for hard prompts: generate -> critique -> paraphrase -> rerank.",
        "Use back-translation and semantic similarity to flag likely 'valid but unlisted' outputs for targeted analysis or data augmentation.",
        "Incorporate paraphrase-expanded training targets during fine-tuning to improve coverage of acceptable variants.",
    ]
    for item in future:
        st.write(f"- {item}")


def main() -> None:
    st.set_page_config(page_title="STAPLE Cumulative Gains Presentation", layout="wide")
    st.title("STAPLE EN→HU: Cumulative Gains Presentation")
    st.caption("Presentation-focused dashboard: pipeline, cumulative gains, coverage, and qualitative error examples.")
    st.info(
        "Goal: given an English prompt, produce a high-coverage set of valid Hungarian translations. "
        "This presentation focuses on cumulative gains from fine-tuning (better candidate generation) and filtering/reranking (better final selection)."
    )

    if not PATHS["dev_gold"].exists():
        st.error("Required dataset files not found.")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Model & Method", "Metrics & Gains", "Coverage", "Error Examples", "Conclusion"])
    with tab1:
        render_story()
    with tab2:
        render_cumulative_gains()
    with tab3:
        render_coverage()
    with tab4:
        render_error_examples()
    with tab5:
        render_conclusion()


if __name__ == "__main__":
    main()
