from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import altair as alt
import pandas as pd
import streamlit as st

from compare_all_metrics import gold_oracle_topk_metrics
from data_io import parse_aws_baseline, parse_gold, parse_prompts
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
    "dev_v26a": ROOT / "reranked_dev_v34e3_big_pairwise_noleak_v26a.json",
    "test_v26a": ROOT / "reranked_test_v34e3_big_pairwise_noleak_v26a.json",
    "dev_v26a_anchor": ROOT / "reranked_dev_v34e3_big_pairwise_noleak_v26a_aws_anchor.json",
    "test_v26a_anchor": ROOT / "reranked_test_v34e3_big_pairwise_noleak_v26a_aws_anchor.json",
    "dev_v27a": ROOT / "reranked_dev_v34e3_big_relevance_v27a.json",
    "dev_v27a_anchor": ROOT / "reranked_dev_v34e3_big_relevance_v27a_aws_anchor.json",
    "test_v27a": ROOT / "reranked_test_v34e3_big_relevance_v27a.json",
    "test_v27a_anchor": ROOT / "reranked_test_v34e3_big_relevance_v27a_aws_anchor.json",
    "official_dev_summary": ROOT / "official_scorer_dev_summary_all_systems.json",
    "official_test_summary": ROOT / "official_scorer_test_summary_all_systems.json",
    "deepl_bt_test_v27a_anchor_k5": ROOT / "deepl_backtranslation_worst5_test_v27a_anchor_k5.json",
    "deepl_bt_test_groups_k5_extended": ROOT / "deepl_backtranslation_test_groups_k5_extended.json",
    "deepl_bt_similarity_summary": ROOT / "deepl_backtranslation_worst_bundle_similarity_summary.json",
}


SYSTEM_CATALOG = [
    ("aws_top1", "AWS Baseline (1 Guess)", "single", None, True),
    ("base_raw", "Base Generator", "raw", "base", True),
    ("ft_raw", "Fine-Tuned Generator", "raw", "ft", True),
    ("v26a_soft", "Neural Ranker (Pairwise)", "reranked", "v26a", True),
    ("v26a_anchor", "Neural Ranker + AWS First Guess", "reranked", "v26a_anchor", True),
    ("v27a_soft", "Frequency-Aware Reranker", "reranked", "v27a", True),
    ("v27a_anchor", "Frequency-Aware Reranker + AWS First Guess", "reranked", "v27a_anchor", True),
]


@st.cache_data(show_spinner=False)
def load_gold(path_str: str) -> Dict[str, Dict[str, Any]]:
    return parse_gold(path_str)


@st.cache_data(show_spinner=False)
def load_prompts(path_str: str) -> Dict[str, str]:
    return parse_prompts(path_str)


@st.cache_data(show_spinner=False)
def load_aws(path_str: str) -> Dict[str, str]:
    return parse_aws_baseline(path_str)


@st.cache_data(show_spinner=False)
def load_json(path_str: str) -> Any:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def _take_topk(cands: Dict[str, List[str]], k: int) -> Dict[str, List[str]]:
    return {pid: lst[:k] for pid, lst in cands.items()}


def _available_systems(split: str) -> List[dict]:
    rows = []
    for sys_id, label, kind, key_suffix, valid in SYSTEM_CATALOG:
        if kind == "single":
            rows.append({"id": sys_id, "label": label, "kind": kind, "valid": valid})
            continue
        p = PATHS.get(f"{split}_{key_suffix}")
        if p and p.exists():
            rows.append({"id": sys_id, "label": label, "kind": kind, "path": p, "valid": valid})
    return rows


def _system_preds_for_k(split: str, sys_row: dict, k: int, aws_map: Dict[str, str]) -> Dict[str, Sequence[str] | str]:
    if sys_row["kind"] == "single":
        return aws_map
    obj = load_json(sys_row["path"].as_posix())
    if sys_row["kind"] == "raw":
        return _take_topk(obj, k)
    return obj[str(k)]


@st.cache_data(show_spinner=False)
def build_metric_table(split: str) -> pd.DataFrame:
    gold_path = PATHS[f"{split}_gold"]
    aws_path = PATHS[f"{split}_aws"]
    gold = load_gold(gold_path.as_posix())
    aws = load_aws(aws_path.as_posix())
    systems = _available_systems(split)
    rows: List[dict] = []
    for sys_row in systems:
        for k in KS:
            if sys_row["id"] == "aws_top1" and k != 1:
                continue
            preds = _system_preds_for_k(split, sys_row, k, aws)
            m = evaluate(gold, preds)
            oracle = gold_oracle_topk_metrics(gold, k)
            rows.append(
                {
                    "split": split,
                    "system_id": sys_row["id"],
                    "label": sys_row["label"],
                    "k": k,
                    "macro_weighted_f1": m["macro_weighted_f1"],
                    "macro_weighted_recall": m["macro_weighted_recall"],
                    "macro_precision": m["macro_precision"],
                    "oracle_f1": oracle["macro_weighted_f1"],
                    "oracle_recall": oracle["macro_weighted_recall"],
                    "oracle_gap_f1": oracle["macro_weighted_f1"] - m["macro_weighted_f1"],
                    "valid": sys_row.get("valid", True),
                }
            )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def build_official_metric_table(split: str) -> pd.DataFrame:
    summary_path = PATHS.get(f"official_{split}_summary")
    if not summary_path or not summary_path.exists():
        return pd.DataFrame()

    summary = load_json(summary_path.as_posix())
    by_system = summary.get("by_system") or {}

    desired_map: Dict[str, str] = {"aws_top1": "aws_top1"}
    for sys_id, label, kind, key_suffix, _valid in SYSTEM_CATALOG:
        if kind == "single":
            continue
        p = PATHS.get(f"{split}_{key_suffix}")
        if p and p.exists():
            desired_map[p.stem] = sys_id

    by_id = {sid: label for sid, label, *_ in SYSTEM_CATALOG}
    rows: List[dict] = []
    for summary_key, sys_id in desired_map.items():
        block = by_system.get(summary_key)
        if not block:
            continue
        for k_str, m in (block.get("by_k") or {}).items():
            rows.append(
                {
                    "split": split,
                    "system_id": sys_id,
                    "label": by_id.get(sys_id, block.get("label", summary_key)),
                    "k": int(k_str),
                    "official_weighted_macro_f1": float(m["weighted_macro_f1"]),
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
            gmap[nk] = gmap.get(nk, 0.0) + float(w)
        pset = {normalize_text(x) for x in preds.get(pid, [])[:k] if normalize_text(x)}
        inter = pset.intersection(gmap.keys())
        if inter:
            prompt_hit += 1
        # "Best gold found" = at least one highest-weight gold variant is present.
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


@st.cache_data(show_spinner=False)
def _pred_map_for_examples(split: str, system_id: str, k: int) -> Dict[str, List[str]]:
    aws = load_aws(PATHS[f"{split}_aws"].as_posix())
    for s in _available_systems(split):
        if s["id"] == system_id:
            preds = _system_preds_for_k(split, s, k, aws)
            return {pid: ([x] if isinstance(x, str) else list(x)) for pid, x in preds.items()}
    return {}


@st.cache_data(show_spinner=False)
def build_prompt_delta_table(split: str, compare_system_id: str, baseline_system_id: str, k: int) -> pd.DataFrame:
    gold = load_gold(PATHS[f"{split}_gold"].as_posix())
    comp = _pred_map_for_examples(split, compare_system_id, k)
    base = _pred_map_for_examples(split, baseline_system_id, k)
    rows = []
    for pid, g in gold.items():
        s_base = score_prompt(g["translations"], g["weights"], base.get(pid, []))
        s_cmp = score_prompt(g["translations"], g["weights"], comp.get(pid, []))
        rows.append(
            {
                "prompt_id": pid,
                "english": g["english"],
                "base_f1": s_base["weighted_f1"],
                "cmp_f1": s_cmp["weighted_f1"],
                "delta_f1": s_cmp["weighted_f1"] - s_base["weighted_f1"],
                "base_rec": s_base["weighted_recall"],
                "cmp_rec": s_cmp["weighted_recall"],
                "delta_rec": s_cmp["weighted_recall"] - s_base["weighted_recall"],
            }
        )
    return pd.DataFrame(rows).sort_values("delta_f1", ascending=False)


@st.cache_data(show_spinner=False)
def cached_coverage_stats(split: str, system_id: str, k: int) -> Dict[str, float]:
    gold = load_gold(PATHS[f"{split}_gold"].as_posix())
    preds = _pred_map_for_examples(split, system_id, k)
    out = coverage_stats(gold, preds, k)
    # Backward-safe defaults in case Streamlit cache serves an older shape.
    out.setdefault("best_gold_hit_ratio", 0.0)
    return out


def render_overview() -> None:
    st.subheader("Model Comparison")
    split = st.radio("Split", ["dev", "test"], horizontal=True, key="overview_split")
    df = build_official_metric_table(split)
    if df.empty:
        st.warning("No official scorer summary found for this split. Build the summary JSON first.")
        return

    st.caption("Overview uses only the Duolingo/STAPLE official scorer values.")
    st.info(
        "STAPLE uses weighted macro F1 as the main score because each prompt has multiple accepted translations with weights. "
        "The metric rewards covering high-weight valid translations (weighted recall) while penalizing extra incorrect guesses (precision), "
        "so it is a good balance for this task's set prediction objective."
    )
    k_sel = st.selectbox("Metric table k", KS, index=3, key="overview_k")
    table_k = df[df["k"] == k_sel].copy().sort_values("official_weighted_macro_f1", ascending=False)
    table_k = table_k.rename(columns={"official_weighted_macro_f1": "Official Weighted Macro F1"})
    st.dataframe(
        table_k[["label", "Official Weighted Macro F1"]].rename(columns={"label": "System"}),
        use_container_width=True,
        hide_index=True,
    )

    pivot = df.pivot(index="k", columns="label", values="official_weighted_macro_f1")
    st.line_chart(pivot)
    st.markdown("### Official Scorer Table (all k)")
    full_tbl = df.pivot(index="label", columns="k", values="official_weighted_macro_f1").reset_index()
    full_tbl = full_tbl.rename(columns={"label": "System"})
    st.dataframe(full_tbl, use_container_width=True, hide_index=True)


def render_coverage_and_examples() -> None:
    st.subheader("Coverage and Prompt Examples")
    split = st.radio("Split", ["dev", "test"], horizontal=True, key="coverage_split")
    available = _available_systems(split)
    by_id = {x["id"]: x for x in available}
    inspect_options = [x["id"] for x in available if x["id"] != "aws_top1"]
    if not inspect_options:
        st.warning("No model outputs found for this split.")
        return

    default_sys = "v27a_anchor" if "v27a_anchor" in by_id else ("v26a_anchor" if "v26a_anchor" in by_id else "ft_raw")
    default_sys_idx = inspect_options.index(default_sys) if default_sys in inspect_options else 0

    ctrl_left, ctrl_right = st.columns([1, 2])
    with ctrl_left:
        system_id = st.selectbox(
            "Selected model",
            options=inspect_options,
            index=default_sys_idx,
            format_func=lambda sid: by_id[sid]["label"],
            key="coverage_system",
        )
        k = st.selectbox("k", KS, index=3, key="coverage_k")
        baseline_candidates = [x["id"] for x in available]
        baseline_default = "ft_raw" if "ft_raw" in baseline_candidates else baseline_candidates[0]
        baseline_id = st.selectbox(
            "Baseline for comparison",
            options=baseline_candidates,
            index=baseline_candidates.index(baseline_default),
            format_func=lambda sid: by_id[sid]["label"] if sid in by_id else sid,
            key="coverage_baseline",
        )
        mode = st.radio(
            "Example mode",
            ["Best improved", "Worst degraded", "Search prompt"],
            key="coverage_mode",
        )
        n_examples = st.slider("How many examples", min_value=3, max_value=20, value=5, step=1, key="coverage_n_examples")
    with ctrl_right:
        st.caption("Coverage summary focused on: (1) how many gold variants are found, and (2) how much gold frequency weight is covered.")
        metric_df = build_metric_table(split)
        row_sel = metric_df[(metric_df["system_id"] == system_id) & (metric_df["k"] == k)]
        row_base = metric_df[(metric_df["system_id"] == baseline_id) & (metric_df["k"] == k)]
        cov = cached_coverage_stats(split, system_id, k)
        cov_base = cached_coverage_stats(split, baseline_id, k)
        vals = row_sel.iloc[0].to_dict() if not row_sel.empty else {}
        cards_top = st.columns(5)
        cards_top[0].metric(
            "Gold Variant Coverage (%)",
            f"{100.0 * cov['macro_variant_coverage']:.1f}%",
            f"{100.0 * (cov['macro_variant_coverage'] - cov_base['macro_variant_coverage']):+.1f} pts",
        )
        cards_top[1].metric(
            "Gold Frequency Covered (%)",
            f"{100.0 * cov['macro_weighted_recall']:.1f}%",
            f"{100.0 * (cov['macro_weighted_recall'] - cov_base['macro_weighted_recall']):+.1f} pts",
        )
        cards_top[2].metric(
            "Prompt Hit Rate (%)",
            f"{100.0 * cov['prompt_hit_rate']:.1f}%",
            f"{100.0 * (cov['prompt_hit_rate'] - cov_base['prompt_hit_rate']):+.1f} pts",
        )
        cards_top[3].metric(
            "Best Gold Found (%)",
            f"{100.0 * cov['best_gold_hit_ratio']:.1f}%",
            f"{100.0 * (cov['best_gold_hit_ratio'] - cov_base['best_gold_hit_ratio']):+.1f} pts",
        )
        cards_top[4].metric(
            "Micro Variant Coverage (%)",
            f"{100.0 * cov['micro_variant_coverage']:.1f}%",
            f"{100.0 * (cov['micro_variant_coverage'] - cov_base['micro_variant_coverage']):+.1f} pts",
        )

        gold_for_oracle = load_gold(PATHS[f"{split}_gold"].as_posix())
        oracle_preds_cov = {
            pid: [t for t, _w in sorted(zip(g["translations"], g["weights"]), key=lambda x: float(x[1]), reverse=True)[:k]]
            for pid, g in gold_for_oracle.items()
        }
        oracle_cov = coverage_stats(gold_for_oracle, oracle_preds_cov, k)

        compare_cov_df = pd.DataFrame(
            [
                {
                    "System": by_id[baseline_id]["label"] if baseline_id in by_id else baseline_id,
                    "Gold Variant Coverage %": 100.0 * cov_base["macro_variant_coverage"],
                    "Gold Frequency Covered %": 100.0 * cov_base["macro_weighted_recall"],
                    "Prompt Hit Rate %": 100.0 * cov_base["prompt_hit_rate"],
                    "Best Gold Found %": 100.0 * cov_base["best_gold_hit_ratio"],
                },
                {
                    "System": by_id[system_id]["label"] if system_id in by_id else system_id,
                    "Gold Variant Coverage %": 100.0 * cov["macro_variant_coverage"],
                    "Gold Frequency Covered %": 100.0 * cov["macro_weighted_recall"],
                    "Prompt Hit Rate %": 100.0 * cov["prompt_hit_rate"],
                    "Best Gold Found %": 100.0 * cov["best_gold_hit_ratio"],
                },
                {
                    "System": f"Oracle @ {k}",
                    "Gold Variant Coverage %": 100.0 * oracle_cov["macro_variant_coverage"],
                    "Gold Frequency Covered %": 100.0 * oracle_cov["macro_weighted_recall"],
                    "Prompt Hit Rate %": 100.0 * oracle_cov["prompt_hit_rate"],
                    "Best Gold Found %": 100.0 * oracle_cov["best_gold_hit_ratio"],
                },
            ]
        )
        st.markdown("**Coverage comparison (baseline vs selected vs oracle)**")
        st.dataframe(compare_cov_df, use_container_width=True, hide_index=True)
        st.markdown("**Coverage bars**")
        cov_plot_df = compare_cov_df.melt(
            id_vars=["System"],
            value_vars=[
                "Gold Variant Coverage %",
                "Gold Frequency Covered %",
                "Prompt Hit Rate %",
                "Best Gold Found %",
            ],
            var_name="Coverage metric",
            value_name="Percent",
        )
        # Keep Oracle at the far right for easier visual comparison.
        system_order = [s for s in compare_cov_df["System"].tolist() if not str(s).startswith("Oracle @")] + [
            s for s in compare_cov_df["System"].tolist() if str(s).startswith("Oracle @")
        ]
        cov_chart = (
            alt.Chart(cov_plot_df)
            .mark_bar()
            .encode(
                x=alt.X("Coverage metric:N", title=None),
                xOffset=alt.XOffset("System:N", sort=system_order),
                y=alt.Y("Percent:Q", title="Percent"),
                color=alt.Color("System:N", title=None, sort=system_order),
                tooltip=["System", "Coverage metric", alt.Tooltip("Percent:Q", format=".2f")],
            )
            .properties(height=320)
        )
        st.altair_chart(cov_chart, use_container_width=True)
        st.caption(
            "Gold Variant Coverage = percentage of accepted gold variants found (macro over prompts). "
            "Gold Frequency Covered = percentage of gold frequency weight captured by the guesses (weighted recall). "
            "Best Gold Found = percentage of prompts where the highest-weight gold translation is present in the guesses. "
            f"Oracle @ {k} = top-{k} highest-weight gold variants (upper bound for this coverage view)."
        )

    gold = load_gold(PATHS[f"{split}_gold"].as_posix())
    preds = _pred_map_for_examples(split, system_id, k)
    base_preds = _pred_map_for_examples(split, baseline_id, k)
    aws_map = load_aws(PATHS[f"{split}_aws"].as_posix())
    prompt_map = load_prompts(PATHS[f"{split}_gold"].as_posix())

    st.markdown("### Prompt Examples")
    st.caption("Use Best/Worst to inspect typical wins and failures, or Search to inspect a specific prompt.")

    delta_df = pd.DataFrame()
    if mode != "Search prompt":
        run_prompt_analysis = st.button("Load prompt examples", key="run_prompt_level_examples", use_container_width=True)
        if run_prompt_analysis:
            with st.spinner("Computing prompt-level deltas..."):
                delta_df = build_prompt_delta_table(split, system_id, baseline_id, k)
        else:
            st.info("Click 'Load prompt examples' to see best/worst prompt examples.")
    else:
        st.session_state["run_prompt_level_examples"] = False

    def render_prompt_card(pid: str) -> None:
        g = gold[pid]
        sel_preds = list(preds.get(pid, []))
        base_list = list(base_preds.get(pid, []))
        s_sel = score_prompt(g["translations"], g["weights"], sel_preds)
        s_base = score_prompt(g["translations"], g["weights"], base_list)
        delta_f1 = float(s_sel["weighted_f1"] - s_base["weighted_f1"])
        delta_rec = float(s_sel["weighted_recall"] - s_base["weighted_recall"])
        delta_color = "green" if delta_f1 > 1e-9 else ("red" if delta_f1 < -1e-9 else "gray")

        gold_weight_by_norm: Dict[str, float] = {}
        for t, w in zip(g["translations"], g["weights"]):
            nk = normalize_text(t)
            if not nk:
                continue
            gold_weight_by_norm[nk] = gold_weight_by_norm.get(nk, 0.0) + float(w)
        best_gold_found_sel = False
        best_gold_found_base = False
        if gold_weight_by_norm:
            max_w = max(gold_weight_by_norm.values())
            best_keys = {nk for nk, w in gold_weight_by_norm.items() if w == max_w}
            sel_norm = {normalize_text(x) for x in sel_preds if normalize_text(x)}
            base_norm = {normalize_text(x) for x in base_list if normalize_text(x)}
            best_gold_found_sel = bool(sel_norm.intersection(best_keys))
            best_gold_found_base = bool(base_norm.intersection(best_keys))

        st.markdown("---")
        st.markdown(f"**Prompt {pid}**")
        st.write(g["english"])
        st.caption(f"AWS: {aws_map.get(pid, '')}")
        c0, c1, c2 = st.columns(3)
        c0.metric("Selected F1", f"{s_sel['weighted_f1']:.3f}", f"{delta_f1:+.3f}")
        c1.metric("Baseline F1", f"{s_base['weighted_f1']:.3f}")
        c2.metric("Recall change", f"{s_sel['weighted_recall']:.3f}", f"{delta_rec:+.3f}")
        st.markdown(f":{delta_color}[F1 change vs baseline: {delta_f1:+.3f}]")
        st.caption(
            f"Best gold found: selected = {'Yes' if best_gold_found_sel else 'No'} | "
            f"baseline = {'Yes' if best_gold_found_base else 'No'}"
        )

        top_gold = (
            pd.DataFrame({"gold_translation": g["translations"], "weight": g["weights"]})
            .sort_values("weight", ascending=False)
            .head(5)
        )
        st.markdown("**Top weighted gold translations**")
        st.dataframe(top_gold, use_container_width=True, hide_index=True)

        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown(f"**Selected model top-{k}**")
            st.write(sel_preds)
        with c_right:
            st.markdown(f"**Baseline top-{k}**")
            st.write(base_list)

        sel_set = {normalize_text(x) for x in sel_preds if normalize_text(x)}
        missing = []
        for t, w in sorted(zip(g["translations"], g["weights"]), key=lambda x: float(x[1]), reverse=True):
            if normalize_text(t) not in sel_set:
                missing.append({"gold_translation": t, "weight": float(w)})

        hint = []
        if delta_f1 > 0 and delta_rec > 0:
            hint.append("Selected model covers more useful gold frequency mass and improves final quality.")
        if delta_f1 > 0 and delta_rec <= 0:
            hint.append("Selected model improved precision/front-of-list ordering.")
        if delta_f1 < 0 and delta_rec > 0:
            hint.append("Selected model covers more gold frequency mass but likely adds noise (precision drop).")
        if not hint:
            hint.append("Difference is small; ordering changes likely affected precision/recall balance.")
        st.caption(" ".join(hint))

        with st.expander("Details (full gold list, missing variants, exact predictions)"):
            full_gold = pd.DataFrame({"gold_translation": g["translations"], "weight": g["weights"]}).sort_values("weight", ascending=False)
            st.markdown("**Full gold list**")
            st.dataframe(full_gold, use_container_width=True, hide_index=True)
            st.markdown("**Missing gold variants (selected model)**")
            st.dataframe(pd.DataFrame(missing).head(50), use_container_width=True, hide_index=True)
            st.markdown("**Selected predictions**")
            st.write(sel_preds)
            st.markdown("**Baseline predictions**")
            st.write(base_list)

    if mode in {"Best improved", "Worst degraded"} and not delta_df.empty:
        if mode == "Best improved":
            show_df = delta_df.head(n_examples)
        else:
            show_df = delta_df.tail(n_examples).sort_values("delta_f1")
        preview = show_df[["prompt_id", "delta_f1", "delta_rec"]].copy()
        preview = preview.rename(columns={"delta_f1": "F1 change", "delta_rec": "Recall change"})
        st.dataframe(preview, use_container_width=True, hide_index=True)
        for pid in show_df["prompt_id"].tolist():
            render_prompt_card(str(pid))
    elif mode == "Search prompt":
        pid_options = list(prompt_map.keys())
        search_text = st.text_input("Search prompt text or id", key="coverage_search_text").strip().lower()
        filtered = [
            pid for pid in pid_options
            if (search_text in pid.lower()) or (search_text in prompt_map[pid].lower())
        ] if search_text else pid_options
        if not filtered:
            st.warning("No prompts matched the search.")
            return
        pid = st.selectbox("Prompt", options=filtered[:200], format_func=lambda x: f"{x} | {prompt_map[x]}", key="coverage_prompt_pick")
        render_prompt_card(pid)


def render_architecture() -> None:
    st.subheader("Pipeline and Model Structure")
    st.markdown("### Big Picture: Generator → Candidates → Filters")
    st.graphviz_chart(
        """
        digraph BIG {
          rankdir=LR;
          node [shape=box, style=rounded];
          prompts [label="English prompts"];
          gen [label="Generator\\n(base / LoRA-finetuned NLLB)", shape=component];
          cands [label="Candidate pool\\nbeam + sampling + dedupe", shape=folder];
          fA [label="Filter A\\nPairwise Neural Ranker", shape=component];
          fB [label="Filter B\\nFrequency-Aware Regressor", shape=component];
          policy [label="Policy\\nsoft or AWS anchor", shape=diamond];
          out [label="Top-k predictions"];
          prompts -> gen -> cands;
          cands -> fA;
          cands -> fB;
          fA -> policy;
          fB -> policy;
          policy -> out;
        }
        """
    )
    st.caption("Finetuning happens inside the Generator block: LoRA adapters are trained on STAPLE data and attached to the base NLLB model.")

    st.markdown("### Click a Component To Inspect Internal Structure")
    if "best_dash_arch_view" not in st.session_state:
        st.session_state["best_dash_arch_view"] = "Generator"
    cols = st.columns(3)
    if cols[0].button("Generator", use_container_width=True):
        st.session_state["best_dash_arch_view"] = "Generator"
    if cols[1].button("Filter A: Pairwise Neural Ranker", use_container_width=True):
        st.session_state["best_dash_arch_view"] = "Filter A: Pairwise Neural Ranker"
    if cols[2].button("Filter B: Frequency-Aware Regressor", use_container_width=True):
        st.session_state["best_dash_arch_view"] = "Filter B: Frequency-Aware Regressor"

    view = st.session_state["best_dash_arch_view"]
    st.markdown(f"#### {view}")

    if view == "Generator":
        st.graphviz_chart(
            """
            digraph GEN {
              rankdir=TB;
              node [shape=box, style=rounded];
              p [label="English prompt"];
              train [label="STAPLE train pairs + weights\\n(data prep / sampling)", shape=folder];
              tok [label="NLLB tokenizer\\neng_Latn -> hun_Latn"];
              base [label="Base NLLB seq2seq"];
              lora [label="LoRA adapters\\n(trainable parameters)", style="rounded,filled", fillcolor="#fff2cc"];
              ft [label="Finetuned generator\\n(Base NLLB + LoRA)", style="rounded,filled", fillcolor="#e8f5e9"];
              beam [label="Beam decode"];
              samp [label="Sampling decode"];
              merge [label="Merge + preserve order\\n(beam first, sampling after)"];
              dedupe [label="Normalized dedupe"];
              out [label="candidates_*.json"];
              p -> tok -> ft;
              base -> ft;
              lora -> ft;
              train -> lora [label="finetunes"];
              ft -> beam -> merge;
              ft -> samp -> merge;
              merge -> dedupe -> out;
            }
            """
        )
        st.caption(
            "Finetuning occurs by training LoRA adapters on STAPLE data (weighted sampling variants), then using Base NLLB + LoRA for generation. "
            "Output keeps an implicit rank/order (beam-first then sampling), but candidate JSON stores plain text only."
        )

    elif view == "Filter A: Pairwise Neural Ranker":
        st.graphviz_chart(
            """
            digraph V26A {
              rankdir=TB;
              node [shape=box, style=rounded];
              prompt [label="Prompt text"];
              aws [label="AWS translation"];
              cands [label="Candidate list"];
              enc [label="Sentence embedding encoder"];
              poolgeo [label="Candidate-pool geometry\\nunsupervised K-centers"];
              feats [label="Feature builder neural network\\ncombines embedding similarities,\\nraw rank, string overlap, and\\ncandidate-pool geometry features"];
              ranker [label="MLP pairwise ranker\\ntrain: pairwise loss"];
              scores [label="Candidate scores"];
              sort [label="Sort -> top-k"];
              prompt -> enc;
              aws -> enc;
              cands -> enc;
              cands -> poolgeo;
              enc -> feats;
              poolgeo -> feats;
              feats -> ranker -> scores -> sort;
            }
            """
        )
        st.caption(
            "This filter first builds a feature vector with a small neural feature builder from embeddings and ranking/string signals, then a pairwise MLP ranks candidates. "
            "Geometric filtering uses unsupervised centers over the candidate pool and distance/margin-to-center features to capture structure and diversity."
        )

    elif view == "Filter B: Frequency-Aware Regressor":
        st.graphviz_chart(
            """
            digraph V27A {
              rankdir=TB;
              node [shape=box, style=rounded];
              prompt [label="Prompt text"];
              aws [label="AWS translation"];
              cands [label="Candidate list"];
              enc [label="Sentence embedding encoder"];
              poolgeo [label="Candidate-pool geometry\\n(optional feature block)"];
              feats [label="Feature builder neural network\\ncombines embeddings, overlap\\nsignals, raw-rank cues, and\\noptional pool-geometry features"];
              reg [label="MLP regressor\\nrelevance ~ gold frequency"];
              scores [label="Predicted relevance score"];
              sort [label="Sort -> top-k"];
              target [label="Train target:\\nmatched gold weight or 0", shape=note];
              prompt -> enc;
              aws -> enc;
              cands -> enc;
              cands -> poolgeo;
              enc -> feats;
              poolgeo -> feats;
              feats -> reg -> scores -> sort;
              target -> reg;
            }
            """
        )
        st.caption(
            "This filter uses a neural feature builder plus an MLP regressor to predict candidate relevance (proxy for learner frequency weight). "
            "Geometric filtering signals can be added as candidate-pool distance/margin features so the regressor sees structural cues too."
        )
    else:
        st.session_state["best_dash_arch_view"] = "Generator"
        st.info("Select a component to inspect.")

    st.markdown("### Notes")
    st.info(
        "Generator targets coverage/diversity; filters/policies target final weighted F1. "
        "A change can improve coverage but lower F1 (or vice versa), so inspect both."
    )


def render_extra_ideas() -> None:
    st.subheader("What Else To Add")
    rows = [
        ("Official Scorer Panel", "Paper-style weighted macro F1 from Duolingo scorer"),
        ("Oracle Gap", "oracle_f1 - model_f1 by k"),
        ("AWS Help/Hurt Stats", "Prompt counts and delta distributions under anchoring"),
        ("Source Mix", "Beam vs sample contribution in top-k using metadata sidecars"),
        ("Near-Duplicates", "Similarity-based redundancy analysis, not only exact duplicates"),
        ("Runtime/Cost", "Generation + reranking latency and memory"),
        ("Error Tags", "Morphology / word order / formality categories for bad prompts"),
        ("Confidence Calibration", "Score histograms and confidence vs correctness"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Feature", "Why it helps"]), use_container_width=True, hide_index=True)


def render_error_analysis() -> None:
    st.subheader("Error Analysis (Worst Prompt-Level F1)")
    st.caption("Worst prompts for the selected model, with the model's Hungarian guesses and DeepL back-translations (HU → EN).")

    split = st.radio("Split", ["test", "dev"], horizontal=True, key="error_split")
    available = _available_systems(split)
    by_id = {x["id"]: x for x in available}
    model_options = [x["id"] for x in available if x["id"] != "aws_top1"]
    if not model_options:
        st.warning("No model outputs found for this split.")
        return

    default_model = "v27a_anchor" if "v27a_anchor" in model_options else ("v26a_anchor" if "v26a_anchor" in model_options else model_options[0])
    cols = st.columns([2, 1, 1])
    with cols[0]:
        system_id = st.selectbox(
            "Model",
            options=model_options,
            index=model_options.index(default_model),
            format_func=lambda sid: by_id[sid]["label"],
            key="error_model",
        )
    with cols[1]:
        k = st.selectbox("k", KS, index=3, key="error_k")
    with cols[2]:
        n_show = st.slider("Worst prompts to show", min_value=3, max_value=30, value=10, step=1, key="error_n_show")

    gold = load_gold(PATHS[f"{split}_gold"].as_posix())
    aws_map = load_aws(PATHS[f"{split}_aws"].as_posix())
    preds = _pred_map_for_examples(split, system_id, k)
    deepl_bt_index: Dict[str, Any] = {}
    if split == "test" and system_id == "v27a_anchor" and k == 5 and PATHS["deepl_bt_test_v27a_anchor_k5"].exists():
        try:
            bt_rows = load_json(PATHS["deepl_bt_test_v27a_anchor_k5"].as_posix())
            if isinstance(bt_rows, list):
                deepl_bt_index = {str(x.get("prompt_id")): x for x in bt_rows if isinstance(x, dict)}
        except Exception:
            deepl_bt_index = {}
    if split == "test" and k == 5 and PATHS["deepl_bt_test_groups_k5_extended"].exists():
        try:
            groups_obj = load_json(PATHS["deepl_bt_test_groups_k5_extended"].as_posix())
            for b in (groups_obj.get("bundles") or []):
                bname = str(b.get("name", ""))
                for row in (b.get("rows") or []):
                    pid = str(row.get("prompt_id", ""))
                    if not pid:
                        continue
                    if system_id == "v27a_anchor" and bname.startswith("best_model_"):
                        deepl_bt_index.setdefault(pid, row)
        except Exception:
            pass

    @st.cache_data(show_spinner=False)
    def build_worst_prompt_table(split_: str, system_id_: str, k_: int) -> pd.DataFrame:
        gold_ = load_gold(PATHS[f"{split_}_gold"].as_posix())
        preds_ = _pred_map_for_examples(split_, system_id_, k_)
        rows = []
        for pid, g in gold_.items():
            s = score_prompt(g["translations"], g["weights"], preds_.get(pid, []))
            rows.append(
                {
                    "prompt_id": pid,
                    "english": g["english"],
                    "f1": s["weighted_f1"],
                    "weighted_recall": s["weighted_recall"],
                    "precision": s["precision"],
                }
            )
        return pd.DataFrame(rows).sort_values(["f1", "weighted_recall", "precision"], ascending=[True, True, True])

    worst_df = build_worst_prompt_table(split, system_id, k)
    st.markdown("### Worst Prompts (preview)")
    st.dataframe(worst_df.head(n_show), use_container_width=True, hide_index=True)

    st.markdown("### Detailed Prompt Inspection")
    for pid in worst_df.head(n_show)["prompt_id"].tolist():
        g = gold[str(pid)]
        pred_list = list(preds.get(str(pid), []))
        s = score_prompt(g["translations"], g["weights"], pred_list)
        gold_rows = (
            pd.DataFrame({"gold_translation": g["translations"], "weight": g["weights"]})
            .sort_values("weight", ascending=False)
        )
        gold_norms = {normalize_text(t) for t in g["translations"] if normalize_text(t)}
        bt_item = deepl_bt_index.get(str(pid))
        bt_map = {}
        if bt_item:
            for r in bt_item.get("predictions", []) or []:
                hu = str(r.get("hu", ""))
                bt_map[hu] = str(r.get("back_en", ""))

        st.markdown("---")
        st.markdown(f"### {g['english']}")
        m1, m2 = st.columns(2)
        m1.metric("Prompt F1", f"{s['weighted_f1']:.3f}")
        m2.metric("Weighted Recall", f"{s['weighted_recall']:.3f}")

        st.markdown(f"**Model guesses and DeepL back-translations (top-{k})**")
        if pred_list:
            for idx, hu in enumerate(pred_list, start=1):
                left, right = st.columns(2)
                with left:
                    st.markdown(f"**Guess {idx} (HU)**")
                    st.write(hu)
                with right:
                    st.markdown(f"**DeepL back-translation {idx} (EN)**")
                    st.write(bt_map.get(hu, "(not available)"))
        elif split == "test" and system_id == "v27a_anchor" and k == 5:
            st.caption(
                "DeepL back-translation file not found for this view. Save "
                "`deepl_backtranslation_worst5_test_v27a_anchor_k5.json` to display HU→EN checks here."
            )
        else:
            st.write(pred_list if pred_list else ["(none)"])

        # Intentionally kept minimal: no prompt IDs or full-detail expander in the main error-analysis view.


def main() -> None:
    st.set_page_config(page_title="STAPLE Best Model Dashboard", layout="wide")
    st.title("STAPLE EN→HU Best Model Dashboard")
    st.caption("Focused view on generator/reranker comparisons, coverage analysis, and architecture.")

    if not PATHS["dev_gold"].exists():
        st.error(f"Missing required file: {PATHS['dev_gold']}")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Architecture", "Overview", "Coverage & Examples", "Error Analysis", "Additions"])
    with tab1:
        render_architecture()
    with tab2:
        render_overview()
    with tab3:
        render_coverage_and_examples()
    with tab4:
        render_error_analysis()
    with tab5:
        render_extra_ideas()


if __name__ == "__main__":
    main()
