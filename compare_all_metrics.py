from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from data_io import parse_aws_baseline, parse_gold
from metrics import evaluate
from textnorm import normalize_text


def take_topk(cands: Dict[str, List[str]], k: int) -> Dict[str, List[str]]:
    return {pid: lst[:k] for pid, lst in cands.items()}


def fmt(x: float) -> str:
    return f"{x:.6f}"


def gold_oracle_topk_metrics(gold: Dict[str, Dict[str, object]], k: int) -> Dict[str, float]:
    """Oracle metrics if we could pick the best k items from each gold set itself.

    We dedupe gold translations by normalized form (same as eval matching), sum weights
    for duplicates, sort by weight desc, and take top-k.
    """
    p_vals = []
    r_vals = []
    f_vals = []
    count_cov_vals = []

    for g in gold.values():
        gmap: Dict[str, float] = {}
        for t, w in zip(g["translations"], g["weights"]):
            key = normalize_text(t)
            gmap[key] = gmap.get(key, 0.0) + float(w)

        weights_sorted = sorted(gmap.values(), reverse=True)
        covered = sum(weights_sorted[:k])
        precision = 1.0 if min(k, len(weights_sorted)) > 0 else 0.0
        f1 = (2.0 * precision * covered / (precision + covered)) if (precision + covered) > 0 else 0.0
        count_cov = (min(k, len(weights_sorted)) / len(weights_sorted)) if weights_sorted else 0.0

        p_vals.append(precision)
        r_vals.append(covered)
        f_vals.append(f1)
        count_cov_vals.append(count_cov)

    n = max(len(f_vals), 1)
    return {
        "macro_precision": sum(p_vals) / n,
        "macro_weighted_recall": sum(r_vals) / n,
        "macro_weighted_f1": sum(f_vals) / n,
        "macro_count_coverage": sum(count_cov_vals) / n,
        "num_prompts": float(len(f_vals)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare AWS baseline vs raw generator order vs filter-reranked metrics on dev."
    )
    ap.add_argument("--gold", required=True, help="Path to dev gold file")
    ap.add_argument("--aws", required=False, help="Path to AWS baseline predictions (top-1)")
    ap.add_argument("--candidates-json", required=True, help="Original generator candidates (e.g. candidates_dev.json)")
    ap.add_argument("--reranked-json", required=True, help='Output of rerank_and_predict.py (keys like "1","2",...)')
    ap.add_argument("--k-list", default="1,2,3,5,10,20")
    ap.add_argument("--out-json", default=None, help="Optional JSON report path")
    args = ap.parse_args()

    gold = parse_gold(args.gold)
    raw_cands = json.loads(Path(args.candidates_json).read_text(encoding="utf-8"))
    reranked = json.loads(Path(args.reranked_json).read_text(encoding="utf-8"))
    ks = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]

    aws_metrics = None
    if args.aws:
        aws = parse_aws_baseline(args.aws)
        aws_metrics = evaluate(gold, aws)

    report = {
        "aws_top1": aws_metrics,
        "by_k": {},
    }

    if aws_metrics is not None:
        print("AWS baseline (top-1)")
        print(
            f"  precision={fmt(aws_metrics['macro_precision'])} "
            f"recall={fmt(aws_metrics['macro_weighted_recall'])} "
            f"f1={fmt(aws_metrics['macro_weighted_f1'])}"
        )
        print()

    print("Comparison on same candidate pool (before = generator order, after = filter rerank)")
    if aws_metrics is not None:
        print("k | aws_f1   | raw_f1   | rerank_f1 | delta_f1  | raw_rec  | rerank_rec | delta_rec | oracle_rec | oracle_f1")
        print("-" * 119)
    else:
        print("k | raw_f1   | rerank_f1 | delta_f1  | raw_rec  | rerank_rec | delta_rec | oracle_rec | oracle_f1")
        print("-" * 108)

    for k in ks:
        raw_preds = take_topk(raw_cands, k)
        rerank_preds = reranked[str(k)]

        m_raw = evaluate(gold, raw_preds)
        m_rerank = evaluate(gold, rerank_preds)
        m_oracle = gold_oracle_topk_metrics(gold, k)

        delta = {
            "macro_precision": m_rerank["macro_precision"] - m_raw["macro_precision"],
            "macro_weighted_recall": m_rerank["macro_weighted_recall"] - m_raw["macro_weighted_recall"],
            "macro_weighted_f1": m_rerank["macro_weighted_f1"] - m_raw["macro_weighted_f1"],
        }

        report["by_k"][str(k)] = {
            "raw": m_raw,
            "reranked": m_rerank,
            "gold_topk_oracle": m_oracle,
            "delta": delta,
        }

        if aws_metrics is not None:
            print(
                f"{k:<2} | "
                f"{aws_metrics['macro_weighted_f1']:.6f} | "
                f"{m_raw['macro_weighted_f1']:.6f} | {m_rerank['macro_weighted_f1']:.6f} | {delta['macro_weighted_f1']:+.6f} | "
                f"{m_raw['macro_weighted_recall']:.6f} | {m_rerank['macro_weighted_recall']:.6f} | {delta['macro_weighted_recall']:+.6f} | "
                f"{m_oracle['macro_weighted_recall']:.6f} | {m_oracle['macro_weighted_f1']:.6f}"
            )
        else:
            print(
                f"{k:<2} | "
                f"{m_raw['macro_weighted_f1']:.6f} | {m_rerank['macro_weighted_f1']:.6f} | {delta['macro_weighted_f1']:+.6f} | "
                f"{m_raw['macro_weighted_recall']:.6f} | {m_rerank['macro_weighted_recall']:.6f} | {delta['macro_weighted_recall']:+.6f} | "
                f"{m_oracle['macro_weighted_recall']:.6f} | {m_oracle['macro_weighted_f1']:.6f}"
            )

    print("\nOracle note: `oracle_rec` is weighted recall if we could return the top-k highest-weight gold variants.")

    if aws_metrics is not None and "1" in report["by_k"]:
        d_vs_aws = {
            "raw_top1_minus_aws_f1": report["by_k"]["1"]["raw"]["macro_weighted_f1"] - aws_metrics["macro_weighted_f1"],
            "rerank_top1_minus_aws_f1": report["by_k"]["1"]["reranked"]["macro_weighted_f1"] - aws_metrics["macro_weighted_f1"],
        }
        report["top1_vs_aws"] = d_vs_aws
        print()
        print("Top-1 vs AWS baseline")
        print(f"  raw_top1 - aws_f1    = {d_vs_aws['raw_top1_minus_aws_f1']:+.6f}")
        print(f"  rerank_top1 - aws_f1 = {d_vs_aws['rerank_top1_minus_aws_f1']:+.6f}")

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved report to {args.out_json}")


if __name__ == "__main__":
    main()
