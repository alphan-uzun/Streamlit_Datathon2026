from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from data_io import parse_aws_baseline, parse_gold
from metrics import evaluate, score_prompt
from textnorm import normalize_text


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str, obj: dict) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def anchor_aws(reranked: dict, aws: Dict[str, str]) -> dict:
    out = {}
    for k_str, pred_map in reranked.items():
        k = int(k_str)
        if not isinstance(pred_map, dict):
            continue
        out[k_str] = {}
        for pid, cand_list in pred_map.items():
            merged: List[str] = []
            seen = set()
            aws_pred = aws.get(pid, "")
            if aws_pred:
                nk = normalize_text(aws_pred)
                if nk:
                    merged.append(aws_pred)
                    seen.add(nk)
            for c in cand_list:
                nk = normalize_text(c)
                if not nk or nk in seen:
                    continue
                merged.append(c)
                seen.add(nk)
                if len(merged) >= k:
                    break
            out[k_str][pid] = merged[:k]
    return out


def aws_is_gold(gold_entry: Dict[str, object], aws_pred: str) -> tuple[bool, float]:
    nk = normalize_text(aws_pred)
    if not nk:
        return False, 0.0
    w = 0.0
    for t, wt in zip(gold_entry["translations"], gold_entry["weights"]):
        if normalize_text(t) == nk:
            w += float(wt)
    return (w > 0.0), w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--aws", required=True)
    ap.add_argument("--reranked-json", required=True, help="Existing reranked predictions with keys '1','2',...")
    ap.add_argument("--raw-candidates-json", required=False, help="Optional raw candidates file for compare_all style context")
    ap.add_argument("--out-json", required=True, help="Anchored reranked output file")
    ap.add_argument("--k", type=int, default=5, help="k for helped/hurt prompt diagnostics")
    ap.add_argument("--topn", type=int, default=20)
    args = ap.parse_args()

    gold = parse_gold(args.gold)
    aws = parse_aws_baseline(args.aws)
    rer = load_json(args.reranked_json)
    anchored = anchor_aws(rer, aws)
    save_json(args.out_json, anchored)
    print(f"Saved {args.out_json}")

    # Summary metrics by k
    print("\nAnchored metrics:")
    for k_str in sorted(anchored.keys(), key=lambda x: int(x)):
        m = evaluate(gold, anchored[k_str])
        print(
            f"k={k_str} f1={m['macro_weighted_f1']:.6f} "
            f"rec={m['macro_weighted_recall']:.6f} prec={m['macro_precision']:.6f}"
        )

    # Compare anchored vs original reranked at selected k
    k_str = str(args.k)
    if k_str not in rer or k_str not in anchored:
        print(f"\nNo k={args.k} found in reranked predictions. Available ks: {sorted(rer.keys())}")
        return

    rows = []
    for pid, g in gold.items():
        orig_preds = rer[k_str].get(pid, [])
        anc_preds = anchored[k_str].get(pid, [])
        s_orig = score_prompt(g["translations"], g["weights"], orig_preds)
        s_anc = score_prompt(g["translations"], g["weights"], anc_preds)
        aws_pred = aws.get(pid, "")
        aws_gold, aws_gold_w = aws_is_gold(g, aws_pred)
        rows.append(
            {
                "prompt_id": pid,
                "english": g["english"],
                "orig_f1": s_orig["weighted_f1"],
                "anchored_f1": s_anc["weighted_f1"],
                "delta_f1": s_anc["weighted_f1"] - s_orig["weighted_f1"],
                "orig_rec": s_orig["weighted_recall"],
                "anchored_rec": s_anc["weighted_recall"],
                "delta_rec": s_anc["weighted_recall"] - s_orig["weighted_recall"],
                "aws_pred": aws_pred,
                "aws_is_gold": aws_gold,
                "aws_gold_weight": aws_gold_w,
                "orig_top1": orig_preds[0] if orig_preds else "",
                "anchored_top1": anc_preds[0] if anc_preds else "",
            }
        )

    rows_sorted_help = sorted(rows, key=lambda r: r["delta_f1"], reverse=True)
    rows_sorted_hurt = sorted(rows, key=lambda r: r["delta_f1"])

    print(f"\nMost helped prompts (AWS anchor vs reranked) at k={args.k}:")
    for r in rows_sorted_help[: args.topn]:
        print(
            f"{r['prompt_id']} | delta_f1={r['delta_f1']:+.4f} | aws_gold={r['aws_is_gold']} "
            f"(w={r['aws_gold_weight']:.4f}) | EN={r['english']}"
        )
        print(f"  AWS: {r['aws_pred']}")
        print(f"  orig_top1: {r['orig_top1']}")
        print(f"  anchored_top1: {r['anchored_top1']}")

    print(f"\nMost hurt prompts (AWS anchor vs reranked) at k={args.k}:")
    for r in rows_sorted_hurt[: args.topn]:
        print(
            f"{r['prompt_id']} | delta_f1={r['delta_f1']:+.4f} | aws_gold={r['aws_is_gold']} "
            f"(w={r['aws_gold_weight']:.4f}) | EN={r['english']}"
        )
        print(f"  AWS: {r['aws_pred']}")
        print(f"  orig_top1: {r['orig_top1']}")
        print(f"  anchored_top1: {r['anchored_top1']}")

    # Optional compact stats
    n = len(rows)
    helped = sum(1 for r in rows if r["delta_f1"] > 0)
    hurt = sum(1 for r in rows if r["delta_f1"] < 0)
    neutral = n - helped - hurt
    aws_gold_rate = sum(1 for r in rows if r["aws_is_gold"]) / max(n, 1)
    print("\nPrompt-level summary:")
    print(
        {
            "num_prompts": n,
            "helped": helped,
            "hurt": hurt,
            "neutral": neutral,
            "aws_is_gold_rate": round(aws_gold_rate, 6),
            "mean_delta_f1": round(sum(r["delta_f1"] for r in rows) / max(n, 1), 6),
        }
    )


if __name__ == "__main__":
    main()

