"""STAPLE metrics: precision, weighted recall, weighted F1, macro averages."""

from __future__ import annotations

import argparse
from typing import Dict, Iterable, List, Sequence

from data_io import parse_aws_baseline, parse_gold
from textnorm import normalize_text


def _gold_map(translations: Sequence[str], weights: Sequence[float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for t, w in zip(translations, weights):
        key = normalize_text(t)
        out[key] = out.get(key, 0.0) + float(w)
    return out


def _norm_unique(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        k = normalize_text(x)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def score_prompt(gold_translations: Sequence[str], gold_weights: Sequence[float], predictions: Sequence[str]) -> Dict[str, float]:
    gmap = _gold_map(gold_translations, gold_weights)
    pred = _norm_unique(predictions)
    pred_set = set(pred)
    inter = pred_set.intersection(gmap.keys())

    precision = (len(inter) / len(pred_set)) if pred_set else 0.0
    w_recall = sum(gmap[k] for k in inter)
    f1 = (2.0 * precision * w_recall / (precision + w_recall)) if (precision + w_recall) > 0 else 0.0
    return {"precision": precision, "weighted_recall": w_recall, "weighted_f1": f1}


def evaluate(gold: Dict[str, Dict[str, object]], predictions: Dict[str, Sequence[str] | str]) -> Dict[str, float]:
    p_vals = []
    r_vals = []
    f_vals = []
    for pid, g in gold.items():
        pred_obj = predictions.get(pid, [])
        pred_list = [pred_obj] if isinstance(pred_obj, str) else list(pred_obj)
        s = score_prompt(g["translations"], g["weights"], pred_list)
        p_vals.append(s["precision"])
        r_vals.append(s["weighted_recall"])
        f_vals.append(s["weighted_f1"])
    n = max(len(f_vals), 1)
    return {
        "macro_precision": sum(p_vals) / n,
        "macro_weighted_recall": sum(r_vals) / n,
        "macro_weighted_f1": sum(f_vals) / n,
        "num_prompts": float(len(f_vals)),
    }


def _format_metrics(m: Dict[str, float]) -> str:
    return (
        f"prompts={int(m['num_prompts'])} "
        f"macro_precision={m['macro_precision']:.6f} "
        f"macro_weighted_recall={m['macro_weighted_recall']:.6f} "
        f"macro_weighted_f1={m['macro_weighted_f1']:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, help="Path to *.gold.txt")
    parser.add_argument("--aws", required=True, help="Path to *.aws_baseline.pred.txt")
    args = parser.parse_args()

    gold = parse_gold(args.gold)
    aws = parse_aws_baseline(args.aws)
    metrics = evaluate(gold, aws)
    print(_format_metrics(metrics))


if __name__ == "__main__":
    main()

