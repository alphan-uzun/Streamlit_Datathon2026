from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(".")
REPO = ROOT / "external" / "duolingo-sharedtask-2020"
if str(REPO.resolve()) not in sys.path:
    sys.path.insert(0, str(REPO.resolve()))

from staple_2020_scorer import score  # type: ignore
from utils import read_transfile  # type: ignore

from data_io import parse_prompts

KS = [1, 2, 3, 5, 10, 20]


def pred_lines_from_map(prompts: Dict[str, str], pred_map: Dict[str, List[str] | str]) -> List[str]:
    lines: List[str] = []
    for pid, english in prompts.items():
        lines.append(f"{pid}|{english}")
        obj = pred_map.get(pid, [])
        vals = [obj] if isinstance(obj, str) else list(obj)
        for v in vals:
            t = str(v).strip()
            if t:
                lines.append(t)
        lines.append("")
    return [x + "\n" for x in lines]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def infer_leakage_status(name: str) -> str:
    lower = name.lower()
    if "pairwise_noleak" in lower or "relevance_v27" in lower or "fusion_meta" in lower:
        return "valid_non_leaky"
    if "_geo_" in lower or "awsproto" in lower or "geo_pairwise_v25" in lower:
        return "diagnostic_leaky"
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build official Duolingo scorer summary for available test systems.")
    ap.add_argument("--gold", default="staple-2020/en_hu/test.en_hu.2020-02-20.gold.txt")
    ap.add_argument("--aws", default="staple-2020/en_hu/test.en_hu.aws_baseline.pred.txt")
    ap.add_argument("--prompts-file", default="staple-2020/en_hu/test.en_hu.2020-02-20.gold.txt")
    ap.add_argument("--raw-pattern", default="candidates_test*.json")
    ap.add_argument("--reranked-pattern", default="reranked_test*.json")
    ap.add_argument("--out-json", default="official_scorer_test_summary_all_systems.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    prompts = parse_prompts(args.prompts_file)
    with Path(args.gold).open(encoding="utf-8") as f:
        gold = read_transfile(f.readlines(), weighted=True)
    with Path(args.aws).open(encoding="utf-8") as f:
        aws_pred = read_transfile(f.readlines(), weighted=False)

    raw_files = sorted(ROOT.glob(args.raw_pattern))
    reranked_files = sorted(ROOT.glob(args.reranked_pattern))

    summary = {
        "gold": str(Path(args.gold)).replace("\\", "/"),
        "metric": "official_duolingo_staple_weighted_macro_f1",
        "by_system": {},
    }

    # AWS baseline
    aws_val = score(gold, aws_pred, verbose=False)
    summary["by_system"]["aws_top1"] = {
        "label": "AWS baseline top-1",
        "system_type": "single",
        "leakage_status": "valid_non_leaky",
        "source_file": str(Path(args.aws)).replace("\\", "/"),
        "by_k": {"1": {"weighted_macro_f1": aws_val}},
    }

    for p in raw_files:
        obj = load_json(p)
        entry = {
            "label": f"{p.stem} (raw)",
            "system_type": "raw_candidates",
            "leakage_status": "valid_non_leaky",
            "source_file": p.as_posix(),
            "by_k": {},
        }
        for k in KS:
            pred_map = {pid: lst[:k] for pid, lst in obj.items()}
            pred = read_transfile(pred_lines_from_map(prompts, pred_map), weighted=False)
            val = score(gold, pred, verbose=False)
            entry["by_k"][str(k)] = {"weighted_macro_f1": val}
        summary["by_system"][p.stem] = entry

    for p in reranked_files:
        obj = load_json(p)
        if not isinstance(obj, dict):
            continue
        # Expect multi-k dict; skip plain maps if any weird file appears.
        if not all(isinstance(v, dict) for v in obj.values()):
            continue
        entry = {
            "label": p.stem,
            "system_type": "reranked_multik",
            "leakage_status": infer_leakage_status(p.stem),
            "source_file": p.as_posix(),
            "by_k": {},
        }
        for k in KS:
            k_str = str(k)
            if k_str not in obj:
                continue
            pred = read_transfile(pred_lines_from_map(prompts, obj[k_str]), weighted=False)
            val = score(gold, pred, verbose=False)
            entry["by_k"][k_str] = {"weighted_macro_f1": val}
        summary["by_system"][p.stem] = entry

    Path(args.out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.quiet:
        print(f"Saved {args.out_json}")
        print(f"systems={len(summary['by_system'])} raw={len(raw_files)} reranked={len(reranked_files)}")


if __name__ == "__main__":
    main()
