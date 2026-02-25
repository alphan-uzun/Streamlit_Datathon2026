from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from data_io import parse_prompts


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_pred_blocks(
    prompt_file: str | Path,
    pred_map: Dict[str, List[str] | str],
) -> List[str]:
    prompts = parse_prompts(prompt_file)
    lines: List[str] = []
    for pid, english in prompts.items():
        lines.append(f"{pid}|{english}")
        pred_obj = pred_map.get(pid, [])
        preds = [pred_obj] if isinstance(pred_obj, str) else list(pred_obj)
        for p in preds:
            if p is None:
                continue
            txt = str(p).strip()
            if txt:
                lines.append(txt)
        lines.append("")
    return lines


def select_pred_map(reranked_obj: dict, k: int | None) -> Dict[str, List[str] | str]:
    # Supports:
    # 1) shared task style map {prompt_id: [..]}
    # 2) our reranked format {"1": {...}, "5": {...}}
    if not reranked_obj:
        return {}
    if all(isinstance(v, dict) for v in reranked_obj.values()):
        if k is None:
            raise ValueError("Input looks like multi-k reranked JSON. Please provide --k.")
        k_str = str(k)
        if k_str not in reranked_obj:
            raise KeyError(f"k={k} not found. Available keys: {sorted(reranked_obj.keys())}")
        return reranked_obj[k_str]
    return reranked_obj  # already prompt->preds


def run_duolingo_scorer(scorer_path: Path, goldfile: Path, predfile: Path, verbose: bool = False) -> int:
    cmd = [sys.executable, str(scorer_path), "--goldfile", str(goldfile), "--predfile", str(predfile)]
    if verbose:
        cmd.append("--verbose")
    print("Running official scorer:", " ".join(cmd))
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(cmd, check=False, env=env)
    return int(proc.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export reranked JSON to official STAPLE .pred.txt format.")
    ap.add_argument("--input-json", required=True, help="reranked_*.json or prompt->pred map")
    ap.add_argument("--prompt-file", required=True, help="STAPLE gold/prompt file used for header IDs and English prompt text")
    ap.add_argument("--out-pred", required=True, help="Output .pred.txt path (shared-task format)")
    ap.add_argument("--k", type=int, default=None, help="Select k from multi-k reranked JSON (required for our reranked outputs)")
    ap.add_argument("--score-goldfile", default=None, help="Optional gold file to score immediately with official scorer")
    ap.add_argument("--scorer-path", default="external/duolingo-sharedtask-2020/staple_2020_scorer.py")
    ap.add_argument("--score-verbose", action="store_true")
    args = ap.parse_args()

    obj = load_json(args.input_json)
    pred_map = select_pred_map(obj, args.k)
    lines = build_pred_blocks(args.prompt_file, pred_map)
    out_path = Path(args.out_pred)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {args.out_pred}")

    if args.score_goldfile:
        rc = run_duolingo_scorer(
            scorer_path=Path(args.scorer_path),
            goldfile=Path(args.score_goldfile),
            predfile=Path(args.out_pred),
            verbose=args.score_verbose,
        )
        if rc != 0:
            raise SystemExit(rc)


if __name__ == "__main__":
    main()
