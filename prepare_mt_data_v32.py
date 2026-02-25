from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from data_io import parse_gold


def flatten_weights(weights, mode: str):
    ws = [float(w) for w in weights]
    if mode == "weighted":
        out = ws
    elif mode == "uniform":
        out = [1.0 for _ in ws]
    elif mode == "sqrt":
        # flattens distribution: rare variants get relatively more chance
        out = [math.sqrt(max(w, 1e-12)) for w in ws]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    s = sum(out)
    return [w / s for w in out] if s > 0 else [1.0 / len(out) for _ in out]


def sample_target(rng: random.Random, translations, weights, mode: str):
    probs = flatten_weights(weights, mode)
    return rng.choices(list(translations), weights=probs, k=1)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3, help="synthetic epochs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-weighted", action="store_true", help="include one weighted sample per prompt per epoch")
    ap.add_argument("--include-uniform", action="store_true", help="include one uniform sample per prompt per epoch")
    ap.add_argument("--include-sqrt", action="store_true", help="include one sqrt-flattened sample per prompt per epoch")
    args = ap.parse_args()

    if not (args.include_weighted or args.include_uniform or args.include_sqrt):
        # sensible default: all three
        args.include_weighted = True
        args.include_uniform = True
        args.include_sqrt = True

    rng = random.Random(args.seed)
    gold = parse_gold(args.gold)
    prompt_ids = list(gold.keys())

    modes = []
    if args.include_weighted:
        modes.append("weighted")
    if args.include_uniform:
        modes.append("uniform")
    if args.include_sqrt:
        modes.append("sqrt")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    mode_counts = {m: 0 for m in modes}

    with out_path.open("w", encoding="utf-8") as f:
        for ep in range(args.epochs):
            for pid in prompt_ids:
                g = gold[pid]
                for mode in modes:
                    tgt = sample_target(rng, g["translations"], g["weights"], mode)
                    row = {
                        "prompt_id": pid,
                        "src_text": g["english"],
                        "tgt_text": tgt,
                        "epoch_id": ep,
                        "sampling_mode": mode,
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_rows += 1
                    mode_counts[mode] += 1

    print(f"Saved {n_rows} rows to {out_path}")
    print("mode_counts:", mode_counts)
    print("rows_per_prompt_per_epoch:", len(modes))

    print("Sanity sample:")
    with out_path.open("r", encoding="utf-8") as f:
        for _ in range(6):
            line = f.readline().strip()
            if not line:
                break
            print(line)


if __name__ == "__main__":
    main()
