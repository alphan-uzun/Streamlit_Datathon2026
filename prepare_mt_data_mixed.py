from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from data_io import parse_gold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weighted-ratio", type=float, default=0.5, help="probability of weighted sampling; remainder uses uniform-over-variants")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    gold = parse_gold(args.gold)
    prompt_ids = list(gold.keys())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    n_weighted = 0
    n_uniform = 0

    with out_path.open("w", encoding="utf-8") as f:
        for ep in range(args.epochs):
            for pid in prompt_ids:
                g = gold[pid]
                use_weighted = rng.random() < args.weighted_ratio
                if use_weighted:
                    tgt = rng.choices(g["translations"], weights=g["weights"], k=1)[0]
                    n_weighted += 1
                else:
                    tgt = rng.choice(g["translations"])
                    n_uniform += 1

                row = {
                    "prompt_id": pid,
                    "src_text": g["english"],
                    "tgt_text": tgt,
                    "epoch_id": ep,
                    "sampling_mode": "weighted" if use_weighted else "uniform",
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1

    print(f"Saved {n_rows} rows to {out_path}")
    print(f"weighted_rows={n_weighted} uniform_rows={n_uniform} weighted_ratio_actual={n_weighted/max(n_rows,1):.4f}")

    print("Sanity sample:")
    with out_path.open("r", encoding="utf-8") as f:
        for _ in range(3):
            line = f.readline().strip()
            if not line:
                break
            print(line)


if __name__ == "__main__":
    main()
