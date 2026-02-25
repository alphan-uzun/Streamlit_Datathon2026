"""Prepare EN->HU MT finetuning data with weighted target sampling per epoch."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from data_io import parse_gold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    gold = parse_gold(args.gold)
    prompt_ids = list(gold.keys())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ep in range(args.epochs):
            for pid in prompt_ids:
                g = gold[pid]
                target = rng.choices(g["translations"], weights=g["weights"], k=1)[0]
                row = {
                    "prompt_id": pid,
                    "src_text": g["english"],
                    "tgt_text": target,
                    "epoch_id": ep,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1

    print(f"Saved {n_rows} rows to {out_path}")
    print("Sanity sample:")
    with out_path.open("r", encoding="utf-8") as f:
        for i in range(2):
            line = f.readline().strip()
            if not line:
                break
            print(line)


if __name__ == "__main__":
    main()

