"""Generate candidate translations via beam + sampling and dedupe by normalized text.

Default output format remains unchanged:
    {prompt_id: [candidate_text, ...]}

Optional sidecar metadata can be written without changing the candidate JSON schema.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from data_io import parse_prompts
from textnorm import normalize_text


def batched(items: List[str], batch_size: int) -> List[List[str]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        k = normalize_text(x)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def dedupe_keep_order_with_meta(items: List[Tuple[str, dict]]) -> Tuple[List[str], List[dict]]:
    seen = set()
    out_txt: List[str] = []
    out_meta: List[dict] = []
    for x, meta in items:
        k = normalize_text(x)
        if not k or k in seen:
            continue
        seen.add(k)
        out_txt.append(x)
        out_meta.append(meta)
    return out_txt, out_meta


def load_model_tokenizer(base_model: str, adapter_dir: str | None, device: str):
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir or base_model, src_lang="eng_Latn")
    base = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    if adapter_dir:
        model = PeftModel.from_pretrained(base, adapter_dir)
    else:
        model = base
    model = model.to(device).eval()
    return model, tokenizer


def run_split(
    split_path: str,
    out_path: str,
    model,
    tokenizer,
    batch_size: int,
    beam_size: int,
    sample_n: int,
    max_new_tokens: int,
    out_meta_path: str | None = None,
    split_name: str | None = None,
    progress_every_batches: int = 1,
) -> None:
    prompts = parse_prompts(split_path)
    prompt_ids = list(prompts.keys())
    src_texts = [prompts[pid] for pid in prompt_ids]
    batches_src = batched(src_texts, batch_size)
    total_batches = len(batches_src)
    total_prompts = len(prompt_ids)
    tgt_id = tokenizer.convert_tokens_to_ids("hun_Latn")
    device = next(model.parameters()).device

    out: Dict[str, List[str]] = {}
    out_meta: Dict[str, List[dict]] = {}
    split_label = split_name or Path(out_path).stem
    print(f"[{split_label}] start: prompts={total_prompts} batch_size={batch_size} batches={total_batches} beam={beam_size} sample_n={sample_n}")
    with torch.inference_mode():
        for bi, batch in enumerate(batches_src):
            if bi == 0 or ((bi + 1) % max(progress_every_batches, 1) == 0) or (bi + 1 == total_batches):
                done_prompts = min(bi * batch_size, total_prompts)
                print(f"[{split_label}] progress batch {bi+1}/{total_batches} | prompts {done_prompts}/{total_prompts}")
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            beam_out = model.generate(
                **enc,
                forced_bos_token_id=tgt_id,
                num_beams=beam_size,
                num_return_sequences=beam_size,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
            samp_out = model.generate(
                **enc,
                forced_bos_token_id=tgt_id,
                do_sample=True,
                top_p=0.9,
                temperature=0.9,
                num_return_sequences=sample_n,
                max_new_tokens=max_new_tokens,
            )
            beam_txt = tokenizer.batch_decode(beam_out, skip_special_tokens=True)
            samp_txt = tokenizer.batch_decode(samp_out, skip_special_tokens=True)
            for i in range(len(batch)):
                pid = prompt_ids[bi * batch_size + i]
                b = beam_txt[i * beam_size : (i + 1) * beam_size]
                s = samp_txt[i * sample_n : (i + 1) * sample_n]
                if out_meta_path:
                    combined: List[Tuple[str, dict]] = []
                    for j, txt in enumerate(b):
                        combined.append(
                            (
                                txt,
                                {
                                    "source": "beam",
                                    "source_rank": j,
                                    "pre_dedupe_index": len(combined),
                                },
                            )
                        )
                    for j, txt in enumerate(s):
                        combined.append(
                            (
                                txt,
                                {
                                    "source": "sample",
                                    "source_rank": j,
                                    "pre_dedupe_index": len(combined),
                                },
                            )
                        )
                    deduped_txt, deduped_meta = dedupe_keep_order_with_meta(combined)
                    # Add final rank after dedupe and normalized form for debugging.
                    for r, m in enumerate(deduped_meta):
                        m["final_rank"] = r
                        m["norm"] = normalize_text(deduped_txt[r])
                    out[pid] = deduped_txt
                    out_meta[pid] = deduped_meta
                else:
                    out[pid] = dedupe_keep_order(b + s)
        print(f"[{split_label}] progress batch {total_batches}/{total_batches} | prompts {total_prompts}/{total_prompts}")

    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{split_label}] saved {len(out)} prompts to {out_path}")
    if out_meta_path:
        meta_payload = {
            "schema": "candidate_metadata_v1",
            "notes": "Aligned by index with candidates JSON list for each prompt_id",
            "by_prompt": out_meta,
        }
        Path(out_meta_path).write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{split_label}] saved metadata sidecar to {out_meta_path}")
    for pid in list(out.keys())[:2]:
        print(f"[{split_label}] sample {pid}", out[pid][:3])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--sample-n", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dev-prompts", default="staple-2020/en_hu/dev.en_hu.2020-02-20.gold.txt")
    parser.add_argument("--test-prompts", default="staple-2020/en_hu/test.en_hu.2020-02-20.gold.txt")
    parser.add_argument("--train-prompts", default=None)
    parser.add_argument("--out-dev", default="candidates_dev.json")
    parser.add_argument("--out-test", default="candidates_test.json")
    parser.add_argument("--out-train", default=None)
    parser.add_argument("--out-dev-meta", default=None, help="Optional sidecar metadata JSON for dev candidates")
    parser.add_argument("--out-test-meta", default=None, help="Optional sidecar metadata JSON for test candidates")
    parser.add_argument("--out-train-meta", default=None, help="Optional sidecar metadata JSON for train candidates")
    parser.add_argument("--progress-every-batches", type=int, default=5, help="Print progress every N batches per split")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_tokenizer(args.base_model, args.adapter_dir, device)
    print(f"device={device}")

    run_split(
        args.dev_prompts,
        args.out_dev,
        model,
        tokenizer,
        args.batch_size,
        args.beam_size,
        args.sample_n,
        args.max_new_tokens,
        out_meta_path=args.out_dev_meta,
        split_name="dev",
        progress_every_batches=args.progress_every_batches,
    )
    run_split(
        args.test_prompts,
        args.out_test,
        model,
        tokenizer,
        args.batch_size,
        args.beam_size,
        args.sample_n,
        args.max_new_tokens,
        out_meta_path=args.out_test_meta,
        split_name="test",
        progress_every_batches=args.progress_every_batches,
    )

    if args.train_prompts and args.out_train:
        run_split(
            args.train_prompts,
            args.out_train,
            model,
            tokenizer,
            args.batch_size,
            args.beam_size,
            args.sample_n,
            args.max_new_tokens,
            out_meta_path=args.out_train_meta,
            split_name="train",
            progress_every_batches=args.progress_every_batches,
        )


if __name__ == "__main__":
    main()
