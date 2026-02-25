from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model


class JsonlMtDataset(Dataset):
    def __init__(self, path: str):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return {"src_text": r["src_text"], "tgt_text": r["tgt_text"]}


def build_collate_fn(tokenizer, max_source_len: int, max_target_len: int):
    def collate(batch: List[dict]):
        src = [x["src_text"] for x in batch]
        tgt = [x["tgt_text"] for x in batch]

        enc = tokenizer(
            src,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_source_len,
        )
        lab = tokenizer(
            text_target=tgt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_target_len,
        )
        labels = lab["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }

    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--output-dir", default="artifacts/nllb_lora_manual")
    ap.add_argument("--max-source-len", type=int, default=96)
    ap.add_argument("--max-target-len", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = device == "cuda" and not use_bf16
    print(f"device={device} bf16={use_bf16} fp16={use_fp16}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, src_lang="eng_Latn", tgt_lang="hun_Latn")
    base = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(base, lora_cfg).to(device)
    model.train()
    model.print_trainable_parameters()

    ds = JsonlMtDataset(args.train_jsonl)
    print(f"train_rows={len(ds)}")
    if len(ds) > 0:
        print("sample:", ds[0])

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=build_collate_fn(tokenizer, args.max_source_len, args.max_target_len),
        pin_memory=(device == "cuda"),
    )

    total_micro_steps = len(loader) * args.epochs
    total_steps = max(1, math.ceil(total_micro_steps / args.grad_accum))
    warmup_steps = int(total_steps * args.warmup_ratio)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        # linear decay
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 1.0 - progress)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    global_step = 0
    micro_step = 0
    running_loss = 0.0

    for epoch in range(args.epochs):
        for batch in loader:
            micro_step += 1
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.autocast(
                device_type=device,
                dtype=torch.bfloat16 if use_bf16 else torch.float16,
                enabled=(device == "cuda"),
            ):
                out = model(**batch)
                loss = out.loss / args.grad_accum

            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item() * args.grad_accum

            if micro_step % args.grad_accum == 0:
                if use_fp16:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                global_step += 1

                if global_step % args.log_every == 0:
                    # compute grad norm over trainable params (after step grads may be cleared; this is mostly for visibility)
                    lr = opt.param_groups[0]["lr"]
                    print(
                        {
                            "step": global_step,
                            "epoch": epoch + 1,
                            "loss": f"{running_loss / max(args.log_every,1):.4f}",
                            "lr": f"{lr:.6g}",
                        }
                    )
                    running_loss = 0.0

        # flush remaining accumulated grads if dataset size not divisible by grad_accum
        if micro_step % args.grad_accum != 0:
            if use_fp16:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            global_step += 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved LoRA adapter + tokenizer to {out_dir}")


if __name__ == "__main__":
    main()



