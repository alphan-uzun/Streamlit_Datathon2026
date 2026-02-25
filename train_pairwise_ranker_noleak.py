from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from data_io import parse_aws_baseline, parse_gold
from metrics import evaluate
from textnorm import normalize_text
from train_geo_filter import GeoSphereFilter
from train_geo_pairwise_ranker import (
    EmbeddingCache,
    PairwiseGeoAwsRanker,
    PromptFeaturePack,
    _pairwise_rank_loss,
    _soft_f1_loss,
    _soft_weighted_recall_loss,
    weighted_k_centers,
)

KS_OUT = [1, 2, 3, 5, 10, 20]
BASE_FEATURE_NAMES = [
    "cand_pool_margin_best",
    "cand_pool_margin_mean_top2",
    "cand_pool_margin_gap12",
    "cand_aws_cos",
    "cand_prompt_cos",
    "aws_prompt_cos",
    "raw_prior",
    "rank_recip",
    "cand_norm_len",
    "aws_exact_norm_match",
    "aws_token_jaccard",
    "aws_char_jaccard",
    "cand_top1_cos",
]


def load_candidates(path: str) -> Dict[str, List[str]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _set_jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def _char_set(s: str) -> set[str]:
    return set(ch for ch in s if not ch.isspace())


def _aggregate_gold_relevance(gold_translations: Sequence[str], gold_weights: Sequence[float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for t, w in zip(gold_translations, gold_weights):
        nk = normalize_text(t)
        out[nk] = out.get(nk, 0.0) + float(w)
    return out


def _parse_hidden_dims(hidden_dims_arg: str | None, fallback_hidden_dim: int) -> List[int]:
    if hidden_dims_arg:
        dims = [int(x.strip()) for x in hidden_dims_arg.split(",") if x.strip()]
        if not dims:
            raise ValueError("Invalid --hidden-dims")
        return dims
    return [int(fallback_hidden_dim), int(fallback_hidden_dim)]


def augment_features(features: torch.Tensor, feature_names: Sequence[str], mode: str) -> tuple[torch.Tensor, List[str]]:
    if mode == "none":
        return features, list(feature_names)
    if mode != "basic":
        raise ValueError(f"Unsupported feature-crosses mode: {mode}")
    idx = {n: i for i, n in enumerate(feature_names)}
    pairs = [
        ("cand_pool_margin_best", "raw_prior"),
        ("cand_pool_margin_best", "cand_aws_cos"),
        ("cand_pool_margin_best", "cand_prompt_cos"),
        ("cand_aws_cos", "raw_prior"),
        ("cand_prompt_cos", "raw_prior"),
        ("cand_aws_cos", "cand_prompt_cos"),
        ("aws_exact_norm_match", "cand_aws_cos"),
        ("aws_token_jaccard", "cand_aws_cos"),
        ("aws_char_jaccard", "cand_aws_cos"),
    ]
    cols = [features]
    names = list(feature_names)
    for a, b in pairs:
        cols.append((features[:, idx[a]] * features[:, idx[b]]).unsqueeze(1))
        names.append(f"{a}__x__{b}")
    for a in ["cand_pool_margin_best", "cand_aws_cos", "cand_prompt_cos"]:
        v = features[:, idx[a]]
        cols.append((v * v).unsqueeze(1))
        names.append(f"{a}__sq")
    return torch.cat(cols, dim=1), names


def _fit_feature_norm(packs: List[PromptFeaturePack]) -> tuple[torch.Tensor, torch.Tensor]:
    xs = torch.cat([p.features for p in packs], dim=0)
    return xs.mean(dim=0), xs.std(dim=0, unbiased=False).clamp_min(1e-4)


def _apply_feature_norm(packs: List[PromptFeaturePack], mean: torch.Tensor, std: torch.Tensor) -> List[PromptFeaturePack]:
    out = []
    for p in packs:
        out.append(
            PromptFeaturePack(
                prompt_id=p.prompt_id,
                features=(p.features - mean) / std,
                rel=p.rel,
                raw_prior=p.raw_prior,
                candidates=p.candidates,
            )
        )
    return out


def _build_prompt_pack(
    model: GeoSphereFilter,
    emb_cache: EmbeddingCache | None,
    prompt_id: str,
    prompt_text: str,
    gold_entry: Dict[str, object] | None,
    aws_text: str,
    cand_list: List[str],
    device: str,
    num_pool_centers: int = 3,
) -> PromptFeaturePack | None:
    if not cand_list:
        return None

    enc = emb_cache.encode if emb_cache else None
    with torch.no_grad():
        prompt_z = (enc(model, [prompt_text], device) if enc else model.encode_texts([prompt_text], device))[0]
        zc = (enc(model, cand_list, device) if enc else model.encode_texts(cand_list, device))
        aws_text = aws_text or ""
        if aws_text.strip():
            aws_z = (enc(model, [aws_text], device) if enc else model.encode_texts([aws_text], device))[0]
        else:
            aws_z = torch.zeros_like(prompt_z)

        cand_aws_cos = (zc * aws_z.unsqueeze(0)).sum(dim=-1) if aws_text.strip() else torch.zeros(zc.size(0), device=device)
        cand_prompt_cos = (zc * prompt_z.unsqueeze(0)).sum(dim=-1)
        aws_prompt_cos = float((aws_z * prompt_z).sum().item()) if aws_text.strip() else 0.0
        cand_top1_cos = (zc * zc[0].unsqueeze(0)).sum(dim=-1) if zc.size(0) > 0 else torch.zeros(0, device=device)

        n = len(cand_list)
        raw_prior = torch.tensor([1.0 - (i / max(n - 1, 1)) for i in range(n)], dtype=torch.float32)
        rank_recip = torch.tensor([1.0 / float(i + 1) for i in range(n)], dtype=torch.float32)
        # Unsupervised candidate-pool geometry: derive centers/radii from candidate embeddings only.
        cw = (0.5 + raw_prior.to(device))
        cw = cw / cw.sum().clamp(min=1e-6)
        pool_centers = weighted_k_centers(zc, cw, k=max(1, num_pool_centers))
        d_pool = 1.0 - torch.einsum("nd,kd->nk", zc, pool_centers)
        assign = torch.argmin(d_pool, dim=1)
        pool_radii = []
        for kk in range(pool_centers.size(0)):
            mask = (assign == kk).float()
            wk = cw * mask
            if wk.sum() < 1e-8:
                pool_radii.append(0.12)
                continue
            wk = wk / wk.sum()
            dk = d_pool[:, kk]
            mu = (wk * dk).sum()
            var = (wk * (dk - mu) ** 2).sum()
            rk = float((mu + torch.sqrt(var + 1e-8)).item())
            pool_radii.append(max(1e-4, min(rk, 0.35)))
        pool_radii_t = torch.tensor(pool_radii, dtype=torch.float32, device=device)
        pool_margins = pool_radii_t.unsqueeze(0) - d_pool
        pool_best = pool_margins.max(dim=1).values
        if pool_margins.size(1) >= 2:
            pool_top2 = torch.topk(pool_margins, k=2, dim=1).values
            pool_mean_top2 = pool_top2.mean(dim=1)
            pool_gap12 = pool_top2[:, 0] - pool_top2[:, 1]
        else:
            pool_mean_top2 = pool_best.clone()
            pool_gap12 = torch.zeros_like(pool_best)

        gmap = _aggregate_gold_relevance(gold_entry["translations"], gold_entry["weights"]) if gold_entry else {}
        aws_norm = normalize_text(aws_text) if aws_text else ""
        aws_tokens = set(aws_norm.split()) if aws_norm else set()
        aws_chars = _char_set(aws_norm) if aws_norm else set()

        feat_rows = []
        rel = []
        for i, cand in enumerate(cand_list):
            cand_norm = normalize_text(cand)
            cand_tokens = set(cand_norm.split()) if cand_norm else set()
            cand_chars = _char_set(cand_norm) if cand_norm else set()
            exact = 1.0 if (aws_norm and cand_norm == aws_norm) else 0.0
            feat_rows.append(
                [
                    float(pool_best[i].item()),
                    float(pool_mean_top2[i].item()),
                    float(pool_gap12[i].item()),
                    float(cand_aws_cos[i].item()),
                    float(cand_prompt_cos[i].item()),
                    float(aws_prompt_cos),
                    float(raw_prior[i].item()),
                    float(rank_recip[i].item()),
                    float(min(len(cand_norm), 200) / 200.0),
                    exact,
                    float(_set_jaccard(cand_tokens, aws_tokens)),
                    float(_set_jaccard(cand_chars, aws_chars)),
                    float(cand_top1_cos[i].item()),
                ]
            )
            rel.append(float(gmap.get(cand_norm, 0.0)))

    feats = torch.tensor(feat_rows, dtype=torch.float32)
    rel_t = torch.tensor(rel, dtype=torch.float32)
    raw_prior_t = raw_prior.clone().float()
    return PromptFeaturePack(prompt_id=prompt_id, features=feats, rel=rel_t, raw_prior=raw_prior_t, candidates=list(cand_list))


def build_feature_packs(
    model: GeoSphereFilter,
    gold_path: str,
    aws_path: str,
    candidates_path: str,
    device: str,
    emb_cache: EmbeddingCache | None = None,
    num_pool_centers: int = 3,
) -> List[PromptFeaturePack]:
    gold = parse_gold(gold_path)
    aws = parse_aws_baseline(aws_path)
    cands = load_candidates(candidates_path)
    packs: List[PromptFeaturePack] = []
    pids = list(gold.keys())
    for idx, pid in enumerate(pids):
        p = _build_prompt_pack(
            model=model,
            emb_cache=emb_cache,
            prompt_id=pid,
            prompt_text=str(gold[pid]["english"]),
            gold_entry=gold[pid],
            aws_text=aws.get(pid, ""),
            cand_list=cands.get(pid, []),
            device=device,
            num_pool_centers=num_pool_centers,
        )
        if p is not None:
            packs.append(p)
        if (idx + 1) % 200 == 0:
            print(f"feature_build progress {idx+1}/{len(pids)}")
    return packs


def build_inference_feature_packs(
    model: GeoSphereFilter,
    prompts_file: str,
    aws_path: str,
    candidates_path: str,
    device: str,
    emb_cache: EmbeddingCache | None = None,
    num_pool_centers: int = 3,
) -> List[PromptFeaturePack]:
    from data_io import parse_prompts

    prompts = parse_prompts(prompts_file)
    aws = parse_aws_baseline(aws_path)
    cands = load_candidates(candidates_path)
    packs: List[PromptFeaturePack] = []
    pids = list(prompts.keys())
    for idx, pid in enumerate(pids):
        p = _build_prompt_pack(
            model=model,
            emb_cache=emb_cache,
            prompt_id=pid,
            prompt_text=str(prompts[pid]),
            gold_entry=None,
            aws_text=aws.get(pid, ""),
            cand_list=cands.get(pid, []),
            device=device,
            num_pool_centers=num_pool_centers,
        )
        if p is not None:
            packs.append(p)
        if (idx + 1) % 200 == 0:
            print(f"infer_feature_build progress {idx+1}/{len(pids)}")
    return packs


def rerank_from_packs(ranker: PairwiseGeoAwsRanker, packs: List[PromptFeaturePack], device: str) -> Dict[str, Dict[str, List[str]]]:
    out = {str(k): {} for k in KS_OUT}
    ranker.eval()
    with torch.no_grad():
        for p in packs:
            scores = ranker(p.features.to(device)).cpu()
            order = torch.argsort(scores, descending=True).tolist()
            ranked = [p.candidates[i] for i in order]
            for k in KS_OUT:
                out[str(k)][p.prompt_id] = ranked[:k]
    return out


def evaluate_packs(ranker: PairwiseGeoAwsRanker, packs: List[PromptFeaturePack], gold_path: str, device: str) -> Dict[str, Dict[str, float]]:
    preds = rerank_from_packs(ranker, packs, device)
    gold = parse_gold(gold_path)
    return {str(k): evaluate(gold, preds[str(k)]) for k in KS_OUT}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train non-leaky pairwise reranker (prompt/candidate/AWS features only).")
    ap.add_argument("--encoder-geo-model", required=True, help="Geo model checkpoint used only as text embedding encoder")
    ap.add_argument("--train-gold", required=True)
    ap.add_argument("--train-aws", required=True)
    ap.add_argument("--train-candidates", required=True)
    ap.add_argument("--dev-gold", required=True)
    ap.add_argument("--dev-aws", required=True)
    ap.add_argument("--dev-candidates", required=True)
    ap.add_argument("--output-dir", default="artifacts/pairwise_ranker_v26a")
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--hidden-dims", default=None)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--feature-crosses", choices=["none", "basic"], default="basic")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--pair-hardneg-boost", type=float, default=2.0)
    ap.add_argument("--pair-mode", choices=["all", "hard"], default="hard")
    ap.add_argument("--hard-pair-topn", type=int, default=10)
    ap.add_argument("--hard-pair-margin", type=float, default=0.2)
    ap.add_argument("--hard-pair-max-pairs", type=int, default=128)
    ap.add_argument("--topk-weight-scheme", choices=["none", "staple", "topheavy"], default="staple")
    ap.add_argument("--lambda-f1", type=float, default=0.0)
    ap.add_argument("--lambda-f1-after-epoch", type=int, default=0)
    ap.add_argument("--f1-ks", default="1,3,5,10")
    ap.add_argument("--f1-tau", type=float, default=0.25)
    ap.add_argument("--lambda-rec", type=float, default=0.0)
    ap.add_argument("--lambda-rec-after-epoch", type=int, default=0)
    ap.add_argument("--rec-ks", default="5,10,20")
    ap.add_argument("--rec-tau", type=float, default=0.25)
    ap.add_argument("--select-k", type=int, default=5)
    ap.add_argument("--embedding-cache", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-dev-reranked-json", default=None)
    ap.add_argument("--num-pool-centers", type=int, default=3, help="Unsupervised candidate-pool geometry centers (non-leaky)")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    enc_ckpt = torch.load(args.encoder_geo_model, map_location=device)
    encoder_model = GeoSphereFilter(enc_ckpt["encoder_name"], proj_dim=enc_ckpt["proj_dim"]).to(device)
    encoder_model.load_state_dict(enc_ckpt["state_dict"])
    encoder_model.eval()
    for p in encoder_model.parameters():
        p.requires_grad = False

    emb_cache = EmbeddingCache(args.embedding_cache)
    emb_cache.validate_or_reset(encoder_name=enc_ckpt["encoder_name"], proj_dim=int(enc_ckpt["proj_dim"]))

    print("Building train feature packs...")
    train_packs = build_feature_packs(
        encoder_model, args.train_gold, args.train_aws, args.train_candidates, device, emb_cache, num_pool_centers=args.num_pool_centers
    )
    print("Building dev feature packs...")
    dev_packs = build_feature_packs(
        encoder_model, args.dev_gold, args.dev_aws, args.dev_candidates, device, emb_cache, num_pool_centers=args.num_pool_centers
    )
    emb_cache.save()
    if not train_packs or not dev_packs:
        raise RuntimeError("No feature packs built")

    model_feature_names = list(BASE_FEATURE_NAMES)
    train_aug = []
    for p in train_packs:
        xa, model_feature_names = augment_features(p.features, BASE_FEATURE_NAMES, args.feature_crosses)
        train_aug.append(PromptFeaturePack(p.prompt_id, xa, p.rel, p.raw_prior, p.candidates))
    dev_aug = []
    for p in dev_packs:
        xa, _ = augment_features(p.features, BASE_FEATURE_NAMES, args.feature_crosses)
        dev_aug.append(PromptFeaturePack(p.prompt_id, xa, p.rel, p.raw_prior, p.candidates))
    train_packs = train_aug
    dev_packs = dev_aug

    feat_mean, feat_std = _fit_feature_norm(train_packs)
    train_packs = _apply_feature_norm(train_packs, feat_mean, feat_std)
    dev_packs = _apply_feature_norm(dev_packs, feat_mean, feat_std)

    ranker = PairwiseGeoAwsRanker(
        input_dim=train_packs[0].features.size(1),
        hidden_dims=_parse_hidden_dims(args.hidden_dims, args.hidden_dim),
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(ranker.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    f1_ks = [int(x.strip()) for x in args.f1_ks.split(",") if x.strip()]
    rec_ks = [int(x.strip()) for x in args.rec_ks.split(",") if x.strip()]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "base_feature_names": BASE_FEATURE_NAMES,
                "feature_crosses": args.feature_crosses,
                "args": vars(args),
                "encoder_name": enc_ckpt["encoder_name"],
                "encoder_proj_dim": int(enc_ckpt["proj_dim"]),
                "num_pool_centers": args.num_pool_centers,
                "non_leaky": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    best_score = -float("inf")
    interrupted = False
    try:
        for ep in range(1, args.epochs + 1):
            ranker.train()
            random.shuffle(train_packs)
            total_loss = total_rank = total_f1_aux = total_rec_aux = 0.0
            used = 0
            use_lambda_f1 = args.lambda_f1 if (args.lambda_f1_after_epoch <= 0 or ep >= args.lambda_f1_after_epoch) else 0.0
            use_lambda_rec = args.lambda_rec if (args.lambda_rec_after_epoch <= 0 or ep >= args.lambda_rec_after_epoch) else 0.0
            for p in train_packs:
                x = p.features.to(device)
                rel = p.rel.to(device)
                raw_prior = p.raw_prior.to(device)
                opt.zero_grad(set_to_none=True)
                scores = ranker(x)
                rank_loss = _pairwise_rank_loss(
                    scores,
                    rel,
                    raw_prior,
                    topk_hardneg_boost=args.pair_hardneg_boost,
                    pair_mode=args.pair_mode,
                    hard_pair_topn=args.hard_pair_topn,
                    hard_pair_margin=args.hard_pair_margin,
                    hard_pair_max_pairs=args.hard_pair_max_pairs,
                    topk_weight_scheme=args.topk_weight_scheme,
                )
                if rank_loss.detach().abs().item() == 0.0 and rel.max().item() <= 0:
                    continue
                f1_aux = _soft_f1_loss(scores, rel, f1_ks, args.f1_tau) if use_lambda_f1 > 0 else scores.new_zeros(())
                rec_aux = _soft_weighted_recall_loss(scores, rel, rec_ks, args.rec_tau) if use_lambda_rec > 0 else scores.new_zeros(())
                loss = rank_loss + use_lambda_f1 * f1_aux + use_lambda_rec * rec_aux
                loss.backward()
                opt.step()
                total_loss += float(loss.item())
                total_rank += float(rank_loss.item())
                total_f1_aux += float(f1_aux.item()) if use_lambda_f1 > 0 else 0.0
                total_rec_aux += float(rec_aux.item()) if use_lambda_rec > 0 else 0.0
                used += 1

            dev_metrics = evaluate_packs(ranker, dev_packs, args.dev_gold, device)
            print(
                f"epoch={ep} used_prompts={used} train_loss={total_loss/max(used,1):.6f} "
                f"rank={total_rank/max(used,1):.6f} f1_aux={total_f1_aux/max(used,1):.6f} "
                f"rec_aux={total_rec_aux/max(used,1):.6f} lambda_f1={use_lambda_f1:.4f} lambda_rec={use_lambda_rec:.4f} "
                f"dev_f1@1={dev_metrics['1']['macro_weighted_f1']:.6f} "
                f"dev_f1@{args.select_k}={dev_metrics[str(args.select_k)]['macro_weighted_f1']:.6f}"
            )

            ckpt = {
                "state_dict": ranker.state_dict(),
                "feature_mean": feat_mean,
                "feature_std": feat_std,
                "base_feature_names": BASE_FEATURE_NAMES,
                "feature_names": model_feature_names,
                "feature_crosses": args.feature_crosses,
                "hidden_dim": args.hidden_dim,
                "hidden_dims": _parse_hidden_dims(args.hidden_dims, args.hidden_dim),
                "dropout": args.dropout,
                "encoder_name": enc_ckpt["encoder_name"],
                "encoder_proj_dim": enc_ckpt["proj_dim"],
                "num_pool_centers": args.num_pool_centers,
                "non_leaky": True,
            }
            torch.save(ckpt, out_dir / "ranker_last.pt")
            score = float(dev_metrics[str(args.select_k)]["macro_weighted_f1"])
            if score > best_score:
                best_score = score
                torch.save(ckpt, out_dir / "ranker_best.pt")
                (out_dir / "best_dev_metrics.json").write_text(json.dumps(dev_metrics, indent=2), encoding="utf-8")
                print(f"  new_best dev_f1@{args.select_k}={score:.6f}")
    except KeyboardInterrupt:
        interrupted = True
        print("\nKeyboardInterrupt received. Keeping saved best checkpoint (if any).")
    finally:
        if (out_dir / "ranker_best.pt").exists():
            print(f"Best checkpoint available: {out_dir / 'ranker_best.pt'}")
            if args.save_dev_reranked_json:
                best = torch.load(out_dir / "ranker_best.pt", map_location=device)
                best_ranker = PairwiseGeoAwsRanker(
                    input_dim=len(best["feature_names"]),
                    hidden_dims=best.get("hidden_dims", [int(best.get("hidden_dim", 64))] * 2),
                    dropout=float(best["dropout"]),
                ).to(device)
                best_ranker.load_state_dict(best["state_dict"])
                preds = rerank_from_packs(best_ranker, dev_packs, device)
                Path(args.save_dev_reranked_json).write_text(json.dumps(preds, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"Saved dev reranked predictions to {args.save_dev_reranked_json}")
        else:
            print("No best checkpoint saved.")
        emb_cache.save()
    print(f"Saved artifact to {out_dir}")
    if interrupted:
        return


if __name__ == "__main__":
    main()
