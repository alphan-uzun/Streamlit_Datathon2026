from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data_io import parse_gold
from metrics import evaluate
from train_geo_filter import GeoSphereFilter
from train_geo_pairwise_ranker import EmbeddingCache
from train_pairwise_ranker_noleak import (
    KS_OUT,
    _apply_feature_norm,
    augment_features,
    build_inference_feature_packs,
    rerank_from_packs,
)
from train_relevance_regressor_noleak import PairwiseGeoAwsRanker


def main() -> None:
    ap = argparse.ArgumentParser(description="Standalone non-leaky relevance regressor reranker inference (v27a).")
    ap.add_argument("--encoder-geo-model", required=True)
    ap.add_argument("--ranker-ckpt", required=True)
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--aws-file", required=True)
    ap.add_argument("--candidates-json", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--embedding-cache", default=None)
    ap.add_argument("--eval-gold", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc_ckpt = torch.load(args.encoder_geo_model, map_location=device)
    encoder_model = GeoSphereFilter(enc_ckpt["encoder_name"], proj_dim=enc_ckpt["proj_dim"]).to(device)
    encoder_model.load_state_dict(enc_ckpt["state_dict"])
    encoder_model.eval()
    for p in encoder_model.parameters():
        p.requires_grad = False

    rank_ckpt = torch.load(args.ranker_ckpt, map_location="cpu")
    model = PairwiseGeoAwsRanker(
        input_dim=len(rank_ckpt["feature_names"]),
        hidden_dims=rank_ckpt.get("hidden_dims", [int(rank_ckpt.get("hidden_dim", 64))] * 2),
        dropout=float(rank_ckpt["dropout"]),
    ).to(device)
    model.load_state_dict(rank_ckpt["state_dict"])
    model.eval()

    emb_cache = EmbeddingCache(args.embedding_cache)
    emb_cache.validate_or_reset(encoder_name=enc_ckpt["encoder_name"], proj_dim=int(enc_ckpt["proj_dim"]))

    packs = build_inference_feature_packs(
        model=encoder_model,
        prompts_file=args.prompts_file,
        aws_path=args.aws_file,
        candidates_path=args.candidates_json,
        device=device,
        emb_cache=emb_cache,
        num_pool_centers=int(rank_ckpt.get("num_pool_centers", 3)),
    )
    packs_aug = []
    base_feature_names = rank_ckpt.get("base_feature_names", rank_ckpt["feature_names"])
    feature_crosses = rank_ckpt.get("feature_crosses", "none")
    for p in packs:
        x_aug, _ = augment_features(p.features, base_feature_names, feature_crosses)
        packs_aug.append(type(p)(prompt_id=p.prompt_id, features=x_aug, rel=p.rel, raw_prior=p.raw_prior, candidates=p.candidates))
    packs = _apply_feature_norm(packs_aug, rank_ckpt["feature_mean"].cpu(), rank_ckpt["feature_std"].cpu())

    preds = rerank_from_packs(model, packs, device)
    Path(args.out_json).write_text(json.dumps(preds, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {args.out_json}")
    emb_cache.save()

    if args.eval_gold:
        gold = parse_gold(args.eval_gold)
        for k in KS_OUT:
            m = evaluate(gold, preds[str(k)])
            print(f"k={k} f1={m['macro_weighted_f1']:.6f} rec={m['macro_weighted_recall']:.6f} prec={m['macro_precision']:.6f}")


if __name__ == "__main__":
    main()
