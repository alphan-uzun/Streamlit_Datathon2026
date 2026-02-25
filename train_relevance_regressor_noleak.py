from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

from train_geo_filter import GeoSphereFilter
from train_geo_pairwise_ranker import EmbeddingCache, PairwiseGeoAwsRanker, PromptFeaturePack
from train_pairwise_ranker_noleak import (
    KS_OUT,
    _apply_feature_norm,
    _fit_feature_norm,
    _parse_hidden_dims,
    augment_features,
    build_feature_packs,
    evaluate_packs,
    rerank_from_packs,
)


def _transform_rel(rel: torch.Tensor, mode: str, alpha: float) -> torch.Tensor:
    if mode == "raw":
        return rel
    if mode == "sqrt":
        return torch.sqrt(rel.clamp_min(0.0))
    if mode == "log":
        return torch.log1p(alpha * rel.clamp_min(0.0)) / math.log1p(alpha)
    raise ValueError(f"Unsupported reg-target: {mode}")


def _regression_loss(pred: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "smoothl1":
        per = F.smooth_l1_loss(pred, target, reduction="none")
    elif loss_type == "mse":
        per = F.mse_loss(pred, target, reduction="none")
    else:
        raise ValueError(f"Unsupported loss-type: {loss_type}")
    w = sample_weight / sample_weight.mean().clamp_min(1e-6)
    return (per * w).mean()


def _score_pack_metrics(ranker: PairwiseGeoAwsRanker, packs: List[PromptFeaturePack], device: str, reg_target: str, reg_alpha: float) -> dict:
    ranker.eval()
    abs_err = []
    pos_abs_err = []
    with torch.no_grad():
        for p in packs:
            pred = ranker(p.features.to(device)).cpu()
            tgt = _transform_rel(p.rel, reg_target, reg_alpha)
            ae = (pred - tgt).abs()
            abs_err.append(ae.mean().item())
            pos_mask = p.rel > 0
            if pos_mask.any():
                pos_abs_err.append(ae[pos_mask].mean().item())
    return {
        "mae_all": sum(abs_err) / max(len(abs_err), 1),
        "mae_pos": sum(pos_abs_err) / max(len(pos_abs_err), 1) if pos_abs_err else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train standalone non-leaky relevance regressor reranker (v27a).")
    ap.add_argument("--encoder-geo-model", required=True, help="Geo checkpoint used only as text embedding encoder")
    ap.add_argument("--train-gold", required=True)
    ap.add_argument("--train-aws", required=True)
    ap.add_argument("--train-candidates", required=True)
    ap.add_argument("--dev-gold", required=True)
    ap.add_argument("--dev-aws", required=True)
    ap.add_argument("--dev-candidates", required=True)
    ap.add_argument("--output-dir", default="artifacts/relevance_regressor_v27a")
    ap.add_argument("--num-pool-centers", type=int, default=3)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--hidden-dims", default=None)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--feature-crosses", choices=["none", "basic"], default="basic")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--loss-type", choices=["smoothl1", "mse"], default="smoothl1")
    ap.add_argument("--reg-target", choices=["raw", "sqrt", "log"], default="sqrt")
    ap.add_argument("--reg-alpha", type=float, default=20.0, help="Only used for reg-target=log")
    ap.add_argument("--weight-positives", type=float, default=2.0, help="Extra sample weight multiplier for gold-matched candidates")
    ap.add_argument("--weight-by-target", action="store_true", help="Further weight examples by transformed target value")
    ap.add_argument("--select-k", type=int, default=5, help="Select best checkpoint by reranked dev F1@k (using regressor score)")
    ap.add_argument("--embedding-cache", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-dev-reranked-json", default=None)
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
        raise RuntimeError("No feature packs built.")

    # Feature augmentation then normalization (same as v26a)
    from train_pairwise_ranker_noleak import BASE_FEATURE_NAMES

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

    model = PairwiseGeoAwsRanker(
        input_dim=train_packs[0].features.size(1),
        hidden_dims=_parse_hidden_dims(args.hidden_dims, args.hidden_dim),
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "task": "relevance_regression_reranker_noleak",
                "args": vars(args),
                "encoder_name": enc_ckpt["encoder_name"],
                "encoder_proj_dim": int(enc_ckpt["proj_dim"]),
                "feature_crosses": args.feature_crosses,
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
            model.train()
            random.shuffle(train_packs)
            total = 0.0
            used = 0
            for p in train_packs:
                x = p.features.to(device)
                rel = p.rel.to(device)
                target = _transform_rel(rel, args.reg_target, args.reg_alpha)
                sample_w = torch.ones_like(target)
                sample_w = sample_w + (rel > 0).float() * max(args.weight_positives - 1.0, 0.0)
                if args.weight_by_target:
                    sample_w = sample_w * (1.0 + target)
                opt.zero_grad(set_to_none=True)
                pred = model(x)
                loss = _regression_loss(pred, target, sample_w, args.loss_type)
                loss.backward()
                opt.step()
                total += float(loss.item())
                used += 1

            dev_rank_metrics = evaluate_packs(model, dev_packs, args.dev_gold, device)
            dev_reg_metrics = _score_pack_metrics(model, dev_packs, device, args.reg_target, args.reg_alpha)
            train_loss = total / max(used, 1)
            sel = float(dev_rank_metrics[str(args.select_k)]["macro_weighted_f1"])
            print(
                f"epoch={ep} train_reg_loss={train_loss:.6f} "
                f"dev_mae_all={dev_reg_metrics['mae_all']:.6f} dev_mae_pos={dev_reg_metrics['mae_pos']:.6f} "
                f"dev_f1@1={dev_rank_metrics['1']['macro_weighted_f1']:.6f} "
                f"dev_f1@{args.select_k}={sel:.6f}"
            )

            ckpt = {
                "state_dict": model.state_dict(),
                "feature_mean": feat_mean,
                "feature_std": feat_std,
                "feature_names": model_feature_names,
                "base_feature_names": BASE_FEATURE_NAMES,
                "feature_crosses": args.feature_crosses,
                "hidden_dim": args.hidden_dim,
                "hidden_dims": _parse_hidden_dims(args.hidden_dims, args.hidden_dim),
                "dropout": args.dropout,
                "encoder_name": enc_ckpt["encoder_name"],
                "encoder_proj_dim": enc_ckpt["proj_dim"],
                "num_pool_centers": args.num_pool_centers,
                "non_leaky": True,
                "model_type": "relevance_regressor",
                "reg_target": args.reg_target,
                "reg_alpha": args.reg_alpha,
            }
            torch.save(ckpt, out_dir / "ranker_last.pt")
            if sel > best_score:
                best_score = sel
                torch.save(ckpt, out_dir / "ranker_best.pt")
                (out_dir / "best_dev_metrics.json").write_text(json.dumps(dev_rank_metrics, indent=2), encoding="utf-8")
                print(f"  new_best dev_f1@{args.select_k}={sel:.6f}")
    except KeyboardInterrupt:
        interrupted = True
        print("\nKeyboardInterrupt received. Keeping saved best checkpoint (if any).")
    finally:
        if (out_dir / "ranker_best.pt").exists() and args.save_dev_reranked_json:
            best = torch.load(out_dir / "ranker_best.pt", map_location=device)
            best_model = PairwiseGeoAwsRanker(
                input_dim=len(best["feature_names"]),
                hidden_dims=best.get("hidden_dims", [int(best.get("hidden_dim", 64))] * 2),
                dropout=float(best["dropout"]),
            ).to(device)
            best_model.load_state_dict(best["state_dict"])
            preds = rerank_from_packs(best_model, dev_packs, device)
            Path(args.save_dev_reranked_json).write_text(json.dumps(preds, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved dev reranked predictions to {args.save_dev_reranked_json}")
        emb_cache.save()
        print(f"Saved artifact to {out_dir}")
    if interrupted:
        return


if __name__ == "__main__":
    main()
