#!/usr/bin/env python3
"""CodeDeltaTok hard-negative evaluation.

Tests whether delta tokens can discriminate between changes when the
starting code is similar. This is the key reviewer defense: "MRR 0.997
with random negatives is too easy."

Hard negative construction:
  For each query, gallery contains K hard negatives (similar before-state)
  plus the ground-truth match. The task: find the correct after-state
  among changes to similar code.

Three difficulty levels:
  - Easy: random gallery (baseline, already tested)
  - Medium: top-100 most similar before-states as gallery
  - Hard: top-20 most similar before-states as gallery

Usage:
    python eval_hard_negatives.py \
        --features data/commitpackft_unixcoder_features.h5 \
        --checkpoint checkpoints/code_deltatok_final.pt \
        --num-query 500
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def mrr_at_k(sim_row: np.ndarray, gt_idx: int) -> float:
    """MRR for a single query given similarity scores and ground-truth index."""
    rank = int((sim_row > sim_row[gt_idx]).sum()) + 1
    return 1.0 / rank


def eval_hard_negatives(
    delta_reps: np.ndarray,
    after_feat: np.ndarray,
    before_feat: np.ndarray,
    num_query: int = 500,
    gallery_sizes: tuple[int, ...] = (20, 50, 100, 500, 1000),
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Evaluate with hard negatives based on before-state similarity.

    For each query, selects gallery items whose before-state is most
    similar to the query's before-state. The ground-truth match (same
    index) is always included in the gallery.
    """
    np.random.seed(seed)
    N = len(delta_reps)

    # Normalize before features for similarity
    bf_norm = before_feat / (np.linalg.norm(before_feat, axis=1, keepdims=True) + 1e-8)

    # Normalize delta reps and after features for retrieval
    dr_norm = delta_reps / (np.linalg.norm(delta_reps, axis=1, keepdims=True) + 1e-8)
    af_norm = after_feat / (np.linalg.norm(after_feat, axis=1, keepdims=True) + 1e-8)

    # Select query indices
    query_idx = np.random.choice(N, size=min(num_query, N), replace=False)

    results = {}

    for gs in gallery_sizes:
        mrrs = []
        r1s = []
        r5s = []

        for qi in query_idx:
            # Find top-gs most similar before-states (excluding self)
            bf_sim = bf_norm[qi] @ bf_norm.T
            bf_sim[qi] = -1  # exclude self
            hard_neg_idx = np.argsort(bf_sim)[-gs:]  # top-gs most similar

            # Gallery = hard negatives + ground truth (qi itself)
            gallery_idx = np.concatenate([hard_neg_idx, [qi]])
            gt_pos = len(gallery_idx) - 1  # ground truth is last

            # Cross-modal retrieval: delta_rep[qi] vs after_feat[gallery]
            query_vec = dr_norm[qi]
            gallery_vecs = af_norm[gallery_idx]
            sim = query_vec @ gallery_vecs.T

            rank = int((sim > sim[gt_pos]).sum()) + 1
            mrrs.append(1.0 / rank)
            r1s.append(1.0 if rank <= 1 else 0.0)
            r5s.append(1.0 if rank <= 5 else 0.0)

        results[f"hard_{gs}"] = {
            "gallery_size": gs + 1,  # +1 for ground truth
            "MRR": float(np.mean(mrrs)),
            "R@1": float(np.mean(r1s)),
            "R@5": float(np.mean(r5s)),
            "MRR_std": float(np.std(mrrs)),
        }

    # Also run random-gallery baseline at same query count
    rand_mrrs = []
    rand_r1s = []
    for qi in query_idx:
        sim = dr_norm[qi] @ af_norm.T
        rank = int((sim > sim[qi]).sum()) + 1
        rand_mrrs.append(1.0 / rank)
        rand_r1s.append(1.0 if rank <= 1 else 0.0)

    results["random_full"] = {
        "gallery_size": N,
        "MRR": float(np.mean(rand_mrrs)),
        "R@1": float(np.mean(rand_r1s)),
        "R@5": float("nan"),
        "MRR_std": float(np.std(rand_mrrs)),
    }

    return results


def load_deltatok_reps(before_feat, after_feat, checkpoint_path):
    """Load checkpoint and encode all deltas."""
    from architectures.code_deltatok.code_deltatok import CodeDeltaTok

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    num_blocks = sum(1 for k in sd if k.startswith("encoder.") and k.endswith(".norm1.weight"))
    num_tokens = sd["z_embed"].shape[1]
    feature_dim = sd["z_embed"].shape[2]

    model = CodeDeltaTok(feature_dim=feature_dim, num_blocks=num_blocks, num_delta_tokens=num_tokens)
    model.load_state_dict(sd)
    model.requires_grad_(False)

    prev_t = torch.from_numpy(before_feat.astype(np.float32))
    next_t = torch.from_numpy(after_feat.astype(np.float32))

    all_z = []
    bs = 256
    for i in range(0, len(prev_t), bs):
        end = min(i + bs, len(prev_t))
        with torch.no_grad():
            dt = model.encode(prev_t[i:end], next_t[i:end])
        z = dt.mean(dim=1).numpy() if dt.shape[1] > 1 else dt[:, 0].numpy()
        all_z.append(z)

    return np.concatenate(all_z)


def main():
    parser = argparse.ArgumentParser(description="CodeDeltaTok hard-negative eval")
    parser.add_argument("--features", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-query", type=int, default=500)
    parser.add_argument("--num-eval", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    import h5py
    print(f"Loading features from {args.features}")
    f = h5py.File(args.features, "r")
    all_before = f["before_features"][:]
    all_after = f["after_features"][:]
    N = all_before.shape[0]
    print(f"  {N} samples, {all_before.shape[1]}-dim")

    # Subsample for speed
    perm = np.random.permutation(N)[:args.num_eval]
    before_feat = all_before[perm]
    after_feat = all_after[perm]

    # Filter zero-norm
    deltas = after_feat - before_feat
    valid = np.linalg.norm(deltas, axis=1) > 1e-6
    before_feat = before_feat[valid]
    after_feat = after_feat[valid]
    n = len(before_feat)
    print(f"  {n} valid samples after filtering")

    # Encode with DeltaTok
    print(f"\nEncoding with {args.checkpoint}...")
    t0 = time.time()
    delta_reps = load_deltatok_reps(before_feat, after_feat, args.checkpoint)
    print(f"  Encoded in {time.time()-t0:.1f}s")

    # Also compute raw delta baseline
    raw_delta = after_feat - before_feat

    # Run hard-negative eval for both
    print(f"\n=== DeltaTok (contrastive) ===")
    dt_results = eval_hard_negatives(
        delta_reps, after_feat, before_feat,
        num_query=args.num_query, seed=args.seed,
    )
    for name, r in dt_results.items():
        print(f"  {name:<15} gallery={r['gallery_size']:>5}  MRR={r['MRR']:.4f} R@1={r['R@1']:.4f} R@5={r.get('R@5', float('nan')):.4f}")

    print(f"\n=== Raw Delta (baseline) ===")
    rd_results = eval_hard_negatives(
        raw_delta, after_feat, before_feat,
        num_query=args.num_query, seed=args.seed,
    )
    for name, r in rd_results.items():
        print(f"  {name:<15} gallery={r['gallery_size']:>5}  MRR={r['MRR']:.4f} R@1={r['R@1']:.4f} R@5={r.get('R@5', float('nan')):.4f}")

    # Summary comparison
    print(f"\n{'='*80}")
    print(f"{'Difficulty':<15} {'Gallery':>7} {'DeltaTok MRR':>12} {'Raw Delta MRR':>13} {'Delta Win':>10}")
    print(f"{'-'*80}")
    for name in dt_results:
        dt_mrr = dt_results[name]["MRR"]
        rd_mrr = rd_results[name]["MRR"]
        win = dt_mrr - rd_mrr
        gs = dt_results[name]["gallery_size"]
        print(f"{name:<15} {gs:>7} {dt_mrr:>12.4f} {rd_mrr:>13.4f} {win:>+10.4f}")
    print(f"{'='*80}")

    f.close()


if __name__ == "__main__":
    main()
