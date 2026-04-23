#!/usr/bin/env python3
"""CodeDeltaTok benchmark — proper downstream evaluation.

Three evaluation tasks:

Task 1: Cross-modal retrieval
  Query = delta representation (from method under test)
  Gallery = actual after-state features (frozen UniXcoder)
  Ground truth: query[i] should match gallery[i] (same code change)
  Tests: does the delta representation carry enough info to identify
         WHICH next-state it corresponds to?

Task 2: Edit-type classification
  Linear probe on delta representations -> 3-class edit type
  Tests: does the representation preserve edit semantics?

Task 3: Reconstruction retrieval
  Decode delta_token + before_feat -> predicted after_feat
  Find actual after_feat in gallery via cosine
  Tests: is the reconstruction discriminative, not just high-cosine?

Usage:
    python eval_deltatok.py \
        --features data/commitpackft_unixcoder_features.h5 \
        [--checkpoint path/to/code_deltatok_final.pt] \
        --num-eval 5000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def retrieval_metrics(
    query_reps: np.ndarray,
    gallery_reps: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10, 50),
    bootstrap_n: int = 1000,
) -> dict[str, float]:
    """Cosine KNN retrieval with bootstrap CIs. query[i] matches gallery[i]."""
    q_norm = query_reps / (np.linalg.norm(query_reps, axis=1, keepdims=True) + 1e-8)
    g_norm = gallery_reps / (np.linalg.norm(gallery_reps, axis=1, keepdims=True) + 1e-8)

    sim = q_norm @ g_norm.T
    nq = sim.shape[0]

    ranks = []
    for i in range(nq):
        rank = int((sim[i] > sim[i, i]).sum()) + 1
        ranks.append(rank)
    ranks = np.array(ranks, dtype=np.float64)

    results = {}
    for k in ks:
        results[f"R@{k}"] = float((ranks <= k).mean())
    results["MRR"] = float((1.0 / ranks).mean())
    results["med_rank"] = float(np.median(ranks))

    # Paired bootstrap 95% CI on MRR
    if bootstrap_n > 0:
        rr = 1.0 / ranks  # per-query reciprocal rank
        boot_mrrs = []
        for _ in range(bootstrap_n):
            idx = np.random.choice(nq, size=nq, replace=True)
            boot_mrrs.append(rr[idx].mean())
        boot_mrrs = np.array(boot_mrrs)
        results["MRR_ci_lo"] = float(np.percentile(boot_mrrs, 2.5))
        results["MRR_ci_hi"] = float(np.percentile(boot_mrrs, 97.5))

    return results


def effective_rank(X: np.ndarray) -> float:
    X_c = X - X.mean(axis=0)
    s = np.linalg.svd(X_c, compute_uv=False)
    p = s / (s.sum() + 1e-12)
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))


def eval_method(
    name: str,
    delta_reps: np.ndarray,
    after_feat: np.ndarray,
    before_feat: np.ndarray | None = None,
    recon_feat: np.ndarray | None = None,
    dims: int = 0,
    bootstrap_n: int = 1000,
) -> dict[str, object]:
    """Run all eval tasks on a delta representation."""
    N = len(delta_reps)
    result: dict[str, object] = {"name": name, "dims": dims or delta_reps.shape[-1]}

    # Task 1: Cross-modal retrieval (delta -> after-state)
    # Only works when delta_reps and after_feat have same dims
    if delta_reps.shape[1] == after_feat.shape[1]:
        t1 = retrieval_metrics(delta_reps, after_feat, bootstrap_n=bootstrap_n)
        for k, v in t1.items():
            result[f"xmodal_{k}"] = v
    else:
        result["xmodal_MRR"] = float("nan")
        result["xmodal_R@1"] = float("nan")
        result["xmodal_R@10"] = float("nan")

    # Task 1b: Self-retrieval (delta -> delta)
    # Skipped — deterministic methods always get MRR=1.0 on self-retrieval

    # Effective rank
    result["eff_rank"] = effective_rank(delta_reps)

    # Task 3: Reconstruction retrieval (if available)
    if recon_feat is not None:
        t3 = retrieval_metrics(recon_feat, after_feat, bootstrap_n=bootstrap_n)
        for k, v in t3.items():
            result[f"recon_{k}"] = v
        # Recon cosine
        cos = np.sum(recon_feat * after_feat, axis=1) / (
            np.linalg.norm(recon_feat, axis=1) * np.linalg.norm(after_feat, axis=1) + 1e-8
        )
        result["recon_cos"] = float(cos.mean())

    return result


def get_raw_delta(before_feat, after_feat):
    return after_feat - before_feat


def get_pca_delta(before_feat, after_feat, n_components):
    from sklearn.decomposition import PCA
    deltas = after_feat - before_feat
    pca = PCA(n_components=n_components)
    compressed = pca.fit_transform(deltas)
    recon = pca.inverse_transform(compressed)
    return compressed, recon


def get_linear_ae_delta(before_feat, after_feat, bottleneck):
    import torch
    import torch.nn as nn

    deltas = (after_feat - before_feat).astype(np.float32)
    D = deltas.shape[1]
    data = torch.from_numpy(deltas)

    encoder = nn.Linear(D, bottleneck)
    decoder = nn.Linear(bottleneck, D)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

    for _ in range(500):
        z = encoder(data)
        recon = decoder(z)
        loss = ((recon - data) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        z = encoder(data).numpy()
        recon = decoder(encoder(data)).numpy()
    return z, recon


def get_deltatok_reps(before_feat, after_feat, checkpoint_path):
    import torch
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

    all_z, all_recon = [], []
    bs = 256
    for i in range(0, len(prev_t), bs):
        end = min(i + bs, len(prev_t))
        with torch.no_grad():
            dt = model.encode(prev_t[i:end], next_t[i:end])
            recon = model.decode(dt, prev_t[i:end])
        z = dt.mean(dim=1).numpy() if dt.shape[1] > 1 else dt[:, 0].numpy()
        all_z.append(z)
        all_recon.append(recon.numpy())

    return np.concatenate(all_z), np.concatenate(all_recon), num_tokens, num_blocks


def main():
    parser = argparse.ArgumentParser(description="CodeDeltaTok benchmark (proper eval)")
    parser.add_argument("--features", required=True)
    parser.add_argument("--checkpoints", nargs="*", default=[])
    parser.add_argument("--num-eval", type=int, default=5000,
                        help="Total samples (split: first 20%% query, rest gallery)")
    parser.add_argument("--num-query", type=int, default=None,
                        help="Override query count (default: 20%% of num-eval)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="Bootstrap iterations for CIs (0=skip)")
    args = parser.parse_args()

    np.random.seed(args.seed)

    import h5py
    print(f"Loading features from {args.features}")
    f = h5py.File(args.features, "r")
    all_before = f["before_features"][:]
    all_after = f["after_features"][:]
    N = all_before.shape[0]
    D = all_before.shape[1]
    print(f"  {N} samples, {D}-dim")

    # Use first num_eval samples (deterministic after seed shuffle)
    perm = np.random.permutation(N)[:args.num_eval]
    before_feat = all_before[perm]
    after_feat = all_after[perm]
    n = len(perm)

    # Split into query + gallery (query[i] matches gallery[i])
    nq = args.num_query or max(int(n * 0.2), 100)
    ng = n - nq
    print(f"  Eval: {nq} queries, {ng} gallery (from {n} samples)")

    # Filter zero-norm deltas
    deltas = after_feat - before_feat
    norms = np.linalg.norm(deltas, axis=1)
    valid = norms > 1e-6
    if valid.sum() < n:
        print(f"  Filtered {n - valid.sum()} zero-norm deltas")
        before_feat = before_feat[valid]
        after_feat = after_feat[valid]
        n = len(before_feat)
        print(f"  {n} valid samples")

    all_results = []

    # ---- Baselines ----
    print("\n--- Raw Delta (768-dim) ---")
    t0 = time.time()
    raw = get_raw_delta(before_feat, after_feat)
    bn = args.bootstrap
    r = eval_method("raw_delta_768", raw, after_feat, before_feat, dims=768, bootstrap_n=bn)
    print(f"  xmodal MRR={r['xmodal_MRR']:.4f} R@1={r['xmodal_R@1']:.4f} R@10={r['xmodal_R@10']:.4f} rank={r['eff_rank']:.0f} ({time.time()-t0:.1f}s)")
    all_results.append(r)

    for nd in [1, 10, 64, 128, 384]:
        print(f"\n--- PCA-{nd} ---")
        t0 = time.time()
        z, recon = get_pca_delta(before_feat, after_feat, nd)
        r = eval_method(f"pca_{nd}", z, after_feat, before_feat, recon, dims=nd, bootstrap_n=bn)
        print(f"  xmodal MRR={r['xmodal_MRR']:.4f} R@1={r['xmodal_R@1']:.4f} R@10={r['xmodal_R@10']:.4f} recon_cos={r.get('recon_cos',0):.3f} rank={r['eff_rank']:.0f} ({time.time()-t0:.1f}s)")
        all_results.append(r)

    for bd in [64, 128]:
        print(f"\n--- Linear AE-{bd} ---")
        t0 = time.time()
        z, recon = get_linear_ae_delta(before_feat, after_feat, bd)
        r = eval_method(f"linear_ae_{bd}", z, after_feat, before_feat, recon, dims=bd, bootstrap_n=bn)
        print(f"  xmodal MRR={r['xmodal_MRR']:.4f} R@1={r['xmodal_R@1']:.4f} R@10={r['xmodal_R@10']:.4f} recon_cos={r.get('recon_cos',0):.3f} rank={r['eff_rank']:.0f} ({time.time()-t0:.1f}s)")
        all_results.append(r)

    # ---- DeltaTok ----
    for ckpt in args.checkpoints:
        print(f"\n--- DeltaTok: {Path(ckpt).name} ---")
        t0 = time.time()
        z, recon, K, nb = get_deltatok_reps(before_feat, after_feat, ckpt)
        r = eval_method(f"deltatok_K{K}_{nb}blk", z, after_feat, before_feat, recon, dims=z.shape[1], bootstrap_n=bn)
        print(f"  xmodal MRR={r['xmodal_MRR']:.4f} R@1={r['xmodal_R@1']:.4f} R@10={r['xmodal_R@10']:.4f} recon_cos={r.get('recon_cos',0):.3f} rank={r['eff_rank']:.0f} K={K} ({time.time()-t0:.1f}s)")
        all_results.append(r)

    # ---- Summary ----
    print("\n" + "=" * 115)
    print(f"{'Method':<25} {'xmodal_MRR':>10} {'95% CI':>16} {'R@1':>8} {'R@10':>8} {'recon_cos':>10} {'rank':>6} {'dims':>5}")
    print("-" * 115)
    for r in all_results:
        mrr = r.get('xmodal_MRR', float('nan'))
        ci_lo = r.get('xmodal_MRR_ci_lo', float('nan'))
        ci_hi = r.get('xmodal_MRR_ci_hi', float('nan'))
        r1 = r.get('xmodal_R@1', float('nan'))
        r10 = r.get('xmodal_R@10', float('nan'))
        rc = r.get('recon_cos', '-')
        rc_s = f"{rc:.3f}" if isinstance(rc, float) else "-"
        ci_s = f"[{ci_lo:.4f},{ci_hi:.4f}]" if not np.isnan(ci_lo) else "-"
        mrr_s = f"{mrr:.4f}" if not np.isnan(mrr) else "-"
        r1_s = f"{r1:.4f}" if not np.isnan(r1) else "-"
        r10_s = f"{r10:.4f}" if not np.isnan(r10) else "-"
        print(f"{r['name']:<25} {mrr_s:>10} {ci_s:>16} {r1_s:>8} {r10_s:>8} {rc_s:>10} {r['eff_rank']:>6.0f} {r['dims']:>5}")
    print("=" * 115)

    f.close()


if __name__ == "__main__":
    main()
