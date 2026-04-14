#!/usr/bin/env python3
"""Phase 0.3 — Isotropy analysis of encoder latent space.

Measures how close the frozen encoder's representation space is to an
isotropic Gaussian (the optimal geometry for delta prediction per
LeJEPA/SIGReg theory). An anisotropic space means some delta directions
are privileged over others, making delta prediction inconsistent.

Metrics:
  1. Singular value spectrum — how uniform?
  2. Anisotropy ratio — top-k SVs vs rest
  3. Per-dimension variance — should be uniform for isotropy
  4. Intrinsic dimensionality estimate
  5. Partition function (Arora et al.) — isotropy score

Usage:
    python isotropy_analysis.py \
        --checkpoint ~/.crucible-hub/taps/crucible-community-tap/checkpoints/phase5/contrast_15k_seed42/code_wm_best.pt \
        --data ~/.crucible-hub/taps/crucible-community-tap/data/commitpackft_with_diffs.h5 \
        --device cpu
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _shared import load_codewm


def isotropy_score(embeddings: np.ndarray) -> float:
    """Compute isotropy score (Mu et al. 2018 / Arora et al. 2016).

    I(Z) = min_c exp(E[c^T z]) / max_c exp(E[c^T z])

    In practice, computed via eigenvalues of the covariance matrix.
    Score of 1.0 = perfectly isotropic, close to 0 = highly anisotropic.
    """
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / len(centered)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 1e-10)
    return float(eigvals.min() / eigvals.max())


def intrinsic_dimensionality(singular_values: np.ndarray, threshold: float = 0.95) -> int:
    """Number of dimensions needed to capture threshold fraction of variance."""
    total = (singular_values ** 2).sum()
    cumulative = np.cumsum(singular_values ** 2)
    return int(np.searchsorted(cumulative, threshold * total)) + 1


def main():
    parser = argparse.ArgumentParser(description="Phase 0.3: Isotropy analysis")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device

    # ── Load ──
    print(f"Loading checkpoint: {args.checkpoint}")
    model, cfg = load_codewm(args.checkpoint, device=device)
    model_dim = cfg.get("model_dim", 128)
    print(f"  model_dim={model_dim}")

    print(f"Loading data: {args.data}")
    f = h5py.File(args.data, "r")
    N = f["before_tokens"].shape[0]
    n = min(args.n_samples, N)
    idx = np.sort(np.random.permutation(N)[:n])

    before = torch.from_numpy(f["before_tokens"][idx.tolist()].astype(np.int64)).to(device)
    after = torch.from_numpy(f["after_tokens"][idx.tolist()].astype(np.int64)).to(device)
    f.close()
    print(f"  samples: {n}")

    # ── Encode ──
    print("Encoding states...")
    z_before_list = []
    z_after_list = []
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            zb = model.state_encoder(before[start:end]).cpu().numpy()
            za = model.state_encoder(after[start:end]).cpu().numpy()
            z_before_list.append(zb)
            z_after_list.append(za)

    z_before = np.concatenate(z_before_list, axis=0)  # [N, D]
    z_after = np.concatenate(z_after_list, axis=0)
    z_delta = z_after - z_before

    # Also encode with target encoder for comparison
    print("Encoding with target encoder...")
    z_target_list = []
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            zt = model.target_encoder(before[start:end]).cpu().numpy()
            z_target_list.append(zt)
    z_target = np.concatenate(z_target_list, axis=0)

    # ── Analysis ──
    for name, Z in [("State encoder (before)", z_before),
                     ("Target encoder (before)", z_target),
                     ("Deltas (z_after - z_before)", z_delta)]:
        print(f"\n{'=' * 60}")
        print(f"ISOTROPY ANALYSIS: {name}")
        print(f"{'=' * 60}")
        print(f"  Shape: {Z.shape}")

        # Center
        Z_centered = Z - Z.mean(axis=0, keepdims=True)

        # SVD
        U, S, Vt = np.linalg.svd(Z_centered, full_matrices=False)
        total_var = (S ** 2).sum()

        # Singular value spectrum
        print(f"\n  --- Singular value spectrum ---")
        print(f"  Top 10 singular values:")
        for i in range(min(10, len(S))):
            pct = 100 * S[i] ** 2 / total_var
            bar = "#" * int(pct)
            print(f"    SV[{i:2d}] = {S[i]:8.4f}  ({pct:5.1f}%) {bar}")

        # Variance ratios
        top1_ratio = S[0] ** 2 / total_var
        top5_ratio = (S[:5] ** 2).sum() / total_var
        top10_ratio = (S[:10] ** 2).sum() / total_var
        top50_ratio = (S[:min(50, len(S))] ** 2).sum() / total_var

        print(f"\n  --- Variance concentration ---")
        print(f"  Top-1  captures: {top1_ratio:.3f} ({100*top1_ratio:.1f}%)")
        print(f"  Top-5  captures: {top5_ratio:.3f} ({100*top5_ratio:.1f}%)")
        print(f"  Top-10 captures: {top10_ratio:.3f} ({100*top10_ratio:.1f}%)")
        if len(S) >= 50:
            print(f"  Top-50 captures: {top50_ratio:.3f} ({100*top50_ratio:.1f}%)")

        # Intrinsic dimensionality
        id_95 = intrinsic_dimensionality(S, 0.95)
        id_99 = intrinsic_dimensionality(S, 0.99)
        print(f"\n  --- Intrinsic dimensionality ---")
        print(f"  95% variance: {id_95}/{model_dim} dims")
        print(f"  99% variance: {id_99}/{model_dim} dims")

        # Isotropy score
        iso = isotropy_score(Z)
        print(f"\n  --- Isotropy score ---")
        print(f"  Score: {iso:.6f}")
        print(f"  (1.0 = perfect isotropy, ~0 = highly anisotropic)")
        if iso > 0.1:
            print(f"  Interpretation: GOOD — reasonably isotropic")
        elif iso > 0.01:
            print(f"  Interpretation: MODERATE — some anisotropy")
        else:
            print(f"  Interpretation: POOR — highly anisotropic, deltas will be direction-biased")

        # Per-dimension variance
        dim_var = Z_centered.var(axis=0)
        print(f"\n  --- Per-dimension variance ---")
        print(f"  mean: {dim_var.mean():.6f}")
        print(f"  std:  {dim_var.std():.6f}")
        print(f"  min:  {dim_var.min():.6f}")
        print(f"  max:  {dim_var.max():.6f}")
        print(f"  coefficient of variation: {dim_var.std() / dim_var.mean():.3f}")
        print(f"  (CV=0 is perfectly uniform, high CV = some dims dominate)")

        # Effective rank (exponential of entropy of normalized SVs squared)
        s2_norm = (S ** 2) / total_var
        entropy = -np.sum(s2_norm * np.log(s2_norm + 1e-10))
        effective_rank = np.exp(entropy)
        print(f"\n  --- Effective rank ---")
        print(f"  Effective rank: {effective_rank:.1f}/{model_dim}")
        print(f"  (closer to model_dim = more isotropic)")

    # ── Delta-specific analysis ──
    print(f"\n{'=' * 60}")
    print(f"DELTA-SPECIFIC GEOMETRY")
    print(f"{'=' * 60}")

    # Delta norms
    delta_norms = np.linalg.norm(z_delta, axis=1)
    state_norms = np.linalg.norm(z_before, axis=1)
    ratios = delta_norms / (state_norms + 1e-8)

    print(f"\n  --- Delta / State norm ratio ---")
    print(f"  ||delta||:  mean={delta_norms.mean():.4f}  std={delta_norms.std():.4f}")
    print(f"  ||state||:  mean={state_norms.mean():.4f}  std={state_norms.std():.4f}")
    print(f"  ratio:      mean={ratios.mean():.4f}  std={ratios.std():.4f}")
    print(f"  median:     {np.median(ratios):.4f}")
    print(f"  p10:        {np.percentile(ratios, 10):.4f}")
    print(f"  p90:        {np.percentile(ratios, 90):.4f}")

    if ratios.mean() < 0.05:
        print(f"  WARNING: Deltas are very small relative to states.")
        print(f"  This confirms the signal-to-noise problem: changes are tiny.")

    # Mean delta direction (is there a dominant direction?)
    mean_delta = z_delta.mean(axis=0)
    mean_delta_norm = np.linalg.norm(mean_delta)
    per_sample_alignment = z_delta @ mean_delta / (delta_norms * mean_delta_norm + 1e-8)
    print(f"\n  --- Mean delta direction ---")
    print(f"  ||mean(delta)||: {mean_delta_norm:.4f}")
    print(f"  Alignment of individual deltas with mean direction:")
    print(f"    mean cos: {per_sample_alignment.mean():.4f}")
    print(f"    std:      {per_sample_alignment.std():.4f}")
    if per_sample_alignment.mean() > 0.5:
        print(f"  WARNING: Deltas cluster around a single direction.")
        print(f"  This suggests a dominant 'drift' rather than diverse transitions.")

    print("\nPhase 0.3 complete.")


if __name__ == "__main__":
    main()
