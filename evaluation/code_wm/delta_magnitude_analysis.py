#!/usr/bin/env python3
"""Phase 0.4 — Delta magnitude distribution analysis.

Computes ||z_{t+1} - z_t|| / ||z_t|| across datasets to quantify the
signal-to-noise ratio for delta prediction. Designed to compare CodeWM
(Python, expected low signal) vs KernelWM (CUDA kernels, expected high
signal) to confirm the domain hypothesis.

Also computes copy-cosine (how close is z_{t+1} to z_t?) as a direct
measure of how hard delta prediction needs to work.

Usage:
    # CodeWM only
    python delta_magnitude_analysis.py \
        --checkpoint ~/.crucible-hub/taps/crucible-community-tap/checkpoints/phase5/contrast_15k_seed42/code_wm_best.pt \
        --data ~/.crucible-hub/taps/crucible-community-tap/data/commitpackft_with_diffs.h5 \
        --device cpu

    # Cross-domain comparison (if kernel checkpoint exists)
    python delta_magnitude_analysis.py \
        --checkpoint <code_wm_ckpt> --data <code_data> \
        --kernel-checkpoint <kernel_wm_ckpt> --kernel-data <kernel_data> \
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from _shared import load_codewm


def analyze_deltas(
    model,
    before_tokens: torch.Tensor,
    after_tokens: torch.Tensor,
    batch_size: int = 256,
    device: str = "cpu",
    label: str = "dataset",
) -> dict:
    """Encode before/after states and compute delta statistics."""
    n = before_tokens.shape[0]

    z_before_list = []
    z_after_list = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            zb = model.state_encoder(before_tokens[start:end]).cpu()
            za = model.state_encoder(after_tokens[start:end]).cpu()
            z_before_list.append(zb)
            z_after_list.append(za)

    z_before = torch.cat(z_before_list, dim=0)
    z_after = torch.cat(z_after_list, dim=0)
    z_delta = z_after - z_before

    # Norms
    delta_norms = z_delta.norm(dim=1).numpy()
    state_norms = z_before.norm(dim=1).numpy()
    ratios = delta_norms / (state_norms + 1e-8)

    # Copy cosine: cos(z_t, z_{t+1})
    copy_cos = F.cosine_similarity(z_before, z_after, dim=1).numpy()

    # Delta direction cosine: cos(delta, mean_delta)
    mean_delta = z_delta.mean(dim=0)
    delta_cos_mean = F.cosine_similarity(z_delta, mean_delta.unsqueeze(0), dim=1).numpy()

    stats = {
        "label": label,
        "n": n,
        "delta_norm_mean": float(delta_norms.mean()),
        "delta_norm_std": float(delta_norms.std()),
        "delta_norm_median": float(np.median(delta_norms)),
        "state_norm_mean": float(state_norms.mean()),
        "ratio_mean": float(ratios.mean()),
        "ratio_std": float(ratios.std()),
        "ratio_median": float(np.median(ratios)),
        "ratio_p10": float(np.percentile(ratios, 10)),
        "ratio_p90": float(np.percentile(ratios, 90)),
        "copy_cos_mean": float(copy_cos.mean()),
        "copy_cos_std": float(copy_cos.std()),
        "copy_cos_median": float(np.median(copy_cos)),
        "copy_cos_p10": float(np.percentile(copy_cos, 10)),
        "delta_cos_mean_alignment": float(delta_cos_mean.mean()),
    }

    # Distribution buckets for ratio
    bins = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, float("inf")]
    bin_labels = ["<0.01 (near-zero)", "0.01-0.05 (tiny)", "0.05-0.1 (small)",
                  "0.1-0.2 (moderate)", "0.2-0.5 (significant)", "0.5-1.0 (large)", ">1.0 (restructure)"]
    dist = {}
    for i in range(len(bins) - 1):
        mask = (ratios >= bins[i]) & (ratios < bins[i + 1])
        dist[bin_labels[i]] = int(mask.sum())
    stats["ratio_distribution"] = dist

    # Copy-cosine distribution
    copy_bins = [0, 0.9, 0.95, 0.99, 0.999, 1.0001]
    copy_labels = ["<0.90 (large change)", "0.90-0.95", "0.95-0.99 (small change)",
                   "0.99-0.999 (near-identical)", ">0.999 (trivial)"]
    copy_dist = {}
    for i in range(len(copy_bins) - 1):
        mask = (copy_cos >= copy_bins[i]) & (copy_cos < copy_bins[i + 1])
        copy_dist[copy_labels[i]] = int(mask.sum())
    stats["copy_cos_distribution"] = copy_dist

    return stats


def print_stats(stats: dict):
    """Pretty-print delta statistics."""
    label = stats["label"]
    n = stats["n"]

    print(f"\n  --- {label} (n={n}) ---")
    print(f"  Delta/State ratio (||delta||/||state||):")
    print(f"    mean:   {stats['ratio_mean']:.4f} +/- {stats['ratio_std']:.4f}")
    print(f"    median: {stats['ratio_median']:.4f}")
    print(f"    p10:    {stats['ratio_p10']:.4f}")
    print(f"    p90:    {stats['ratio_p90']:.4f}")

    print(f"  Copy cosine (cos(z_t, z_{{t+1}})):")
    print(f"    mean:   {stats['copy_cos_mean']:.4f} +/- {stats['copy_cos_std']:.4f}")
    print(f"    median: {stats['copy_cos_median']:.4f}")
    print(f"    p10:    {stats['copy_cos_p10']:.4f}")

    print(f"  Raw norms:")
    print(f"    ||delta|| mean: {stats['delta_norm_mean']:.4f} +/- {stats['delta_norm_std']:.4f}")
    print(f"    ||state|| mean: {stats['state_norm_mean']:.4f}")

    print(f"  Mean delta alignment: {stats['delta_cos_mean_alignment']:.4f}")

    print(f"\n  Ratio distribution:")
    for label_b, count in stats["ratio_distribution"].items():
        pct = 100 * count / n
        bar = "#" * int(pct / 2)
        print(f"    {label_b:30s}: {count:5d} ({pct:5.1f}%) {bar}")

    print(f"\n  Copy-cosine distribution:")
    for label_b, count in stats["copy_cos_distribution"].items():
        pct = 100 * count / n
        bar = "#" * int(pct / 2)
        print(f"    {label_b:30s}: {count:5d} ({pct:5.1f}%) {bar}")


def main():
    parser = argparse.ArgumentParser(description="Phase 0.4: Delta magnitude analysis")
    parser.add_argument("--checkpoint", required=True, help="CodeWM checkpoint")
    parser.add_argument("--data", required=True, help="CodeWM HDF5 data")
    parser.add_argument("--kernel-checkpoint", default=None, help="KernelWM checkpoint (optional)")
    parser.add_argument("--kernel-data", default=None, help="KernelWM HDF5 data (optional)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device

    print("=" * 60)
    print("DELTA MAGNITUDE ANALYSIS — Signal-to-Noise Assessment")
    print("=" * 60)

    # ── CodeWM analysis ──
    print(f"\nLoading CodeWM: {args.checkpoint}")
    model, cfg = load_codewm(args.checkpoint, device=device)

    f = h5py.File(args.data, "r")
    N = f["before_tokens"].shape[0]
    n = min(args.n_samples, N)
    idx = np.sort(np.random.permutation(N)[:n])

    before = torch.from_numpy(f["before_tokens"][idx.tolist()].astype(np.int64)).to(device)
    after = torch.from_numpy(f["after_tokens"][idx.tolist()].astype(np.int64)).to(device)
    f.close()

    code_stats = analyze_deltas(model, before, after, args.batch_size, device, "CodeWM (Python)")
    print_stats(code_stats)

    # ── KernelWM analysis (optional) ──
    if args.kernel_checkpoint and args.kernel_data:
        print(f"\nLoading KernelWM: {args.kernel_checkpoint}")
        kernel_model, kernel_cfg = load_codewm(args.kernel_checkpoint, device=device)

        fk = h5py.File(args.kernel_data, "r")
        Nk = fk["before_tokens"].shape[0]
        nk = min(args.n_samples, Nk)
        idxk = np.sort(np.random.permutation(Nk)[:nk])

        k_before = torch.from_numpy(fk["before_tokens"][idxk.tolist()].astype(np.int64)).to(device)
        k_after = torch.from_numpy(fk["after_tokens"][idxk.tolist()].astype(np.int64)).to(device)
        fk.close()

        kernel_stats = analyze_deltas(kernel_model, k_before, k_after, args.batch_size, device, "KernelWM (CUDA)")
        print_stats(kernel_stats)

        # ── Cross-domain comparison ──
        print(f"\n{'=' * 60}")
        print("CROSS-DOMAIN COMPARISON")
        print(f"{'=' * 60}")
        print(f"  {'Metric':<35s} {'CodeWM':>10s} {'KernelWM':>10s} {'Ratio':>10s}")
        print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10}")

        metrics = [
            ("Delta/State ratio", "ratio_mean"),
            ("Copy cosine", "copy_cos_mean"),
            ("||delta|| mean", "delta_norm_mean"),
        ]
        for label, key in metrics:
            cv = code_stats[key]
            kv = kernel_stats[key]
            r = kv / (cv + 1e-8)
            print(f"  {label:<35s} {cv:10.4f} {kv:10.4f} {r:10.2f}x")

        print(f"\n  Interpretation:")
        code_ratio = code_stats["ratio_mean"]
        kernel_ratio = kernel_stats["ratio_mean"]
        if code_ratio < 0.05 and kernel_ratio > 0.1:
            print(f"    CONFIRMED: Python changes ({code_ratio:.4f}) are much smaller than kernel changes ({kernel_ratio:.4f}).")
            print(f"    Delta prediction has {kernel_ratio/code_ratio:.0f}x more signal in the kernel domain.")
        elif code_ratio < kernel_ratio:
            print(f"    Kernel domain has more signal ({kernel_ratio:.4f} vs {code_ratio:.4f}).")
        else:
            print(f"    Surprising: Code domain has comparable or more signal.")
    else:
        print("\n  (No kernel checkpoint provided — skipping cross-domain comparison)")
        print("  To compare: add --kernel-checkpoint and --kernel-data")

    # ── Verdict ──
    print(f"\n{'=' * 60}")
    print("VERDICT")
    print(f"{'=' * 60}")
    ratio = code_stats["ratio_mean"]
    copy = code_stats["copy_cos_mean"]
    if copy > 0.99:
        print(f"  Copy cosine {copy:.4f} is VERY HIGH.")
        print(f"  The copy-last baseline is near-perfect for most samples.")
        print(f"  Delta prediction must beat a very strong null model.")
    elif copy > 0.95:
        print(f"  Copy cosine {copy:.4f} is HIGH.")
        print(f"  Copy-last is a strong baseline but beatable.")
    else:
        print(f"  Copy cosine {copy:.4f} leaves room for improvement.")

    if ratio < 0.05:
        print(f"  Delta/state ratio {ratio:.4f} is VERY SMALL.")
        print(f"  Recommendation: focus on edit-type segmentation and KernelWM first.")
    elif ratio < 0.15:
        print(f"  Delta/state ratio {ratio:.4f} is moderate.")
        print(f"  Delta prediction may work with the right approach.")
    else:
        print(f"  Delta/state ratio {ratio:.4f} is healthy.")
        print(f"  Good signal for delta prediction.")

    print("\nPhase 0.4 complete.")


if __name__ == "__main__":
    main()
