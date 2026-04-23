#!/usr/bin/env python3
"""CodeDeltaTok efficiency benchmark.

Measures the systems advantage of 1-token delta representation:
  - Encoding latency (ms/change)
  - Storage per change (bytes)
  - ANN retrieval latency at scale
  - Comparison to full-feature storage

Usage:
    python eval_efficiency.py \
        --features data/commitpackft_unixcoder_features.h5 \
        --checkpoint checkpoints/code_deltatok_final.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def main():
    parser = argparse.ArgumentParser(description="CodeDeltaTok efficiency benchmark")
    parser.add_argument("--features", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    import h5py
    f = h5py.File(args.features, "r")
    before_feat = f["before_features"][:args.num_samples].astype(np.float32)
    after_feat = f["after_features"][:args.num_samples].astype(np.float32)
    N, D = before_feat.shape
    print(f"Loaded {N} samples, {D}-dim features")

    # Load model
    from architectures.code_deltatok.code_deltatok import CodeDeltaTok
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    sd = ckpt["model_state_dict"]
    num_blocks = sum(1 for k in sd if k.startswith("encoder.") and k.endswith(".norm1.weight"))
    num_tokens = sd["z_embed"].shape[1]
    feature_dim = sd["z_embed"].shape[2]

    model = CodeDeltaTok(feature_dim=feature_dim, num_blocks=num_blocks, num_delta_tokens=num_tokens)
    model.load_state_dict(sd)
    model.requires_grad_(False)
    model = model.to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params, K={num_tokens}, blocks={num_blocks}")
    print(f"Device: {args.device}")

    # ---- Encoding latency ----
    print("\n=== Encoding Latency ===")
    prev_t = torch.from_numpy(before_feat).to(args.device)
    next_t = torch.from_numpy(after_feat).to(args.device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model.encode(prev_t[:32], next_t[:32])
    if args.device == "cuda":
        torch.cuda.synchronize()

    # Batch encoding
    for bs in [1, 16, 64, 256]:
        times = []
        n_iters = min(50, N // bs)
        for i in range(n_iters):
            s = i * bs
            e = s + bs
            if args.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model.encode(prev_t[s:e], next_t[s:e])
            if args.device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        mean_ms = np.mean(times) * 1000
        per_sample = mean_ms / bs
        print(f"  batch={bs:>4}: {mean_ms:.2f}ms/batch, {per_sample:.3f}ms/change")

    # ---- Storage comparison ----
    print("\n=== Storage per 1M Changes ===")

    # Raw before+after features
    raw_bytes_per_change = D * 4 * 2  # float32, both before and after
    # Raw delta (after - before)
    delta_bytes_per_change = D * 4
    # DeltaTok: K tokens of D dims
    dt_bytes_per_change = num_tokens * D * 4
    # DeltaTok quantized to float16
    dt_f16_bytes = num_tokens * D * 2

    M = 1_000_000
    print(f"  Raw before+after features:  {raw_bytes_per_change:>6} bytes/change = {raw_bytes_per_change * M / 1e9:.1f} GB/M")
    print(f"  Raw delta (after-before):   {delta_bytes_per_change:>6} bytes/change = {delta_bytes_per_change * M / 1e9:.1f} GB/M")
    print(f"  DeltaTok K={num_tokens} (float32):   {dt_bytes_per_change:>6} bytes/change = {dt_bytes_per_change * M / 1e9:.1f} GB/M")
    print(f"  DeltaTok K={num_tokens} (float16):   {dt_f16_bytes:>6} bytes/change = {dt_f16_bytes * M / 1e9:.1f} GB/M")

    # Compression ratios
    print(f"\n  Compression vs raw features: {raw_bytes_per_change / dt_bytes_per_change:.1f}x")
    print(f"  Compression vs raw delta:    {delta_bytes_per_change / dt_bytes_per_change:.1f}x")
    if num_tokens == 1:
        print(f"  (K=1: delta token is same size as raw delta — win is in quality, not compression)")

    # ---- ANN retrieval simulation ----
    print("\n=== Retrieval Latency (brute-force cosine) ===")
    # Encode all
    with torch.no_grad():
        all_z = []
        for i in range(0, N, 256):
            e = min(i + 256, N)
            dt = model.encode(prev_t[i:e], next_t[i:e])
            all_z.append(dt[:, 0] if dt.shape[1] == 1 else dt.mean(dim=1))
        all_z = torch.cat(all_z, dim=0)  # [N, D]
        all_z_n = torch.nn.functional.normalize(all_z, dim=-1)

    gallery_af = torch.from_numpy(after_feat).to(args.device)
    gallery_af_n = torch.nn.functional.normalize(gallery_af, dim=-1)

    # Single query retrieval
    times = []
    for i in range(min(100, N)):
        q = all_z_n[i:i+1]
        if args.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        sim = q @ gallery_af_n.T
        top_idx = sim.argmax()
        if args.device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    print(f"  Single query vs {N} gallery: {np.mean(times)*1000:.3f}ms/query")
    print(f"  (brute force, {args.device})")

    # ---- Summary ----
    print("\n=== Summary ===")
    print(f"  Tokens per change:     {num_tokens}")
    print(f"  Feature dim:           {D}")
    print(f"  Trainable params:      {n_params:,}")
    print(f"  Encoding:              {per_sample:.3f} ms/change (batch=256)")
    print(f"  Storage:               {dt_bytes_per_change} bytes/change ({dt_bytes_per_change * M / 1e9:.1f} GB/M)")
    print(f"  vs raw features:       same size (K=1 win is quality, not size)")
    print(f"  Retrieval:             {np.mean(times)*1000:.3f} ms/query (brute force, N={N})")

    f.close()


if __name__ == "__main__":
    main()
