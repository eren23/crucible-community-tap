#!/usr/bin/env python3
"""Phase 0.2 — Trajectory curvature measurement in latent space.

For sequences of 3+ consecutive commits, computes curvature of the
trajectory in encoder latent space:

    kappa = ||delta_{t} - delta_{t-1}|| / ||delta_{t-1}||

where delta_t = z_{t+1} - z_t.

Low curvature (straight trajectories) means deltas are consistent regardless
of starting state -- this is what makes delta prediction work (Temporal
Straightening, Wang et al. 2025).

High curvature means the same semantic transition produces different delta
vectors depending on position -- delta prediction is unreliable here.

Usage:
    python curvature_analysis.py \
        --checkpoint ~/.crucible-hub/taps/crucible-community-tap/checkpoints/phase5/contrast_15k_seed42/code_wm_best.pt \
        --data ~/.crucible-hub/taps/crucible-community-tap/data/trajectories_quality_563k.h5 \
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


def main():
    parser = argparse.ArgumentParser(description="Phase 0.2: Trajectory curvature analysis")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="HDF5 with trajectory group")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-trajectories", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device

    # ── Load model ──
    print(f"Loading checkpoint: {args.checkpoint}")
    model, cfg = load_codewm(args.checkpoint, device=device)
    print(f"  model_dim={cfg.get('model_dim', 128)}, encoder_loops={cfg.get('encoder_loops', 6)}")

    # ── Load trajectory data ──
    print(f"Loading data: {args.data}")
    f = h5py.File(args.data, "r")

    if "trajectory" not in f:
        print("ERROR: No trajectory group in this HDF5. Use a trajectory dataset.")
        f.close()
        return

    traj_lengths = f["trajectory"]["traj_lengths"][:]
    traj_offsets = f["trajectory"]["traj_offsets"][:]
    n_traj = len(traj_lengths)
    n_use = min(args.max_trajectories, n_traj)

    # Filter to trajectories with length >= 3 (need 3 points for curvature)
    valid_mask = traj_lengths >= 3
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) < n_use:
        n_use = len(valid_indices)
    selected = valid_indices[:n_use]
    print(f"  total trajectories: {n_traj}, valid (len>=3): {len(valid_indices)}, using: {n_use}")

    # ── Encode all states in selected trajectories ──
    # Gather unique row indices we need to encode
    row_indices = []
    traj_row_ranges = []
    for ti in selected:
        offset = traj_offsets[ti]
        length = traj_lengths[ti]
        traj_row_ranges.append((len(row_indices), length))
        row_indices.extend(range(offset, offset + length))

    row_indices = np.array(row_indices)
    print(f"  total states to encode: {len(row_indices)}")

    # Load token data
    # before_tokens[i] is state at step i, after_tokens[i] is state at step i+1
    # For trajectories: before_tokens[offset+j] is the j-th state in the trajectory
    # after_tokens[offset+j] is the (j+1)-th state
    # So the full trajectory states are: before[offset], before[offset+1], ..., before[offset+L-1], after[offset+L-1]
    # But simpler: just use before_tokens for all steps, which gives us the state before each edit

    tokens_all = f["before_tokens"][row_indices.tolist()]
    actions_all = f["edit_actions"][row_indices.tolist()]

    # Also load after_tokens for the last step of each trajectory
    after_last_indices = []
    for ti in selected:
        offset = traj_offsets[ti]
        length = traj_lengths[ti]
        after_last_indices.append(offset + length - 1)
    after_last = f["after_tokens"][after_last_indices]

    f.close()

    tokens_tensor = torch.from_numpy(tokens_all.astype(np.int64)).to(device)
    actions_np = actions_all.astype(np.float32)

    # Encode all states
    print("Encoding states...")
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(tokens_tensor), args.batch_size):
            end = min(start + args.batch_size, len(tokens_tensor))
            z = model.state_encoder(tokens_tensor[start:end])
            embeddings.append(z.cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)  # [total_states, D]

    # Also encode after-last states
    after_last_tensor = torch.from_numpy(after_last.astype(np.int64)).to(device)
    after_last_embeddings = []
    with torch.no_grad():
        for start in range(0, len(after_last_tensor), args.batch_size):
            end = min(start + args.batch_size, len(after_last_tensor))
            z = model.state_encoder(after_last_tensor[start:end])
            after_last_embeddings.append(z.cpu().numpy())
    after_last_embeddings = np.concatenate(after_last_embeddings, axis=0)

    # ── Compute curvatures per trajectory ──
    print("\nComputing curvatures...")
    all_curvatures = []
    all_delta_norms = []
    all_state_norms = []
    all_cos_consecutive = []
    traj_mean_curvatures = []

    for i, ti in enumerate(selected):
        row_start, length = traj_row_ranges[i]

        # Build state sequence: z_0, z_1, ..., z_{L-1}, z_L
        # z_0..z_{L-1} from before_tokens, z_L from after_tokens[last]
        z_seq = np.zeros((length + 1, embeddings.shape[1]))
        z_seq[:length] = embeddings[row_start:row_start + length]
        z_seq[length] = after_last_embeddings[i]

        # Compute deltas: delta_j = z_{j+1} - z_j for j in 0..L-1
        deltas = z_seq[1:] - z_seq[:-1]  # [L, D]

        # Delta norms
        d_norms = np.linalg.norm(deltas, axis=1)
        s_norms = np.linalg.norm(z_seq[:-1], axis=1)
        all_delta_norms.extend(d_norms.tolist())
        all_state_norms.extend(s_norms.tolist())

        # Curvature: kappa_j = ||delta_j - delta_{j-1}|| / ||delta_{j-1}||
        # For j in 1..L-1
        traj_curvatures = []
        for j in range(1, len(deltas)):
            d_diff = np.linalg.norm(deltas[j] - deltas[j - 1])
            d_prev = np.linalg.norm(deltas[j - 1])
            if d_prev > 1e-8:
                kappa = d_diff / d_prev
                traj_curvatures.append(kappa)

                # Also cosine between consecutive deltas
                cos = np.dot(deltas[j], deltas[j - 1]) / (
                    np.linalg.norm(deltas[j]) * np.linalg.norm(deltas[j - 1]) + 1e-8
                )
                all_cos_consecutive.append(cos)

        all_curvatures.extend(traj_curvatures)
        if traj_curvatures:
            traj_mean_curvatures.append(np.mean(traj_curvatures))

    all_curvatures = np.array(all_curvatures)
    all_delta_norms = np.array(all_delta_norms)
    all_state_norms = np.array(all_state_norms)
    all_cos_consecutive = np.array(all_cos_consecutive)
    traj_mean_curvatures = np.array(traj_mean_curvatures)

    # ── Report ──
    print("\n" + "=" * 60)
    print("CURVATURE STATISTICS")
    print("=" * 60)

    print(f"\nTotal curvature measurements: {len(all_curvatures)}")
    print(f"Trajectories analyzed: {len(traj_mean_curvatures)}")

    print(f"\n--- Per-step curvature (kappa) ---")
    print(f"  mean:   {all_curvatures.mean():.4f}")
    print(f"  median: {np.median(all_curvatures):.4f}")
    print(f"  std:    {all_curvatures.std():.4f}")
    print(f"  p25:    {np.percentile(all_curvatures, 25):.4f}")
    print(f"  p75:    {np.percentile(all_curvatures, 75):.4f}")
    print(f"  p90:    {np.percentile(all_curvatures, 90):.4f}")

    print(f"\n--- Per-trajectory mean curvature ---")
    print(f"  mean:   {traj_mean_curvatures.mean():.4f}")
    print(f"  median: {np.median(traj_mean_curvatures):.4f}")
    print(f"  std:    {traj_mean_curvatures.std():.4f}")

    print(f"\n--- Consecutive delta cosine similarity ---")
    print(f"  mean:   {all_cos_consecutive.mean():.4f}")
    print(f"  median: {np.median(all_cos_consecutive):.4f}")
    print(f"  std:    {all_cos_consecutive.std():.4f}")
    print(f"  Interpretation: >0.5 = mostly straight, <0 = curved/random")

    print(f"\n--- Delta magnitude ---")
    ratios = all_delta_norms / (all_state_norms + 1e-8)
    print(f"  ||delta|| mean: {all_delta_norms.mean():.4f} +/- {all_delta_norms.std():.4f}")
    print(f"  ||state|| mean: {all_state_norms.mean():.4f}")
    print(f"  ||delta||/||state|| mean: {ratios.mean():.4f} +/- {ratios.std():.4f}")

    # ── Curvature histogram (text-based) ──
    print(f"\n--- Curvature distribution ---")
    bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, float("inf")]
    labels = ["<0.5 (straight)", "0.5-1.0 (mild)", "1.0-1.5 (moderate)",
              "1.5-2.0 (curved)", "2.0-3.0 (sharp)", "3.0-5.0 (very sharp)", ">5.0 (chaotic)"]
    for i in range(len(bins) - 1):
        mask = (all_curvatures >= bins[i]) & (all_curvatures < bins[i + 1])
        count = mask.sum()
        pct = 100 * count / len(all_curvatures)
        bar = "#" * int(pct / 2)
        print(f"  {labels[i]:25s}: {count:5d} ({pct:5.1f}%) {bar}")

    # ── Straightness score (fraction of low-curvature steps) ──
    straight_frac = (all_curvatures < 1.0).mean()
    print(f"\n--- Straightness score ---")
    print(f"  Fraction with kappa < 1.0: {straight_frac:.3f}")
    print(f"  Interpretation:")
    if straight_frac > 0.7:
        print(f"    GOOD: Trajectories are mostly straight. Delta prediction should work.")
    elif straight_frac > 0.4:
        print(f"    MIXED: Some trajectories are straight, some curved. Per-edit-type segmentation needed.")
    else:
        print(f"    POOR: Trajectories are mostly curved. Delta prediction will be inconsistent.")

    print("\nPhase 0.2 complete.")


if __name__ == "__main__":
    main()
