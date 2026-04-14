#!/usr/bin/env python3
"""Phase 0.1 — Layer-wise linear probes for CodeWM encoder.

Probes intermediate representations after each loop iteration of the
weight-shared LoopedTransformerBlock to answer:
  1. Which loop iterations contain transition information?
  2. Can linear probes predict the next-state delta direction?
  3. Which action features are recoverable from each layer?

The encoder is weight-shared (single block iterated encoder_loops times),
so "layers" here are loop iterations, not distinct parameter sets. Despite
shared weights, each iteration produces a different representation because
the input evolves.

Usage:
    python geometry_probes.py \
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
import torch.nn.functional as F
from torch import Tensor

# ── Setup imports ──
sys.path.insert(0, str(Path(__file__).parent))
from _shared import load_codewm, resolve_tap_root


# ---------------------------------------------------------------------------
# Extract intermediate loop representations
# ---------------------------------------------------------------------------

def extract_loop_intermediates(
    model,
    tokens: Tensor,
    device: str = "cpu",
) -> list[Tensor]:
    """Run the state encoder and capture output after each loop iteration.

    Returns a list of [B, D] tensors, one per loop iteration, using the
    model's configured readout (attention pooling by default).
    """
    encoder = model.state_encoder
    B = tokens.shape[0]

    # Embedding + CLS prepend + positional encoding
    h = encoder.embedding(tokens)
    cls = encoder.cls_token.expand(B, -1, -1)
    h = torch.cat([cls, h], dim=1)
    h = encoder.pos_enc(h)

    intermediates = []
    for _ in range(encoder.encoder_loops):
        h = encoder.block(h)
        # Apply readout to get [B, D] from [B, S+1, D]
        if encoder.pool_mode == "cls":
            pooled = h[:, 0]
        elif encoder.pool_mode == "attn":
            pooled = encoder.attn_pool(h)
        else:
            pooled = h.mean(dim=1)
        pooled = encoder.norm(pooled)
        intermediates.append(pooled.detach())

    return intermediates


# ---------------------------------------------------------------------------
# Linear probe training (sklearn-free, pure torch)
# ---------------------------------------------------------------------------

def train_linear_probe(
    X: np.ndarray,    # [N, D]
    y: np.ndarray,    # [N] for classification, [N, D] for regression
    task: str = "classification",  # or "regression"
    lr: float = 1e-2,
    epochs: int = 200,
    val_frac: float = 0.2,
) -> dict:
    """Train a linear probe and return metrics on held-out validation set."""
    N = X.shape[0]
    perm = np.random.permutation(N)
    n_val = max(1, int(N * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    X_tr = torch.from_numpy(X[train_idx]).float()
    X_va = torch.from_numpy(X[val_idx]).float()

    if task == "classification":
        y_tr = torch.from_numpy(y[train_idx]).long()
        y_va = torch.from_numpy(y[val_idx]).long()
        n_classes = int(y.max()) + 1
        probe = torch.nn.Linear(X.shape[1], n_classes)
        criterion = torch.nn.CrossEntropyLoss()
    else:
        y_tr = torch.from_numpy(y[train_idx]).float()
        y_va = torch.from_numpy(y[val_idx]).float()
        out_dim = y_tr.shape[1] if y_tr.ndim > 1 else 1
        if y_tr.ndim == 1:
            y_tr = y_tr.unsqueeze(1)
            y_va = y_va.unsqueeze(1)
        probe = torch.nn.Linear(X.shape[1], out_dim)
        criterion = torch.nn.MSELoss()

    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    for _ in range(epochs):
        probe.train()
        opt.zero_grad()
        loss = criterion(probe(X_tr), y_tr)
        loss.backward()
        opt.step()

    # Validation
    probe.train(False)
    with torch.no_grad():
        if task == "classification":
            logits = probe(X_va)
            preds = logits.argmax(dim=1)
            acc = (preds == y_va).float().mean().item()
            return {"accuracy": acc, "n_val": n_val}
        else:
            pred = probe(X_va)
            target = y_va
            cos = F.cosine_similarity(pred, target, dim=1).mean().item()
            mse = F.mse_loss(pred, target).item()
            return {"cosine_sim": cos, "mse": mse, "n_val": n_val}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 0.1: Layer-wise linear probes")
    parser.add_argument("--checkpoint", required=True, help="CodeWM checkpoint path")
    parser.add_argument("--data", required=True, help="HDF5 data path (commitpackft_with_diffs.h5)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-samples", type=int, default=5000,
                        help="Number of samples to use (default: 5000)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device

    # ── Load model ──
    print(f"Loading checkpoint: {args.checkpoint}")
    model, cfg = load_codewm(args.checkpoint, device=device)
    encoder_loops = cfg.get("encoder_loops", 6)
    model_dim = cfg.get("model_dim", 128)
    print(f"  model_dim={model_dim}, encoder_loops={encoder_loops}")
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Load data ──
    print(f"Loading data: {args.data}")
    f = h5py.File(args.data, "r")
    N = f["before_tokens"].shape[0]
    n = min(args.n_samples, N)
    idx = np.random.permutation(N)[:n]
    idx.sort()

    before = torch.from_numpy(f["before_tokens"][idx.tolist()].astype(np.int64)).to(device)
    after = torch.from_numpy(f["after_tokens"][idx.tolist()].astype(np.int64)).to(device)
    actions = torch.from_numpy(f["edit_actions"][idx.tolist()].astype(np.float32))
    f.close()
    print(f"  samples: {n}, seq_len: {before.shape[1]}, action_dim: {actions.shape[1]}")

    # ── Extract intermediates for before and after states ──
    print("\nExtracting encoder intermediates...")
    all_before_layers = [[] for _ in range(encoder_loops)]
    all_after_layers = [[] for _ in range(encoder_loops)]

    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            b_inter = extract_loop_intermediates(model, before[start:end], device)
            a_inter = extract_loop_intermediates(model, after[start:end], device)
            for li in range(encoder_loops):
                all_before_layers[li].append(b_inter[li].cpu())
                all_after_layers[li].append(a_inter[li].cpu())

    before_layers = [torch.cat(bl, dim=0).numpy() for bl in all_before_layers]
    after_layers = [torch.cat(al, dim=0).numpy() for al in all_after_layers]

    # Final encoder output (standard forward)
    print("Computing final encoder outputs (standard forward)...")
    final_before = []
    final_after = []
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            fb = model.state_encoder(before[start:end])
            fa = model.state_encoder(after[start:end])
            final_before.append(fb.cpu())
            final_after.append(fa.cpu())
    final_before_np = torch.cat(final_before, dim=0).numpy()
    final_after_np = torch.cat(final_after, dim=0).numpy()

    # ── Compute deltas (z_{t+1} - z_t) for each layer ──
    deltas = [after_layers[li] - before_layers[li] for li in range(encoder_loops)]
    final_delta = final_after_np - final_before_np

    # ── Derive edit type labels from action vectors ──
    action_np = actions.numpy()
    if action_np.shape[1] >= 3:
        edit_type_labels = action_np[:, :3].argmax(axis=1)
    else:
        edit_type_labels = np.zeros(n, dtype=np.int64)

    # ── PROBE 1: Edit type classification from each layer ──
    print("\n" + "=" * 60)
    print("PROBE 1: Edit type classification (linear probe)")
    print("=" * 60)
    for li in range(encoder_loops):
        result = train_linear_probe(
            before_layers[li], edit_type_labels, task="classification"
        )
        print(f"  Loop {li+1}/{encoder_loops}: accuracy={result['accuracy']:.3f} (n_val={result['n_val']})")
    result = train_linear_probe(final_before_np, edit_type_labels, task="classification")
    print(f"  Final output:    accuracy={result['accuracy']:.3f} (n_val={result['n_val']})")

    # ── PROBE 2: Delta direction prediction ──
    print("\n" + "=" * 60)
    print("PROBE 2: Next-state delta direction prediction")
    print("  Target: z_{t+1} - z_t at final layer")
    print("  Probe: linear map from loop-i representation to final delta")
    print("=" * 60)
    for li in range(encoder_loops):
        result = train_linear_probe(
            before_layers[li], final_delta, task="regression"
        )
        print(f"  Loop {li+1}/{encoder_loops}: cos_sim={result['cosine_sim']:.4f}  mse={result['mse']:.6f}")
    result = train_linear_probe(final_before_np, final_delta, task="regression")
    print(f"  Final output:    cos_sim={result['cosine_sim']:.4f}  mse={result['mse']:.6f}")

    # ── PROBE 3: Action feature recovery ──
    print("\n" + "=" * 60)
    print("PROBE 3: Action feature recovery (per-dim regression)")
    print("  Which of the 15 action dims are recoverable from the state?")
    print("=" * 60)
    action_dim = action_np.shape[1]
    per_dim_mse = []
    for d in range(action_dim):
        target = action_np[:, d:d+1]
        result = train_linear_probe(final_before_np, target, task="regression")
        per_dim_mse.append(result["mse"])
    print(f"  Action dim MSEs (from final encoder, lower=more recoverable):")
    for d in range(action_dim):
        bar = "#" * max(1, int(50 * (1.0 - min(per_dim_mse[d], 1.0))))
        print(f"    dim {d:2d}: mse={per_dim_mse[d]:.4f} {bar}")

    # ── PROBE 4: Cross-layer delta consistency ──
    print("\n" + "=" * 60)
    print("PROBE 4: Cross-layer delta consistency")
    print("  Cosine similarity between deltas at different loop iterations")
    print("=" * 60)
    delta_tensors = [torch.from_numpy(d) for d in deltas]
    for i in range(encoder_loops):
        row = []
        for j in range(encoder_loops):
            cos = F.cosine_similarity(delta_tensors[i], delta_tensors[j], dim=1).mean().item()
            row.append(f"{cos:.3f}")
        print(f"  Loop {i+1}: [{', '.join(row)}]")

    # ── PROBE 5: Delta magnitude per layer ──
    print("\n" + "=" * 60)
    print("PROBE 5: Delta magnitude per loop iteration")
    print("  ||z_{t+1} - z_t|| and ||delta||/||z_t||")
    print("=" * 60)
    for li in range(encoder_loops):
        d_norm = np.linalg.norm(deltas[li], axis=1)
        z_norm = np.linalg.norm(before_layers[li], axis=1)
        ratio = d_norm / (z_norm + 1e-8)
        print(f"  Loop {li+1}: ||delta||={d_norm.mean():.4f} +/- {d_norm.std():.4f}  "
              f"||z||={z_norm.mean():.4f}  ratio={ratio.mean():.4f} +/- {ratio.std():.4f}")
    d_norm = np.linalg.norm(final_delta, axis=1)
    z_norm = np.linalg.norm(final_before_np, axis=1)
    ratio = d_norm / (z_norm + 1e-8)
    print(f"  Final:  ||delta||={d_norm.mean():.4f} +/- {d_norm.std():.4f}  "
          f"||z||={z_norm.mean():.4f}  ratio={ratio.mean():.4f} +/- {ratio.std():.4f}")

    print("\nPhase 0.1 complete.")


if __name__ == "__main__":
    main()
