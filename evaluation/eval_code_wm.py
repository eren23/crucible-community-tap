#!/usr/bin/env python3
"""Evaluate a Code World Model on held-out git edit transitions.

Computes:
    - MSE in latent space (pred vs target)
    - Cosine similarity between predicted and target embeddings
    - Nearest-neighbor accuracy (does nearest neighbor of pred = actual target?)

Compares against baselines:
    - Random: random embedding predictions
    - Copy-last: predict z_t as z_{t+1} (no change baseline)

Usage::

    python eval_code_wm.py --checkpoint model.pt --data edits.h5

Or without a trained model (baseline-only evaluation)::

    python eval_code_wm.py --data edits.h5 --baseline-only
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, device: str = "cpu") -> Any:
    """Load a saved CodeWorldModel checkpoint."""
    # Import code_wm module which registers the model
    code_wm_path = Path(__file__).parent.parent / "architectures" / "code_wm" / "code_wm.py"
    if code_wm_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("code_wm", code_wm_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # Crucible checkpoint format
        from crucible.models.registry import build_model

        model = build_model("code_wm", None)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = checkpoint

    model.to(device)
    model.set_mode_inference()
    return model


def load_data(
    data_path: str,
    num_samples: int = 500,
) -> dict[str, np.ndarray]:
    """Load transitions from HDF5."""
    import h5py

    with h5py.File(data_path, "r") as f:
        meta = f["metadata"]
        num_edits = int(meta.attrs["num_edits"])
        context_window = int(meta.attrs["context_window"])

        # Sample indices
        n = min(num_samples, num_edits)
        indices = np.random.choice(num_edits, size=n, replace=False)
        indices.sort()

        before_tokens = np.zeros((n, context_window), dtype=np.int64)
        edit_actions = np.zeros((n, 4), dtype=np.float32)
        after_tokens = np.zeros((n, context_window), dtype=np.int64)

        edits = f["edits"]
        for i, idx in enumerate(indices):
            g = edits[str(idx)]
            before_tokens[i] = np.array(g["before_tokens"], dtype=np.int64)
            edit_actions[i] = np.array(g["edit_action"], dtype=np.float32)
            after_tokens[i] = np.array(g["after_tokens"], dtype=np.int64)

    return {
        "before_tokens": before_tokens,
        "edit_actions": edit_actions,
        "after_tokens": after_tokens,
        "num_samples": n,
        "context_window": context_window,
    }


@torch.no_grad()
def run_model_evaluation(
    model: Any,
    data: dict[str, np.ndarray],
    device: str = "cpu",
    batch_size: int = 64,
) -> dict[str, float]:
    """Run model on held-out data and compute metrics."""
    n = data["num_samples"]
    before = torch.from_numpy(data["before_tokens"]).to(device)
    actions = torch.from_numpy(data["edit_actions"]).to(device)
    after = torch.from_numpy(data["after_tokens"]).to(device)

    all_pred = []
    all_target = []
    all_z_current = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        b_states = before[start:end]
        b_actions = actions[start:end]
        b_next = after[start:end]

        # Encode current and next states
        z_current = model.state_encoder(b_states)       # [B, D]
        z_target = model.target_encoder(b_next)          # [B, D]

        # Predict next state
        z_action = model.action_encoder(b_actions)
        z_pred = model.predictor(z_current.detach(), z_action)

        all_pred.append(z_pred.cpu())
        all_target.append(z_target.cpu())
        all_z_current.append(z_current.cpu())

    pred = torch.cat(all_pred)       # [N, D]
    target = torch.cat(all_target)   # [N, D]
    current = torch.cat(all_z_current)  # [N, D]

    # --- Metrics ---
    # 1. MSE in latent space
    mse = F.mse_loss(pred, target).item()

    # 2. Cosine similarity
    cos_sim = F.cosine_similarity(pred, target, dim=-1).mean().item()

    # 3. Nearest-neighbor accuracy
    # For each prediction, find its nearest neighbor in the target set
    dists = torch.cdist(pred, target)  # [N, N]
    nn_indices = dists.argmin(dim=1)   # [N]
    correct = (nn_indices == torch.arange(n)).float()
    nn_accuracy = correct.mean().item()

    # Top-5 accuracy
    _, top5_indices = dists.topk(5, dim=1, largest=False)
    top5_correct = (top5_indices == torch.arange(n).unsqueeze(1)).any(dim=1).float()
    nn_top5_accuracy = top5_correct.mean().item()

    # --- Baselines ---
    # Random baseline: random vectors
    random_pred = torch.randn_like(pred)
    random_mse = F.mse_loss(random_pred, target).item()
    random_cos = F.cosine_similarity(random_pred, target, dim=-1).mean().item()

    # Copy-last baseline: predict current state as next state
    copy_mse = F.mse_loss(current, target).item()
    copy_cos = F.cosine_similarity(current, target, dim=-1).mean().item()

    return {
        "model/mse": mse,
        "model/cosine_sim": cos_sim,
        "model/nn_accuracy": nn_accuracy,
        "model/nn_top5_accuracy": nn_top5_accuracy,
        "baseline_random/mse": random_mse,
        "baseline_random/cosine_sim": random_cos,
        "baseline_copy/mse": copy_mse,
        "baseline_copy/cosine_sim": copy_cos,
        "num_samples": n,
    }


def compute_baseline_stats(
    data: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Compute baseline data statistics without a trained model."""
    n = data["num_samples"]
    cw = data["context_window"]

    before = torch.from_numpy(data["before_tokens"]).float()
    after = torch.from_numpy(data["after_tokens"]).float()

    # Token-level overlap (how much does code change between transitions?)
    match_rate = (before == after).float().mean().item()

    # Edit diversity
    actions = data["edit_actions"]
    edit_types = actions[:, :3].argmax(axis=1)
    unique_types, counts = np.unique(edit_types, return_counts=True)
    type_dist = {int(t): int(c) for t, c in zip(unique_types, counts)}

    return {
        "data/num_samples": n,
        "data/context_window": cw,
        "data/token_match_rate": match_rate,
        "data/edit_type_distribution": type_dist,
    }


def main():
    parser = argparse.ArgumentParser(description="Code World Model Evaluation")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to HDF5 data")
    parser.add_argument("--num-samples", type=int, default=500, help="Num samples")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--baseline-only", action="store_true", help="Baselines only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print(f"Loading data from {args.data}...")
    data = load_data(args.data, num_samples=args.num_samples)
    print(f"Loaded {data['num_samples']} transitions")

    if args.baseline_only:
        results = compute_baseline_stats(data)
        print("\n=== Data Statistics ===")
        for k, v in results.items():
            print(f"  {k}: {v}")
        return

    if not args.checkpoint:
        print("Error: --checkpoint required unless --baseline-only", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device=args.device)
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    print("Running evaluation...")
    results = run_model_evaluation(
        model, data, device=args.device, batch_size=args.batch_size,
    )

    print("\n=== Code World Model Results ===")
    print(f"  Samples: {results['num_samples']}")
    print()
    print("  Model:")
    print(f"    MSE (latent):     {results['model/mse']:.6f}")
    print(f"    Cosine sim:       {results['model/cosine_sim']:.4f}")
    print(f"    NN accuracy:      {results['model/nn_accuracy']:.4f}")
    print(f"    NN top-5 acc:     {results['model/nn_top5_accuracy']:.4f}")
    print()
    print("  Baseline (random):")
    print(f"    MSE (latent):     {results['baseline_random/mse']:.6f}")
    print(f"    Cosine sim:       {results['baseline_random/cosine_sim']:.4f}")
    print()
    print("  Baseline (copy-last):")
    print(f"    MSE (latent):     {results['baseline_copy/mse']:.6f}")
    print(f"    Cosine sim:       {results['baseline_copy/cosine_sim']:.4f}")
    print()

    # Verdict
    if results["model/mse"] < results["baseline_copy/mse"]:
        print("  >> Model beats copy-last baseline on MSE")
    else:
        print("  >> Model does NOT beat copy-last on MSE (needs more training)")

    if results["model/cosine_sim"] > results["baseline_random/cosine_sim"]:
        print("  >> Model beats random baseline on cosine similarity")


if __name__ == "__main__":
    main()
