#!/usr/bin/env python3
"""Full-val KNN + eff_rank evaluation on a CodeWM checkpoint.

Addresses the B1 audit finding: training-time val/knn metrics run on a
single 128-sample batch and are noisy. This script runs the same
metrics on the FULL val set (typically 5000 samples) for a defensible
paper number.

Computes:
  - eff_rank(online/target/pred) via SVD entropy on normalized embeddings
  - KNN@1/5/10/50 — does target[i] appear in top-k nearest neighbors of pred[i]?
  - Cross diagonal (mean cos(pred[i], target[i])) vs off-diagonal mean
  - Random baselines for comparison

Usage::

    python full_val_knn.py \\
        --checkpoint /path/to/code_wm_step100000.pt \\
        --data /path/to/commitpackft_with_diffs.h5 \\
        --device cpu \\
        [--wandb-run-id RUN_ID]   # optional: log to existing W&B run

    # Or write to a JSON file:
    python full_val_knn.py --checkpoint ... --data ... --out results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import h5py

# Import shared checkpoint loader (handles WM_POOL_MODE + plugin loading)
_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from _shared import load_codewm  # noqa: E402


def eff_rank(X: torch.Tensor) -> float:
    """SVD-entropy effective rank. X: [N, D]."""
    try:
        s = torch.linalg.svdvals(X.float())
        s = s / (s.sum() + 1e-12)
        return float(torch.exp(-(s * (s + 1e-12).log()).sum()).item())
    except Exception:
        return float("nan")


def encode_val_set(
    model: Any,
    data_path: str,
    *,
    device: str,
    seed: int,
    val_frac: float,
    chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Encode the full val split via state/target/predictor."""
    with h5py.File(data_path, "r") as f:
        before_ds = f["before_tokens"]
        after_ds = f["after_tokens"]
        action_ds = f["edit_actions"]
        num_edits = before_ds.shape[0]

        # Reproduce the training split (same seed + val_frac).
        np.random.seed(seed)
        all_indices = np.random.permutation(num_edits)
        n_val = max(int(num_edits * val_frac), chunk)
        val_indices = np.sort(all_indices[:n_val])
        N_full = len(val_indices)
        print(f"  val split: seed={seed} val_frac={val_frac} N_full={N_full}")

        z_before_all: list[torch.Tensor] = []
        z_after_all: list[torch.Tensor] = []
        z_pred_all: list[torch.Tensor] = []

        model.train(False)
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, N_full, chunk):
                idx = np.sort(val_indices[i:i + chunk]).tolist()
                before = torch.from_numpy(before_ds[idx].astype(np.int64)).to(device)
                after = torch.from_numpy(after_ds[idx].astype(np.int64)).to(device)
                acts = torch.from_numpy(action_ds[idx].astype(np.float32)).to(device)
                zb = model.state_encoder(before)
                za = model.target_encoder(after)
                zc = model.action_encoder(acts)
                zp = model.predictor(zb, zc)
                z_before_all.append(F.normalize(zb, dim=-1).cpu())
                z_after_all.append(F.normalize(za, dim=-1).cpu())
                z_pred_all.append(F.normalize(zp, dim=-1).cpu())
        dt = time.time() - t0
        print(f"  encoded {N_full} samples in {dt:.1f}s ({1000 * dt / N_full:.1f}ms/sample)")

    return (
        torch.cat(z_before_all, dim=0),
        torch.cat(z_after_all, dim=0),
        torch.cat(z_pred_all, dim=0),
        N_full,
    )


def retrieval_metrics(
    z_pred: torch.Tensor, z_target: torch.Tensor, *, chunk: int,
) -> dict[str, float]:
    """KNN@k and cross-diag/off-diag stats. Inputs are L2-normalized."""
    N = z_pred.shape[0]
    topk_hits = {1: 0, 5: 0, 10: 0, 50: 0}
    diag_sum = 0.0
    off_sum = 0.0
    off_count = 0

    for i in range(0, N, chunk):
        pred_chunk = z_pred[i:i + chunk]
        sims = pred_chunk @ z_target.T  # [chunk, N]
        chunk_n = sims.shape[0]
        true_idx = torch.arange(i, i + chunk_n)

        diag_vals = sims[torch.arange(chunk_n), true_idx]
        diag_sum += float(diag_vals.sum())

        # Top-50 once; slice for smaller k
        top_k_max = min(50, N)
        top_idx = sims.topk(top_k_max, dim=-1).indices
        true_col = true_idx.unsqueeze(-1)
        for k in (1, 5, 10, 50):
            if k > N:
                continue
            hits = (top_idx[:, :k] == true_col).any(-1)
            topk_hits[k] += int(hits.sum())

        mask = torch.ones_like(sims, dtype=torch.bool)
        mask[torch.arange(chunk_n), true_idx] = False
        off_sum += float(sims[mask].sum())
        off_count += int(mask.sum())

    return {
        "knn_top1": topk_hits[1] / N,
        "knn_top5": topk_hits[5] / N,
        "knn_top10": topk_hits[10] / N,
        "knn_top50": topk_hits[50] / N,
        "cross_diag_mean": diag_sum / N,
        "cross_off_mean": off_sum / max(off_count, 1),
        "random_top1": 1.0 / N,
        "random_top5": 5.0 / N,
        "random_top10": 10.0 / N,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-val KNN eval on a CodeWM checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to code_wm_*.pt")
    parser.add_argument("--data", required=True, help="HDF5 data path (commitpackft_with_diffs.h5)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42, help="Must match training split seed")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Must match training val_frac")
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--out", default="", help="Optional JSON output path")
    parser.add_argument("--wandb-run-id", default="", help="Optional W&B run ID to attach to")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-entity", default="")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    model, cfg = load_codewm(args.checkpoint, device=args.device)
    model.train(False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model_dim={cfg.get('model_dim')}, encoder_loops={cfg.get('encoder_loops')}, "
          f"params={n_params:,}")

    print(f"Encoding full val set from {args.data}")
    z_before, z_after, z_pred, N_full = encode_val_set(
        model, args.data,
        device=args.device, seed=args.seed, val_frac=args.val_frac, chunk=args.chunk,
    )

    print("Computing eff_rank + KNN + gap metrics")
    eff_o = eff_rank(z_before)
    eff_t = eff_rank(z_after)
    eff_p = eff_rank(z_pred)
    ret = retrieval_metrics(z_pred, z_after, chunk=args.chunk)

    results = {
        "N_full": N_full,
        "model_params": n_params,
        "eff_rank_online": eff_o,
        "eff_rank_target": eff_t,
        "eff_rank_pred": eff_p,
        **ret,
    }
    results["cross_gap"] = results["cross_diag_mean"] - results["cross_off_mean"]

    print()
    print("=" * 66)
    print(f"  Full-val retrieval (N={N_full})")
    print("=" * 66)
    print(f"  eff_rank:  online={eff_o:.2f}  target={eff_t:.2f}  pred={eff_p:.2f}")
    print(f"  KNN@1  = {ret['knn_top1']:.4f}  (random {ret['random_top1']:.4f})")
    print(f"  KNN@5  = {ret['knn_top5']:.4f}  (random {ret['random_top5']:.4f})")
    print(f"  KNN@10 = {ret['knn_top10']:.4f}  (random {ret['random_top10']:.4f})")
    print(f"  KNN@50 = {ret['knn_top50']:.4f}")
    print(f"  cross: diag={ret['cross_diag_mean']:+.4f}  off={ret['cross_off_mean']:+.4f}  "
          f"gap={results['cross_gap']:+.4f}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\n  wrote {args.out}")

    if args.wandb_run_id:
        try:
            import wandb
            run = wandb.init(
                project=args.wandb_project or os.environ.get("WANDB_PROJECT"),
                entity=args.wandb_entity or os.environ.get("WANDB_ENTITY"),
                id=args.wandb_run_id,
                resume="must",
            )
            run.log({f"val_full/{k}": v for k, v in results.items() if isinstance(v, (int, float))})
            for k, v in results.items():
                if isinstance(v, (int, float)):
                    run.summary[f"val_full/{k}"] = v
            run.finish()
            print(f"  logged to W&B run {args.wandb_run_id}")
        except Exception as exc:
            print(f"  W&B logging failed: {exc}")


if __name__ == "__main__":
    main()
