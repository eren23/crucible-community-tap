#!/usr/bin/env python3
"""CodeDeltaTok training script.

Trains a DeltaTok-inspired delta tokenizer on pre-computed frozen backbone
features. No backbone inference during training -- just loads feature vectors
from HDF5 and learns to compress deltas.

Env vars:
    CDT_HDF5_PATH:     Path to pre-computed features HDF5
    CDT_FEATURE_DIM:   Feature dimension (default: 768)
    CDT_NUM_BLOCKS:    Transformer blocks per encoder/decoder (default: 6)
    CDT_NUM_HEADS:     Attention heads (default: 12)
    CDT_NUM_TOKENS:    Delta tokens K (default: 1)
    CDT_LR:            Learning rate (default: 1e-4)
    CDT_BATCH_SIZE:    Batch size (default: 256)
    CDT_STEPS:         Total training steps (default: 5000)
    CDT_WARMUP:        Warmup steps (default: 200)
    CDT_GRAD_CLIP:     Gradient clipping (default: 0.01, from DeltaTok)
    CDT_SEED:          Random seed (default: 42)
    WANDB_PROJECT:     W&B project (default: crucible-code-deltatok)
    WANDB_RUN_NAME:    W&B run name
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add tap root for imports
tap_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, tap_root)


def main():
    # ---- Config ----
    hdf5_path = os.environ.get("CDT_HDF5_PATH", "")
    if not hdf5_path:
        print("ERROR: CDT_HDF5_PATH is required", file=sys.stderr)
        sys.exit(1)

    lr = float(os.environ.get("CDT_LR", "1e-4"))
    batch_size = int(os.environ.get("CDT_BATCH_SIZE", "256"))
    total_steps = int(os.environ.get("CDT_STEPS", "5000"))
    warmup_steps = int(os.environ.get("CDT_WARMUP", "200"))
    grad_clip = float(os.environ.get("CDT_GRAD_CLIP", "0.01"))
    seed = int(os.environ.get("CDT_SEED", "42"))
    log_interval = int(os.environ.get("CDT_LOG_INTERVAL", "100"))
    save_interval = int(os.environ.get("CDT_SAVE_INTERVAL", "1000"))
    output_dir = os.environ.get("OUTPUT_DIR", "./checkpoints")
    wandb_project = os.environ.get("WANDB_PROJECT", "crucible-code-deltatok")
    wandb_run_name = os.environ.get("WANDB_RUN_NAME", "")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    # ---- Seed ----
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ---- Load data ----
    import h5py

    # Auto-download from HuggingFace if local file doesn't exist
    if not os.path.exists(hdf5_path) and "/" not in hdf5_path:
        hf_repo = os.environ.get("CDT_HF_REPO", "eren23/codewm-data")
        hf_path = os.environ.get("CDT_HF_PATH", f"deltatok/{hdf5_path}")
        print(f"Local file not found, downloading from HF: {hf_repo}/{hf_path}")
        try:
            from huggingface_hub import hf_hub_download
            hdf5_path = hf_hub_download(
                repo_id=hf_repo, filename=hf_path, repo_type="dataset",
            )
            print(f"  Downloaded to: {hdf5_path}")
        except Exception as e:
            print(f"ERROR: Failed to download from HF: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"Loading features from {hdf5_path}")
    f = h5py.File(hdf5_path, "r")
    before_features = f["before_features"]  # [N, D]
    after_features = f["after_features"]    # [N, D]
    N = before_features.shape[0]
    D = before_features.shape[1]
    model_name = str(f["metadata"].attrs.get("model_name", "unknown"))
    print(f"  Samples: {N:,}, feature_dim: {D}")
    print(f"  Backbone: {model_name}")

    # Train/val split
    all_idx = np.random.permutation(N)
    n_val = max(int(N * 0.1), batch_size)
    val_idx = np.sort(all_idx[:n_val])
    train_idx = np.sort(all_idx[n_val:])
    print(f"  Split: {len(train_idx):,} train, {len(val_idx):,} val")

    # ---- Build model ----
    from architectures.code_deltatok.code_deltatok import codedeltatik_from_env
    os.environ.setdefault("CDT_FEATURE_DIM", str(D))
    model = codedeltatik_from_env()
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    print(f"Device: {device}")
    print(f"Training: {total_steps} steps, batch={batch_size}, lr={lr}")

    # ---- W&B ----
    use_wandb = False
    try:
        import wandb
        if not wandb_run_name:
            K = int(os.environ.get("CDT_NUM_TOKENS", "1"))
            nb = int(os.environ.get("CDT_NUM_BLOCKS", "6"))
            wandb_run_name = f"cdt-{D}d-{nb}blk-K{K}"
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "feature_dim": D,
                "num_blocks": int(os.environ.get("CDT_NUM_BLOCKS", "6")),
                "num_heads": int(os.environ.get("CDT_NUM_HEADS", "12")),
                "num_delta_tokens": int(os.environ.get("CDT_NUM_TOKENS", "1")),
                "lr": lr, "batch_size": batch_size,
                "total_steps": total_steps, "n_params": n_params,
                "backbone": model_name, "seed": seed,
                "grad_clip": grad_clip,
                "launcher": "code_deltatok/train_deltatok.py",
            },
        )
        use_wandb = True
        wandb.run.tags = [f"seed-{seed}", f"K-{os.environ.get('CDT_NUM_TOKENS', '1')}"]
        print(f"W&B run: {wandb.run.url}")
    except Exception as e:
        print(f"W&B init failed ({e}), training without logging")

    # ---- Optimizer + scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def lr_schedule(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # ---- Training loop ----
    print("\n=== CodeDeltaTok Training ===")
    start_time = time.time()
    best_loss = float("inf")

    for step in range(total_steps):
        model.train()

        # Sample batch
        idx = np.random.choice(train_idx, size=batch_size, replace=False)
        prev = torch.from_numpy(before_features[idx.tolist()]).to(device)
        nxt = torch.from_numpy(after_features[idx.tolist()]).to(device)

        # Forward
        out = model(prev, nxt)
        loss = out["loss"]

        # Backward
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val

        # Log
        if use_wandb and step % 10 == 0:
            wandb.log({
                "train/loss": loss_val,
                "train/recon_cos": out["recon_cos"].item(),
                "train/delta_eff_rank": out["delta_eff_rank"].item(),
                "train/raw_cos": out["raw_before_after_cos"].item(),
                "train/grad_norm": grad_norm.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
            }, step=step)

        if step % log_interval == 0:
            # Validation
            model.eval()
            with torch.no_grad():
                v_idx = np.random.choice(val_idx, size=min(batch_size, len(val_idx)), replace=False)
                v_prev = torch.from_numpy(before_features[v_idx.tolist()]).to(device)
                v_nxt = torch.from_numpy(after_features[v_idx.tolist()]).to(device)
                v_out = model(v_prev, v_nxt)

            elapsed = time.time() - start_time
            sps = (step + 1) / elapsed if elapsed > 0 else 0
            print(
                f"step {step:5d}/{total_steps} | "
                f"loss={loss_val:.4f} recon_cos={out['recon_cos'].item():.3f} "
                f"rank={out['delta_eff_rank'].item():.1f} | "
                f"val_loss={v_out['loss'].item():.4f} val_cos={v_out['recon_cos'].item():.3f} "
                f"val_rank={v_out['delta_eff_rank'].item():.1f} | "
                f"best={best_loss:.4f} grad={grad_norm.item():.3f} | {sps:.0f} steps/s"
            )

            if use_wandb:
                wandb.log({
                    "val/loss": v_out["loss"].item(),
                    "val/recon_cos": v_out["recon_cos"].item(),
                    "val/delta_eff_rank": v_out["delta_eff_rank"].item(),
                }, step=step)

        if save_interval > 0 and step > 0 and step % save_interval == 0:
            ckpt_path = os.path.join(output_dir, f"code_deltatok_step{step}.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss_val,
            }, ckpt_path)
            print(f"  checkpoint: {ckpt_path}")

    # ---- Final section ----
    print("\n=== Final Results ===")
    model.eval()

    # Full validation set
    all_recon_cos = []
    all_delta_rank = []
    with torch.no_grad():
        for i in range(0, len(val_idx), batch_size):
            end = min(i + batch_size, len(val_idx))
            v_idx_batch = val_idx[i:end]
            v_prev = torch.from_numpy(before_features[v_idx_batch.tolist()]).to(device)
            v_nxt = torch.from_numpy(after_features[v_idx_batch.tolist()]).to(device)
            v_out = model(v_prev, v_nxt)
            all_recon_cos.append(v_out["recon_cos"].item())
            all_delta_rank.append(v_out["delta_eff_rank"].item())

    mean_cos = np.mean(all_recon_cos)
    mean_rank = np.mean(all_delta_rank)
    print(f"  Val recon_cos: {mean_cos:.4f}")
    print(f"  Val delta_rank: {mean_rank:.1f}")

    # Change retrieval probe: encode all val deltas, do KNN
    print("\n  Change retrieval probe (val set)...")
    with torch.no_grad():
        all_deltas = []
        for i in range(0, len(val_idx), batch_size):
            end = min(i + batch_size, len(val_idx))
            v_idx_batch = val_idx[i:end]
            v_prev = torch.from_numpy(before_features[v_idx_batch.tolist()]).to(device)
            v_nxt = torch.from_numpy(after_features[v_idx_batch.tolist()]).to(device)
            dt = model.encode(v_prev, v_nxt)
            all_deltas.append(dt[:, 0].cpu())  # first delta token
        all_deltas = torch.cat(all_deltas, dim=0)  # [N_val, D]

        # Effective rank of delta tokens
        all_deltas_centered = all_deltas - all_deltas.mean(dim=0)
        svd = torch.linalg.svdvals(all_deltas_centered)
        p = svd / svd.sum()
        full_eff_rank = torch.exp(-torch.sum(p * torch.log(p + 1e-8))).item()

        # Cross-gap (discriminability)
        all_deltas_n = F.normalize(all_deltas, dim=-1)
        Nv = all_deltas_n.shape[0]
        # Sample pairs for large val sets
        if Nv > 2000:
            sample_n = 2000
            sample_idx = np.random.choice(Nv, sample_n, replace=False)
            sim_sub = all_deltas_n[sample_idx] @ all_deltas_n[sample_idx].T
            eye_sub = torch.eye(sample_n, dtype=torch.bool)
            diag_mean = sim_sub[eye_sub].mean().item()
            off_mean = sim_sub[~eye_sub].mean().item()
        else:
            sim_matrix = all_deltas_n @ all_deltas_n.T
            eye_mask = torch.eye(Nv, dtype=torch.bool)
            diag_mean = sim_matrix[eye_mask].mean().item()
            off_mean = sim_matrix[~eye_mask].mean().item()
        cross_gap = diag_mean - off_mean

    print(f"  Delta token eff_rank: {full_eff_rank:.1f}")
    print(f"  Cross-gap: {cross_gap:.4f} (diag={diag_mean:.4f}, off={off_mean:.4f})")

    if use_wandb:
        wandb.run.summary["val/recon_cos"] = mean_cos
        wandb.run.summary["val/delta_eff_rank_full"] = full_eff_rank
        wandb.run.summary["val/cross_gap"] = cross_gap
        wandb.finish()

    # Save final checkpoint
    final_path = os.path.join(output_dir, "code_deltatok_final.pt")
    torch.save({
        "step": total_steps,
        "model_state_dict": model.state_dict(),
        "loss": best_loss,
        "recon_cos": mean_cos,
        "delta_eff_rank": full_eff_rank,
    }, final_path)
    print(f"\nFinal checkpoint: {final_path}")
    print(f"Total time: {time.time() - start_time:.0f}s")

    f.close()


if __name__ == "__main__":
    main()
