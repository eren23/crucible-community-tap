#!/usr/bin/env python3
"""Standalone Code World Model training script for RunPod / GPU execution.

Trains a CodeWorldModel on preprocessed CommitPackFT data (flat HDF5) with
W&B logging. Designed to run on a provisioned pod via Crucible's fleet system.

Env vars:
    WM_HDF5_PATH:      Path to preprocessed HDF5 data (required)
    WM_MODEL_DIM:      Embedding dimension (default: 128)
    WM_NUM_LOOPS:      Looped predictor iterations (default: 6)
    WM_NUM_HEADS:      Attention heads (default: 4)
    WM_VOCAB_SIZE:     AST tokenizer vocab (default: 662)
    WM_MAX_SEQ_LEN:    Max token sequence length (default: 512)
    WM_ENCODER_LOOPS:  Encoder loop count (default: 6)
    ACTION_DIM:        Action vector dim (default: 7, or 15 for rich)
    WM_EMA_DECAY:      EMA target encoder momentum (default: 0.996)
    WM_SIGREG_WEIGHT:  Variance regularization weight (default: 0.1)
    WM_LR:             Learning rate (default: 3e-4)
    WM_BATCH_SIZE:     Batch size (default: 128)
    WM_STEPS:          Training steps (default: 2000)
    WM_EVAL_INTERVAL:  Steps between evals (default: 100)
    WM_SAVE_INTERVAL:  Steps between checkpoints (default: 500)
    WANDB_PROJECT:     W&B project name (default: crucible-code-wm)
    WANDB_RUN_NAME:    W&B run name (default: auto-generated)
    OUTPUT_DIR:        Checkpoint output dir (default: ./checkpoints)
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F


def main():
    # ---- Config from env ------------------------------------------------
    hdf5_path = os.environ.get("WM_HDF5_PATH", "")
    if not hdf5_path:
        print("ERROR: WM_HDF5_PATH is required", file=sys.stderr)
        sys.exit(1)

    model_dim = int(os.environ.get("WM_MODEL_DIM", "128"))
    num_loops = int(os.environ.get("WM_NUM_LOOPS", "6"))
    num_heads = int(os.environ.get("WM_NUM_HEADS", "4"))
    vocab_size = int(os.environ.get("WM_VOCAB_SIZE", "662"))
    max_seq_len = int(os.environ.get("WM_MAX_SEQ_LEN", "512"))
    encoder_loops = int(os.environ.get("WM_ENCODER_LOOPS", "6"))
    action_dim = int(os.environ.get("ACTION_DIM", "7"))
    ema_decay = float(os.environ.get("WM_EMA_DECAY", "0.996"))
    lr = float(os.environ.get("WM_LR", "3e-4"))
    batch_size = int(os.environ.get("WM_BATCH_SIZE", "128"))
    total_steps = int(os.environ.get("WM_STEPS", "2000"))
    warmup_steps = int(os.environ.get("WM_WARMUP_STEPS", "200"))
    eval_interval = int(os.environ.get("WM_EVAL_INTERVAL", "100"))
    save_interval = int(os.environ.get("WM_SAVE_INTERVAL", "500"))
    output_dir = os.environ.get("OUTPUT_DIR", "./checkpoints")
    wandb_project = os.environ.get("WANDB_PROJECT", "crucible-code-wm")
    wandb_run_name = os.environ.get("WANDB_RUN_NAME", "")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    print("=== Code World Model Training ===")
    print(f"  Data:       {hdf5_path}")
    print(f"  Device:     {device}")
    print(f"  Model:      dim={model_dim}, loops={num_loops}, heads={num_heads}")
    print(f"  Vocab:      {vocab_size}, seq_len={max_seq_len}")
    print(f"  Actions:    {action_dim}-dim")
    print(f"  Training:   {total_steps} steps, batch={batch_size}, lr={lr}")
    print(f"  W&B:        {wandb_project}")
    print()

    # ---- Import tap modules ---------------------------------------------
    # __file__ = tap/launchers/code_wm/train_code_wm.py → need 3 levels up
    tap_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, tap_root)

    from architectures.code_wm.code_wm import CodeWorldModel
    from architectures.wm_base.wm_base import wm_base_kwargs_from_env

    # ---- Build model ----------------------------------------------------
    kwargs = wm_base_kwargs_from_env(None)
    model = CodeWorldModel(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        encoder_loops=encoder_loops,
        **kwargs,
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    # ---- Load data ------------------------------------------------------
    import h5py
    import numpy as np

    f = h5py.File(hdf5_path, "r")
    num_edits = f["before_tokens"].shape[0]
    ctx_window = f["before_tokens"].shape[1]
    data_action_dim = f["edit_actions"].shape[1]
    print(f"Data: {num_edits:,} transitions, ctx={ctx_window}, action_dim={data_action_dim}")

    if data_action_dim != action_dim:
        print(f"NOTE: Using data action_dim={data_action_dim} (ACTION_DIM env was {action_dim})")
        action_dim = data_action_dim

    # ---- W&B init -------------------------------------------------------
    use_wandb = False
    try:
        import wandb
        if not wandb_run_name:
            wandb_run_name = f"code-wm-{model_dim}d-{num_loops}L-{action_dim}act"
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "model_dim": model_dim, "num_loops": num_loops,
                "num_heads": num_heads, "vocab_size": vocab_size,
                "max_seq_len": max_seq_len, "encoder_loops": encoder_loops,
                "action_dim": action_dim, "ema_decay": ema_decay,
                "lr": lr, "batch_size": batch_size, "total_steps": total_steps,
                "n_params": n_params, "data_file": os.path.basename(hdf5_path),
                "num_edits": num_edits,
            },
        )
        use_wandb = True
        print(f"W&B run: {wandb.run.url}")
    except Exception as e:
        print(f"W&B init failed ({e}), training without logging")

    # ---- Optimizer + scheduler ------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # Linear warmup then cosine decay
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps - warmup_steps, 1),
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps],
    )

    # ---- Batch loader ---------------------------------------------------
    def get_batch():
        indices = np.sort(np.random.choice(num_edits, size=batch_size, replace=False))
        before = torch.from_numpy(
            f["before_tokens"][indices.tolist()].astype(np.int64)
        ).to(device)
        actions = torch.from_numpy(
            f["edit_actions"][indices.tolist()].astype(np.float32)
        ).to(device)
        after = torch.from_numpy(
            f["after_tokens"][indices.tolist()].astype(np.int64)
        ).to(device)
        states = torch.stack([before, after], dim=1)
        act = actions.unsqueeze(1)
        return {"states": states, "actions": act}

    # ---- Training loop --------------------------------------------------
    model.train()
    start_time = time.time()
    best_loss = float("inf")

    for step in range(total_steps):
        batch = get_batch()
        optimizer.zero_grad()
        out = model.forward(**batch)
        out["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        optimizer.step()
        scheduler.step()

        loss_val = out["loss"].item()
        pred_loss = out["pred_loss"].item()
        reg_loss = out.get("reg_loss", torch.tensor(0.0)).item()

        if loss_val < best_loss:
            best_loss = loss_val

        if use_wandb and step % 10 == 0:
            wandb.log({
                "train/loss": loss_val, "train/pred_loss": pred_loss,
                "train/reg_loss": reg_loss, "train/grad_norm": grad_norm,
                "train/lr": optimizer.param_groups[0]["lr"],
            }, step=step)

        if step % eval_interval == 0:
            elapsed = time.time() - start_time
            sps = (step + 1) / elapsed if elapsed > 0 else 0
            print(
                f"step {step:5d}/{total_steps} | "
                f"loss={loss_val:.4f} pred={pred_loss:.4f} reg_loss={reg_loss:.4f} | "
                f"best={best_loss:.4f} | grad={grad_norm:.2f} | {sps:.1f} steps/s"
            )

            model.eval_mode = True
            with torch.no_grad():
                eb = get_batch()
                eo = model.forward(**eb)
                zp = eo["pred_embeddings"].reshape(-1, model_dim)
                zt = eo["target_embeddings"].reshape(-1, model_dim)
                cos_sim = F.cosine_similarity(zp, zt, dim=-1).mean().item()
                if use_wandb:
                    wandb.log({"val/cosine_sim": cos_sim, "val/pred_loss": eo["pred_loss"].item()}, step=step)
                print(f"  val cosine_sim={cos_sim:.4f}")
            model.train()

        if step > 0 and step % save_interval == 0:
            ckpt_path = os.path.join(output_dir, f"code_wm_step{step}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step, "loss": loss_val, "best_loss": best_loss,
                "config": {
                    "model_dim": model_dim, "num_loops": num_loops,
                    "num_heads": num_heads, "vocab_size": vocab_size,
                    "max_seq_len": max_seq_len, "encoder_loops": encoder_loops,
                    "action_dim": action_dim, "ema_decay": ema_decay,
                },
            }, ckpt_path)
            print(f"  checkpoint: {ckpt_path}")

    # ---- Final save -----------------------------------------------------
    elapsed = time.time() - start_time
    final_path = os.path.join(output_dir, "code_wm_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "step": total_steps, "loss": loss_val, "best_loss": best_loss,
        "config": {
            "model_dim": model_dim, "num_loops": num_loops,
            "num_heads": num_heads, "vocab_size": vocab_size,
            "max_seq_len": max_seq_len, "encoder_loops": encoder_loops,
            "action_dim": action_dim, "ema_decay": ema_decay,
        },
    }, final_path)

    print(f"\n=== Training Complete ===")
    print(f"  Steps: {total_steps}, Final loss: {loss_val:.4f}, Best: {best_loss:.4f}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  Checkpoint: {final_path}")

    if use_wandb:
        wandb.finish()
    f.close()


if __name__ == "__main__":
    main()
