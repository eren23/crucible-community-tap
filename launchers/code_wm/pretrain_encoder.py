#!/usr/bin/env python3
"""Pre-train CodeWM encoder with supervised objectives (Phase 12, Direction B).

Trains the state encoder to predict edit properties from the before-state
encoding. This forces the encoder to learn representations that distinguish
different code contexts — preventing the rank collapse seen with JEPA.

Supported objectives (WM_PRETRAIN_OBJECTIVE):
  - "action_predict": Predict the 7-dim action vector (binary dims + location)
  - "mlm": Masked language modeling on AST tokens

After pre-training, the encoder is saved and can be loaded via
WM_PRETRAINED_ENCODER for frozen-encoder dynamics training.

Usage:
    WM_PRETRAIN_OBJECTIVE=action_predict WM_STEPS=5000 \
    WANDB_RUN_NAME=pretrain-action-5k \
    python launchers/code_wm/pretrain_encoder.py
"""
from __future__ import annotations

import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_float32_matmul_precision("high")


def main():
    # ---- Config from env ------------------------------------------------
    hdf5_path = os.environ.get("WM_HDF5_PATH", "data/commitpackft_python_ast.h5")
    model_dim = int(os.environ.get("WM_MODEL_DIM", "128"))
    num_loops = int(os.environ.get("WM_NUM_LOOPS", "6"))
    num_heads = int(os.environ.get("WM_NUM_HEADS", "4"))
    encoder_loops = int(os.environ.get("WM_ENCODER_LOOPS", "6"))
    vocab_size = int(os.environ.get("WM_VOCAB_SIZE", "662"))
    max_seq_len = int(os.environ.get("WM_MAX_SEQ_LEN", "512"))
    lr = float(os.environ.get("WM_LR", "3e-4"))
    batch_size = int(os.environ.get("WM_BATCH_SIZE", "128"))
    total_steps = int(os.environ.get("WM_STEPS", "5000"))
    seed = int(os.environ.get("WM_SEED", "42"))
    objective = os.environ.get("WM_PRETRAIN_OBJECTIVE", "action_predict")
    mlp_ratio = float(os.environ.get("WM_MLP_RATIO", "4.0"))
    dropout = float(os.environ.get("WM_DROPOUT", "0.1"))
    output_dir = os.environ.get("OUTPUT_DIR", "./checkpoints")
    wandb_project = os.environ.get("WANDB_PROJECT", "crucible-code-wm")
    wandb_run_name = os.environ.get("WANDB_RUN_NAME", f"pretrain-{objective}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    # ---- Import tap modules ---------------------------------------------
    tap_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, tap_root)
    from architectures.code_wm.code_wm import CodeStateEncoder

    # ---- Load data ------------------------------------------------------
    import h5py
    import numpy as np

    if not os.path.exists(hdf5_path) and "/" not in hdf5_path:
        hf_repo = os.environ.get("WM_HF_REPO", "eren23/codewm-data")
        hf_path = os.environ.get("WM_HF_PATH", f"trajectories/{hdf5_path}")
        print(f"Downloading from HF: {hf_repo}/{hf_path}")
        from huggingface_hub import hf_hub_download
        hdf5_path = hf_hub_download(repo_id=hf_repo, filename=hf_path, repo_type="dataset")

    f = h5py.File(hdf5_path, "r")
    num_edits = f["before_tokens"].shape[0]
    ctx_window = f["before_tokens"].shape[1]
    data_vocab = int(f["metadata"].attrs.get("vocab_size", 662))
    if vocab_size < data_vocab:
        vocab_size = data_vocab
    action_dim = f["edit_actions"].shape[1]

    print(f"=== Encoder Pre-Training ({objective}) ===")
    print(f"  Data: {hdf5_path} ({num_edits} edits, ctx={ctx_window}, actions={action_dim})")
    print(f"  Encoder: dim={model_dim}, loops={encoder_loops}, heads={num_heads}")
    print(f"  Training: {total_steps} steps, batch={batch_size}, lr={lr}")
    print(f"  Device: {device}")

    # ---- Train/val split ------------------------------------------------
    np.random.seed(seed)
    indices = np.random.permutation(num_edits)
    n_val = max(int(num_edits * 0.1), batch_size)
    val_idx = np.sort(indices[:n_val])
    train_idx = np.sort(indices[n_val:])

    # ---- Build encoder + head -------------------------------------------
    encoder = CodeStateEncoder(
        vocab_size=vocab_size,
        model_dim=model_dim,
        max_seq_len=max_seq_len,
        encoder_loops=encoder_loops,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
    ).to(device)

    if objective == "action_predict":
        # Predict the 7-dim action vector: 6 binary dims + 1 continuous
        head = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, action_dim),
        ).to(device)
    elif objective == "mlm":
        # Masked language modeling: predict masked tokens
        head = nn.Linear(model_dim, vocab_size).to(device)
    else:
        raise ValueError(f"Unknown objective: {objective}")

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_head = sum(p.numel() for p in head.parameters())
    print(f"  Encoder: {n_enc:,} params, Head: {n_head:,} params")

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=lr, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    # ---- W&B init -------------------------------------------------------
    use_wandb = False
    try:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name, config={
            "objective": objective, "model_dim": model_dim, "encoder_loops": encoder_loops,
            "num_heads": num_heads, "lr": lr, "batch_size": batch_size, "steps": total_steps,
            "vocab_size": vocab_size, "n_encoder_params": n_enc, "n_head_params": n_head,
        })
        use_wandb = True
        print(f"  W&B: {wandb_project}/{wandb_run_name}")
    except Exception as e:
        print(f"  W&B init failed ({e}), training without logging")

    # ---- Training loop --------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float("inf")
    t0 = time.time()
    mask_prob = 0.15  # for MLM
    pad_id = 612

    for step in range(total_steps + 1):
        encoder.train()
        head.train()

        # Sample batch
        idx = np.random.choice(train_idx, batch_size, replace=False)
        before = torch.from_numpy(f["before_tokens"][idx].astype(np.int64)).to(device)
        actions = torch.from_numpy(f["edit_actions"][idx].astype(np.float32)).to(device)

        if objective == "action_predict":
            z = encoder(before)  # [B, D]
            pred = head(z)  # [B, 7]
            # Binary cross-entropy for dims 0-5, MSE for dim 6
            loss_binary = F.binary_cross_entropy_with_logits(pred[:, :6], actions[:, :6])
            loss_location = F.mse_loss(torch.sigmoid(pred[:, 6]), actions[:, 6])
            loss = loss_binary + loss_location

        elif objective == "mlm":
            # Mask random tokens (not PAD)
            mask = (torch.rand_like(before.float()) < mask_prob) & (before != pad_id)
            masked_input = before.clone()
            masked_input[mask] = pad_id  # replace with PAD as mask token
            z_seq = encoder.embedding(masked_input)
            cls = encoder.cls_token.expand(before.shape[0], -1, -1)
            z_seq = torch.cat([cls, z_seq], dim=1)
            z_seq = encoder.pos_enc(z_seq)
            for _ in range(encoder.encoder_loops):
                z_seq = encoder.block(z_seq)
            # Predict masked tokens from sequence output (skip CLS at pos 0)
            logits = head(z_seq[:, 1:])  # [B, S, V]
            loss = F.cross_entropy(logits[mask], before[mask])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(head.parameters()), 1.0
        )
        optimizer.step()
        scheduler.step()

        # ---- Logging + validation ----------------------------------------
        if step % 100 == 0:
            encoder.train(False)
            with torch.no_grad():
                # Val batch
                vidx = np.random.choice(val_idx, min(batch_size, len(val_idx)), replace=False)
                vbefore = torch.from_numpy(f["before_tokens"][vidx].astype(np.int64)).to(device)
                vactions = torch.from_numpy(f["edit_actions"][vidx].astype(np.float32)).to(device)

                vz = encoder(vbefore)

                if objective == "action_predict":
                    vpred = head(vz)
                    vl_bin = F.binary_cross_entropy_with_logits(vpred[:, :6], vactions[:, :6])
                    vl_loc = F.mse_loss(torch.sigmoid(vpred[:, 6]), vactions[:, 6])
                    val_loss = (vl_bin + vl_loc).item()
                    # Accuracy on binary dims
                    acc = ((vpred[:, :6] > 0) == (vactions[:, :6] > 0.5)).float().mean().item()
                elif objective == "mlm":
                    val_loss = loss.item()  # approximate
                    acc = 0.0

                # Effective rank of encoder output
                vz_n = F.normalize(vz, dim=-1)
                s = torch.linalg.svdvals(vz_n.float())
                s = s / (s.sum() + 1e-12)
                eff_rank = float(torch.exp(-(s * (s + 1e-12).log()).sum()).item())

            elapsed = time.time() - t0
            sps = (step + 1) / elapsed if elapsed > 0 else 0
            print(f"step {step:>5d}/{total_steps} | loss={loss.item():.4f} val_loss={val_loss:.4f} "
                  f"acc={acc:.3f} eff_rank={eff_rank:.1f}/128 | {sps:.1f} steps/s")

            if use_wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "val/loss": val_loss,
                    "val/accuracy": acc,
                    "val/eff_rank_encoder": eff_rank,
                    "lr": scheduler.get_last_lr()[0],
                }, step=step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "model_state_dict": encoder.state_dict(),
                    "step": step,
                    "val_loss": val_loss,
                    "eff_rank": eff_rank,
                    "objective": objective,
                    "config": {
                        "model_dim": model_dim, "num_loops": num_loops,
                        "num_heads": num_heads, "vocab_size": vocab_size,
                        "max_seq_len": max_seq_len, "encoder_loops": encoder_loops,
                    },
                }, os.path.join(output_dir, "encoder_pretrained_best.pt"))

        # Periodic checkpoint
        if step > 0 and step % 500 == 0:
            torch.save({
                "model_state_dict": encoder.state_dict(),
                "step": step,
                "config": {
                    "model_dim": model_dim, "num_loops": num_loops,
                    "num_heads": num_heads, "vocab_size": vocab_size,
                    "max_seq_len": max_seq_len, "encoder_loops": encoder_loops,
                },
            }, os.path.join(output_dir, f"encoder_pretrained_step{step}.pt"))

    # ---- Final save -----------------------------------------------------
    final_path = os.path.join(output_dir, "encoder_pretrained_final.pt")
    torch.save({
        "model_state_dict": encoder.state_dict(),
        "step": total_steps,
        "val_loss": best_val_loss,
        "objective": objective,
        "config": {
            "model_dim": model_dim, "num_loops": num_loops,
            "num_heads": num_heads, "vocab_size": vocab_size,
            "max_seq_len": max_seq_len, "encoder_loops": encoder_loops,
        },
    }, final_path)
    print(f"\nFinal encoder saved: {final_path}")
    print(f"Best encoder saved: {os.path.join(output_dir, 'encoder_pretrained_best.pt')}")
    print(f"Best val_loss: {best_val_loss:.4f}")

    f.close()
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
