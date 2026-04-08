#!/usr/bin/env python3
"""Standalone Code World Model training script for RunPod / GPU execution.

Supports two data modes:
  - **Single-step** (default): classic (before, action, after) pairs from flat HDF5
  - **Trajectory**: multi-step windows from trajectory-structured HDF5
    (auto-detected via /metadata.has_trajectories flag)

Phase 2 features for fixing the 700-step training ceiling:
  - Bounded residual predictor: z_next = z + tanh(f(z,a)) * scale
  - Direction loss scheduling: kill delta-dir loss after N steps
  - Aggressive cosine LR decay with early peak
  - Dropout scheduling: ramp from 0 to target after warmup

Env vars (new, in addition to existing):
    WM_WINDOW_LEN:       Trajectory window length (default: 4, auto 1 if no trajectories)
    WM_BOUNDED_RESIDUAL: "1" to use bounded residual predictor (default: "0")
    WM_RESIDUAL_SCALE:   Max residual norm (default: 1.0)
    WM_DIR_LOSS_UNTIL:   Step to kill direction loss (default: 0 = never kill)
    WM_DROPOUT_RAMP:     "1" to ramp dropout 0 -> target over warmup (default: "0")
    WM_LR_PEAK_STEP:     Step of LR peak if > 0, overrides warmup (default: 0)
    WM_EMA_START:        Initial EMA decay, anneals to WM_EMA_DECAY (default: same)
    WM_ROLLOUT_STEPS:    Multi-step rollout loss steps (default: 0)
    WM_LAMBDA_ROLLOUT:   Rollout loss weight (default: 0.0)
    WM_LAMBDA_PATH_CONSISTENCY: Path consistency weight (default: 0.0)
"""
from __future__ import annotations

import math
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

    # Phase 2 config
    window_len = int(os.environ.get("WM_WINDOW_LEN", "4"))
    bounded_residual = os.environ.get("WM_BOUNDED_RESIDUAL", "0") == "1"
    residual_scale = float(os.environ.get("WM_RESIDUAL_SCALE", "1.0"))
    dir_loss_until = int(os.environ.get("WM_DIR_LOSS_UNTIL", "0"))
    dropout_ramp = os.environ.get("WM_DROPOUT_RAMP", "0") == "1"
    lr_peak_step = int(os.environ.get("WM_LR_PEAK_STEP", "0"))
    ema_start = float(os.environ.get("WM_EMA_START", str(ema_decay)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    # ---- Import tap modules ---------------------------------------------
    tap_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, tap_root)

    from architectures.code_wm.code_wm import CodeWorldModel
    from architectures.wm_base.wm_base import wm_base_kwargs_from_env

    # ---- Load data + detect trajectory mode -----------------------------
    import h5py
    import numpy as np

    # Auto-download from HuggingFace if local path doesn't exist
    if not os.path.exists(hdf5_path) and "/" not in hdf5_path:
        # Treat as a HF Hub path: repo_id/filename
        hf_repo = os.environ.get("WM_HF_REPO", "eren23/codewm-data")
        hf_path = os.environ.get("WM_HF_PATH", f"trajectories/{hdf5_path}")
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

    f = h5py.File(hdf5_path, "r")
    num_edits = f["before_tokens"].shape[0]
    ctx_window = f["before_tokens"].shape[1]
    data_action_dim = f["edit_actions"].shape[1]
    has_trajectories = bool(f["metadata"].attrs.get("has_trajectories", False))

    # Always sync action_dim from data into env (wm_base_kwargs_from_env reads it)
    action_dim = data_action_dim
    os.environ["ACTION_DIM"] = str(action_dim)

    # Auto-adjust window length
    if has_trajectories:
        traj_lengths = f["trajectory/traj_lengths"][:]
        num_trajectories = len(traj_lengths)
        eligible = int((traj_lengths >= window_len).sum())
        if eligible == 0:
            max_avail = int(traj_lengths.max())
            print(f"WARNING: No trajectories with length >= {window_len}, using {max_avail}")
            window_len = max_avail
            eligible = int((traj_lengths >= window_len).sum())
        mode = "trajectory"
    else:
        window_len = 1
        num_trajectories = 0
        eligible = 0
        mode = "single-step"

    print("=== Code World Model Training ===")
    print(f"  Data:       {hdf5_path}")
    print(f"  Mode:       {mode} (window={window_len})")
    if has_trajectories:
        print(f"  Trajs:      {num_trajectories} total, {eligible} eligible (>={window_len} transitions)")
    print(f"  Transitions: {num_edits:,}, ctx={ctx_window}, action_dim={action_dim}")
    print(f"  Device:     {device}")
    print(f"  Model:      dim={model_dim}, loops={num_loops}, heads={num_heads}")
    if bounded_residual:
        print(f"  Predictor:  BOUNDED RESIDUAL (scale={residual_scale})")
    if dir_loss_until > 0:
        print(f"  Dir loss:   killed after step {dir_loss_until}")
    if dropout_ramp:
        print(f"  Dropout:    ramped 0 -> target over {warmup_steps} steps")
    if ema_start != ema_decay:
        print(f"  EMA:        {ema_start} -> {ema_decay} annealed")
    print(f"  Training:   {total_steps} steps, batch={batch_size}, lr={lr}")
    print(f"  W&B:        {wandb_project}")
    print()

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

    # ---- W&B init -------------------------------------------------------
    use_wandb = False
    try:
        import wandb
        if not wandb_run_name:
            tag = f"traj{window_len}" if has_trajectories else "single"
            wandb_run_name = f"code-wm-{model_dim}d-{num_loops}L-{tag}"
            if bounded_residual:
                wandb_run_name += "-bounded"
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
                "num_edits": num_edits, "mode": mode, "window_len": window_len,
                "bounded_residual": bounded_residual, "residual_scale": residual_scale,
                "dir_loss_until": dir_loss_until, "dropout_ramp": dropout_ramp,
                "ema_start": ema_start, "num_trajectories": num_trajectories,
            },
        )
        use_wandb = True
        print(f"W&B run: {wandb.run.url}")
    except Exception as e:
        print(f"W&B init failed ({e}), training without logging")

    # ---- Optimizer + scheduler ------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    actual_warmup = lr_peak_step if lr_peak_step > 0 else warmup_steps
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=actual_warmup,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps - actual_warmup, 1),
        eta_min=lr * 0.01,  # decay to 1% of peak (more aggressive than default 0)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[actual_warmup],
    )

    # ---- Train/Val split ------------------------------------------------
    val_frac = float(os.environ.get("WM_VAL_FRAC", "0.1"))
    seed = int(os.environ.get("WM_SEED", "42"))
    np.random.seed(seed)
    torch.manual_seed(seed)

    if has_trajectories:
        # Split by trajectory, not by transition
        all_traj_indices = np.random.permutation(num_trajectories)
        n_val_traj = max(int(num_trajectories * val_frac), 1)
        val_traj_set = set(all_traj_indices[:n_val_traj].tolist())
        train_traj_set = set(all_traj_indices[n_val_traj:].tolist())

        traj_offsets = f["trajectory/traj_offsets"][:]

        # Build eligible trajectory pools
        train_eligible = np.array([
            t for t in range(num_trajectories)
            if t in train_traj_set and traj_lengths[t] >= window_len
        ])
        val_eligible = np.array([
            t for t in range(num_trajectories)
            if t in val_traj_set and traj_lengths[t] >= window_len
        ])
        # Fallback: if val has no eligible trajs, use some from train
        if len(val_eligible) == 0:
            val_eligible = train_eligible[:max(len(train_eligible) // 10, 1)]

        print(f"Split: {len(train_eligible)} train trajs, {len(val_eligible)} val trajs (seed={seed})")
    else:
        all_indices = np.random.permutation(num_edits)
        n_val = max(int(num_edits * val_frac), batch_size)
        val_indices = np.sort(all_indices[:n_val])
        train_indices = np.sort(all_indices[n_val:])
        print(f"Split: {len(train_indices):,} train, {len(val_indices):,} val (seed={seed})")

    # ---- Batch loaders --------------------------------------------------
    before_ds = f["before_tokens"]
    after_ds = f["after_tokens"]
    action_ds = f["edit_actions"]

    def get_single_step_batch(pool_indices):
        """Classic single-step batch: states=[B,2,S], actions=[B,1,A]."""
        idx = np.sort(np.random.choice(pool_indices, size=batch_size, replace=False))
        before = torch.from_numpy(before_ds[idx.tolist()].astype(np.int64)).to(device)
        actions = torch.from_numpy(action_ds[idx.tolist()].astype(np.float32)).to(device)
        after = torch.from_numpy(after_ds[idx.tolist()].astype(np.int64)).to(device)
        return {"states": torch.stack([before, after], dim=1), "actions": actions.unsqueeze(1)}

    def get_trajectory_batch(eligible_pool):
        """Multi-step trajectory batch: states=[B,W+1,S], actions=[B,W,A]."""
        W = window_len
        traj_ids = eligible_pool[np.random.randint(0, len(eligible_pool), size=batch_size)]

        all_states = np.zeros((batch_size, W + 1, ctx_window), dtype=np.int64)
        all_actions = np.zeros((batch_size, W, data_action_dim), dtype=np.float32)

        for b, tid in enumerate(traj_ids):
            offset = int(traj_offsets[tid])
            length = int(traj_lengths[tid])
            start = np.random.randint(0, length - W + 1)
            g_start = offset + start
            g_end = g_start + W

            idx_range = list(range(g_start, g_end))
            befores = before_ds[idx_range].astype(np.int64)
            actions = action_ds[idx_range].astype(np.float32)
            last_after = after_ds[g_end - 1].astype(np.int64)

            all_states[b, :W] = befores
            all_states[b, W] = last_after
            all_actions[b] = actions

        return {
            "states": torch.from_numpy(all_states).to(device),
            "actions": torch.from_numpy(all_actions).to(device),
        }

    def get_batch(split="train"):
        if has_trajectories:
            pool = train_eligible if split == "train" else val_eligible
            return get_trajectory_batch(pool)
        else:
            pool = train_indices if split == "train" else val_indices
            return get_single_step_batch(pool)

    # ---- Phase 2: Bounded residual wrapper ------------------------------
    original_predict = None
    if bounded_residual:
        original_predict = model.predictor.forward

        def bounded_predict(z_state, z_action):
            raw = original_predict(z_state, z_action)
            residual = raw - z_state
            bounded = torch.tanh(residual / residual_scale) * residual_scale
            return z_state + bounded

        model.predictor.forward = bounded_predict
        print(f"Installed bounded residual predictor (scale={residual_scale})")

    # ---- Training loop --------------------------------------------------
    model.train()
    start_time = time.time()
    best_loss = float("inf")
    best_val_dcos = -1.0
    patience = int(os.environ.get("WM_PATIENCE", "0"))
    patience_counter = 0

    # Save original lambda_dir for scheduling
    original_lambda_dir = model.lambda_dir
    original_dropout = float(os.environ.get("WM_DROPOUT", "0.1"))

    for step in range(total_steps):
        # --- Phase 2: Dynamic scheduling ---
        # Direction loss scheduling: kill after dir_loss_until
        if dir_loss_until > 0:
            model.lambda_dir = original_lambda_dir if step < dir_loss_until else 0.0

        # Dropout ramping: 0 -> target over warmup
        if dropout_ramp and step < warmup_steps and original_dropout > 0:
            ramp_frac = step / max(warmup_steps, 1)
            current_dropout = original_dropout * ramp_frac
            for block in model.predictor.blocks:
                if hasattr(block, 'attn'):
                    block.attn.dropout = current_dropout

        # EMA decay annealing: ema_start -> ema_decay
        if ema_start != ema_decay:
            progress = min(step / max(total_steps, 1), 1.0)
            model.ema_decay = ema_start + (ema_decay - ema_start) * progress

        # --- Forward + backward ---
        batch = get_batch()
        optimizer.zero_grad()
        out = model.forward(**batch)
        out["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        optimizer.step()
        scheduler.step()

        loss_val = out["loss"].item()
        loss_pred = out.get("loss_pred", out.get("pred_loss", torch.tensor(0.0))).item()
        loss_dir = out.get("loss_dir", torch.tensor(0.0)).item()
        loss_mag = out.get("loss_mag", torch.tensor(0.0)).item()
        loss_cov = out.get("loss_cov", torch.tensor(0.0)).item()
        delta_cos = out.get("delta_cos_sim", torch.tensor(0.0)).item()
        delta_ratio = out.get("delta_norm_ratio", torch.tensor(0.0)).item()
        loss_rollout = out.get("loss_rollout", torch.tensor(0.0)).item()
        loss_path = out.get("loss_path_consistency", torch.tensor(0.0)).item()

        if loss_val < best_loss:
            best_loss = loss_val

        if use_wandb and step % 10 == 0:
            log_dict = {
                "train/loss": loss_val, "train/loss_pred": loss_pred,
                "train/loss_dir": loss_dir, "train/loss_mag": loss_mag,
                "train/loss_cov": loss_cov,
                "train/delta_cos_sim": delta_cos,
                "train/delta_norm_ratio": delta_ratio,
                "train/grad_norm": grad_norm,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/ema_decay": getattr(model, 'ema_decay', ema_decay),
            }
            if loss_rollout > 0:
                log_dict["train/loss_rollout"] = loss_rollout
            if loss_path > 0:
                log_dict["train/loss_path_consistency"] = loss_path
            if dir_loss_until > 0:
                log_dict["train/lambda_dir"] = model.lambda_dir
            wandb.log(log_dict, step=step)

        if step % eval_interval == 0:
            elapsed = time.time() - start_time
            sps = (step + 1) / elapsed if elapsed > 0 else 0
            extra = ""
            if loss_rollout > 0:
                extra += f" roll={loss_rollout:.4f}"
            if loss_path > 0:
                extra += f" path={loss_path:.4f}"
            print(
                f"step {step:5d}/{total_steps} | "
                f"loss={loss_val:.4f} pred={loss_pred:.4f} dir={loss_dir:.4f}{extra} | "
                f"dcos={delta_cos:.3f} dratio={delta_ratio:.2f} | "
                f"best={best_loss:.4f} | grad={grad_norm:.2f} | {sps:.1f} steps/s"
            )

            # --- Validation ---
            model.train(False)
            with torch.no_grad():
                eb = get_batch(split="val")
                eo = model.forward(**eb)

                # For multi-step: evaluate step-1 predictions specifically
                zp = eo["pred_embeddings"][:, 0].reshape(-1, model_dim)  # first step
                zt = eo["target_embeddings"][:, 0].reshape(-1, model_dim)
                z_curr = model.state_encoder(eb["states"][:, 0]).reshape(-1, model_dim)

                cos_sim = F.cosine_similarity(zp, zt, dim=-1).mean().item()
                cos_copy = F.cosine_similarity(z_curr, zt, dim=-1).mean().item()
                delta_true = zt - z_curr
                delta_pred = zp - z_curr
                dt_n = F.normalize(delta_true, dim=-1)
                dp_n = F.normalize(delta_pred, dim=-1)
                val_delta_cos = (dt_n * dp_n).sum(dim=-1).mean().item()
                lift = cos_sim - cos_copy

                # Multi-step metrics (if window > 1)
                val_multistep = {}
                n_pred_steps = eo["pred_embeddings"].shape[1]
                if n_pred_steps > 1:
                    for k in range(min(n_pred_steps, 4)):
                        zp_k = eo["pred_embeddings"][:, k].reshape(-1, model_dim)
                        zt_k = eo["target_embeddings"][:, k].reshape(-1, model_dim)
                        z_prev = model.state_encoder(eb["states"][:, k]).reshape(-1, model_dim)
                        dt_k = zt_k - z_prev
                        dp_k = zp_k - z_prev
                        dcos_k = (F.normalize(dt_k, dim=-1) * F.normalize(dp_k, dim=-1)).sum(-1).mean().item()
                        val_multistep[f"val/delta_cos_step{k+1}"] = dcos_k

                if use_wandb:
                    val_log = {
                        "val/cosine_sim": cos_sim,
                        "val/cos_copy_baseline": cos_copy,
                        "val/lift_over_copy": lift,
                        "val/delta_cos_sim": val_delta_cos,
                        "val/pred_loss": eo.get("loss_pred", eo.get("pred_loss", torch.tensor(0.0))).item(),
                    }
                    val_log.update(val_multistep)
                    wandb.log(val_log, step=step)

                step_str = f"  val dcos={val_delta_cos:.4f} cos={cos_sim:.4f} (copy={cos_copy:.4f}, lift={lift:+.4f})"
                if val_multistep:
                    dcos_vals = [f"s{k+1}={v:.3f}" for k, v in enumerate(val_multistep.values())]
                    step_str += f" | multi-step: {' '.join(dcos_vals)}"
                print(step_str)

                # Early stopping on val delta_cos
                if val_delta_cos > best_val_dcos:
                    best_val_dcos = val_delta_cos
                    patience_counter = 0
                    best_path = os.path.join(output_dir, "code_wm_best.pt")
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "step": step, "val_delta_cos": val_delta_cos,
                        "config": {
                            "model_dim": model_dim, "num_loops": num_loops,
                            "num_heads": num_heads, "vocab_size": vocab_size,
                            "max_seq_len": max_seq_len, "encoder_loops": encoder_loops,
                            "action_dim": action_dim, "ema_decay": ema_decay,
                            "mode": mode, "window_len": window_len,
                            "bounded_residual": bounded_residual,
                        },
                    }, best_path)
                else:
                    patience_counter += 1

                if patience > 0 and patience_counter >= patience:
                    print(f"\n  Early stopping: val delta_cos didn't improve for "
                          f"{patience} evals (best={best_val_dcos:.4f})")
                    break
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
                    "mode": mode, "window_len": window_len,
                    "bounded_residual": bounded_residual,
                },
            }, ckpt_path)
            print(f"  checkpoint: {ckpt_path}")

    # ---- Final save -----------------------------------------------------
    elapsed = time.time() - start_time
    final_path = os.path.join(output_dir, "code_wm_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "step": total_steps, "loss": loss_val, "best_loss": best_loss,
        "best_val_dcos": best_val_dcos,
        "config": {
            "model_dim": model_dim, "num_loops": num_loops,
            "num_heads": num_heads, "vocab_size": vocab_size,
            "max_seq_len": max_seq_len, "encoder_loops": encoder_loops,
            "action_dim": action_dim, "ema_decay": ema_decay,
            "mode": mode, "window_len": window_len,
            "bounded_residual": bounded_residual,
        },
    }, final_path)

    print(f"\n=== Training Complete ===")
    print(f"  Steps: {total_steps}, Final loss: {loss_val:.4f}, Best: {best_loss:.4f}")
    print(f"  Best val delta_cos: {best_val_dcos:.4f}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  Checkpoint: {final_path}")

    if use_wandb:
        wandb.finish()
    f.close()


def _apply_cli_overrides():
    """Parse --set KEY=VALUE args (Crucible scheduler compat) + ignore unknown flags."""
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    # Accept and ignore Crucible-native flags so this script can be used as run_script
    parser.add_argument("--backend", default=None)
    parser.add_argument("--preset", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--timeout", default=None)
    parser.add_argument("--tag", action="append", default=[])
    args, _ = parser.parse_known_args()
    for item in args.overrides:
        if "=" in item:
            key, value = item.split("=", 1)
            os.environ[key] = value


if __name__ == "__main__":
    _apply_cli_overrides()
    main()
