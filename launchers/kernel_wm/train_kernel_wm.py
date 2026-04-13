#!/usr/bin/env python3
"""KernelWM training script — trains CodeWM on GPU kernel migration pairs.

Uses the same CodeWorldModel architecture with adjusted vocab (400) and
action_dim (12). Adds example table logging: during validation, decodes
predicted kernel configs back to readable form and saves comparison tables.

Env vars (in addition to standard WM_* vars):
    KWM_SAVE_EXAMPLES:   Number of example tables to save per eval (default: 10)
    KWM_EXAMPLE_DIR:     Directory for example tables (default: ./examples)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F


# ── Token decoding (inverse of pairs_to_hdf5.py encoding) ──

PAD, BOS, EOS, SEP = 0, 1, 2, 3

FIELD_NAMES = {
    4: "arch", 5: "kernel_type", 6: "element_a", 7: "element_b",
    8: "element_c", 9: "layout_a", 10: "layout_b", 11: "tile_m",
    12: "tile_n", 13: "tile_k", 14: "cluster_m", 15: "cluster_n",
    16: "cluster_k", 17: "stages", 18: "mma_class", 19: "mainloop",
}

CAT_VALUES = {
    20: "sm80", 21: "sm90", 22: "sm100", 23: "unknown",
    24: "gemm", 25: "conv", 26: "reduce",
    27: "f16", 28: "bf16", 29: "f32", 30: "f64",
    31: "tf32", 32: "f8e4m3", 33: "f8e5m2",
    34: "i8", 35: "u8",
    36: "row", 37: "col",
    38: "hmma", 39: "wgmma", 40: "tcgen05", 41: "simt",
    42: "cp_async", 43: "tma", 44: "tma_warp_specialized",
    46: "default", 47: "visitor", 48: "evt",
}

NUM_OFFSET = 100
NUM_FIELDS = {"tile_m", "tile_n", "tile_k", "cluster_m", "cluster_n", "cluster_k", "stages"}


def decode_tokens(tokens) -> dict:
    """Decode a token sequence back to a kernel config dict."""
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.cpu().numpy().tolist()

    config = {}
    i = 0
    if tokens[0] == BOS:
        i = 1

    current_field = None
    while i < len(tokens):
        tok = tokens[i]
        if tok == EOS or tok == PAD:
            break
        if tok == SEP:
            current_field = None
            i += 1
            continue
        if tok in FIELD_NAMES:
            current_field = FIELD_NAMES[tok]
            i += 1
            continue
        if current_field:
            if current_field in NUM_FIELDS:
                raw_val = tok - NUM_OFFSET if tok >= NUM_OFFSET else tok
                if current_field in ("tile_m", "tile_n", "tile_k"):
                    config[current_field] = 2 ** raw_val if raw_val >= 0 else 0
                else:
                    config[current_field] = raw_val
            elif tok in CAT_VALUES:
                config[current_field] = CAT_VALUES[tok]
            else:
                config[current_field] = f"tok_{tok}"
            current_field = None
        i += 1

    return config


def config_to_str(cfg: dict) -> str:
    """One-line summary of a kernel config."""
    parts = []
    if "arch" in cfg:
        parts.append(cfg["arch"])
    if "kernel_type" in cfg:
        parts.append(cfg["kernel_type"])
    dtypes = []
    for k in ("element_a", "element_b"):
        if k in cfg:
            dtypes.append(str(cfg[k]))
    if dtypes:
        parts.append("x".join(dtypes))
    tiles = []
    for k in ("tile_m", "tile_n", "tile_k"):
        if k in cfg:
            tiles.append(str(cfg[k]))
    if tiles:
        parts.append(f"tile={'x'.join(tiles)}")
    if "stages" in cfg:
        parts.append(f"stg={cfg['stages']}")
    if "mma_class" in cfg:
        parts.append(cfg["mma_class"])
    if "mainloop" in cfg:
        parts.append(cfg["mainloop"])
    clusters = []
    for k in ("cluster_m", "cluster_n", "cluster_k"):
        if k in cfg:
            clusters.append(str(cfg[k]))
    if clusters and any(c != "1" for c in clusters):
        parts.append(f"cluster={'x'.join(clusters)}")
    return " | ".join(parts)


def generate_example_table(model, batch, n_examples=10, device="cuda"):
    """Generate a comparison table of before -> predicted -> actual after configs.

    Returns list of dicts with before, after_true, after_pred configs,
    cosine similarity, and config field match counts.
    """
    model_was_training = model.training
    model.train(False)
    with torch.no_grad():
        states = batch["states"][:n_examples].to(device)
        actions = batch["actions"][:n_examples].to(device)

        B = states.shape[0]
        before_tokens = states[:, 0]
        after_tokens = states[:, 1]

        z_before = model.state_encoder(before_tokens)
        z_after_true = model.state_encoder(after_tokens)
        z_action = model.action_encoder(actions[:, 0])
        z_pred = model.predictor(z_before.detach(), z_action)

        cos_sim = F.cosine_similarity(z_pred, z_after_true, dim=-1)

        # Nearest-neighbor decode: find closest after-state in full val batch
        all_after = model.state_encoder(batch["states"][:, 1].to(device))

        examples = []
        for i in range(B):
            before_cfg = decode_tokens(before_tokens[i])
            after_cfg = decode_tokens(after_tokens[i])

            sims = F.cosine_similarity(z_pred[i:i+1], all_after, dim=-1)
            best_idx = sims.argmax().item()
            pred_cfg = decode_tokens(batch["states"][best_idx, 1])

            match_keys = ["arch", "mma_class", "mainloop", "tile_m", "tile_n", "tile_k", "stages"]
            matches = sum(1 for k in match_keys if pred_cfg.get(k) == after_cfg.get(k))
            total = len(match_keys)

            examples.append({
                "before": before_cfg,
                "after_true": after_cfg,
                "after_pred": pred_cfg,
                "before_str": config_to_str(before_cfg),
                "after_true_str": config_to_str(after_cfg),
                "after_pred_str": config_to_str(pred_cfg),
                "cosine_sim": cos_sim[i].item(),
                "nn_cosine": sims[best_idx].item(),
                "config_match": f"{matches}/{total}",
                "exact_match": matches == total,
            })

    if model_was_training:
        model.train()
    return examples


def save_example_table(examples, step, output_dir):
    """Save examples as JSON and readable markdown."""
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, f"examples_step{step}.json")
    with open(json_path, "w") as f:
        json.dump(examples, f, indent=2)

    md_path = os.path.join(output_dir, f"examples_step{step}.md")
    with open(md_path, "w") as f:
        f.write(f"# KernelWM Example Predictions -- Step {step}\n\n")
        f.write(f"| # | Before | After (true) | After (pred) | cos | match |\n")
        f.write(f"|---|--------|-------------|-------------|-----|-------|\n")
        for i, ex in enumerate(examples):
            f.write(f"| {i+1} | {ex['before_str']} | {ex['after_true_str']} | "
                    f"{ex['after_pred_str']} | {ex['cosine_sim']:.3f} | {ex['config_match']} |\n")

        n_exact = sum(1 for ex in examples if ex["exact_match"])
        avg_cos = sum(ex["cosine_sim"] for ex in examples) / max(len(examples), 1)
        f.write(f"\n**Exact match: {n_exact}/{len(examples)}** | **Avg cosine: {avg_cos:.4f}**\n")

    print(f"  examples saved: {json_path}, {md_path}")
    return json_path, md_path


def main():
    import numpy as np

    hdf5_path = os.environ.get("WM_HDF5_PATH", "")
    if not hdf5_path:
        print("ERROR: WM_HDF5_PATH required", file=sys.stderr)
        sys.exit(1)

    model_dim = int(os.environ.get("WM_MODEL_DIM", "128"))
    num_loops = int(os.environ.get("WM_NUM_LOOPS", "6"))
    num_heads = int(os.environ.get("WM_NUM_HEADS", "4"))
    vocab_size = int(os.environ.get("WM_VOCAB_SIZE", "400"))
    max_seq_len = int(os.environ.get("WM_MAX_SEQ_LEN", "512"))
    encoder_loops = int(os.environ.get("WM_ENCODER_LOOPS", "6"))
    action_dim = int(os.environ.get("ACTION_DIM", "12"))
    lr = float(os.environ.get("WM_LR", "3e-4"))
    batch_size = int(os.environ.get("WM_BATCH_SIZE", "128"))
    total_steps = int(os.environ.get("WM_STEPS", "2000"))
    warmup_steps = int(os.environ.get("WM_WARMUP_STEPS", "200"))
    eval_interval = int(os.environ.get("WM_EVAL_INTERVAL", "100"))
    save_interval = int(os.environ.get("WM_SAVE_INTERVAL", "500"))
    output_dir = os.environ.get("OUTPUT_DIR", "./checkpoints")
    example_dir = os.environ.get("KWM_EXAMPLE_DIR", "./examples")
    n_examples = int(os.environ.get("KWM_SAVE_EXAMPLES", "10"))
    wandb_project = os.environ.get("WANDB_PROJECT", "crucible-kernel-wm")
    wandb_run_name = os.environ.get("WANDB_RUN_NAME", "")
    seed = int(os.environ.get("WM_SEED", "42"))

    dense_mode = os.environ.get("KWM_DENSE", "0") == "1"
    os.environ["ACTION_DIM"] = str(action_dim)
    os.environ["WM_POOL_MODE"] = os.environ.get("WM_POOL_MODE", "attn")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    tap_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, tap_root)

    from architectures.wm_base.wm_base import wm_base_kwargs_from_env
    if dense_mode:
        from architectures.kernel_wm.kernel_wm import KernelWorldModel, kernel_wm_kwargs_from_env
    else:
        from architectures.code_wm.code_wm import CodeWorldModel

    import h5py
    f = h5py.File(hdf5_path, "r")
    has_dense = "before_dense" in f
    if dense_mode and not has_dense:
        print("ERROR: KWM_DENSE=1 but HDF5 has no before_dense field. Rebuild with pairs_to_hdf5.py.", file=sys.stderr)
        sys.exit(1)
    if dense_mode:
        num_pairs = f["before_dense"].shape[0]
        input_dim = f["before_dense"].shape[1]
        ctx_window = input_dim
    else:
        num_pairs = f["before_tokens"].shape[0]
        ctx_window = f["before_tokens"].shape[1]
    data_action_dim = f["edit_actions"].shape[1]
    action_dim = data_action_dim
    os.environ["ACTION_DIM"] = str(action_dim)

    np.random.seed(seed)
    torch.manual_seed(seed)

    val_frac = 0.1
    all_idx = np.random.permutation(num_pairs)
    n_val = max(int(num_pairs * val_frac), batch_size)
    val_indices = np.sort(all_idx[:n_val])
    train_indices = np.sort(all_idx[n_val:])

    print("=== KernelWM Training ===")
    print(f"  Data:     {hdf5_path} ({num_pairs:,} pairs, action_dim={action_dim})")
    if dense_mode:
        print(f"  Mode:     DENSE (input_dim={input_dim})")
    else:
        print(f"  Mode:     TOKEN (ctx={ctx_window}, vocab={vocab_size})")
    print(f"  Split:    {len(train_indices):,} train, {len(val_indices):,} val")
    print(f"  Model:    dim={model_dim}, loops={num_loops}, heads={num_heads}")
    print(f"  Training: {total_steps} steps, batch={batch_size}, lr={lr}")
    print(f"  Examples: {n_examples} per eval -> {example_dir}")
    print(f"  Device:   {device}")
    print()

    if dense_mode:
        kwargs = kernel_wm_kwargs_from_env()
        kwargs["input_dim"] = input_dim
        model = KernelWorldModel(**kwargs).to(device)
    else:
        kwargs = wm_base_kwargs_from_env(None)
        model = CodeWorldModel(
            vocab_size=vocab_size, max_seq_len=max_seq_len,
            encoder_loops=encoder_loops, **kwargs,
        ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    use_wandb = False
    try:
        import wandb
        if not wandb_run_name:
            wandb_run_name = f"kernel-wm-{model_dim}d-{action_dim}act"
        wandb.init(project=wandb_project, name=wandb_run_name, config={
            "model_dim": model_dim, "num_loops": num_loops, "vocab_size": vocab_size,
            "action_dim": action_dim, "lr": lr, "batch_size": batch_size,
            "total_steps": total_steps, "n_params": n_params,
            "data_file": os.path.basename(hdf5_path), "num_pairs": num_pairs,
        })
        use_wandb = True
        print(f"W&B: {wandb.run.url}")
    except Exception as e:
        print(f"W&B init failed ({e}), training without logging")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=lr * 0.01,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps],
    )

    # Preload ALL data into numpy for fast random access (736x vs HDF5 random reads)
    print("  Preloading data into memory...", end=" ", flush=True)
    action_np = f["edit_actions"][:].astype(np.float32)
    if dense_mode:
        before_np = f["before_dense"][:].astype(np.float32)
        after_np = f["after_dense"][:].astype(np.float32)
    else:
        before_np = f["before_tokens"][:].astype(np.int64)
        after_np = f["after_tokens"][:].astype(np.int64)
    diff_np = f["diff_tokens"][:].astype(np.int64) if "diff_tokens" in f else None
    total_mb = (before_np.nbytes + after_np.nbytes + action_np.nbytes) / 1024 / 1024
    if diff_np is not None:
        total_mb += diff_np.nbytes / 1024 / 1024
    print(f"{total_mb:.0f} MB loaded")
    f.close()  # HDF5 no longer needed

    def get_batch(pool_indices):
        idx = np.random.choice(pool_indices, size=min(batch_size, len(pool_indices)), replace=False)
        before = torch.from_numpy(before_np[idx]).to(device)
        after = torch.from_numpy(after_np[idx]).to(device)
        actions = torch.from_numpy(action_np[idx]).to(device)
        batch = {
            "states": torch.stack([before, after], dim=1),
            "actions": actions.unsqueeze(1),
        }
        if diff_np is not None:
            batch["diff_tokens"] = torch.from_numpy(diff_np[idx].astype(np.int64)).to(device).unsqueeze(1)
        batch["_indices"] = idx
        return batch

    model.train()
    start_time = time.time()
    best_val_dcos = -1.0

    for step in range(total_steps):
        batch = get_batch(train_indices)
        optimizer.zero_grad()
        fwd_args = {k: v for k, v in batch.items() if not k.startswith("_")}
        out = model.forward(**fwd_args)
        out["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        optimizer.step()
        scheduler.step()

        loss_val = out["loss"].item()
        delta_cos = out.get("delta_cos_sim", torch.tensor(0.0)).item()
        delta_ratio = out.get("delta_norm_ratio", torch.tensor(0.0)).item()

        if step % eval_interval == 0:
            elapsed = time.time() - start_time
            sps = (step + 1) / elapsed if elapsed > 0 else 0
            print(f"step {step:5d}/{total_steps} | loss={loss_val:.4f} | "
                  f"dcos={delta_cos:.3f} dratio={delta_ratio:.2f} | grad={grad_norm:.2f} | {sps:.1f} s/s")

            model.train(False)
            with torch.no_grad():
                vb = get_batch(val_indices)
                vb_fwd = {k: v for k, v in vb.items() if not k.startswith("_")}
                vo = model.forward(**vb_fwd)
                zp = vo["pred_embeddings"][:, 0].reshape(-1, model_dim)
                zt = vo["target_embeddings"][:, 0].reshape(-1, model_dim)
                z_curr = model.state_encoder(vb["states"][:, 0]).reshape(-1, model_dim)
                val_cos = F.cosine_similarity(zp, zt, dim=-1).mean().item()
                cos_copy = F.cosine_similarity(z_curr, zt, dim=-1).mean().item()
                dt = F.normalize(zt - z_curr, dim=-1)
                dp = F.normalize(zp - z_curr, dim=-1)
                val_dcos = (dt * dp).sum(dim=-1).mean().item()
                lift = val_cos - cos_copy

            print(f"  val dcos={val_dcos:.4f} cos={val_cos:.4f} "
                  f"(copy={cos_copy:.4f}, lift={lift:+.4f})")

            # Example tables every 5 evals or at step 0
            if step % (eval_interval * 5) == 0 or step == 0:
                examples = generate_example_table(model, vb, n_examples=n_examples, device=device)
                save_example_table(examples, step, example_dir)
                if use_wandb:
                    try:
                        table = wandb.Table(columns=["before", "after_true", "after_pred", "cosine", "match"])
                        for ex in examples:
                            table.add_data(ex["before_str"], ex["after_true_str"],
                                           ex["after_pred_str"], ex["cosine_sim"], ex["config_match"])
                        wandb.log({"examples": table}, step=step)
                    except Exception:
                        pass

            if val_dcos > best_val_dcos:
                best_val_dcos = val_dcos
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "step": step, "val_delta_cos": val_dcos,
                    "config": {"model_dim": model_dim, "num_loops": num_loops,
                               "vocab_size": vocab_size, "action_dim": action_dim,
                               "dense": dense_mode, "input_dim": input_dim if dense_mode else 0},
                }, os.path.join(output_dir, "kernel_wm_best.pt"))

            if use_wandb:
                wandb.log({
                    "train/loss": loss_val, "train/delta_cos": delta_cos,
                    "train/delta_norm_ratio": delta_ratio, "train/grad_norm": grad_norm,
                    "val/delta_cos": val_dcos, "val/cosine_sim": val_cos,
                    "val/copy_baseline": cos_copy, "val/lift": lift,
                }, step=step)

            model.train()

        if step > 0 and step % save_interval == 0:
            torch.save({
                "model_state_dict": model.state_dict(), "step": step,
                "config": {"model_dim": model_dim, "num_loops": num_loops,
                           "vocab_size": vocab_size, "action_dim": action_dim},
            }, os.path.join(output_dir, f"kernel_wm_step{step}.pt"))

    # Final save + examples
    torch.save({
        "model_state_dict": model.state_dict(), "step": total_steps,
        "best_val_dcos": best_val_dcos,
        "config": {"model_dim": model_dim, "num_loops": num_loops,
                   "vocab_size": vocab_size, "action_dim": action_dim},
    }, os.path.join(output_dir, "kernel_wm_final.pt"))

    model.train(False)
    with torch.no_grad():
        vb = get_batch(val_indices)
    examples = generate_example_table(model, vb, n_examples=min(20, n_examples * 2), device=device)
    save_example_table(examples, total_steps, example_dir)

    elapsed = time.time() - start_time
    print(f"\n=== Training Complete ===")
    print(f"  Steps: {total_steps}, Best val dcos: {best_val_dcos:.4f}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  Examples: {example_dir}/")

    if use_wandb:
        wandb.finish()
    f.close()


if __name__ == "__main__":
    main()
