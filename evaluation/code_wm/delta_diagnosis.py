#!/usr/bin/env python3
"""DeltaCodeWM Generalization Diagnosis.

Tests why in-batch training metrics (dcos=0.957, lift=+0.128) don't transfer
to independent holdout (pred_vs_diff cosine=0.05).

Usage:
    python delta_diagnosis.py --checkpoint checkpoints_delta/kernel_wm_best.pt --data data/commitpackft_with_diffs.h5
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


def load_delta_model(checkpoint_path, device="cuda"):
    tap_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(tap_root))
    os.environ.setdefault("WM_POOL_MODE", "attn")
    os.environ["WM_DELTA_MODE"] = "1"
    os.environ["WM_DIFF_VOCAB_SIZE"] = "700"

    from architectures.code_wm.code_wm import CodeWorldModel
    from architectures.wm_base.wm_base import wm_base_kwargs_from_env

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    os.environ["WM_MODEL_DIM"] = str(cfg.get("model_dim", 128))
    os.environ["WM_NUM_LOOPS"] = str(cfg.get("num_loops", 6))
    os.environ["ACTION_DIM"] = str(cfg.get("action_dim", 15))

    kwargs = wm_base_kwargs_from_env()
    model = CodeWorldModel(vocab_size=700, max_seq_len=512, encoder_loops=6, **kwargs)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device)
    model.train(False)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-samples", type=int, default=1000)
    args = parser.parse_args()

    device = args.device
    model = load_delta_model(args.checkpoint, device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    f = h5py.File(args.data, "r")
    N = f["before_tokens"].shape[0]
    n = min(args.n_samples, N)

    np.random.seed(42)
    all_idx = np.random.permutation(N)
    train_idx = np.sort(all_idx[:int(N * 0.9)])
    val_idx = np.sort(all_idx[int(N * 0.9):])
    print(f"Data: {N} total, {len(train_idx)} train, {len(val_idx)} val\n")

    def load_batch(indices):
        idx = indices.tolist()
        return {
            "before": torch.from_numpy(f["before_tokens"][idx].astype(np.int64)).to(device),
            "after": torch.from_numpy(f["after_tokens"][idx].astype(np.int64)).to(device),
            "diffs": torch.from_numpy(f["diff_tokens"][idx].astype(np.int64)).to(device),
            "actions": torch.from_numpy(f["edit_actions"][idx].astype(np.float32)).to(device),
        }

    # TEST 1: Training-style batch vs independent holdout
    print("=" * 60)
    print("TEST 1: Training-style batch vs independent holdout")
    print("=" * 60)

    with torch.no_grad():
        sample_idx = np.sort(np.random.choice(train_idx, size=min(128, len(train_idx)), replace=False))
        batch = load_batch(sample_idx)
        states = torch.stack([batch["before"], batch["after"]], dim=1)
        actions = batch["actions"].unsqueeze(1)
        out = model.forward(states=states, actions=actions, diff_tokens=batch["diffs"].unsqueeze(1))
        train_style_dcos = out["delta_cos_sim"].item()
        print(f"  Training-style forward dcos: {train_style_dcos:.4f}")

        z_before = model.state_encoder(batch["before"])
        z_action = model.action_encoder(batch["actions"])
        z_pred = model.predictor(z_before, z_action)
        z_diff = model.diff_encoder(batch["diffs"])
        indep_cos = F.cosine_similarity(z_pred, z_diff, dim=-1).mean().item()
        print(f"  Independent encode cos:      {indep_cos:.4f}")

        val_sample = np.sort(np.random.choice(val_idx, size=min(128, len(val_idx)), replace=False))
        vbatch = load_batch(val_sample)
        z_vb = model.state_encoder(vbatch["before"])
        z_va = model.action_encoder(vbatch["actions"])
        z_vp = model.predictor(z_vb, z_va)
        z_vd = model.diff_encoder(vbatch["diffs"])
        val_cos = F.cosine_similarity(z_vp, z_vd, dim=-1).mean().item()
        print(f"  Val holdout cos:             {val_cos:.4f}")
        print(f"  Gap (train - val):           {train_style_dcos - val_cos:+.4f}")

    # TEST 2: Easy vs hard edits
    print(f"\n{'=' * 60}")
    print("TEST 2: Easy vs hard edits (by diff token count)")
    print("=" * 60)

    with torch.no_grad():
        sample = np.sort(np.random.choice(train_idx, size=min(n, len(train_idx)), replace=False))
        batch = load_batch(sample)
        diff_sizes = (batch["diffs"] > 0).sum(dim=-1).cpu().numpy()
        z_b = model.state_encoder(batch["before"])
        z_a = model.action_encoder(batch["actions"])
        z_p = model.predictor(z_b, z_a)
        z_d = model.diff_encoder(batch["diffs"])
        cos_all = F.cosine_similarity(z_p, z_d, dim=-1).cpu().numpy()

        for label, lo, hi in [("small (<30)", 0, 30), ("medium (30-100)", 30, 100),
                               ("large (100-300)", 100, 300), ("xlarge (300+)", 300, 9999)]:
            mask = (diff_sizes >= lo) & (diff_sizes < hi)
            if mask.sum() > 0:
                print(f"  {label:20s}: cos={cos_all[mask].mean():.4f}  (n={mask.sum()})")

    # TEST 3: Action vector ablation
    print(f"\n{'=' * 60}")
    print("TEST 3: Action vector ablation")
    print("=" * 60)

    with torch.no_grad():
        cos_normal = F.cosine_similarity(z_p, z_d, dim=-1).mean().item()
        z_zero_a = model.action_encoder(torch.zeros_like(batch["actions"]))
        cos_zero = F.cosine_similarity(model.predictor(z_b, z_zero_a), z_d, dim=-1).mean().item()
        z_rand_a = model.action_encoder(torch.randn_like(batch["actions"]))
        cos_rand = F.cosine_similarity(model.predictor(z_b, z_rand_a), z_d, dim=-1).mean().item()

        print(f"  Normal action:  cos={cos_normal:.4f}")
        print(f"  Zero action:    cos={cos_zero:.4f}")
        print(f"  Random action:  cos={cos_rand:.4f}")
        delta = abs(cos_normal - cos_zero)
        print(f"  Action matters? {'YES' if delta > 0.05 else 'NO'} (delta={delta:.4f})")

    # TEST 4: NN retrieval in diff space
    print(f"\n{'=' * 60}")
    print("TEST 4: NN retrieval in diff space")
    print("=" * 60)

    with torch.no_grad():
        cb_idx = np.sort(np.random.choice(train_idx, size=min(2000, len(train_idx)), replace=False))
        cb_diffs = torch.from_numpy(f["diff_tokens"][cb_idx.tolist()].astype(np.int64)).to(device)
        z_cb = F.normalize(model.diff_encoder(cb_diffs), dim=-1)

        z_vd_n = F.normalize(z_vd, dim=-1)
        sims = z_vd_n @ z_cb.T
        nn_cos = sims.max(dim=-1).values.mean().item()

        z_vp_n = F.normalize(z_vp, dim=-1)
        pred_nn = (z_vp_n @ z_cb.T).max(dim=-1).values.mean().item()

        print(f"  Val diff -> nearest train diff:  {nn_cos:.4f}")
        print(f"  Val pred -> nearest train diff:  {pred_nn:.4f}")
        print(f"  Diff encoder generalizes? {'YES' if nn_cos > 0.5 else 'WEAK'}")
        print(f"  Predictor generalizes?    {'YES' if pred_nn > 0.5 else 'WEAK'}")

    # TEST 5: Diversity analysis
    print(f"\n{'=' * 60}")
    print("TEST 5: Diversity analysis")
    print("=" * 60)

    with torch.no_grad():
        cos_ba = F.cosine_similarity(z_b[:128], model.state_encoder(batch["after"][:128]), dim=-1).mean().item()
        z_p_n = F.normalize(z_p[:200], dim=-1)
        cross_pred = (z_p_n @ z_p_n.T).fill_diagonal_(0).mean().item()
        z_d_n2 = F.normalize(z_d[:200], dim=-1)
        cross_diff = (z_d_n2 @ z_d_n2.T).fill_diagonal_(0).mean().item()

        print(f"  State: before vs after cos:  {cos_ba:.4f} (1.0=identical)")
        print(f"  Diff embedding std/dim:      {z_d.std(dim=0).mean():.4f}")
        print(f"  Pred embedding std/dim:      {z_p.std(dim=0).mean():.4f}")
        print(f"  Cross-sample sim (pred):     {cross_pred:.4f} (high=collapse)")
        print(f"  Cross-sample sim (diff):     {cross_diff:.4f}")

    f.close()
    print(f"\n{'=' * 60}")
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
