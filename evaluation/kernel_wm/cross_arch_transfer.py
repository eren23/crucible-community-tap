#!/usr/bin/env python3
"""KernelWM Cross-Architecture Transfer Evaluation.

The moonshot test: train on SM80->SM90 + SM90->SM100, test zero-shot
on SM80->SM100. Compares model against copy-last, random, and NN baselines.

Usage:
    python cross_arch_transfer.py \
        --checkpoint kernel_wm_best.pt \
        --train-data kernel_pairs_v2.h5 \
        --eval-data kernel_pairs_v2_eval.h5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F


# Token decoding constants
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
    31: "tf32", 32: "f8e4m3", 33: "f8e5m2", 34: "i8", 35: "u8",
    36: "row", 37: "col",
    38: "hmma", 39: "wgmma", 40: "tcgen05", 41: "simt",
    42: "cp_async", 43: "tma", 44: "tma_warp_specialized",
    46: "default", 47: "visitor", 48: "evt",
}
NUM_OFFSET = 100
NUM_FIELDS = {"tile_m", "tile_n", "tile_k", "cluster_m", "cluster_n", "cluster_k", "stages"}
MATCH_KEYS = ["arch", "mma_class", "mainloop", "tile_m", "tile_n", "tile_k", "stages"]


def decode_tokens(tokens) -> dict:
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.cpu().numpy().tolist()
    config = {}
    i = 1 if tokens[0] == BOS else 0
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
            current_field = None
        i += 1
    return config


def config_to_str(cfg: dict) -> str:
    parts = []
    for k in ["arch", "kernel_type"]:
        if k in cfg:
            parts.append(str(cfg[k]))
    tiles = [str(cfg.get(k, "?")) for k in ("tile_m", "tile_n", "tile_k")]
    parts.append("tile=" + "x".join(tiles))
    if "stages" in cfg:
        parts.append("stg=" + str(cfg["stages"]))
    for k in ["mma_class", "mainloop"]:
        if k in cfg:
            parts.append(str(cfg[k]))
    return " | ".join(parts)


def load_model(checkpoint_path: str, device: str = "cuda", dense: bool = False):
    tap_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(tap_root))
    os.environ.setdefault("WM_POOL_MODE", "attn")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    os.environ["WM_MODEL_DIM"] = str(config.get("model_dim", 128))
    os.environ["WM_NUM_LOOPS"] = str(config.get("num_loops", 6))
    os.environ["WM_NUM_HEADS"] = str(config.get("num_heads", 4))
    os.environ["ACTION_DIM"] = str(config.get("action_dim", 12))

    if dense:
        from architectures.kernel_wm.kernel_wm import KernelWorldModel, kernel_wm_kwargs_from_env
        kwargs = kernel_wm_kwargs_from_env()
        kwargs["input_dim"] = config.get("input_dim", 41)
        model = KernelWorldModel(**kwargs)
    else:
        from architectures.code_wm.code_wm import CodeWorldModel
        from architectures.wm_base.wm_base import wm_base_kwargs_from_env
        kwargs = wm_base_kwargs_from_env(None)
        model = CodeWorldModel(
            vocab_size=config.get("vocab_size", 400),
            max_seq_len=512, encoder_loops=6, **kwargs,
        )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device)
    model.train(False)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params ({'dense' if dense else 'token'}) from {checkpoint_path}")
    return model


def batch_encode(data_np, encoder, device, batch_size=512, dtype=None):
    all_z = []
    if dtype is None:
        dtype = np.float32 if data_np.dtype == np.float32 else np.int64
    for i in range(0, len(data_np), batch_size):
        batch = torch.from_numpy(data_np[i:i + batch_size].astype(dtype)).to(device)
        with torch.no_grad():
            z = encoder(batch)
        all_z.append(z.cpu())
    return torch.cat(all_z, dim=0)


def main():
    parser = argparse.ArgumentParser(description="KernelWM Cross-Arch Transfer")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="cross_arch_results.json")
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument("--dense", action="store_true", help="Use dense config vectors instead of tokens")
    args = parser.parse_args()

    device = args.device
    model = load_model(args.checkpoint, device, dense=args.dense)
    model_dim = model.model_dim

    # Load data — dense or token mode
    sk = "before_dense" if args.dense else "before_tokens"
    ak = "after_dense" if args.dense else "after_tokens"

    print(f"\nLoading eval: {args.eval_data} ({'dense' if args.dense else 'token'})")
    f_ev = h5py.File(args.eval_data, "r")
    ev_before = f_ev[sk][:]
    ev_after = f_ev[ak][:]
    ev_actions = f_ev["edit_actions"][:]
    ev_before_tok = f_ev["before_tokens"][:] if "before_tokens" in f_ev else None
    ev_after_tok = f_ev["after_tokens"][:] if "after_tokens" in f_ev else None
    N_ev = ev_before.shape[0]
    print(f"  {N_ev} SM80->SM100 pairs")

    print(f"Loading train: {args.train_data}")
    f_tr = h5py.File(args.train_data, "r")
    tr_after = f_tr[ak][:]
    tr_before = f_tr[sk][:]
    tr_after_tok = f_tr["after_tokens"][:] if "after_tokens" in f_tr else None
    tr_source = f_tr["source_arch"][:]
    tr_target = f_tr["target_arch"][:]
    N_tr = tr_after.shape[0]
    print(f"  {N_tr} train pairs (codebook)")

    # Encode
    print("\nEncoding...")
    t0 = time.time()
    z_ev_before = batch_encode(ev_before, model.state_encoder, device)
    z_ev_after = batch_encode(ev_after, model.state_encoder, device)
    z_tr_after = batch_encode(tr_after, model.state_encoder, device)
    z_tr_before = batch_encode(tr_before, model.state_encoder, device)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Predict
    print("Predicting...")
    z_pred_list = []
    sd = np.float32 if args.dense else np.int64
    for i in range(0, N_ev, 512):
        b = torch.from_numpy(ev_before[i:i+512].astype(sd)).to(device)
        a = torch.from_numpy(ev_actions[i:i+512].astype(np.float32)).to(device)
        with torch.no_grad():
            z_b = model.state_encoder(b)
            z_a = model.action_encoder(a)
            z_p = model.predictor(z_b, z_a)
        z_pred_list.append(z_p.cpu())
    z_pred = torch.cat(z_pred_list, dim=0)

    # Metrics
    z_true = z_ev_after
    z_before = z_ev_before
    delta_true = z_true - z_before
    delta_pred = z_pred - z_before

    methods = {}

    # Model
    methods["model"] = {
        "cosine_sim": F.cosine_similarity(z_pred, z_true, dim=-1).mean().item(),
        "delta_cos": F.cosine_similarity(
            F.normalize(delta_pred, dim=-1), F.normalize(delta_true, dim=-1), dim=-1
        ).mean().item(),
        "delta_norm_ratio": (delta_pred.norm(dim=-1) / delta_true.norm(dim=-1).clamp_min(1e-6)).mean().item(),
    }

    # Copy-last
    methods["copy_last"] = {
        "cosine_sim": F.cosine_similarity(z_before, z_true, dim=-1).mean().item(),
        "delta_cos": 0.0,
        "delta_norm_ratio": 0.0,
    }

    # Random
    z_rand = z_true[torch.randperm(N_ev)]
    methods["random"] = {
        "cosine_sim": F.cosine_similarity(z_rand, z_true, dim=-1).mean().item(),
        "delta_cos": 0.0,
        "delta_norm_ratio": 0.0,
    }

    # NN retrieval from train codebook
    print("Computing NN retrieval...")
    z_pred_n = F.normalize(z_pred, dim=-1)
    z_cb_n = F.normalize(z_tr_after, dim=-1)
    nn_idx = []
    for i in range(0, N_ev, 512):
        sims = z_pred_n[i:i+512] @ z_cb_n.T
        nn_idx.append(sims.argmax(dim=-1))
    nn_idx = torch.cat(nn_idx, dim=0)
    z_nn = z_tr_after[nn_idx]
    delta_nn = z_nn - z_before
    methods["nn_retrieval"] = {
        "cosine_sim": F.cosine_similarity(z_nn, z_true, dim=-1).mean().item(),
        "delta_cos": F.cosine_similarity(
            F.normalize(delta_nn, dim=-1), F.normalize(delta_true, dim=-1), dim=-1
        ).mean().item(),
        "delta_norm_ratio": (delta_nn.norm(dim=-1) / delta_true.norm(dim=-1).clamp_min(1e-6)).mean().item(),
    }

    # Delta transfer alignment
    print("Delta transfer alignment...")
    train_deltas = z_tr_after - z_tr_before
    mask_80_90 = (tr_source == 0) & (tr_target == 1)
    mask_90_100 = (tr_source == 1) & (tr_target == 2)

    mean_d_80_90 = train_deltas[mask_80_90].mean(dim=0) if mask_80_90.sum() > 0 else torch.zeros(model_dim)
    mean_d_90_100 = train_deltas[mask_90_100].mean(dim=0) if mask_90_100.sum() > 0 else torch.zeros(model_dim)
    mean_d_80_100 = delta_true.mean(dim=0)

    transfer = {
        "delta_80_90_vs_80_100": F.cosine_similarity(mean_d_80_90.unsqueeze(0), mean_d_80_100.unsqueeze(0)).item(),
        "delta_90_100_vs_80_100": F.cosine_similarity(mean_d_90_100.unsqueeze(0), mean_d_80_100.unsqueeze(0)).item(),
        "composed_vs_80_100": F.cosine_similarity((mean_d_80_90 + mean_d_90_100).unsqueeze(0), mean_d_80_100.unsqueeze(0)).item(),
        "n_80_90": int(mask_80_90.sum()),
        "n_90_100": int(mask_90_100.sum()),
    }

    # Config accuracy via NN decode
    print("Config accuracy...")
    n_ex = min(args.examples, N_ev)
    examples = []
    total_m, total_f = 0, 0
    for i in range(n_ex):
        # Use token data for human-readable decode (even in dense mode)
        b_tok = ev_before_tok[i] if ev_before_tok is not None else ev_before[i]
        a_tok = ev_after_tok[i] if ev_after_tok is not None else ev_after[i]
        p_tok = tr_after_tok[nn_idx[i].item()] if tr_after_tok is not None else tr_after[nn_idx[i].item()]
        before_cfg = decode_tokens(b_tok)
        true_cfg = decode_tokens(a_tok)
        pred_cfg = decode_tokens(p_tok)
        matches = sum(1 for k in MATCH_KEYS if pred_cfg.get(k) == true_cfg.get(k))
        total_m += matches
        total_f += len(MATCH_KEYS)
        examples.append({
            "before": config_to_str(before_cfg),
            "true": config_to_str(true_cfg),
            "pred": config_to_str(pred_cfg),
            "cos": round(F.cosine_similarity(z_pred[i:i+1], z_true[i:i+1], dim=-1).item(), 4),
            "match": f"{matches}/{len(MATCH_KEYS)}",
            "exact": matches == len(MATCH_KEYS),
        })
    config_acc = total_m / max(total_f, 1)

    # Print
    print("\n" + "=" * 70)
    print("CROSS-ARCH TRANSFER: SM80->SM100 (zero-shot)")
    print("=" * 70)
    print(f"\n{'Method':<18} {'cosine':<10} {'delta_cos':<12} {'dratio':<8}")
    print("-" * 48)
    for name, m in methods.items():
        print(f"{name:<18} {m['cosine_sim']:<10.4f} {m['delta_cos']:<12.4f} {m['delta_norm_ratio']:<8.2f}")

    lift = methods["model"]["cosine_sim"] - methods["copy_last"]["cosine_sim"]
    print(f"\nModel lift over copy-last: {lift:+.4f}")
    print(f"Model lift over NN:        {methods['model']['cosine_sim'] - methods['nn_retrieval']['cosine_sim']:+.4f}")

    print(f"\nDelta Transfer Alignment:")
    print(f"  delta(80->90) vs delta(80->100):             {transfer['delta_80_90_vs_80_100']:.4f}")
    print(f"  delta(90->100) vs delta(80->100):            {transfer['delta_90_100_vs_80_100']:.4f}")
    print(f"  composed(80->90 + 90->100) vs delta(80->100): {transfer['composed_vs_80_100']:.4f}")

    print(f"\nConfig Accuracy (NN decode): {config_acc:.1%}")
    print(f"\nExamples (first 10):")
    print(f"{'#':<4} {'Before':<35} {'True':<35} {'Pred':<35} {'cos':<6} {'match'}")
    print("-" * 120)
    for ex in examples[:10]:
        print(f"{examples.index(ex):<4} {ex['before']:<35} {ex['true']:<35} {ex['pred']:<35} {ex['cos']:.3f} {ex['match']}")

    n_exact = sum(1 for ex in examples if ex["exact"])
    print(f"\nExact: {n_exact}/{n_ex}")

    # Save
    results = {
        "type": "cross_arch_transfer", "holdout": "sm80->sm100",
        "n_eval": N_ev, "n_codebook": N_tr,
        "methods": methods, "transfer": transfer,
        "config_accuracy": config_acc, "examples": examples,
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output}")

    f_ev.close()
    f_tr.close()


if __name__ == "__main__":
    main()
