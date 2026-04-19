#!/usr/bin/env python3
"""Latent gradient planning eval for CodeWM.

Per the temporal-straightening / planning prior art (arxiv 2603.12231):
  given z_before = encoder(state_t) and z_target = target_encoder(state_{t+1}),
  can we recover the latent action via gradient descent in latent space?

  z_a* = argmin_z_a  || predictor(z_before, z_a) - z_target ||^2

If yes → latent space supports planning. If no → predictor isn't an
inverse model. This is a downstream usefulness check beyond retrieval KNN.

We compare:
  - z_a_opt: latent action recovered by gradient descent (K steps)
  - z_a_true: action_encoder(true_action_vector)
  - z_a_zero: zero vector (no-op baseline)
  - z_a_rand: random vector (chance baseline)

Metrics per pair:
  - cos(predictor(z_before, z_a_opt), z_target)  -- planning convergence
  - cos(z_a_opt, z_a_true)                        -- action recovery
  - improvement of opt over zero/rand baselines

Usage::

    python latent_planning.py \\
        --checkpoint /tmp/vicreg_promotion_step28000.pt \\
        --data data/commitpackft_with_diffs.h5 \\
        --device cpu --n-samples 200 --opt-steps 100 --opt-lr 0.05

Output JSON includes per-method aggregate stats and a per-sample histogram.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import h5py

TAP = Path.home() / ".crucible-hub/taps/crucible-community-tap"
sys.path.insert(0, str(TAP))
sys.path.insert(0, str(TAP / "evaluation/code_wm"))

from _shared import load_codewm  # noqa: E402


def encode_pair_batch(model, before_ids, after_ids):
    """Returns (z_before [N,D], z_target [N,D])."""
    model.train(False)
    with torch.no_grad():
        z_before = model.state_encoder(before_ids)
        z_target = model.target_encoder(after_ids)
    return z_before, z_target


def optimize_action(model, z_before, z_target, *, n_steps, lr, init="rand"):
    """Gradient descent on latent action z_a to minimize predictor MSE.

    Returns (z_a_opt [N,D], final_loss_per_sample [N]).
    """
    N, D = z_before.shape
    if init == "rand":
        z_a = torch.randn(N, D, device=z_before.device) * 0.1
    elif init == "zero":
        z_a = torch.zeros(N, D, device=z_before.device)
    else:
        raise ValueError(init)
    z_a.requires_grad_(True)
    opt = torch.optim.Adam([z_a], lr=lr)

    z_before_d = z_before.detach()
    z_target_d = z_target.detach()
    for _ in range(n_steps):
        opt.zero_grad()
        pred = model.predictor(z_before_d, z_a)
        loss = F.mse_loss(pred, z_target_d, reduction="none").mean(-1)
        loss.sum().backward()
        opt.step()
    with torch.no_grad():
        pred = model.predictor(z_before_d, z_a)
        per_sample_mse = F.mse_loss(pred, z_target_d, reduction="none").mean(-1)
    return z_a.detach(), per_sample_mse.detach()


def cos(a, b):
    return F.cosine_similarity(a, b, dim=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--opt-steps", type=int, default=100)
    p.add_argument("--opt-lr", type=float, default=0.05)
    p.add_argument("--out", default="")
    args = p.parse_args()

    print(f"loading checkpoint: {args.checkpoint}", file=sys.stderr)
    model, cfg = load_codewm(args.checkpoint, device=args.device)
    D = cfg.get("model_dim", 128)
    print(f"  model_dim={D}, encoder_loops={cfg.get('encoder_loops')}", file=sys.stderr)

    print(f"loading data + reproducing val split (seed={args.seed})", file=sys.stderr)
    with h5py.File(args.data, "r") as f:
        N_total = f["before_tokens"].shape[0]
        np.random.seed(args.seed)
        all_idx = np.random.permutation(N_total)
        n_val = max(int(N_total * args.val_frac), 128)
        val_idx = np.sort(all_idx[:n_val])[: args.n_samples]
        before = torch.from_numpy(f["before_tokens"][val_idx].astype(np.int64))
        after = torch.from_numpy(f["after_tokens"][val_idx].astype(np.int64))
        actions = torch.from_numpy(f["edit_actions"][val_idx].astype(np.float32))
    N = len(val_idx)
    print(f"  using N={N} val samples", file=sys.stderr)

    print("encoding pairs", file=sys.stderr)
    t0 = time.time()
    z_before, z_target = encode_pair_batch(model, before, after)
    print(f"  done in {time.time()-t0:.1f}s", file=sys.stderr)

    # ─── Baselines ───
    with torch.no_grad():
        z_a_true = model.action_encoder(actions)
        pred_true = model.predictor(z_before, z_a_true)
        mse_true = F.mse_loss(pred_true, z_target, reduction="none").mean(-1)
        cos_true = cos(pred_true, z_target)

        z_a_zero = torch.zeros(N, D, device=z_before.device)
        pred_zero = model.predictor(z_before, z_a_zero)
        mse_zero = F.mse_loss(pred_zero, z_target, reduction="none").mean(-1)
        cos_zero = cos(pred_zero, z_target)

        torch.manual_seed(args.seed)
        z_a_rand = torch.randn(N, D, device=z_before.device) * 0.1
        pred_rand = model.predictor(z_before, z_a_rand)
        mse_rand = F.mse_loss(pred_rand, z_target, reduction="none").mean(-1)
        cos_rand = cos(pred_rand, z_target)

    # ─── Gradient planning ───
    print(f"running gradient planning ({args.opt_steps} steps, lr={args.opt_lr})", file=sys.stderr)
    t0 = time.time()
    z_a_opt, mse_opt = optimize_action(
        model, z_before, z_target,
        n_steps=args.opt_steps, lr=args.opt_lr, init="rand",
    )
    with torch.no_grad():
        pred_opt = model.predictor(z_before, z_a_opt)
        cos_opt = cos(pred_opt, z_target)
        cos_action_recovery = cos(z_a_opt, z_a_true)
    print(f"  done in {time.time()-t0:.1f}s", file=sys.stderr)

    # ─── Aggregate ───
    def stats(t):
        return {"mean": float(t.mean()), "std": float(t.std()),
                "median": float(t.median()), "min": float(t.min()),
                "max": float(t.max())}

    results = {
        "checkpoint": args.checkpoint,
        "n_samples": N,
        "opt_steps": args.opt_steps, "opt_lr": args.opt_lr,
        "predictor_mse": {
            "true_action": stats(mse_true),
            "zero_action": stats(mse_zero),
            "rand_action": stats(mse_rand),
            "opt_action": stats(mse_opt),
        },
        "cos_pred_target": {
            "true_action": stats(cos_true),
            "zero_action": stats(cos_zero),
            "rand_action": stats(cos_rand),
            "opt_action": stats(cos_opt),
        },
        "cos_z_a_opt_vs_z_a_true": stats(cos_action_recovery),
    }

    print("\n=== Predictor MSE (lower=better) ===")
    for k, s in results["predictor_mse"].items():
        print(f"  {k:15s}  mean={s['mean']:.6f}  median={s['median']:.6f}")

    print("\n=== cos(pred(z_before, z_a), z_target) (1=perfect) ===")
    for k, s in results["cos_pred_target"].items():
        print(f"  {k:15s}  mean={s['mean']:+.4f}  median={s['median']:+.4f}")

    print("\n=== cos(z_a_opt, z_a_true) (does planning recover the right action?) ===")
    s = results["cos_z_a_opt_vs_z_a_true"]
    print(f"  mean={s['mean']:+.4f}  median={s['median']:+.4f}  "
          f"min={s['min']:+.4f}  max={s['max']:+.4f}")

    # Useful summary: does opt beat baselines on predictor MSE?
    opt_vs_zero = float((mse_opt < mse_zero).float().mean())
    opt_vs_rand = float((mse_opt < mse_rand).float().mean())
    opt_vs_true = float((mse_opt < mse_true).float().mean())
    print(f"\n=== Planning beats baseline on MSE? (fraction of samples) ===")
    print(f"  opt < zero_action:  {opt_vs_zero:.3f}")
    print(f"  opt < rand_action:  {opt_vs_rand:.3f}")
    print(f"  opt < true_action:  {opt_vs_true:.3f}  "
          f"  ← if >0.5 then planning is *better* than the true action embedding")
    results["opt_vs_baseline_winrate"] = {
        "vs_zero": opt_vs_zero, "vs_rand": opt_vs_rand, "vs_true": opt_vs_true,
    }

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
