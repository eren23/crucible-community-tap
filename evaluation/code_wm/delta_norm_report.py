#!/usr/bin/env python3
"""Delta magnitude / norm reporting for CodeWM.

Closes the "cosine alone can hide whether the model is predicting a tiny
vector" concern from the 2026-04-11 reviewer-response pass. For each
checkpoint we sample N single-step transitions (and optionally longer
trajectories for s2/s3) from the trajectory HDF5 used in training, encode
with both online and target encoders, and report:

  ||z_0||           mean magnitude of the state embedding
  ||delta_true||    mean magnitude of (target(s_{k+1}) - online(s_k))
                    which is the training-time delta target
  ||delta_pred||    mean magnitude of (predictor_output - online(s_k))
  ratio_true        ||delta_true|| / ||z_0||  -- how much of the state is
                    actually moving per edit?
  ratio_pred        ||delta_pred|| / ||z_0||
  cos(delta_pred, delta_true)  -- the existing delta_cos number for context

Report mean, median, q10/q25/q75/q90 across queries and a short histogram
so a reviewer can see whether the delta_cos claim is "predict direction of
a meaningful vector" or "predict direction of noise".

Usage (reuse /tmp/codewm_eval_venv or any venv with torch + h5py)::

    python delta_norm_report.py \\
        --checkpoint /path/to/code_wm_best.pt \\
        --data ~/.crucible-hub/taps/crucible-community-tap/data/commitpack_python_trajectories_1.5m.h5 \\
        --num-samples 2000 --max-steps 3
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _shared import load_codewm as load_model  # noqa: E402


def _h5_gather(ds, flat_indices: np.ndarray) -> np.ndarray:
    """Read rows from an h5py dataset by arbitrary (possibly duplicated or
    unsorted) indices. h5py requires strictly-increasing unique indices, so
    we dedupe + sort, do one read, and then reshape back to the original
    order via np.unique's inverse.
    """
    uniq, inv = np.unique(flat_indices, return_inverse=True)
    data = ds[uniq.tolist()]
    return np.asarray(data)[inv]


def _sample_windows(
    traj_offsets: np.ndarray,
    traj_lengths: np.ndarray,
    window_len: int,
    n_target: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample n_target (traj_id, start_step) tuples where the window fits.

    Returns an int array of shape [n_target, window_len] with absolute
    indices into the flat before_tokens/edit_actions arrays.
    """
    # Trajectories that can fit a window of window_len transitions.
    valid = np.where(traj_lengths >= window_len)[0]
    if len(valid) == 0:
        return np.zeros((0, window_len), dtype=np.int64)
    out: list[list[int]] = []
    for _ in range(n_target):
        t = int(rng.choice(valid))
        off = int(traj_offsets[t])
        max_start = int(traj_lengths[t]) - window_len
        s = int(rng.integers(0, max_start + 1))
        out.append([off + s + k for k in range(window_len)])
    return np.asarray(out, dtype=np.int64)


def _summarize(vals: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "q10": float(np.quantile(vals, 0.10)),
        "q25": float(np.quantile(vals, 0.25)),
        "q75": float(np.quantile(vals, 0.75)),
        "q90": float(np.quantile(vals, 0.90)),
        "std": float(np.std(vals)),
        "n": int(len(vals)),
    }


@torch.no_grad()
def run(
    checkpoint: str,
    data_path: str,
    num_samples: int = 2000,
    max_steps: int = 3,
    batch_size: int = 64,
    device: str = "cpu",
    seed: int = 42,
):
    import h5py

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    model, cfg = load_model(checkpoint, device)
    action_dim = cfg["action_dim"]
    ckpt_name = Path(checkpoint).name
    ckpt_parent = Path(checkpoint).parent.name
    label = f"{ckpt_parent}/{ckpt_name}" if ckpt_parent else ckpt_name
    print(f"Model: dim={cfg['model_dim']}  checkpoint={label}")

    f = h5py.File(data_path, "r")
    before = f["before_tokens"]
    after = f["after_tokens"]
    actions_ds = f["edit_actions"]
    traj_offsets = f["trajectory"]["traj_offsets"][:]
    traj_lengths = f["trajectory"]["traj_lengths"][:]

    per_step_summary: list[dict] = []
    for step in range(1, max_steps + 1):
        window_len = step  # step=1 -> 1 transition (pair), step=2 -> 2 transitions, etc.
        windows = _sample_windows(traj_offsets, traj_lengths, window_len, num_samples, rng)
        if len(windows) == 0:
            print(f"  [warn] no trajectories with {window_len} transitions, skipping step {step}")
            continue

        # For step k, we need the ORIGINAL state (before tokens of the first
        # transition) and the FINAL state (after tokens of the k-th
        # transition). We also need all k intermediate actions to run the
        # predictor forward from z_0.
        idx0 = windows[:, 0]  # first transition of the window
        idxk = windows[:, -1]  # last transition of the window

        # Gather via the dedupe-then-read helper (h5py requires strictly
        # increasing unique indices).
        t0 = time.time()
        before_0 = _h5_gather(before, idx0)
        after_k = _h5_gather(after, idxk)
        action_seq = np.zeros((len(windows), window_len, action_dim), dtype=np.float32)
        for j in range(window_len):
            action_seq[:, j] = _h5_gather(actions_ds, windows[:, j])

        print(f"  step={step}: loaded {len(windows)} windows in {time.time() - t0:.1f}s")

        # Encode in batches.
        z0_list, zk_target_list = [], []
        for i in range(0, len(windows), batch_size):
            b0 = torch.from_numpy(before_0[i:i + batch_size].astype(np.int64)).to(device)
            bk = torch.from_numpy(after_k[i:i + batch_size].astype(np.int64)).to(device)
            z0_list.append(model.state_encoder(b0).cpu())
            zk_target_list.append(model.target_encoder(bk).cpu())
        z0 = torch.cat(z0_list, dim=0)
        zk_target = torch.cat(zk_target_list, dim=0)

        # Predictor rollout from z0 with the sequence of actions.
        actions_t = torch.from_numpy(action_seq).to(device)
        z_cur = z0.clone().to(device)
        for j in range(window_len):
            z_cur = model.predict_next(z_cur, actions_t[:, j, :])
        z_pred = z_cur.cpu()

        # Compute metrics.
        delta_true = zk_target - z0
        delta_pred = z_pred - z0
        z0_norm = z0.norm(dim=-1).numpy()
        true_norm = delta_true.norm(dim=-1).numpy()
        pred_norm = delta_pred.norm(dim=-1).numpy()
        ratio_true = true_norm / np.clip(z0_norm, 1e-8, None)
        ratio_pred = pred_norm / np.clip(z0_norm, 1e-8, None)
        cos_dp = F.cosine_similarity(delta_pred, delta_true, dim=-1).numpy()

        summary = {
            "step": step,
            "n": int(len(windows)),
            "z0_norm": _summarize(z0_norm),
            "delta_true_norm": _summarize(true_norm),
            "delta_pred_norm": _summarize(pred_norm),
            "ratio_delta_true_over_z0": _summarize(ratio_true),
            "ratio_delta_pred_over_z0": _summarize(ratio_pred),
            "delta_cos_mixed": _summarize(cos_dp),
        }
        per_step_summary.append(summary)

    f.close()

    print()
    print(f"=== Delta norm summary  {label} ===")
    print(f"  {'metric':<28}" + "  ".join(f"{'s'+str(r['step']):>9}" for r in per_step_summary))
    rows = [
        ("||z_0|| mean", "z0_norm", "mean"),
        ("||delta_true|| mean", "delta_true_norm", "mean"),
        ("||delta_pred|| mean", "delta_pred_norm", "mean"),
        ("||delta_true||/||z_0|| mean", "ratio_delta_true_over_z0", "mean"),
        ("||delta_true||/||z_0|| median", "ratio_delta_true_over_z0", "median"),
        ("||delta_true||/||z_0|| q10", "ratio_delta_true_over_z0", "q10"),
        ("||delta_true||/||z_0|| q90", "ratio_delta_true_over_z0", "q90"),
        ("||delta_pred||/||z_0|| mean", "ratio_delta_pred_over_z0", "mean"),
        ("cos(delta_pred, delta_true)", "delta_cos_mixed", "mean"),
    ]
    for label_row, top_key, inner_key in rows:
        cells = "  ".join(
            f"{r[top_key][inner_key]:>9.4f}" for r in per_step_summary
        )
        print(f"  {label_row:<28}{cells}")
    return {"checkpoint": label, "per_step": per_step_summary, "config": cfg}


def main():
    p = argparse.ArgumentParser(description="Delta norm / magnitude reporting for CodeWM.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--data",
        default=str(
            Path.home()
            / ".crucible-hub/taps/crucible-community-tap/data/commitpack_python_trajectories_1.5m.h5"
        ),
    )
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--max-steps", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="")
    args = p.parse_args()

    result = run(
        args.checkpoint,
        args.data,
        num_samples=args.num_samples,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, default=float)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
