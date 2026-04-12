#!/usr/bin/env python3
"""Multi-step rollout evaluation for Code World Model.

Two subcommands:

- **drift**: Phase 1 sanity check. Sample N transitions from the training h5,
  roll out the predictor with a fixed action for several steps, report
  magnitude + cosine drift. No ground truth required.

- **trajectory**: Phase 2 real eval. Extract per-file edit chains from a local
  git repo (ordered commits), encode each state, compute rolled-out vs
  teacher-forced vs copy-last prediction quality at step 1, 2, 3, 4.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# Shared checkpoint loader — see _shared.py.
_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from _shared import load_codewm as load_model  # noqa: E402


def _get_tap_module(name: str, rel_path: str):
    tap_root = Path(__file__).resolve().parent.parent.parent
    spec = importlib.util.spec_from_file_location(name, tap_root / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def run_drift(
    checkpoint: str, data_path: str, num_samples: int = 500, max_steps: int = 5,
    device: str = "cpu", seed: int = 42,
):
    import h5py
    np.random.seed(seed)
    torch.manual_seed(seed)

    model, cfg = load_model(checkpoint, device)
    print(f"Model: dim={cfg['model_dim']}, checkpoint={Path(checkpoint).name}")

    f = h5py.File(data_path, "r")
    n_total = f["before_tokens"].shape[0]
    indices = np.sort(np.random.choice(n_total, size=num_samples, replace=False))
    before = torch.from_numpy(f["before_tokens"][indices.tolist()].astype(np.int64)).to(device)
    actions = torch.from_numpy(f["edit_actions"][indices.tolist()].astype(np.float32)).to(device)
    f.close()

    z0 = model.encode(before)
    print(f"Encoded {z0.shape[0]} initial states, dim={z0.shape[1]}")

    z = z0.clone()
    trace = []
    z0_norm = z0.norm(dim=-1).mean().item()
    trace.append({"step": 0, "norm_mean": z0_norm, "norm_ratio_to_step0": 1.0,
                  "cos_to_step0": 1.0, "cos_to_prev": 1.0})
    z_prev = z0.clone()
    for t in range(1, max_steps + 1):
        z = model.predict_next(z, actions)
        norm = z.norm(dim=-1).mean().item()
        cos0 = F.cosine_similarity(z, z0, dim=-1).mean().item()
        cos_prev = F.cosine_similarity(z, z_prev, dim=-1).mean().item()
        trace.append({"step": t, "norm_mean": norm,
                      "norm_ratio_to_step0": norm / max(z0_norm, 1e-8),
                      "cos_to_step0": cos0, "cos_to_prev": cos_prev})
        z_prev = z.clone()

    print(f"\nDrift trace (N={num_samples}, fixed action, {max_steps} steps):")
    print(f"{'step':>4} {'|z|':>10} {'|z|/|z0|':>10} {'cos(z,z0)':>12} {'cos(z,z-1)':>12}")
    for r in trace:
        print(f"{r['step']:>4} {r['norm_mean']:>10.4f} {r['norm_ratio_to_step0']:>10.4f} "
              f"{r['cos_to_step0']:>12.4f} {r['cos_to_prev']:>12.4f}")

    final = trace[-1]
    print("\nVerdict:")
    if final["norm_ratio_to_step0"] > 10.0:
        print(f"  [FAIL] magnitude blew up: {final['norm_ratio_to_step0']:.1f}x")
    elif final["norm_ratio_to_step0"] < 0.1:
        print(f"  [FAIL] magnitude collapsed: {final['norm_ratio_to_step0']:.3f}x")
    else:
        print(f"  [PASS] magnitude stable: {final['norm_ratio_to_step0']:.2f}x at step {max_steps}")
    if abs(final["cos_to_prev"] - 1.0) < 1e-4:
        print(f"  [FAIL] fixed-point collapse: cos(z,z-1)={final['cos_to_prev']:.4f}")
    else:
        print(f"  [PASS] no fixed-point collapse: cos(z,z-1)={final['cos_to_prev']:.4f}")
    return trace


def _run_git(repo: Path, *args: str) -> str:
    try:
        res = subprocess.run(["git", "-C", str(repo), *args],
                             capture_output=True, text=True, timeout=30)
        return res.stdout if res.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def _file_at_rev(repo: Path, rev: str, path: str) -> str:
    return _run_git(repo, "show", f"{rev}:{path}")


def collect_file_trajectories(repo: Path, min_edits: int = 4,
                              max_commits: int = 2000, max_trajs: int = 200):
    log = _run_git(repo, "log", "--no-merges", f"-n{max_commits}",
                   "--pretty=format:%H%x00%s", "--name-only")
    if not log:
        return []
    file_shas: dict[str, list] = defaultdict(list)
    for block in log.strip().split("\n\n"):
        lines = block.strip().split("\n")
        if not lines or "\x00" not in lines[0]:
            continue
        sha, subject = lines[0].split("\x00", 1)
        files = [f for f in lines[1:] if f.strip().endswith(".py")]
        for py_file in files:
            file_shas[py_file].append((sha, subject))
    trajectories = []
    for py_file, history in file_shas.items():
        if len(history) < min_edits + 1:
            continue
        chain = list(reversed(history[: min_edits + 1]))
        shas = [c[0] for c in chain]
        states = []
        skip = False
        for sha in shas:
            content = _file_at_rev(repo, sha, py_file)
            if len(content) > 40000 or not content.strip():
                skip = True
                break
            states.append(content)
        if skip or len(states) != min_edits + 1:
            continue
        trajectories.append({
            "file": py_file, "shas": shas,
            "messages": [c[1][:100] for c in chain], "states": states,
        })
        if len(trajectories) >= max_trajs:
            break
    return trajectories


@torch.no_grad()
def run_trajectory(checkpoint: str, repo: str, min_edits: int = 4, max_steps: int = 4,
                   max_commits: int = 2000, max_trajs: int = 200,
                   device: str = "cpu", out_path=None):
    model, cfg = load_model(checkpoint, device)
    tok_mod = _get_tap_module("ast_tokenizer", "collectors/ast_tokenizer.py")
    diff_mod = _get_tap_module("ast_diff", "collectors/ast_diff.py")
    ast_tokenize = tok_mod.ast_tokenize
    compute_rich_action = diff_mod.compute_rich_action
    action_dim = cfg["action_dim"]
    max_len = cfg["max_seq_len"]

    print(f"Model: dim={cfg['model_dim']}, action_dim={action_dim}, "
          f"checkpoint={Path(checkpoint).name}")
    print(f"Walking git history in {repo}...")
    t0 = time.time()
    trajs = collect_file_trajectories(Path(repo).resolve(), min_edits=min_edits,
                                      max_commits=max_commits, max_trajs=max_trajs)
    print(f"  {len(trajs)} trajectories (>= {min_edits+1} states) in {time.time()-t0:.1f}s")
    if not trajs:
        print("  No trajectories collected — aborting.")
        return None

    max_steps = min(max_steps, min_edits)
    all_states = []
    for t in trajs:
        all_states.extend(t["states"][: max_steps + 1])
    print(f"Encoding {len(all_states)} states...")
    t0 = time.time()
    toks = np.stack([ast_tokenize(s, max_len=max_len) for s in all_states], axis=0)
    toks_t = torch.from_numpy(toks.astype(np.int64)).to(device)
    # Compute BOTH online encoder outputs (for state-space metrics and rollout
    # starting point) and target encoder outputs (for mixed-formula deltas that
    # match the training metric: delta_true = target(s_{k+1}) - online(s_k)).
    z_all = model.state_encoder(toks_t)             # online encoder
    z_target_all = model.target_encoder(toks_t)    # target (EMA) encoder
    print(f"  encoded in {time.time()-t0:.1f}s")

    N_traj = len(trajs)
    D = z_all.shape[1]
    z_states = z_all.reshape(N_traj, max_steps + 1, D)
    z_target_states = z_target_all.reshape(N_traj, max_steps + 1, D)

    print(f"Computing actions ({max_steps} transitions × {N_traj} trajs)...")
    actions_all = np.zeros((N_traj, max_steps, action_dim), dtype=np.float32)
    for i, t in enumerate(trajs):
        for k in range(max_steps):
            before, after = t["states"][k], t["states"][k + 1]
            rich = compute_rich_action(before, after)
            actions_all[i, k] = rich[:action_dim]
    actions_t = torch.from_numpy(actions_all).to(device)

    # Teacher-forced: pred starts from online(s_k)
    teacher_forced = []
    for k in range(max_steps):
        z_prev_gt = z_states[:, k, :]
        a_k = actions_t[:, k, :]
        teacher_forced.append(model.predict_next(z_prev_gt, a_k))

    # Rolled-out: pred starts from online(s_0), then feeds its own output back
    rolled_out = []
    z_cur = z_states[:, 0, :].clone()
    for k in range(max_steps):
        a_k = actions_t[:, k, :]
        z_cur = model.predict_next(z_cur, a_k)
        rolled_out.append(z_cur.clone())

    results = {"per_step": [], "model_dim": cfg["model_dim"], "n_traj": N_traj}
    print(f"\nPer-step metrics (N_traj={N_traj}):")
    print(f"{'step':>4} {'teach_cos':>10} {'rolled_cos':>11} {'copy_cos':>10} "
          f"{'rand_cos':>10} {'norm_rat':>10} {'dcos_on':>9} {'dcos_mix':>9}")
    for k in range(max_steps):
        z_gt = z_states[:, k + 1, :]                   # online(s_{k+1})
        z_gt_target = z_target_states[:, k + 1, :]     # target(s_{k+1}) — training's "true" target
        z_prev_gt = z_states[:, k, :]                  # online(s_k)
        z_teach = teacher_forced[k]
        z_roll = rolled_out[k]
        teach_cos = F.cosine_similarity(z_teach, z_gt, dim=-1).mean().item()
        rolled_cos = F.cosine_similarity(z_roll, z_gt, dim=-1).mean().item()
        copy_cos = F.cosine_similarity(z_prev_gt, z_gt, dim=-1).mean().item()
        z_rand = torch.randn_like(z_gt)
        z_rand = z_rand * (z_gt.norm(dim=-1, keepdim=True) /
                           z_rand.norm(dim=-1, keepdim=True).clamp_min(1e-8))
        rand_cos = F.cosine_similarity(z_rand, z_gt, dim=-1).mean().item()
        norm_ratio = (z_roll.norm(dim=-1) / z_teach.norm(dim=-1).clamp_min(1e-8)).mean().item()

        # (A) online delta (legacy): delta_true = online(s_{k+1}) - online(s_k),
        #     delta_pred = rolled_pred - previous_rolled_or_online(s_0)
        pred_delta_online = z_roll - (z_states[:, 0, :] if k == 0 else rolled_out[k - 1])
        true_delta_online = z_gt - z_prev_gt
        delta_cos_online = F.cosine_similarity(pred_delta_online, true_delta_online, dim=-1).mean().item()

        # (B) MIXED delta (matches training metric wm_base.py:432-434):
        #     delta_true = target(s_{k+1}) - online(s_k)
        #     delta_pred = teacher_forced_pred - online(s_k)
        #     Teacher-forced only; rollout in this space is ill-defined because
        #     predict_next is trained for online-space inputs.
        pred_delta_mixed = z_teach - z_prev_gt
        true_delta_mixed = z_gt_target - z_prev_gt
        delta_cos_mixed = F.cosine_similarity(pred_delta_mixed, true_delta_mixed, dim=-1).mean().item()

        step_r = {"step": k + 1, "teacher_forced_cos": teach_cos, "rolled_out_cos": rolled_cos,
                  "copy_last_cos": copy_cos, "random_cos": rand_cos,
                  "rolled_norm_ratio_to_teach": norm_ratio,
                  "delta_cos_online": delta_cos_online,
                  "delta_cos_mixed_teacher_forced": delta_cos_mixed}
        results["per_step"].append(step_r)
        print(f"{k+1:>4} {teach_cos:>10.4f} {rolled_cos:>11.4f} {copy_cos:>10.4f} "
              f"{rand_cos:>10.4f} {norm_ratio:>10.4f} {delta_cos_online:>9.4f} {delta_cos_mixed:>9.4f}")

    print("\nVerdict:")
    step_checks = []
    for sr in results["per_step"]:
        beats_copy = sr["rolled_out_cos"] > sr["copy_last_cos"]
        beats_rand = sr["rolled_out_cos"] > sr["random_cos"]
        step_checks.append((sr["step"], beats_copy, beats_rand))
        tag_c = "PASS" if beats_copy else "FAIL"
        tag_r = "PASS" if beats_rand else "FAIL"
        print(f"  step {sr['step']}: rolled vs copy-last [{tag_c}] "
              f"(delta={sr['rolled_out_cos']-sr['copy_last_cos']:+.4f}), vs random [{tag_r}]")
    beat_copy_all = all(c[1] for c in step_checks)
    beat_copy_any = any(c[1] for c in step_checks)
    if beat_copy_all:
        print("  [PASS] rolled-out beats copy-last at ALL steps")
    elif beat_copy_any:
        first_fail = next(c[0] for c in step_checks if not c[1])
        print(f"  [PARTIAL] rolled-out beats copy-last until step {first_fail - 1}")
    else:
        print("  [FAIL] rolled-out never beats copy-last baseline")

    if out_path:
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")
    return results


def main():
    p = argparse.ArgumentParser(description="Multi-step rollout for Code WM")
    sp = p.add_subparsers(dest="cmd", required=True)
    pd = sp.add_parser("drift")
    pd.add_argument("--checkpoint", required=True)
    pd.add_argument("--data", required=True)
    pd.add_argument("--num-samples", type=int, default=500)
    pd.add_argument("--max-steps", type=int, default=5)
    pd.add_argument("--device", default="cpu")
    pd.add_argument("--seed", type=int, default=42)
    pt = sp.add_parser("trajectory")
    pt.add_argument("--checkpoint", required=True)
    pt.add_argument("--repo", required=True)
    pt.add_argument("--min-edits", type=int, default=4)
    pt.add_argument("--max-steps", type=int, default=4)
    pt.add_argument("--max-commits", type=int, default=2000)
    pt.add_argument("--max-trajs", type=int, default=200)
    pt.add_argument("--device", default="cpu")
    pt.add_argument("--out", default=None)
    args = p.parse_args()
    if args.cmd == "drift":
        run_drift(args.checkpoint, args.data, args.num_samples, args.max_steps,
                  args.device, args.seed)
    elif args.cmd == "trajectory":
        run_trajectory(args.checkpoint, args.repo, args.min_edits, args.max_steps,
                       args.max_commits, args.max_trajs, args.device, args.out)


if __name__ == "__main__":
    main()
