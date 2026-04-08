#!/usr/bin/env python3
"""Cross-repo retrieval eval with modern baselines (CodeBERT, CodeT5+, jina).

Extends ``cross_repo_eval.py`` to also embed the (before, after) pairs with
modern HF code embedders, not just CodeWM + BoW. Uses the same 20-repo basket
and the same leave-one-repo-out protocol so results are directly comparable
to both ``cross_repo_eval.py`` and ``modern_baselines_compare.py``.

For each HF model we embed ``before`` and ``after`` sources separately and
compute a delta vector ``z_after - z_before``. That's the same delta-space
representation CodeWM uses internally, so the head-to-head is apples-to-apples.

Usage::

    python cross_repo_modern_compare.py \\
        --checkpoint ~/.crucible-hub/taps/crucible-community-tap/checkpoints/phase5/phase5-contrast-15k-high-best.pt \\
        --out /tmp/cross_repo_modern.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))

from cross_repo_eval import (  # type: ignore
    DEFAULT_REPOS,
    bow_encode_deltas,
    clone_if_needed,
    compute_actions_for_pairs,
    encode_edit_pairs,
    extract_edit_pairs,
    joint_labels_from_actions,
    retrieval_with_joint_labels,
)
from codesearchnet_eval import load_codewm  # type: ignore


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def encode_hf_deltas(
    model_id: str,
    pairs: list[tuple[str, str]],
    device: str,
    max_length: int = 512,
    batch_size: int = 8,
    pooling: str = "mean",
):
    """Embed a list of (before, after) pairs with a HF model.

    Returns (delta_embeds, n_params, ms_per_pair) or None on load failure.
    """
    try:
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except ImportError:
        print(f"  [skip {model_id}] transformers not installed")
        return None

    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
        model.train(False)
    except Exception as exc:
        print(f"  [skip {model_id}] load failed: {type(exc).__name__}: {exc}")
        return None

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {model_id}: {n_params:,} params")

    def embed(texts: list[str]) -> torch.Tensor:
        out_vecs: list[torch.Tensor] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            try:
                out = model(**enc)
            except Exception as exc:
                print(f"  [skip {model_id}] forward failed: {exc}")
                raise
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                lhs = out.last_hidden_state
                if pooling == "mean":
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    pooled = (lhs * mask).sum(1) / mask.sum(1).clamp_min(1)
                else:
                    pooled = lhs[:, 0]
            elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                pooled = out.pooler_output
            else:
                raise RuntimeError(f"unknown output type: {type(out).__name__}")
            out_vecs.append(pooled.detach().cpu())
        return torch.cat(out_vecs, dim=0)

    befores = [p[0] for p in pairs]
    afters = [p[1] for p in pairs]

    t0 = time.time()
    try:
        z_b = embed(befores)
        z_a = embed(afters)
    except Exception:
        return None
    elapsed = time.time() - t0
    lat = elapsed * 1000 / (2 * len(pairs))  # two embeds per pair

    return z_a - z_b, n_params, lat


def run_leave_one_out(
    deltas: torch.Tensor,
    joint: np.ndarray,
    repo_slice: dict[str, tuple[int, int]],
    name: str,
) -> tuple[list[float], dict[str, dict]]:
    """Run leave-one-repo-out retrieval on a delta tensor. Returns (mrrs, per_repo)."""
    mrrs: list[float] = []
    per_repo: dict[str, dict] = {}
    for repo, (start, end) in repo_slice.items():
        q_mask = np.zeros(len(deltas), dtype=bool)
        q_mask[start:end] = True
        g_mask = ~q_mask
        q = deltas[q_mask]
        g = deltas[g_mask]
        q_joint = joint[q_mask]
        g_joint = joint[g_mask]
        metrics = retrieval_with_joint_labels(q, g, q_joint, g_joint)
        per_repo[repo] = metrics
        mrrs.append(metrics["mrr@10"])
    return mrrs, per_repo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--workdir", default="/tmp/cross_repo_eval")
    parser.add_argument("--commits-per-repo", type=int, default=150)
    parser.add_argument("--max-pairs-per-repo", type=int, default=200)
    parser.add_argument("--depth", type=int, default=500)
    parser.add_argument("--max-repos", type=int, default=20)
    parser.add_argument("--out", default="")
    parser.add_argument("--device", default=pick_device())
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "microsoft/codebert-base",
            "Salesforce/codet5p-110m-embedding",
            "jinaai/jina-embeddings-v2-base-code",
        ],
    )
    args = parser.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: extract pairs (reuse cloned repos) ---
    repo_data: dict[str, dict] = {}
    for name, url in DEFAULT_REPOS[: args.max_repos]:
        print(f"\n[{name}]")
        repo = clone_if_needed(name, url, workdir, depth=args.depth)
        if repo is None:
            continue
        pairs = extract_edit_pairs(
            repo,
            max_commits=args.commits_per_repo,
            max_pairs_per_repo=args.max_pairs_per_repo,
        )
        print(f"  {len(pairs)} edit pairs")
        if len(pairs) < 10:
            continue
        repo_data[name] = {"pairs": pairs}

    all_pairs: list[tuple[str, str]] = []
    repo_slice: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, data in repo_data.items():
        start = cursor
        all_pairs.extend(data["pairs"])
        cursor += len(data["pairs"])
        repo_slice[name] = (start, cursor)

    print(f"\nTotal: {len(all_pairs)} pairs across {len(repo_data)} repos")

    # --- Phase 2: compute joint labels once ---
    print("\n--- Computing actions + joint labels ---")
    actions = compute_actions_for_pairs(all_pairs)
    joint = joint_labels_from_actions(actions)

    summary: dict[str, dict] = {}

    # --- CodeWM ---
    print(f"\n--- CodeWM ({args.checkpoint}) ---")
    cw_model, cw_cfg = load_codewm(args.checkpoint, args.device)
    cw_params = sum(p.numel() for p in cw_model.parameters())
    t0 = time.time()
    cw_deltas = encode_edit_pairs(cw_model, all_pairs, cw_cfg["max_seq_len"], args.device)
    cw_lat = (time.time() - t0) * 1000 / len(all_pairs)
    cw_mrrs, cw_per_repo = run_leave_one_out(cw_deltas, joint, repo_slice, "CodeWM")
    summary["CodeWM"] = {
        "params": cw_params,
        "latency_ms_per_pair": cw_lat,
        "aggregate_mrr10": float(np.mean(cw_mrrs)),
        "per_repo": cw_per_repo,
        "checkpoint": str(args.checkpoint),
    }
    print(f"  aggregate MRR@10 = {np.mean(cw_mrrs):.4f}  latency={cw_lat:.1f} ms/pair")
    del cw_model, cw_deltas
    gc.collect()
    if args.device == "mps":
        torch.mps.empty_cache()

    # --- HF modern baselines ---
    for model_id in args.models:
        print(f"\n--- {model_id} ---")
        result = encode_hf_deltas(model_id, all_pairs, args.device)
        if result is None:
            summary[model_id] = {"skipped": True}
            continue
        deltas, n_params, lat = result
        mrrs, per_repo = run_leave_one_out(deltas, joint, repo_slice, model_id)
        summary[model_id] = {
            "params": n_params,
            "latency_ms_per_pair": lat,
            "aggregate_mrr10": float(np.mean(mrrs)),
            "per_repo": per_repo,
        }
        print(f"  aggregate MRR@10 = {np.mean(mrrs):.4f}  latency={lat:.1f} ms/pair")
        del deltas
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()

    # --- BoW ---
    print("\n--- BoW (AST tokens) ---")
    bow_deltas = bow_encode_deltas(all_pairs, cw_cfg["max_seq_len"], vocab_size=cw_cfg.get("vocab_size", 664))
    bow_mrrs, bow_per_repo = run_leave_one_out(bow_deltas, joint, repo_slice, "BoW")
    summary["BoW (AST)"] = {
        "params": 0,
        "latency_ms_per_pair": 0.0,
        "aggregate_mrr10": float(np.mean(bow_mrrs)),
        "per_repo": bow_per_repo,
    }
    print(f"  aggregate MRR@10 = {np.mean(bow_mrrs):.4f}")

    # --- Random baseline ---
    rng = np.random.default_rng(42)
    rand_deltas = torch.from_numpy(rng.standard_normal((len(all_pairs), 128), dtype=np.float32))
    rand_mrrs, _ = run_leave_one_out(rand_deltas, joint, repo_slice, "Random")
    summary["Random"] = {
        "params": 0,
        "latency_ms_per_pair": 0.0,
        "aggregate_mrr10": float(np.mean(rand_mrrs)),
    }
    print(f"\nRandom aggregate MRR@10 = {np.mean(rand_mrrs):.4f}")

    # --- Summary table ---
    print(f"\n=== Cross-repo MRR@10 (leave-one-repo-out, {len(all_pairs)} pairs, {len(repo_data)} repos) ===")
    print(f"{'Model':<48} {'Params':>12} {'Latency':>12} {'Aggregate':>10}")
    print("-" * 86)
    for name, r in summary.items():
        if r.get("skipped"):
            print(f"{name:<48} {'—':>12} {'—':>12} {'skipped':>10}")
            continue
        lat_str = f"{r.get('latency_ms_per_pair', 0.0):.1f} ms/pair"
        print(
            f"{name:<48} "
            f"{r.get('params', 0):>12,} "
            f"{lat_str:>12} "
            f"{r['aggregate_mrr10']:>10.4f}"
        )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "total_pairs": len(all_pairs),
                    "n_repos": len(repo_data),
                    "repos": list(repo_data.keys()),
                    "results": summary,
                },
                f,
                indent=2,
            )
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
