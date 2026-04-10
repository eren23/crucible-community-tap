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


def class_prior_chance_mrr10(
    joint: np.ndarray,
    repo_slice: dict[str, tuple[int, int]],
) -> tuple[float, dict[str, float]]:
    """Analytic expected MRR@10 under a uniform-random ranking given the joint-label
    distribution in the leave-one-repo-out gallery.

    For each query q in repo r with label L, let p = fraction of gallery items
    (i.e. all pairs NOT in repo r) with label L. Under a uniform-random ranking
    the position of the first same-label gallery item follows a geometric-ish
    distribution; for the top-k slice we use the exact formula

        E[1/rank | first-hit <= k] * P(first-hit <= k)
            = sum_{j=1..k} (1-p)^{j-1} * p / j

    and 0 if the query's label is absent from the gallery. Averaging over all
    queries gives an interpretable "label-collision chance" number that
    explains why 0.65 is not a model-quality signal but a property of the
    coarse 9-class by_joint metric.

    Returns (aggregate_mean_of_per_repo_mrr, per_repo_mrr).
    """
    per_repo: dict[str, float] = {}
    n_total = len(joint)
    k = 10
    for repo, (start, end) in repo_slice.items():
        q_labels = joint[start:end]
        # Gallery is everything NOT in this repo's slice.
        g_mask = np.ones(n_total, dtype=bool)
        g_mask[start:end] = False
        g_labels = joint[g_mask]
        g_n = len(g_labels)
        if g_n == 0 or len(q_labels) == 0:
            per_repo[repo] = 0.0
            continue
        # Per-class hit rate in the gallery.
        unique, counts = np.unique(g_labels, return_counts=True)
        prior = dict(zip(unique.tolist(), (counts / g_n).tolist()))
        mrrs = np.zeros(len(q_labels))
        for i, L in enumerate(q_labels):
            p = prior.get(int(L), 0.0)
            if p <= 0:
                mrrs[i] = 0.0
                continue
            # Sum p*(1-p)^(j-1)/j for j=1..k.
            total = 0.0
            one_minus_p = 1.0 - p
            factor = p
            for j in range(1, k + 1):
                total += factor / j
                factor *= one_minus_p
            mrrs[i] = total
        per_repo[repo] = float(mrrs.mean())
    agg = float(np.mean(list(per_repo.values()))) if per_repo else 0.0
    return agg, per_repo


def run_hard_negative_rerank(
    deltas: torch.Tensor,
    joint: np.ndarray,
    repo_slice: dict[str, tuple[int, int]],
    k_distractors: int = 9,
    seed: int = 42,
) -> tuple[float, dict[str, float]]:
    """Hard-negative reranking: for each query, build a per-query gallery of
    1 true positive (another pair in the SAME held-out repo with the SAME joint
    label) plus K distractors (drawn at random from the leave-one-repo-out
    gallery with a DIFFERENT joint label). Rank by cosine similarity, compute
    MRR@K+1. Expected random ≈ 1/(K+1).

    Returns (aggregate_mean_of_per_repo_mrr, per_repo_mrr).
    Queries with no matching positive in their repo are skipped.
    """
    rng = np.random.default_rng(seed)
    dn = F.normalize(deltas, dim=-1).numpy()
    per_repo: dict[str, float] = {}
    n_total = len(joint)

    for repo, (start, end) in repo_slice.items():
        q_idxs = np.arange(start, end)
        q_labels = joint[start:end]
        g_mask = np.ones(n_total, dtype=bool)
        g_mask[start:end] = False
        g_idxs_all = np.where(g_mask)[0]
        g_labels_all = joint[g_idxs_all]

        rrs: list[float] = []
        for qi, L in zip(q_idxs, q_labels):
            # Positive: another pair in the SAME repo with the same joint label.
            in_repo = np.arange(start, end)
            in_repo_labels = joint[in_repo]
            pos_candidates = in_repo[(in_repo_labels == L) & (in_repo != qi)]
            if len(pos_candidates) == 0:
                continue
            pos = int(rng.choice(pos_candidates))

            # Distractors: different joint label, drawn from the leave-one-repo-out gallery.
            neg_pool = g_idxs_all[g_labels_all != L]
            if len(neg_pool) < k_distractors:
                continue
            negs = rng.choice(neg_pool, size=k_distractors, replace=False)

            cand = np.concatenate([[pos], negs])
            q_vec = dn[qi]
            sims = dn[cand] @ q_vec
            order = np.argsort(-sims)
            rank_of_pos = int(np.where(order == 0)[0][0]) + 1
            rrs.append(1.0 / rank_of_pos)

        per_repo[repo] = float(np.mean(rrs)) if rrs else 0.0

    agg = float(np.mean(list(per_repo.values()))) if per_repo else 0.0
    return agg, per_repo


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
    # Keep a copy of cw_deltas for the hard-negative rerank; free the model only.
    cw_deltas_backup = cw_deltas.clone()
    del cw_model, cw_deltas
    gc.collect()
    if args.device == "mps":
        torch.mps.empty_cache()

    # --- HF modern baselines ---
    # Keep deltas alive for the hard-negative rerank pass below; None if loading failed.
    hf_deltas_backup: dict[str, torch.Tensor | None] = {}
    for model_id in args.models:
        print(f"\n--- {model_id} ---")
        result = encode_hf_deltas(model_id, all_pairs, args.device)
        if result is None:
            summary[model_id] = {"skipped": True}
            hf_deltas_backup[model_id] = None
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
        hf_deltas_backup[model_id] = deltas.clone()
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

    # --- Random baseline (uniform Gaussian features; still ranked via cosine) ---
    rng = np.random.default_rng(42)
    rand_deltas = torch.from_numpy(rng.standard_normal((len(all_pairs), 128), dtype=np.float32))
    rand_mrrs, _ = run_leave_one_out(rand_deltas, joint, repo_slice, "Random (Gaussian)")
    summary["Random (Gaussian)"] = {
        "params": 0,
        "latency_ms_per_pair": 0.0,
        "aggregate_mrr10": float(np.mean(rand_mrrs)),
    }
    print(f"\nRandom (Gaussian) aggregate MRR@10 = {np.mean(rand_mrrs):.4f}")

    # --- Class-prior analytic chance (Track A) ---
    # The Gaussian random above produces a suspiciously high MRR@10 not because
    # the model peeks at labels but because the by_joint label has only 9
    # classes and one or two of them dominate the gallery. This computes the
    # exact expected MRR@10 under a uniform-random ranking given the observed
    # joint-label distribution. It should reproduce the Gaussian random number
    # within a few percent.
    cp_agg, cp_per_repo = class_prior_chance_mrr10(joint, repo_slice)
    summary["Class-prior chance"] = {
        "params": 0,
        "latency_ms_per_pair": 0.0,
        "aggregate_mrr10": cp_agg,
        "per_repo": cp_per_repo,
        "note": "Analytic chance under uniform-random ranking with the observed by_joint distribution.",
    }
    print(f"Class-prior chance aggregate MRR@10 = {cp_agg:.4f}")

    # --- Hard-negative reranking (Track A) ---
    # Build a per-query mini-gallery of 1 same-label positive + K wrong-label
    # distractors and compute MRR@(K+1) for each model. Expected random ≈ 1/(K+1).
    # This converts the coarse 9-class by_joint metric into a clean 10-way
    # discrimination task where the random floor is interpretable.
    K = 9
    hn_random_agg, _ = run_hard_negative_rerank(rand_deltas, joint, repo_slice, k_distractors=K)
    hn_cw_agg, _ = run_hard_negative_rerank(cw_deltas_backup, joint, repo_slice, k_distractors=K)
    hn_bow_agg, _ = run_hard_negative_rerank(bow_deltas, joint, repo_slice, k_distractors=K)
    summary[f"Hard-neg K={K} Random"] = {
        "params": 0,
        "aggregate_mrr10": hn_random_agg,
        "note": f"1 same-label positive + {K} wrong-label distractors per query.",
    }
    summary[f"Hard-neg K={K} CodeWM"] = {
        "params": cw_params,
        "aggregate_mrr10": hn_cw_agg,
    }
    summary[f"Hard-neg K={K} BoW"] = {
        "params": 0,
        "aggregate_mrr10": hn_bow_agg,
    }
    # HF baselines hard-negative numbers — run for each successfully-loaded baseline.
    for model_id, hf_deltas in hf_deltas_backup.items():
        if hf_deltas is None:
            continue
        hn_agg, _ = run_hard_negative_rerank(hf_deltas, joint, repo_slice, k_distractors=K)
        summary[f"Hard-neg K={K} {model_id}"] = {
            "params": summary[model_id].get("params", 0),
            "aggregate_mrr10": hn_agg,
        }
    print(f"\nHard-negative K={K} reranking (1 pos + {K} wrong-label distractors; expected random ≈ {1/(K+1):.4f}):")
    print(f"  Random  = {hn_random_agg:.4f}")
    print(f"  CodeWM  = {hn_cw_agg:.4f}")
    print(f"  BoW     = {hn_bow_agg:.4f}")
    for model_id in hf_deltas_backup:
        if hf_deltas_backup[model_id] is None:
            continue
        print(f"  {model_id:<40} = {summary[f'Hard-neg K={K} {model_id}']['aggregate_mrr10']:.4f}")

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
