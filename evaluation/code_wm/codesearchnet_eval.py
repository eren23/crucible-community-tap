#!/usr/bin/env python3
"""External-benchmark retrieval eval on CodeSearchNet Python test split.

Protocol (code-to-code retrieval on a diverse external corpus):

1. Load ``code_search_net`` Python test split (22,176 functions from open-source
   Python projects, each annotated with its origin ``repository_name``).
2. Sample N functions as QUERIES and a disjoint M as GALLERY.
3. Encode every function with CodeWM (via ``model.encode()`` on AST-tokenized
   source) and, optionally, a BoW baseline built from the same AST tokens.
4. For each query, rank the gallery by cosine similarity and compute:
   * ``by_repo`` — relevance = gallery item is from the same repository as the
     query. Measures whether embeddings cluster by repo-level coding style /
     API usage. Weak semantic signal but fully reproducible, external, and
     model-agnostic.
5. Report MRR, Recall@{1,5,10} for both CodeWM and BoW.

This connects CodeWM's numbers to an external, standard dataset that any
reviewer can reproduce, closing the "all self-defined metrics on self-
constructed data" gap flagged in the Phase 5C audit.

Usage::

    python codesearchnet_eval.py \\
        --checkpoint ~/.crucible-hub/taps/crucible-community-tap/checkpoints/ema-frozen-15k-best.pt \\
        --num-query 500 --num-gallery 1500 \\
        --out /tmp/csn_ema_frozen_15k.json
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

# Shared checkpoint loader — see _shared.py. `load_codewm` is re-exported
# here so downstream scripts (cross_repo_eval.py, cross_repo_modern_compare.py,
# modern_baselines_compare.py) can continue to `from codesearchnet_eval import load_codewm`
# without a mass-rename.
_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from _shared import load_codewm, resolve_tap_root  # noqa: E402  (re-exported for historical callers)


def fetch_codesearchnet_samples(num_samples: int, seed: int = 42):
    """Fetch Python test split from CodeSearchNet, return list of dicts."""
    from datasets import load_dataset

    print("Loading code_search_net python test split...")
    ds = load_dataset("code_search_net", "python", split="test")
    print(f"  Dataset loaded: {len(ds)} functions")

    rng = np.random.default_rng(seed)
    total = len(ds)
    # Over-sample to allow dropping parse failures / short functions.
    target = min(num_samples * 3, total)
    indices = rng.choice(total, size=target, replace=False)

    samples = []
    for idx in indices:
        row = ds[int(idx)]
        code = row.get("func_code_string") or row.get("whole_func_string") or ""
        repo = row.get("repository_name") or "<unknown>"
        if not isinstance(code, str):
            continue
        code = code.strip()
        # Skip trivial/empty functions and huge ones
        if len(code) < 30 or len(code) > 20000:
            continue
        samples.append({
            "code": code,
            "repo": repo,
            "name": row.get("func_name", ""),
        })
        if len(samples) >= num_samples:
            break

    print(f"  Kept {len(samples)} usable functions (after length filter)")
    return samples


@torch.no_grad()
def encode_codewm(model, samples, max_seq_len, device, batch_size=32):
    tap_root = resolve_tap_root()
    collectors_dir = str(tap_root)
    if collectors_dir not in sys.path:
        sys.path.insert(0, collectors_dir)
    from collectors.ast_tokenizer import ast_tokenize  # type: ignore

    tokens = np.stack([ast_tokenize(s["code"], max_seq_len) for s in samples])
    n = tokens.shape[0]
    embeds = []
    for i in range(0, n, batch_size):
        batch = torch.from_numpy(tokens[i:i + batch_size].astype(np.int64)).to(device)
        z = model.encode(batch).cpu()
        embeds.append(z)
    return torch.cat(embeds, dim=0)


def encode_bow(samples, max_seq_len, vocab_size=664):
    tap_root = resolve_tap_root()
    collectors_dir = str(tap_root)
    if collectors_dir not in sys.path:
        sys.path.insert(0, collectors_dir)
    from collectors.ast_tokenizer import ast_tokenize  # type: ignore

    bows = np.zeros((len(samples), vocab_size), dtype=np.float32)
    for i, s in enumerate(samples):
        tokens = ast_tokenize(s["code"], max_seq_len)
        for t in tokens:
            if 0 <= t < vocab_size:
                bows[i, int(t)] += 1.0
        norm = np.linalg.norm(bows[i])
        if norm > 1e-8:
            bows[i] /= norm
    return torch.from_numpy(bows)


def compute_retrieval_metrics(
    query_embeds: torch.Tensor,
    gallery_embeds: torch.Tensor,
    query_repos: list[str],
    gallery_repos: list[str],
    k_max: int = 10,
) -> dict:
    """Compute MRR, Recall@k using 'same repository' as relevance."""
    qn = F.normalize(query_embeds, dim=-1)
    gn = F.normalize(gallery_embeds, dim=-1)
    sim = qn @ gn.T  # [Q, G]
    topk = sim.topk(k_max, dim=-1).indices.numpy()

    query_repos_arr = np.array(query_repos)
    gallery_repos_arr = np.array(gallery_repos)

    rel = gallery_repos_arr[topk] == query_repos_arr[:, None]  # [Q, k_max]

    n = len(query_repos)
    rr = np.zeros(n)
    for i in range(n):
        hits = np.where(rel[i])[0]
        rr[i] = 1.0 / (hits[0] + 1) if len(hits) > 0 else 0.0

    return {
        "mrr@10": float(rr.mean()),
        "recall@1": float(rel[:, :1].any(axis=1).mean()),
        "recall@5": float(rel[:, :5].any(axis=1).mean()),
        "recall@10": float(rel[:, :10].any(axis=1).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-query", type=int, default=500)
    parser.add_argument("--num-gallery", type=int, default=1500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    parser.add_argument("--out", default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    total = args.num_query + args.num_gallery

    print(f"Loading CodeWM from {args.checkpoint}...")
    model, cfg = load_codewm(args.checkpoint, args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  CodeWM: {n_params:,} params on {args.device}")

    samples = fetch_codesearchnet_samples(total, seed=args.seed)
    if len(samples) < total:
        print(f"WARNING: got {len(samples)} samples, wanted {total}. Adjusting splits.")
        args.num_query = max(1, len(samples) // 4)
        args.num_gallery = len(samples) - args.num_query

    # Disjoint query/gallery split
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(samples))
    query_idx = perm[: args.num_query]
    gallery_idx = perm[args.num_query : args.num_query + args.num_gallery]
    query_samples = [samples[i] for i in query_idx]
    gallery_samples = [samples[i] for i in gallery_idx]
    q_repos = [s["repo"] for s in query_samples]
    g_repos = [s["repo"] for s in gallery_samples]

    # Sanity: how many unique repos?
    n_q_repos = len(set(q_repos))
    n_g_repos = len(set(g_repos))
    n_overlap = len(set(q_repos) & set(g_repos))
    print(f"\nQuery set: {len(query_samples)} samples, {n_q_repos} unique repos")
    print(f"Gallery set: {len(gallery_samples)} samples, {n_g_repos} unique repos")
    print(f"Repos appearing in both query and gallery: {n_overlap}")
    if n_overlap == 0:
        print("WARNING: no overlap in repos — every query is a guaranteed miss.")

    # --- CodeWM ---
    print("\n--- CodeWM encoding ---")
    t0 = time.time()
    cw_q = encode_codewm(model, query_samples, cfg["max_seq_len"], args.device)
    cw_g = encode_codewm(model, gallery_samples, cfg["max_seq_len"], args.device)
    cw_time = time.time() - t0
    print(f"  Encoded in {cw_time:.1f}s ({cw_time * 1000 / len(samples):.1f} ms/sample)")
    cw_metrics = compute_retrieval_metrics(cw_q, cw_g, q_repos, g_repos)
    print(f"  CodeWM: {cw_metrics}")

    # --- BoW baseline ---
    print("\n--- BoW (AST tokens) ---")
    t0 = time.time()
    bow_q = encode_bow(query_samples, cfg["max_seq_len"], vocab_size=cfg.get("vocab_size", 664))
    bow_g = encode_bow(gallery_samples, cfg["max_seq_len"], vocab_size=cfg.get("vocab_size", 664))
    bow_time = time.time() - t0
    print(f"  Encoded in {bow_time:.1f}s")
    bow_metrics = compute_retrieval_metrics(bow_q, bow_g, q_repos, g_repos)
    print(f"  BoW: {bow_metrics}")

    # --- Random baseline ---
    rng2 = np.random.default_rng(args.seed + 1)
    rand_q = torch.from_numpy(rng2.standard_normal((len(query_samples), cfg["model_dim"]), dtype=np.float32))
    rand_g = torch.from_numpy(rng2.standard_normal((len(gallery_samples), cfg["model_dim"]), dtype=np.float32))
    rand_metrics = compute_retrieval_metrics(rand_q, rand_g, q_repos, g_repos)
    print(f"  Random: {rand_metrics}")

    result = {
        "checkpoint": args.checkpoint,
        "n_query": len(query_samples),
        "n_gallery": len(gallery_samples),
        "n_query_repos": n_q_repos,
        "n_gallery_repos": n_g_repos,
        "n_repo_overlap": n_overlap,
        "codewm": cw_metrics,
        "codewm_latency_ms_per_sample": cw_time * 1000 / len(samples),
        "codewm_params": n_params,
        "bow": bow_metrics,
        "random": rand_metrics,
        "dataset": "code_search_net:python:test",
        "seed": args.seed,
    }

    print("\n=== Summary ===")
    print(f"CodeWM   ({n_params:,} params): MRR@10={cw_metrics['mrr@10']:.4f}  R@1={cw_metrics['recall@1']:.4f}  R@10={cw_metrics['recall@10']:.4f}")
    print(f"BoW      (AST tokens)          : MRR@10={bow_metrics['mrr@10']:.4f}  R@1={bow_metrics['recall@1']:.4f}  R@10={bow_metrics['recall@10']:.4f}")
    print(f"Random   (normal)              : MRR@10={rand_metrics['mrr@10']:.4f}  R@1={rand_metrics['recall@1']:.4f}  R@10={rand_metrics['recall@10']:.4f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
