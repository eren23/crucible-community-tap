#!/usr/bin/env python3
"""Modern code embedding baselines vs CodeWM on CodeSearchNet retrieval.

Closes the "stale baselines" gap (CodeBERT 2020 only) from the Phase 5C
benchmark audit by running 2024-2026 code embedding models on the same
CodeSearchNet code-to-code retrieval protocol that ``codesearchnet_eval.py``
uses.

Models scored:

- **CodeWM** (ours, 1.1M) loaded from a checkpoint
- **CodeBERT** (microsoft/codebert-base, 124M, 2020)
- **CodeT5+** (Salesforce/codet5p-110m-embedding, ~110M, 2022)
- **jina-code-embeddings** (jinaai/jina-embeddings-v2-base-code, 161M, 2024)
- **BoW (AST tokens)** baseline
- **Random** baseline

All models embed the same 400 query / 1200 gallery split from the
``code_search_net:python:test`` shard. Relevance = same repository; metrics
= MRR@10 + Recall@{1,5,10}.

Any model that fails to download or instantiate is skipped with a warning
instead of failing the whole run.

Usage::

    python modern_baselines_compare.py \\
        --checkpoint ~/.crucible-hub/taps/crucible-community-tap/checkpoints/phase5/phase5-contrast-extreme-3k-best.pt \\
        --num-query 400 --num-gallery 1200 \\
        --out /tmp/modern_baselines.json
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

# Reuse the CodeSearchNet helpers from the sister script
THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))
from codesearchnet_eval import (  # type: ignore
    compute_retrieval_metrics,
    encode_bow,
    encode_codewm,
    fetch_codesearchnet_samples,
    load_codewm,
)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def encode_hf_model(
    model_id: str,
    samples: list[dict],
    device: str,
    max_length: int = 512,
    batch_size: int = 8,
    pooling: str = "mean",
):
    """Load a HF AutoModel, embed samples, return (embeddings, n_params, ms_per_sample).

    Returns ``None`` on any loading or forward failure so the overall run
    survives individual model issues.
    """
    try:
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except ImportError:
        print(f"  [skip {model_id}] transformers not installed")
        return None

    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
        model.train(False)  # inference mode
    except Exception as exc:
        print(f"  [skip {model_id}] load failed: {type(exc).__name__}: {exc}")
        return None

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {model_id}: {n_params:,} params")

    embeds: list[torch.Tensor] = []
    t0 = time.time()
    for i in range(0, len(samples), batch_size):
        batch = [s["code"] for s in samples[i:i + batch_size]]
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
            return None

        # Handle different output shapes.
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            lhs = out.last_hidden_state
            if pooling == "mean":
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (lhs * mask).sum(1) / mask.sum(1).clamp_min(1)
            elif pooling == "cls":
                pooled = lhs[:, 0]
            else:
                raise ValueError(f"Unknown pooling: {pooling}")
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            # Some models (e.g. sentence-transformers-wrapped) expose a
            # ``sentence_embedding`` or similar attribute. Try a fallback.
            for attr in ("sentence_embedding", "embeddings", "text_embeds"):
                tensor = getattr(out, attr, None)
                if tensor is not None:
                    pooled = tensor
                    break
            else:
                print(f"  [skip {model_id}] unknown output shape: {type(out).__name__}")
                return None

        embeds.append(pooled.detach().cpu())

    elapsed = time.time() - t0
    lat_ms_per_sample = elapsed * 1000 / len(samples)
    return torch.cat(embeds, dim=0), n_params, lat_ms_per_sample


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-query", type=int, default=400)
    parser.add_argument("--num-gallery", type=int, default=1200)
    parser.add_argument("--device", default=pick_device())
    parser.add_argument("--out", default="")
    parser.add_argument("--seed", type=int, default=42)
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

    total = args.num_query + args.num_gallery
    samples = fetch_codesearchnet_samples(total, seed=args.seed)
    if len(samples) < total:
        args.num_query = max(1, len(samples) // 4)
        args.num_gallery = len(samples) - args.num_query

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(samples))
    query_idx = perm[: args.num_query]
    gallery_idx = perm[args.num_query : args.num_query + args.num_gallery]
    query_samples = [samples[i] for i in query_idx]
    gallery_samples = [samples[i] for i in gallery_idx]
    q_repos = [s["repo"] for s in query_samples]
    g_repos = [s["repo"] for s in gallery_samples]

    n_q_repos = len(set(q_repos))
    n_g_repos = len(set(g_repos))
    n_overlap = len(set(q_repos) & set(g_repos))
    print(f"Query: {len(query_samples)} samples, {n_q_repos} repos")
    print(f"Gallery: {len(gallery_samples)} samples, {n_g_repos} repos")
    print(f"Overlapping repos: {n_overlap}")

    results: dict[str, dict] = {}

    # ---- CodeWM ----
    print(f"\n--- CodeWM ({args.checkpoint}) ---")
    cw_model, cw_cfg = load_codewm(args.checkpoint, args.device)
    cw_params = sum(p.numel() for p in cw_model.parameters())
    t0 = time.time()
    cw_q = encode_codewm(cw_model, query_samples, cw_cfg["max_seq_len"], args.device)
    cw_g = encode_codewm(cw_model, gallery_samples, cw_cfg["max_seq_len"], args.device)
    cw_lat = (time.time() - t0) * 1000 / len(samples)
    cw_metrics = compute_retrieval_metrics(cw_q, cw_g, q_repos, g_repos)
    results["CodeWM"] = {
        "params": cw_params,
        "latency_ms": cw_lat,
        "metrics": cw_metrics,
        "checkpoint": str(args.checkpoint),
    }
    print(f"  {cw_metrics}")
    del cw_model
    gc.collect()
    if args.device == "mps":
        torch.mps.empty_cache()

    # ---- HF modern baselines ----
    for model_id in args.models:
        print(f"\n--- {model_id} ---")
        result = encode_hf_model(
            model_id,
            samples=query_samples + gallery_samples,
            device=args.device,
        )
        if result is None:
            results[model_id] = {"skipped": True}
            continue
        all_embeds, n_params, lat_ms = result
        m_q = all_embeds[: len(query_samples)]
        m_g = all_embeds[len(query_samples):]
        metrics = compute_retrieval_metrics(m_q, m_g, q_repos, g_repos)
        results[model_id] = {
            "params": n_params,
            "latency_ms": lat_ms,
            "metrics": metrics,
        }
        print(f"  MRR@10={metrics['mrr@10']:.4f}  R@1={metrics['recall@1']:.4f}  R@10={metrics['recall@10']:.4f}  latency={lat_ms:.1f}ms")
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()

    # ---- BoW ----
    print("\n--- BoW (AST tokens) ---")
    bow_q = encode_bow(query_samples, cw_cfg["max_seq_len"], vocab_size=cw_cfg.get("vocab_size", 664))
    bow_g = encode_bow(gallery_samples, cw_cfg["max_seq_len"], vocab_size=cw_cfg.get("vocab_size", 664))
    bow_metrics = compute_retrieval_metrics(bow_q, bow_g, q_repos, g_repos)
    results["BoW (AST)"] = {
        "params": 0,
        "latency_ms": 0.0,
        "metrics": bow_metrics,
    }
    print(f"  {bow_metrics}")

    # ---- Random ----
    rng2 = np.random.default_rng(args.seed + 1)
    rand_q = torch.from_numpy(rng2.standard_normal((len(query_samples), 128), dtype=np.float32))
    rand_g = torch.from_numpy(rng2.standard_normal((len(gallery_samples), 128), dtype=np.float32))
    rand_metrics = compute_retrieval_metrics(rand_q, rand_g, q_repos, g_repos)
    results["Random"] = {
        "params": 0,
        "latency_ms": 0.0,
        "metrics": rand_metrics,
    }
    print(f"  Random: {rand_metrics}")

    # ---- Summary table ----
    print("\n=== Head-to-head on CodeSearchNet Python test (n_q=400, n_g=1200) ===")
    print(
        f"{'Model':<48} {'Params':>12} {'Latency':>10} {'MRR@10':>8} "
        f"{'R@1':>7} {'R@5':>7} {'R@10':>7}"
    )
    print("-" * 108)
    for name, r in results.items():
        if r.get("skipped"):
            print(f"{name:<48} {'—':>12} {'—':>10} {'skipped':>8}")
            continue
        m = r["metrics"]
        print(
            f"{name:<48} "
            f"{r['params']:>12,} "
            f"{r['latency_ms']:>9.1f}ms "
            f"{m['mrr@10']:>8.4f} "
            f"{m['recall@1']:>7.4f} "
            f"{m['recall@5']:>7.4f} "
            f"{m['recall@10']:>7.4f}"
        )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "n_query": len(query_samples),
                    "n_gallery": len(gallery_samples),
                    "n_query_repos": n_q_repos,
                    "n_gallery_repos": n_g_repos,
                    "n_repo_overlap": n_overlap,
                    "seed": args.seed,
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
