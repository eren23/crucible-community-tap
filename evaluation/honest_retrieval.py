#!/usr/bin/env python3
"""Honest retrieval evaluation that doesn't get fooled by dataset imbalance.

The existing semantic_eval.py reports MRR/Recall@k with "same edit_type" as the
relevance criterion. But CommitPack is 98.8% MODIFY, so this metric is trivially
saturated — random retrieval scores ~0.99.

This script reports retrieval quality under multiple relevance criteria:

1. edit_type only — the easy/baseline metric
2. joint (edit_type × scope) — 9 classes, harder
3. joint + location bucket — even harder
4. action_cos > 0.9 — strict: relevant = the edit action vector matches closely
5. action_cos > 0.95 — very strict

Also: train/gallery vs query split. The existing semantic_eval uses all-vs-all
with self exclusion, so near-duplicate trajectory edits inflate the metric.
This script uses a clean split.

Usage::

    python honest_retrieval.py \
        --checkpoint /workspace/parameter-golf/checkpoints/code_wm_best.pt \
        --data /path/to/data.h5 \
        --num-samples 5000
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F


def _load_code_wm_modules():
    tap_root = Path(__file__).parent.parent
    if not (tap_root / "architectures" / "wm_base" / "wm_base.py").exists():
        tap_root = Path("/workspace/crucible-community-tap")
    for mod_name, mod_path in [
        ("wm_base", tap_root / "architectures" / "wm_base" / "wm_base.py"),
        ("code_wm", tap_root / "architectures" / "code_wm" / "code_wm.py"),
    ]:
        spec = importlib.util.spec_from_file_location(mod_name, mod_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    import code_wm
    return code_wm


def load_model(checkpoint_path: str, device: str = "cpu"):
    os.environ.setdefault("WM_POOL_MODE", "cls")
    code_wm = _load_code_wm_modules()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = code_wm.CodeWorldModel(
        vocab_size=cfg["vocab_size"],
        max_seq_len=cfg["max_seq_len"],
        encoder_loops=cfg["encoder_loops"],
        model_dim=cfg["model_dim"],
        num_loops=cfg["num_loops"],
        num_heads=cfg["num_heads"],
        predictor_depth=2,
        ema_decay=cfg["ema_decay"],
        action_dim=cfg["action_dim"],
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.train(False)
    return model, cfg


def load_samples(data_path: str, num_query: int, num_gallery: int, seed: int = 42):
    """Load disjoint query and gallery sets."""
    np.random.seed(seed)
    f = h5py.File(data_path, "r")
    n_total = f["before_tokens"].shape[0]
    total_needed = num_query + num_gallery
    if total_needed > n_total:
        raise ValueError(f"Need {total_needed} samples, only have {n_total}")

    indices = np.random.choice(n_total, size=total_needed, replace=False)
    indices = np.sort(indices)
    query_idx = indices[:num_query]
    gallery_idx = indices[num_query:]

    def fetch(idx):
        idx_list = idx.tolist()
        before = f["before_tokens"][idx_list].astype(np.int64)
        after = f["after_tokens"][idx_list].astype(np.int64)
        actions = f["edit_actions"][idx_list].astype(np.float32)
        return before, after, actions

    qb, qa, qact = fetch(query_idx)
    gb, ga, gact = fetch(gallery_idx)
    f.close()
    return (qb, qa, qact), (gb, ga, gact)


@torch.no_grad()
def encode_deltas(model, before, after, device, batch_size=64):
    """Compute target embeddings for before/after and the delta between them."""
    n = before.shape[0]
    z_before = []
    z_after = []
    for i in range(0, n, batch_size):
        b = torch.from_numpy(before[i:i+batch_size]).to(device)
        a = torch.from_numpy(after[i:i+batch_size]).to(device)
        z_before.append(model.encode(b).cpu())
        z_after.append(model.encode(a).cpu())
    z_before = torch.cat(z_before, dim=0)
    z_after = torch.cat(z_after, dim=0)
    delta = z_after - z_before
    return z_before, z_after, delta


def compute_retrieval_metrics(
    query_delta, query_actions, query_labels,
    gallery_delta, gallery_actions, gallery_labels,
    k_max=10,
):
    """Compute MRR and Recall@k under multiple relevance criteria.

    Returns dict with metrics for:
    - by_edit_type: relevance = same edit_type
    - by_joint: relevance = same (edit_type, scope) joint label
    - by_action_cos_0.9: relevance = action vector cos > 0.9
    - by_action_cos_0.95: relevance = action vector cos > 0.95
    """
    qd_n = F.normalize(query_delta, dim=-1)
    gd_n = F.normalize(gallery_delta, dim=-1)

    # Cosine similarity matrix [Q, G]
    sim = qd_n @ gd_n.T
    topk_idx = sim.topk(k_max, dim=-1).indices.numpy()  # [Q, k_max]

    n_query = len(query_actions)

    # Compute action cosines for action-based relevance
    qa_n = F.normalize(query_actions, dim=-1)
    ga_n = F.normalize(gallery_actions, dim=-1)
    action_sim_full = qa_n @ ga_n.T  # [Q, G]

    results = {}

    # Method 1: by edit_type
    q_et = query_labels[:, 0]  # edit_type column
    g_et = gallery_labels[:, 0]
    neighbor_et = g_et[topk_idx]
    relevant_et = (neighbor_et == q_et[:, None])
    results["by_edit_type"] = _summary(relevant_et, n_query, k_max)

    # Method 2: by joint label (edit_type * 3 + scope)
    q_joint = query_labels[:, 0] * 3 + query_labels[:, 1]
    g_joint = gallery_labels[:, 0] * 3 + gallery_labels[:, 1]
    neighbor_joint = g_joint[topk_idx]
    relevant_joint = (neighbor_joint == q_joint[:, None])
    results["by_joint"] = _summary(relevant_joint, n_query, k_max)

    # Method 3: action cosine > 0.9
    relevant_a09 = np.zeros_like(topk_idx, dtype=bool)
    for i in range(n_query):
        for j_pos, j in enumerate(topk_idx[i]):
            relevant_a09[i, j_pos] = action_sim_full[i, j].item() > 0.9
    results["by_action_cos_0.9"] = _summary(relevant_a09, n_query, k_max)

    # Method 4: action cosine > 0.95
    relevant_a095 = np.zeros_like(topk_idx, dtype=bool)
    for i in range(n_query):
        for j_pos, j in enumerate(topk_idx[i]):
            relevant_a095[i, j_pos] = action_sim_full[i, j].item() > 0.95
    results["by_action_cos_0.95"] = _summary(relevant_a095, n_query, k_max)

    return results, topk_idx, sim


def _summary(relevant, n_query, k_max):
    """Compute MRR + Recall@{1,5,10} from a [Q, k_max] boolean array."""
    rr_list = []
    for i in range(n_query):
        hits = np.where(relevant[i])[0]
        rr_list.append(1.0 / (hits[0] + 1) if len(hits) > 0 else 0.0)
    return {
        "mrr": float(np.mean(rr_list)),
        "recall@1": float(relevant[:, :1].any(axis=1).mean()),
        "recall@5": float(relevant[:, :5].any(axis=1).mean()),
        "recall@10": float(relevant[:, :10].any(axis=1).mean()),
        "n_relevant_in_top10_mean": float(relevant.sum(axis=1).mean()),
    }


def random_baseline(query_actions, gallery_actions, n_query, k_max=10, seed=42):
    """Random retrieval baseline. Picks k random gallery items per query."""
    rng = np.random.default_rng(seed)
    n_gallery = len(gallery_actions)
    rand_idx = rng.integers(0, n_gallery, size=(n_query, k_max))

    qa_n = F.normalize(query_actions, dim=-1)
    ga_n = F.normalize(gallery_actions, dim=-1)
    action_sim = qa_n @ ga_n.T

    # by joint
    q_joint = ((query_actions[:, :3].argmax(axis=1) * 3) +
               query_actions[:, 3:6].argmax(axis=1)).numpy()
    g_joint = ((gallery_actions[:, :3].argmax(axis=1) * 3) +
               gallery_actions[:, 3:6].argmax(axis=1)).numpy()
    neighbor_joint = g_joint[rand_idx]
    relevant_joint = (neighbor_joint == q_joint[:, None])

    # by edit_type
    q_et = query_actions[:, :3].argmax(axis=1).numpy()
    g_et = gallery_actions[:, :3].argmax(axis=1).numpy()
    neighbor_et = g_et[rand_idx]
    relevant_et = (neighbor_et == q_et[:, None])

    # by action_cos > 0.9
    relevant_a09 = np.zeros_like(rand_idx, dtype=bool)
    relevant_a095 = np.zeros_like(rand_idx, dtype=bool)
    for i in range(n_query):
        for j_pos, j in enumerate(rand_idx[i]):
            cos = action_sim[i, j].item()
            relevant_a09[i, j_pos] = cos > 0.9
            relevant_a095[i, j_pos] = cos > 0.95

    return {
        "by_edit_type": _summary(relevant_et, n_query, k_max),
        "by_joint": _summary(relevant_joint, n_query, k_max),
        "by_action_cos_0.9": _summary(relevant_a09, n_query, k_max),
        "by_action_cos_0.95": _summary(relevant_a095, n_query, k_max),
    }


def bow_baseline(query_before, query_after, gallery_before, gallery_after,
                 query_actions, gallery_actions, vocab_size, k_max=10):
    """BoW-based retrieval: bag of token counts → delta → cosine."""

    def bag(tokens):
        n = tokens.shape[0]
        bows = np.zeros((n, vocab_size), dtype=np.float32)
        for i in range(n):
            for t in tokens[i]:
                if 0 <= t < vocab_size:
                    bows[i, t] += 1.0
        norms = np.linalg.norm(bows, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return bows / norms

    qb_bow = bag(query_before)
    qa_bow = bag(query_after)
    gb_bow = bag(gallery_before)
    ga_bow = bag(gallery_after)

    q_delta = qa_bow - qb_bow
    g_delta = ga_bow - gb_bow

    # Normalize and cosine similarity
    q_dn = q_delta / np.maximum(np.linalg.norm(q_delta, axis=1, keepdims=True), 1e-8)
    g_dn = g_delta / np.maximum(np.linalg.norm(g_delta, axis=1, keepdims=True), 1e-8)

    sim = q_dn @ g_dn.T  # [Q, G]
    topk_idx = np.argsort(-sim, axis=1)[:, :k_max]

    n_query = len(query_actions)

    qa_n = F.normalize(torch.from_numpy(query_actions), dim=-1)
    ga_n = F.normalize(torch.from_numpy(gallery_actions), dim=-1)
    action_sim = (qa_n @ ga_n.T).numpy()

    q_et = query_actions[:, :3].argmax(axis=1)
    g_et = gallery_actions[:, :3].argmax(axis=1)
    q_joint = q_et * 3 + query_actions[:, 3:6].argmax(axis=1)
    g_joint = g_et * 3 + gallery_actions[:, 3:6].argmax(axis=1)

    neighbor_et = g_et[topk_idx]
    neighbor_joint = g_joint[topk_idx]
    relevant_et = (neighbor_et == q_et[:, None])
    relevant_joint = (neighbor_joint == q_joint[:, None])

    relevant_a09 = np.zeros_like(topk_idx, dtype=bool)
    relevant_a095 = np.zeros_like(topk_idx, dtype=bool)
    for i in range(n_query):
        for j_pos, j in enumerate(topk_idx[i]):
            cos = action_sim[i, j]
            relevant_a09[i, j_pos] = cos > 0.9
            relevant_a095[i, j_pos] = cos > 0.95

    return {
        "by_edit_type": _summary(relevant_et, n_query, k_max),
        "by_joint": _summary(relevant_joint, n_query, k_max),
        "by_action_cos_0.9": _summary(relevant_a09, n_query, k_max),
        "by_action_cos_0.95": _summary(relevant_a095, n_query, k_max),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--num-query", type=int, default=1000)
    parser.add_argument("--num-gallery", type=int, default=4000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model, cfg = load_model(args.checkpoint, args.device)
    print(f"  Model: dim={cfg['model_dim']}, ema_decay={cfg['ema_decay']}, "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    print(f"Loading {args.num_query} query + {args.num_gallery} gallery samples...")
    (qb, qa, qact), (gb, ga, gact) = load_samples(
        args.data, args.num_query, args.num_gallery, args.seed,
    )

    print("Encoding query set...")
    t0 = time.time()
    q_z_before, q_z_after, q_delta = encode_deltas(model, qb, qa, args.device)
    print(f"  Encoded {len(qb)} query samples in {time.time()-t0:.1f}s")

    print("Encoding gallery set...")
    t0 = time.time()
    g_z_before, g_z_after, g_delta = encode_deltas(model, gb, ga, args.device)
    print(f"  Encoded {len(gb)} gallery samples in {time.time()-t0:.1f}s")

    # Action labels for relevance
    q_labels = np.stack([
        qact[:, :3].argmax(axis=1),
        qact[:, 3:6].argmax(axis=1),
    ], axis=1)
    g_labels = np.stack([
        gact[:, :3].argmax(axis=1),
        gact[:, 3:6].argmax(axis=1),
    ], axis=1)

    qact_t = torch.from_numpy(qact)
    gact_t = torch.from_numpy(gact)

    print("\n" + "=" * 70)
    print("CodeWM (delta-NN retrieval)")
    print("=" * 70)
    t0 = time.time()
    cw_results, topk_idx, sim = compute_retrieval_metrics(
        q_delta, qact_t, q_labels,
        g_delta, gact_t, g_labels,
    )
    print(f"  Retrieval computed in {time.time()-t0:.1f}s")
    for criterion, metrics in cw_results.items():
        print(f"  {criterion}:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")

    print("\n" + "=" * 70)
    print("BoW baseline (delta-NN on bag-of-AST-tokens)")
    print("=" * 70)
    t0 = time.time()
    bow_results = bow_baseline(qb, qa, gb, ga, qact, gact, vocab_size=cfg["vocab_size"])
    print(f"  Computed in {time.time()-t0:.1f}s")
    for criterion, metrics in bow_results.items():
        print(f"  {criterion}:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")

    print("\n" + "=" * 70)
    print("Random retrieval baseline")
    print("=" * 70)
    rand_results = random_baseline(qact_t, gact_t, len(qact), seed=args.seed)
    for criterion, metrics in rand_results.items():
        print(f"  {criterion}:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")

    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD: MRR by relevance criterion")
    print("=" * 70)
    print(f"{'Criterion':<25} {'CodeWM':<10} {'BoW':<10} {'Random':<10} {'CW lift':<10}")
    print("-" * 70)
    for criterion in ["by_edit_type", "by_joint", "by_action_cos_0.9", "by_action_cos_0.95"]:
        cw = cw_results[criterion]["mrr"]
        bow = bow_results[criterion]["mrr"]
        rand = rand_results[criterion]["mrr"]
        lift = cw - bow
        print(f"{criterion:<25} {cw:<10.4f} {bow:<10.4f} {rand:<10.4f} {lift:+.4f}")

    print("\nRecall@1 head-to-head:")
    print(f"{'Criterion':<25} {'CodeWM':<10} {'BoW':<10} {'Random':<10} {'CW lift':<10}")
    print("-" * 70)
    for criterion in ["by_edit_type", "by_joint", "by_action_cos_0.9", "by_action_cos_0.95"]:
        cw = cw_results[criterion]["recall@1"]
        bow = bow_results[criterion]["recall@1"]
        rand = rand_results[criterion]["recall@1"]
        lift = cw - bow
        print(f"{criterion:<25} {cw:<10.4f} {bow:<10.4f} {rand:<10.4f} {lift:+.4f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "checkpoint": args.checkpoint,
                "config": cfg,
                "num_query": args.num_query,
                "num_gallery": args.num_gallery,
                "codewm": cw_results,
                "bow": bow_results,
                "random": rand_results,
            }, f, indent=2)
        print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
