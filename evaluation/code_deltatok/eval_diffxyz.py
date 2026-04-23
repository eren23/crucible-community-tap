#!/usr/bin/env python3
"""CodeDeltaTok evaluation on Diff-XYZ public benchmark.

Downloads Diff-XYZ test set from HuggingFace, encodes with UniXcoder,
runs CodeDeltaTok, and evaluates cross-modal retrieval.

This anchors the paper's claims to a public benchmark.

Usage:
    python eval_diffxyz.py \
        --checkpoint checkpoints/code_deltatok_final.pt \
        --model microsoft/unixcoder-base \
        --device cuda
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def encode_texts(model, tokenizer, texts, device, batch_size=32, max_length=512):
    """Encode texts to CLS features using a HuggingFace model."""
    all_feats = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :]
        all_feats.append(cls.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def retrieval_metrics(query, gallery, ks=(1, 5, 10)):
    """Cross-modal retrieval. query[i] matches gallery[i]."""
    q_n = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
    g_n = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-8)
    sim = q_n @ g_n.T
    nq = sim.shape[0]

    ranks = []
    for i in range(nq):
        rank = int((sim[i] > sim[i, i]).sum()) + 1
        ranks.append(rank)
    ranks = np.array(ranks, dtype=np.float64)

    results = {}
    for k in ks:
        results[f"R@{k}"] = float((ranks <= k).mean())
    results["MRR"] = float((1.0 / ranks).mean())

    # Bootstrap CI
    rr = 1.0 / ranks
    boot = [np.random.choice(rr, size=len(rr), replace=True).mean() for _ in range(1000)]
    results["MRR_ci_lo"] = float(np.percentile(boot, 2.5))
    results["MRR_ci_hi"] = float(np.percentile(boot, 97.5))
    return results


def main():
    parser = argparse.ArgumentParser(description="CodeDeltaTok on Diff-XYZ")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="microsoft/unixcoder-base")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load Diff-XYZ
    print("Loading Diff-XYZ test set...")
    from datasets import load_dataset
    ds = load_dataset("JetBrains-Research/diff-xyz", split="test", streaming=True)

    old_codes, new_codes, langs, kinds = [], [], [], []
    for sample in ds:
        if len(old_codes) >= args.max_samples:
            break
        old_codes.append(sample["old_code"])
        new_codes.append(sample["new_code"])
        langs.append(sample["lang"])
        kinds.append(sample["change_kind"])

    N = len(old_codes)
    print(f"  {N} samples, langs={set(langs)}")

    # Encode with UniXcoder
    print(f"\nEncoding with {args.model}...")
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    backbone = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    backbone = backbone.to(args.device)
    backbone.requires_grad_(False)

    t0 = time.time()
    before_feat = encode_texts(backbone, tokenizer, old_codes, args.device)
    after_feat = encode_texts(backbone, tokenizer, new_codes, args.device)
    D = before_feat.shape[1]
    print(f"  Encoded {N} pairs in {time.time()-t0:.1f}s ({D}-dim)")

    # Filter zero-norm deltas
    deltas = after_feat - before_feat
    valid = np.linalg.norm(deltas, axis=1) > 1e-6
    before_feat = before_feat[valid]
    after_feat = after_feat[valid]
    langs_valid = [l for l, v in zip(langs, valid) if v]
    kinds_valid = [k for k, v in zip(kinds, valid) if v]
    n = len(before_feat)
    print(f"  {n} valid after filtering")

    # Load DeltaTok
    print(f"\nLoading DeltaTok from {args.checkpoint}...")
    from architectures.code_deltatok.code_deltatok import CodeDeltaTok
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    num_blocks = sum(1 for k in sd if k.startswith("encoder.") and k.endswith(".norm1.weight"))
    num_tokens = sd["z_embed"].shape[1]
    feature_dim = sd["z_embed"].shape[2]

    model = CodeDeltaTok(feature_dim=feature_dim, num_blocks=num_blocks, num_delta_tokens=num_tokens)
    model.load_state_dict(sd)
    model.requires_grad_(False)
    model = model.to(args.device)

    # Encode deltas
    prev_t = torch.from_numpy(before_feat.astype(np.float32)).to(args.device)
    next_t = torch.from_numpy(after_feat.astype(np.float32)).to(args.device)

    all_z = []
    bs = 64
    for i in range(0, n, bs):
        e = min(i + bs, n)
        with torch.no_grad():
            dt = model.encode(prev_t[i:e], next_t[i:e])
        z = dt[:, 0].cpu().numpy() if dt.shape[1] == 1 else dt.mean(dim=1).cpu().numpy()
        all_z.append(z)
    delta_reps = np.concatenate(all_z)

    raw_delta = after_feat - before_feat

    # ---- Overall eval ----
    print(f"\n=== Overall ({n} samples) ===")
    dt_r = retrieval_metrics(delta_reps, after_feat)
    rd_r = retrieval_metrics(raw_delta, after_feat)
    print(f"  DeltaTok:  MRR={dt_r['MRR']:.4f} [{dt_r['MRR_ci_lo']:.4f},{dt_r['MRR_ci_hi']:.4f}] R@1={dt_r['R@1']:.4f} R@10={dt_r['R@10']:.4f}")
    print(f"  Raw delta: MRR={rd_r['MRR']:.4f} [{rd_r['MRR_ci_lo']:.4f},{rd_r['MRR_ci_hi']:.4f}] R@1={rd_r['R@1']:.4f} R@10={rd_r['R@10']:.4f}")

    # ---- Per-language eval ----
    print(f"\n=== Per Language ===")
    unique_langs = sorted(set(langs_valid))
    for lang in unique_langs:
        mask = np.array([l == lang for l in langs_valid])
        if mask.sum() < 10:
            continue
        dt_lang = retrieval_metrics(delta_reps[mask], after_feat[mask])
        rd_lang = retrieval_metrics(raw_delta[mask], after_feat[mask])
        print(f"  {lang:<12} DeltaTok MRR={dt_lang['MRR']:.4f} R@1={dt_lang['R@1']:.4f} | Raw MRR={rd_lang['MRR']:.4f} R@1={rd_lang['R@1']:.4f} (n={mask.sum()})")

    # ---- Per change-kind eval ----
    print(f"\n=== Per Change Kind ===")
    unique_kinds = sorted(set(kinds_valid))
    for kind in unique_kinds:
        mask = np.array([k == kind for k in kinds_valid])
        if mask.sum() < 10:
            continue
        dt_kind = retrieval_metrics(delta_reps[mask], after_feat[mask])
        rd_kind = retrieval_metrics(raw_delta[mask], after_feat[mask])
        print(f"  {kind:<12} DeltaTok MRR={dt_kind['MRR']:.4f} R@1={dt_kind['R@1']:.4f} | Raw MRR={rd_kind['MRR']:.4f} R@1={rd_kind['R@1']:.4f} (n={mask.sum()})")

    print("\nDone.")


if __name__ == "__main__":
    main()
