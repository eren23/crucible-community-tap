#!/usr/bin/env python3
"""Evaluate CodeWM vs CodeBERT vs BoW on CommitChronicle subset_llm.

JetBrains-Research/commit-chronicle subset_llm has 4,030 commits with per-file
diffs and reference commit messages. The retrieval task: given a (before, after)
code edit, retrieve the most similar edit by delta-NN.

Relevance criterion: same edit_type AND same scope (joint label).

Usage::

    python commit_chronicle_eval.py \\
        --checkpoint /workspace/parameter-golf/checkpoints/code_wm_best.pt \\
        --num-query 500 --num-gallery 1500
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


def load_codewm(checkpoint_path: str, device: str = "cpu"):
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


def reconstruct_before_after_from_diff(diff_text: str) -> tuple[str, str] | None:
    """Reconstruct before/after code from a unified diff string.

    CommitChronicle stores diffs as unified-diff text. We parse the hunks and
    apply them to reconstruct both versions.
    """
    if not diff_text or not diff_text.strip():
        return None
    before_lines = []
    after_lines = []
    for line in diff_text.split("\n"):
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("@@"):
            continue
        if line.startswith("-"):
            before_lines.append(line[1:])
        elif line.startswith("+"):
            after_lines.append(line[1:])
        elif line.startswith(" ") or line == "":
            # Context line — include in both
            before_lines.append(line[1:] if line.startswith(" ") else line)
            after_lines.append(line[1:] if line.startswith(" ") else line)

    before = "\n".join(before_lines)
    after = "\n".join(after_lines)
    if not before.strip() and not after.strip():
        return None
    return before, after


def fetch_commit_chronicle_samples(num_samples: int, max_file_size: int = 50_000):
    """Fetch samples from JetBrains-Research/commit-chronicle subset_llm."""
    from datasets import load_dataset

    print("Loading commit-chronicle subset_llm...")
    try:
        ds = load_dataset(
            "JetBrains-Research/commit-chronicle",
            "subset_llm",
            split="test",
            trust_remote_code=True,
        )
        print(f"  Dataset loaded: {len(ds)} commits")
    except Exception as e:
        print(f"  ERROR loading: {e}")
        # Try fallback
        try:
            ds = load_dataset(
                "JetBrains-Research/commit-chronicle",
                "default",
                split="test[:5000]",
                trust_remote_code=True,
            )
            print(f"  Fallback: {len(ds)} commits")
        except Exception as e2:
            print(f"  Both failed: {e2}")
            return []

    samples = []
    for record in ds:
        if len(samples) >= num_samples:
            break
        # commit-chronicle has "mods" list with per-file changes
        mods = record.get("mods", [])
        for mod in mods:
            if len(samples) >= num_samples:
                break
            # Different schemas: try multiple field names
            old_content = mod.get("old_path") or mod.get("old_content") or ""
            new_content = mod.get("new_path") or mod.get("new_content") or ""
            diff = mod.get("diff", "")
            change_type = mod.get("change_type", "")

            # If we have old_content and new_content directly, use them
            if isinstance(old_content, str) and isinstance(new_content, str):
                if old_content.strip() or new_content.strip():
                    if len(old_content) > max_file_size or len(new_content) > max_file_size:
                        continue
                    samples.append({
                        "old": old_content,
                        "new": new_content,
                        "change_type": change_type,
                    })
                    continue

            # Otherwise reconstruct from diff
            if diff:
                result = reconstruct_before_after_from_diff(diff)
                if result is None:
                    continue
                old, new = result
                if len(old) > max_file_size or len(new) > max_file_size:
                    continue
                samples.append({"old": old, "new": new, "change_type": change_type})

    print(f"  Extracted {len(samples)} usable file edits")
    return samples


def compute_actions(samples):
    sys.path.insert(0, "/workspace/crucible-community-tap")
    from collectors.commitpack_processor import compute_action
    return np.stack([compute_action(s["old"], s["new"]) for s in samples])


@torch.no_grad()
def encode_codewm(model, samples, max_seq_len, device, batch_size=32):
    sys.path.insert(0, "/workspace/crucible-community-tap")
    from collectors.ast_tokenizer import ast_tokenize

    n = len(samples)
    befores = np.stack([ast_tokenize(s["old"], max_seq_len) for s in samples])
    afters = np.stack([ast_tokenize(s["new"], max_seq_len) for s in samples])

    z_b, z_a = [], []
    for i in range(0, n, batch_size):
        b = torch.from_numpy(befores[i:i+batch_size].astype(np.int64)).to(device)
        a = torch.from_numpy(afters[i:i+batch_size].astype(np.int64)).to(device)
        z_b.append(model.encode(b).cpu())
        z_a.append(model.encode(a).cpu())
    return torch.cat(z_b), torch.cat(z_a)


@torch.no_grad()
def encode_codebert(tok, model, samples, device, batch_size=8, max_length=512):
    n = len(samples)
    z_b, z_a = [], []
    for i in range(0, n, batch_size):
        batch = samples[i:i+batch_size]
        olds = [s["old"][:max_length*4] for s in batch]
        news = [s["new"][:max_length*4] for s in batch]
        for texts, out_list in [(olds, z_b), (news, z_a)]:
            enc = tok(texts, padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt").to(device)
            out = model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1)
            out_list.append(pooled.cpu())
    return torch.cat(z_b), torch.cat(z_a)


def bow_encode(samples, vocab_size=664):
    sys.path.insert(0, "/workspace/crucible-community-tap")
    from collectors.ast_tokenizer import ast_tokenize

    bows_b, bows_a = [], []
    for s in samples:
        bb = np.zeros(vocab_size, dtype=np.float32)
        ba = np.zeros(vocab_size, dtype=np.float32)
        for t in ast_tokenize(s["old"], 512):
            if 0 <= t < vocab_size:
                bb[t] += 1
        for t in ast_tokenize(s["new"], 512):
            if 0 <= t < vocab_size:
                ba[t] += 1
        nb = max(np.linalg.norm(bb), 1e-8)
        na = max(np.linalg.norm(ba), 1e-8)
        bows_b.append(bb / nb)
        bows_a.append(ba / na)
    return torch.tensor(np.stack(bows_b)), torch.tensor(np.stack(bows_a))


def retrieval_metrics(query_delta, gallery_delta, query_actions, gallery_actions, k=10):
    qd = F.normalize(query_delta, dim=-1)
    gd = F.normalize(gallery_delta, dim=-1)
    sim = qd @ gd.T
    topk = sim.topk(k, dim=-1).indices.numpy()

    n = len(query_actions)
    qa_n = F.normalize(query_actions, dim=-1)
    ga_n = F.normalize(gallery_actions, dim=-1)
    action_sim = (qa_n @ ga_n.T).numpy()

    q_et = query_actions[:, :3].argmax(axis=1).numpy()
    g_et = gallery_actions[:, :3].argmax(axis=1).numpy()
    q_joint = (q_et * 3 + query_actions[:, 3:6].argmax(axis=1).numpy())
    g_joint = (g_et * 3 + gallery_actions[:, 3:6].argmax(axis=1).numpy())

    rel_et = (g_et[topk] == q_et[:, None])
    rel_joint = (g_joint[topk] == q_joint[:, None])

    rel_a09 = np.zeros_like(topk, dtype=bool)
    rel_a095 = np.zeros_like(topk, dtype=bool)
    for i in range(n):
        for j_pos, j in enumerate(topk[i]):
            cs = action_sim[i, j]
            rel_a09[i, j_pos] = cs > 0.9
            rel_a095[i, j_pos] = cs > 0.95

    def summarize(rel):
        rr = []
        for i in range(n):
            hits = np.where(rel[i])[0]
            rr.append(1.0 / (hits[0] + 1) if len(hits) > 0 else 0.0)
        return {
            "mrr": float(np.mean(rr)),
            "recall@1": float(rel[:, :1].any(axis=1).mean()),
            "recall@5": float(rel[:, :5].any(axis=1).mean()),
            "recall@10": float(rel[:, :10].any(axis=1).mean()),
        }

    return {
        "by_edit_type": summarize(rel_et),
        "by_joint": summarize(rel_joint),
        "by_action_cos_0.9": summarize(rel_a09),
        "by_action_cos_0.95": summarize(rel_a095),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-query", type=int, default=400)
    parser.add_argument("--num-gallery", type=int, default=1200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    print(f"Loading CodeWM from {args.checkpoint}...")
    cw_model, cw_cfg = load_codewm(args.checkpoint, args.device)
    cw_params = sum(p.numel() for p in cw_model.parameters())
    print(f"  CodeWM: {cw_params:,} params")

    from transformers import AutoTokenizer, AutoModel
    print("Loading CodeBERT...")
    cb_tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    cb_model = AutoModel.from_pretrained("microsoft/codebert-base").to(args.device)
    cb_model.train(False)
    cb_params = sum(p.numel() for p in cb_model.parameters())

    total_needed = args.num_query + args.num_gallery
    samples = fetch_commit_chronicle_samples(total_needed)
    if len(samples) < total_needed:
        print(f"Only got {len(samples)} samples, adjusting splits")
        total_needed = len(samples)
        args.num_query = total_needed // 4
        args.num_gallery = total_needed - args.num_query

    np.random.seed(42)
    perm = np.random.permutation(total_needed)
    query_samples = [samples[i] for i in perm[:args.num_query]]
    gallery_samples = [samples[i] for i in perm[args.num_query:total_needed]]

    print(f"\nQuery: {len(query_samples)}, Gallery: {len(gallery_samples)}")

    print("\nComputing action vectors...")
    qact = compute_actions(query_samples)
    gact = compute_actions(gallery_samples)
    qact_t = torch.from_numpy(qact)
    gact_t = torch.from_numpy(gact)

    print("\n--- Encoding with CodeWM ---")
    t0 = time.time()
    cw_qb, cw_qa = encode_codewm(cw_model, query_samples, cw_cfg["max_seq_len"], args.device)
    cw_gb, cw_ga = encode_codewm(cw_model, gallery_samples, cw_cfg["max_seq_len"], args.device)
    cw_time = time.time() - t0
    print(f"  CodeWM: {cw_time:.1f}s ({cw_time*1000/total_needed:.1f}ms/sample)")

    print("\n--- Encoding with CodeBERT ---")
    t0 = time.time()
    cb_qb, cb_qa = encode_codebert(cb_tok, cb_model, query_samples, args.device)
    cb_gb, cb_ga = encode_codebert(cb_tok, cb_model, gallery_samples, args.device)
    cb_time = time.time() - t0
    print(f"  CodeBERT: {cb_time:.1f}s ({cb_time*1000/total_needed:.1f}ms/sample)")

    print("\n--- BoW (AST) baseline ---")
    bow_qb, bow_qa = bow_encode(query_samples)
    bow_gb, bow_ga = bow_encode(gallery_samples)

    cw_qd = cw_qa - cw_qb
    cw_gd = cw_ga - cw_gb
    cb_qd = cb_qa - cb_qb
    cb_gd = cb_ga - cb_gb
    bow_qd = bow_qa - bow_qb
    bow_gd = bow_ga - bow_gb

    cw_results = retrieval_metrics(cw_qd, cw_gd, qact_t, gact_t)
    cb_results = retrieval_metrics(cb_qd, cb_gd, qact_t, gact_t)
    bow_results = retrieval_metrics(bow_qd, bow_gd, qact_t, gact_t)

    print("\n=== CommitChronicle subset_llm — Head-to-head MRR ===")
    print(f"{'Criterion':<25} {'CodeWM':<10} {'CodeBERT':<12} {'BoW':<10}")
    print("-" * 60)
    for criterion in ["by_edit_type", "by_joint", "by_action_cos_0.9", "by_action_cos_0.95"]:
        cw = cw_results[criterion]["mrr"]
        cb = cb_results[criterion]["mrr"]
        bow = bow_results[criterion]["mrr"]
        print(f"{criterion:<25} {cw:<10.4f} {cb:<12.4f} {bow:<10.4f}")

    print("\n=== Recall@1 ===")
    print(f"{'Criterion':<25} {'CodeWM':<10} {'CodeBERT':<12} {'BoW':<10}")
    print("-" * 60)
    for criterion in ["by_edit_type", "by_joint", "by_action_cos_0.9", "by_action_cos_0.95"]:
        cw = cw_results[criterion]["recall@1"]
        cb = cb_results[criterion]["recall@1"]
        bow = bow_results[criterion]["recall@1"]
        print(f"{criterion:<25} {cw:<10.4f} {cb:<12.4f} {bow:<10.4f}")

    print(f"\nCodeWM:   {cw_params:,} params, {cw_time*1000/total_needed:.1f}ms/sample")
    print(f"CodeBERT: {cb_params:,} params, {cb_time*1000/total_needed:.1f}ms/sample")
    print(f"Size ratio: {cb_params/cw_params:.1f}x | Speed ratio: {cb_time/cw_time:.1f}x")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "dataset": "JetBrains-Research/commit-chronicle subset_llm",
                "codewm": {"params": cw_params, "ms_per_sample": cw_time*1000/total_needed, "results": cw_results},
                "codebert": {"params": cb_params, "ms_per_sample": cb_time*1000/total_needed, "results": cb_results},
                "bow": {"results": bow_results},
            }, f, indent=2)


if __name__ == "__main__":
    main()
