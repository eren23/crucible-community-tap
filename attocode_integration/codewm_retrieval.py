#!/usr/bin/env python3
"""Delta-NN edit retrieval for Code World Model.

Indexes a git repo's commit history as (before, after) edit pairs,
encodes each edit through the CodeWM encoder, and stores the delta
vectors. Queries return the top-k historically similar edits.

This is the Track B attocode-integration demo: it answers "how did we
handle this edit before?" for any Python git repo, using a learned
transition geometry rather than static symbol/text similarity.

Usage
-----
Build an index::

    WM_POOL_MODE=cls python codewm_retrieval.py index \\
        --repo /path/to/python/repo \\
        --checkpoint g8_sigreg_dir.pt \\
        --out ./idx \\
        --max-commits 500

Query::

    WM_POOL_MODE=cls python codewm_retrieval.py query \\
        --index ./idx \\
        --checkpoint g8_sigreg_dir.pt \\
        --before before.py --after after.py \\
        --top-k 5

Benchmark latency::

    WM_POOL_MODE=cls python codewm_retrieval.py benchmark \\
        --index ./idx --checkpoint g8_sigreg_dir.pt --num-queries 100
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Model loading (mirrors evaluation/semantic_eval.py)
# ---------------------------------------------------------------------------

def _load_code_wm_modules():
    """Dynamically load wm_base and code_wm modules from the tap."""
    tap_root = Path(__file__).parent.parent
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
    """Load a CodeWorldModel checkpoint in inference mode."""
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


def _get_tokenizer():
    """Load ast_tokenize from the tap's collectors."""
    tap_root = Path(__file__).parent.parent
    spec = importlib.util.spec_from_file_location(
        "ast_tokenizer", tap_root / "collectors" / "ast_tokenizer.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ast_tokenize


def encode_sources(
    model, sources: list[str], device: str = "cpu", batch_size: int = 32
) -> np.ndarray:
    """Tokenize and encode a batch of Python sources. Returns [N, dim] np array."""
    ast_tokenize = _get_tokenizer()
    max_len = model.state_encoder.max_seq_len if hasattr(model.state_encoder, "max_seq_len") else 512

    toks = np.stack([ast_tokenize(s, max_len=max_len) for s in sources], axis=0)
    toks = torch.from_numpy(toks.astype(np.int64)).to(device)

    out = []
    with torch.no_grad():
        for i in range(0, toks.shape[0], batch_size):
            z = model.state_encoder(toks[i : i + batch_size])
            out.append(z.cpu().numpy())
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# Git history walking
# ---------------------------------------------------------------------------

def _run_git(repo: Path, *args: str) -> str:
    """Run a git command in repo and return stdout (empty on failure)."""
    try:
        res = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, timeout=30,
        )
        return res.stdout if res.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def _file_at_rev(repo: Path, rev: str, path: str) -> str:
    """Read file content at a specific git revision (empty if missing)."""
    return _run_git(repo, "show", f"{rev}:{path}")


def iter_edit_pairs(
    repo: Path, max_commits: int = 500, max_pairs: int = 2000
) -> Iterator[dict]:
    """Walk git history, yielding (before, after) pairs per Python file touched.

    Skips merge commits, binary files, and files larger than ~40KB (after
    which AST parsing gets slow and the 512-token budget saturates).
    """
    # List commits (newest first), skip merges.
    log = _run_git(
        repo, "log", "--no-merges", f"-n{max_commits}",
        "--pretty=format:%H%x00%s", "--name-only",
    )
    if not log:
        return

    # Parse: each commit block is "SHA\x00subject\n<files>\n\n"
    emitted = 0
    for block in log.strip().split("\n\n"):
        lines = block.strip().split("\n")
        if not lines or "\x00" not in lines[0]:
            continue
        sha, subject = lines[0].split("\x00", 1)
        files = [f for f in lines[1:] if f.strip().endswith(".py")]

        for py_file in files:
            if emitted >= max_pairs:
                return
            before = _file_at_rev(repo, f"{sha}^", py_file)
            after = _file_at_rev(repo, sha, py_file)
            if not before.strip() and not after.strip():
                continue
            if len(before) > 40000 or len(after) > 40000:
                continue
            yield {
                "sha": sha,
                "file": py_file,
                "message": subject[:200],
                "before": before,
                "after": after,
            }
            emitted += 1


# ---------------------------------------------------------------------------
# Index / query
# ---------------------------------------------------------------------------

def build_index(
    repo: str, checkpoint: str, out_dir: str,
    max_commits: int = 500, max_pairs: int = 2000,
    device: str = "cpu",
) -> None:
    repo_path = Path(repo).resolve()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {checkpoint}...")
    model, _ = load_model(checkpoint, device)

    print(f"Walking git history in {repo_path} (max {max_commits} commits)...")
    t0 = time.time()
    pairs = list(iter_edit_pairs(repo_path, max_commits=max_commits, max_pairs=max_pairs))
    print(f"  collected {len(pairs)} edit pairs in {time.time() - t0:.1f}s")
    if not pairs:
        print("No pairs collected — aborting.")
        return

    print("Encoding before/after pairs...")
    t1 = time.time()
    befores = [p["before"] for p in pairs]
    afters = [p["after"] for p in pairs]
    z_before = encode_sources(model, befores, device=device)
    z_after = encode_sources(model, afters, device=device)
    deltas = z_after - z_before
    # L2-normalize for cosine search
    norms = np.linalg.norm(deltas, axis=1, keepdims=True) + 1e-8
    deltas_unit = deltas / norms
    print(f"  encoded {len(pairs)} pairs in {time.time() - t1:.1f}s")

    # Save
    np.save(out / "deltas.npy", deltas_unit.astype(np.float32))
    np.save(out / "deltas_raw.npy", deltas.astype(np.float32))
    meta = [
        {"sha": p["sha"], "file": p["file"], "message": p["message"]}
        for p in pairs
    ]
    with open(out / "meta.json", "w") as f:
        json.dump({"repo": str(repo_path), "entries": meta}, f, indent=2)
    print(f"Index written to {out}/ ({len(pairs)} entries, {deltas.nbytes / 1024:.0f} KB)")


def query_index(
    index_dir: str, checkpoint: str, before_code: str, after_code: str,
    top_k: int = 5, device: str = "cpu",
) -> list[dict]:
    idx = Path(index_dir)
    deltas_unit = np.load(idx / "deltas.npy")
    with open(idx / "meta.json") as f:
        meta = json.load(f)["entries"]

    model, _ = load_model(checkpoint, device)
    t0 = time.time()
    zs = encode_sources(model, [before_code, after_code], device=device)
    delta = zs[1] - zs[0]
    q = delta / (np.linalg.norm(delta) + 1e-8)
    scores = deltas_unit @ q
    top = np.argsort(-scores)[:top_k]
    elapsed_ms = (time.time() - t0) * 1000

    results = []
    for i in top:
        results.append({
            "score": float(scores[i]),
            "sha": meta[i]["sha"],
            "file": meta[i]["file"],
            "message": meta[i]["message"],
        })
    results_meta = {"query_latency_ms": round(elapsed_ms, 2), "index_size": len(meta)}
    return results, results_meta


def benchmark(index_dir: str, checkpoint: str, num_queries: int = 100, device: str = "cpu"):
    """Measure query latency by sampling random indexed edits as queries."""
    idx = Path(index_dir)
    with open(idx / "meta.json") as f:
        meta_doc = json.load(f)
    deltas_unit = np.load(idx / "deltas.npy")
    n = len(meta_doc["entries"])

    model, _ = load_model(checkpoint, device)

    # Warm up
    dummy = "def foo(): return 1"
    encode_sources(model, [dummy, dummy], device=device)

    # Pretend we have arbitrary input by grabbing indexed deltas directly
    # (skips re-tokenization; measures pure NN search)
    print(f"Benchmarking NN search over {n} entries...")
    rng = np.random.default_rng(0)
    qs = rng.integers(0, n, size=num_queries)
    t0 = time.time()
    for qi in qs:
        q = deltas_unit[qi]
        scores = deltas_unit @ q
        _ = np.argsort(-scores)[:5]
    t_search = (time.time() - t0) * 1000 / num_queries

    # Measure full pipeline (tokenize + encode + search) with a fixed code pair
    before = (Path(meta_doc["repo"]) / meta_doc["entries"][0]["file"]).read_text(
        errors="ignore"
    ) if (Path(meta_doc["repo"]) / meta_doc["entries"][0]["file"]).exists() else "x=1"
    after = before + "\n# edit\n"
    warm_z = encode_sources(model, [before, after], device=device)  # warm
    t0 = time.time()
    for _ in range(num_queries):
        zs = encode_sources(model, [before, after], device=device)
        d = zs[1] - zs[0]
        q = d / (np.linalg.norm(d) + 1e-8)
        scores = deltas_unit @ q
        _ = np.argsort(-scores)[:5]
    t_full = (time.time() - t0) * 1000 / num_queries

    print(f"  Pure NN search:        {t_search:.2f} ms / query")
    print(f"  Full (encode+search):  {t_full:.2f} ms / query")
    print(f"  Index size:            {n} entries")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Code WM delta-NN edit retrieval")
    sp = p.add_subparsers(dest="cmd", required=True)

    pi = sp.add_parser("index", help="Index a repo's git history")
    pi.add_argument("--repo", required=True)
    pi.add_argument("--checkpoint", required=True)
    pi.add_argument("--out", required=True)
    pi.add_argument("--max-commits", type=int, default=500)
    pi.add_argument("--max-pairs", type=int, default=2000)
    pi.add_argument("--device", default="cpu")

    pq = sp.add_parser("query", help="Query an index with a (before, after) pair")
    pq.add_argument("--index", required=True)
    pq.add_argument("--checkpoint", required=True)
    pq.add_argument("--before", required=True, help="path to before file OR raw code")
    pq.add_argument("--after", required=True, help="path to after file OR raw code")
    pq.add_argument("--top-k", type=int, default=5)
    pq.add_argument("--device", default="cpu")

    pb = sp.add_parser("benchmark", help="Measure query latency")
    pb.add_argument("--index", required=True)
    pb.add_argument("--checkpoint", required=True)
    pb.add_argument("--num-queries", type=int, default=100)
    pb.add_argument("--device", default="cpu")

    args = p.parse_args()

    if args.cmd == "index":
        build_index(
            args.repo, args.checkpoint, args.out,
            max_commits=args.max_commits, max_pairs=args.max_pairs,
            device=args.device,
        )
    elif args.cmd == "query":
        def _read_or_literal(x: str) -> str:
            pth = Path(x)
            return pth.read_text() if pth.exists() and pth.is_file() else x

        before_code = _read_or_literal(args.before)
        after_code = _read_or_literal(args.after)
        results, meta = query_index(
            args.index, args.checkpoint, before_code, after_code,
            top_k=args.top_k, device=args.device,
        )
        print(f"\nTop-{args.top_k} similar edits (query in {meta['query_latency_ms']} ms, "
              f"over {meta['index_size']} indexed edits):\n")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:+.3f}] {r['sha'][:8]}  {r['file']}")
            print(f"       \"{r['message']}\"")
    elif args.cmd == "benchmark":
        benchmark(args.index, args.checkpoint, args.num_queries, device=args.device)


if __name__ == "__main__":
    main()
