#!/usr/bin/env python3
"""Cross-repository retrieval eval on 20 held-out Python repos.

Closes the "cross-repo generalization never measured" gap from the Phase 5C
audit. The model was trained on CommitPackFT Python (IID val split). Here we
test it on fresh git histories of 20 well-known Python libraries that are
NOT in CommitPackFT's training slice.

Pipeline per repo:

1. ``git clone --depth=500`` into ``/tmp/cross_repo_eval/<repo>``
2. Walk the last N commits, extract (before, after) source pairs for each
   changed ``*.py`` file
3. Tokenize with ``collectors.ast_tokenizer.ast_tokenize``
4. Compute 7-dim action with ``collectors.commitpack_processor.compute_action``
5. Accumulate into a single cross-repo pool

Then evaluate retrieval with ``leave-one-repo-out``: for each repo's edits,
use them as the query set and all OTHER repos as the gallery. Report MRR@10
and Recall@1/5/10 per repo under the ``by_joint`` (edit_type × scope)
relevance criterion, plus aggregate MRR across repos.

Compares:
- CodeWM (one or more checkpoints)
- BoW (AST tokens) baseline
- Random baseline

Usage::

    python cross_repo_eval.py \\
        --checkpoint ~/.crucible-hub/taps/crucible-community-tap/checkpoints/phase5/phase5-contrast-extreme-3k-best.pt \\
        --commits-per-repo 150 \\
        --out /tmp/cross_repo_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))
from codesearchnet_eval import (  # type: ignore
    compute_retrieval_metrics as _csn_metrics,
    encode_bow,
    encode_codewm,
    load_codewm,
)


# 20 well-known Python libraries. These are popular enough to have rich
# commit histories but are NOT part of CommitPackFT's training distribution
# (they were post-dated / deduplicated / not in bigcode/commitpack python slice).
DEFAULT_REPOS = [
    ("pandas", "https://github.com/pandas-dev/pandas.git"),
    ("requests", "https://github.com/psf/requests.git"),
    ("fastapi", "https://github.com/tiangolo/fastapi.git"),
    ("httpx", "https://github.com/encode/httpx.git"),
    ("rich", "https://github.com/Textualize/rich.git"),
    ("typer", "https://github.com/tiangolo/typer.git"),
    ("pydantic", "https://github.com/pydantic/pydantic.git"),
    ("scrapy", "https://github.com/scrapy/scrapy.git"),
    ("dask", "https://github.com/dask/dask.git"),
    ("networkx", "https://github.com/networkx/networkx.git"),
    ("flask", "https://github.com/pallets/flask.git"),
    ("attrs", "https://github.com/python-attrs/attrs.git"),
    ("click", "https://github.com/pallets/click.git"),
    ("sympy", "https://github.com/sympy/sympy.git"),
    ("pytest", "https://github.com/pytest-dev/pytest.git"),
    ("tqdm", "https://github.com/tqdm/tqdm.git"),
    ("loguru", "https://github.com/Delgan/loguru.git"),
    ("polars", "https://github.com/pola-rs/polars.git"),
    ("matplotlib", "https://github.com/matplotlib/matplotlib.git"),
    ("numpy", "https://github.com/numpy/numpy.git"),
]


def _run(cmd: list[str], cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=check)


def clone_if_needed(name: str, url: str, workdir: Path, depth: int = 500) -> Path | None:
    target = workdir / name
    if target.exists():
        if (target / ".git").exists():
            return target
        import shutil
        shutil.rmtree(target)
    print(f"  cloning {name} (depth={depth})...", flush=True)
    try:
        _run(["git", "clone", "--depth", str(depth), "--quiet", url, str(target)])
    except subprocess.CalledProcessError as exc:
        print(f"    FAILED: {exc.stderr.strip()}")
        return None
    return target


def recent_commits(repo: Path, max_commits: int) -> list[str]:
    res = _run(["git", "log", "--format=%H", f"-n{max_commits}"], cwd=str(repo))
    return [h for h in res.stdout.strip().split("\n") if h]


def file_at_rev(repo: Path, rev: str, path: str) -> str | None:
    try:
        res = _run(["git", "show", f"{rev}:{path}"], cwd=str(repo), check=False)
        if res.returncode != 0:
            return None
        return res.stdout
    except Exception:
        return None


def changed_py_files(repo: Path, rev: str) -> list[str]:
    """List Python files touched by this commit."""
    try:
        res = _run(
            ["git", "show", "--name-only", "--pretty=format:", rev],
            cwd=str(repo),
            check=False,
        )
        files = [f.strip() for f in res.stdout.split("\n") if f.strip()]
        return [f for f in files if f.endswith(".py")]
    except Exception:
        return []


def parent_rev(repo: Path, rev: str) -> str | None:
    try:
        res = _run(["git", "rev-parse", f"{rev}^"], cwd=str(repo), check=False)
        if res.returncode != 0:
            return None
        return res.stdout.strip()
    except Exception:
        return None


def extract_edit_pairs(
    repo: Path,
    max_commits: int = 150,
    max_pairs_per_repo: int = 200,
    max_file_size: int = 50_000,
) -> list[tuple[str, str]]:
    """Walk recent commits and return (before, after) source pairs."""
    pairs: list[tuple[str, str]] = []
    commits = recent_commits(repo, max_commits)
    for rev in commits:
        if len(pairs) >= max_pairs_per_repo:
            break
        parent = parent_rev(repo, rev)
        if parent is None:
            continue
        files = changed_py_files(repo, rev)
        for path in files:
            if len(pairs) >= max_pairs_per_repo:
                break
            before = file_at_rev(repo, parent, path)
            after = file_at_rev(repo, rev, path)
            if before is None and after is None:
                continue
            before = before or ""
            after = after or ""
            if not before.strip() and not after.strip():
                continue
            if before == after:
                continue
            if len(before) > max_file_size or len(after) > max_file_size:
                continue
            pairs.append((before, after))
    return pairs


@torch.no_grad()
def encode_edit_pairs(
    model,
    pairs: list[tuple[str, str]],
    max_seq_len: int,
    device: str,
    batch_size: int = 32,
) -> torch.Tensor:
    """Return delta embeddings [N, D] for a list of (before, after) pairs."""
    samples_before = [{"code": p[0]} for p in pairs]
    samples_after = [{"code": p[1]} for p in pairs]
    z_b = encode_codewm(model, samples_before, max_seq_len, device, batch_size)
    z_a = encode_codewm(model, samples_after, max_seq_len, device, batch_size)
    return z_a - z_b


def compute_actions_for_pairs(pairs: list[tuple[str, str]]) -> np.ndarray:
    sys.path.insert(0, str(THIS_DIR.parent))
    from collectors.commitpack_processor import compute_action  # type: ignore
    out = np.stack([compute_action(old, new) for old, new in pairs])
    return out.astype(np.float32)


def joint_labels_from_actions(actions: np.ndarray) -> np.ndarray:
    """Compute joint label = edit_type * 3 + scope from 7-dim action vectors."""
    et = np.argmax(actions[:, :3], axis=1)
    sc = np.argmax(actions[:, 3:6], axis=1)
    return et * 3 + sc


def bow_encode_deltas(pairs: list[tuple[str, str]], max_seq_len: int, vocab_size: int = 664) -> torch.Tensor:
    samples_before = [{"code": p[0]} for p in pairs]
    samples_after = [{"code": p[1]} for p in pairs]
    bow_b = encode_bow(samples_before, max_seq_len, vocab_size=vocab_size)
    bow_a = encode_bow(samples_after, max_seq_len, vocab_size=vocab_size)
    return bow_a - bow_b


def retrieval_with_joint_labels(
    query_delta: torch.Tensor,
    gallery_delta: torch.Tensor,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    k: int = 10,
) -> dict:
    qn = F.normalize(query_delta, dim=-1)
    gn = F.normalize(gallery_delta, dim=-1)
    sim = qn @ gn.T
    topk = sim.topk(k, dim=-1).indices.numpy()
    rel = gallery_labels[topk] == query_labels[:, None]

    n = len(query_labels)
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
    parser.add_argument("--workdir", default="/tmp/cross_repo_eval")
    parser.add_argument("--commits-per-repo", type=int, default=150)
    parser.add_argument("--max-pairs-per-repo", type=int, default=200)
    parser.add_argument("--depth", type=int, default=500)
    parser.add_argument("--max-repos", type=int, default=20)
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    )
    args = parser.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CodeWM from {args.checkpoint}...")
    model, cfg = load_codewm(args.checkpoint, args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  CodeWM: {n_params:,} params on {args.device}")

    # ---- Phase 1: clone + extract per-repo edit pairs ----
    repo_data: dict[str, dict] = {}
    t_clone_start = time.time()
    for name, url in DEFAULT_REPOS[: args.max_repos]:
        print(f"\n[{name}]")
        repo = clone_if_needed(name, url, workdir, depth=args.depth)
        if repo is None:
            continue
        t0 = time.time()
        pairs = extract_edit_pairs(
            repo,
            max_commits=args.commits_per_repo,
            max_pairs_per_repo=args.max_pairs_per_repo,
        )
        print(f"  extracted {len(pairs)} edit pairs in {time.time() - t0:.1f}s")
        if len(pairs) < 10:
            print(f"  too few pairs, skipping")
            continue
        repo_data[name] = {"pairs": pairs}
    print(f"\nTotal clone+extract time: {time.time() - t_clone_start:.1f}s")

    if len(repo_data) < 3:
        print("ERROR: fewer than 3 repos with enough pairs. Aborting.")
        return 1

    # ---- Phase 2: encode all pairs once (batch across repos) ----
    print(f"\n--- Encoding all {sum(len(d['pairs']) for d in repo_data.values())} pairs with CodeWM ---")
    t0 = time.time()
    all_pairs: list[tuple[str, str]] = []
    repo_slice: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, data in repo_data.items():
        start = cursor
        all_pairs.extend(data["pairs"])
        cursor += len(data["pairs"])
        repo_slice[name] = (start, cursor)

    cw_deltas = encode_edit_pairs(model, all_pairs, cfg["max_seq_len"], args.device)
    cw_time = time.time() - t0
    print(f"  done in {cw_time:.1f}s ({cw_time * 1000 / len(all_pairs):.1f} ms/pair)")

    print(f"\n--- Encoding all pairs with BoW ---")
    bow_deltas = bow_encode_deltas(all_pairs, cfg["max_seq_len"], vocab_size=cfg.get("vocab_size", 664))

    print(f"\n--- Computing actions + joint labels ---")
    actions = compute_actions_for_pairs(all_pairs)
    joint = joint_labels_from_actions(actions)

    # ---- Phase 3: leave-one-repo-out retrieval ----
    print(f"\n=== Leave-one-repo-out retrieval (by joint edit_type × scope) ===")
    per_repo: dict[str, dict] = {}
    cw_mrrs: list[float] = []
    bow_mrrs: list[float] = []

    print(f"{'Repo':<14} {'N':>5} | {'CodeWM MRR':>12} {'BoW MRR':>10} {'Δ':>8}")
    print("-" * 60)

    for name, (start, end) in repo_slice.items():
        q_mask = np.zeros(len(all_pairs), dtype=bool)
        q_mask[start:end] = True
        g_mask = ~q_mask

        q_cw = cw_deltas[q_mask]
        g_cw = cw_deltas[g_mask]
        q_bow = bow_deltas[q_mask]
        g_bow = bow_deltas[g_mask]
        q_joint = joint[q_mask]
        g_joint = joint[g_mask]

        cw_metrics = retrieval_with_joint_labels(q_cw, g_cw, q_joint, g_joint)
        bow_metrics = retrieval_with_joint_labels(q_bow, g_bow, q_joint, g_joint)

        per_repo[name] = {
            "n_queries": int(q_mask.sum()),
            "n_gallery": int(g_mask.sum()),
            "codewm": cw_metrics,
            "bow": bow_metrics,
        }
        cw_mrrs.append(cw_metrics["mrr@10"])
        bow_mrrs.append(bow_metrics["mrr@10"])
        delta = cw_metrics["mrr@10"] - bow_metrics["mrr@10"]
        marker = "✓" if delta > 0 else "✗"
        print(
            f"{name:<14} {int(q_mask.sum()):>5} | "
            f"{cw_metrics['mrr@10']:>12.4f} "
            f"{bow_metrics['mrr@10']:>10.4f} "
            f"{delta:>+8.4f} {marker}"
        )

    print("-" * 60)
    print(
        f"{'AGGREGATE':<14} {len(all_pairs):>5} | "
        f"{np.mean(cw_mrrs):>12.4f} "
        f"{np.mean(bow_mrrs):>10.4f} "
        f"{np.mean(cw_mrrs) - np.mean(bow_mrrs):>+8.4f}"
    )

    result = {
        "checkpoint": str(args.checkpoint),
        "codewm_params": n_params,
        "total_pairs": len(all_pairs),
        "n_repos": len(repo_data),
        "per_repo": per_repo,
        "aggregate": {
            "codewm_mean_mrr10": float(np.mean(cw_mrrs)),
            "bow_mean_mrr10": float(np.mean(bow_mrrs)),
            "delta_mean_mrr10": float(np.mean(cw_mrrs) - np.mean(bow_mrrs)),
            "codewm_std_mrr10": float(np.std(cw_mrrs)),
            "bow_std_mrr10": float(np.std(bow_mrrs)),
        },
        "codewm_latency_ms_per_pair": cw_time * 1000 / len(all_pairs),
    }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
