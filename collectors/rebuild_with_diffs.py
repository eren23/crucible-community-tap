#!/usr/bin/env python3
"""Rebuild CommitPack HDF5 with diff tokens for DeltaCodeWM.

Re-downloads CommitPackFT from HuggingFace, AST-tokenizes before/after
(same as existing pipeline), AND tokenizes the unified diff. Produces
a new HDF5 with all three token arrays.

Usage:
    python rebuild_with_diffs.py --output data/commitpackft_with_diffs.h5 [--max-samples 50000]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Add tap root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collectors.ast_tokenizer import ast_tokenize
from collectors.ast_diff import compute_rich_action
from collectors.diff_tokenizer import tokenize_diff, DIFF_VOCAB_SIZE

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Rebuild CommitPack HDF5 with diff tokens")
    parser.add_argument("--output", default="data/commitpackft_with_diffs.h5")
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--context-window", type=int, default=512)
    parser.add_argument("--max-file-size", type=int, default=40000)
    parser.add_argument("--use-ft", action="store_true", default=True)
    parser.add_argument("--min-diff-tokens", type=int, default=5,
                        help="Skip samples where diff has fewer than N non-pad tokens")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Load CommitPackFT
    logger.info("Loading CommitPackFT from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset(
        "json",
        data_files="hf://datasets/bigcode/commitpackft/data/python/data.jsonl",
        split="train",
        streaming=True,
    )

    before_list = []
    after_list = []
    diff_list = []
    action_list = []

    skipped_empty = 0
    skipped_size = 0
    skipped_nodiff = 0
    processed = 0

    for sample in ds:
        if processed >= args.max_samples:
            break

        old_src = sample.get("old_contents", "") or ""
        new_src = sample.get("new_contents", "") or ""

        if not old_src.strip() and not new_src.strip():
            skipped_empty += 1
            continue

        if len(old_src) > args.max_file_size or len(new_src) > args.max_file_size:
            skipped_size += 1
            continue

        # Tokenize diff
        try:
            diff_tokens = tokenize_diff(old_src, new_src, max_len=args.context_window)
        except Exception:
            continue

        # Skip trivial diffs (empty or near-empty)
        n_nonpad = int((diff_tokens > 0).sum())
        if n_nonpad < args.min_diff_tokens:
            skipped_nodiff += 1
            continue

        # AST-tokenize before/after
        try:
            before_tokens = ast_tokenize(old_src, args.context_window)
            after_tokens = ast_tokenize(new_src, args.context_window)
            action = compute_rich_action(old_src, new_src)
        except Exception:
            continue

        before_list.append(before_tokens)
        after_list.append(after_tokens)
        diff_list.append(diff_tokens)
        action_list.append(action)
        processed += 1

        if processed % 5000 == 0:
            logger.info(
                "Processed %d (skipped: %d empty, %d size, %d nodiff)",
                processed, skipped_empty, skipped_size, skipped_nodiff,
            )

    if not before_list:
        logger.error("No valid samples")
        sys.exit(1)

    # Write HDF5
    import h5py
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    N = len(before_list)
    logger.info("Writing %d samples to %s", N, output)

    with h5py.File(output, "w") as f:
        f.create_dataset("before_tokens", data=np.stack(before_list), dtype="int64",
                         compression="gzip", chunks=(min(256, N), args.context_window))
        f.create_dataset("after_tokens", data=np.stack(after_list), dtype="int64",
                         compression="gzip", chunks=(min(256, N), args.context_window))
        f.create_dataset("diff_tokens", data=np.stack(diff_list), dtype="int64",
                         compression="gzip", chunks=(min(256, N), args.context_window))
        f.create_dataset("edit_actions", data=np.stack(action_list), dtype="float32",
                         compression="gzip", chunks=(min(256, N), action_list[0].shape[0]))

        meta = f.create_group("metadata")
        meta.attrs["num_edits"] = N
        meta.attrs["context_window"] = args.context_window
        meta.attrs["action_dim"] = action_list[0].shape[0]
        meta.attrs["has_trajectories"] = False
        meta.attrs["has_diffs"] = True
        meta.attrs["vocab_size"] = 662  # AST vocab
        meta.attrs["diff_vocab_size"] = DIFF_VOCAB_SIZE
        meta.attrs["source"] = "bigcode/commitpackft"
        meta.attrs["min_diff_tokens"] = args.min_diff_tokens

    size_mb = output.stat().st_size / 1024 / 1024
    logger.info("Done: %d samples, %.1f MB", N, size_mb)
    logger.info("Skipped: %d empty, %d oversized, %d no-diff", skipped_empty, skipped_size, skipped_nodiff)

    # Quick stats
    diffs = np.stack(diff_list)
    nonpad = (diffs > 0).sum(axis=1)
    logger.info("Diff token stats: mean=%.1f, median=%.0f, max=%d non-pad tokens per sample",
                nonpad.mean(), np.median(nonpad), nonpad.max())


if __name__ == "__main__":
    main()
