#!/usr/bin/env python3
"""Rebuild CommitPack HDF5 with raw byte tokens for AST-vs-raw comparison.

Re-downloads CommitPackFT from HuggingFace and produces an HDF5 with BOTH
AST tokens and raw byte tokens for the same samples. This enables a direct
comparison: train identical JEPA models on AST vs raw-byte tokenizations
of the same data.

If JEPA on raw bytes doesn't collapse the same way, the thesis "AST
tokenization removes nuisance that JEPA needs" becomes a causal claim.

Output HDF5 schema::

    /before_tokens       (N, ctx)  int64   -- AST tokens (same as existing)
    /after_tokens        (N, ctx)  int64   -- AST tokens
    /before_bytes        (N, ctx)  uint16  -- raw UTF-8 byte tokens (0-255, PAD=256)
    /after_bytes         (N, ctx)  uint16  -- raw UTF-8 byte tokens
    /diff_tokens         (N, ctx)  int64   -- diff tokens (same as existing)
    /edit_actions        (N, act)  float32 -- action vectors
    /metadata            attrs: vocab_size, byte_vocab_size, has_bytes, ...

Usage:
    python rebuild_with_bytes.py --output data/commitpackft_ast_and_bytes.h5
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collectors.ast_tokenizer import ast_tokenize
from collectors.ast_diff import compute_rich_action
from collectors.diff_tokenizer import tokenize_diff, DIFF_VOCAB_SIZE
from collectors.byte_tokenizer import byte_tokenize, BYTE_VOCAB_SIZE

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild CommitPack HDF5 with AST + raw byte tokens",
    )
    parser.add_argument("--output", default="data/commitpackft_ast_and_bytes.h5")
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--context-window", type=int, default=512)
    parser.add_argument("--max-file-size", type=int, default=40000)
    parser.add_argument("--min-diff-tokens", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("Loading CommitPackFT from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset(
        "json",
        data_files="hf://datasets/bigcode/commitpackft/data/python/data.jsonl",
        split="train",
        streaming=True,
    )

    before_ast_list: list[np.ndarray] = []
    after_ast_list: list[np.ndarray] = []
    before_byte_list: list[np.ndarray] = []
    after_byte_list: list[np.ndarray] = []
    diff_list: list[np.ndarray] = []
    action_list: list[np.ndarray] = []

    skipped_empty = 0
    skipped_size = 0
    skipped_nodiff = 0
    processed = 0

    for sample in ds:
        if processed >= args.max_samples:
            break

        old_src: str = sample.get("old_contents", "") or ""
        new_src: str = sample.get("new_contents", "") or ""

        if not old_src.strip() and not new_src.strip():
            skipped_empty += 1
            continue

        if len(old_src) > args.max_file_size or len(new_src) > args.max_file_size:
            skipped_size += 1
            continue

        # Diff tokens
        try:
            diff_tokens = tokenize_diff(old_src, new_src, max_len=args.context_window)
        except Exception:
            continue

        n_nonpad = int((diff_tokens > 0).sum())
        if n_nonpad < args.min_diff_tokens:
            skipped_nodiff += 1
            continue

        # AST tokens + byte tokens + action
        try:
            before_ast = ast_tokenize(old_src, args.context_window)
            after_ast = ast_tokenize(new_src, args.context_window)
            before_bytes = byte_tokenize(old_src, args.context_window)
            after_bytes = byte_tokenize(new_src, args.context_window)
            action = compute_rich_action(old_src, new_src)
        except Exception:
            continue

        before_ast_list.append(before_ast)
        after_ast_list.append(after_ast)
        before_byte_list.append(before_bytes)
        after_byte_list.append(after_bytes)
        diff_list.append(diff_tokens)
        action_list.append(action)
        processed += 1

        if processed % 5000 == 0:
            logger.info(
                "Processed %d (skipped: %d empty, %d size, %d nodiff)",
                processed, skipped_empty, skipped_size, skipped_nodiff,
            )

    if not before_ast_list:
        logger.error("No valid samples")
        sys.exit(1)

    # Write HDF5
    import h5py
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    N = len(before_ast_list)
    cw = args.context_window
    chunk = min(256, N)
    logger.info("Writing %d samples to %s", N, output)

    with h5py.File(output, "w") as f:
        # AST tokens (same format as existing files)
        f.create_dataset("before_tokens", data=np.stack(before_ast_list),
                         dtype="int64", compression="gzip", chunks=(chunk, cw))
        f.create_dataset("after_tokens", data=np.stack(after_ast_list),
                         dtype="int64", compression="gzip", chunks=(chunk, cw))

        # Raw byte tokens (new)
        f.create_dataset("before_bytes", data=np.stack(before_byte_list),
                         dtype="uint16", compression="gzip", chunks=(chunk, cw))
        f.create_dataset("after_bytes", data=np.stack(after_byte_list),
                         dtype="uint16", compression="gzip", chunks=(chunk, cw))

        # Diff tokens + actions (same as existing)
        f.create_dataset("diff_tokens", data=np.stack(diff_list),
                         dtype="int64", compression="gzip", chunks=(chunk, cw))
        f.create_dataset("edit_actions", data=np.stack(action_list),
                         dtype="float32", compression="gzip",
                         chunks=(chunk, action_list[0].shape[0]))

        meta = f.create_group("metadata")
        meta.attrs["num_edits"] = N
        meta.attrs["context_window"] = cw
        meta.attrs["action_dim"] = action_list[0].shape[0]
        meta.attrs["has_trajectories"] = False
        meta.attrs["has_diffs"] = True
        meta.attrs["has_bytes"] = True
        meta.attrs["vocab_size"] = 662  # AST vocab
        meta.attrs["byte_vocab_size"] = BYTE_VOCAB_SIZE
        meta.attrs["diff_vocab_size"] = DIFF_VOCAB_SIZE
        meta.attrs["source"] = "bigcode/commitpackft"
        meta.attrs["min_diff_tokens"] = args.min_diff_tokens

    size_mb = output.stat().st_size / 1024 / 1024
    logger.info("Done: %d samples, %.1f MB", N, size_mb)
    logger.info("Skipped: %d empty, %d oversized, %d no-diff",
                skipped_empty, skipped_size, skipped_nodiff)

    # Quick byte stats
    bytes_arr = np.stack(before_byte_list)
    nonpad = (bytes_arr < 256).sum(axis=1)
    logger.info("Byte token stats: mean=%.0f, min=%d, max=%d non-pad per sample",
                nonpad.mean(), nonpad.min(), nonpad.max())


if __name__ == "__main__":
    main()
