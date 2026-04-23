#!/usr/bin/env python3
"""Pre-compute frozen backbone features for CodeDeltaTok.

Runs a pretrained code encoder (UniXcoder, CodeBERT, CodeT5+) on all
before/after code pairs in CommitPackFT and stores the CLS-pooled
embeddings in HDF5. These frozen features are the input to CodeDeltaTok
training — no backbone inference needed during delta token learning.

Usage:
    python precompute_backbone_features.py \
        --model microsoft/unixcoder-base \
        --output data/commitpackft_unixcoder_features.h5 \
        --max-samples 50000 \
        --batch-size 64
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute backbone features for CodeDeltaTok",
    )
    parser.add_argument("--model", default="microsoft/unixcoder-base",
                        help="HuggingFace model name")
    parser.add_argument("--output", default="data/commitpackft_unixcoder_features.h5")
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max token length for backbone tokenizer")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-file-size", type=int, default=40000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Load model and tokenizer
    logger.info(f"Loading model: {args.model}")
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    model = model.to(args.device)
    model.requires_grad_(False)

    hidden_size = model.config.hidden_size
    logger.info(f"Model loaded: hidden_size={hidden_size}, device={args.device}")

    # Load CommitPackFT
    logger.info("Loading CommitPackFT from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset(
        "json",
        data_files="hf://datasets/bigcode/commitpackft/data/python/data.jsonl",
        split="train",
        streaming=True,
    )

    # Collect source code pairs
    old_sources: list[str] = []
    new_sources: list[str] = []
    skipped = 0

    for sample in ds:
        if len(old_sources) >= args.max_samples:
            break
        old_src = sample.get("old_contents", "") or ""
        new_src = sample.get("new_contents", "") or ""
        if not old_src.strip() and not new_src.strip():
            skipped += 1
            continue
        if len(old_src) > args.max_file_size or len(new_src) > args.max_file_size:
            skipped += 1
            continue
        old_sources.append(old_src)
        new_sources.append(new_src)
        if len(old_sources) % 10000 == 0:
            logger.info(f"Collected {len(old_sources)} pairs (skipped {skipped})")

    N = len(old_sources)
    logger.info(f"Collected {N} pairs total (skipped {skipped})")

    # Encode in batches
    def encode_batch(texts: list[str]) -> np.ndarray:
        """Encode a batch of texts to CLS-pooled features [B, hidden_size]."""
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(args.device)
        with torch.no_grad():
            outputs = model(**inputs)
            # CLS token is position 0
            cls_features = outputs.last_hidden_state[:, 0, :]
        return cls_features.cpu().numpy()

    before_features = np.zeros((N, hidden_size), dtype=np.float32)
    after_features = np.zeros((N, hidden_size), dtype=np.float32)

    bs = args.batch_size
    for i in range(0, N, bs):
        end = min(i + bs, N)
        batch_old = old_sources[i:end]
        batch_new = new_sources[i:end]

        before_features[i:end] = encode_batch(batch_old)
        after_features[i:end] = encode_batch(batch_new)

        if (i // bs) % 50 == 0:
            logger.info(f"Encoded {end}/{N} pairs")

    logger.info(f"All {N} pairs encoded. Writing HDF5...")

    # Write HDF5
    import h5py
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output, "w") as f:
        f.create_dataset("before_features", data=before_features,
                         dtype="float32", compression="gzip",
                         chunks=(min(256, N), hidden_size))
        f.create_dataset("after_features", data=after_features,
                         dtype="float32", compression="gzip",
                         chunks=(min(256, N), hidden_size))

        meta = f.create_group("metadata")
        meta.attrs["num_samples"] = N
        meta.attrs["hidden_size"] = hidden_size
        meta.attrs["model_name"] = args.model
        meta.attrs["max_length"] = args.max_length
        meta.attrs["source"] = "bigcode/commitpackft"
        meta.attrs["pooling"] = "cls"

    size_mb = output.stat().st_size / 1024 / 1024
    logger.info(f"Done: {N} pairs, {hidden_size}-dim features, {size_mb:.1f} MB")

    # Quick stats
    cos_sims = np.sum(before_features * after_features, axis=1) / (
        np.linalg.norm(before_features, axis=1) *
        np.linalg.norm(after_features, axis=1) + 1e-8
    )
    logger.info(f"Before/after cosine sim: mean={cos_sims.mean():.3f}, "
                f"std={cos_sims.std():.3f}, min={cos_sims.min():.3f}, max={cos_sims.max():.3f}")


if __name__ == "__main__":
    main()
