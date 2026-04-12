#!/usr/bin/env python3
"""Convert kernel migration pairs (JSON) to HDF5 matching CodeWM schema.

Takes the output of generate_pairs.py and produces an HDF5 file with:
  - before_tokens: int[N, 512]  (encoded source kernel config)
  - after_tokens:  int[N, 512]  (encoded target kernel config)
  - edit_actions:  float[N, 12] (migration action vector)
  - source_arch:   int[N]       (0=SM80, 1=SM90, 2=SM100)
  - target_arch:   int[N]
  - kernel_type:   int[N]       (0=gemm, 1=conv, 2=reduce)
  - metadata.has_trajectories = False (single-step pairs)

Encoding: Each kernel config is encoded as a fixed-length token sequence.
For Level 1 (structured configs), we encode the config fields as categorical
and numerical tokens into a 512-length sequence with padding.

Usage:
    python pairs_to_hdf5.py --pairs kernel_pairs.json --output kernel_pairs.h5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np


# ── Token vocabulary for structured kernel configs ──
# We encode each config field as a sequence of tokens.
# Vocab layout:
#   0:   PAD
#   1:   BOS (begin of sequence)
#   2:   EOS (end of sequence)
#   3:   SEP (field separator)
#   4-19:  field type tokens (arch, kernel_type, element_a, ...)
#   20-99: categorical value tokens
#   100-355: numerical tokens (0-255 for tile dims, stages, cluster)
#   356-399: reserved

PAD, BOS, EOS, SEP = 0, 1, 2, 3

# Field type tokens
FIELD_TOKENS = {
    "arch": 4,
    "kernel_type": 5,
    "element_a": 6,
    "element_b": 7,
    "element_c": 8,
    "layout_a": 9,
    "layout_b": 10,
    "tile_m": 11,
    "tile_n": 12,
    "tile_k": 13,
    "cluster_m": 14,
    "cluster_n": 15,
    "cluster_k": 16,
    "stages": 17,
    "mma_class": 18,
    "mainloop": 19,
}

# Categorical value tokens
CAT_TOKENS = {
    # Architectures
    "sm80": 20, "sm90": 21, "sm100": 22, "unknown": 23,
    # Kernel types
    "gemm": 24, "conv": 25, "reduce": 26,
    # Element types
    "f16": 27, "bf16": 28, "f32": 29, "f64": 30,
    "tf32": 31, "f8e4m3": 32, "f8e5m2": 33,
    "i8": 34, "u8": 35,
    # Layouts
    "row": 36, "col": 37,
    # MMA classes
    "hmma": 38, "wgmma": 39, "tcgen05": 40, "simt": 41,
    # Mainloop types
    "cp_async": 42, "tma": 43, "tma_warp_specialized": 44,
    # Epilogue
    "default": 46, "visitor": 47, "evt": 48,
}

ARCH_MAP = {"sm80": 0, "sm90": 1, "sm100": 2}
KTYPE_MAP = {"gemm": 0, "conv": 1, "reduce": 2}

# Numerical token offset: value N encoded as token 100 + N
NUM_OFFSET = 100


def encode_numerical(value: int, max_val: int = 255) -> int:
    """Encode a numerical value as a token."""
    clamped = max(0, min(value, max_val))
    return NUM_OFFSET + clamped


def encode_config(cfg: dict, seq_len: int = 512) -> np.ndarray:
    """Encode a kernel config as a fixed-length token sequence.

    Format: BOS [field_type value SEP]* EOS PAD...

    For categorical fields: field_type + cat_token
    For numerical fields: field_type + num_token (log2-encoded for tile dims)
    """
    tokens = [BOS]

    # Categorical fields
    for field in ["arch", "kernel_type", "element_a", "element_b", "element_c",
                  "layout_a", "layout_b", "mma_class", "mainloop"]:
        val = cfg.get(field, "unknown")
        tokens.append(FIELD_TOKENS.get(field, 3))
        tokens.append(CAT_TOKENS.get(str(val), CAT_TOKENS.get("unknown", 23)))
        tokens.append(SEP)

    # Numerical fields (tile dims encoded as log2 for compactness)
    import math
    for field in ["tile_m", "tile_n", "tile_k"]:
        val = cfg.get(field, 128)
        log_val = int(math.log2(max(val, 1))) if val > 0 else 0  # 0-12 range
        tokens.append(FIELD_TOKENS.get(field, 3))
        tokens.append(encode_numerical(log_val))
        tokens.append(SEP)

    # Cluster dims (small integers 1-4)
    for field in ["cluster_m", "cluster_n", "cluster_k"]:
        val = cfg.get(field, 1)
        tokens.append(FIELD_TOKENS.get(field, 3))
        tokens.append(encode_numerical(val))
        tokens.append(SEP)

    # Stages (small integer 1-8)
    tokens.append(FIELD_TOKENS.get("stages", 3))
    tokens.append(encode_numerical(cfg.get("stages", 2)))
    tokens.append(SEP)

    tokens.append(EOS)

    # Pad or truncate to seq_len
    if len(tokens) > seq_len:
        tokens = tokens[:seq_len - 1] + [EOS]
    else:
        tokens = tokens + [PAD] * (seq_len - len(tokens))

    return np.array(tokens, dtype=np.int64)


def encode_action(action: dict) -> np.ndarray:
    """Encode migration action as 12D float vector."""
    keys = [
        "migration_dir", "arch_delta",
        "tile_ratio_m", "tile_ratio_n", "tile_ratio_k",
        "pipeline_delta",
        "mma_changed", "mem_changed", "dtype_changed",
        "cluster_changed", "epilogue_changed", "reserved",
    ]
    return np.array([action.get(k, 0.0) for k in keys], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Convert kernel pairs to HDF5")
    parser.add_argument("--pairs", required=True, help="Input JSON from generate_pairs.py")
    parser.add_argument("--output", default="kernel_pairs.h5", help="Output HDF5 file")
    parser.add_argument("--seq-len", type=int, default=512, help="Token sequence length")
    parser.add_argument("--holdout-migration", default="sm80->sm100",
                        help="Migration direction to hold out for zero-shot eval")
    args = parser.parse_args()

    with open(args.pairs) as f:
        pairs = json.load(f)
    print(f"Loaded {len(pairs)} pairs")

    # Split by migration direction for train/eval
    train_pairs = []
    eval_pairs = []
    for p in pairs:
        direction = f"{p['before']['arch']}->{p['after']['arch']}"
        if direction == args.holdout_migration:
            eval_pairs.append(p)
        else:
            train_pairs.append(p)

    print(f"Train pairs: {len(train_pairs)} (excluding {args.holdout_migration})")
    print(f"Eval pairs (zero-shot): {len(eval_pairs)} ({args.holdout_migration})")

    # Encode all pairs
    for split_name, split_pairs, suffix in [
        ("train", train_pairs, ""),
        ("eval", eval_pairs, "_eval"),
    ]:
        if not split_pairs:
            continue

        N = len(split_pairs)
        before_tokens = np.zeros((N, args.seq_len), dtype=np.int64)
        after_tokens = np.zeros((N, args.seq_len), dtype=np.int64)
        edit_actions = np.zeros((N, 12), dtype=np.float32)
        source_arch = np.zeros(N, dtype=np.int32)
        target_arch = np.zeros(N, dtype=np.int32)
        kernel_type = np.zeros(N, dtype=np.int32)

        for i, pair in enumerate(split_pairs):
            before_tokens[i] = encode_config(pair["before"], args.seq_len)
            after_tokens[i] = encode_config(pair["after"], args.seq_len)
            edit_actions[i] = encode_action(pair["action"])
            source_arch[i] = ARCH_MAP.get(pair["before"]["arch"], 0)
            target_arch[i] = ARCH_MAP.get(pair["after"]["arch"], 0)
            kernel_type[i] = KTYPE_MAP.get(pair["before"]["kernel_type"], 0)

        # Write HDF5
        out_path = args.output.replace(".h5", f"{suffix}.h5")
        with h5py.File(out_path, "w") as f:
            f.create_dataset("before_tokens", data=before_tokens, compression="gzip")
            f.create_dataset("after_tokens", data=after_tokens, compression="gzip")
            f.create_dataset("edit_actions", data=edit_actions, compression="gzip")
            f.create_dataset("source_arch", data=source_arch)
            f.create_dataset("target_arch", data=target_arch)
            f.create_dataset("kernel_type", data=kernel_type)

            # Metadata matching CodeWM schema
            meta = f.create_group("metadata")
            meta.attrs["has_trajectories"] = False
            meta.attrs["num_pairs"] = N
            meta.attrs["seq_len"] = args.seq_len
            meta.attrs["action_dim"] = 12
            meta.attrs["split"] = split_name
            meta.attrs["holdout_migration"] = args.holdout_migration
            meta.attrs["vocab_size"] = 400  # PAD..399

        size_mb = Path(out_path).stat().st_size / 1024 / 1024
        print(f"  {split_name}: {N} pairs -> {out_path} ({size_mb:.1f} MB)")

    print("\nDone!")


if __name__ == "__main__":
    main()
