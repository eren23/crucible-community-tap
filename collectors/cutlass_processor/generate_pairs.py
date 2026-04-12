#!/usr/bin/env python3
"""Generate architecture migration pairs from extracted CUTLASS kernel configs.

Takes the output of extract_kernel_configs.py and produces (before, after, action)
triples for KernelWM training. Pairs are matched by semantic equivalence: same
kernel_type, element types, and layouts, but different architecture.

Two modes:
  1. Real pairs: match configs extracted from actual CUTLASS source
  2. Synthetic augmentation: for each real config, generate plausible
     configs for other architectures using known migration rules

Usage:
    python generate_pairs.py --configs kernel_configs.json --output pairs.json [--synthetic]

Output: JSON array of migration pairs, each with:
    - before: dict (source kernel config)
    - after: dict (target kernel config)
    - action: dict (migration action vector)
    - pair_type: str ("real" or "synthetic")
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from itertools import product


# ── Architecture migration rules ──

# Known tile shape migrations (SM80 -> SM90 -> SM100)
TILE_MIGRATIONS = {
    ("sm80", "sm90"): {
        # SM90 supports larger K tiles due to TMA
        (128, 128, 32): [(128, 128, 64), (128, 256, 64), (64, 128, 64)],
        (128, 256, 32): [(128, 256, 64)],
        (256, 128, 32): [(256, 128, 64)],
        (64, 64, 32): [(64, 64, 64), (64, 128, 64)],
    },
    ("sm90", "sm100"): {
        # SM100 keeps most SM90 tiles, adds some larger ones
        (128, 128, 64): [(128, 128, 64), (128, 128, 128)],
        (128, 256, 64): [(128, 256, 64), (128, 256, 128)],
        (64, 128, 64): [(64, 128, 64), (64, 128, 128)],
    },
    ("sm80", "sm100"): {
        (128, 128, 32): [(128, 128, 64), (128, 128, 128)],
        (128, 256, 32): [(128, 256, 64), (128, 256, 128)],
        (64, 64, 32): [(64, 64, 64), (64, 128, 64)],
    },
}

# Default stage counts per arch
DEFAULT_STAGES = {"sm80": 2, "sm90": 4, "sm100": 4}

# MMA class per arch
DEFAULT_MMA = {"sm80": "hmma", "sm90": "wgmma", "sm100": "tcgen05"}

# Mainloop per arch
DEFAULT_MAINLOOP = {"sm80": "cp_async", "sm90": "tma_warp_specialized", "sm100": "tcgen05"}

# Cluster shapes per arch (sm80 has no clusters)
DEFAULT_CLUSTER = {"sm80": (1, 1, 1), "sm90": (1, 1, 1), "sm100": (1, 1, 1)}
VALID_CLUSTERS = {
    "sm80": [(1, 1, 1)],
    "sm90": [(1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1)],
    "sm100": [(1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1), (4, 1, 1)],
}

# Valid tile shapes per arch (common ones)
VALID_TILES = {
    "sm80": [
        (64, 64, 32), (128, 64, 32), (64, 128, 32),
        (128, 128, 32), (128, 256, 32), (256, 128, 32),
    ],
    "sm90": [
        (64, 64, 64), (64, 128, 64), (128, 64, 64),
        (128, 128, 64), (128, 256, 64), (256, 128, 64),
        (64, 64, 128), (128, 128, 128),
    ],
    "sm100": [
        (64, 64, 64), (64, 128, 64), (128, 64, 64),
        (128, 128, 64), (128, 256, 64), (256, 128, 64),
        (64, 64, 128), (128, 128, 128), (128, 256, 128),
    ],
}

# Valid element types per arch
VALID_ELEMENTS = {
    "sm80": ["f16", "bf16", "tf32", "f32"],
    "sm90": ["f16", "bf16", "tf32", "f32", "f8e4m3", "f8e5m2"],
    "sm100": ["f16", "bf16", "tf32", "f32", "f8e4m3", "f8e5m2"],
}

ARCH_ORDER = {"sm80": 0, "sm90": 1, "sm100": 2}


def compute_action(before: dict, after: dict) -> dict:
    """Compute a 12D migration action vector from a before/after pair."""
    arch_delta = ARCH_ORDER.get(after["arch"], 0) - ARCH_ORDER.get(before["arch"], 0)
    migration_dir = 1.0 if arch_delta > 0 else (-1.0 if arch_delta < 0 else 0.0)

    # Tile ratios (log-scale, 0 = no change)
    import math
    tile_ratio_m = math.log2(after["tile_m"] / max(before["tile_m"], 1)) if before["tile_m"] > 0 else 0.0
    tile_ratio_n = math.log2(after["tile_n"] / max(before["tile_n"], 1)) if before["tile_n"] > 0 else 0.0
    tile_ratio_k = math.log2(after["tile_k"] / max(before["tile_k"], 1)) if before["tile_k"] > 0 else 0.0

    # Pipeline stage delta
    pipeline_delta = after["stages"] - before["stages"]

    # MMA class change (categorical -> binary flags)
    mma_changed = 1.0 if before["mma_class"] != after["mma_class"] else 0.0

    # Memory path change
    mem_changed = 1.0 if before["mainloop"] != after["mainloop"] else 0.0

    # Element type change
    dtype_changed = 1.0 if (before["element_a"] != after["element_a"] or
                            before["element_b"] != after["element_b"]) else 0.0

    # Cluster shape change
    cluster_changed = 1.0 if (before["cluster_m"] != after["cluster_m"] or
                              before["cluster_n"] != after["cluster_n"]) else 0.0

    # Epilogue change
    epilogue_changed = 1.0 if before.get("epilogue") != after.get("epilogue") else 0.0

    return {
        "migration_dir": migration_dir,
        "arch_delta": float(arch_delta),
        "tile_ratio_m": tile_ratio_m,
        "tile_ratio_n": tile_ratio_n,
        "tile_ratio_k": tile_ratio_k,
        "pipeline_delta": float(pipeline_delta),
        "mma_changed": mma_changed,
        "mem_changed": mem_changed,
        "dtype_changed": dtype_changed,
        "cluster_changed": cluster_changed,
        "epilogue_changed": epilogue_changed,
        "reserved": 0.0,
    }


def find_real_pairs(configs: list[dict]) -> list[dict]:
    """Find pairs of configs that represent the same kernel on different archs."""
    pairs = []

    # Group by semantic equivalence key (kernel_type + elements + layouts)
    groups: dict[tuple, list[dict]] = {}
    for cfg in configs:
        key = (cfg["kernel_type"], cfg["element_a"], cfg["element_b"],
               cfg["element_c"], cfg["layout_a"], cfg["layout_b"])
        groups.setdefault(key, []).append(cfg)

    for key, group in groups.items():
        # Find configs at different arch levels
        by_arch: dict[str, list[dict]] = {}
        for cfg in group:
            by_arch.setdefault(cfg["arch"], []).append(cfg)

        # Generate pairs across architectures
        archs = sorted(by_arch.keys(), key=lambda a: ARCH_ORDER.get(a, 99))
        for i in range(len(archs)):
            for j in range(i + 1, len(archs)):
                for before in by_arch[archs[i]]:
                    for after in by_arch[archs[j]]:
                        action = compute_action(before, after)
                        pairs.append({
                            "before": before,
                            "after": after,
                            "action": action,
                            "pair_type": "real",
                        })

    return pairs


def generate_synthetic_config(base: dict, target_arch: str) -> list[dict]:
    """Generate plausible configs for target_arch based on a base config."""
    results = []
    base_tile = (base["tile_m"], base["tile_n"], base["tile_k"])
    pair_key = (base["arch"], target_arch)

    # Get target tiles from migration table or use valid tiles
    if pair_key in TILE_MIGRATIONS and base_tile in TILE_MIGRATIONS[pair_key]:
        target_tiles = TILE_MIGRATIONS[pair_key][base_tile]
    else:
        # Find closest valid tile for target arch
        target_tiles = []
        for vt in VALID_TILES.get(target_arch, []):
            # Keep M and N similar, allow K to change
            if vt[0] == base_tile[0] and vt[1] == base_tile[1]:
                target_tiles.append(vt)
        if not target_tiles:
            # Just use the base tile if it's valid for target
            target_tiles = [base_tile]

    for tile in target_tiles:
        for cluster in VALID_CLUSTERS.get(target_arch, [(1, 1, 1)])[:2]:  # limit combos
            cfg = copy.deepcopy(base)
            cfg["arch"] = target_arch
            cfg["tile_m"], cfg["tile_n"], cfg["tile_k"] = tile
            cfg["cluster_m"], cfg["cluster_n"], cfg["cluster_k"] = cluster
            cfg["stages"] = DEFAULT_STAGES[target_arch]
            cfg["mma_class"] = DEFAULT_MMA[target_arch]
            cfg["mainloop"] = DEFAULT_MAINLOOP[target_arch]
            cfg["source_file"] = f"synthetic/{base['source_file']}"

            # FP8 only available on SM90+
            if target_arch == "sm80" and cfg["element_a"] in ("f8e4m3", "f8e5m2"):
                continue

            results.append(cfg)

    return results


def generate_synthetic_pairs(configs: list[dict]) -> list[dict]:
    """Generate synthetic migration pairs by applying known rules."""
    pairs = []
    target_archs = ["sm80", "sm90", "sm100"]

    for cfg in configs:
        for target in target_archs:
            if target == cfg["arch"]:
                continue
            # Only forward migrations (or backward for data augmentation)
            synth_configs = generate_synthetic_config(cfg, target)
            for synth in synth_configs:
                if ARCH_ORDER.get(target, 0) > ARCH_ORDER.get(cfg["arch"], 0):
                    before, after = cfg, synth
                else:
                    before, after = synth, cfg
                action = compute_action(before, after)
                pairs.append({
                    "before": before,
                    "after": after,
                    "action": action,
                    "pair_type": "synthetic",
                })

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate kernel migration pairs")
    parser.add_argument("--configs", required=True, help="Input JSON from extract_kernel_configs.py")
    parser.add_argument("--output", default="kernel_pairs.json", help="Output JSON file")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic augmentation")
    parser.add_argument("--max-synthetic", type=int, default=50000, help="Max synthetic pairs")
    args = parser.parse_args()

    with open(args.configs) as f:
        configs = json.load(f)
    print(f"Loaded {len(configs)} kernel configs")

    # Real pairs
    real_pairs = find_real_pairs(configs)
    print(f"Real pairs: {len(real_pairs)}")

    all_pairs = real_pairs

    # Synthetic augmentation
    if args.synthetic:
        synth_pairs = generate_synthetic_pairs(configs)
        if len(synth_pairs) > args.max_synthetic:
            import random
            random.seed(42)
            synth_pairs = random.sample(synth_pairs, args.max_synthetic)
        print(f"Synthetic pairs: {len(synth_pairs)}")
        all_pairs = real_pairs + synth_pairs

    print(f"Total pairs: {len(all_pairs)}")

    # Stats
    by_migration = {}
    for p in all_pairs:
        key = f"{p['before']['arch']}->{p['after']['arch']}"
        by_migration[key] = by_migration.get(key, 0) + 1
    for key, count in sorted(by_migration.items()):
        print(f"  {key}: {count}")

    with open(args.output, "w") as f:
        json.dump(all_pairs, f, indent=2)
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
