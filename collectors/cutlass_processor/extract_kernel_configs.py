#!/usr/bin/env python3
"""Extract structured kernel configurations from CUTLASS source tree.

Parses CUTLASS examples and include headers to extract template instantiations
for different GPU architectures (SM80, SM90, SM100). Produces structured config
vectors that can be paired across architectures for KernelWM training.

Usage:
    python extract_kernel_configs.py --cutlass-root /path/to/cutlass --output configs.json

Output: JSON array of kernel configs, each with:
    - arch: str (sm80, sm90, sm100)
    - kernel_type: str (gemm, conv, reduce)
    - element_a/b/c: str (f16, bf16, f32, f8e4m3, ...)
    - layout_a/b: str (row, col)
    - tile_m/n/k: int
    - cluster_m/n/k: int (1 for pre-Hopper)
    - stages: int (pipeline depth)
    - mma_class: str (hmma, wgmma, tcgen05, simt)
    - mainloop: str (cp_async, tma, tma_warp_specialized, ...)
    - epilogue: str (default, visitor, evt, ...)
    - source_file: str (relative path)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class KernelConfig:
    """Structured representation of a CUTLASS kernel configuration."""
    arch: str = "unknown"           # sm80, sm90, sm100
    kernel_type: str = "gemm"       # gemm, conv, reduce
    element_a: str = "f16"
    element_b: str = "f16"
    element_c: str = "f32"
    layout_a: str = "row"           # row, col
    layout_b: str = "col"
    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 32
    cluster_m: int = 1
    cluster_n: int = 1
    cluster_k: int = 1
    stages: int = 2
    mma_class: str = "unknown"      # hmma, wgmma, tcgen05, simt
    mainloop: str = "unknown"       # cp_async, tma, tma_warp_specialized, ...
    epilogue: str = "default"
    source_file: str = ""
    source_line: int = 0


# ── Regex patterns for CUTLASS template parameter extraction ──

# Architecture detection from file paths and pragmas
ARCH_PATTERNS = {
    "sm100": [
        re.compile(r'Sm100', re.IGNORECASE),
        re.compile(r'sm_100'),
        re.compile(r'blackwell', re.IGNORECASE),
        re.compile(r'tcgen05'),
    ],
    "sm90": [
        re.compile(r'Sm90', re.IGNORECASE),
        re.compile(r'sm_90'),
        re.compile(r'hopper', re.IGNORECASE),
        re.compile(r'[Ww]gm?ma'),
    ],
    "sm80": [
        re.compile(r'Sm80', re.IGNORECASE),
        re.compile(r'sm_80'),
        re.compile(r'ampere', re.IGNORECASE),
        re.compile(r'[Cc]p[Aa]sync'),
    ],
}

# Element type normalization
ELEMENT_MAP = {
    "half_t": "f16", "cutlass::half_t": "f16", "cute::half_t": "f16",
    "bfloat16_t": "bf16", "cutlass::bfloat16_t": "bf16", "cute::bfloat16_t": "bf16",
    "float": "f32", "double": "f64",
    "float_e4m3_t": "f8e4m3", "cutlass::float_e4m3_t": "f8e4m3",
    "float_e5m2_t": "f8e5m2", "cutlass::float_e5m2_t": "f8e5m2",
    "int8_t": "i8", "uint8_t": "u8",
    "tfloat32_t": "tf32", "cutlass::tfloat32_t": "tf32",
}

# Tile shape extraction: Shape<_128, _128, _64> or Shape<128, 128, 64>
TILE_SHAPE_RE = re.compile(
    r'(?:Tile)?Shape\s*<\s*_?(\d+)\s*,\s*_?(\d+)\s*,\s*_?(\d+)\s*>'
)

# Cluster shape extraction
CLUSTER_SHAPE_RE = re.compile(
    r'ClusterShape\s*.*?Shape\s*<\s*_?(\d+)\s*,\s*_?(\d+)\s*,\s*_?(\d+)\s*>'
)

# Stage count
STAGES_RE = re.compile(r'Stages?\s*(?:=|:)?\s*(\d+)')
STAGE_COUNT_RE = re.compile(r'StageCount(?:Auto)?(?:<\s*(\d+)\s*>)?')

# Mainloop / collective patterns
MAINLOOP_PATTERNS = {
    "tma_warp_specialized": re.compile(r'TmaWarpSpecialized', re.IGNORECASE),
    "tma": re.compile(r'Tma(?!Warp)', re.IGNORECASE),
    "cp_async": re.compile(r'CpAsync', re.IGNORECASE),
    "tcgen05": re.compile(r'tcgen05|Sm100', re.IGNORECASE),
}

# MMA class
MMA_PATTERNS = {
    "tcgen05": re.compile(r'tcgen05'),
    "wgmma": re.compile(r'wgmma|WarpGroupMma', re.IGNORECASE),
    "hmma": re.compile(r'hmma|OpClassTensorOp.*Sm80', re.IGNORECASE),
    "simt": re.compile(r'OpClassSimt', re.IGNORECASE),
}

# Layout
LAYOUT_ROW_RE = re.compile(r'RowMajor|layout::RowMajor|cute::C', re.IGNORECASE)
LAYOUT_COL_RE = re.compile(r'ColumnMajor|ColMajor|layout::ColumnMajor|cute::F', re.IGNORECASE)

# Element type in template params
ELEMENT_RE = re.compile(
    r'(?:Element[ABC]|typename\s+Element[ABC])\s*(?:=\s*|,\s*)([\w:]+)'
)


def detect_arch(text: str, filepath: str) -> str:
    """Detect GPU architecture from file content and path."""
    combined = text + " " + filepath
    # Check most specific first
    for arch in ["sm100", "sm90", "sm80"]:
        for pattern in ARCH_PATTERNS[arch]:
            if pattern.search(combined):
                return arch
    return "unknown"


def detect_mma_class(text: str, arch: str) -> str:
    """Detect MMA instruction class."""
    for mma, pattern in MMA_PATTERNS.items():
        if pattern.search(text):
            return mma
    # Default by arch
    return {"sm100": "tcgen05", "sm90": "wgmma", "sm80": "hmma"}.get(arch, "unknown")


def detect_mainloop(text: str, arch: str) -> str:
    """Detect mainloop type."""
    for ml, pattern in MAINLOOP_PATTERNS.items():
        if pattern.search(text):
            return ml
    return {"sm100": "tcgen05", "sm90": "tma", "sm80": "cp_async"}.get(arch, "unknown")


def normalize_element(raw: str) -> str:
    """Normalize element type string."""
    raw = raw.strip()
    return ELEMENT_MAP.get(raw, raw.lower().replace("cutlass::", "").replace("cute::", ""))


def extract_tile_shape(text: str) -> tuple[int, int, int] | None:
    """Extract tile shape (M, N, K) from text."""
    m = TILE_SHAPE_RE.search(text)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def extract_cluster_shape(text: str) -> tuple[int, int, int]:
    """Extract cluster shape, defaulting to (1,1,1)."""
    m = CLUSTER_SHAPE_RE.search(text)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return (1, 1, 1)


def extract_stages(text: str, arch: str) -> int:
    """Extract pipeline stage count."""
    m = STAGES_RE.search(text)
    if m:
        return int(m.group(1))
    m = STAGE_COUNT_RE.search(text)
    if m and m.group(1):
        return int(m.group(1))
    # Arch defaults
    return {"sm100": 4, "sm90": 4, "sm80": 2}.get(arch, 2)


def extract_elements(text: str) -> dict[str, str]:
    """Extract element types for A, B, C."""
    result = {"element_a": "f16", "element_b": "f16", "element_c": "f32"}

    # Look for using declarations or template params
    for suffix, key in [("A", "element_a"), ("B", "element_b"),
                        ("C", "element_c"), ("D", "element_c")]:
        pattern = re.compile(
            rf'Element{suffix}\s*(?:=\s*|,\s*)([\w:]+)'
        )
        m = pattern.search(text)
        if m:
            result[key] = normalize_element(m.group(1))

    return result


def extract_layouts(text: str) -> dict[str, str]:
    """Extract layouts for A and B."""
    result = {"layout_a": "row", "layout_b": "col"}

    for suffix, key in [("A", "layout_a"), ("B", "layout_b")]:
        pattern = re.compile(rf'Layout{suffix}\s*(?:=\s*|,\s*)([\w:]+)')
        m = pattern.search(text)
        if m:
            val = m.group(1)
            if LAYOUT_COL_RE.search(val):
                result[key] = "col"
            else:
                result[key] = "row"

    return result


def parse_file(filepath: Path, cutlass_root: Path) -> list[KernelConfig]:
    """Extract kernel configs from a single file."""
    try:
        text = filepath.read_text(errors="replace")
    except Exception:
        return []

    rel_path = str(filepath.relative_to(cutlass_root))
    configs = []

    # Detect architecture
    arch = detect_arch(text, rel_path)
    if arch == "unknown":
        return []

    # Find tile shapes — each one is potentially a kernel config
    tile_matches = list(TILE_SHAPE_RE.finditer(text))

    if not tile_matches:
        # File has arch markers but no tile shapes — create one config from file-level info
        cfg = KernelConfig(
            arch=arch,
            mma_class=detect_mma_class(text, arch),
            mainloop=detect_mainloop(text, arch),
            stages=extract_stages(text, arch),
            source_file=rel_path,
        )
        elements = extract_elements(text)
        layouts = extract_layouts(text)
        for k, v in {**elements, **layouts}.items():
            setattr(cfg, k, v)

        cluster = extract_cluster_shape(text)
        cfg.cluster_m, cfg.cluster_n, cfg.cluster_k = cluster

        configs.append(cfg)
    else:
        # One config per tile shape found
        seen_tiles = set()
        for tm in tile_matches:
            tile = (int(tm.group(1)), int(tm.group(2)), int(tm.group(3)))
            # Filter out cluster shapes misidentified as tiles:
            # real GEMM tiles have all dims >= 16 (no 4x1x1, 2x2x1, etc.)
            if tile[0] < 16 or tile[1] < 16 or tile[2] < 16:
                continue
            if tile in seen_tiles:
                continue
            seen_tiles.add(tile)

            # Use surrounding context (~500 chars around match) for param extraction
            start = max(0, tm.start() - 500)
            end = min(len(text), tm.end() + 500)
            context = text[start:end]

            cfg = KernelConfig(
                arch=arch,
                tile_m=tile[0], tile_n=tile[1], tile_k=tile[2],
                mma_class=detect_mma_class(context + text[:2000], arch),
                mainloop=detect_mainloop(context + text[:2000], arch),
                stages=extract_stages(context, arch),
                source_file=rel_path,
                source_line=text[:tm.start()].count("\n") + 1,
            )

            elements = extract_elements(context)
            layouts = extract_layouts(context)
            for k, v in {**elements, **layouts}.items():
                setattr(cfg, k, v)

            cluster = extract_cluster_shape(context)
            cfg.cluster_m, cfg.cluster_n, cfg.cluster_k = cluster

            # Detect kernel type from path
            if "conv" in rel_path.lower():
                cfg.kernel_type = "conv"
            elif "reduce" in rel_path.lower() or "reduction" in rel_path.lower():
                cfg.kernel_type = "reduce"

            configs.append(cfg)

    return configs


def scan_cutlass(cutlass_root: Path) -> list[KernelConfig]:
    """Scan CUTLASS repo for kernel configurations."""
    all_configs = []

    # Priority directories
    scan_dirs = [
        cutlass_root / "examples",
        cutlass_root / "include" / "cutlass" / "gemm",
        cutlass_root / "include" / "cutlass" / "conv",
        cutlass_root / "test" / "unit" / "gemm",
    ]

    extensions = {".hpp", ".h", ".cu", ".cpp", ".cuh"}

    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            print(f"  skip (not found): {scan_dir.relative_to(cutlass_root)}")
            continue

        file_count = 0
        for filepath in scan_dir.rglob("*"):
            if filepath.suffix not in extensions:
                continue
            if filepath.is_dir():
                continue
            configs = parse_file(filepath, cutlass_root)
            all_configs.extend(configs)
            file_count += 1

        print(f"  {scan_dir.relative_to(cutlass_root)}: {file_count} files -> {len([c for c in all_configs if c.source_file.startswith(str(scan_dir.relative_to(cutlass_root)))])} configs")

    return all_configs


def deduplicate(configs: list[KernelConfig]) -> list[KernelConfig]:
    """Remove duplicate configs (same arch + tile + elements + layouts)."""
    seen = set()
    unique = []
    for cfg in configs:
        key = (cfg.arch, cfg.kernel_type,
               cfg.element_a, cfg.element_b, cfg.element_c,
               cfg.layout_a, cfg.layout_b,
               cfg.tile_m, cfg.tile_n, cfg.tile_k,
               cfg.cluster_m, cfg.cluster_n, cfg.cluster_k,
               cfg.stages, cfg.mma_class, cfg.mainloop)
        if key not in seen:
            seen.add(key)
            unique.append(cfg)
    return unique


def main():
    parser = argparse.ArgumentParser(description="Extract kernel configs from CUTLASS")
    parser.add_argument("--cutlass-root", required=True, help="Path to CUTLASS repo")
    parser.add_argument("--output", default="kernel_configs.json", help="Output JSON file")
    args = parser.parse_args()

    cutlass_root = Path(args.cutlass_root)
    if not cutlass_root.exists():
        print(f"ERROR: {cutlass_root} not found")
        sys.exit(1)

    print(f"Scanning CUTLASS at {cutlass_root}...")
    configs = scan_cutlass(cutlass_root)
    print(f"\nRaw configs extracted: {len(configs)}")

    configs = deduplicate(configs)
    print(f"After dedup: {len(configs)}")

    # Stats
    by_arch = {}
    for cfg in configs:
        by_arch.setdefault(cfg.arch, []).append(cfg)
    for arch, cfgs in sorted(by_arch.items()):
        print(f"  {arch}: {len(cfgs)} configs")

    # Write output
    output = [asdict(c) for c in configs]
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
