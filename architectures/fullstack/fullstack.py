"""SOTA-Inspired V1 — declarative YAML spec wrapper.

Registers the sota_inspired_v1 architecture from an inline YAML spec.
Competition-winning design with encoder-decoder U-Net skips, gated residuals,
3x MLP, SmearGate, BigramHash, TrigramHash, and orthogonal init.

Usage:
    MODEL_FAMILY=sota_inspired_v1
    NUM_LAYERS=11
    MLP_MULT=3
"""
from __future__ import annotations

from pathlib import Path

from crucible.models.composer import register_from_spec

_SPEC_PATH = Path(__file__).resolve().parent.parent.parent / "architectures" / "sota_inspired_v1.yaml"

if _SPEC_PATH.exists():
    register_from_spec("sota_inspired_v1", _SPEC_PATH, source="local")
