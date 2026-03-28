"""Looped Transformer + Augmentations — declarative YAML spec wrapper.

Registers the looped_augmented architecture from an inline YAML spec.
Uses weight-shared recurrence (12 steps, 3 unique blocks) with SmearGate
and BigramHash augmentations for parameter-efficient training.

Usage:
    MODEL_FAMILY=looped_augmented
    RECURRENCE_STEPS=12
    SHARE_BLOCKS=3
"""
from __future__ import annotations

from pathlib import Path

from crucible.models.composer import register_from_spec

_SPEC_PATH = Path(__file__).resolve().parent.parent.parent / "architectures" / "looped_augmented.yaml"

if _SPEC_PATH.exists():
    register_from_spec("looped_augmented", _SPEC_PATH, source="local")
