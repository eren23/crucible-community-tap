"""Mixed-precision quantization-aware training callback.

Parses a per-module bit-width pattern from ``QAT_MIXED_PATTERN`` and applies
the matching fake-quantization function to each module during forward passes.
This lets different parts of the model run at different precisions (e.g.,
encoder at 8-bit, predictor at 4-bit).

Pattern format (``QAT_MIXED_PATTERN``):
    - Single value: ``"8"`` -- all modules at 8 bits (default).
    - Comma-separated: ``"encoder:8,predictor:4,lm_head:8"`` -- modules whose
      name contains the prefix get the specified bit-width.

Env vars:
    QAT_MIXED_PATTERN: Bit-width mapping (default "8").
    QAT_MIXED_WARMUP_STEPS: Steps before enabling fake quant (default 500).
    QAT_MIXED_EXCLUDE_PATTERNS: Comma-separated module name patterns to skip
        (default "tok_emb,embed_low").
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Import shared QAT utilities from sibling _qat_common module
_callbacks_dir = str(Path(__file__).parent.parent)
if _callbacks_dir not in sys.path:
    sys.path.insert(0, _callbacks_dir)

from callbacks._qat_common import BaseQATCallback, get_quant_fn, parse_exclude_patterns

from crucible.training.callbacks import TrainingCallback, register_callback


# ---------------------------------------------------------------------------
# Pattern parsing
# ---------------------------------------------------------------------------

def _parse_pattern(raw: str) -> dict[str, int] | int:
    """Parse QAT_MIXED_PATTERN into either a global bit-width or prefix map.

    Returns:
        int -- if the pattern is a single number (applied to all modules).
        dict -- mapping of name-prefix to bit-width.
    """
    raw = raw.strip()
    if ":" not in raw:
        return int(raw)

    mapping: dict[str, int] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        prefix, _, bits_str = part.partition(":")
        mapping[prefix.strip()] = int(bits_str.strip())
    return mapping


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class QatMixedCallback(BaseQATCallback, TrainingCallback):
    """Mixed-precision QAT with per-module bit-width assignment."""

    priority = 5  # Run early, before pruning callbacks

    def __init__(self, **kwargs: Any) -> None:
        self.warmup_steps = int(os.environ.get("QAT_MIXED_WARMUP_STEPS", "500"))
        pattern_raw = os.environ.get("QAT_MIXED_PATTERN", "8")
        self.pattern = _parse_pattern(pattern_raw)
        self.exclude_patterns = parse_exclude_patterns("QAT_MIXED_EXCLUDE_PATTERNS", "tok_emb,embed_low")
        self._hooks: list[Any] = []
        self._enabled = False
        self._bit_assignments: dict[str, int] = {}

    def _resolve_bits(self, module_name: str) -> int:
        """Determine the bit-width for a given module name."""
        if isinstance(self.pattern, int):
            return self.pattern
        for prefix, bits in self.pattern.items():
            if prefix in module_name:
                return bits
        # No match -- fall back to 8-bit
        return 8

    def on_model_ready(self, state: dict[str, Any]) -> None:
        """Register QAT hooks BEFORE torch.compile so they're in the graph."""
        model = state.get("model")
        if model is None:
            return

        def _resolve_fn(name: str):
            bits = self._resolve_bits(name)
            return get_quant_fn(bits), bits

        self._register_quant_hooks_base(model, None, bits=0, resolve_fn=_resolve_fn)

    # on_train_begin, on_step_begin, on_train_end inherited from BaseQATCallback


register_callback("qat_mixed", QatMixedCallback, source="local")
