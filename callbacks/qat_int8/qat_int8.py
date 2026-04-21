"""INT8 quantization-aware training callback.

Applies fake INT8 quantization via straight-through estimator (STE) during
the forward pass so the model learns to be robust to quantization noise.
Hooks are gated by a warmup period and removed at train end for clean
serialization.

Env vars:
    QAT_INT8_WARMUP_STEPS: Steps before enabling fake quant (default 500).
    QAT_INT8_EXCLUDE_PATTERNS: Comma-separated module name patterns to skip
        (default "tok_emb,embed_low,lm_head").
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

from callbacks._qat_common import BaseQATCallback, fake_int8_quant, parse_exclude_patterns

from crucible.training.callbacks import TrainingCallback, register_callback


class QatInt8Callback(BaseQATCallback, TrainingCallback):
    """INT8 QAT via per-row fake quantization with straight-through estimator."""

    priority = 5  # Run early, before pruning callbacks

    def __init__(self, **kwargs: Any) -> None:
        self.warmup_steps = int(os.environ.get("QAT_INT8_WARMUP_STEPS", "500"))
        self.exclude_patterns = parse_exclude_patterns("QAT_INT8_EXCLUDE_PATTERNS")
        self._hooks: list[Any] = []
        self._enabled = False
        self._bit_assignments: dict[str, int] = {}

    def on_model_ready(self, state: dict[str, Any]) -> None:
        """Register QAT hooks BEFORE torch.compile so they're in the graph."""
        model = state.get("model")
        if model is None:
            return
        self._register_quant_hooks_base(model, fake_int8_quant, bits=8)

    # on_train_begin, on_step_begin, on_train_end inherited from BaseQATCallback


register_callback("qat_int8", QatInt8Callback, source="local")
