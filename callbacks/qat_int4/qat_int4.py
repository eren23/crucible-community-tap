"""INT4 quantization-aware training callback.

Applies fake INT4 quantization with per-group scale factors via
straight-through estimator (STE).  Groups of ``GROUP_SIZE`` weights share
a single scale, giving finer granularity than per-row quantization.

Env vars:
    QAT_INT4_WARMUP_STEPS: Steps before enabling fake quant (default 1000).
    QAT_INT4_GROUP_SIZE: Number of weights per quantization group (default 128).
    QAT_INT4_EXCLUDE_PATTERNS: Comma-separated module name patterns to skip
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

from callbacks._qat_common import BaseQATCallback, fake_int4_quant, parse_exclude_patterns

from crucible.training.callbacks import TrainingCallback, register_callback


class QatInt4Callback(BaseQATCallback, TrainingCallback):
    """INT4 QAT via per-group fake quantization with straight-through estimator."""

    priority = 5  # Run early, before pruning callbacks

    def __init__(self, **kwargs: Any) -> None:
        self.warmup_steps = int(os.environ.get("QAT_INT4_WARMUP_STEPS", "1000"))
        self.group_size = int(os.environ.get("QAT_INT4_GROUP_SIZE", "128"))
        self.exclude_patterns = parse_exclude_patterns("QAT_INT4_EXCLUDE_PATTERNS")
        self._hooks: list[Any] = []
        self._enabled = False
        self._bit_assignments: dict[str, int] = {}

    def on_model_ready(self, state: dict[str, Any]) -> None:
        """Register QAT hooks BEFORE torch.compile so they're in the graph."""
        model = state.get("model")
        if model is None:
            return

        gs = self.group_size

        def _make_group_quant(mod_name: str, group_sz: int):
            def _fn(w):
                return fake_int4_quant(w, group_size=group_sz)
            return _fn

        from crucible.training.compression_utils import iter_prunable_layers
        self._bit_assignments = {}
        for name, module in iter_prunable_layers(model, exclude_patterns=self.exclude_patterns):
            quant_fn = _make_group_quant(name, gs)

            def _make_hook(mod_name: str, fn):
                def _hook(mod, inputs):
                    if self._enabled:
                        mod.weight.data = fn(mod.weight.data)
                return _hook

            handle = module.register_forward_pre_hook(_make_hook(name, quant_fn))
            self._hooks.append(handle)
            self._bit_assignments[name] = 4

    # on_train_begin, on_step_begin, on_train_end inherited from BaseQATCallback


register_callback("qat_int4", QatInt4Callback, source="local")
