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
from typing import Any

from crucible.training.callbacks import TrainingCallback, register_callback


# ---------------------------------------------------------------------------
# Fake quantization (STE)
# ---------------------------------------------------------------------------

def fake_int8_quant(w):
    """Straight-through estimator for int8 QAT -- gradient flows through w."""
    import torch

    if w.ndim != 2:
        return w
    with torch.no_grad():
        scale = w.abs().amax(dim=1, keepdim=True) / 127.0
        scale = scale.clamp_min(1.0 / 127.0)
    q = torch.clamp(torch.round(w / scale), -128, 127)
    return w + (q * scale - w).detach()  # STE


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class QatInt8Callback(TrainingCallback):
    """INT8 QAT via per-row fake quantization with straight-through estimator."""

    priority = 5  # Run early, before pruning callbacks

    def __init__(self, **kwargs: Any) -> None:
        self.warmup_steps = int(os.environ.get("QAT_INT8_WARMUP_STEPS", "500"))
        exclude_raw = os.environ.get("QAT_INT8_EXCLUDE_PATTERNS", "tok_emb,embed_low,lm_head")
        self.exclude_patterns = tuple(p.strip() for p in exclude_raw.split(",") if p.strip())
        self._hooks: list[Any] = []
        self._enabled = False

    def on_model_ready(self, state: dict[str, Any]) -> None:
        """Register QAT hooks BEFORE torch.compile so they're in the graph."""
        model = state.get("model")
        if model is None:
            return

        from crucible.training.compression_utils import iter_prunable_layers

        self._bit_assignments: dict[str, int] = {}

        for name, module in iter_prunable_layers(model, exclude_patterns=self.exclude_patterns):

            def _make_hook(mod_name: str):
                def _hook(mod, inputs):
                    if self._enabled:
                        mod.weight.data = fake_int8_quant(mod.weight.data)
                return _hook

            handle = module.register_forward_pre_hook(_make_hook(name))
            self._hooks.append(handle)
            self._bit_assignments[name] = 8

    def on_train_begin(self, state: dict[str, Any]) -> None:
        state["qat_bit_assignments"] = self._bit_assignments

    def on_step_begin(self, step: int, state: dict[str, Any]) -> None:
        if not self._enabled and step >= self.warmup_steps:
            self._enabled = True

    def on_train_end(self, state: dict[str, Any]) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._enabled = False


register_callback("qat_int8", QatInt8Callback, source="local")
